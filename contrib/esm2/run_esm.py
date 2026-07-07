"""
ESM-2 protein language model inference on Neuron devices.

Loads a pretrained ESM-2 model from HuggingFace, compiles with
torch.compile for Neuron, and runs masked language modeling inference
on protein sequences.

Usage (single device, small models):
    torchrun --nproc-per-node 1 run_esm.py --model-size 650M

Usage (tensor parallel, large models):
    torchrun --nproc-per-node 4 run_esm.py --model-size 3B
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration mapping
MODEL_CONFIGS = {
    "8M": {"hf_model_id": "facebook/esm2_t6_8M_UR50D", "tp_degree": 1},
    "35M": {"hf_model_id": "facebook/esm2_t12_35M_UR50D", "tp_degree": 1},
    "150M": {"hf_model_id": "facebook/esm2_t30_150M_UR50D", "tp_degree": 1},
    "650M": {"hf_model_id": "facebook/esm2_t33_650M_UR50D", "tp_degree": 1},
    "3B": {"hf_model_id": "facebook/esm2_t36_3B_UR50D", "tp_degree": 4},
    "15B": {"hf_model_id": "facebook/esm2_t48_15B_UR50D", "tp_degree": 4},
}

# Example protein sequences
DEFAULT_SEQUENCES = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]

MAX_SEQ_LEN = 128

torch.set_default_dtype(torch.float32)


def apply_esm_tp(model, tp_mesh):
    """Apply tensor parallelism to HuggingFace ESM-2 model."""
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    layer_tp_plan = {
        "attention.self.query": ColwiseParallel(),
        "attention.self.key": ColwiseParallel(),
        "attention.self.value": ColwiseParallel(),
        "attention.output.dense": RowwiseParallel(),
        "intermediate.dense": ColwiseParallel(),
        "output.dense": RowwiseParallel(),
    }

    for layer in model.esm.encoder.layer:
        parallelize_module(layer, tp_mesh, layer_tp_plan)

    return model


def run_esm_inference(**kwargs):
    model_size = kwargs.get("model_size", "650M")
    sequences = kwargs.get("sequences", DEFAULT_SEQUENCES)
    max_seq_len = kwargs.get("max_seq_len", MAX_SEQ_LEN)

    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_size]
    model_id = config["hf_model_id"]
    expected_tp = config["tp_degree"]

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != expected_tp:
        raise ValueError(
            f"Model {model_size} requires TP degree {expected_tp}, got world size {world_size}. "
            f"Use: torchrun --nproc-per-node {expected_tp} run_esm.py --model-size {model_size}"
        )

    device = torch.neuron.current_device()

    # Load tokenizer
    logger.info(f"Rank {rank}: Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    logger.info(f"Rank {rank}: Loading model from {model_id}...")
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    logger.info(f"Rank {rank}: Model loaded in {time.time() - t0:.2f}s")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Rank {rank}: {total_params:,} parameters")

    # Apply TP if needed
    if world_size > 1:
        from torch.distributed.device_mesh import DeviceMesh

        device_mesh = DeviceMesh("neuron", list(range(world_size)))
        logger.info(f"Rank {rank}: Applying tensor parallelism...")
        model = apply_esm_tp(model, device_mesh)

    # Move to device
    logger.info(f"Rank {rank}: Moving to {device}...")
    t0 = time.time()
    model = model.to(device)
    logger.info(f"Rank {rank}: Moved in {time.time() - t0:.2f}s")
    model.eval()

    # Tokenize sequences with mask tokens inserted at middle position
    masked_sequences = []
    mask_positions = []
    for seq in sequences:
        mid = len(seq) // 2
        masked = seq[:mid] + tokenizer.mask_token + seq[mid + 1:]
        masked_sequences.append(masked)
        mask_positions.append(mid)

    logger.info(f"Rank {rank}: Tokenizing {len(sequences)} sequences (max_len={max_seq_len})...")
    inputs = tokenizer(
        masked_sequences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Compile
    logger.info(f"Rank {rank}: Compiling model with torch.compile(backend='neuron')...")
    model.forward = torch.compile(model.forward, backend="neuron", fullgraph=True, dynamic=False)

    dist.barrier()

    # Warmup
    logger.info(f"Rank {rank}: Warmup forward pass...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    warmup_time = time.time() - t0
    logger.info(f"Rank {rank}: Warmup: {warmup_time:.2f}s (includes compilation)")

    # Timed inference
    logger.info(f"Rank {rank}: Timed forward pass...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    inference_time = time.time() - t0
    logits = outputs.logits

    if rank == 0:
        logger.info(f"Rank {rank}: Logits shape: {logits.shape}")
        logger.info(f"Rank {rank}: Inference time: {inference_time:.4f}s")

        # Decode predictions at masked positions
        for i, (seq, mpos) in enumerate(zip(sequences, mask_positions)):
            # +1 for CLS token prepended by tokenizer
            token_pos = mpos + 1
            predicted_id = logits[i, token_pos].argmax(dim=-1).item()
            predicted_aa = tokenizer.decode([predicted_id]).strip()
            original_aa = seq[mpos]
            logger.info(
                f"  Seq {i}: masked position {mpos}, "
                f"original='{original_aa}', predicted='{predicted_aa}'"
            )

    dist.barrier()
    dist.destroy_process_group()
    logger.info(f"Rank {rank}: Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="ESM-2 inference on Neuron devices")
    parser.add_argument(
        "--model-size", type=str, choices=list(MODEL_CONFIGS.keys()),
        default="650M", help="ESM-2 model size",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=MAX_SEQ_LEN,
        help="Maximum sequence length (padded)",
    )
    parser.add_argument(
        "--sequences", type=str, nargs="+", default=None,
        help="Protein sequences to run inference on (masks middle residue)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_esm_inference(
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        sequences=args.sequences or DEFAULT_SEQUENCES,
    )
