"""
ESM-2 training with FSDP (Fully Sharded Data Parallelism) on Neuron.

FSDP shards model weights across devices, enabling training of models
that don't fit on a single device. Each device sees the full model
during forward (via all-gather) but only stores 1/N of the weights.

Usage:
    torchrun --nproc-per-node 4 train_esm_fsdp.py --model-size 3B --num-steps 10
    torchrun --nproc-per-node 16 train_esm_fsdp.py --model-size 15B --num-steps 10
"""

import argparse
import logging
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.models.esm.modeling_esm import EsmLayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
    "15B": "facebook/esm2_t48_15B_UR50D",
}

MASK_PROB = 0.15

SAMPLE_SEQUENCES = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
    "MKFLILLFNILCLFPVLAADNHGVSMAQTQGFVDLQAGVHRLMQLLRDNLTLRSGAFNQEFNISYCQVYCKDLLQEADGIGTQWTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHERCP",
    "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVGTSKEAKQDGIDLYKHMFEHYPAMKKYFKHRENYTPADVQKDPFFIKQGQNILLACHVLCATYDDRETFNAYTRELLDRHARDHVHMPPEVWTDFWKLFEEYLGKKTTLDEPTKQAWHEIGREFAKEINK",
    "MGDIQVQVNIDDSGKNFDYIASQHFTKVLEHYAAGKDIAHTKFADPELVAQKQAELNAAGKIKGEQLGKFDDLVKKLDDNHALDTDFKQKIDKLAKELGINYQVHGAKVEGDTKLMISLDNFESDKFTTEHAKEKFDELAKNHGIAFNFVKMMFAQK",
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHQRQVTRERDQKLNQLEESINAINNKDSINKLQDKGKFLIPSIETKE",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGIIEPSLKALASKYNCDKSVCRKCYARLPPRATNCRKRKCGHTNQLRPKKKLK",
]


def mask_tokens(input_ids, tokenizer):
    labels = input_ids.clone()
    prob_matrix = torch.full(input_ids.shape, MASK_PROB)
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for tid in [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if tid is not None:
            special_mask |= (input_ids == tid)
    prob_matrix.masked_fill_(special_mask, 0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100
    input_ids[masked_indices] = tokenizer.mask_token_id
    return input_ids, labels


def create_batch(sequences, tokenizer, max_seq_len, device):
    encoded = tokenizer(
        sequences, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_seq_len,
    )
    input_ids, labels = mask_tokens(encoded["input_ids"], tokenizer)
    return input_ids.to(device), encoded["attention_mask"].to(device), labels.to(device)


def forward_with_loss(model, input_ids, attention_mask, labels):
    """Forward pass with HuggingFace's built-in loss (labels=-100 ignored)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def run_training(**kwargs):
    model_size = kwargs["model_size"]
    max_seq_len = kwargs.get("max_seq_len", 128)
    batch_size = kwargs.get("batch_size", 2)
    num_steps = kwargs.get("num_steps", 10)
    lr = kwargs.get("lr", 5e-5)

    model_id = MODEL_CONFIGS[model_size]

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    logger.info(f"Rank {rank}/{world_size}: Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # Wrap with FSDP - shard at EsmLayer granularity
    from functools import partial
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={EsmLayer},
    )

    logger.info(f"Rank {rank}: Wrapping with FSDP (world_size={world_size})...")
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Rank {rank}: {total_params:,} params (sharded)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False)

    # Note: torch.compile + FSDP is not yet supported on Neuron (FSDP uses compiler.disable()
    # internally which breaks fullgraph=True, and fullgraph=False causes SIGSEGV).
    # For compiled training, use TorchTitan which has a compatible integration path.
    # Eager FSDP works well: ~4-5s/step for 3B, ~18-20s/step for 15B.
    no_compile = kwargs.get("no_compile", True)
    if not no_compile:
        logger.warning(f"Rank {rank}: torch.compile + FSDP not supported on Neuron, using eager")

    logger.info(f"Rank {rank}: Training {num_steps} steps, batch_size={batch_size}, seq_len={max_seq_len}")

    t_start = time.time()
    for step in range(num_steps):
        batch_seqs = random.choices(SAMPLE_SEQUENCES, k=batch_size)
        input_ids, attention_mask, labels = create_batch(batch_seqs, tokenizer, max_seq_len, device)

        optimizer.zero_grad()
        loss = forward_with_loss(model, input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()

        if rank == 0:
            elapsed = time.time() - t_start
            logger.info(f"  Step {step}/{num_steps}: loss={loss.item():.4f}, elapsed={elapsed:.1f}s")

    elapsed = time.time() - t_start
    if rank == 0:
        logger.info(f"Training complete: {num_steps} steps in {elapsed:.1f}s")
        logger.info(f"  Steps/sec: {num_steps / elapsed:.3f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="3B", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (pure eager)")
    args = parser.parse_args()
    run_training(**vars(args))
