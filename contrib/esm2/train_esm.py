"""
ESM-2 masked language model training on Neuron devices.

Features:
- Masked language modeling (MLM) with 15% random masking
- FASTA dataset loader with train/eval split
- Cosine LR schedule with linear warmup
- Checkpoint saving/loading (resume training)
- Periodic evaluation with perplexity reporting

Usage:
    torchrun --nproc-per-node 1 train_esm.py --model-size 8M --num-steps 100
    torchrun --nproc-per-node 1 train_esm.py --model-size 650M --fasta proteins.fasta
    torchrun --nproc-per-node 1 train_esm.py --model-size 650M --resume checkpoint/
"""

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForMaskedLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "8M": {"hf_model_id": "facebook/esm2_t6_8M_UR50D", "tp_degree": 1},
    "35M": {"hf_model_id": "facebook/esm2_t12_35M_UR50D", "tp_degree": 1},
    "150M": {"hf_model_id": "facebook/esm2_t30_150M_UR50D", "tp_degree": 1},
    "650M": {"hf_model_id": "facebook/esm2_t33_650M_UR50D", "tp_degree": 1},
    "3B": {"hf_model_id": "facebook/esm2_t36_3B_UR50D", "tp_degree": 4},
    "15B": {"hf_model_id": "facebook/esm2_t48_15B_UR50D", "tp_degree": 8},
}

MASK_PROB = 0.15

# Built-in demo sequences (used when no --fasta provided)
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


# ── Dataset ─────────────────────────────────────────────────────────────

def load_fasta(path):
    """Load sequences from a FASTA file, filtering by length."""
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    seq = "".join(current)
                    if len(seq) >= 10:  # skip very short sequences
                        sequences.append(seq)
                    current = []
            elif line:
                current.append(line)
    if current:
        seq = "".join(current)
        if len(seq) >= 10:
            sequences.append(seq)
    return sequences


def split_dataset(sequences, eval_fraction=0.1, seed=42):
    """Split sequences into train/eval sets."""
    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


# ── Masking ──────────────────────────────────────────────────────────────

def mask_tokens(input_ids, tokenizer, mask_prob=MASK_PROB):
    """BERT-style random masking for MLM."""
    labels = input_ids.clone()
    prob_matrix = torch.full(input_ids.shape, mask_prob)

    # Don't mask special tokens
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if token_id is not None:
            special_mask |= (input_ids == token_id)
    prob_matrix.masked_fill_(special_mask, 0.0)

    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100

    # 80% [MASK], 10% random, 10% original
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[replace_mask] = tokenizer.mask_token_id

    random_mask = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~replace_mask
    random_tokens = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[random_mask] = random_tokens[random_mask]

    return input_ids, labels


def create_batch(sequences, tokenizer, max_seq_len, device):
    """Tokenize and mask a batch of sequences."""
    encoded = tokenizer(
        sequences, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_seq_len,
    )
    input_ids, labels = mask_tokens(encoded["input_ids"], tokenizer)
    # Precompute loss mask on CPU (avoids int64 comparison on Neuron)
    loss_mask = (labels != -100).float()
    # Replace -100 with 0 and cast to int32 (Neuron doesn't support int64 in cross_entropy)
    safe_labels = labels.clamp(min=0).to(torch.int32)
    return (input_ids.to(device), encoded["attention_mask"].to(device),
            safe_labels.to(device), loss_mask.to(device))


# ── LR Schedule ──────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine decay with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpointing ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, loss, checkpoint_dir):
    """Save model, optimizer, scheduler state."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Save to CPU to avoid device issues on reload
    state = {
        "step": step,
        "loss": loss,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    path = checkpoint_dir / "checkpoint.pt"
    torch.save(state, path)
    logger.info(f"Checkpoint saved at step {step} to {path}")


def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device):
    """Load checkpoint and return the step to resume from."""
    path = Path(checkpoint_dir) / "checkpoint.pt"
    if not path.exists():
        logger.info(f"No checkpoint found at {path}, starting from scratch")
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    step = state["step"]
    logger.info(f"Resumed from checkpoint at step {step} (loss={state['loss']:.4f})")
    return step


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(model, compiled_forward, eval_sequences, tokenizer, max_seq_len, batch_size, device, vocab_size):
    """Compute perplexity on eval set using compiled forward+loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    num_batches = max(1, len(eval_sequences) // batch_size)
    with torch.no_grad():
        for i in range(num_batches):
            batch_seqs = eval_sequences[i * batch_size:(i + 1) * batch_size]
            if not batch_seqs:
                break
            input_ids, attention_mask, safe_labels, loss_mask = create_batch(batch_seqs, tokenizer, max_seq_len, device)
            loss = compiled_forward(model, input_ids, attention_mask, safe_labels, loss_mask)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    avg_loss = total_loss / max(1, n_batches)
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, perplexity


# ── Forward ──────────────────────────────────────────────────────────────

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


def forward_with_loss(model, input_ids, attention_mask, safe_labels, loss_mask):
    """Compiled forward+loss with manual cross-entropy (no dynamic shapes)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    vocab_size = logits.shape[-1]
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = safe_labels.view(-1)
    mask_flat = loss_mask.view(-1)
    # Manual cross-entropy: -log_softmax[target_class], masked
    log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # Gather the log prob of the correct class
    per_token_loss = -log_probs.gather(1, labels_flat.unsqueeze(1).long()).squeeze(1)
    # Average over all positions, weighted by mask (static divisor)
    total_tokens = mask_flat.shape[0]
    loss = (per_token_loss * mask_flat).sum() / total_tokens
    return loss


# ── Main ─────────────────────────────────────────────────────────────────

def run_training(**kwargs):
    model_size = kwargs["model_size"]
    max_seq_len = kwargs["max_seq_len"]
    batch_size = kwargs["batch_size"]
    num_steps = kwargs["num_steps"]
    lr = kwargs["lr"]
    fasta_path = kwargs.get("fasta")
    checkpoint_dir = kwargs["checkpoint_dir"]
    resume_path = kwargs.get("resume")
    eval_every = kwargs["eval_every"]
    save_every = kwargs["save_every"]
    warmup_steps = kwargs["warmup_steps"]
    eval_fraction = kwargs["eval_fraction"]

    config = MODEL_CONFIGS[model_size]
    model_id = config["hf_model_id"]

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    # Load tokenizer and model
    logger.info(f"Rank {rank}: Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # Apply tensor parallelism if needed
    tp_degree = config["tp_degree"]
    if world_size > 1 and tp_degree > 1:
        from torch.distributed.device_mesh import DeviceMesh
        device_mesh = DeviceMesh("neuron", list(range(world_size)))
        logger.info(f"Rank {rank}: Applying TP (degree={world_size})...")
        model = apply_esm_tp(model, device_mesh)

    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Rank {rank}: {total_params:,} params on {device}")

    # Load and split dataset
    sequences = load_fasta(fasta_path) if fasta_path else SAMPLE_SEQUENCES
    train_seqs, eval_seqs = split_dataset(sequences, eval_fraction=eval_fraction)
    logger.info(f"Rank {rank}: {len(train_seqs)} train, {len(eval_seqs)} eval sequences")

    # Optimizer (disable foreach when using TP to avoid DTensor/Tensor mixing)
    use_foreach = (tp_degree == 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=use_foreach)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)

    # Resume from checkpoint if specified
    start_step = 0
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer, scheduler, device)

    # Compile forward+loss together (avoids eager-mode cross_entropy issues with TP)
    logger.info(f"Rank {rank}: Compiling forward+loss...")
    compiled_train_step = torch.compile(forward_with_loss, backend="neuron", fullgraph=True, dynamic=False)
    compiled_forward = compiled_train_step

    vocab_size = model.config.vocab_size

    # Training loop
    logger.info(f"Rank {rank}: Training steps {start_step}→{num_steps}, batch_size={batch_size}, seq_len={max_seq_len}")
    logger.info(f"Rank {rank}: LR={lr}, warmup={warmup_steps}, eval_every={eval_every}, save_every={save_every}")

    total_loss = 0.0
    t_start = time.time()

    for step in range(start_step, num_steps):
        batch_seqs = random.choices(train_seqs, k=batch_size)
        input_ids, attention_mask, safe_labels, loss_mask = create_batch(batch_seqs, tokenizer, max_seq_len, device)

        optimizer.zero_grad()
        loss = compiled_train_step(model, input_ids, attention_mask, safe_labels, loss_mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val

        if rank == 0 and (step % 5 == 0 or step == num_steps - 1):
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_start
            logger.info(f"  Step {step}/{num_steps}: loss={loss_val:.4f}, lr={current_lr:.2e}, elapsed={elapsed:.1f}s")

        # Evaluation
        if rank == 0 and eval_every > 0 and (step + 1) % eval_every == 0:
            eval_loss, ppl = evaluate(
                model, compiled_forward, eval_seqs, tokenizer,
                max_seq_len, batch_size, device, vocab_size,
            )
            logger.info(f"  [Eval] step={step+1}: loss={eval_loss:.4f}, perplexity={ppl:.2f}")

        # Checkpoint
        if rank == 0 and save_every > 0 and (step + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step + 1, loss_val, checkpoint_dir)

    elapsed = time.time() - t_start
    avg_loss = total_loss / max(1, num_steps - start_step)

    if rank == 0:
        logger.info(f"Training complete: steps {start_step}→{num_steps} in {elapsed:.1f}s")
        logger.info(f"  Avg loss: {avg_loss:.4f}")
        logger.info(f"  Steps/sec: {(num_steps - start_step) / elapsed:.2f}")
        logger.info(f"  Samples/sec: {(num_steps - start_step) * batch_size / elapsed:.2f}")

        # Final eval
        eval_loss, ppl = evaluate(
            model, compiled_forward, eval_seqs, tokenizer,
            max_seq_len, batch_size, device, vocab_size,
        )
        logger.info(f"  Final eval: loss={eval_loss:.4f}, perplexity={ppl:.2f}")

        # Final checkpoint
        save_checkpoint(model, optimizer, scheduler, num_steps, avg_loss, checkpoint_dir)

    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="ESM-2 MLM training on Neuron")
    parser.add_argument("--model-size", type=str, choices=list(MODEL_CONFIGS.keys()), default="8M")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=10, help="Linear warmup steps")
    parser.add_argument("--fasta", type=str, default=None, help="FASTA file (train+eval split automatically)")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Fraction held out for eval")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate every N steps (0=disable)")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N steps (0=disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        fasta=args.fasta,
        eval_fraction=args.eval_fraction,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
