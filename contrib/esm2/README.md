# ESM-2 on AWS Neuron

Runs HuggingFace ESM-2 protein language models on Neuron devices using `torch.compile`. The script loads pretrained weights, applies tensor parallelism for large models, and performs masked language modeling (predicting masked amino acids in protein sequences).

## Requirements

```
transformers>=4.50.0
```

## Model Sizes

| Size | HuggingFace ID | Params | TP Degree | Instance |
|------|---------------|--------|-----------|----------|
| 8M | facebook/esm2_t6_8M_UR50D | 8M | 1 | trn2.3xlarge |
| 35M | facebook/esm2_t12_35M_UR50D | 35M | 1 | trn2.3xlarge |
| 150M | facebook/esm2_t30_150M_UR50D | 150M | 1 | trn2.3xlarge |
| 650M | facebook/esm2_t33_650M_UR50D | 650M | 1 | trn2.3xlarge |
| 3B | facebook/esm2_t36_3B_UR50D | 3B | 4 | trn2.12xlarge+ |
| 15B | facebook/esm2_t48_15B_UR50D | 15B | 32 | trn2.48xlarge |

## Setup (on Trainium2 instance)

```bash
# Activate your Neuron venv
source /root/workspace/native_venv/bin/activate

# Install transformers if not already present
pip install transformers
```

## Examples

**650M model (single device):**
```bash
torchrun --nproc-per-node 1 run_esm.py --model-size 650M
```

**8M model (quick test):**
```bash
torchrun --nproc-per-node 1 run_esm.py --model-size 8M
```

**3B model (4-way TP):**
```bash
torchrun --nproc-per-node 4 run_esm.py --model-size 3B
```

**Custom sequences:**
```bash
torchrun --nproc-per-node 1 run_esm.py --model-size 650M \
    --sequences "MLKNVQVQLV" "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-size` | str | 650M | ESM-2 model size (8M, 35M, 150M, 650M, 3B, 15B) |
| `--max-seq-len` | int | 128 | Max sequence length (sequences are padded to this) |
| `--sequences` | str[] | (built-in) | Protein sequences to run inference on |

## What It Does (Inference)

1. Loads a pretrained ESM-2 model from HuggingFace
2. Masks the middle amino acid of each input sequence
3. Compiles the model with `torch.compile(backend="neuron")`
4. Runs forward pass and predicts the masked residue
5. Reports the original vs predicted amino acid

## Training

`train_esm.py` fine-tunes ESM-2 using masked language modeling (MLM) with:
- Cosine LR schedule with linear warmup
- Periodic evaluation with perplexity reporting
- Checkpoint saving/loading (resume interrupted training)
- Automatic train/eval split from FASTA input

```bash
# Quick smoke test (8M, built-in sequences)
torchrun --nproc-per-node 1 train_esm.py --model-size 8M --num-steps 50

# 650M model with custom data
torchrun --nproc-per-node 1 train_esm.py --model-size 650M \
    --fasta proteins.fasta --num-steps 1000 --batch-size 4 --max-seq-len 512

# Resume from checkpoint
torchrun --nproc-per-node 1 train_esm.py --model-size 650M \
    --fasta proteins.fasta --num-steps 2000 --resume ./checkpoints
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-size` | 8M | Model to fine-tune |
| `--batch-size` | 4 | Batch size per step |
| `--max-seq-len` | 128 | Sequence length (padded) |
| `--num-steps` | 100 | Total training steps |
| `--lr` | 5e-5 | Peak learning rate |
| `--warmup-steps` | 10 | Linear warmup steps |
| `--fasta` | (built-in) | Path to FASTA file |
| `--eval-fraction` | 0.1 | Fraction held out for eval |
| `--eval-every` | 25 | Evaluate every N steps |
| `--save-every` | 50 | Save checkpoint every N steps |
| `--checkpoint-dir` | ./checkpoints | Where to save checkpoints |
| `--resume` | None | Resume from this checkpoint dir |

### Validated Configurations (trn2.3xlarge, 96GB HBM)

Framework: **PyTorch Native** (`torch.compile(backend="neuron")`). These are
end-to-end MLM fine-tuning **training** steps (forward + backward + optimizer),
not inference.

| Model | Batch | Seq Len | Compile Time | Step Time | Samples/sec |
|-------|-------|---------|--------------|-----------|-------------|
| 8M | 4 | 128 | ~70s | ~0.7s | 5.7 |
| 8M | 8 | 256 | ~90s | ~2.2s | 3.6 |
| 650M | 2 | 128 | ~385s | ~3s | 0.7 |
| 650M | 4 | 512 | ~64s* | ~2.6s | 1.5 |

*NEFF cache hit from prior run

### Using UniRef50 Data

Download UniRef50 FASTA from UniProt and pass it directly:
```bash
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
gunzip uniref50.fasta.gz
torchrun --nproc-per-node 1 train_esm.py --model-size 650M \
    --fasta uniref50.fasta --num-steps 10000 --batch-size 4 --max-seq-len 512
```

## Transferring to Trainium2

From your dev machine, upload the script via S3:
```bash
aws s3 cp examples/esm/run_esm.py s3://YOUR-BUCKET/esm/run_esm.py
```

On the Trainium2 instance:
```bash
aws s3 cp s3://YOUR-BUCKET/esm/run_esm.py /root/workspace/esm/run_esm.py
cd /root/workspace/esm
torchrun --nproc-per-node 1 run_esm.py --model-size 8M
```
