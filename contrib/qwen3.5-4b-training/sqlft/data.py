"""DuckDB text2sql data pipeline, built on plain `torch.utils.data` -- no HF Trainer.

Rubric coverage: Dataset / DataLoader / collate_fn / num_workers / packing, and the
label-masking that makes this supervised fine-tuning rather than plain LM pretraining.

    python3 -m sqlft.data --self-test
    python3 -m sqlft.data --stats --seq 512 --seq 1024 --seq 2048
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

DATA_JSONL = pathlib.Path("/work/data/duckdb-text2sql-25k/train.jsonl")
# SQLFT_MODEL selects the checkpoint for every lab (container.sh passes it through), e.g.
# SQLFT_MODEL=/work/models/Qwen3.5-9B. Same family = same tokenizer and vocab.
MODEL_DIR = pathlib.Path(os.environ.get("SQLFT_MODEL", "/work/models/Qwen3.5-4B"))

# Loss is computed only on completion tokens. -100 is torch's cross_entropy ignore_index.
IGNORE = -100

# The GDN NKI kernel fixes chunk_size at 128, so every sequence length we train at must be
# a multiple of 128 or the kernel rejects it.
GDN_CHUNK = 128


@dataclasses.dataclass(frozen=True)
class Row:
    prompt: str
    query: str
    schema: str
    category: str
    # Pre-rendered text for formats other than the plain completion format (e.g. the
    # chat-template reasoning-trace rows built by labs/lab15_trace_data.py). When set,
    # render_prompt/render_completion return them verbatim; `query` stays the SQL alone
    # so filter_parseable still checks it.
    prompt_text: str | None = None
    completion_text: str | None = None


def load_rows(path: pathlib.Path = DATA_JSONL) -> list[Row]:
    rows = []
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            rows.append(Row(d["prompt"], d["query"], d["schema"], d.get("category", ""),
                            d.get("prompt_text"), d.get("completion_text")))
    return rows


def filter_parseable(rows: Sequence[Row], *, report: bool = False) -> list[Row]:
    """Drop rows whose GOLD SQL does not parse in DuckDB.

    ~2.8% of this synthetic dataset's target queries are not valid DuckDB (malformed COPY
    options, unbalanced parens, and at least one row that is Python rather than SQL). The
    whole objective is "emit syntactically correct SQL", so training on those rows teaches
    exactly the failure we are measuring. Applied to TRAIN only -- the eval split keeps its
    noise so the reported ceiling stays honest.
    """
    import duckdb

    con = duckdb.connect()
    kept = []
    for r in rows:
        try:
            con.extract_statements(r.query.strip())
        except duckdb.ParserException:
            continue
        except Exception:
            pass  # catalog/unimplemented -- well-formed SQL, keep it
        kept.append(r)
    if report:
        n = len(rows)
        print(f"filter_parseable: kept {len(kept)}/{n} ({100 * (n - len(kept)) / max(n, 1):.2f}% "
              f"of gold SQL does not parse and was dropped from TRAIN)")
    return kept


def split_rows(rows: Sequence[Row], eval_frac: float = 0.05) -> tuple[list[Row], list[Row]]:
    """Deterministic split. The dataset ships a single `train` split, so we make our own.

    Hashing the row content rather than shuffling by index means the split is stable even
    if the file is reordered or extended -- an eval row can never leak into train later.
    """
    train, evl = [], []
    cut = int(eval_frac * 2**32)
    for r in rows:
        h = int.from_bytes(hashlib.sha256(r.prompt.encode()).digest()[:4], "big")
        (evl if h < cut else train).append(r)
    return train, evl


def render_prompt(row: Row) -> str:
    """Plain completion format (unless the row carries pre-rendered prompt_text).

    Deliberately NOT the chat template: `plain` keeps the prompt/completion boundary exact
    (so masking is unambiguous) and avoids Qwen3.5's thinking mode emitting a <think> block
    before the SQL. The model learns this format during fine-tuning -- that is the point.
    """
    if row.prompt_text is not None:
        return row.prompt_text
    return (
        "-- DuckDB SQL generation\n"
        "-- Schema:\n"
        f"{row.schema.strip()}\n"
        f"-- Question: {row.prompt.strip()}\n"
        "-- Answer:\n"
    )


def render_completion(row: Row) -> str:
    if row.completion_text is not None:
        return row.completion_text
    return row.query.strip()


class SqlDataset(Dataset):
    """Tokenises up front, then serves fixed-length (seq_len) padded examples.

    Rows whose prompt+completion exceed seq_len are DROPPED, not truncated: truncating
    would cut the tail off the SQL and teach the model to emit unparseable fragments,
    which is precisely the metric we are trying to move. `--stats` reports the drop rate.
    """

    def __init__(self, rows: Sequence[Row], tokenizer, seq_len: int, *, report: bool = False):
        if seq_len % GDN_CHUNK:
            raise ValueError(f"seq_len must be a multiple of {GDN_CHUNK} (GDN kernel), got {seq_len}")
        self.seq_len = seq_len
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        self.examples: list[tuple[list[int], int]] = []  # (ids, n_prompt_tokens)
        dropped = 0

        eos = tokenizer.eos_token_id
        for row in rows:
            p_ids = tokenizer(render_prompt(row), add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(render_completion(row), add_special_tokens=False)["input_ids"] + [eos]
            if len(p_ids) + len(c_ids) > seq_len:
                dropped += 1
                continue
            self.examples.append((p_ids + c_ids, len(p_ids)))

        self.dropped = dropped
        if report:
            total = len(rows)
            print(f"seq_len={seq_len}: kept {len(self.examples)}/{total} "
                  f"({100 * dropped / max(total, 1):.1f}% dropped as too long)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ids, n_prompt = self.examples[i]
        pad = self.seq_len - len(ids)

        input_ids = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
        input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.long)

        attention_mask = torch.zeros(self.seq_len, dtype=torch.long)
        attention_mask[: len(ids)] = 1

        # Labels are shifted inside the model, so labels[i] supervises the token AFTER i.
        # Mask the prompt span and the padding; keep only the completion.
        labels = torch.full((self.seq_len,), IGNORE, dtype=torch.long)
        labels[n_prompt : len(ids)] = input_ids[n_prompt : len(ids)]

        assert pad >= 0
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class PackedSqlDataset(Dataset):
    """Concatenate examples back-to-back into full seq_len blocks -- no padding waste.

    Padding at seq_len=512 with a ~200-token example wastes >50% of every step's FLOPs.
    Packing removes that, at the cost of examples bleeding across the block boundary
    (the model can attend from one example into the previous one). Lab 00 measures the
    tokens/step difference so the trade is visible rather than assumed.
    """

    def __init__(self, rows: Sequence[Row], tokenizer, seq_len: int):
        if seq_len % GDN_CHUNK:
            raise ValueError(f"seq_len must be a multiple of {GDN_CHUNK}, got {seq_len}")
        self.seq_len = seq_len
        eos = tokenizer.eos_token_id

        stream_ids: list[int] = []
        stream_sup: list[bool] = []  # is this token supervised (part of a completion)?
        for row in rows:
            p = tokenizer(render_prompt(row), add_special_tokens=False)["input_ids"]
            c = tokenizer(render_completion(row), add_special_tokens=False)["input_ids"] + [eos]
            stream_ids += p + c
            stream_sup += [False] * len(p) + [True] * len(c)

        n_blocks = len(stream_ids) // seq_len
        self.ids = torch.tensor(stream_ids[: n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)
        self.sup = torch.tensor(stream_sup[: n_blocks * seq_len], dtype=torch.bool).view(n_blocks, seq_len)

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ids = self.ids[i]
        labels = torch.where(self.sup[i], ids, torch.full_like(ids, IGNORE))
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": labels,
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Everything is already fixed-length, so this is a pure stack.

    Kept explicit rather than relying on the default collate: the labels tensor must NOT
    be padded with 0 (a real token id) and writing it out makes that impossible to forget.
    """
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_loader(
    rows: Sequence[Row],
    tokenizer,
    seq_len: int,
    batch_size: int,
    *,
    packed: bool = False,
    shuffle: bool = True,
    num_workers: int = 2,
    seed: int = 0,
    report: bool = False,
    distributed: bool = False,
) -> DataLoader:
    """`distributed=True` shards the dataset across ranks with no overlap.

    Seeding each rank's shuffle differently (which the throughput labs do) is fine for a
    benchmark but wrong for a real epoch: ranks would draw overlapping samples, so "one
    epoch" would neither cover the data once nor be reproducible. DistributedSampler
    partitions instead. Remember to call `sampler.set_epoch(e)` or every epoch is identical.
    """
    ds = (PackedSqlDataset(rows, tokenizer, seq_len) if packed
          else SqlDataset(rows, tokenizer, seq_len, report=report))

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=True)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        # shuffle and sampler are mutually exclusive.
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        generator=gen,
        collate_fn=collate,
        # The box has 12 vCPUs shared with up to 8 ranks of neuronx-cc; oversubscribing
        # workers here starves the compiler and can wedge the host.
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader


def infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def get_tokenizer(model_dir: pathlib.Path = MODEL_DIR):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


# --------------------------------------------------------------------------------------


def _self_test() -> int:
    tok = get_tokenizer()
    rows = load_rows()
    print(f"rows: {len(rows)}")

    train, evl = split_rows(rows)
    print(f"split: {len(train)} train / {len(evl)} eval ({100 * len(evl) / len(rows):.1f}%)")
    assert 0.03 < len(evl) / len(rows) < 0.07, "eval fraction way off 5%"

    # Split must be stable across runs and disjoint.
    t2, e2 = split_rows(rows)
    assert [r.prompt for r in evl] == [r.prompt for r in e2], "split is not deterministic"
    assert not ({r.prompt for r in train} & {r.prompt for r in evl}), "train/eval overlap"

    ds = SqlDataset(train[:2000], tok, 1024, report=True)
    assert len(ds) > 0
    ex = ds[0]
    assert ex["input_ids"].shape == (1024,)

    # Masking: supervised positions must be exactly the completion span.
    sup = (ex["labels"] != IGNORE)
    assert sup.any(), "nothing is supervised"
    assert (ex["labels"][sup] == ex["input_ids"][sup]).all(), "labels != input_ids on supervised span"
    first, last = int(sup.nonzero()[0]), int(sup.nonzero()[-1])
    assert sup[first : last + 1].all(), "supervised span is not contiguous"
    assert not sup[:first].any(), "prompt tokens are supervised"
    assert int(ex["attention_mask"].sum()) == last + 1, "attention mask disagrees with content length"
    decoded = tok.decode(ex["input_ids"][sup])
    print(f"supervised span ({sup.sum().item()} tok): {decoded[:120]!r}")

    # Padding must never be supervised.
    assert not sup[last + 1 :].any(), "padding is supervised"

    # Packing wastes no positions.
    packed = PackedSqlDataset(train[:2000], tok, 1024)
    pex = packed[0]
    assert int(pex["attention_mask"].sum()) == 1024, "packed block should be fully occupied"
    pad_frac = 1 - ds[0]["attention_mask"].float().mean().item()
    print(f"padded example wastes {100 * pad_frac:.1f}% of positions; packed wastes 0%")

    batch = collate([ds[0], ds[1]])
    assert batch["input_ids"].shape == (2, 1024)

    # seq_len must be a GDN multiple.
    try:
        SqlDataset(train[:10], tok, 500)
    except ValueError as e:
        print(f"non-multiple-of-128 rejected: {e}")
    else:
        raise AssertionError("seq_len=500 should have been rejected")

    print("DATA SELF-TEST OK")
    return 0


def _stats(seq_lens: list[int]) -> int:
    tok = get_tokenizer()
    rows = load_rows()
    train, _ = split_rows(rows)
    lens = []
    for row in train[:4000]:
        p = len(tok(render_prompt(row), add_special_tokens=False)["input_ids"])
        c = len(tok(render_completion(row), add_special_tokens=False)["input_ids"]) + 1
        lens.append(p + c)
    lens_t = torch.tensor(lens, dtype=torch.float)
    qs = torch.tensor([0.5, 0.9, 0.95, 0.99])
    print("prompt+completion token length over 4000 train rows:")
    print(f"  mean {lens_t.mean():.0f}  max {lens_t.max():.0f}  "
          + "  ".join(f"p{int(q*100)} {torch.quantile(lens_t, q):.0f}" for q in qs))
    for s in seq_lens:
        keep = (lens_t <= s).float().mean().item()
        print(f"  seq_len={s:5d}: keeps {100 * keep:5.1f}% of rows")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--seq", type=int, action="append", default=None)
    args = ap.parse_args()
    if args.stats:
        raise SystemExit(_stats(args.seq or [512, 1024, 2048]))
    raise SystemExit(_self_test())
