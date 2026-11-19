# Handoff: where the qwen3.5 training work stands (2026-11-19 17:30 UTC)

## Running on the box (`$QWEN35_TRAIN_REMOTE_ROOT`), independent of any pi session
- `labs/run_trace_ft.sh` (nohup) -> status in `runs/trace9b.status`
  - DONE: probe, `trace9b_tierA` train (adapter `runs/trace9b_tierA/adapter_final.pt`),
    `b128_trace9b_tierA` eval = 0.578 on the 128-subset
  - RUNNING since 09:45: `bird4k_trace9b_tierA` full-dev chat-think eval (4 shards,
    seq 4096, max_new 2048); ~94% done at 17:29, ETA ~18:00 UTC, then merge.
  - Check: `env/remote.sh ssh "tail -3 $QWEN35_TRAIN_REMOTE_ROOT/runs/trace9b.status"`
  - Then: `env/remote.sh pull`, report full-dev + clean-gold (gold_audit_dev.json) numbers
    and paired tests vs bird2k_bird9b_r16.

## Results so far (128-question stratified dev subset, seed 0, row-aligned)
| arm | exec |
|---|---:|
| 9B untuned, plain format | 0.242 |
| 9B untuned, chat no-think | 0.477 |
| 9B untuned, chat think (32 rows, ~2.8-3.9k budget) | 0.094 (27/32 never closed </think>) |
| 9B FT plain (bird9b_r16) | 0.555 |
| 9B FT tier-A traces, chat think | 0.578 (think p50 307 tok, 0 unclosed) |
traces vs nothink base p=0.035; traces vs FT plain p=0.74 (tie). Clean-gold: 0.570 vs 0.581.

## Full-dev result (bird4k_trace9b_tierA, DONE 17:58 UTC 11-19, pulled locally)
n=1493 exec 0.534 [0.509,0.560] vs bird2k_bird9b_r16 0.498; McNemar 235/181 p=0.009.
Confound: the baseline ran at seq 2048 and cut 122 prompts. On rows uncut in both: +0.028, p=0.057.
Ignore the clean-gold lift (+0.09): the audit is defined by agreement with gpt-oss-120b,
the same model that produced the traces. By stratum (trace/plain):
agrees_with_teacher 0.858/0.682,
judged_correct 0.263/0.337 (p=0.02), judged_incorrect 0.069/0.241.
The trace model tracks its teacher, and the plain FT wins where the teacher is wrong.
Join key: gold_audit idx = row position in $QWEN35_TRAIN_REMOTE_ROOT/bird/bird_duckdb_dev.json.

## Proposed next step (awaiting user OK)
Control: chat-format, thinking-OFF, SQL-only fine-tune on the SAME 4,429 tier-A rows,
to separate format from traces. Needs lab15 variant (completion = ```sql``` only, prompt
with enable_thinking=False) + run_trace_ft.sh variant + chat-nothink eval.

## Data/provenance decisions
- Teacher: gpt-oss-120b (Bedrock). R1 piloted, not used.
- Opus-scrubbed (rewritten) traces were DELETED at the user's request; S2/S3 hinted
  traces excluded. Trace pool = `runs/trace_pool_train.jsonl` (tier A 5,068 execution-
  verified, tier B 2,072 judge-selected). Trained on tier A only.
- Gold audit (Opus judge): `runs/gold_audit_{train,dev}.json`; ~20% train / ~19% dev gold
  judged bad. Review CSVs: `runs/gold_review_{train,dev}.csv`.
- Local copies of the jsonl used by lab14 live in /tmp (re-fetch from
  $QWEN35_TRAIN_REMOTE_ROOT/bird/ if /tmp is cleared).

## Nothing is committed yet (git status in the worktree).
