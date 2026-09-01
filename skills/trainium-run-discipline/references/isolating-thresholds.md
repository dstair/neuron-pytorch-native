# Isolating a size-dependent failure

For any bug described as "it works up to N and fails above" — a hang, an OOB, a wrong result, a
compile failure. The method is domain-agnostic; the worked example is in
`moe-kernel-perf/references/cte-blocking.md`.

The failure mode this prevents: **spending weeks optimizing or escalating the wrong variable**,
because the first quantity someone named happened to correlate.

## 1. Write down the working and failing sets

One row per run you have actually observed. Do not include runs you believe would work.

## 2. Enumerate EVERY quantity that co-varies — including non-numeric ones

The trap is stopping at the obvious number. Enumerate:

- the named quantity (block count, batch, sequence length…)
- everything derived from it (assignments = tokens × K; buffer lengths; tile counts)
- **allocation sizes** that scale with it
- **code paths and configuration branches** selected by it — `if size <= LIMIT: fast_path()`
- **booleans derived from it**, especially synchronization flags. A barrier that switches off
  above a threshold turns a capacity story into a synchronization story, and the two are
  indistinguishable from the size axis alone.
- library-internal branches (a different compute function above some tile size)

Grep the code path for comparisons against constants, not just for the quantity you suspect:

```bash
grep -n "<=\|>=\|< _\|> _" <suspect files> | grep -i "max\|limit\|_SIZE\|THRESH"
```

In our case the numeric enumeration was thorough and still missed the real suspect, because it
was a **boolean derived from** one of the numbers.

## 3. Apply the separation test

A candidate variable is only viable if its values in the working set **do not overlap** its
values in the failing set.

- Overlap ⇒ **refuted**, immediately. Our "block count > 64" theory died because 96 blocks ran
  while 80 hung.
- The same value on both sides ⇒ refuted, and it is worth stating in both directions
  (packed_len 32768 both ran and hung; 28672 hung while 32768 ran).

Most theories die here, cheaply, on data you already have.

## 4. Find a zero-code knob that decouples the survivors

Look for an existing, already-supported parameter that changes one variable while holding
another fixed. In our case `block_size = 256` was accepted end-to-end with no code change and
broke block-count away from token-count.

**Design the discriminator to hold the known-good value fixed and change only the suspect.** Our
first experiment kept the token count at its proven-good value and only tripled the block count —
so a pass was unambiguous.

Order the experiments by cost. Two ~20-minute runs at 4 layers refuted the premise; the
alternative was days of kernel work.

## 5. Run the discriminator BEFORE building the workaround

The workaround is usually more expensive than the experiment and sometimes unnecessary. Ours
would have been a ~32× FLOP regression, correctly abandoned once a config change opened the same
capability.

## 6. State honestly what remains confounded

If two or three variables survive and are confounded **by construction** (one determines the
other), say so and name the experiment that would separate them — do not pick the most
convenient survivor and call it the root cause.

Note which direction is even testable: if the fast path cannot exist above the threshold (an
SBUF limit, say), the confound can only be broken by forcing the *slow* path below it.

## 7. Only then escalate

An escalation should name a variable that survived the separation test, list what was eliminated,
and give a reproduction. "Above 2048 tokens per call it deadlocks" was crisp — and still wrong,
because step 2 was incomplete. An escalation that turns out to be your own conditional barrier is
worse than no escalation.

## Checklist

```
[ ] working/failing sets tabulated from observed runs only
[ ] all co-varying quantities listed
[ ] derived allocations listed
[ ] CODE PATHS / branches selected by size listed
[ ] BOOLEANS (esp. barriers, sync flags) derived from size listed
[ ] separation test applied; overlapping candidates struck out
[ ] zero-code knob identified that decouples the survivors
[ ] discriminator holds the known-good value fixed
[ ] cheapest discriminator run FIRST, before any workaround
[ ] remaining confounds stated, with the experiment that would break them
```
