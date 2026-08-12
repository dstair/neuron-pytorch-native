"""Validate the DeltaNet spec-verify ROLLBACK bookkeeping (algorithm only).

EAGLE3 spec decode feeds the target N = num_spec+1 tokens of ONE sequence per
decode step to VERIFY them. DeltaNet is a sequential recurrence (state_{t} depends
on state_{t-1}), so verifying N tokens = rolling the recurrence forward N steps
from the confirmed state. After the forward, rejection sampling accepts k of the N
(1 <= k <= N). The recurrent state has NO per-position addressing (unlike a KV
cache), so we must EXPLICITLY roll it back to the accepted prefix.

Design under test (traceable, no Python-int control flow on the hot path):
  Per layer, persist between steps:
    - confirmed recurrent state S
    - verify trajectory T[0..N-1] = states AFTER processing each verify token
    - verify_start_pos
  Each verify step at start position p (= positions[0]):
    1. k = p - verify_start_pos          # accepted count from the PREVIOUS verify
    2. if not first step: S = T[k-1]      # roll back to the accepted prefix
    3. roll recurrence forward over the N current tokens from S, saving T'
    4. verify_start_pos = p ; persist T'
The invariant this must satisfy: after any sequence of (verify N, accept k) steps,
the confirmed state used at the START of a step equals a CLEAN sequential recurrence
over exactly the accepted token stream so far.

This test models the "recurrence" as a trivial running fold (sum of token values)
so the bookkeeping is checked independently of the real gated-delta-rule math
(which is already proven per-token by deltanet_full). If the fold matches a clean
replay over accepted tokens, the index logic is correct.
"""
import numpy as np


def recurrence_step(state, token):
    """Stand-in for one DeltaNet step: any associative-ish update is fine for
    checking bookkeeping. Use state' = state*0.9 + token (order-sensitive, like
    the real decayed recurrence) so a wrong rollback index gives a wrong result."""
    return state * 0.9 + token


def clean_replay(tokens):
    """Ground truth: sequential recurrence over the exact accepted token stream."""
    s = 0.0
    for t in tokens:
        s = recurrence_step(s, t)
    return s


def run(num_spec=3, seed=0, steps=12):
    rng = np.random.default_rng(seed)
    N = num_spec + 1

    # --- model-side persistent state (what the layer would hold) ---
    S = 0.0                      # confirmed recurrent state
    traj = np.zeros(N)           # verify trajectory (states after each verify token)
    verify_start = -1            # sentinel: no prior verify
    first = True

    # --- ground-truth bookkeeping ---
    accepted_stream = []         # the real accepted token stream
    confirmed_pos = 0            # absolute position of next token to confirm

    for step in range(steps):
        p = confirmed_pos        # this verify step starts here (positions[0])

        # 1+2: roll back to the accepted prefix of the PREVIOUS verify
        if not first:
            k = p - verify_start            # accepted count from previous step
            assert 1 <= k <= N, f"k={k} out of range"
            S = traj[k - 1]                 # restore confirmed state

        # The N candidate tokens this step proposes (draft+bonus). The accepted
        # prefix of these becomes real; model can't know which yet.
        cand = rng.standard_normal(N)

        # 3: roll recurrence forward over the N candidates from S, save trajectory
        s = S
        new_traj = np.zeros(N)
        for i in range(N):
            s = recurrence_step(s, cand[i])
            new_traj[i] = s
        traj = new_traj
        verify_start = p
        first = False

        # --- simulate acceptance: 1..N tokens accepted this step ---
        k_acc = int(rng.integers(1, N + 1))
        accepted_stream.extend(cand[:k_acc].tolist())
        confirmed_pos += k_acc

        # --- CHECK: at the start of the NEXT step we'll restore traj[k_acc-1];
        #     that must equal a clean replay over the full accepted stream. ---
        restored = traj[k_acc - 1]
        truth = clean_replay(accepted_stream)
        err = abs(restored - truth)
        ok = err < 1e-9
        print(f"step {step:2d}: start_pos={p} N={N} accepted={k_acc} "
              f"confirmed_pos={confirmed_pos} restored={restored:.6f} "
              f"truth={truth:.6f} err={err:.2e} {'OK' if ok else 'FAIL'}")
        if not ok:
            print("FAIL"); return False
    print("PASS")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_spec", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=12)
    a = p.parse_args()
    all_ok = True
    for sd in range(5):
        print(f"=== seed {sd} ===")
        all_ok &= run(num_spec=a.num_spec, seed=sd, steps=a.steps)
    print("ALL PASS" if all_ok else "SOME FAILED")
