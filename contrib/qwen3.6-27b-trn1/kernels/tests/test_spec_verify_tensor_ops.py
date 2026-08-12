"""Validate the EXACT tensor ops in Qwen3_5LinearAttention._decode_spec_verify
(rollback gather + trajectory persist) against a clean sequential replay.

Models the per-token DeltaNet step as a simple decayed fold on a [K,V] state
(stand-in for the proven deltanet_full kernel) so we exercise the bookkeeping
TENSOR ops — index_select rollback, the is_first select-mask, trajectory cat,
prev_verify_start update — with the same shapes/dtypes the real method uses.
A wrong gather index or mask gives a wrong state, so this catches shape/index
bugs before any device compile.
"""
import torch

K = 8           # tiny stand-in for v_heads*k_dim
V = 4           # tiny stand-in for v_dim


def step(state, token_val):
    # order-sensitive decayed update, like the real recurrence
    return state * 0.9 + token_val


def clean_replay(tokens):
    s = torch.zeros(K, V)
    for tv in tokens:
        s = step(s, tv)
    return s


class FakeLayer:
    """Mirror of the spec-verify state mgmt with the real tensor ops."""
    def __init__(self, N):
        self.N = N
        self.recurrent_state = torch.zeros(1, K, V)        # prefill-produced
        self.traj_recurrent = torch.zeros(N, K, V)
        self.prev_verify_start = torch.full((1,), -1, dtype=torch.long)

    def verify(self, positions, token_vals):
        N = self.N
        start = positions[0:1].to(torch.long)
        k = (start - self.prev_verify_start).clamp(min=1, max=N)
        is_first = (self.prev_verify_start < 0)
        gather_idx = (k - 1).clamp(min=0, max=N - 1)
        traj_pick = self.traj_recurrent.index_select(0, gather_idx.reshape(1))[0]
        sel = is_first.to(torch.float32).reshape(1, 1)
        state = sel * self.recurrent_state[0] + (1.0 - sel) * traj_pick

        new_states = []
        for t in range(N):
            state = step(state, token_vals[t])
            new_states.append(state.unsqueeze(0))
        self.traj_recurrent = torch.cat(new_states, dim=0)
        self.prev_verify_start = start
        self.recurrent_state = self.traj_recurrent[N - 1:N]
        # return the state trajectory the runner would (implicitly) pick from
        return self.traj_recurrent


def run(num_spec=3, seed=0, steps=10):
    torch.manual_seed(seed)
    N = num_spec + 1
    layer = FakeLayer(N)
    accepted = []
    confirmed_pos = 0
    for st in range(steps):
        p = confirmed_pos
        cand = [torch.randn(K, V) for _ in range(N)]
        traj = layer.verify(torch.tensor([p + i for i in range(N)]), cand)
        k_acc = int(torch.randint(1, N + 1, (1,)).item())
        accepted.extend(cand[:k_acc])
        confirmed_pos += k_acc
        # next step will restore traj[k_acc-1]; must equal clean replay
        restored = traj[k_acc - 1]
        truth = clean_replay(accepted)
        err = (restored - truth).abs().max().item()
        ok = err < 1e-5
        print(f"step {st:2d} start={p} N={N} accepted={k_acc} err={err:.2e} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            print("FAIL"); return False
    print("PASS")
    return True


if __name__ == "__main__":
    ok = True
    for sd in range(5):
        print(f"=== seed {sd} ===")
        ok &= run(seed=sd)
    print("ALL PASS" if ok else "SOME FAILED")
