"""Benchmark DeltaNet NKI kernel at 27B model scale (48 heads, K=V=128)."""
import time
import torch


def benchmark_recurrent(device, num_heads=48, num_iters=10, warmup=3):
    """Benchmark single-token decode step."""
    from deltanet_recurrent import nki_deltanet_recurrent_step, K_DIM, V_DIM

    state = torch.randn(num_heads * K_DIM, V_DIM, dtype=torch.float32, device=device)
    query = torch.randn(num_heads, K_DIM, dtype=torch.float32, device=device)
    key = torch.randn(num_heads, K_DIM, dtype=torch.float32, device=device)
    value = torch.randn(num_heads, V_DIM, dtype=torch.float32, device=device)
    g = torch.randn(num_heads, 1, dtype=torch.float32, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(num_heads, 1, dtype=torch.float32, device=device))

    # Warmup
    for _ in range(warmup):
        nki_deltanet_recurrent_step(state, query, key, value, g, beta)
    torch.neuron.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(num_iters):
        state, _ = nki_deltanet_recurrent_step(state, query, key, value, g, beta)
    torch.neuron.synchronize()
    elapsed = time.time() - t0

    per_step_ms = elapsed / num_iters * 1000
    print(f"Recurrent decode: {per_step_ms:.2f} ms/step ({num_heads} heads, K={K_DIM}, V={V_DIM})")
    return per_step_ms


def benchmark_prefill(device, num_heads=48, seq_len=64, warmup=2, num_iters=5):
    """Benchmark prefill (full sequence)."""
    from deltanet_prefill import nki_deltanet_prefill, K_DIM, V_DIM

    state = torch.randn(num_heads * K_DIM, V_DIM, dtype=torch.float32, device=device)
    query = torch.randn(num_heads * seq_len, K_DIM, dtype=torch.float32, device=device)
    key = torch.randn(num_heads * seq_len, K_DIM, dtype=torch.float32, device=device)
    value = torch.randn(num_heads * seq_len, V_DIM, dtype=torch.float32, device=device)
    g = torch.randn(num_heads * seq_len, 1, dtype=torch.float32, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(num_heads * seq_len, 1, dtype=torch.float32, device=device))

    # Warmup
    for _ in range(warmup):
        nki_deltanet_prefill(state, query, key, value, g, beta, num_heads, seq_len)
    torch.neuron.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(num_iters):
        nki_deltanet_prefill(state, query, key, value, g, beta, num_heads, seq_len)
    torch.neuron.synchronize()
    elapsed = time.time() - t0

    per_call_ms = elapsed / num_iters * 1000
    tok_per_sec = seq_len / (per_call_ms / 1000)
    print(f"Prefill: {per_call_ms:.1f} ms ({num_heads} heads, seq={seq_len}, {tok_per_sec:.0f} tok/s)")
    return per_call_ms


if __name__ == "__main__":
    import torch_neuronx
    device = torch.device("neuron")
    print("=== DeltaNet NKI Kernel Benchmark (27B scale) ===\n")

    # Correctness at 48 heads
    from deltanet_recurrent import nki_deltanet_recurrent_step, K_DIM, V_DIM
    NUM_HEADS = 48
    state = torch.randn(NUM_HEADS * K_DIM, V_DIM, dtype=torch.float32)
    query = torch.randn(NUM_HEADS, K_DIM, dtype=torch.float32)
    key = torch.randn(NUM_HEADS, K_DIM, dtype=torch.float32)
    value = torch.randn(NUM_HEADS, V_DIM, dtype=torch.float32)
    g_t = torch.randn(NUM_HEADS, 1, dtype=torch.float32) * 0.1
    beta_t = torch.sigmoid(torch.randn(NUM_HEADS, 1, dtype=torch.float32))

    # Quick correctness check
    ref_state = state.clone().reshape(NUM_HEADS, K_DIM, V_DIM)
    for h in range(NUM_HEADS):
        s = ref_state[h] * g_t[h].exp()
        k_vec = key[h].unsqueeze(1)
        kv_mem = s.T @ k_vec
        v_vec = value[h].unsqueeze(1)
        delta = (v_vec - kv_mem) * beta_t[h]
        ref_state[h] = s + k_vec @ delta.T

    nki_s, _ = nki_deltanet_recurrent_step(
        state.to(device), query.to(device), key.to(device),
        value.to(device), g_t.to(device), beta_t.to(device),
    )
    diff = (nki_s.cpu() - ref_state.reshape(NUM_HEADS * K_DIM, V_DIM)).abs().max().item()
    print(f"48-head correctness: max_diff={diff:.6f} {'PASS' if diff < 0.01 else 'FAIL'}\n")

    # Benchmarks
    print("--- Decode (single token) ---")
    benchmark_recurrent(device, num_heads=48, num_iters=20)

    print("\n--- Prefill ---")
    benchmark_prefill(device, num_heads=48, seq_len=16)
    benchmark_prefill(device, num_heads=48, seq_len=64)
