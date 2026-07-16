# BS=4 Compiler Cache Validation - 2026-07-16

## Artifact

The complete Native `/tmp` compiler-cache root for the validated BS=4 long
prefill graph was staged through the configurable cache S3 URI under:

```text
trn2-3xl-bs4-c16-s20000-tp4-b512-fused-direct512
```

The artifact contains 664 files, including 66 NEFFs, and occupies 3.4 GiB. It
preserves `hlo_cache`, `neff_cache`, NKI compiler subtrees, and the other cache
metadata. The cache helper scripts are `deploy/cache/{push,pull,inspect}.sh`.

## Restore Validation

The complete artifact was restored into a fresh NVMe directory, mounted at
container `/tmp`, and run with the same image, TP=4/LNC=2 topology, BS=4,
S=20,000, bucket 512, paired-C16, and fused NKI MoE route packing settings.

- No active `neuronx-cc` or `walrus_driver` process appeared during the restored
  run.
- Warmup: 116,077.3 ms / 689.2 aggregate prompt tok/s.
- Timed: 39,788.3 ms / 2,010.6 aggregate prompt tok/s.
- Fingerprint:
  `sum=-5.81636438e+05 norm=1.68448743e+03 top5=[517, 607, 261, 294, 290]`.
- Logits, DeltaNet state, convolution state, and K/V state were all finite.

The timed result matches the original three-run median of 39,757.0 ms /
2,012.2 aggregate prompt tok/s within normal run variation.

## Operational Requirement

The instance role needs `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`,
`s3:ListBucket`, and `s3:GetBucketLocation` on the configured cache bucket and
prefix. Use immutable cache keys for validated builds; only use `--replace` to
refresh a deliberately selected key.
