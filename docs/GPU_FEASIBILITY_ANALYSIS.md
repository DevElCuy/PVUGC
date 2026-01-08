# GPU Feasibility Analysis for PVUGC Circuits

This document analyzes whether enterprise GPUs (H100, A100) can practically handle full PVUGC circuit CRS generation, and provides baseline metrics for tracking algorithmic improvements.

## Executive Summary

**Full circuit CRS generation is computationally infeasible with current throughput.** Benchmarks show ~65-95 pairs/sec for MNT4-298 on GTX 1660 (6GB). The bottleneck is computational complexity, not GPU memory.

| Curve | GTX 1660 (measured) | H100 (projected) | Status |
|-------|---------------------|------------------|--------|
| MNT4-298 | ~65-95 pairs/sec → **~83 days** | ~520-950/sec → **~8-15 days** | Implemented |
| MNT6-298 | ~20-24 pairs/sec → **~1 year** | ~160-240/sec → **~1-2 months** | Implemented |
| BW6-761 | ~17-21 pairs/sec → **~1.2 years** | ~140-200/sec → **~1.5-2 months** | Implemented |

**Critical observation:** Recent algorithmic optimizations (precomputing `omega^d * inv_n_one_minus_omega[d]`) improved MNT4-298 throughput by ~3-4×. MNT6/BW6 numbers are from prior measurements and may improve with the same optimizations.

---

## GPU Implementation Status

| Curve | Sparse Quotient GPU | MSM GPU | Notes |
|-------|---------------------|---------|-------|
| MNT4-298 | ✅ Yes | ✅ Yes | Full support, benchmarked |
| MNT6-298 | ✅ Yes | ✅ Yes | Full support, benchmarked |
| BW6-761 | ✅ Yes | ✅ Yes | Full support, benchmarked |

All kernels verified available via `benchmark_gpu_scalability` test (January 2026).

See `GPU_plan.md` for the implementation roadmap.

---

## Circuit Complexity Overview

The sparse quotient computation generates coefficient pairs from the Cartesian product of R1CS matrix columns:

```
pairs ≈ |columns_in_A| × |columns_in_B| ≈ O(constraints²)
```

| Circuit Configuration | Constraints | Coefficient Pairs | Scaling Factor |
|-----------------------|-------------|-------------------|----------------|
| MNT4/MNT6 Full | ~31,000 | ~680 million | 1.0× |
| BW6-761 Full | ~50,000+ | ~2.5 billion | ~3.7× |
| Skip Verifier (MNT) | ~14,000 | ~160 million | 0.24× |
| Trivial Test | ~10 | ~100 | negligible |

**Key insight:** Halving constraints only reduces pairs by 75% due to O(n²) scaling.

---

## Memory Requirements

### Per-Pair Memory Formula

From `src/pvugc_outer.rs`:
```rust
bytes_per_pair = (max_col_a * 40) + (max_col_b * 40) + (max_diag * 4) + (max_diag * 40)
```

Where:
- `max_col_a`, `max_col_b` = maximum non-zero entries in any column of matrices A, B
- `max_diag = min(max_col_a, max_col_b)`
- 40 bytes = size of a CGBN scalar (320-bit for MNT4-298)

### Observed Values

| Circuit | max_col | bytes_per_pair | Source |
|---------|---------|----------------|--------|
| MNT4/MNT6 ~31K constraints | ~2,000 | ~161 KB | docs/MNT_GPU_ACCELERATION.md |
| Trivial (~10 constraints) | ~10 | ~1.2 KB | estimated |

---

## Enterprise GPU Capacity

### Memory Analysis

| GPU | Total Memory | Usable (80%) | Pairs per Batch | Batches for 680M Pairs |
|-----|--------------|--------------|-----------------|------------------------|
| GTX 1660 (reference) | 6 GB | 4.8 GB | ~30,000 | ~22,700 |
| RTX 4090 | 24 GB | 19.2 GB | ~120,000 | ~5,700 |
| **A100 40GB** | 40 GB | 32 GB | ~200,000 | ~3,400 |
| **A100 80GB** | 80 GB | 64 GB | ~400,000 | ~1,700 |
| **H100 80GB** | 80 GB | 64 GB | ~400,000 | ~1,700 |
| H100 NVL 94GB | 94 GB | 75 GB | ~470,000 | ~1,450 |

**Conclusion:** Memory is NOT the bottleneck. Even consumer GPUs can process the workload with batching.

### Compute Time Projections for Enterprise GPUs

Based on measured GTX 1660 throughput (~20 pairs/sec), projecting to enterprise GPUs:

| GPU | Est. Speedup vs GTX 1660 | Est. Throughput | 680M Pairs (MNT4) | 680M Pairs (BW6) |
|-----|--------------------------|-----------------|-------------------|------------------|
| GTX 1660 (measured) | 1× | ~20 pairs/sec | **~1.08 years** | **~1.20 years** |
| RTX 4090 | ~3-4× | ~60-80 pairs/sec | ~100-130 days | ~110-145 days |
| **A100 80GB** | ~5-8× | ~100-160 pairs/sec | ~50-80 days | ~55-90 days |
| **H100 80GB** | ~8-12× | ~160-240 pairs/sec | ~33-50 days | ~37-55 days |

**Critical finding:** Even with an H100 at optimistic 12× speedup, full circuit CRS generation still requires **~1-2 months**. This is fundamentally impractical for iterative development.

**Note:** These projections assume:
- Linear scaling with GPU compute capability (may not hold due to memory bandwidth limits)
- No kernel launch overhead reduction (larger batches may help)
- Same algorithmic approach (no optimizations)

The ~1000× gap between estimated and measured throughput suggests the bottleneck is algorithmic, not hardware. Enterprise GPUs may provide only modest improvements without kernel optimization.

---

## Compute Time Analysis

### Measured CGBN Throughput (January 2026)

Benchmark run on GTX 1660 (6GB) with `benchmark_gpu_scalability` test:

```
GPU Memory: Total 5926MB, Available 5443MB, Target 4354MB (80%)
Domain size: 1024, NNZ per column: 100
```

#### MNT4-298 Measured Results (with precomputation optimization)

| Pairs | Time (ms) | Throughput (pairs/sec) | bytes_per_pair |
|-------|-----------|------------------------|----------------|
| 100 | 1,054 | **94.88** | 12,400 |
| 1,000 | 14,424 | **69.33** | 12,400 |
| 10,000 | 154,206 | **64.85** | 12,400 |

#### MNT6-298 / BW6-761 (prior measurements, before optimization)

| Curve | Pairs | Time (ms) | Throughput (pairs/sec) | bytes_per_pair |
|-------|-------|-----------|------------------------|----------------|
| MNT6-298 | 100 | 4,086 | **24.47** | 12,400 |
| MNT6-298 | 1,000 | 48,709 | **20.53** | 12,400 |
| BW6-761 | 100 | 4,686 | **21.34** | 14,800 |
| BW6-761 | 1,000 | 56,264 | **17.77** | 14,800 |

**Key finding:** The precomputation optimization (computing `omega^d * inv_n_one_minus_omega[d]` on CPU) improved MNT4-298 throughput by ~3-4× compared to prior measurements. MNT6/BW6 have the same optimization applied but need re-benchmarking to confirm similar gains.

### Time Projections Based on Measured Throughput

Using measured MNT4-298 (~65 pairs/sec) and prior MNT6/BW6 (~20/18 pairs/sec):

| Circuit | Pairs | MNT4-298 (65/s) | MNT6-298 (20/s) | BW6-761 (18/s) |
|---------|-------|-----------------|-----------------|----------------|
| **Full (680M)** | 680,000,000 | **~121 days** | **~1.08 years** | **~1.20 years** |
| Skip Verifier (160M) | 160,000,000 | ~28 days | ~93 days | ~103 days |
| 1M pairs | 1,000,000 | ~4.3 hours | ~14 hours | ~15.4 hours |
| 10K pairs | 10,000 | ~2.6 min | ~8.3 min | ~9.3 min |
| 1K pairs | 1,000 | ~15 sec | ~50 sec | ~56 sec |
| 100 pairs | 100 | ~1 sec | ~4-5 sec | ~4-5 sec |

**Note:** MNT4-298 numbers reflect the precomputation optimization. MNT6/BW6 should see similar ~3-4× improvement once re-benchmarked.

---

## The Fundamental Problem: O(n²) Scaling

```
Time ∝ constraints² × time_per_pair
```

| Constraints | Pairs | Relative Time |
|-------------|-------|---------------|
| 31,000 | 961M | 1.0× |
| 20,000 | 400M | 0.42× |
| 10,000 | 100M | 0.10× |
| 5,000 | 25M | 0.026× |
| 1,000 | 1M | 0.001× |

**Even with perfect GPU utilization, O(n²) scaling dominates.** Going from 31K to 10K constraints provides 10× speedup, but still leaves ~15-30 minutes for a single CRS generation.

---

## Practical Recommendations

### For Development/Testing

| Goal | Configuration | Expected Time (GTX 1660) |
|------|---------------|--------------------------|
| GPU kernel correctness | `TRIVIAL_TEST_CIRCUIT=1` (~100 pairs) | ~4-5 seconds |
| Performance benchmarking | Benchmark test (~1K pairs) | ~50 seconds |
| Small-scale integration | ~10K pairs | ~8-10 minutes |

### For Production CRS Generation

| Approach | Notes |
|----------|-------|
| **Not feasible with current throughput** | 680M pairs at 20/sec = ~1 year |
| Multi-GPU parallelization | Would need ~1000× improvement to be practical |
| Precomputed CRS | Generate once offline, reuse |
| Algorithmic breakthrough required | See investigation items below |

### Identified Bottlenecks (January 2026)

Analysis of `sparse_quotient_mnt4_298_cgbn.cu` reveals severe GPU underutilization:

#### 1. One Block Per Pair (Critical)

```cuda
num_blocks = num_pairs;  // Line 575
```

The kernel launches **one CUDA block per pair**. For 100 pairs = 100 blocks × 256 threads = 25,600 threads. A GTX 1660 has 1408 CUDA cores - this severely underutilizes the GPU when processing small batches.

| Pairs | Blocks | Threads | GPU Utilization |
|-------|--------|---------|-----------------|
| 100 | 100 | 25,600 | ~2% |
| 1,000 | 1,000 | 256,000 | ~18% |
| 10,000 | 10,000 | 2,560,000 | ~180% (good) |

**Fix:** Process multiple pairs per block, or parallelize the inner loop across blocks.

#### 2. Serial O(n²) Inner Loops (Critical)

Each block processes one pair with nested loops (lines 294-339, 363-395):

```
Phase 1: for idx_u in 0..n_u:      # 100 iterations
           for idx_v in 0..n_v:    # 100 iterations each
             field_mul, field_add, etc.

Phase 2: for idx_v in 0..n_v:      # 100 iterations
           for idx_u in 0..n_u:    # 100 iterations each
             field_mul, field_sub, etc.
```

With only 32 CGBN instances per block (256 threads / 8 TPI), work is divided but still largely sequential: **~20,000 loop iterations per pair**.

**Fix:** Parallelize inner loop across threads/warps, not just outer loop.

#### 3. Memory Allocation Per Call (Moderate)

For **every batch call**, the kernel (lines 494-596):
- `cudaMalloc` for all buffers (15+ allocations)
- `cudaMemcpy` all input data (sparse matrices, domain tables)
- Run kernel
- `cudaMemcpy` all output data
- `cudaFree` all buffers

This creates **massive host-device transfer overhead**, especially for small batches.

**Fix:** Persistent GPU memory allocation, reuse across calls. Cache domain tables on GPU.

#### 4. Work Per Pair is O(n_u × n_v)

With 100 NNZ per column, each pair performs:
- 100 × 100 = 10,000 field multiplications
- Each `field_mul` = wide multiply (320×320→640 bit) + modular reduction
- × 2 phases = ~20,000 CGBN operations
- Each CGBN operation uses 8 threads cooperatively

**Estimated ~1,000,000+ GPU operations per pair** explains the low throughput.

#### Bottleneck Summary

| Bottleneck | Impact | Estimated Speedup if Fixed |
|------------|--------|---------------------------|
| One block per pair | Severe GPU underutilization | 10-50× |
| Serial inner loop | Sequential bottleneck | 5-20× |
| Memory alloc/copy per call | High overhead | 2-5× |
| No domain table caching | Redundant transfers | 1.5-2× |
| **Combined potential** | | **100-1000×** |

Fixing these bottlenecks could bring throughput from ~20 pairs/sec to ~20,000+ pairs/sec, making the original estimates achievable.

### Algorithmic Improvements to Track

1. **Sparse matrix exploitation**: Skip zero columns entirely
2. **FFT-based convolution**: Could reduce O(n²) to O(n log n) for certain structures
3. **Incremental computation**: Reuse partial results across similar circuits
4. **Mixed precision**: Use lower precision for intermediate results where safe
5. **Parallelism restructuring**: Process multiple pairs concurrently within kernel

---

## Benchmark Metrics for Tracking Progress

The benchmark test (`tests/benchmark_gpu_scalability.rs`) outputs structured metrics:

```
=== GPU Scalability Benchmark Result ===
curve: MNT4-298
num_pairs: 1000
num_columns_a: 32
num_columns_b: 32
nnz_per_col: 100
domain_size: 1024
bytes_per_pair: 12400
gpu_memory_total_mb: 5926
gpu_memory_available_mb: 5418
time_ms: 48625
throughput_pairs_per_sec: 20.57
=========================================
```

### Key Metrics to Track

| Metric | Unit | What it Measures |
|--------|------|------------------|
| `throughput_pairs_per_sec` | pairs/s | Raw GPU compute throughput |
| `bytes_per_pair` | bytes | Memory efficiency |
| `time_ms` | ms | End-to-end latency |
| `gpu_memory_mb` | MB | Peak GPU memory usage |

### Projection Formula

```
estimated_time_sec = total_pairs / throughput_pairs_per_sec
estimated_memory_mb = batch_size × bytes_per_pair / (1024 × 1024)

# Example with measured values:
# 680M pairs / 20 pairs/sec = 34,000,000 seconds ≈ 1.08 years
```

---

## Appendix: Detailed Memory Breakdown

### Per-Pair Allocations (MNT4-298)

| Buffer | Size | Formula |
|--------|------|---------|
| Column A coefficients | max_col_a × 40 bytes | CGBN scalars |
| Column B coefficients | max_col_b × 40 bytes | CGBN scalars |
| Diagonal indices | max_diag × 4 bytes | uint32 indices |
| Diagonal products | max_diag × 40 bytes | CGBN scalars |
| **Total per pair** | **(max_col_a + max_col_b) × 40 + max_diag × 44** | |

### Measured Values (January 2026)

From `benchmark_gpu_scalability` with nnz_per_col=100, domain_size=1024:

| Curve | bytes_per_pair | Notes |
|-------|----------------|-------|
| MNT4-298 | 12,400 | 320-bit CGBN scalars |
| MNT6-298 | 12,400 | 320-bit CGBN scalars |
| BW6-761 | 14,800 | 384-bit CGBN scalars |

**Note:** These are smaller than initial estimates because the benchmark uses smaller column sizes (10×10 columns with 100 NNZ each) compared to full circuits (~2000 max_col).

---

## Running the Benchmark

The `tests/benchmark_gpu_scalability.rs` test generates structured output for tracking:

```bash
# Default: test 100, 1K, 10K, 100K pairs
cargo test --features gpu --release --test benchmark_gpu_scalability -- --ignored --nocapture

# Quick sanity check (100 pairs, seconds)
BENCHMARK_QUICK=1 cargo test --features gpu --release --test benchmark_gpu_scalability -- --ignored --nocapture

# Specific scale
BENCHMARK_PAIRS=50000 cargo test --features gpu --release --test benchmark_gpu_scalability -- --ignored --nocapture

# Extended benchmark for accurate projections (up to 1M pairs)
cargo test --features gpu --release --test benchmark_gpu_scalability benchmark_gpu_extended -- --ignored --nocapture
```

### Interpreting Results

The benchmark outputs structured results for each curve at each scale:

```
========================================================================
  SCALE: 1000 pairs
========================================================================

--- MNT4-298 @ 1000 pairs ---
  [MNT4-298] Setting up 1000 pairs with 32x32 columns, 100 entries/col, domain=1024...
  Running GPU kernel...
[sparse_quotient MNT4] pairs=1000, max_col_a=100, max_col_b=100, acc_u=3.8MB, acc_v=3.8MB

=== GPU Scalability Benchmark Result ===
curve: MNT4-298
num_pairs: 1000
num_columns_a: 32
num_columns_b: 32
nnz_per_col: 100
domain_size: 1024
bytes_per_pair: 12400
gpu_memory_total_mb: 5926
gpu_memory_available_mb: 5418
time_ms: 48625
throughput_pairs_per_sec: 20.57
=========================================
```

**Key insight from measurements:** At ~20 pairs/sec throughput, full circuit CRS generation is not practical. The benchmark reveals a fundamental bottleneck that requires algorithmic investigation.

---

## References

- `src/pvugc_outer.rs`: Sparse quotient GPU integration
- `sppark-msm/src/sparse_quotient_mnt4_298_cgbn.cu`: CUDA kernel implementation
- `docs/CIRCUIT_SIZE_OPTIONS.md`: Circuit configuration options
- `tests/benchmark_gpu_scalability.rs`: Scalability benchmark test
