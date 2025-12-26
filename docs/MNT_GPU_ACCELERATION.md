# MNT4/MNT6-298 GPU Acceleration

## Overview

This document covers the GPU acceleration strategy for MNT4-298 and MNT6-298 curves, including:
1. MSM (Multi-Scalar Multiplication) CGBN kernels
2. Sparse quotient coefficient computation kernels
3. Testing and validation

## Priority Alignment (Lean e2e)

Lean e2e uses the MNT4-298/MNT6-298 cycle (DefaultCycle). Within that scope,
the highest-impact pending items are:

- **MNT6 G1 MSM dispatch in the lean prover** (`msm_backend::msm_g1` still routes only MNT4).
- **Pippenger MSM for CGBN kernels** (replace serial double-and-add).
- **G2 MSM** remains CPU-only (optional, lower impact).

SP1 e2e uses the BLS12-377/BW6-761 cycle; the top missing item there is the BW6
sparse-quotient GPU path (see `GPU_plan.md`).
See `docs/GPU_INDEX.md` for the full GPU documentation tree.

---

## Problem Statement

The lean prover PVUGC setup computes sparse quotient bases H_{ij} in `src/pvugc_outer.rs` (lines 603-754). This was the primary bottleneck:

- **680 million pairs** (i,j) to process
- Each pair involves sparse matrix lookups and small MSMs
- Original CPU rate: Very slow
- Estimated CPU time: Several hours

The computation has two phases:
1. **Diagonal Q-vector** (lines 500-575): Already efficient (~25s) using FFT convolution
2. **Main pair loop** (lines 603-754): The bottleneck - O(n²) pairs with irregular work per pair

---

## Solution: GPU CGBN Kernels

### Implemented Kernels

| Kernel | File | Purpose |
|--------|------|---------|
| MSM MNT4-298 | `msm_mnt4_298_cgbn.cu` | Multi-scalar multiplication for MNT4-298 curve |
| MSM MNT6-298 | `msm_mnt6_298_cgbn.cu` | Multi-scalar multiplication for MNT6-298 curve |
| Sparse Quotient MNT4 | `sparse_quotient_mnt4_298_cgbn.cu` | Coefficient accumulation for sparse quotient bases |
| Sparse Quotient MNT6 | `sparse_quotient_mnt6_298_cgbn.cu` | Coefficient accumulation for sparse quotient bases |

### Architecture

- Uses NVIDIA CGBN (Cooperative Group Big Numbers) library
- TPI (Threads Per Instance): 8 threads cooperate per 320-bit field element
- Uses Montgomery representation for field arithmetic
- XYZZ extended Jacobian coordinates for point operations

---

## Sparse Quotient Algorithm

### Original CPU Algorithm

For each pair (i, j) in 680M active pairs:
```
1. Load sparse column entries:
   rows_u = col_a[i]  // sparse: [(row_idx, value), ...]
   rows_v = col_b[j]  // sparse: [(row_idx, value), ...]

2. For each (k, val_u) in rows_u:
   For each (m, val_v) in rows_v:
     prod = val_u * val_v
     if k == m:
       # Diagonal: accumulate for Q-vector MSM
       diag_terms.push((k, prod))
     else:
       # Off-diagonal: field arithmetic for Lagrange basis MSM
       inv_denom = -ω^{-m} * inv_n_one_minus_omega[(k-m) mod n]
       common = prod * inv_denom
       acc_u[idx_u] += common * ω^m
       acc_v[idx_v] -= common * ω^k

3. Build MSM task from accumulated coefficients
4. Run small MSM (uses GPU via msm_with_gpu_fallback)
5. Store non-zero result as (i, j, H_ij)
```

### Why CPU Was Slow

1. **680M loop iterations** with irregular access patterns
2. **Small MSMs per pair** - GPU launch overhead dominates for small MSMs
3. **Sparse matrix lookups** - cache-unfriendly memory access
4. **Field arithmetic in inner loop** - MNT4/MNT6 298-bit field ops on CPU

### GPU Kernel Design

**Two-Phase Algorithm** (solves race conditions):
- **Phase 1:** Each CGBN instance owns `idx_u` values, loops over ALL `idx_v` → writes `acc_u` exclusively
- **Phase 2:** Each CGBN instance owns `idx_v` values, loops over ALL `idx_u` → writes `acc_v` exclusively

This doubles compute but ensures deterministic, race-free accumulation without requiring 320-bit atomic operations.

**Data Format** (CSR - Compressed Sparse Row):
- `col_ptr[i+1] - col_ptr[i]` = number of non-zeros in column i
- `row_idx[k]` = row index for k-th non-zero
- `values[k]` = scalar value [u32; 10] for MNT4/MNT6-298 fields (320 bits)

---

## PVUGC Integration (Sparse Quotient GPU Path)

### Activation Conditions

GPU path is used when:
- `gpu` feature is enabled
- `sparse_quotient_gpu_available()` is true
- The curve is the MNT4/MNT6 cycle (`is_mnt_cycle::<C>()` in `src/pvugc_outer.rs`)

### Key Types and FFI

- `SparseMatrixCsr` and `SparseQuotientPairOutput` live in `sppark-msm/src/lib.rs` and define CSR inputs and per-pair outputs.
- GPU entrypoint: `compute_sparse_quotient_coeffs_mnt4_298_gpu(...)`.
- CUDA kernels exist for both MNT4 and MNT6; the Rust wrapper uses the MNT4 variant because MNT4 Fr == MNT6 Fq in the cycle.

```rust
pub struct SparseMatrixCsr {
    pub col_ptr: Vec<u32>,
    pub row_idx: Vec<u32>,
    pub values: Vec<[u32; 10]>,
}

pub struct SparseQuotientPairOutput {
    pub acc_u: Vec<[u32; 10]>,
    pub acc_v: Vec<[u32; 10]>,
    pub diag_terms: Vec<(u32, [u32; 10])>,
}

pub fn compute_sparse_quotient_coeffs_mnt4_298_gpu(
    col_a: &SparseMatrixCsr,
    col_b: &SparseMatrixCsr,
    domain_elements: &[[u32; 10]],
    inv_domain_elements: &[[u32; 10]],
    inv_n_one_minus_omega: &[[u32; 10]],
    domain_size: u32,
    pairs: &[(u32, u32)],
    max_col_a: u32,
    max_col_b: u32,
    max_diag_per_pair: u32,
) -> Result<Vec<SparseQuotientPairOutput>, SparseQuotientGpuError>
```

### Integration Flow (compute_witness_bases)

1. Prepare GPU data once:
   - Convert sparse columns to CSR (`sparse_columns_to_csr`)
   - Convert domain tables to `[u32; 10]` arrays (`scalar_to_u32_array`)
2. Main loop selects GPU or CPU path; GPU batches pairs (default 10,000 per chunk, larger than CPU).
3. GPU kernel computes `acc_u`, `acc_v`, and `diag_terms`.
4. Build MSM tasks from GPU output and run MSM via `msm_with_gpu_fallback`.
5. CPU path uses the same MSM helper as fallback.

### Helper Functions (src/pvugc_outer.rs)

- `is_mnt_cycle::<C>()`
- `GpuQuotientData`
- `compute_quotient_bases_gpu`
- `compute_quotient_bases_cpu`
- `build_msm_task_from_gpu_output`

---

## Testing Strategy

### Phase 1: Low-Level CUDA Field Arithmetic Tests

Tests in `tests/test_cgbn_mnt4_field.cu` and `tests/test_cgbn_mnt6_field.cu`:

```
Test Categories:
1. Field Addition
   - a + b mod p (basic)
   - a + b where a + b >= p (reduction needed)
   - 0 + a = a (identity)
   - a + (p - a) = 0 (additive inverse)

2. Field Subtraction
   - a - b mod p (basic, a > b)
   - a - b mod p (underflow, a < b)
   - a - 0 = a (identity)
   - a - a = 0

3. Field Multiplication
   - a * b mod p (basic)
   - a * 1 = a (identity)
   - a * 0 = 0
   - (p-1) * (p-1) mod p (large values)

4. Field Negation
   - -a mod p = p - a
   - -0 = 0

5. Wide Multiplication & Reduction
   - Verify cgbn_mul_wide + cgbn_rem_wide produces correct results
   - Test with values that produce full 596-bit products
```

**Validation Method**: Compare GPU results against GMP reference calculations on CPU.

### Phase 2: CPU vs GPU MSM Consistency Tests

Tests in `tests/test_mnt4_cpu_gpu_consistency.rs` and `tests/test_mnt6_cpu_gpu_consistency.rs`:

```rust
Test Cases:
1. Basic Tests
   - generator_simple: 2*G + 3*G = 5*G
   - single_point: 42*G
   - empty: empty input returns identity
   - zero_scalars: 0*G + 0*G + 0*G = identity
   - identity_points: scalar * infinity = identity

2. Cancellation Tests (Critical)
   - cancellation: a*G + (-a)*G = 0
   - multiple_cancellations: pattern with multiple cancel pairs
   - sequential_partial_sums_to_zero: intermediate zeros during accumulation

3. Random MSM Tests (Various Sizes)
   - random_small_n4: n=4 random points/scalars
   - random_medium_n16: n=16
   - random_larger_n64: n=64
   - random_256: n=256
   - random_512: n=512
   - random_1024: n=1024

4. Edge Cases
   - mixed_edge_cases: mix of generator, identity, zero scalars
   - large_scalars: scalars near field modulus (p-1)
   - negative_scalars: scalars > p/2
   - identical_points: same point with different scalars
   - max_u64_scalars: u64::MAX as scalar

5. Behavior Tests
   - deterministic: same inputs produce same outputs
   - groth16_like_pattern: realistic MSM pattern from proving
```

### Phase 3: Sparse Quotient Coefficient Kernel Tests

Tests in `tests/test_sparse_quotient_gpu.rs`:

```rust
Test Cases:
1. Basic Functionality
   - Single pair with small sparse columns
   - Multiple pairs batch processing
   - Empty columns handling

2. Diagonal vs Off-Diagonal Terms
   - Pairs with only diagonal matches (k == m)
   - Pairs with only off-diagonal matches (k != m)
   - Mixed diagonal and off-diagonal

3. Field Arithmetic Correctness
   - Verify coefficient accumulation matches CPU reference
   - Test inv_n_one_minus_omega table usage
   - Test domain element lookups

4. Edge Cases
   - Large sparse columns
   - Single element columns
   - Maximum diagonal terms per pair

5. CPU vs GPU Consistency
   - Compare full coefficient computation against CPU implementation
   - Use actual sparse matrices from test circuits
```

### Phase 4: GPU vs CPU Integration Tests ✅

**Purpose:** Verify the full pipeline matches between GPU and CPU:
```
(i, j) pairs -> coefficient computation -> MSM task building -> H_ij bases
```

**Location:** `src/pvugc_outer.rs` module `sparse_quotient_integration_tests`

**Tests implemented:**

| Test | Description |
|------|-------------|
| `test_sparse_quotient_gpu_cpu_consistency` | Main integration test - compares GPU and CPU paths on synthetic sparse data (6×5 columns, 30 pairs) |
| `test_sparse_quotient_gpu_determinism` | Verifies GPU results are identical across 3 runs |
| `test_sparse_quotient_empty_columns` | Edge case handling for empty columns |
| `test_sparse_quotient_large_columns` | Stress test with 20×16=320 element combinations |

**Run command:**
```bash
cargo test --release --features gpu sparse_quotient_integration_tests
```

**Implementation details:**
- Uses MNT4-298 Fr (scalar field) and MNT6-298 G1 (curve points)
- Generates synthetic sparse columns with random field elements
- Compares full H_ij affine points (not just coefficients)
- Tests sort by (i,j) for deterministic comparison

---

## Test Execution Commands

### Run MNT4 Tests
```bash
# Basic kernel launch tests
cargo test --release --features gpu --test test_gpu_cgbn_mnt4

# Full consistency tests
cargo test --release --features gpu --test test_mnt4_cpu_gpu_consistency
```

### Run MNT6 Tests
```bash
cargo test --release --features gpu --test test_gpu_cgbn_mnt6
cargo test --release --features gpu --test test_mnt6_cpu_gpu_consistency
```

### Run CUDA Field Tests
```bash
cd sppark-msm/tests
nvcc -o test_cgbn_mnt4_field test_cgbn_mnt4_field.cu -I../cgbn-lib/include -lgmp -arch=sm_75
./test_cgbn_mnt4_field
```

### Run Sparse Quotient Tests
```bash
cargo test --release --features gpu --test test_sparse_quotient_gpu
```

### Run GPU/CPU Integration Tests
```bash
cargo test --release --features gpu sparse_quotient_integration_tests -- --nocapture
```

### Run Lean Prover Integration Tests
```bash
cargo test --features gpu test_lean_prover_end_to_end -- --ignored --nocapture
```

---

## Implementation Status

### Completed ✅

| Component | File | Tests | Status |
|-----------|------|-------|--------|
| MNT4-298 CUDA Field Tests | `test_cgbn_mnt4_field.cu` | 100/100 | ✅ DONE |
| MNT6-298 CUDA Field Tests | `test_cgbn_mnt6_field.cu` | 100/100 | ✅ DONE |
| MNT4-298 CPU/GPU Consistency | `test_mnt4_cpu_gpu_consistency.rs` | 21/21 | ✅ DONE |
| MNT6-298 CPU/GPU Consistency | `test_mnt6_cpu_gpu_consistency.rs` | 21/21 | ✅ DONE |
| Sparse Quotient GPU Tests | `test_sparse_quotient_gpu.rs` | 19/19 | ✅ DONE |
| MSM GPU Integration | `src/pvugc_outer.rs` | - | ✅ DONE |
| Sparse Quotient Kernel | `sparse_quotient_mnt4_298_cgbn.cu` | - | ✅ DONE |
| Sparse Quotient GPU Integration | `src/pvugc_outer.rs` | - | ✅ DONE |
| GPU/CPU Integration Tests | `src/pvugc_outer.rs` | 4/4 | ✅ DONE |

### Integration Status

- ✅ MSM GPU kernels integrated via `msm_with_gpu_fallback()` in `src/pvugc_outer.rs`
- ✅ Sparse quotient GPU kernel integrated into `compute_witness_bases()` with CPU fallback
- ✅ Full pipeline integration tests comparing GPU and CPU H_ij output

---

## Resolved Issues

### Issue 1: Sparse Quotient Kernel Hang Bug (Fixed 2025-12-24)

**Problem:** The sparse quotient CUDA kernel hung due to incorrect use of `__shfl_sync(0xffffffff, ...)`. This warp shuffle requires all 32 threads in a warp to participate, but when CGBN instances (TPI=8, so 4 instances per warp) diverge, some instances skip the loop body while others enter it. The divergent execution with full-warp shuffle caused a deadlock.

**Fix Applied:** Changed to instance-scoped shuffle with `width=TPI`:
```cuda
// Before (BROKEN):
diag_slot = __shfl_sync(0xffffffff, diag_slot, 0);

// After (FIXED):
constexpr uint32_t TPI = sparse_quotient_cgbn_params_t::TPI;
uint32_t lane_in_warp = threadIdx.x & 31;
uint32_t group_thread = threadIdx.x % TPI;
uint32_t instance_mask = ((1u << TPI) - 1u) << (lane_in_warp - group_thread);
diag_slot = __shfl_sync(instance_mask, diag_slot, 0, TPI);
```

### Issue 2: Accumulator Race Condition (Fixed 2025-12-24)

**Problem:** The off-diagonal accumulations (`out_acc_u`, `out_acc_v`) used unsynchronized load/add/store operations. With 32 CGBN instances per block distributing combinations across instances, multiple instances could write to the same output slot simultaneously, causing incorrect results.

**Fix Applied:** Two-phase algorithm where each instance owns exclusive output indices:
- **Phase 1:** Each instance owns `idx_u` values, loops over ALL `idx_v` → writes `acc_u`
- **Phase 2:** Each instance owns `idx_v` values, loops over ALL `idx_u` → writes `acc_v`

This doubles compute but ensures deterministic, race-free accumulation without requiring 320-bit atomic operations.

### Issue 3: CGBN Duplicate Symbol Linker Error (Fixed 2025-12-25)

**Problem:** When linking multiple CGBN kernels (MSM + sparse quotient), the linker reported duplicate symbol errors for `cgbn_error_report_alloc`, `cgbn_error_report_free`, etc. These functions are defined in `cgbn.cu` which is included by `cgbn.h`, causing each kernel object file to contain its own definition.

**Fix Applied:**
1. Added `-Xcompiler -fvisibility=hidden` to all CGBN kernel builds in `sppark-msm/build.rs`
2. Added `--allow-multiple-definition` linker flag in `.cargo/config.toml`:
   ```toml
   [target.x86_64-unknown-linux-gnu]
   rustflags = ["-C", "link-arg=-Wl,--allow-multiple-definition"]
   ```

---

## Technical Notes

### MNT4/MNT6 Cycle Pair

MNT4-298 and MNT6-298 form a cycle pair:
- MNT4-298 scalar field (Fr) == MNT6-298 base field (Fq)
- MNT6-298 scalar field (Fr) == MNT4-298 base field (Fq)

This means the field moduli in the CUDA kernels are swapped:
- `msm_mnt4_298_cgbn.cu` uses MNT4_298_R (scalar field) for scalar operations
- `msm_mnt6_298_cgbn.cu` uses MNT6_298_R (scalar field) for scalar operations

### Custom Test Harness

The existing MNT tests use `harness = false` to avoid Rust test framework CUDA cleanup conflicts. New tests should follow this pattern.

### ManuallyDrop Pattern

GPU test code uses `ManuallyDrop` to prevent Rust from dropping CUDA memory during test execution. This is a workaround for CUDA/Rust interaction issues.

### Open Questions / Tunables

1. **MNT6 sparse quotient wrapper:** CUDA kernel exists for MNT6, but Rust FFI wraps the MNT4 variant today. Confirm if a dedicated MNT6 wrapper is needed.
2. **Batch size:** GPU uses larger batch size than CPU; tune based on memory/throughput.
3. **Error handling:** Decide whether GPU failures fall back to CPU mid-run or fail fast.
4. **Lean prover dispatch:** Add MNT6 GPU routing in `msm_backend::msm_g1` to avoid CPU MSM in lean e2e.

---

## Performance Expectations

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Coeff accumulation | ~6 hours | ~10 min | 30-40x |
| MSM per pair | ~2 hours | ~5 min | 20-30x |
| **Total** | **~8 hours** | **~15 min** | **~30x** |

### Memory Requirements

- Sparse matrix data: ~500 MB
- Domain element tables: ~10 MB
- Output coefficients: ~2 GB (temporary)
- GPU memory needed: ~4 GB minimum

---

## Files Reference

### GPU Kernels
- `sppark-msm/src/msm_mnt4_298_cgbn.cu` - MNT4-298 MSM kernel
- `sppark-msm/src/msm_mnt6_298_cgbn.cu` - MNT6-298 MSM kernel
- `sppark-msm/src/sparse_quotient_mnt4_298_cgbn.cu` - Sparse quotient coefficient kernel
- `sppark-msm/src/sparse_quotient_mnt6_298_cgbn.cu` - Sparse quotient coefficient kernel

### Integration Points
- `src/pvugc_outer.rs` - Main integration, GPU dispatch
- `sppark-msm/src/lib.rs` - FFI bindings, MSM implementations
- `sppark-msm/build.rs` - Parallel CUDA compilation

### Tests
- `sppark-msm/tests/test_cgbn_mnt4_field.cu` - Low-level field arithmetic
- `sppark-msm/tests/test_cgbn_mnt6_field.cu` - Low-level field arithmetic
- `sppark-msm/tests/test_mnt4_cpu_gpu_consistency.rs` - MSM consistency
- `sppark-msm/tests/test_mnt6_cpu_gpu_consistency.rs` - MSM consistency
- `sppark-msm/tests/test_sparse_quotient_gpu.rs` - Sparse quotient kernel
