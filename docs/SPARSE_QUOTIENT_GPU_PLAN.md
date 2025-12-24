# GPU Acceleration Plan for Sparse Quotient Computation

## Problem Statement

The lean prover PVUGC setup computes sparse quotient bases H_{ij} in `src/pvugc_outer.rs` (lines 603-754). This is currently the bottleneck:

- **680 million pairs** (i,j) to process
- Each pair involves sparse matrix lookups and small MSMs
- Current rate: ~0 pair/s initially (CPU-bound)
- Estimated time: Several hours on CPU

The computation has two phases:
1. **Diagonal Q-vector** (lines 500-575): Already efficient (~25s) using FFT convolution
2. **Main pair loop** (lines 603-754): The bottleneck - O(n²) pairs with irregular work per pair

## Current Algorithm Structure

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
4. Run small MSM (currently uses GPU via msm_with_gpu_fallback)
5. Store non-zero result as (i, j, H_ij)
```

## Why It's Slow

1. **680M loop iterations** with Python-like irregular access patterns
2. **Small MSMs per pair** - GPU launch overhead dominates for small MSMs
3. **Sparse matrix lookups** - cache-unfriendly memory access
4. **Field arithmetic in inner loop** - MNT4/MNT6 298-bit field ops on CPU

## GPU Acceleration Options

### Option 1: Batched MSM Only (Minimal Change)

**Approach**: Keep CPU coefficient computation, batch MSM tasks into larger chunks.

**Changes**:
- Collect 10K-100K MSM tasks before dispatching to GPU
- Concatenate bases/scalars into single large MSM
- Use prefix sums to track which results belong to which pair

**Pros**:
- Minimal code changes
- Leverages existing GPU MSM infrastructure

**Cons**:
- CPU coefficient computation still bottleneck
- Memory pressure from batching (need to store intermediate bases/scalars)
- Estimated speedup: 2-3x (MSM is ~30% of time)

**Estimated effort**: Low

---

### Option 2: GPU Field Arithmetic Kernel (CGBN)

**Approach**: Move the inner coefficient accumulation loop to GPU using CGBN.

**Changes**:
- New CUDA kernel: `sparse_quotient_coeffs_mnt4_298.cu`
- Input: Sparse columns col_a, col_b, domain elements, inv_n_one_minus_omega
- Output: Accumulated coefficients (acc_u, acc_v, diag_terms) per pair
- Use CGBN for 320-bit MNT4/MNT6 field arithmetic

**Kernel structure**:
```cuda
__global__ void compute_quotient_coeffs(
    // Sparse matrix data (CSR format)
    const uint32_t* col_a_ptr,      // column pointers
    const uint32_t* col_a_idx,      // row indices
    const cgbn_mem_t* col_a_val,    // field values
    const uint32_t* col_b_ptr,
    const uint32_t* col_b_idx,
    const cgbn_mem_t* col_b_val,
    // Precomputed tables
    const cgbn_mem_t* domain_elements,
    const cgbn_mem_t* inv_domain_elements,
    const cgbn_mem_t* inv_n_one_minus_omega,
    // Pair assignments
    const uint32_t* pairs_i,
    const uint32_t* pairs_j,
    uint32_t num_pairs,
    // Output: accumulated coefficients
    cgbn_mem_t* out_coeffs,
    uint32_t* out_indices
) {
    // Each thread block handles one (i,j) pair
    // Threads cooperate on the rows_u × rows_v nested loop
}
```

**Pros**:
- Attacks the actual bottleneck (field arithmetic)
- Good GPU utilization for regular work distribution
- Estimated speedup: 10-50x

**Cons**:
- Significant CUDA development
- Need to handle irregular work (varying rows_u/rows_v sizes)
- Memory layout conversion (Rust sparse → GPU CSR)

**Estimated effort**: Medium-High

---

### Option 3: Reformulate as Sparse Matrix-Matrix Multiply (SpMM)

**Approach**: Express the computation as sparse matrix operations using cuSPARSE.

**Observation**: The coefficient computation is essentially:
```
C = A^T @ B  (sparse matrix multiply)
```
where A and B are the constraint matrices in sparse format.

**Changes**:
- Convert col_a, col_b to CSR/CSC format
- Use cuSPARSE `cusparseSpGEMM` for the multiply
- Post-process to apply the ω^k, ω^m scaling

**Pros**:
- Leverages highly optimized cuSPARSE library
- Handles sparsity patterns efficiently
- Potentially fastest option

**Cons**:
- Field arithmetic must be custom (cuSPARSE is for floats)
- Would need CGBN-based custom SpMM kernel
- Complex to implement correctly

**Estimated effort**: High

---

### Option 4: Hybrid CPU-GPU Pipeline (Recommended)

**Approach**: Pipeline the work - CPU prepares batches while GPU processes previous batch.

**Architecture**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CPU Core  │────▶│  GPU Queue  │────▶│    GPU      │
│  Coeff Acc  │     │   Batch     │     │    MSM      │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                    │
      ▼                   ▼                    ▼
   Prepare            Accumulate           Execute
   Batch N+1          Batch N              Batch N-1
```

**Changes**:
1. Split pairs into large batches (1M pairs each)
2. CPU threads compute coefficients for batch N+1
3. Meanwhile, GPU executes MSMs for batch N
4. Triple-buffer to hide latency

**Pros**:
- Overlaps CPU and GPU work
- No new CUDA kernels needed
- Incremental improvement path

**Cons**:
- Still CPU-bound overall
- Complex synchronization
- Estimated speedup: 3-5x

**Estimated effort**: Medium

---

## Recommended Approach: Option 2 (GPU Field Arithmetic)

### Rationale

1. The inner loop (field arithmetic) is 70%+ of the time
2. CGBN infrastructure already exists for MNT4/MNT6 (320-bit)
3. Similar pattern to existing MSM kernels
4. Can be combined with batched MSM for additional gains

### Implementation Plan

#### Phase 1: Data Structure Preparation
- [ ] Convert sparse columns to GPU-friendly CSR format
- [ ] Pre-upload domain_elements, inv_domain_elements, inv_n_one_minus_omega to GPU
- [ ] Create pair assignment array (which thread handles which pair)

#### Phase 2: CGBN Coefficient Kernel
- [ ] Create `sparse_quotient_coeffs_mnt4_298.cu`
- [ ] Implement CGBN field multiply, add, sub for MNT4-298 scalar field
- [ ] Handle diagonal vs off-diagonal branching
- [ ] Output: per-pair coefficient arrays

#### Phase 3: Batched MSM Integration
- [ ] Collect coefficient outputs into batched MSM format
- [ ] Use existing GPU MSM for the batched computation
- [ ] Reconstruct per-pair results using prefix sums

#### Phase 4: Rust FFI Integration
- [ ] Add FFI functions in `sppark-msm/src/lib.rs`
- [ ] Modify `pvugc_outer.rs` to use GPU path when available
- [ ] CPU fallback for non-GPU builds

### Required Components

1. **New CUDA kernel**: `sppark-msm/src/sparse_quotient_mnt4_298_cgbn.cu`
   - CGBN 320-bit field arithmetic
   - CSR sparse matrix handling
   - Per-pair coefficient accumulation

2. **Rust FFI bindings**: `sppark-msm/src/lib.rs`
   - `sparse_quotient_coeffs_mnt4_298_gpu()`
   - Data marshalling for sparse matrices

3. **Modified prover**: `src/pvugc_outer.rs`
   - GPU dispatch logic
   - Format conversion (sparse columns → CSR)

### Performance Expectations

| Component | Current (CPU) | After GPU | Speedup |
|-----------|---------------|-----------|---------|
| Coeff accumulation | ~6 hours | ~10 min | 30-40x |
| MSM per pair | ~2 hours | ~5 min | 20-30x |
| **Total** | **~8 hours** | **~15 min** | **30x** |

### Memory Requirements

- Sparse matrix data: ~500 MB
- Domain element tables: ~10 MB
- Output coefficients: ~2 GB (temporary)
- GPU memory needed: ~4 GB minimum

## Alternative: Precomputation Caching

If GPU acceleration is not feasible, consider:

1. **Cache the H_{ij} bases to disk** after first computation
2. Load from cache on subsequent runs
3. Cache key: hash of (circuit, curve parameters)

This doesn't speed up the first run but makes subsequent runs instant.

## Files to Modify

1. `sppark-msm/src/sparse_quotient_mnt4_298_cgbn.cu` (new)
2. `sppark-msm/src/sparse_quotient_mnt6_298_cgbn.cu` (new, similar)
3. `sppark-msm/src/lib.rs` - FFI bindings
4. `sppark-msm/build.rs` - compile new CUDA files
5. `src/pvugc_outer.rs` - GPU dispatch in `compute_witness_bases_outer_for()`

## References

- Current implementation: `src/pvugc_outer.rs:603-754`
- CGBN MNT4 MSM kernel: `sppark-msm/src/msm_mnt4_298_cgbn.cu`
- CGBN library: `cgbn-lib/include/cgbn/`
