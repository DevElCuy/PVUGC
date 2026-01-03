# GPU Acceleration Support

## Overview

This codebase includes GPU acceleration for Multi-Scalar Multiplication (MSM) operations using NVIDIA CUDA via the [sppark](https://github.com/supranational/sppark) library. GPU support is **optional** and controlled by the `gpu` feature flag.

For the documentation tree and prioritized roadmap, see `docs/GPU_INDEX.md` and `GPU_plan.md`.

## Current Status

### ✅ Fully Supported: BLS12-377 G1

**Curve**: BLS12-377 G1 (base field Fp, scalar field Fr)
**Status**: Production ready with comprehensive test coverage
**Use Cases**: Inner proofs in the Bls12Bw6Cycle

### ✅ Supported (Experimental): BW6-761 G1

**Curve**: BW6-761 G1 (761-bit base field)
**Status**: Working via CGBN kernel, gated behind `ENABLE_CGBN_STUB=1`
**Implementation**: Uses NVIDIA CGBN library for cooperative-group field arithmetic
**Use Cases**: Outer proofs in the Bls12Bw6Cycle

See `sppark-msm/docs/BW6_761_CGBN.md` for implementation details.

**Test Command**:
```bash
ENABLE_CGBN_STUB=1 cargo test --release --features gpu --test test_gpu_cgbn_bw6
```

### ✅ Supported (Experimental): MNT4-298 G1

**Curve**: MNT4-298 G1 (298-bit base field, curve parameter a=2)
**Status**: Working via CGBN kernel (experimental)
**Implementation**: Uses NVIDIA CGBN library with 320-bit field arithmetic (TPI=8)
**Use Cases**: Inner proofs in the Mnt4Mnt6Cycle

**Test Command**:
```bash
ENABLE_CGBN_STUB=1 cargo test --release --features gpu --test test_gpu_cgbn_mnt4
ENABLE_CGBN_STUB=1 cargo test --release --features gpu --test layout_validation_mnt4_298
```

### ✅ Supported (Experimental): MNT6-298 G1

**Curve**: MNT6-298 G1 (298-bit base field, curve parameter a=11)
**Status**: Working via CGBN kernel (experimental)
**Implementation**: Uses NVIDIA CGBN library with 320-bit field arithmetic (TPI=8)
**Use Cases**: Outer proofs in the Mnt4Mnt6Cycle
**Note**: Sparse quotient GPU path is integrated for the MNT cycle (see `docs/MNT_GPU_ACCELERATION.md`); BW6 sparse quotient is still pending (see `GPU_plan.md`).

**Test Command**:
```bash
ENABLE_CGBN_STUB=1 cargo test --release --features gpu --test test_gpu_cgbn_mnt6
ENABLE_CGBN_STUB=1 cargo test --release --features gpu --test layout_validation_mnt6_298
```

### ❌ Not Supported

- **BLS12-377 G2**: No GPU implementation
- **BW6-761 G2**: No GPU implementation
- **MNT4-298 G2**: No GPU implementation
- **MNT6-298 G2**: No GPU implementation

## Important Limitations

### 1. BW6-761 GPU Requires Environment Variable

The BW6-761 CGBN kernel is gated behind `ENABLE_CGBN_STUB=1` until correctness is fully validated in production. Without this variable, BW6-761 MSMs fall back to CPU.

```bash
# To enable BW6-761 GPU acceleration
ENABLE_CGBN_STUB=1 cargo test --release --features gpu test_pvugc_on_outer_proof_e2e
```

### 2. GPU Architecture Requirements

**Default**: `sm_75` (Turing architecture - RTX 2060, GTX 1660, etc.)

The build will silently fall back to CPU if:
- CUDA is not available
- GPU architecture is incompatible
- CUDA build fails for any reason

**To use a different architecture**, set the `CUDA_ARCH` environment variable:

```bash
# For Ampere GPUs (RTX 3000 series)
export CUDA_ARCH=sm_80

# For Ada Lovelace (RTX 4000 series)
export CUDA_ARCH=sm_89

# For Hopper (H100)
export CUDA_ARCH=sm_90

# For older Pascal GPUs (GTX 1080, P100)
export CUDA_ARCH=sm_60
```

### 3. Safety Considerations

**Layout Safety Checks**: The code includes runtime assertions to verify that arkworks struct layouts match sppark's expectations. If arkworks changes `G1Affine` or `BigInt` layout in a future version, these assertions will catch the mismatch early.

**Type Safety**: GPU dispatch uses `TypeId` checking to ensure only supported curve types are sent to GPU kernels.

## Building with GPU Support

### Prerequisites

1. **CUDA Toolkit**: Version 11.0 or later
2. **Compatible GPU**: NVIDIA GPU matching your `CUDA_ARCH` setting
3. **C++ Compiler**: Supporting C++17

### Build Commands

```bash
# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.6  # Adjust to your CUDA version
export PATH="$CUDA_HOME/bin:$PATH"

# Optional: Set GPU architecture
export CUDA_ARCH=sm_75  # Default, can be omitted

# Build with GPU support
cargo build --features gpu --release

# Run GPU-specific tests
cargo test --features gpu --test test_gpu_integration_bls12_377
cargo test --features gpu -p sppark-msm --test gpu_msm_bls12_377
```

### Verifying GPU Usage

The build system will print warnings if GPU compilation fails:

```
cargo:warning=Build disabled: <error details>
```

If you see this warning, the code will compile with CPU-only fallbacks.

## Test Coverage

### GPU Unit Tests (sppark-msm)

Located in `sppark-msm/tests/gpu_msm_bls12_377.rs`:

- ✅ Small inputs (16 points)
- ✅ Medium inputs (256 points)
- ✅ Large inputs (1024 points)
- ✅ Zero scalars → identity
- ✅ Identity points → identity
- ✅ Single point MSM
- ✅ Mismatched sizes (error handling)
- ✅ Empty inputs → identity
- ✅ Known values (5 * G)
- ✅ Identical points
- ✅ Varying sizes [8, 16, 32, 64, 128, 256, 512]
- ✅ Maximum scalar values
- ✅ Deterministic behavior

**Run with**: `cargo test --features gpu -p sppark-msm`

### GPU Integration Tests

Located in `tests/test_gpu_integration_bls12_377.rs`:

- ✅ Full msm_backend dispatch path
- ✅ Layout safety assertions
- ✅ Various workload sizes
- ✅ Edge case handling
- ✅ Deterministic execution

**Run with**: `cargo test --features gpu --test test_gpu_integration_bls12_377`

### CGBN Kernel Tests (Experimental Curves)

**BW6-761 G1**: `cargo test --features gpu --test test_gpu_cgbn_bw6`
**MNT4-298 G1**: `cargo test --features gpu --test test_gpu_cgbn_mnt4`
**MNT6-298 G1**: `cargo test --features gpu --test test_gpu_cgbn_mnt6`

Layout validation tests ensure correct FFI data layout:
- `cargo test --features gpu --test layout_validation_bw6_761`
- `cargo test --features gpu --test layout_validation_mnt4_298`
- `cargo test --features gpu --test layout_validation_mnt6_298`

### GPU/CPU Integration Tests (Sparse Quotient)

Located in `src/pvugc_outer.rs` module `sparse_quotient_integration_tests`:

- ✅ `test_sparse_quotient_gpu_cpu_consistency` - Full H_ij bases match (30 pairs)
- ✅ `test_sparse_quotient_gpu_determinism` - Results identical across 3 runs
- ✅ `test_sparse_quotient_empty_columns` - Edge case handling
- ✅ `test_sparse_quotient_large_columns` - Stress test (320 combinations)

**Run with**: `cargo test --release --features gpu --lib sparse_quotient_integration_tests`

### What Tests DON'T Cover

❌ **G2 operations** - No GPU implementation for any curve
❌ **Pippenger algorithm** - Current CGBN kernels use serial double-and-add

## Architecture Details

### Code Flow

```
User Code (Groth16 prover)
    ↓
msm_backend::msm_g1<G>()  [src/msm_backend.rs]
    ↓
TypeId check: G type?
    ↓
    ├─ BLS12-377 G1Affine
    │    ↓
    │  Layout safety assertions
    │    ↓
    │  GpuMsm::msm_gpu()  [sppark-msm/src/lib.rs]
    │    ↓
    │  CUDA kernel  [sppark-msm/src/msm_bls12_377.cu]
    │    ↓
    │  sppark mont_t field arithmetic
    │
    ├─ MNT4-298 G1Affine
    │    ↓
    │  GpuMsm::msm_gpu() → msm_mnt4_298_gpu_cgbn()
    │    ↓
    │  CGBN kernel [sppark-msm/src/msm_mnt4_298_cgbn.cu]
    │    ↓
    │  CGBN 320-bit field arithmetic (TPI=8, a=2)
    │
    └─ MNT6-298 G1Affine (via pvugc_outer.rs)
         ↓
       msm_mnt6_298_gpu_cgbn()
         ↓
       CGBN kernel [sppark-msm/src/msm_mnt6_298_cgbn.cu]
         ↓
       CGBN 320-bit field arithmetic (TPI=8, a=11)
```

### Safety Mechanisms

1. **Compile-time**: TypeId checking ensures only registered types dispatch to GPU
2. **Runtime**: Layout assertions verify struct compatibility
3. **Fallback**: Any GPU error falls back to CPU automatically
4. **Size verification**: FFI passes struct size for additional validation

## Performance Expectations

### When GPU Helps

- Large MSMs (1024+ points): Significant speedup
- Batch operations: Multiple MSMs can overlap
- BLS12-377 inner proofs: Direct acceleration

### When GPU Doesn't Help

- Small MSMs (<64 points): CPU may be faster due to overhead
- Systems without compatible CUDA GPU
- G2 operations on any curve: No GPU implementation

## Troubleshooting

### "GPU MSM failed" Errors

Check:
1. CUDA runtime is available: `nvidia-smi`
2. Architecture matches your GPU: `CUDA_ARCH` setting
3. Build succeeded: Look for build warnings

### Silent CPU Fallback

If GPU is not being used but no errors appear:
1. Check build output for CUDA compilation warnings
2. Verify `cfg(sppark_cuda_built)` is set: `cargo build --features gpu -vv`
3. Check that your curve type is BLS12-377 G1Affine

### Layout Assertion Failures

If you see "size mismatch" or "alignment mismatch" panics:
1. Arkworks version may have changed struct layout
2. Rebuild from clean: `cargo clean && cargo build --features gpu`
3. Report as a bug - this indicates an incompatibility

## Future Work

### Priority for SP1 e2e + Lean e2e (ordered by expected gain)

1. **BW6 sparse-quotient GPU path** (SP1 e2e bottleneck)  
   Est gain: ~10–30x on quotient phase; ~3–10x overall SP1 e2e.
2. **Pippenger MSM for CGBN kernels (BW6/MNT)**  
   Est gain: ~5–15x vs CPU for large MSMs; ~2–6x vs current GPU.
3. **G1 GPU MSM dispatch in lean prover (BW6 + MNT6)**  
   Est gain: ~2–8x on MSM-heavy steps; ~1.5–4x overall.
4. **G2 GPU MSM (BW6/MNT6)**  
   Est gain: ~1.5–4x on G2 MSM; ~1.1–2x overall.
5. **GPU pairings (verification path)**  
   Est gain: ~1.1–1.3x verification time.

### CGBN Kernel Optimizations (Details)

The CGBN kernels now support both serial double-and-add and Pippenger algorithms:

1. **Pippenger Algorithm**: ✅ Implemented for BW6-761, MNT4-298, MNT6-298
   - O(n + nwins × 2^wbits) complexity
   - Automatic window size selection
   - Falls back to serial for n < 64
2. **Parallel Scalar Multiplications**: Multiple thread groups + tree reduction (future)
3. **Remove ENABLE_CGBN_STUB Gate**: BW6-761 bug is now fixed (2026-01-03); gate can be removed
4. **TPI Tuning**: Optimize thread-per-instance count for MNT curves (currently TPI=8)

**Key Differences in MNT Curves**:
- MNT4-298/MNT6-298 use 320-bit fields (10 x u32 limbs) vs BW6-761's 768-bit fields (24 x u32 limbs)
- MNT4 has curve parameter a=2, MNT6 has a=11 (vs BW6's a=0)
- Point doubling formulas include a*ZZ^2 term for MNT curves

See `sppark-msm/docs/BW6_761_CGBN.md` for CGBN implementation details.
See `sppark-msm/docs/MNT_CGBN.md` and `docs/MNT_GPU_ACCELERATION.md` for MNT implementation details.

## References

- **sppark Library**: https://github.com/supranational/sppark
- **arkworks**: https://github.com/arkworks-rs
- **CUDA Compute Capabilities**: https://developer.nvidia.com/cuda-gpus

## Summary of Supported Curves

| Curve | G1 GPU | Implementation | Status | Test Coverage |
|-------|--------|---------------|--------|---------------|
| BLS12-377 | ✅ | sppark Pippenger | Production | Comprehensive |
| BW6-761 | ✅ | CGBN (serial + Pippenger) | Production | 21/21 tests passing |
| MNT4-298 | ✅ | CGBN (serial + Pippenger) | Experimental | Basic |
| MNT6-298 | ✅ | CGBN (serial + Pippenger) | Experimental | Basic |
| *-* G2 | ❌ | N/A | Not implemented | N/A |

---
*Last Updated: 2026-01-03*
