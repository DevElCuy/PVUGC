# GPU Acceleration Support

## Overview

This codebase includes GPU acceleration for Multi-Scalar Multiplication (MSM) operations using NVIDIA CUDA via the [sppark](https://github.com/supranational/sppark) library. GPU support is **optional** and controlled by the `gpu` feature flag.

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

### ❌ Not Supported

- **BLS12-377 G2**: No GPU implementation
- **BW6-761 G2**: No GPU implementation

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

### What Tests DON'T Cover

❌ **BW6-761 GPU acceleration** - Not implemented yet
❌ **test_pvugc_on_outer_proof_e2e with GPU** - Outer proof uses BW6-761 (CPU only)
❌ **BLS12-377 G2 operations** - No GPU implementation

## Architecture Details

### Code Flow

```
User Code (Groth16 prover)
    ↓
msm_backend::msm_g1<G>()  [src/msm_backend.rs]
    ↓
TypeId check: Is G == BLS12-377 G1Affine?
    ↓ Yes (with gpu feature)
Layout safety assertions
    ↓
GpuMsm::msm_gpu()  [sppark-msm/src/lib.rs]
    ↓
CUDA kernel  [sppark-msm/src/msm_bls12_377.cu]
    ↓
sppark mont_t field arithmetic  [sppark-msm/sppark/ff/bls12-377.hpp]
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
- BW6-761 operations: Falls back to CPU
- Systems without compatible CUDA GPU

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

### BW6-761 Optimizations

The current CGBN kernel uses serial double-and-add (O(n × log(scalar_bits))). Future improvements:

1. **Pippenger Algorithm**: O(n / log(n)) for large MSMs
2. **Parallel Scalar Multiplications**: Multiple thread groups + tree reduction
3. **Remove ENABLE_CGBN_STUB Gate**: Once correctness validated in production

See `sppark-msm/docs/BW6_761_CGBN.md` for implementation details.

## References

- **sppark Library**: https://github.com/supranational/sppark
- **arkworks**: https://github.com/arkworks-rs
- **CUDA Compute Capabilities**: https://developer.nvidia.com/cuda-gpus

---
*Last Updated: 2025-12-23*
