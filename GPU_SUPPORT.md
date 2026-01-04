# GPU Acceleration Support

## Overview

This codebase includes GPU acceleration for multi-scalar multiplication (MSM) using NVIDIA CUDA via the [sppark](https://github.com/supranational/sppark) library. GPU support is optional and controlled by the `gpu` feature flag.

For supported curves, status, and the roadmap, see `GPU_plan.md` and `docs/GPU_INDEX.md`. For MNT sparse-quotient integration details, see `docs/MNT_GPU_ACCELERATION.md`.

## Configuration

### GPU Architecture Requirements

**Default**: `sm_75` (Turing architecture - RTX 2060, GTX 1660, etc.)

The build will fall back to CPU if:
- CUDA is not available
- GPU architecture is incompatible
- CUDA build fails for any reason

To use a different architecture, set `CUDA_ARCH`:

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

### GPU Memory Configuration

The sparse quotient kernel uses dynamic batch sizing based on detected GPU memory:

```bash
# Default: 80% of available GPU memory
cargo test --features gpu --release

# Custom percentage (10-95%)
GPU_MEMORY_PERCENT=90 cargo test --features gpu --release
```

**Environment Variables:**
- `GPU_MEMORY_PERCENT`: Target GPU memory utilization (default: 80%, range: 10-95%)

**API Functions** (in `sppark-msm`):
- `get_gpu_available_memory()` - Returns free GPU memory in bytes
- `get_gpu_total_memory()` - Returns total GPU memory in bytes
- `get_gpu_memory_percent()` - Returns configured percentage (from env var)
- `get_gpu_target_memory()` - Returns calculated target memory for batch operations

**Fallback behavior**: If GPU memory detection fails, falls back to 500MB target.

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

- Small inputs, large inputs, edge cases
- Deterministic behavior and error handling

**Run with**: `cargo test --features gpu -p sppark-msm`

### GPU Integration Tests

Located in `tests/test_gpu_integration_bls12_377.rs`:

- Full `msm_backend` dispatch path
- Layout safety assertions
- Various workload sizes

**Run with**: `cargo test --features gpu --test test_gpu_integration_bls12_377`

### CGBN Kernel Tests (BW6/MNT)

- **BW6-761 G1**: `cargo test --features gpu --test test_gpu_cgbn_bw6`
- **MNT4-298 G1**: `cargo test --features gpu --test test_gpu_cgbn_mnt4`
- **MNT6-298 G1**: `cargo test --features gpu --test test_gpu_cgbn_mnt6`

**CPU/GPU consistency**:
- `cargo test --release --features gpu --test test_mnt4_cpu_gpu_consistency -- --nocapture`
- `cargo test --release --features gpu --test test_mnt6_cpu_gpu_consistency -- --nocapture`

Layout validation tests ensure correct FFI data layout:
- `cargo test --features gpu --test layout_validation_bw6_761`
- `cargo test --features gpu --test layout_validation_mnt4_298`
- `cargo test --features gpu --test layout_validation_mnt6_298`

### GPU/CPU Integration Tests (Sparse Quotient)

Located in `src/pvugc_outer.rs` module `sparse_quotient_integration_tests`:

- `test_sparse_quotient_gpu_cpu_consistency` - Full H_ij bases match (30 pairs)
- `test_sparse_quotient_gpu_determinism` - Results identical across runs
- `test_sparse_quotient_empty_columns` - Edge case handling
- `test_sparse_quotient_large_columns` - Stress test (320 combinations)

**Run with**: `cargo test --release --features gpu --lib sparse_quotient_integration_tests`

## Troubleshooting

1. If GPU tests are skipped, check `cargo:warning=CUDA build skipped` messages.
2. Verify `CUDA_HOME` and `CUDA_ARCH` are set correctly.
3. Check that your curve type is supported and the GPU path is selected.

### Layout Assertion Failures

If you see "size mismatch" or "alignment mismatch" panics:
1. Arkworks version may have changed struct layout.
2. Rebuild from clean: `cargo clean && cargo build --features gpu`.
3. Report as a bug - this indicates an incompatibility.

## Safety Notes

**Layout Safety Checks**: The code includes runtime assertions to verify that arkworks struct layouts match sppark's expectations. If arkworks changes `G1Affine` or `BigInt` layout in a future version, these assertions will catch the mismatch early.

**Type Safety**: GPU dispatch uses `TypeId` checking to ensure only supported curve types are sent to GPU kernels.

## References

- **sppark Library**: https://github.com/supranational/sppark
- **arkworks**: https://github.com/arkworks-rs
- **CUDA Compute Capabilities**: https://developer.nvidia.com/cuda-gpus

---
*Last Updated: 2026-01-03*
