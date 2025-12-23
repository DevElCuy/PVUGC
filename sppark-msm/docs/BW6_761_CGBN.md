# BW6-761 CGBN GPU MSM Implementation

**Last Updated**: 2025-12-23

## Overview

This document describes the CGBN-based GPU MSM implementation for BW6-761, why it was needed, and how it works.

---

## Why CGBN? (The Problem with sppark)

BW6-761 has a **761-bit base field** (vs 377-bit for BLS12-377). When using sppark's standard Pippenger implementation, the kernel crashes due to excessive stack usage:

| Metric | BLS12-377 | BW6-761 | Impact |
|--------|-----------|---------|--------|
| Field bits | 377 | 761 | 2x larger |
| Limbs (32-bit) | 12 | 24 | 2x more operations |
| Element size | 48 bytes | 96 bytes | 2x memory |
| Stack per point add | 17.8 KB | 21-25 KB | Exceeds GPU limits |

**Root Cause**: sppark's `field_large_t<761>` template generates massive stack frames (~22KB per thread) for point operations. With CUDA's default stack limits, the kernel hangs or crashes.

**Solution**: Use NVIDIA's CGBN library which distributes field element limbs across multiple cooperating threads (TPI=8), avoiding per-thread stack overflow.

---

## Architecture

### CGBN Parameters

```cpp
class bw6_cgbn_params_t {
  static const uint32_t TPI = 8;      // 8 threads cooperate per big number
  static const uint32_t BITS = 768;   // Round up from 761 bits
};
```

### Data Flow

```
Rust (arkworks)                    CUDA (CGBN)
───────────────                    ───────────
G1Affine (Montgomery form)    →    Convert to plain form (CPU)
  x: Fp (u64[12])                    x: u32[24]
  y: Fp (u64[12])                    y: u32[24]
  infinity: bool                     infinity: bool

BigInt<6> (scalar)            →    scalar_cgbn_t
  limbs: u64[6]                      limbs: u32[12]
```

### Montgomery Conversion (Critical)

Arkworks stores field elements in Montgomery form (`a * R mod p`), but CGBN expects plain form. The Rust FFI layer converts using `into_bigint()`:

```rust
// src/lib.rs - Montgomery → plain conversion
let x_bigint = point.x.into_bigint();  // Performs Montgomery reduction
let x_u32: [u32; 24] = bigint_to_u32_array(x_bigint);
```

This conversion happens on the CPU before GPU transfer.

---

## Implementation Details

### Files

| File | Purpose |
|------|---------|
| `src/msm_bw6_761_cgbn.cu` | CGBN kernel implementation |
| `src/lib.rs` | Rust FFI with Montgomery conversion |
| `tests/test_gpu_cgbn_bw6.rs` | Integration tests (custom harness) |

### Kernel Algorithm

**Current**: Serial double-and-add MSM (MVP)

```
For each (point_i, scalar_i):
  1. term = scalar_mul(point_i, scalar_i)  // Double-and-add
  2. accumulator = point_add(accumulator, term)
Return accumulator
```

**Complexity**: O(n × log(scalar_bits)) - suitable for small MSMs

### CGBN Class Structure

```cpp
template<class params>
class bw6_msm_t {
  context_t _context;
  env_t     _env;
  int32_t   _instance;

  // Field operations (all TPI threads cooperate)
  void field_add(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P);
  void field_sub(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P);
  void field_mul(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P);

  // Point operations (XYZZ coordinates)
  void point_add_mixed(...)   // XYZZ + Affine → XYZZ
  void point_double(...)      // 2P
  void point_add(...)         // XYZZ + XYZZ → XYZZ
  void scalar_mul(...)        // k * P (double-and-add)
  void xyzz_to_jacobian(...)  // Convert to output format
};
```

---

## Usage

### Build

```bash
cargo build --release --features gpu
```

### Run Tests

```bash
cargo test --release --features gpu --test test_gpu_cgbn_bw6 -- --nocapture
```

### Integration

```rust
use sppark_msm::{GpuMsm, msm_bw6_761_gpu_cgbn};
use ark_bw6_761::G1Affine;

// Via trait (recommended)
let result = G1Affine::msm_gpu(&points, &scalars)?;

// Direct function call
let result = msm_bw6_761_gpu_cgbn(&points, &scalars)?;
```

---

## Custom Test Harness

GPU tests use `harness = false` in Cargo.toml to avoid SIGSEGV crashes caused by Rust test framework / CUDA cleanup order conflicts.

### The Problem

1. Rust test framework runs tests on worker threads
2. CUDA context is thread-local
3. When worker thread exits, CUDA destroys context automatically
4. Rust then tries to Drop variables that reference destroyed CUDA state
5. Result: SIGSEGV

### The Solution

```toml
# Cargo.toml
[[test]]
name = "test_gpu_cgbn_bw6"
path = "tests/test_gpu_cgbn_bw6.rs"
harness = false  # Custom main(), no worker threads
```

```rust
// tests/test_gpu_cgbn_bw6.rs
fn main() {
    let tests = vec![
        ("test_kernel_launches", test_kernel_launches),
        // ...
    ];
    for (name, test_fn) in tests {
        if !test_fn() {
            std::process::exit(1);
        }
    }
    std::process::exit(0);
}
```

**Pattern**: Use this for ALL GPU tests in Rust.

---

## Current Status

| Component | Status |
|-----------|--------|
| Field arithmetic (add/sub/mul) | ✅ Working |
| Point doubling | ✅ Working |
| Mixed addition (XYZZ + Affine) | ✅ Working |
| Full addition (XYZZ + XYZZ) | ✅ Working |
| Scalar multiplication | ✅ Working (double-and-add) |
| MSM accumulation | ✅ Working (serial) |
| Rust FFI | ✅ Working |
| Montgomery conversion | ✅ Working |
| Tests | ✅ 5/5 passing |

---

## Future Optimizations

### 1. Pippenger Algorithm (High Impact)

Current serial double-and-add is O(n × log(scalar_bits)).
Pippenger bucket method would be O(n / log(n)) for large MSMs.

```
1. Partition scalars into windows (e.g., 16-bit)
2. Accumulate points into buckets per window (parallel)
3. Combine buckets with weighted sum
```

### 2. Parallel Scalar Multiplications

Current: Single thread group processes all points serially.
Future: Multiple thread groups in parallel, then tree reduction.

### 3. Production Hardening

Add comprehensive error handling and diagnostics for production deployments.

---

## Environment

- **GPU**: NVIDIA GeForce GTX 1660 SUPER (sm_75, Turing)
- **CUDA**: 12.6
- **CGBN**: TPI=8, BITS=768

---

## Historical: Deprecated sppark Specialized Kernel

> **Note**: This section documents a failed approach that has been removed from the codebase.
> It is preserved here for historical reference and to explain why CGBN was necessary.

### What Was Attempted

Before CGBN, we tried to use sppark's standard Pippenger MSM implementation with aggressive
register management. The specialized kernel (`msm_bw6_761_specialized.cu`) attempted to:

1. **Reduce window size**: `MSM_WBITS=6` (64 buckets instead of default)
2. **Limit threads**: `MSM_NTHREADS=64` (conservative thread count)
3. **Cap registers**: `--maxrregcount=96` (force register spilling)
4. **Increase stack**: `cudaDeviceSetLimit(cudaLimitStackSize, 128KB)`

### Why It Failed

The fundamental problem is **stack frame size**, not register count:

```
sppark point_add() for BW6-761:
├── 6 temporary field elements (XYZZ coordinates)
├── Each field element: 96 bytes (761-bit → 24 × 32-bit limbs)
├── Total temporaries: ~576 bytes minimum
├── Plus call stack, loop variables, etc.
└── Result: 21-25 KB per thread stack frame
```

CUDA GPUs have limited per-thread stack space (~1KB default, max ~128KB with tuning).
Even with maximum stack limits, the kernel would:

- **Hang indefinitely** on GTX 1660 SUPER (sm_75)
- **Crash with CUDA error** on some configurations
- **Silently produce wrong results** in edge cases

### Symptoms Observed

| Test Scenario | Result |
|--------------|--------|
| Single point (count=1) | Kernel hang (timeout) |
| Multiple points (count>1) | Kernel crash or hang |
| Small MSM (count<16) | Random crashes |
| Any random points | GPU reset required |

### Files Removed

The following files were part of the deprecated specialized kernel approach:

```
src/msm_bw6_761_specialized.cu  # Specialized Pippenger kernel (REMOVED)
src/msm_bw6_761.cu              # Generic wrapper (REMOVED)
```

### Configuration Flags Removed

```rust
// build.rs - These flags are no longer used for BW6-761
BW6_WBITS      // Window size tuning
BW6_NTHREADS   // Thread count tuning
BW6_NSTREAMS   // Stream count tuning
BW6_MAXREG     // Register limit tuning
bw6_gpu_available  // cfg flag for specialized kernel
```

### Lesson Learned

For curves with base fields >512 bits, sppark's `field_large_t<N>` template approach
generates stack frames too large for GPU execution. The solution is to use a
**cooperative threading model** like CGBN where limbs are distributed across threads,
keeping per-thread memory usage within GPU limits.

---

## References

- [NVIDIA CGBN Library](https://github.com/NVlabs/CGBN)
- sppark: `sppark/ff/bw6-761.hpp` (field definition)
- arkworks: `ark-bw6-761` crate
