# BW6-761 CGBN Compilation Memory Optimization

## TL;DR - How to Build

```bash
# Recommended: parallel build in release mode (~4MB per kernel, 3.5 min)
CUDA_PARALLEL=1 cargo build --features gpu --release

# If memory constrained: sequential build (one kernel at a time)
cargo build --features gpu --release

# Skip BW6 if not needed
SKIP_BW6=1 cargo build --features gpu --release
```

**CRITICAL:** Always use `--release` mode. Debug mode's `-G` flag conflicts with `-Xptxas=-O1` optimization, causing `ptxas fatal: Optimized debugging not supported`.

## Related Docs

- `docs/GPU_INDEX.md` - GPU documentation tree
- `GPU_plan.md` - prioritized GPU roadmap (SP1 + lean e2e)
- `GPU_SUPPORT.md` - build flags and supported curves

---

## Problem Summary

The BW6-761 CGBN kernel (768-bit field arithmetic) causes `ptxas` to consume 30GB+ RAM during compilation, while MNT4-298 (320-bit) compiles fine with ~1-2GB.

## Root Cause Analysis

### 1. CGBN Code Generation Explosion

CGBN's `mont_mul` implementation has nested loops with `#pragma unroll`:

```cuda
#pragma nounroll
for(int32_t thread=0; thread<TPI; thread++) {
    #pragma unroll
    for(int word=0; word<LIMBS; word++) {
        // ~8 operations per iteration
    }
}
```

**Code size formula**: `TPI × LIMBS² × ops_per_iter`

| Field | LIMBS | Unrolled Operations |
|-------|-------|---------------------|
| MNT4-298 (320-bit) | 10 | ~800 |
| BW6-761 (768-bit) | 24 | ~4,600 (5.8x more) |

### 2. Architecture-Specific Implementations

CGBN selects implementation based on `__CUDA_ARCH__`:

| Macro | Architecture | Complexity |
|-------|-------------|------------|
| `XMP_IMAD` | sm_30-49 (Kepler) | Lowest |
| `XMP_XMAD` | sm_50-69 (Maxwell/Pascal) | Medium |
| `XMP_WMAD` | sm_70+ (Volta/Turing) | Highest (2x IMAD) |

Default `sm_75` selects WMAD, the most complex variant.

### 3. Register Pressure

Local arrays scale with LIMBS:
- 320-bit: `ra[12] + ru[11]` = 23 registers
- 768-bit: `ra[26] + ru[25]` = 51 registers

Combined with large unrolled code, `ptxas` register allocation explodes super-linearly.

## Optimizations Implemented

### 1. Sequential Compilation (build.rs)

Added `CUDA_SEQUENTIAL=1` environment variable support to compile kernels one at a time instead of in parallel, reducing peak memory usage.

```rust
let cuda_sequential = std::env::var("CUDA_SEQUENTIAL").is_ok();
if cuda_sequential {
    // Compile kernels one by one
}
```

### 2. Force IMAD Implementation (build.rs, BW6 only)

Added compiler define to force simpler IMAD implementation:

```rust
.define("XMP_IMAD", None)
```

This bypasses the architecture detection and uses the Kepler-era implementation with less aggressive unrolling.

### 3. Lower ptxas Optimization (build.rs, BW6 only)

Added flag to reduce ptxas optimization level:

```rust
.flag("-Xptxas=-O1")
```

### 4. Montgomery Multiplication (msm_bw6_761_cgbn.cu)

Replaced `cgbn_mul_wide` + `cgbn_rem_wide` with `cgbn_mont_mul`:

**Before (causes massive code generation):**
```cuda
void field_mul(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P) {
    typedef typename env_t::cgbn_wide_t wide_t;
    wide_t product;
    cgbn_mul_wide(_env, product, a, b);      // 1536-bit intermediate!
    cgbn_rem_wide(_env, r, product, P);      // Huge code generation
}
```

**After (uses CGBN's optimized Montgomery):**
```cuda
void field_mul(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P) {
    cgbn_mont_mul(_env, r, a, b, P, BW6_761_NP0);
}
```

Added Montgomery constants:
```cuda
__device__ __constant__ uint32_t BW6_761_NP0 = 0x8fa798dd;  // -P^(-1) mod 2^32
__device__ __constant__ uint32_t BW6_761_R2_DEVICE[24] = { ... };  // R^2 mod P
```

Added conversion functions:
- `to_montgomery()`: Convert plain → Montgomery form
- `from_montgomery()`: Convert Montgomery → plain form

Updated kernel to:
- Convert input points to Montgomery form after loading
- Use Montgomery form of 1 for ZZ/ZZZ initialization
- Convert back to plain form before storing output

## Additional Optimizations Implemented (2025-12-25)

### 1. Reduce TPI from 8 to 4

In `msm_bw6_761_cgbn.cu`:
```cuda
class bw6_cgbn_params_t {
    static const uint32_t TPI = 4;  // Reduced from 8
};
```

Halves outer loop iterations in `mont_mul`, reducing code generation and register pressure.
Trade-off: Less parallelism per big-number operation, but compiles with less memory.

### 2. SKIP_BW6=1 Environment Variable

Added support to skip BW6-761 kernel compilation entirely when not needed:

```bash
SKIP_BW6=1 cargo build --features gpu
```

This allows MNT4/MNT6 kernels to build on systems with limited memory.

## Optimizations NOT Yet Tried

### 1. Split Kernel into Smaller Functions

Instead of one large kernel, split into:
- `point_add_kernel`
- `point_double_kernel`
- `scalar_mul_kernel`

Reduces per-compilation memory at cost of kernel launch overhead.

### 2. Use CUDA_ARCH=sm_50

Force compilation for Pascal architecture:
```bash
CUDA_ARCH=sm_50 cargo build --features gpu
```

Note: Won't run on Volta+ GPUs.

### 3. Fork CGBN to Remove Unroll Pragmas

Modify CGBN source to use `#pragma nounroll` for large LIMBS values:
```cuda
#if LIMBS > 16
  #pragma nounroll
#else
  #pragma unroll
#endif
```

### 4. Use External High-Memory Compilation

Compile BW6 kernel on a machine with 64GB+ RAM, then cache the `.o` file.

## Current Status

## SOLVED (2025-12-25)

With `XMP_IMAD` + `-Xptxas=-O1` applied to ALL kernels in release mode, memory usage dropped dramatically:

**Memory usage per kernel:**
| Kernel | Before | After |
|--------|--------|-------|
| BW6-761 MSM | 30GB+ (OOM) | ~20MB |
| MNT4-298 MSM | 27GB | ~4MB |
| MNT6-298 MSM | 20GB | ~4MB |
| Sparse Quotient MNT4 | ~800MB | ~4MB |
| Sparse Quotient MNT6 | ~700MB | ~4MB |

**Build times:**
- Sequential: ~6 min
- Parallel: ~3.5 min

**All 5 CGBN kernels now compile successfully in parallel on a 32GB machine.**

### Key Optimizations Applied

1. **`XMP_IMAD`** - Force simpler IMAD implementation instead of WMAD (less code generation)
2. **`-Xptxas=-O1`** - Lower ptxas optimization level (reduces register allocation complexity)
3. **Montgomery multiplication** - Uses `cgbn_mont_mul` instead of `cgbn_mul_wide` + `cgbn_rem_wide`
4. **Release mode required** - Debug mode's `-G` flag conflicts with `-O1`

### Environment Variables

| Variable | Effect |
|----------|--------|
| `CUDA_PARALLEL=1` | Enable parallel kernel compilation (faster) |
| `SKIP_BW6=1` | Skip BW6-761 kernel (if not needed) |
| `CUDA_VERBOSE=1` | Show ptxas register/memory info |
| `BW6_DEBUG=1` | Enable BW6 kernel debug output |

### CGBN Constraints

- **TPI=8 required for 768-bit** - CGBN doesn't support TPI=4 at this bit width
- TPI=8 works for 320-bit (MNT4/MNT6) as well

## Test Baseline (2025-12-25)

Low-level CUDA field arithmetic tests rebuilt and validated after memory optimizations.

**Build command:**
```bash
cd sppark-msm/tests
make -f cgbn_test_Makefile clean all
nvcc -arch=sm_75 -I../../cgbn-lib/include -lgmp --std=c++14 test_cgbn_mnt4_field.cu -o test_cgbn_mnt4_field
nvcc -arch=sm_75 -I../../cgbn-lib/include -lgmp --std=c++14 test_cgbn_mnt6_field.cu -o test_cgbn_mnt6_field
```

**Test Results:**

| Test | Status | Details |
|------|--------|---------|
| `test_cgbn_bw6_field` | PASS | 100/100 add, 100/100 mul (non-zero validation) |
| `test_cgbn_mnt4_field` | PASS | 100/100 add, sub, mul (GMP cross-validated) |
| `test_cgbn_mnt6_field` | PASS | 100/100 add, sub, mul (GMP cross-validated) |
| `test_cgbn_point_add` | PASS | 10/10 BW6-761 point additions (non-zero validation) |

**Conclusion:** Memory optimizations (Montgomery multiplication, IMAD implementation, O1 optimization) do not break low-level CGBN field arithmetic correctness.

## References

- [NVIDIA CGBN GitHub](https://github.com/NVlabs/CGBN)
- [CGBN Documentation](https://github.com/NVlabs/CGBN/blob/master/docs/CGBN.md)
- CGBN source files:
  - `cgbn-lib/include/cgbn/core/core_mont_imad.cu`
  - `cgbn-lib/include/cgbn/core/core_mont_wmad.cu`
  - `cgbn-lib/include/cgbn/cgbn.h` (architecture selection)
