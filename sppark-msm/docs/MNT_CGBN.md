# MNT4-298 / MNT6-298 CGBN GPU MSM Implementation

**Last Updated**: 2026-01-03

## Overview

This document describes the CGBN-based GPU MSM implementation for MNT4-298 and MNT6-298 curves, used in the MNT4/MNT6 recursive proof cycle.

---

## Why MNT GPU Support?

The MNT4-298/MNT6-298 cycle is used for recursive proof composition where:
- **MNT4-298 G1**: Inner curve proofs
- **MNT6-298 G1**: Outer curve proofs

GPU acceleration for these curves enables faster proving in the recursive stack.

### Curve Characteristics

| Property | MNT4-298 | MNT6-298 |
|----------|----------|----------|
| Base field (Fq) | 298 bits | 298 bits |
| Scalar field (Fr) | 298 bits | 298 bits |
| Curve parameter a | 2 | 11 |
| Use case | Inner proofs | Outer proofs |

**Note**: MNT4 Fr == MNT6 Fq (cycle pair property).

---

## Architecture

### CGBN Parameters

```cpp
class mnt_cgbn_params_t {
  static const uint32_t TPI = 8;      // 8 threads cooperate per big number
  static const uint32_t BITS = 320;   // Round up from 298 bits
  static const uint32_t MAX_ROTATION = 4;
  static const uint32_t SHM_LIMIT = 0;
  static const bool CONSTANT_TIME = false;
};
```

### Data Flow

```
Rust (arkworks)                    CUDA (CGBN)
---------------                    -----------
G1Affine (Montgomery form)    ->   Convert to plain form (CPU)
  x: Fp (u64[5])                     x: u32[10]
  y: Fp (u64[5])                     y: u32[10]
  infinity: bool                     infinity: bool

BigInt<5> (scalar)            ->   scalar_cgbn_t
  limbs: u64[5]                      limbs: u32[10]
```

### Montgomery Conversion

Arkworks stores field elements in Montgomery form (`a * R mod p`), but CGBN expects plain form. The Rust FFI layer converts using `into_bigint()`:

```rust
// BigInt<5> (u64[5]) to u32[10] conversion
fn bigint_to_u32_array(bigint: BigInt<5>) -> [u32; 10] {
    let mut result = [0u32; 10];
    for (i, &limb_u64) in bigint.0.iter().enumerate() {
        result[i * 2] = limb_u64 as u32;
        result[i * 2 + 1] = (limb_u64 >> 32) as u32;
    }
    result
}
```

---

## Implementation Details

### Files

| File | Purpose |
|------|---------|
| `src/msm_mnt4_298_cgbn.cu` | MNT4 CGBN kernel (a=2) |
| `src/msm_mnt6_298_cgbn.cu` | MNT6 CGBN kernel (a=11) |
| `src/lib.rs` | Rust FFI with BigInt<5> conversion |
| `tests/test_gpu_cgbn_mnt4.rs` | MNT4 integration tests |
| `tests/test_gpu_cgbn_mnt6.rs` | MNT6 integration tests |

### Kernel Algorithm

**Available Algorithms**:

1. **Serial double-and-add MSM** (`msm_mnt4_298_g1_cgbn` / `msm_mnt6_298_g1_cgbn`)
   - Complexity: O(n × log(scalar_bits))
   - Used for small inputs or as fallback

2. **Pippenger bucket MSM** (`msm_mnt4_298_g1_cgbn_pippenger` / `msm_mnt6_298_g1_cgbn_pippenger`)
   - Complexity: O(n + nwins × 2^wbits)
   - Automatic window size selection based on n
   - Falls back to serial for n < 64

### Point Doubling (Key Difference from BW6)

BW6-761 has curve parameter `a = 0`, but MNT curves have non-zero `a`:

**BW6-761** (a=0):
```
M = 3*X1^2
```

**MNT4-298** (a=2):
```
M = 3*X1^2 + 2*ZZ1^2
```

**MNT6-298** (a=11):
```
M = 3*X1^2 + 11*ZZ1^2
// where 11*ZZ1^2 = 8*ZZ1^2 + 2*ZZ1^2 + ZZ1^2
```

### Modulus Constants

MNT4-298 Fq modulus (little-endian u32[10]):
```c
__device__ __constant__ uint32_t MNT4_298_P[10] = {
    0x71660001, 0xc90cd65a, 0x51200e12, 0x41a9e35e, 0x5d1330ea,
    0xcaeec963, 0xa7b0548e, 0xa266249d, 0xf7bcd473, 0x000003bc
};
```

MNT4-298 Fr / MNT6-298 Fq modulus:
```c
__device__ __constant__ uint32_t MNT4_298_R[10] = {
    0x00000001, 0xbb4334a4, 0x925d6ad3, 0xfb494c07, 0x5cf44194,
    0xcaeec963, 0xa7b0548e, 0xa266249d, 0xf7bcd473, 0x000003bc
};
```

---

## Usage

### Build

```bash
cargo build --release --features gpu -p sppark-msm
```

### Run Tests

```bash
# MNT4-298 tests
cargo test --release --features gpu -p sppark-msm --test test_gpu_cgbn_mnt4

# MNT6-298 tests
cargo test --release --features gpu -p sppark-msm --test test_gpu_cgbn_mnt6

# Layout validation
cargo test --release --features gpu -p sppark-msm --test layout_validation_mnt4_298
cargo test --release --features gpu -p sppark-msm --test layout_validation_mnt6_298
```

### Integration

```rust
use sppark_msm::{msm_mnt4_298_gpu_cgbn, msm_mnt6_298_gpu_cgbn};
use ark_mnt4_298::G1Affine as Mnt4G1;
use ark_mnt6_298::G1Affine as Mnt6G1;

// MNT4-298
let result = msm_mnt4_298_gpu_cgbn(&points, &scalars)?;

// MNT6-298
let result = msm_mnt6_298_gpu_cgbn(&points, &scalars)?;
```

**Note**: The `GpuMsm` trait is NOT implemented for MNT curves because MNT4 and MNT6 share the same underlying `G1Affine` type (cycle pair), causing Rust trait coherence conflicts. Use the direct function calls instead.

---

## Dispatch Integration

### msm_backend.rs

TypeId-based dispatch for MNT4-298:
```rust
if g_id == TypeId::of::<Mnt4_298_G1>() && scalar_id == TypeId::of::<BigInt<5>>() {
    let result = sppark_msm::msm_mnt4_298_gpu_cgbn(bases, scalars).ok()?;
    return Some(result);
}
```

### pvugc_outer.rs

MNT6-298 dispatch for outer proofs:
```rust
if TypeId::of::<E>() == TypeId::of::<ark_mnt6_298::MNT6_298>() {
    match msm_mnt6_298_gpu_cgbn(bases, scalars) {
        Ok(result) => return result,
        Err(e) => {
            eprintln!("[GPU MSM] MNT6-298 failed: {:?}, falling back to CPU", e);
            return E::G1::msm(bases, scalars).unwrap();
        }
    }
}
```

---

## Custom Test Harness

GPU tests use `harness = false` in Cargo.toml to avoid SIGSEGV crashes caused by Rust test framework / CUDA cleanup order conflicts (same pattern as BW6-761).

```toml
[[test]]
name = "test_gpu_cgbn_mnt4"
path = "tests/test_gpu_cgbn_mnt4.rs"
harness = false
```

---

## Current Status

| Component | MNT4-298 | MNT6-298 |
|-----------|----------|----------|
| CGBN kernel | ✅ Working | ✅ Working |
| Field arithmetic | ✅ Working | ✅ Working |
| Point doubling (a != 0) | ✅ Working | ✅ Working |
| Scalar multiplication | ✅ Working | ✅ Working |
| MSM accumulation (serial) | ✅ Working | ✅ Working |
| MSM Pippenger | ✅ Working | ✅ Working |
| Rust FFI | ✅ Working | ✅ Working |
| Layout validation | ✅ Passing | ✅ Passing |
| GPU/CPU consistency | ✅ 21/21 passing | ✅ 21/21 passing |

**Verified**: 2026-01-03

### Implementation Approach (vs BW6-761)

MNT curves use **standard multiplication** (`cgbn_mul_wide` + `cgbn_rem_wide`) instead of Montgomery multiplication. This:
- Avoids the CGBN weak reduction bug that affected BW6-761
- Returns fully reduced results in [0, P) from `cgbn_rem_wide`
- May be slightly slower than Montgomery but is correctness-safe

Detailed review and BW6-761 comparison below.

---

## Implementation Review (2026-01-03)

### Summary

Reviewed MNT4-298 and MNT6-298 CGBN implementations against the BW6-761 weak reduction bug findings. MNT uses standard multiplication (`cgbn_mul_wide` + `cgbn_rem_wide`), so the Montgomery weak-reduction issue does not apply. All GPU/CPU consistency tests pass for both curves (21/21).

### BW6-761 Bug Recap

Root cause: CGBN's `cgbn_mont_mul` returns values in [0, 2P) instead of [0, P). This can break field subtraction when inputs are >= P.

Fix: Add explicit `if (r >= P) r -= P` after every `cgbn_mont_mul` call.

### MNT4-298 Analysis

#### Critical Difference: No Montgomery Form

MNT4-298 uses **standard multiplication** instead of Montgomery:

```cpp
// MNT4-298 field_mul (lines 141-152)
void field_mul(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P) {
    typedef typename env_t::cgbn_wide_t wide_t;
    wide_t product;
    cgbn_mul_wide(_env, product, a, b);    // Full 640-bit product
    cgbn_rem_wide(_env, r, product, P);    // Exact reduction
}
```

vs BW6-761:

```cpp
// BW6-761 field_mul (uses Montgomery)
void field_mul(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P) {
    cgbn_mont_mul(_env, r, a, b, P, BW6_761_NP0);
    // MUST add reduction here!
    if (cgbn_compare(_env, r, P) >= 0) cgbn_sub(_env, r, r, P);
}
```

**Result**: `cgbn_rem_wide` returns a fully reduced result in [0, P). The weak reduction bug does NOT affect MNT4-298's `field_mul`. MNT6-298 follows the same pattern.

### Issue 1: `field_sub` Potential Aliasing Bug (MEDIUM RISK)

**Location**: `sppark-msm/src/msm_mnt4_298_cgbn.cu` lines 123-130

**Current code**:
```cpp
void field_sub(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P) {
    int32_t borrow = cgbn_sub(_env, r, a, b);
    if (borrow != 0) {
        cgbn_add(_env, r, r, P);
    }
}
```

**Problem**: If `r` aliases `b`, then:
1. `cgbn_sub(r, a, b)` computes `a - b` into `r`
2. `b` (aliased to `r`) has been overwritten
3. If borrow occurred, `cgbn_add(r, r, P)` adds P to the wrong value

**Risk level**: Medium. Call sites appear safe today, but patterns like `field_sub(temp, temp, X3, P_mod)` could be problematic.

**Recommended fix** (from BW6-761):
```cpp
void field_sub(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P) {
    bn_t temp_result;
    int32_t cmp = cgbn_compare(_env, a, b);
    if (cmp >= 0) {
        cgbn_sub(_env, temp_result, a, b);
    } else {
        bn_t p_minus_b;
        cgbn_sub(_env, p_minus_b, P, b);
        cgbn_add(_env, temp_result, a, p_minus_b);
    }
    cgbn_set(_env, r, temp_result);
}
```

### Issue 2: `field_add` Uses Expensive `cgbn_rem` (LOW RISK, PERFORMANCE)

**Location**: `sppark-msm/src/msm_mnt4_298_cgbn.cu` lines 110-114

**Current code**:
```cpp
void field_add(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P) {
    cgbn_add(_env, r, a, b);
    cgbn_rem(_env, r, r, P);  // Full modular reduction (expensive!)
}
```

**Issue**: `cgbn_rem` performs a full division. Since `a, b < P`, we have `a + b < 2P`, so a single conditional subtract is sufficient and faster.

**Recommended optimization**:
```cpp
void field_add(bn_t& r, const bn_t& a, const bn_t& b, const bn_t& P) {
    cgbn_add(_env, r, a, b);
    if (cgbn_compare(_env, r, P) >= 0) {
        cgbn_sub(_env, r, r, P);
    }
}
```

### Issue 3: `point_add_mixed` Y3 Calculation (LOW RISK)

**Location**: `sppark-msm/src/msm_mnt4_298_cgbn.cu` lines 217-221

BW6-761 fixed a potential aliasing issue in the Y3 calculation by using dedicated temp variables. MNT4-298 has the same pattern:

```cpp
// Current MNT4-298 code
field_sub(temp, Q, X3, P_mod);
field_mul(Y3, R, temp, P_mod);
field_mul(temp, Y1, PPP, P_mod);
field_sub(Y3, Y3, temp, P_mod);  // temp reused, but not aliased
```

This specific pattern is safe because `temp` is used as intermediate storage and never aliases inputs. For consistency with BW6-761 and defensive coding, consider using the clearer pattern.

### Test Status

All 21/21 GPU/CPU consistency tests pass for both curves (verified 2026-01-03).

Run:
```bash
cargo test --release --features gpu --test test_mnt4_cpu_gpu_consistency -- --nocapture
cargo test --release --features gpu --test test_mnt6_cpu_gpu_consistency -- --nocapture
```

### Comparison Table

| Aspect | BW6-761 (fixed) | MNT4-298 (current) | MNT6-298 (current) |
|--------|-----------------|--------------------|--------------------|
| Field size | 768-bit | 320-bit | 320-bit |
| `field_mul` | Montgomery + reduction | Wide mul + rem | Wide mul + rem |
| Weak reduction risk | Fixed | None | None |
| `field_sub` aliasing | Fixed | Theoretical (not triggered) | Theoretical (not triggered) |
| `field_add` | Conditional sub | Full `cgbn_rem` | Full `cgbn_rem` |
| Test coverage | 21/21 passing | 21/21 passing | 21/21 passing |

### Recommended Actions

#### ✅ Priority 1: Verify Current Tests Pass (DONE 2026-01-03)
```bash
cargo test --release --features gpu --test test_mnt4_cpu_gpu_consistency -- --nocapture
cargo test --release --features gpu --test test_mnt6_cpu_gpu_consistency -- --nocapture
```

**Result**: All 21/21 tests pass for both curves. The theoretical aliasing issues are not triggered by current call patterns.

#### Priority 2: Fix `field_sub` Aliasing (OPTIONAL - for defensive coding)

Apply the BW6-761 pattern using temp variables and explicit comparison. Not required since tests pass.

#### Priority 3: Performance Optimization (OPTIONAL)

Replace `cgbn_rem` in `field_add` with conditional subtract. Low priority since current implementation is correct.

### Notes

1. **Why different approach?** MNT4/MNT6 are 298-bit fields (padded to 320-bit). Montgomery form is most beneficial for repeated multiplications (which MSM has). The choice of standard multiplication may have been:
   - Simplicity (no need for R, R^2, NP0 constants)
   - Correctness-first approach (avoid Montgomery pitfalls)
   - Acceptable performance for 320-bit fields

2. **Trade-off**: Standard multiplication + `cgbn_rem_wide` is likely slower than Montgomery multiplication, but avoids the weak reduction bug entirely. For production, consider converting to Montgomery if performance is critical.

3. **MNT6-298**: Has identical structure to MNT4-298, so the same analysis applies.

---

## Priority Alignment (Lean e2e)

Lean e2e uses the MNT4-298/MNT6-298 cycle (DefaultCycle). Status:

1. ✅ **MNT4 G1 MSM dispatch**: Working, 21/21 tests passing
2. ✅ **MNT6 G1 MSM dispatch**: Working, 21/21 tests passing (added 2025-12-26)
3. ❌ **G2 MSM**: Still CPU-only (no GPU kernel or dispatch)

See `docs/GPU_INDEX.md` for the full GPU documentation tree.

---

## Future Optimizations

1. **Pippenger Algorithm**: ✅ Implemented (2026-01-01)
2. **Parallel Scalar Multiplications**: Multiple thread groups + tree reduction (future)
3. **TPI Tuning**: Optimize for 320-bit fields (currently uses TPI=8 like BW6's 768-bit)

---

## Error Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| -1 | Affine layout mismatch |
| -2 | Scalar layout mismatch |
| -3 | CUDA runtime error |
| -4 | Timeout |
| -5 | Kernel unavailable |

---

## References

- [NVIDIA CGBN Library](https://github.com/NVlabs/CGBN)
- [BW6-761 CGBN Implementation](BW6_761_CGBN.md)
- arkworks: `ark-mnt4-298`, `ark-mnt6-298` crates
