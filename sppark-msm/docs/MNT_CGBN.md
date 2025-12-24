# MNT4-298 / MNT6-298 CGBN GPU MSM Implementation

**Last Updated**: 2025-12-24

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

**Current**: Serial double-and-add MSM (matches BW6-761 pattern)

```
For each (point_i, scalar_i):
  1. term = scalar_mul(point_i, scalar_i)  // Double-and-add
  2. accumulator = point_add(accumulator, term)
Return accumulator
```

**Complexity**: O(n * log(scalar_bits))

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
| CGBN kernel | Working | Working |
| Field arithmetic | Working | Working |
| Point doubling (a != 0) | Working | Working |
| Scalar multiplication | Working | Working |
| MSM accumulation | Working | Working |
| Rust FFI | Working | Working |
| Layout validation | Passing | Passing |

---

## Future Optimizations

1. **Pippenger Algorithm**: O(n / log(n)) for large MSMs
2. **Parallel Scalar Multiplications**: Multiple thread groups + tree reduction
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
