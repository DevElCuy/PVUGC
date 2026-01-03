# BW6-761 CGBN GPU Scalar Mul Bug - Summary

## Status
- Resolved: 2026-01-03

## Context
BW6-761 GPU MSM uses CGBN for 768-bit field math. A single scalar multiplication failed only on GPU for specific scalar bit patterns. CPU (GMP) reference was correct.

Primary failure surfaced in `test_gpu_cpu_consistency.rs`:
- Random large MSM (n=512) failed at point index 131 (seed 512512).
- The failing scalar is 371 bits and has limbs 3-10 and 11 set.

Failing scalar limbs (little-endian 32-bit):
```
0x00000000, 0x00000000, 0x00000000, 0xfdffb2b6,
0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb,
0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43
```

Behavior highlights:
- Power-of-two scalars pass; many smaller or sparser patterns pass.
- The minimal failing pattern requires full limb 3 plus limbs 4-10 and 11.
- Splitting limb 3 or limb 10 into high/low 16-bit halves makes it pass.

## Precise Divergence
`tests/test_probe_bit.cu` found the first divergence at:
- Iteration 261 (scalar bit 109 = limb 3, bit 13).
- The bit is set, so the divergence happens on the add path.
- Pre-add state matches CPU; post-add diverges.
- X matches; Y diverges immediately after add.

Captured data at the divergence shows some Montgomery values exceed P, for example:
- `y1*PPP` has a top limb larger than P's top limb.
- `Y3 post-sub` can have top limb `0xFFFF...` (far above P).

## Root Cause
CGBN Montgomery multiplication is weakly reduced.
`cgbn_mont_mul` and `cgbn_mont_sqr` can return results in `[0, 2P)`, not guaranteed `< P`.

Our field subtraction assumes inputs are `< P`. If `b >= P`, the branch
`a + (P - b)` underflows and produces garbage, which immediately corrupts Y.

This only manifests after enough doublings/adds create an intermediate in `[P, 2P)`,
which explains the sensitivity to specific limb densities.

Supporting evidence:
- CGBN issue #15: `cgbn_mont_sqr` returned `(a^2 mod n) + n` for 971-bit case.
- CGBN montgomery cores end with carry-based reduction, not a compare-and-subtract,
  so results are only conditionally reduced.
- `cgbn_add` and `cgbn_sub` are plain integer ops. They return carry/borrow flags
  but do not auto-reduce mod P.
- Crypto SE report confirmed unreduced intermediates break ECC ops; explicit reduce fixes it.

## Fix (Applied)
Add explicit conditional reduction after every Montgomery multiplication.
A single subtraction is sufficient because CGBN outputs are at most `2P - 1`.

Recommended helper:
```cpp
// r in Montgomery form, P is modulus
__device__ __forceinline__ void field_reduce(bn_env_t env, bn_t& r, const bn_t& P) {
  if (cgbn_compare(env, r, P) >= 0) {
    cgbn_sub(env, r, r, P);
  }
}
```

Usage:
```cpp
cgbn_mont_mul(env, r, a, b, P, NP0);
field_reduce(env, r, P);
```

Additional safe practice (optional but robust):
- Reduce after `cgbn_add` if result >= P.
- If using `cgbn_sub`, add P back on borrow and then reduce if still >= P.

## Verification
- CUDA test `tests/test_bw6_scalar_mul.cu`: 55/55 pass, including the previous failing case.
- Cargo regression: all GPU consistency tests pass.

## Notes on CGBN Usage
- `cgbn_add` returns carry; `cgbn_sub` returns -1 on borrow. You must normalize.
- Montgomery ops assume operands are in a valid range; enforce `[0, P)` at boundaries.

## References
- CGBN issue: https://github.com/NVlabs/CGBN/issues/15
- CGBN montgomery cores:
  - https://raw.githubusercontent.com/NVlabs/CGBN/master/include/cgbn/core/core_mont_imad.cu
  - https://raw.githubusercontent.com/NVlabs/CGBN/master/include/cgbn/core/core_mont_xmad.cu
  - https://raw.githubusercontent.com/NVlabs/CGBN/master/include/cgbn/core/core_mont_wmad.cu
- CGBN docs: https://github.com/NVlabs/CGBN/blob/master/docs/CGBN.md
- Crypto SE discussion: https://crypto.stackexchange.com/questions/112556/elliptic-curve-addtion-not-working-in-some-specific-cases
