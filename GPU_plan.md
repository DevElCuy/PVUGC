# GPU MSM Implementation Plan

**Project:** Open-source GPU Multi-Scalar Multiplication for BLS12-377 and BW6-761
**Target Platform:** NVIDIA CUDA (primary), Apple Metal (future)
**Last Updated:** 2025-12-01

---

## Executive Summary

This plan outlines the implementation of GPU-accelerated MSM (Multi-Scalar Multiplication) for both BLS12-377 and BW6-761 curves using open-source alternatives to Icicle. The chosen approach follows the **Supranational sppark + Aleo snarkVM pattern** as the foundation.

**Current Status:**
- ✅ **BLS12-377**: Fully implemented and production-ready (all 13 tests passing)
- ⚠️ **BW6-761**: Implementation complete but blocked by sppark template instantiation issues with 761-bit fields

---

## 1. Current Implementation Status

### 1.1 BLS12-377 GPU MSM ✅ COMPLETE

**Implementation:**
- **Core Library**: Supranational sppark (Apache 2.0)
- **CUDA Kernel**: `sppark-msm/src/msm_bls12_377.cu`
- **Field Definitions**: `sppark-msm/sppark/ff/bls12-377.hpp`
- **Architecture**: sm_75 (Turing GPUs - GTX 1660)
- **CUDA Version**: 12.6

**Test Results:**
- ✅ All 13 tests passing (1.17s total)
- ✅ Small inputs (16 points)
- ✅ Medium inputs (256 points)
- ✅ Large inputs (1024 points)
- ✅ Edge cases (zero scalars, identity points, empty inputs)
- ✅ Deterministic behavior verified

**Files:**
```
sppark-msm/
├── src/
│   ├── lib.rs                    # Rust FFI exports GpuMsm trait
│   └── msm_bls12_377.cu          # CUDA implementation (working)
├── sppark/                        # Git submodule from supranational/sppark
│   └── ff/bls12-377.hpp          # 12-limb field (377 bits)
├── build.rs                       # CUDA + blst compilation
└── tests/gpu_msm_bls12_377.rs    # Comprehensive test suite
```

### 1.2 BW6-761 GPU MSM ⚠️ BLOCKED

**Implementation:**
- **Core Library**: Same sppark foundation
- **CUDA Kernel**: `sppark-msm/src/msm_bw6_761.cu` (created, disabled)
- **Field Definitions**: `sppark-msm/sppark/ff/bw6-761.hpp` (created)
- **Status**: Compilation disabled (build.rs:33)

**Blocker:**
Template instantiation errors with `mont_t<761, ...>` (24 limbs):
```
error: 'struct bw6_761::fp_t' has no member named 'zero'
error: 'bit_length' is not a member of 'bw6_761::fr_t'
error: 'degree' is not a member of 'bw6_761::fp_t'
```

**Root Cause Analysis:**
- sppark's `mont_t` template uses deep nesting with inline PTX assembly
- Works perfectly for 12-limb fields (BLS12-377, BLS12-381, alt_bn128)
- Template instantiation fails for 24-limb (761-bit) fields
- Likely causes:
  - Compiler template instantiation depth limits
  - PTX register allocation issues with large types (96 bytes)
  - Undocumented size limits in sppark's design

**Files:**
```
sppark-msm/
├── src/msm_bw6_761.cu            # CUDA kernel (ready but won't compile)
├── sppark/ff/bw6-761.hpp         # 24-limb field definition (has issues)
└── BW6_761_STATUS.md             # Detailed status and error analysis
```

---

## 2. Research: Icicle's BW6-761 Implementation

### Key Findings from `/sandbox/icicle`

**Architectural Differences:**

| Aspect | Icicle | sppark |
|--------|--------|--------|
| Template Design | Flat: `Field<CONFIG>` | Deep: `mont_t` → `wide_t` → PTX |
| Field Storage | Simple `storage<N>` array | Template class with inheritance |
| Constants | Compile-time `constexpr` generation | Runtime device constants |
| Arithmetic | Template functions | Inline PTX assembly methods |
| Scalability | Works with any limb count | Optimized for ≤12 limbs |

**Icicle's BW6-761 Success Factors:**
1. **Simpler template structure** - less nesting reduces instantiation depth
2. **Config-driven design** - `PARAMS()` macro generates constants at compile-time
3. **Modular arithmetic** - operations are template functions, not class methods
4. **Trade-off** - Less aggressive optimization than sppark, but more scalable

**Relevant Files:**
- `/sandbox/icicle/icicle/include/fields/snark_fields/bw6_761_base.cuh` - Field config
- `/sandbox/icicle/icicle/include/fields/field.cuh` - Generic Field<> implementation
- `/sandbox/icicle/icicle/include/fields/storage.cuh` - Simple storage template
- `/sandbox/icicle/icicle/include/fields/params_gen.cuh` - Compile-time param generation

---

## 3. Implementation Alternatives for BW6-761

### Option A: Hybrid Approach (Recommended)

**Description:** Use sppark for BLS12-377, implement BW6-761 using Icicle-inspired design

**Rationale:**
- Keep battle-tested, optimized sppark code for BLS12-377
- Avoid fighting sppark's template instantiation limits
- Learn from Icicle's proven BW6-761 implementation
- Maintain open-source independence (no Icicle runtime dependency)

**Implementation Steps:**

1. **Create simplified field template for large fields**
   ```
   sppark-msm/src/field_large.cuh  # New file, Icicle-inspired
   ```
   - Use `storage<N>` with simple limb array
   - Implement config struct with `PARAMS()` macro approach
   - Write template functions for add/sub/mul/reduce
   - Use loop-based PTX (simpler than sppark's aggressive inline assembly)

2. **Implement BW6-761 field using new template**
   ```cpp
   // In bw6-761.hpp
   struct fq_config {
       static constexpr storage<24> modulus = {...};
       PARAMS(modulus)  // Generate all Montgomery constants
   };
   typedef FieldLarge<fq_config> fp_t;  // Not mont_t
   ```

3. **Keep MSM pipeline compatible**
   - Ensure `fp_t` has same API as sppark's `mont_t`
   - Compatible with `jacobian_t<fp_t>`, `xyzz_t<fp_t>`, `pippenger.cuh`
   - No changes needed to higher-level MSM logic

4. **Testing and validation**
   - Port BLS12-377 test suite to BW6-761
   - Verify against arkworks CPU reference
   - Benchmark performance vs expectations

**Pros:**
- ✅ Maintains sppark's BLS12-377 performance
- ✅ Proven approach (Icicle uses it successfully)
- ✅ Clean separation of concerns
- ✅ Lower risk - isolated changes

**Cons:**
- ⚠️ Two different field implementations to maintain
- ⚠️ BW6-761 may be slower than theoretical sppark performance
- ⚠️ Requires understanding Icicle's template patterns

**Effort Estimate:** Medium (2-4 days development + testing)

---

### Option B: Simplify sppark's mont_t for Large Fields

**Description:** Modify sppark to support large fields by reducing template complexity

**Rationale:**
- Keep single unified implementation
- Fix root cause rather than work around it
- Potential to upstream improvements to sppark

**Implementation Steps:**

1. **Create mont_t variant for large fields**
   ```
   sppark/ff/mont_t_large.cuh  # New file, fork of mont_t.cuh
   ```
   - Remove deepest template nesting levels
   - Replace inline PTX assembly with simpler loops for n>16
   - Reduce `wide_t` complexity
   - Keep same external API

2. **Conditional compilation based on field size**
   ```cpp
   #if N <= 384
   #include "mont_t.cuh"        // Original optimized version
   #else
   #include "mont_t_large.cuh"  // Simplified version
   #endif
   ```

3. **Test both paths**
   - BLS12-377 still uses optimized `mont_t`
   - BW6-761 uses simplified `mont_t_large`
   - Ensure API compatibility

4. **Performance tuning**
   - Profile BW6-761 performance
   - Identify bottlenecks in simplified implementation
   - Selectively optimize critical paths

**Pros:**
- ✅ Single template design for all curves
- ✅ Maintains sppark's overall architecture
- ✅ Potential for upstream contribution
- ✅ Future-proof for other large fields

**Cons:**
- ⚠️ Requires deep understanding of sppark internals
- ⚠️ Risk of breaking BLS12-377 during refactoring
- ⚠️ May still hit compiler limits
- ⚠️ Higher complexity to maintain fork

**Effort Estimate:** High (5-7 days development + extensive testing)

---

### Option C: Split Large Field Operations

**Description:** Break 24-limb operations into two 12-limb operations

**Rationale:**
- Leverage sppark's working 12-limb infrastructure
- Avoid template instantiation issues entirely
- Use extended precision arithmetic explicitly

**Implementation Steps:**

1. **Represent 761-bit field as two 380-bit components**
   ```cpp
   struct fp_t_split {
       mont_t<380, ...> lo;  // Lower 380 bits (12 limbs)
       mont_t<380, ...> hi;  // Upper 381 bits (12 limbs)
   };
   ```

2. **Implement field operations with carry handling**
   ```cpp
   fp_t_split add(const fp_t_split& a, const fp_t_split& b) {
       fp_t_split result;
       uint32_t carry;
       result.lo = a.lo + b.lo;  // May overflow
       carry = detect_carry(result.lo);
       result.hi = a.hi + b.hi + carry;
       return reduce(result);
   }
   ```

3. **Handle Montgomery reduction across split**
   - Implement 2-stage Montgomery reduction
   - Careful carry propagation between lo/hi
   - Maintain modular arithmetic correctness

4. **Adapt curve operations**
   - Modify `jacobian_t` and `xyzz_t` to work with split fields
   - Ensure point addition/doubling remain correct

**Pros:**
- ✅ Reuses proven 12-limb sppark code
- ✅ No template instantiation issues
- ✅ Conceptually straightforward

**Cons:**
- ⚠️ Significant performance overhead (carry handling)
- ⚠️ Complex to implement correctly (easy to introduce bugs)
- ⚠️ Non-standard approach (harder to verify)
- ⚠️ May not actually work due to Montgomery arithmetic requirements

**Effort Estimate:** High (7-10 days development + extensive validation)

**Risk:** High - Montgomery arithmetic may not decompose cleanly

---

## 4. Recommended Path Forward

### Phase 1: Implement Option A (Hybrid Approach)

**Timeline:** 2-4 days

**Deliverables:**
1. `sppark-msm/src/field_large.cuh` - Icicle-inspired field template
2. Updated `sppark-msm/sppark/ff/bw6-761.hpp` - Uses new template
3. Working `sppark-msm/src/msm_bw6_761.cu` - Compiles and runs
4. `sppark-msm/tests/gpu_msm_bw6_761.rs` - Full test suite passing

**Success Criteria:**
- ✅ All BW6-761 tests pass
- ✅ Results match arkworks CPU reference
- ✅ BLS12-377 performance unchanged
- ✅ Performance acceptable (benchmark against CPU baseline)

### Phase 2: Performance Optimization (If Needed)

**If Option A performance is insufficient:**

**Option 2.1:** Selective optimization of field_large.cuh
- Profile to identify bottlenecks
- Optimize critical paths with better PTX
- Keep template simplicity where possible

**Option 2.2:** Evaluate Option B (mont_t_large)
- If performance gap is significant (>2x slower than expected)
- If we identify specific sppark optimizations worth porting
- More effort but potentially better long-term result

### Phase 3: Apple Metal Support (Future)

**Current State:**
- No open-source alternative to Icicle for Metal + BW6-761
- Options:
  1. Keep Icicle dependency for Metal builds only
  2. Port field_large.cuh to Metal Shading Language
  3. Accept CPU-only for Apple platforms

**Decision Point:** After CUDA implementation proven

---

## 5. Technical Reference

### 5.1 Field Size Comparison

| Curve | Base Field (Fq) | Scalar Field (Fr) | Limbs (32-bit) | sppark Status |
|-------|-----------------|-------------------|----------------|---------------|
| BLS12-377 | 377 bits | 253 bits | 12 / 8 | ✅ Working |
| BLS12-381 | 381 bits | 255 bits | 12 / 8 | ✅ Working |
| BW6-761 | 761 bits | 377 bits | 24 / 12 | ❌ Blocked |

### 5.2 Key Code Locations

**Current Working Implementation:**
```
sppark-msm/
├── Cargo.toml              # Features: gpu, dependencies
├── build.rs                # CUDA + blst compilation (line 33: BW6-761 disabled)
├── src/
│   ├── lib.rs              # GpuMsm trait, FFI exports
│   ├── msm_bls12_377.cu    # BLS12-377 kernel (working)
│   └── msm_bw6_761.cu      # BW6-761 kernel (ready, won't compile)
├── sppark/                 # Git submodule: supranational/sppark
│   ├── ff/
│   │   ├── bls12-377.hpp   # 12-limb field (working)
│   │   ├── bw6-761.hpp     # 24-limb field (template issues)
│   │   └── mont_t.cuh      # Core field arithmetic template
│   └── msm/pippenger.cuh   # MSM implementation
└── tests/
    └── gpu_msm_bls12_377.rs  # 13 tests passing
```

**Icicle Reference Implementation:**
```
/sandbox/icicle/icicle/include/
├── fields/
│   ├── field.cuh              # Generic Field<CONFIG> template
│   ├── storage.cuh            # Simple storage<N> container
│   ├── params_gen.cuh         # PARAMS() macro for compile-time generation
│   └── snark_fields/
│       ├── bw6_761_base.cuh   # fq_config with modulus
│       └── bw6_761_scalar.cuh # Reuses bls12_377 (377 bits)
└── curves/params/bw6_761.cuh  # Curve parameters
```

### 5.3 Build Configuration

**Environment:**
```bash
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
```

**Build Commands:**
```bash
# BLS12-377 (working)
cargo build --features gpu
cargo test --features gpu --test gpu_msm_bls12_377

# BW6-761 (currently disabled)
# Uncomment build.rs:33 to attempt compilation
cargo build --features gpu  # Will fail with template errors
```

### 5.4 Dependencies

**Rust (Cargo.toml):**
- `ark-bls12-377 = "0.5"` - BLS12-377 curve types
- `ark-bw6-761 = "0.5"` - BW6-761 curve types
- `ark-ec = "0.5"` - Elliptic curve traits
- `ark-ff = "0.5"` - Finite field traits

**Build (build-dependencies):**
- `cc = "1.0"` - CUDA/C++ compilation

**Native:**
- CUDA Toolkit 12.6+
- sppark (git submodule at sppark-msm/sppark)
- blst (assembly for CPU fallback)

---

## 6. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-01 | Chose sppark over ec-gpu | Proven in Aleo production, better performance, BLS12-377 working |
| 2025-12-01 | BLS12-377 implementation complete | All tests passing, production-ready |
| 2025-12-01 | BW6-761 blocked by template issues | mont_t<761> fails to instantiate properly |
| 2025-12-01 | Researched Icicle implementation | Identified simpler template design as success factor |
| 2025-12-01 | Recommend Option A (Hybrid) | Lowest risk, proven approach, maintains BLS12-377 performance |

---

## 7. Open Questions

1. **Performance Target**: What is acceptable BW6-761 GPU MSM performance vs CPU?
2. **Apple Metal Priority**: How important is Metal support vs CUDA-only?
3. **Upstream Contribution**: Should we attempt to upstream large field fixes to sppark?
4. **Alternative Libraries**: Should we evaluate other libraries (cuZK, gnark, etc.) for BW6-761?

---

## 8. Success Metrics

**Technical:**
- ✅ BLS12-377 GPU MSM fully functional
- [ ] BW6-761 GPU MSM fully functional
- [ ] Both curves pass comprehensive test suites
- [ ] Performance acceptable (TBD: define baseline)

**Project:**
- ✅ Open-source implementation (no Icicle dependency)
- ✅ CUDA support on NVIDIA GPUs
- [ ] Documentation and integration guide
- [ ] Performance benchmarks published

---

## References

### Documentation
- [sppark GitHub](https://github.com/supranational/sppark) - Apache 2.0
- [snarkVM Algorithms CUDA](https://docs.rs/snarkvm-algorithms-cuda) - Aleo's GPU backend
- [Icicle Documentation](https://dev.ingonyama.com) - Reference implementation
- GPU_plan.md (this document) - Implementation plan
- BW6_761_STATUS.md - Detailed status and technical analysis

### Key Files
- `sppark-msm/BW6_761_STATUS.md` - Current blocker analysis
- `sppark-msm/build.rs:33` - BW6-761 compilation toggle
- `/sandbox/icicle/icicle/include/fields/` - Reference field implementations

---

*Plan prepared: 2025-12-01*
*Next review: After Option A prototype*
