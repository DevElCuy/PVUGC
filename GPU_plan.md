# GPU Roadmap and Decision Log

**Last Updated**: 2026-01-03

## Hardware Constraints

**Current Dev GPU**: NVIDIA GeForce GTX 1660 SUPER
- VRAM: 6GB
- Compute Capability: sm_75 (Turing)

This constrains which optimizations are viable to develop and test:

| Optimization | Memory Need | Viable on 6GB? | Notes |
|-------------|-------------|----------------|-------|
| MNT6 G1 MSM dispatch | Same as current | ✅ Yes | Routing fix only |
| Pippenger MSM | Same as current | ✅ Yes | Algorithmic change, no extra memory |
| BW6 sparse-quotient | ~4GB with batching | ⚠️ Tight | Needs careful batch sizing like MNT |
| G2 GPU MSM | 2x point size | ❌ Risky | May exceed 6GB for large MSMs |
| GPU pairings | Large intermediate | ❌ Too heavy | Needs 8GB+ GPU |

## Status Snapshot

- **BLS12-377 G1 MSM**: Production via sppark (tests passing).
- **BW6-761 G1 MSM**: CGBN kernel integrated; gated by `ENABLE_CGBN_STUB=1`.
- **MNT4-298 / MNT6-298 G1 MSM + sparse quotient**: GPU path integrated for PVUGC.
- **GPU/CPU Integration Tests**: 4 tests passing, verifying H_ij bases match between paths.

## Priority Scopes (Tests)

- **SP1 e2e**: Uses BLS12-377 (inner) + BW6-761 (outer).
- **Lean e2e**: Uses MNT4-298 (inner) + MNT6-298 (outer) because `DefaultCycle = Mnt4Mnt6Cycle`.

## Where to Look

- `docs/GPU_INDEX.md` - ordered reading path
- `GPU_SUPPORT.md` - build flags, tests, troubleshooting
- `sppark-msm/docs/BW6_761_CGBN.md` - BW6-761 kernel design and history
- `sppark-msm/docs/MNT_CGBN.md` - MNT4/MNT6 MSM kernels
- `docs/MNT_GPU_ACCELERATION.md` - sparse quotient kernels + PVUGC integration
- `docs/BW6_CGBN_MEMORY_OPTIMIZATION.md` - CUDA build memory fixes

## Roadmap (Re-Ordered by Viability + Gain)

Priority now considers what's viable on current 6GB hardware:

### Tier 1: Viable Now (6GB GPU)

1. **MNT6 G1 MSM dispatch in lean prover** ⬅️ Quick win
   - Work: Add MNT6-298 branch in `msm_backend::msm_g1` (kernel exists, just routing).
   - Est gain: **~2–8x** on MNT6 MSM steps in lean prover.
   - Status: ❌ Not started

2. **Pippenger MSM for CGBN kernels (BW6/MNT)** ✅ DONE
   - Work: Replace serial double-and-add with Pippenger in CGBN kernels.
   - Est gain: **~2–6x** over current GPU (serial), **~5–15x** vs CPU.
   - Memory: Same as current - pure algorithmic improvement.
   - Status: ✅ Implemented for all curves (BW6-761, MNT4-298, MNT6-298)
   - BW6-761: ✅ All 21/21 tests passing (CGBN weak reduction bug fixed 2026-01-03)

3. **BW6 sparse-quotient GPU path** (SP1 e2e bottleneck)
   - Work: BW6 kernels + FFI + integration in `compute_witness_bases()` for `Bls12Bw6Cycle`.
   - Est gain: **~10–30x** on quotient coefficient phase; **~3–10x overall** SP1 e2e.
   - Memory: ~4GB with batching (tight on 6GB, needs careful batch sizing like MNT).
   - Status: ❌ Not started

### Tier 2: Needs Larger GPU (8GB+)

4. **G2 GPU MSM (BW6/MNT6)**
   - Work: New kernels + FFI + use in B-term MSM.
   - Est gain: **~1.5–4x** on G2 MSM; **~1.1–2x overall**.
   - Memory: 2x point size vs G1 - risky on 6GB.
   - Status: ❌ Not started (blocked by hardware)

5. **GPU pairings (verification path)**
   - Work: Optional backend acceleration for pairings.
   - Est gain: **~1.1–1.3x** verification time.
   - Memory: Large intermediate state - needs 8GB+ GPU.
   - Status: ❌ Not started (blocked by hardware)

### Validation Gates (Non-Perf, but Required)

- Benchmark GPU vs CPU for BW6/MNT MSM + sparse quotient.
- Remove `ENABLE_CGBN_STUB` gate after correctness validation.
- End-to-end SP1 + lean e2e runs with GPU path.
- Apple Metal feasibility (future work).

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-26 | Re-prioritized roadmap by hardware viability | 6GB GPU limits G2/pairing work; focus on MNT6 dispatch + Pippenger first |
| 2025-12-26 | Lean e2e test scale limitation documented | 680M pairs infeasible on any hardware; not a GPU issue |
| 2025-12-01 | Chose sppark over ec-gpu | Proven in production, BLS12-377 working |
| 2025-12-01 | BLS12-377 implementation complete | All tests passing |
| 2025-12-01 | BW6-761 specialized kernel attempted | Compiles but crashes due to register pressure |
| 2025-12-02 | BW6-761 CGBN kernel attempted | Test infrastructure works, initial stub non-functional |
| 2025-12-03 | CGBN test harness solution | Custom harness avoids CUDA/Rust cleanup SIGSEGV |
| 2025-12-01 | Researched Icicle implementation | Simpler template design informed CGBN choice |
| 2025-12-23 | CGBN kernel fully implemented | Field arithmetic, scalar mul, MSM all working |
| 2025-12-23 | CGBN integrated into pvugc_outer.rs | GPU path used for BW6-761 MSM |
| 2025-12-23 | Removed ICICLE dependency | sppark_snarkvm no longer depends on ICICLE |
| 2025-12-23 | Removed verbose debug logging | Output now concise, errors only to stderr |
| 2025-12-25 | Phase 4 integration tests verified | 4 tests passing: GPU/CPU consistency, determinism, edge cases |
| 2025-12-25 | Fixed CGBN duplicate symbol linker error | Added `--allow-multiple-definition` to `.cargo/config.toml` |
| 2025-12-25 | Added `-fvisibility=hidden` to CGBN builds | Reduces symbol conflicts in multi-kernel builds |
| 2026-01-01 | Pippenger MSM implemented for all CGBN curves | BW6-761, MNT4-298, MNT6-298 all have Pippenger kernels |
| 2026-01-01 | BW6-761 debugging session completed | Found CPU reference bug (aliasing in gmp_field_sub), GPU code was correct |
| 2026-01-01 | Codebase cleanup | Removed ~1500 lines of debug code, kept bug fixes (R2 constant, field_sub, point_add_mixed) |
| 2026-01-03 | BW6-761 CGBN weak reduction bug fixed | Root cause: `cgbn_mont_mul` returns [0,2P), not [0,P). Fix: add `if (r >= P) r -= P` after each Mont mul. All 21/21 tests now pass. |

## Debugging Learnings (2026-01-01 + 2026-01-03)

During BW6-761 debugging, key learnings:

1. **CPU reference code can have bugs too**: The GMP-based debug reference had an aliasing bug in `gmp_field_sub` that produced wrong "expected" values. Always verify reference implementations.

2. **CGBN aliasing considerations**: CGBN's `cgbn_sub`/`cgbn_add` may not be alias-safe. Use temp variables when output overlaps with input.

3. **Montgomery R² constant**: The original R² constant was incorrect. Verified correct value via Python: `R² mod P` where `R = 2^768`.

4. **XYZZ coordinate conversion**: When comparing GPU results against arkworks, remember GPU uses XYZZ coordinates. Convert properly: `x_affine = X/ZZ`, `y_affine = Y/ZZZ`.

5. **CGBN weak Montgomery reduction (2026-01-03)**: CGBN's `cgbn_mont_mul` returns values in [0, 2P) instead of fully reduced [0, P). This is documented in CGBN issue #15. Fix: add explicit `if (r >= P) r -= P` after every `cgbn_mont_mul` call. See `sppark-msm/docs/research-brief-cgbn-bw6-761-RESULTS.md` for full analysis.

## Open Questions

1. ~~When should Pippenger replace serial double-and-add for CGBN kernels?~~ ✅ Done - Pippenger implemented for all curves
2. What is the target benchmark threshold to keep GPU enabled by default?
3. ~~When is it safe to remove the `ENABLE_CGBN_STUB` gate?~~ ✅ Safe now - BW6-761 bug fixed (2026-01-03)
4. How important is Apple Metal support vs CUDA-only for the roadmap?

## Success Metrics

- BLS12-377 GPU MSM remains stable with full test suite.
- BW6-761 + MNT4/MNT6 CGBN tests pass consistently.
- GPU sparse quotient path matches CPU results on integration tests.
- Benchmarks show GPU speedups for large MSMs and quotient batches.
