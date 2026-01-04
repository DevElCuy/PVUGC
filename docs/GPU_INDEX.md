# GPU Docs Index (SP1 + Lean Prover)

**Last Updated**: 2026-01-03

This file orders the GPU/CUDA docs involved in getting lean prover tests ready. It points to existing docs and notes which one to read at each step.

## Priority Scopes
- **SP1 e2e**: BLS12-377 (inner) + BW6-761 (outer)
- **Lean e2e**: MNT4-298 (inner) + MNT6-298 (outer) because `DefaultCycle = Mnt4Mnt6Cycle`

## Doc Map (by purpose)
- `GPU_SUPPORT.md` - build flags, configuration, test commands, troubleshooting.
- `docs/BW6_CGBN_MEMORY_OPTIMIZATION.md` - kernel compile memory fixes and required release build flags.
- `sppark-msm/docs/BW6_761_CGBN.md` - BW6-761 CGBN MSM kernel design, FFI, tests. ✅ All 21/21 tests passing.
- `sppark-msm/docs/BW6_761_CGBN_BUG_SUMMARY.md` - CGBN weak reduction bug analysis and fix.
- `sppark-msm/docs/MNT_CGBN.md` - MNT4/MNT6 CGBN MSM kernels, FFI, tests, and BW6 bug review. ✅ All 21/21 tests passing for both curves.
- `docs/MNT_GPU_ACCELERATION.md` - end-to-end MNT GPU acceleration and sparse quotient kernels.
- `GPU_plan.md` - prioritized roadmap for SP1/lean e2e, decisions, and next steps.

## Ordered Steps
1. Build prerequisites and memory-safe compilation
   - Start with `CLAUDE.md` (release builds) and `docs/BW6_CGBN_MEMORY_OPTIMIZATION.md` (CUDA_PARALLEL, SKIP_BW6, XMP_IMAD).
2. Confirm GPU support and curve coverage
   - Read `GPU_SUPPORT.md` for build flags, configuration, and test commands.
3. Validate kernel-level correctness
   - BW6-761: `sppark-msm/docs/BW6_761_CGBN.md` and its tests.
   - MNT4/MNT6: `sppark-msm/docs/MNT_CGBN.md` and its tests.
   - BLS12-377: tests listed in `GPU_SUPPORT.md`.
4. MNT sparse quotient acceleration details
   - Read `docs/MNT_GPU_ACCELERATION.md` for kernel design, perf expectations, and test strategy.
5. Wire sparse quotient GPU into PVUGC
   - MNT path is integrated; BW6 sparse quotient is still pending (see `GPU_plan.md`).
6. Prioritized roadmap
   - Follow the ordered work list in `GPU_plan.md`.
7. Run lean tests
   - `tests/test_lean_prover.rs` for lean prover e2e.
   - `tests/test_sp1_e2e.rs` for full path.
