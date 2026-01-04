# Circuit Size Options for Testing and Development

This document explains the circuit size reduction options available for faster testing and development iteration. The outer circuit verifies a Groth16 proof inside R1CS, and its size directly impacts CRS generation time due to O(n²) scaling of coefficient pairs.

## Background: Why Circuit Size Matters

The sparse quotient computation generates coefficient pairs from the Cartesian product of matrix columns:

```
pairs ≈ |columns_in_A| × |columns_in_B| ≈ constraints²
```

| Constraints | Approximate Pairs | Feasibility |
|-------------|-------------------|-------------|
| 31,000 | 680 million | Infeasible (~1 year at 20 pairs/sec) |
| 14,000 | 160 million | Infeasible (~3 months at 20 pairs/sec) |
| 1,000 | 1 million | ~14 hours |
| 10 | 100 | ~4-5 seconds |

**Note:** Throughput measured at ~20-24 pairs/sec on GTX 1660 (6GB). See `GPU_FEASIBILITY_ANALYSIS.md` for detailed benchmarks.

This O(n²) scaling means even halving constraints only reduces pairs by 75%.

---

## Comparison Table

| Option | Env Var / Feature | Constraints | Pairs | Time (GTX 1660) | Security | Use Case |
|--------|-------------------|-------------|-------|-----------------|----------|----------|
| Full Circuit | (none) | ~31K | ~680M | ~1 year | Full | Production only |
| Skip Verifier | `SKIP_VERIFIER_GADGET=1` | ~14K | ~160M | ~3 months | None | Not recommended |
| Trivial Circuit | `TRIVIAL_TEST_CIRCUIT=1` | ~10 | ~100 | ~4-5 sec | None | GPU kernel testing |
| BW6 Cycle | `--features bw6-cycle` | ~31K | ~680M | ~1+ years | ~128-bit | Production cycle |
| MNT Cycle | (default) | ~31K | ~680M | ~1 year | ~100-bit | Development cycle |

**Measured throughput:** ~20-24 pairs/sec (MNT4/MNT6), ~17-21 pairs/sec (BW6-761) on GTX 1660 6GB.

---

## Circuit Mode Options

### Option 1: Full Circuit (Default)

**Configuration:** No environment variables set

**What it includes:**
- VerifyingKeyVar allocation (VK elements as circuit variables)
- BooleanInputVar (bit decomposition of public inputs)
- ProofVar allocation (proof elements A, B, C)
- Groth16VerifierGadget::verify (pairing checks)
- Public input binding constraints

**Constraint breakdown (MNT4-298/MNT6-298):**

| Component | Constraints | Description |
|-----------|-------------|-------------|
| VerifyingKeyVar | ~5,000 | VK elements (α, β, γ, δ, γ_abc) with on-curve validation |
| BooleanInputVar | ~300/input | 298-bit decomposition per public input |
| ProofVar | ~2,400 | Proof elements (A∈G1, B∈G2, C∈G1) with on-curve validation |
| Binding constraints | ~600 | Bit reconstruction and public input binding |
| Groth16VerifierGadget | ~23,000 | Miller loop, final exponentiation, pairing checks |
| **Total** | **~31,000** | |

**Pairs generated:** ~680 million

**Feasibility:** Infeasible for practical testing. Process will be killed (exit 137/SIGKILL).

---

### Option 2: Skip Verifier Gadget

**Configuration:** `SKIP_VERIFIER_GADGET=1`

**What it includes:**
- VerifyingKeyVar allocation ✓
- BooleanInputVar ✓
- ProofVar allocation ✓
- ~~Groth16VerifierGadget::verify~~ (skipped)
- Public input binding constraints ✓

**Constraint breakdown (MNT4-298/MNT6-298):**

| Component | Constraints | Description |
|-----------|-------------|-------------|
| VerifyingKeyVar | ~5,000 | Still allocated for potential future use |
| BooleanInputVar | ~300/input | Still needed for input binding |
| ProofVar | ~2,400 | Still allocated (could be removed) |
| Binding constraints | ~600 | Bit reconstruction and public input binding |
| ~~Groth16VerifierGadget~~ | ~~0~~ | Skipped |
| **Total** | **~14,000** | |

**Pairs generated:** ~160 million

**Feasibility:** Still infeasible. O(n²) scaling means 14K constraints still produces too many pairs.

**Warning printed:**
```
================================================================================
WARNING: SKIP_VERIFIER_GADGET is set!
The Groth16 verifier gadget is DISABLED. This circuit provides
NO CRYPTOGRAPHIC SECURITY - it only enforces public input binding.
Use only for GPU kernel testing and development iteration.
================================================================================
```

**Why still slow:** The constraint sources are:

1. **BooleanInputVar::new_witness** - Creates bit decomposition of each inner field element:
   - MNT4-298: 298 bits per field element
   - Each bit needs a Boolean constraint: `bit × (1 - bit) = 0`

2. **ProofVar::new_witness** - Allocates proof elements with on-curve validation:
   - G1 point: ~600 constraints (2 coordinates + curve check)
   - G2 point: ~1,200 constraints (4 coordinates for extension field + curve check)

3. **VerifyingKeyVar** - Allocates VK elements:
   - Multiple G1/G2 points with on-curve validation

---

### Option 3: Trivial Test Circuit (Recommended for GPU Testing)

**Configuration:** `TRIVIAL_TEST_CIRCUIT=1`

**What it includes:**
- ~~VerifyingKeyVar allocation~~ (skipped)
- ~~BooleanInputVar~~ (skipped)
- ~~ProofVar allocation~~ (skipped)
- ~~Groth16VerifierGadget::verify~~ (skipped)
- Minimal public input binding only ✓

**Constraint breakdown:**

| Component | Constraints | Description |
|-----------|-------------|-------------|
| Public input allocation | 0 | Just variable allocation |
| Witness allocation | 0 | Just variable allocation |
| Binding constraint | 1/input | Single `1 × x_wit = x_pub` per input |
| **Total** | **~1-10** | Depends on number of public inputs |

**Pairs generated:** ~100 or fewer

**Feasibility:** Completes in seconds.

**Warning printed:**
```
================================================================================
WARNING: TRIVIAL_TEST_CIRCUIT is set!
Using MINIMAL circuit (~10 constraints) for GPU kernel testing.
This provides NO CRYPTOGRAPHIC FUNCTIONALITY whatsoever.
Use ONLY for GPU kernel correctness and performance testing.
================================================================================
```

**Usage:**
```bash
TRIVIAL_TEST_CIRCUIT=1 cargo test --features gpu --release --test test_c_gap_random_samples -- --ignored --nocapture
```

---

## Recursion Cycle Options

The recursion cycle determines which curves are used for inner/outer proofs. This affects field sizes and consequently constraint costs.

### MNT4-298/MNT6-298 Cycle (Default)

**Configuration:** Default (no feature flag)

| Property | Value |
|----------|-------|
| Inner curve | MNT4-298 |
| Outer curve | MNT6-298 |
| Field size | 298 bits |
| Security level | ~100 bits |
| G1 point constraints | ~600 |
| G2 point constraints | ~1,200 |

**Use case:** Development and fast iteration. Acceptable security for research/testing.

---

### BLS12-377/BW6-761 Cycle

**Configuration:** `--features bw6-cycle`

| Property | Value |
|----------|-------|
| Inner curve | BLS12-377 |
| Outer curve | BW6-761 |
| Field size | 761 bits (outer) |
| Security level | ~128 bits |
| G1 point constraints | ~1,500 |
| G2 point constraints | ~3,000 |

**Use case:** Production deployments requiring higher security.

**Impact on circuit size:**
- BW6-761 has 761-bit field elements vs 298-bit for MNT6
- Approximately 2.5× more constraints per group element
- Full circuit: ~50K+ constraints (vs ~31K for MNT)
- Even more infeasible for testing

**Usage:**
```bash
# With trivial circuit for fast testing
TRIVIAL_TEST_CIRCUIT=1 cargo test --features gpu,bw6-cycle --release ...

# Full circuit (will be killed)
cargo test --features gpu,bw6-cycle --release ...
```

---

## Constraint Sources Deep Dive

### G1 Point Allocation (MNT4-298)

A G1 point has coordinates (x, y) in the base field Fq:
- x coordinate: 298-bit field element
- y coordinate: 298-bit field element
- On-curve check: y² = x³ + ax + b

Constraints needed:
- Coordinate allocation: ~0 (just variable creation)
- On-curve validation: ~300 constraints (field multiplications)
- **Total per G1 point: ~300-600 constraints**

### G2 Point Allocation (MNT4-298)

A G2 point has coordinates in the degree-2 extension field Fq²:
- x coordinate: 2 × 298-bit elements (real, imaginary)
- y coordinate: 2 × 298-bit elements
- On-curve check in extension field

Constraints needed:
- Extension field arithmetic: ~2× cost of base field
- **Total per G2 point: ~600-1,200 constraints**

### BooleanInputVar

For a 298-bit field element:
- 298 Boolean variables allocated
- Each Boolean needs: `b × (1 - b) = 0` constraint
- Plus reconstruction: `Σ bᵢ × 2ⁱ = value`
- **Total: ~300-600 constraints per field element**

---

## Recommendations

| Goal | Recommended Configuration |
|------|---------------------------|
| GPU kernel correctness testing | `TRIVIAL_TEST_CIRCUIT=1` |
| GPU performance benchmarking | `TRIVIAL_TEST_CIRCUIT=1` |
| CRS generation pipeline debugging | `TRIVIAL_TEST_CIRCUIT=1` |
| Development iteration | `TRIVIAL_TEST_CIRCUIT=1` with MNT cycle |
| Security property testing | Full circuit (requires patience or cluster) |
| Production deployment | Full circuit with `--features bw6-cycle` |

---

## Future Options

This document will be updated as new circuit reduction options are added. Potential future options include:

- **SKIP_PROOF_VAR**: Skip ProofVar allocation (saves ~2,400 constraints)
- **SKIP_VK_VAR**: Skip VerifyingKeyVar allocation (saves ~5,000 constraints)
- **MINIMAL_BINDING**: Use scalar binding instead of bit decomposition
- **CONFIGURABLE_BITS**: Reduce bit decomposition precision for testing

---

## References

- `src/outer_compressed.rs`: OuterCircuit implementation with env var checks
- `ark-groth16-pvugc/src/constraints.rs`: ProofVar and VerifyingKeyVar definitions
- `CLAUDE.md`: Quick reference for build and test commands
