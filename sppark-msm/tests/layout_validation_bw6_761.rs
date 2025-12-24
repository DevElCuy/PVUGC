// BW6-761 Layout Validation Tests
// Custom test harness (harness = false) to avoid Rust test framework CUDA cleanup conflicts
//
// CRITICAL: Why Custom Harness?
// ===============================
// The default Rust test framework causes SIGSEGV crashes due to CUDA cleanup order:
// 1. Test runs on worker thread
// 2. CUDA context is thread-local
// 3. When worker thread exits, CUDA runtime destroys context automatically
// 4. Rust test framework then tries to Drop variables that reference destroyed CUDA state
// 5. Result: SIGSEGV (NULL pointer dereference)
//
// Solution: Custom harness = main() function, no worker threads, predictable cleanup order.
//
// Run with: cargo test --release --features gpu --test layout_validation_bw6_761
//
// STATUS: All tests PASS - the CGBN kernel produces correct results.

#![cfg(all(feature = "gpu", bw6_cgbn_available))]

//! Runtime layout validation test for BW6-761
//!
//! This test validates that the actual memory layout of arkworks BW6-761 types
//! matches what the CUDA code expects. It goes beyond size checks to verify
//! field ordering by using known values.
//!
//! CRITICAL: This test catches field reordering that size-only checks miss.
//! If arkworks reorders fields in G1Affine or BigInt<6>, this test will fail.
//!
//! Note: BW6-761 has a 761-bit base field (much larger than BLS12-377's 377 bits)
//! but the same 377-bit scalar field as BLS12-377's base field.

use ark_bw6_761::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInt, PrimeField};
use std::mem::ManuallyDrop;

/// Call CGBN MSM kernel via public wrapper
fn call_msm(
    points: &[G1Affine],
    scalars: &[BigInt<6>],
) -> Result<G1Projective, ()> {
    use sppark_msm::msm_bw6_761_gpu_cgbn;

    msm_bw6_761_gpu_cgbn(points, scalars).map_err(|_| ())
}

// ============================================================================
// Test Functions
// ============================================================================

/// Test that validates BW6-761 G1Affine layout using known generator point
fn test_bw6_761_g1affine_layout_with_generator() -> bool {
    println!("\n=== Test: G1Affine Layout with Generator ===");

    let generator = G1Affine::generator();
    let scalar_one = Fr::from(1u64).into_bigint();

    let points = ManuallyDrop::new(vec![generator]);
    let scalars = ManuallyDrop::new(vec![scalar_one]);

    let result = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ GPU MSM failed");
            return false;
        }
    };

    let result_affine = result.into_affine();

    if result_affine != generator {
        println!("  ❌ Generator MSM failed - CUDA may be reading fields in wrong order");
        println!("  Expected: {:?}", generator);
        println!("  Got: {:?}", result_affine);
        return false;
    }

    println!("  ✅ PASS - G1Affine layout validated");
    true
}

/// Test that validates BigInt<6> layout using known scalar values
fn test_bw6_761_bigint_layout_with_known_scalars() -> bool {
    println!("\n=== Test: BigInt<6> Layout with Known Scalars ===");

    let generator = G1Affine::generator();

    // Test with scalar = 2
    let scalar_2 = Fr::from(2u64).into_bigint();
    let points = ManuallyDrop::new(vec![generator]);
    let scalars = ManuallyDrop::new(vec![scalar_2]);

    let result_2 = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ GPU MSM failed for scalar=2");
            return false;
        }
    };

    let expected_2 = (generator * Fr::from(2u64)).into_affine();

    if result_2.into_affine() != expected_2 {
        println!("  ❌ Scalar=2 MSM failed - BigInt<6> limb ordering issue");
        return false;
    }

    // Test with larger scalar
    let scalar_large = Fr::from(12345678901234567890u64).into_bigint();
    let scalars_large = ManuallyDrop::new(vec![scalar_large]);

    let result_large = match call_msm(&points, &scalars_large) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ GPU MSM failed for large scalar");
            return false;
        }
    };

    let expected_large = (generator * Fr::from(12345678901234567890u64)).into_affine();

    if result_large.into_affine() != expected_large {
        println!("  ❌ Large scalar MSM failed - BigInt<6> limb ordering issue");
        return false;
    }

    println!("  ✅ PASS - BigInt<6> layout validated");
    true
}

/// Test with multiple points to catch subtle layout issues
fn test_bw6_761_multi_point_layout_validation() -> bool {
    println!("\n=== Test: Multi-point Layout Validation ===");

    use ark_std::{UniformRand, test_rng};

    let mut rng = test_rng();
    let count = 10;

    let points: Vec<G1Affine> = (0..count)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<6>> = vec![Fr::from(1u64).into_bigint(); count];

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    let gpu_result = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ Multi-point GPU MSM failed");
            return false;
        }
    };

    // CPU reference: sum all points
    let cpu_result: G1Projective = points.iter()
        .map(|&p| G1Projective::from(p))
        .sum();

    if gpu_result.into_affine() != cpu_result.into_affine() {
        println!("  ❌ Multi-point MSM with scalar=1 failed - layout issue in points");
        return false;
    }

    println!("  ✅ PASS - Multi-point layout validated");
    true
}

/// Test that infinity flag is correctly interpreted
fn test_bw6_761_infinity_flag_layout() -> bool {
    println!("\n=== Test: Infinity Flag Layout ===");

    let identity = G1Affine::identity();
    let generator = G1Affine::generator();

    let points = ManuallyDrop::new(vec![identity, generator, identity]);
    let scalars = ManuallyDrop::new(vec![
        Fr::from(5u64).into_bigint(),
        Fr::from(3u64).into_bigint(),
        Fr::from(7u64).into_bigint(),
    ]);

    let result = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ Infinity flag test MSM failed");
            return false;
        }
    };

    // Expected: 5*∞ + 3*G + 7*∞ = 3*G
    let expected = (generator * Fr::from(3u64)).into_affine();

    if result.into_affine() != expected {
        println!("  ❌ Identity point MSM failed - infinity flag may be misread");
        return false;
    }

    println!("  ✅ PASS - Infinity flag layout validated");
    true
}

/// Comprehensive layout validation combining all checks
fn test_bw6_761_comprehensive_layout_validation() -> bool {
    println!("\n=== Test: Comprehensive Layout Validation ===");

    use ark_std::{UniformRand, test_rng};

    let mut rng = test_rng();

    let points = ManuallyDrop::new(vec![
        G1Affine::identity(),
        G1Affine::generator(),
        G1Projective::rand(&mut rng).into_affine(),
    ]);

    let scalars = ManuallyDrop::new(vec![
        Fr::from(0u64).into_bigint(),
        Fr::from(1u64).into_bigint(),
        Fr::from(u64::MAX).into_bigint(),
    ]);

    let gpu_result = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ Comprehensive layout test MSM failed");
            return false;
        }
    };

    // CPU reference
    let scalars_fr: Vec<Fr> = scalars.iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();

    let cpu_result: G1Projective = points.iter()
        .zip(scalars_fr.iter())
        .map(|(&p, &s)| G1Projective::from(p) * s)
        .sum();

    if gpu_result.into_affine() != cpu_result.into_affine() {
        println!("  ❌ Comprehensive layout validation failed");
        return false;
    }

    println!("  ✅ PASS - Comprehensive layout validation passed");
    println!("     - Point coordinates (761-bit field): ✓");
    println!("     - Infinity flags: ✓");
    println!("     - Scalar limbs (BigInt<6>): ✓");
    true
}

/// Test that validates large field handling
fn test_bw6_761_large_field_arithmetic() -> bool {
    println!("\n=== Test: Large Field (761-bit) Arithmetic ===");

    use ark_std::{UniformRand, test_rng};

    let mut rng = test_rng();

    let points = ManuallyDrop::new((0..5)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect::<Vec<_>>());

    let scalars = ManuallyDrop::new(vec![
        Fr::from(1u64).into_bigint(),
        Fr::from(2u64).into_bigint(),
        Fr::from(u64::MAX).into_bigint(),
        Fr::from(u64::MAX - 1).into_bigint(),
        Fr::from(12345u64).into_bigint(),
    ]);

    let gpu_result = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ Large field MSM failed");
            return false;
        }
    };

    // CPU reference
    let scalars_fr: Vec<Fr> = scalars.iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();

    let cpu_result: G1Projective = points.iter()
        .zip(scalars_fr.iter())
        .map(|(&p, &s)| G1Projective::from(p) * s)
        .sum();

    if gpu_result.into_affine() != cpu_result.into_affine() {
        println!("  ❌ Large field arithmetic failed");
        return false;
    }

    println!("  ✅ PASS - Large field (761-bit) arithmetic validated");
    true
}

// ============================================================================
// Main Test Runner (Custom Harness)
// ============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║  BW6-761 Layout Validation Tests (Custom Harness) ║");
    println!("╚════════════════════════════════════════════════════╝");

    let mut passed = 0;
    let mut failed = 0;

    let tests: Vec<(&str, fn() -> bool)> = vec![
        ("test_bw6_761_g1affine_layout_with_generator", test_bw6_761_g1affine_layout_with_generator),
        ("test_bw6_761_bigint_layout_with_known_scalars", test_bw6_761_bigint_layout_with_known_scalars),
        ("test_bw6_761_multi_point_layout_validation", test_bw6_761_multi_point_layout_validation),
        ("test_bw6_761_infinity_flag_layout", test_bw6_761_infinity_flag_layout),
        ("test_bw6_761_comprehensive_layout_validation", test_bw6_761_comprehensive_layout_validation),
        ("test_bw6_761_large_field_arithmetic", test_bw6_761_large_field_arithmetic),
    ];

    for (name, test_fn) in tests {
        if test_fn() {
            passed += 1;
        } else {
            failed += 1;
            eprintln!("\n❌ Test failed: {}", name);
        }
    }

    println!("\n╔════════════════════════════════════════════════════╗");
    println!("║  Test Summary                                      ║");
    println!("╚════════════════════════════════════════════════════╝");
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);

    if failed == 0 {
        println!("\n  ✅ ALL TESTS PASSED");
        std::process::exit(0);
    } else {
        println!("\n  ❌ SOME TESTS FAILED");
        std::process::exit(1);
    }
}
