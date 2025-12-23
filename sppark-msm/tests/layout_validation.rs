#![cfg(all(feature = "gpu", sppark_cuda_built))]

//! Runtime layout validation test
//!
//! This test validates that the actual memory layout of arkworks types
//! matches what the CUDA code expects. It goes beyond size checks to
//! verify field ordering by using known values.
//!
//! CRITICAL: This test catches field reordering that size-only checks miss.
//! If arkworks reorders fields in G1Affine or BigInt, this test will fail.

use ark_bls12_377::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInt, PrimeField};
use sppark_msm::GpuMsm;

/// Test that validates G1Affine layout using known generator point
///
/// Strategy: Use the generator point which has known x, y coordinates.
/// If CUDA reads the coordinates correctly, field ordering is correct.
#[test]
fn test_g1affine_layout_with_generator() {
    // Generator has known, well-defined x and y coordinates
    let generator = G1Affine::generator();

    // Scalar 1 should give us back the generator
    let scalar_one = Fr::from(1u64).into_bigint();

    // This MSM should compute: 1 * G = G
    let result = G1Affine::msm_gpu(&[generator], &[scalar_one])
        .expect("GPU MSM failed - may indicate layout issue");

    let result_affine = result.into_affine();

    // If field ordering is correct, result should equal generator
    assert_eq!(
        result_affine, generator,
        "Generator MSM failed - CUDA may be reading fields in wrong order"
    );

    // Additionally verify x and y coordinates match
    assert_eq!(
        result_affine.x, generator.x,
        "X coordinate mismatch - field ordering issue"
    );
    assert_eq!(
        result_affine.y, generator.y,
        "Y coordinate mismatch - field ordering issue"
    );

    println!("✅ G1Affine layout validated: generator MSM correct");
}

/// Test that validates BigInt<4> layout using known scalar values
///
/// Strategy: Use prime field operations with known results.
/// If CUDA interprets scalars correctly, limb ordering is correct.
#[test]
fn test_bigint_layout_with_known_scalars() {
    let generator = G1Affine::generator();

    // Test with scalar = 2
    let scalar_2 = Fr::from(2u64).into_bigint();
    let result_2 = G1Affine::msm_gpu(&[generator], &[scalar_2])
        .expect("GPU MSM failed");

    // Expected: 2 * G
    let expected_2 = (generator * Fr::from(2u64)).into_affine();

    assert_eq!(
        result_2.into_affine(), expected_2,
        "Scalar=2 MSM failed - BigInt limb ordering issue"
    );

    // Test with larger scalar
    let scalar_large = Fr::from(12345678901234567890u64).into_bigint();
    let result_large = G1Affine::msm_gpu(&[generator], &[scalar_large])
        .expect("GPU MSM failed");

    let expected_large = (generator * Fr::from(12345678901234567890u64)).into_affine();

    assert_eq!(
        result_large.into_affine(), expected_large,
        "Large scalar MSM failed - BigInt limb ordering issue"
    );

    println!("✅ BigInt<4> layout validated: scalar operations correct");
}

/// Test with multiple points to catch subtle layout issues
///
/// Strategy: Use multiple points with different coordinates.
/// If any coordinate is misread, the MSM result will be wrong.
#[test]
fn test_multi_point_layout_validation() {
    use ark_std::{UniformRand, test_rng};

    let mut rng = test_rng();
    let count = 10;

    // Generate random points
    let points: Vec<G1Affine> = (0..count)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    // Use scalar = 1 for all (identity for multiplication)
    let scalars: Vec<BigInt<4>> = vec![Fr::from(1u64).into_bigint(); count];

    // GPU MSM with scalar=1 should give sum of points
    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("Multi-point GPU MSM failed");

    // CPU reference: sum all points
    let cpu_result: G1Projective = points.iter()
        .map(|&p| G1Projective::from(p))
        .sum();

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "Multi-point MSM with scalar=1 failed - layout issue in points"
    );

    println!("✅ Multi-point layout validated: coordinates read correctly");
}

/// Test that infinity flag is correctly interpreted
///
/// Strategy: Use identity point (infinity = true).
/// If CUDA misinterprets the infinity flag, results will be wrong.
#[test]
fn test_infinity_flag_layout() {
    let identity = G1Affine::identity();
    let generator = G1Affine::generator();

    // Mix identity and generator
    let points = vec![identity, generator, identity];
    let scalars = vec![
        Fr::from(5u64).into_bigint(),
        Fr::from(3u64).into_bigint(),
        Fr::from(7u64).into_bigint(),
    ];

    // Expected: 5*∞ + 3*G + 7*∞ = 3*G
    let result = G1Affine::msm_gpu(&points, &scalars)
        .expect("Infinity flag test failed");

    let expected = (generator * Fr::from(3u64)).into_affine();

    assert_eq!(
        result.into_affine(), expected,
        "Identity point MSM failed - infinity flag may be misread"
    );

    println!("✅ Infinity flag layout validated: identity points handled correctly");
}

/// Comprehensive layout validation combining all checks
#[test]
fn test_comprehensive_layout_validation() {
    use ark_std::{UniformRand, test_rng};

    let mut rng = test_rng();

    // Test various point types
    let points = vec![
        G1Affine::identity(),          // Infinity flag set
        G1Affine::generator(),         // Known coordinates
        G1Projective::rand(&mut rng).into_affine(), // Random point
    ];

    // Test various scalar types
    let scalars = vec![
        Fr::from(0u64).into_bigint(),   // Zero
        Fr::from(1u64).into_bigint(),   // One
        Fr::from(u64::MAX).into_bigint(), // Large value
    ];

    // GPU computation
    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("Comprehensive layout test failed");

    // CPU reference
    let scalars_fr: Vec<Fr> = scalars.iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();

    let cpu_result: G1Projective = points.iter()
        .zip(scalars_fr.iter())
        .map(|(&p, &s)| G1Projective::from(p) * s)
        .sum();

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "Comprehensive layout validation failed"
    );

    println!("✅ Comprehensive layout validation passed");
    println!("   - Point coordinates: ✓");
    println!("   - Infinity flags: ✓");
    println!("   - Scalar limbs: ✓");
    println!("   - All field orderings match CUDA expectations");
}
