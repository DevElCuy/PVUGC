#![cfg(test)]

//! BW6-761 CPU MSM Edge Case Tests
//!
//! Tests edge cases and boundary conditions for MSM on CPU only.
//! These tests establish expected behavior for empty inputs, zero scalars,
//! identity points, and mixed cases.

use ark_bw6_761::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{One, Zero};

/// Edge Case 1: Empty inputs should return infinity
#[test]
fn test_msm_empty_inputs() {
    eprintln!("\n=== Edge Case 1: Empty Inputs ===");

    let points: Vec<G1Affine> = vec![];
    let scalars: Vec<Fr> = vec![];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: Point at infinity (identity element)
    assert!(result.is_zero(), "Empty MSM should return infinity");
    eprintln!("✅ Empty inputs test passed");
}

/// Edge Case 2: Single point with scalar
#[test]
fn test_msm_single_point() {
    eprintln!("\n=== Edge Case 2: Single Point ===");

    let generator = G1Affine::generator();
    let scalar = Fr::from(42u64);

    let result = G1Projective::msm(&[generator], &[scalar]).unwrap();

    // Expected: 42*G (computed directly)
    let expected = generator.into_group() * scalar;

    assert_eq!(
        result.into_affine(),
        expected.into_affine(),
        "Single point MSM should equal scalar multiplication"
    );
    eprintln!("✅ Single point test passed");
}

/// Edge Case 3: All zero scalars should return infinity
#[test]
fn test_msm_all_zero_scalars() {
    eprintln!("\n=== Edge Case 3: All Zero Scalars ===");

    let generator = G1Affine::generator();
    let points = vec![generator, generator];
    let scalars = vec![Fr::zero(), Fr::zero()];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: Point at infinity (0*G + 0*G = ∞)
    assert!(result.is_zero(), "All zero scalars should return infinity");
    eprintln!("✅ All zero scalars test passed");
}

/// Edge Case 4: All identity points should return infinity
#[test]
fn test_msm_all_identity_points() {
    eprintln!("\n=== Edge Case 4: All Identity Points ===");

    let identity = G1Affine::identity(); // Point at infinity
    let points = vec![identity, identity];
    let scalars = vec![Fr::from(1u64), Fr::from(2u64)];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: Point at infinity (1*∞ + 2*∞ = ∞)
    assert!(result.is_zero(), "All identity points should return infinity");
    eprintln!("✅ All identity points test passed");
}

/// Edge Case 5: Mixed identity and non-identity points
#[test]
fn test_msm_mixed_identity() {
    eprintln!("\n=== Edge Case 5: Mixed Identity and Non-Identity ===");

    let generator = G1Affine::generator();
    let identity = G1Affine::identity();
    let points = vec![generator, identity, generator];
    let scalars = vec![Fr::from(2u64), Fr::from(100u64), Fr::from(3u64)];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: 2*G + 100*∞ + 3*G = 2*G + 3*G = 5*G
    let expected = generator.into_group() * Fr::from(5u64);

    assert_eq!(
        result.into_affine(),
        expected.into_affine(),
        "Mixed identity should ignore infinity points"
    );
    eprintln!("✅ Mixed identity test passed");
}

/// Edge Case 6: Single point with zero scalar
#[test]
fn test_msm_single_point_zero_scalar() {
    eprintln!("\n=== Edge Case 6: Single Point with Zero Scalar ===");

    let generator = G1Affine::generator();
    let scalar = Fr::zero();

    let result = G1Projective::msm(&[generator], &[scalar]).unwrap();

    // Expected: Point at infinity (0*G = ∞)
    assert!(result.is_zero(), "Zero scalar should return infinity");
    eprintln!("✅ Single point with zero scalar test passed");
}

/// Edge Case 7: Large number of points with all zeros
#[test]
fn test_msm_many_points_zero_scalars() {
    eprintln!("\n=== Edge Case 7: Many Points with All Zero Scalars ===");

    let generator = G1Affine::generator();
    let points = vec![generator; 100];
    let scalars = vec![Fr::zero(); 100];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: Point at infinity
    assert!(result.is_zero(), "Many points with zero scalars should return infinity");
    eprintln!("✅ Many points with zero scalars test passed");
}

/// Edge Case 8: Scalar = 1 for multiple points
#[test]
fn test_msm_all_scalar_one() {
    eprintln!("\n=== Edge Case 8: All Scalars Equal to 1 ===");

    let generator = G1Affine::generator();
    let points = vec![generator; 5];
    let scalars = vec![Fr::one(); 5];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: 1*G + 1*G + 1*G + 1*G + 1*G = 5*G
    let expected = generator.into_group() * Fr::from(5u64);

    assert_eq!(
        result.into_affine(),
        expected.into_affine(),
        "All scalar=1 should equal sum of points"
    );
    eprintln!("✅ All scalar=1 test passed");
}

/// Edge Case 9: Negation test (positive + negative = 0)
#[test]
fn test_msm_cancellation() {
    eprintln!("\n=== Edge Case 9: Cancellation (a*G + (-a)*G = ∞) ===");

    let generator = G1Affine::generator();
    let scalar_a = Fr::from(42u64);
    let scalar_neg_a = -scalar_a;

    let points = vec![generator, generator];
    let scalars = vec![scalar_a, scalar_neg_a];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: 42*G + (-42)*G = ∞
    assert!(result.is_zero(), "Cancellation should return infinity");
    eprintln!("✅ Cancellation test passed");
}

/// Edge Case 10: Very small and very large scalars together
#[test]
fn test_msm_extreme_scalars() {
    eprintln!("\n=== Edge Case 10: Extreme Scalars (1 and MODULUS-1) ===");

    let generator = G1Affine::generator();
    let scalar_one = Fr::one();
    let scalar_large = Fr::from(-1i64); // MODULUS - 1

    let points = vec![generator, generator];
    let scalars = vec![scalar_one, scalar_large];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: 1*G + (MODULUS-1)*G = 1*G + (-1)*G = ∞
    assert!(result.is_zero(), "1*G + (-1)*G should cancel to infinity");
    eprintln!("✅ Extreme scalars test passed");
}

/// Edge Case 11: Same point multiple times
#[test]
fn test_msm_duplicate_points() {
    eprintln!("\n=== Edge Case 11: Same Point Multiple Times ===");

    let generator = G1Affine::generator();
    let points = vec![generator, generator, generator, generator];
    let scalars = vec![
        Fr::from(1u64),
        Fr::from(2u64),
        Fr::from(3u64),
        Fr::from(4u64),
    ];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: (1+2+3+4)*G = 10*G
    let expected = generator.into_group() * Fr::from(10u64);

    assert_eq!(
        result.into_affine(),
        expected.into_affine(),
        "Duplicate points should sum scalars"
    );
    eprintln!("✅ Duplicate points test passed");
}

/// Edge Case 12: Mixed zeros and non-zeros
#[test]
fn test_msm_mixed_zero_nonzero() {
    eprintln!("\n=== Edge Case 12: Mixed Zero and Non-Zero Scalars ===");

    let generator = G1Affine::generator();
    let points = vec![generator, generator, generator, generator];
    let scalars = vec![
        Fr::from(5u64),
        Fr::zero(),
        Fr::from(7u64),
        Fr::zero(),
    ];

    let result = G1Projective::msm(&points, &scalars).unwrap();

    // Expected: 5*G + 0*G + 7*G + 0*G = 12*G
    let expected = generator.into_group() * Fr::from(12u64);

    assert_eq!(
        result.into_affine(),
        expected.into_affine(),
        "Mixed zero/non-zero should ignore zeros"
    );
    eprintln!("✅ Mixed zero/non-zero test passed");
}
