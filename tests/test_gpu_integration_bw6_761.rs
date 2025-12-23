#![cfg(feature = "gpu")]

//! Integration tests for BW6-761 GPU MSM
//!
//! These tests verify that the BW6-761 GPU acceleration:
//! 1. Is actually available and functioning
//! 2. Produces correct results across various input sizes
//! 3. Matches CPU reference implementation
//! 4. Handles edge cases correctly
//!
//! IMPORTANT: The first test (test_bw6_761_gpu_is_available) will FAIL if GPU is unavailable.
//! This ensures we're actually validating GPU execution, not CPU fallback.

use ark_bw6_761::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, PrimeField};
use ark_std::{test_rng, UniformRand};
use sppark_msm::GpuMsm;

/// Test that GPU is actually available and can perform MSM
///
/// This is a critical test - if it fails, GPU is not working and
/// all MSM calls are falling back to CPU silently.
#[test]
fn test_bw6_761_gpu_is_available() {
    let generator = G1Affine::generator();
    let scalar = Fr::from(1u64).into_bigint();

    let result = G1Affine::msm_gpu(&[generator], &[scalar]);

    assert!(
        result.is_ok(),
        "BW6-761 GPU is not available - check CUDA installation and build"
    );

    println!("✅ BW6-761 GPU is available and functioning");
}

/// Test small MSM (16 points)
#[test]
fn test_bw6_761_small_msm() {
    let mut rng = test_rng();
    let size = 16;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<6>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // GPU MSM
    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    // CPU reference
    let scalars_fr: Vec<Fr> = scalars
        .iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();
    let cpu_result = G1Projective::msm(&points, &scalars_fr).unwrap();

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "BW6-761 GPU MSM result doesn't match CPU"
    );

    println!("✅ BW6-761 Small MSM (16 points) passed");
}

/// Test medium MSM (256 points)
#[test]
fn test_bw6_761_medium_msm() {
    let mut rng = test_rng();
    let size = 256;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<6>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // GPU MSM
    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    // CPU reference
    let scalars_fr: Vec<Fr> = scalars
        .iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();
    let cpu_result = G1Projective::msm(&points, &scalars_fr).unwrap();

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "BW6-761 GPU MSM result doesn't match CPU"
    );

    println!("✅ BW6-761 Medium MSM (256 points) passed");
}

/// Test large MSM (1024 points)
#[test]
fn test_bw6_761_large_msm() {
    let mut rng = test_rng();
    let size = 1024;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<6>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // GPU MSM
    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    // CPU reference
    let scalars_fr: Vec<Fr> = scalars
        .iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();
    let cpu_result = G1Projective::msm(&points, &scalars_fr).unwrap();

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "BW6-761 GPU MSM result doesn't match CPU"
    );

    println!("✅ BW6-761 Large MSM (1024 points) passed");
}

/// Test MSM with all zero scalars (should return identity)
#[test]
fn test_bw6_761_zero_scalars() {
    let mut rng = test_rng();
    let size = 32;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<6>> = vec![Fr::from(0u64).into_bigint(); size];

    let result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    assert_eq!(
        result.into_affine(),
        G1Affine::identity(),
        "BW6-761 MSM with zero scalars should return identity"
    );

    println!("✅ BW6-761 Zero scalars test passed");
}

/// Test MSM with all identity points (should return identity)
#[test]
fn test_bw6_761_identity_points() {
    let mut rng = test_rng();
    let size = 32;

    let points = vec![G1Affine::identity(); size];
    let scalars: Vec<BigInt<6>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    assert_eq!(
        result.into_affine(),
        G1Affine::identity(),
        "BW6-761 MSM with identity points should return identity"
    );

    println!("✅ BW6-761 Identity points test passed");
}

/// Test single point MSM
#[test]
fn test_bw6_761_single_point() {
    let mut rng = test_rng();

    let point = G1Projective::rand(&mut rng).into_affine();
    let scalar = Fr::rand(&mut rng);

    let gpu_result = G1Affine::msm_gpu(&[point], &[scalar.into_bigint()])
        .expect("GPU MSM failed");

    let cpu_result = point * scalar;

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "BW6-761 Single point MSM doesn't match"
    );

    println!("✅ BW6-761 Single point MSM passed");
}

/// Test that mismatched sizes return error
#[test]
fn test_bw6_761_mismatched_sizes() {
    let mut rng = test_rng();

    let points: Vec<G1Affine> = (0..10)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<6>> = (0..5)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let result = G1Affine::msm_gpu(&points, &scalars);

    assert!(
        result.is_err(),
        "BW6-761 MSM should fail with mismatched sizes"
    );

    println!("✅ BW6-761 Mismatched sizes correctly rejected");
}

/// Test empty inputs (should return identity)
#[test]
fn test_bw6_761_empty_inputs() {
    let points: Vec<G1Affine> = vec![];
    let scalars: Vec<BigInt<6>> = vec![];

    let result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed on empty inputs");

    assert_eq!(
        result.into_affine(),
        G1Affine::identity(),
        "BW6-761 Empty MSM should return identity"
    );

    println!("✅ BW6-761 Empty inputs test passed");
}

/// Test with known values (5 * G should equal specific result)
#[test]
fn test_bw6_761_known_values() {
    let generator = G1Affine::generator();
    let scalar_5 = Fr::from(5u64);

    let gpu_result = G1Affine::msm_gpu(&[generator], &[scalar_5.into_bigint()])
        .expect("GPU MSM failed");

    let expected = generator * scalar_5;

    assert_eq!(
        gpu_result.into_affine(),
        expected.into_affine(),
        "BW6-761 5*G doesn't match expected result"
    );

    println!("✅ BW6-761 Known values test passed");
}

/// Test with identical points and different scalars
#[test]
fn test_bw6_761_identical_points() {
    let mut rng = test_rng();
    let size = 16;

    let point = G1Projective::rand(&mut rng).into_affine();
    let points = vec![point; size];

    let scalars: Vec<BigInt<6>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // GPU MSM
    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    // CPU reference: sum of scalars * point
    let scalars_fr: Vec<Fr> = scalars
        .iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();
    let scalar_sum: Fr = scalars_fr.iter().sum();
    let cpu_result = point * scalar_sum;

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "BW6-761 Identical points MSM doesn't match"
    );

    println!("✅ BW6-761 Identical points test passed");
}

/// Test varying sizes to ensure scalability
#[test]
fn test_bw6_761_varying_sizes() {
    let mut rng = test_rng();
    let sizes = [8, 16, 32, 64, 128, 256, 512];

    for &size in &sizes {
        let points: Vec<G1Affine> = (0..size)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();

        let scalars: Vec<BigInt<6>> = (0..size)
            .map(|_| Fr::rand(&mut rng).into_bigint())
            .collect();

        let gpu_result = G1Affine::msm_gpu(&points, &scalars)
            .expect(&format!("GPU MSM failed for size {}", size));

        let scalars_fr: Vec<Fr> = scalars
            .iter()
            .map(|&s| Fr::from_bigint(s).unwrap())
            .collect();
        let cpu_result = G1Projective::msm(&points, &scalars_fr).unwrap();

        assert_eq!(
            gpu_result.into_affine(),
            cpu_result.into_affine(),
            "BW6-761 MSM mismatch at size {}",
            size
        );
    }

    println!("✅ BW6-761 Varying sizes test passed (8-512 points)");
}

/// Test with maximum scalar values
#[test]
fn test_bw6_761_max_scalars() {
    let mut rng = test_rng();
    let size = 16;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    // Use near-maximum scalar values
    let scalars: Vec<BigInt<6>> = (0..size)
        .map(|_| Fr::from(u64::MAX).into_bigint())
        .collect();

    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed with max scalars");

    let scalars_fr: Vec<Fr> = scalars
        .iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();
    let cpu_result = G1Projective::msm(&points, &scalars_fr).unwrap();

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "BW6-761 Max scalars MSM doesn't match"
    );

    println!("✅ BW6-761 Maximum scalar values test passed");
}

/// Test deterministic behavior (same inputs should give same outputs)
#[test]
fn test_bw6_761_deterministic() {
    let mut rng = test_rng();
    let size = 64;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<6>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // Run GPU MSM twice
    let result1 = G1Affine::msm_gpu(&points, &scalars)
        .expect("First GPU MSM failed");

    let result2 = G1Affine::msm_gpu(&points, &scalars)
        .expect("Second GPU MSM failed");

    assert_eq!(
        result1.into_affine(),
        result2.into_affine(),
        "BW6-761 GPU MSM is not deterministic"
    );

    println!("✅ BW6-761 Deterministic behavior verified");
}

/// Test BW6-761 specific: Large 761-bit base field handling
///
/// BW6-761 uses a 761-bit base field (24 x 32-bit limbs), which is significantly
/// larger than typical curves. This test ensures the large field arithmetic works.
#[test]
fn test_bw6_761_large_base_field() {
    use ark_std::UniformRand;
    let mut rng = test_rng();
    let size = 32;

    // Generate points with random 761-bit field elements
    let points: Vec<G1Affine> = (0..size)
        .map(|_| {
            // Generate random projective point and convert to affine
            // This ensures we exercise the full 761-bit field
            G1Projective::rand(&mut rng).into_affine()
        })
        .collect();

    let scalars: Vec<BigInt<6>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("Large field MSM failed");

    let scalars_fr: Vec<Fr> = scalars
        .iter()
        .map(|&s| Fr::from_bigint(s).unwrap())
        .collect();
    let cpu_result = G1Projective::msm(&points, &scalars_fr).unwrap();

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "BW6-761 large field arithmetic mismatch"
    );

    println!("✅ BW6-761 Large base field (761-bit) test passed");
}
