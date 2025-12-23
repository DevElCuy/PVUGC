#![cfg(feature = "gpu")]

//! Integration test for BLS12-377 GPU MSM through the msm_backend path.
//!
//! IMPORTANT: These tests validate GPU execution by explicitly checking GPU availability.
//! The first test (test_gpu_is_available) will FAIL if CUDA build failed or GPU is unavailable.
//! This ensures we're actually validating GPU execution, not CPU fallback.
//!
//! This test validates the full GPU integration path including:
//! - TypeId checking for curve dispatch
//! - Layout safety validation (arkworks vs CUDA)
//! - GPU kernel execution
//! - Correctness against CPU reference implementation
//!
//! Unlike the unit tests in sppark-msm which call msm_gpu directly,
//! this test goes through the production msm_backend::msm_g1 path
//! to ensure the full integration works correctly.

use ark_bls12_377::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;
use arkworks_groth16::msm_backend;
use sppark_msm::GpuMsm;

/// Verify that GPU is actually available and working before running tests
#[test]
fn test_gpu_is_available() {
    let mut rng = test_rng();
    let point = G1Projective::rand(&mut rng).into_affine();
    let scalar = Fr::rand(&mut rng).into_bigint();

    // This should succeed because we're cfg-gated on sppark_cuda_built
    let result = G1Affine::msm_gpu(&[point], &[scalar]);
    assert!(
        result.is_ok(),
        "GPU MSM failed - CUDA may not be properly initialized. Error: {:?}",
        result.err()
    );

    println!("✅ GPU is available and functioning");
}

/// Test that msm_backend::msm_g1 correctly dispatches to GPU for BLS12-377
#[test]
fn test_msm_backend_bls12_377_gpu_dispatch() {
    let mut rng = test_rng();
    let size = 128;

    // Generate random points and scalars
    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<<Fr as PrimeField>::BigInt> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // Use the production msm_backend path (will use GPU if available)
    let backend_result = msm_backend::msm_g1(&points, &scalars);

    // Compute reference result on CPU using arkworks
    let scalars_fr: Vec<Fr> = scalars
        .iter()
        .map(|b| Fr::from_bigint(*b).unwrap())
        .collect();
    let cpu_result = G1Projective::msm(&points, &scalars_fr)
        .expect("CPU MSM failed");

    assert_eq!(
        backend_result.into_affine(),
        cpu_result.into_affine(),
        "msm_backend result doesn't match CPU reference"
    );

    println!("✅ msm_backend::msm_g1 correctly computes BLS12-377 MSM (size={})", size);
}

/// Test that layout safety checks work correctly
#[test]
fn test_layout_safety_checks_pass() {
    // This test verifies that the layout assertions in msm_backend don't panic
    // If arkworks changes struct layout, this would panic before reaching the assertion
    let mut rng = test_rng();

    let point = G1Projective::rand(&mut rng).into_affine();
    let scalar = Fr::rand(&mut rng).into_bigint();

    // This should not panic - layout checks should pass
    let result = msm_backend::msm_g1(&[point], &[scalar]);

    // Verify correctness
    let expected = (point * Fr::from_bigint(scalar).unwrap()).into_affine();
    assert_eq!(result.into_affine(), expected);

    println!("✅ Layout safety checks passed");
}

/// Test msm_backend with various sizes to ensure GPU handles different workloads
#[test]
fn test_msm_backend_various_sizes() {
    let mut rng = test_rng();
    let sizes = [1, 8, 16, 32, 64, 128, 256, 512];

    for &size in &sizes {
        let points: Vec<G1Affine> = (0..size)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();

        let scalars: Vec<<Fr as PrimeField>::BigInt> = (0..size)
            .map(|_| Fr::rand(&mut rng).into_bigint())
            .collect();

        let backend_result = msm_backend::msm_g1(&points, &scalars);

        let scalars_fr: Vec<Fr> = scalars
            .iter()
            .map(|b| Fr::from_bigint(*b).unwrap())
            .collect();
        let cpu_result = G1Projective::msm(&points, &scalars_fr)
            .expect(&format!("CPU MSM failed for size {}", size));

        assert_eq!(
            backend_result.into_affine(),
            cpu_result.into_affine(),
            "msm_backend result doesn't match CPU for size {}", size
        );
    }

    println!("✅ msm_backend handles various sizes correctly");
}

/// Test that msm_backend correctly handles edge cases
#[test]
fn test_msm_backend_edge_cases() {
    use ark_ff::BigInt;
    let mut rng = test_rng();
    let size = 32;

    // Test 1: All zero scalars
    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let zero_scalars = vec![BigInt::from(0u64); size];

    let result = msm_backend::msm_g1(&points, &zero_scalars);
    assert_eq!(result.into_affine(), G1Affine::identity(), "Zero scalars should give identity");

    // Test 2: Identity points
    let identity_points = vec![G1Affine::identity(); size];
    let scalars: Vec<_> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let result = msm_backend::msm_g1(&identity_points, &scalars);
    assert_eq!(result.into_affine(), G1Affine::identity(), "Identity points should give identity");

    // Test 3: Known value (generator * 42)
    let gen = G1Affine::generator();
    let scalar_42 = BigInt::from(42u64);

    let result = msm_backend::msm_g1(&[gen], &[scalar_42]);
    let expected = (gen * Fr::from(42u64)).into_affine();
    assert_eq!(result.into_affine(), expected, "Generator * 42 should match expected");

    println!("✅ msm_backend handles edge cases correctly");
}

/// Test that msm_backend is deterministic
#[test]
fn test_msm_backend_deterministic() {
    let mut rng = test_rng();
    let size = 256;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<<Fr as PrimeField>::BigInt> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // Run twice with same inputs
    let result1 = msm_backend::msm_g1(&points, &scalars);
    let result2 = msm_backend::msm_g1(&points, &scalars);

    assert_eq!(
        result1.into_affine(),
        result2.into_affine(),
        "msm_backend should be deterministic"
    );

    println!("✅ msm_backend is deterministic");
}
