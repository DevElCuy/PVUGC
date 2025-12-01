#![cfg(feature = "gpu")]

use ark_bls12_377::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_std::test_rng;
use sppark_msm::GpuMsm;

/// Helper to convert BigInt scalars to Fr for CPU MSM
fn bigints_to_frs(bigints: &[BigInt<4>]) -> Vec<Fr> {
    bigints.iter().map(|b| Fr::from_bigint(*b).unwrap()).collect()
}

/// Test basic GPU MSM with a small number of points
#[test]
fn test_gpu_msm_small() {
    let mut rng = test_rng();
    let size = 16;

    // Generate random points and scalars
    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<4>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // Compute MSM on GPU
    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    // Compute MSM on CPU for verification
    let scalars_fr = bigints_to_frs(&scalars);
    let cpu_result = G1Projective::msm(&points, &scalars_fr)
        .expect("CPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "GPU and CPU MSM results don't match for size {}", size
    );
}

/// Test GPU MSM with medium-sized inputs
#[test]
fn test_gpu_msm_medium() {
    let mut rng = test_rng();
    let size = 256;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<4>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    let scalars_fr = bigints_to_frs(&scalars);
    let cpu_result = G1Projective::msm(&points, &scalars_fr)
        .expect("CPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "GPU and CPU MSM results don't match for size {}", size
    );
}

/// Test GPU MSM with larger inputs (typical proof size)
#[test]
fn test_gpu_msm_large() {
    let mut rng = test_rng();
    let size = 1024;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<4>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    let scalars_fr = bigints_to_frs(&scalars);
    let cpu_result = G1Projective::msm(&points, &scalars_fr)
        .expect("CPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "GPU and CPU MSM results don't match for size {}", size
    );
}

/// Test GPU MSM with all-zero scalars (should return identity)
#[test]
fn test_gpu_msm_zero_scalars() {
    let mut rng = test_rng();
    let size = 64;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    // All zero scalars
    let scalars: Vec<BigInt<4>> = vec![BigInt::from(0u64); size];

    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        G1Affine::identity(),
        "MSM with zero scalars should return identity"
    );
}

/// Test GPU MSM with identity points
#[test]
fn test_gpu_msm_identity_points() {
    let mut rng = test_rng();
    let size = 64;

    // All identity points
    let points: Vec<G1Affine> = vec![G1Affine::identity(); size];

    let scalars: Vec<BigInt<4>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        G1Affine::identity(),
        "MSM with identity points should return identity"
    );
}

/// Test GPU MSM with single point
#[test]
fn test_gpu_msm_single_point() {
    let mut rng = test_rng();

    let point = G1Projective::rand(&mut rng).into_affine();
    let scalar = Fr::rand(&mut rng).into_bigint();

    let gpu_result = G1Affine::msm_gpu(&[point], &[scalar])
        .expect("GPU MSM failed");

    let scalar_fr = Fr::from_bigint(scalar).unwrap();
    let cpu_result = G1Projective::msm(&[point], &[scalar_fr])
        .expect("CPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "GPU and CPU MSM results don't match for single point"
    );
}

/// Test GPU MSM with mismatched input sizes (should fail)
#[test]
fn test_gpu_msm_mismatched_sizes() {
    let mut rng = test_rng();

    let points: Vec<G1Affine> = (0..10)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<4>> = (0..5)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let result = G1Affine::msm_gpu(&points, &scalars);

    assert!(result.is_err(), "MSM with mismatched sizes should fail");
}

/// Test GPU MSM with empty inputs (should return identity)
#[test]
fn test_gpu_msm_empty() {
    let points: Vec<G1Affine> = vec![];
    let scalars: Vec<BigInt<4>> = vec![];

    let result = G1Affine::msm_gpu(&points, &scalars);

    // Empty MSM should return identity (mathematically, sum of nothing is zero)
    match result {
        Ok(result) => {
            assert_eq!(
                result.into_affine(),
                G1Affine::identity(),
                "MSM with empty inputs should return identity"
            );
        }
        Err(_) => {
            panic!("GPU MSM with empty inputs should succeed and return identity");
        }
    }
}

/// Test GPU MSM with specific known values
#[test]
fn test_gpu_msm_known_values() {
    // Use generator point
    let point = G1Affine::generator();

    // Scalar = 5
    let scalar = BigInt::from(5u64);

    // Expected: 5 * G
    let expected = (G1Affine::generator() * Fr::from(5u64)).into_affine();

    let gpu_result = G1Affine::msm_gpu(&[point], &[scalar])
        .expect("GPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        expected,
        "GPU MSM with known values doesn't match expected result"
    );
}

/// Test GPU MSM with multiple identical points
#[test]
fn test_gpu_msm_identical_points() {
    let mut rng = test_rng();
    let size = 32;

    // All same point
    let point = G1Projective::rand(&mut rng).into_affine();
    let points: Vec<G1Affine> = vec![point; size];

    let scalars: Vec<BigInt<4>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    let scalars_fr = bigints_to_frs(&scalars);
    let cpu_result = G1Projective::msm(&points, &scalars_fr)
        .expect("CPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "GPU and CPU MSM results don't match for identical points"
    );
}

/// Benchmark-style test with varying sizes
#[test]
fn test_gpu_msm_varying_sizes() {
    let mut rng = test_rng();
    let sizes = [8, 16, 32, 64, 128, 256, 512];

    for &size in &sizes {
        let points: Vec<G1Affine> = (0..size)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();

        let scalars: Vec<BigInt<4>> = (0..size)
            .map(|_| Fr::rand(&mut rng).into_bigint())
            .collect();

        let gpu_result = G1Affine::msm_gpu(&points, &scalars)
            .expect(&format!("GPU MSM failed for size {}", size));

        let scalars_fr = bigints_to_frs(&scalars);
        let cpu_result = G1Projective::msm(&points, &scalars_fr)
            .expect(&format!("CPU MSM failed for size {}", size));

        assert_eq!(
            gpu_result.into_affine(),
            cpu_result.into_affine(),
            "GPU and CPU MSM results don't match for size {}", size
        );
    }
}

/// Test GPU MSM with maximum scalar values
#[test]
fn test_gpu_msm_max_scalars() {
    let mut rng = test_rng();
    let size = 32;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    // Use maximum scalar values (modulus - 1)
    let max_scalar = Fr::from(-1).into_bigint();
    let scalars: Vec<BigInt<4>> = vec![max_scalar; size];

    let gpu_result = G1Affine::msm_gpu(&points, &scalars)
        .expect("GPU MSM failed");

    let scalars_fr = bigints_to_frs(&scalars);
    let cpu_result = G1Projective::msm(&points, &scalars_fr)
        .expect("CPU MSM failed");

    assert_eq!(
        gpu_result.into_affine(),
        cpu_result.into_affine(),
        "GPU and CPU MSM results don't match for max scalars"
    );
}

/// Test that GPU MSM is deterministic (same inputs produce same outputs)
#[test]
fn test_gpu_msm_deterministic() {
    let mut rng = test_rng();
    let size = 128;

    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<4>> = (0..size)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    // Run MSM twice with same inputs
    let result1 = G1Affine::msm_gpu(&points, &scalars)
        .expect("First GPU MSM failed");

    let result2 = G1Affine::msm_gpu(&points, &scalars)
        .expect("Second GPU MSM failed");

    assert_eq!(
        result1.into_affine(),
        result2.into_affine(),
        "GPU MSM should be deterministic"
    );
}
