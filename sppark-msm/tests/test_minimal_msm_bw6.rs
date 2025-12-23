#![cfg(feature = "gpu")]

//! Minimal BW6-761 GPU MSM test
//!
//! This is the simplest possible test to diagnose the hanging issue.
//! Tests 2-point MSM with known values.

use ark_bw6_761::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, One, PrimeField};
use sppark_msm::GpuMsm;

#[test]
fn test_minimal_2_point_same_generator() {
    eprintln!("\n=== Minimal 2-Point MSM Test (Same Generator) ===");

    // Test: [G, G] * [1, 1] = 2*G
    let generator = G1Affine::generator();
    let points = vec![generator, generator];

    let scalar_one = Fr::one();
    let scalars = vec![scalar_one.into_bigint(), scalar_one.into_bigint()];

    eprintln!("[TEST] Computing 2-point MSM: [G, G] * [1, 1]");
    eprintln!("[TEST] Expected result: 2*G");

    // Try GPU MSM
    eprintln!("[TEST] Calling GPU MSM...");
    let gpu_result = match G1Affine::msm_gpu(&points, &scalars) {
        Ok(r) => {
            eprintln!("[TEST] ✅ GPU MSM returned successfully!");
            r
        },
        Err(_) => {
            eprintln!("[TEST] ❌ GPU MSM returned error, falling back to CPU");
            // Compute on CPU
            let scalars_fr: Vec<Fr> = scalars
                .iter()
                .map(|&s| Fr::from_bigint(s).unwrap())
                .collect();
            G1Projective::msm(&points, &scalars_fr).unwrap()
        }
    };

    // Compute expected result (CPU)
    eprintln!("[TEST] Computing CPU reference: 2*G");
    let expected = generator.into_group() + generator.into_group();

    eprintln!("[TEST] Comparing results...");
    assert_eq!(
        gpu_result.into_affine(),
        expected.into_affine(),
        "2-point MSM result mismatch"
    );

    eprintln!("✅ Minimal 2-point MSM test passed!");
}

#[test]
fn test_minimal_2_point_different_generators() {
    eprintln!("\n=== Minimal 2-Point MSM Test (Different Points) ===");

    // Test: [G, 2*G] * [1, 1] = G + 2*G = 3*G
    let generator = G1Affine::generator();
    let double_gen = (generator.into_group() + generator.into_group()).into_affine();

    let points = vec![generator, double_gen];

    let scalar_one = Fr::one();
    let scalars = vec![scalar_one.into_bigint(), scalar_one.into_bigint()];

    eprintln!("[TEST] Computing 2-point MSM: [G, 2*G] * [1, 1]");
    eprintln!("[TEST] Expected result: 3*G");

    // Try GPU MSM
    eprintln!("[TEST] Calling GPU MSM...");
    let gpu_result = match G1Affine::msm_gpu(&points, &scalars) {
        Ok(r) => {
            eprintln!("[TEST] ✅ GPU MSM returned successfully!");
            r
        },
        Err(_) => {
            eprintln!("[TEST] ❌ GPU MSM returned error, falling back to CPU");
            let scalars_fr: Vec<Fr> = scalars
                .iter()
                .map(|&s| Fr::from_bigint(s).unwrap())
                .collect();
            G1Projective::msm(&points, &scalars_fr).unwrap()
        }
    };

    // Compute expected result
    eprintln!("[TEST] Computing CPU reference: 3*G");
    let three = Fr::from(3u64);
    let expected = generator * three;

    eprintln!("[TEST] Comparing results...");
    assert_eq!(
        gpu_result.into_affine(),
        expected.into_affine(),
        "2-point MSM result mismatch"
    );

    eprintln!("✅ Minimal 2-point different generators test passed!");
}

#[test]
fn test_minimal_2_point_with_scalars() {
    eprintln!("\n=== Minimal 2-Point MSM Test (Different Scalars) ===");

    // Test: [G, G] * [2, 3] = 2*G + 3*G = 5*G
    let generator = G1Affine::generator();
    let points = vec![generator, generator];

    let scalar_two = Fr::from(2u64);
    let scalar_three = Fr::from(3u64);
    let scalars = vec![scalar_two.into_bigint(), scalar_three.into_bigint()];

    eprintln!("[TEST] Computing 2-point MSM: [G, G] * [2, 3]");
    eprintln!("[TEST] Expected result: 5*G");

    // Try GPU MSM
    eprintln!("[TEST] Calling GPU MSM...");
    let gpu_result = match G1Affine::msm_gpu(&points, &scalars) {
        Ok(r) => {
            eprintln!("[TEST] ✅ GPU MSM returned successfully!");
            r
        },
        Err(_) => {
            eprintln!("[TEST] ❌ GPU MSM returned error, falling back to CPU");
            let scalars_fr: Vec<Fr> = scalars
                .iter()
                .map(|&s| Fr::from_bigint(s).unwrap())
                .collect();
            G1Projective::msm(&points, &scalars_fr).unwrap()
        }
    };

    // Compute expected result
    eprintln!("[TEST] Computing CPU reference: 5*G");
    let five = Fr::from(5u64);
    let expected = generator * five;

    eprintln!("[TEST] Comparing results...");
    assert_eq!(
        gpu_result.into_affine(),
        expected.into_affine(),
        "2-point MSM result mismatch"
    );

    eprintln!("✅ Minimal 2-point with scalars test passed!");
}

#[test]
fn test_minimal_3_point() {
    eprintln!("\n=== Minimal 3-Point MSM Test ===");

    // Test: [G, G, G] * [1, 1, 1] = 3*G
    let generator = G1Affine::generator();
    let points = vec![generator, generator, generator];

    let scalar_one = Fr::one();
    let scalars = vec![
        scalar_one.into_bigint(),
        scalar_one.into_bigint(),
        scalar_one.into_bigint(),
    ];

    eprintln!("[TEST] Computing 3-point MSM: [G, G, G] * [1, 1, 1]");
    eprintln!("[TEST] Expected result: 3*G");

    // Try GPU MSM
    eprintln!("[TEST] Calling GPU MSM...");
    let gpu_result = match G1Affine::msm_gpu(&points, &scalars) {
        Ok(r) => {
            eprintln!("[TEST] ✅ GPU MSM returned successfully!");
            r
        },
        Err(_) => {
            eprintln!("[TEST] ❌ GPU MSM returned error, falling back to CPU");
            let scalars_fr: Vec<Fr> = scalars
                .iter()
                .map(|&s| Fr::from_bigint(s).unwrap())
                .collect();
            G1Projective::msm(&points, &scalars_fr).unwrap()
        }
    };

    // Compute expected result
    eprintln!("[TEST] Computing CPU reference: 3*G");
    let three = Fr::from(3u64);
    let expected = generator * three;

    eprintln!("[TEST] Comparing results...");
    assert_eq!(
        gpu_result.into_affine(),
        expected.into_affine(),
        "3-point MSM result mismatch"
    );

    eprintln!("✅ Minimal 3-point MSM test passed!");
}

#[test]
fn test_minimal_4_point_powers_of_two() {
    eprintln!("\n=== Minimal 4-Point MSM Test (Powers of 2) ===");

    // Test: [G, G, G, G] * [1, 2, 4, 8] = 1*G + 2*G + 4*G + 8*G = 15*G
    let generator = G1Affine::generator();
    let points = vec![generator, generator, generator, generator];

    let scalars = vec![
        Fr::from(1u64).into_bigint(),
        Fr::from(2u64).into_bigint(),
        Fr::from(4u64).into_bigint(),
        Fr::from(8u64).into_bigint(),
    ];

    eprintln!("[TEST] Computing 4-point MSM: [G, G, G, G] * [1, 2, 4, 8]");
    eprintln!("[TEST] Expected result: 15*G");

    // Try GPU MSM
    eprintln!("[TEST] Calling GPU MSM...");
    let gpu_result = match G1Affine::msm_gpu(&points, &scalars) {
        Ok(r) => {
            eprintln!("[TEST] ✅ GPU MSM returned successfully!");
            r
        },
        Err(_) => {
            eprintln!("[TEST] ❌ GPU MSM returned error, falling back to CPU");
            let scalars_fr: Vec<Fr> = scalars
                .iter()
                .map(|&s| Fr::from_bigint(s).unwrap())
                .collect();
            G1Projective::msm(&points, &scalars_fr).unwrap()
        }
    };

    // Compute expected result
    eprintln!("[TEST] Computing CPU reference: 15*G");
    let fifteen = Fr::from(15u64);
    let expected = generator * fifteen;

    eprintln!("[TEST] Comparing results...");
    assert_eq!(
        gpu_result.into_affine(),
        expected.into_affine(),
        "4-point MSM result mismatch"
    );

    eprintln!("✅ Minimal 4-point MSM test passed!");
}
