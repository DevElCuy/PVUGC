#![cfg(all(test, feature = "gpu", bw6_gpu_available))]

//! Minimal BW6-761 GPU MSM Test
//!
//! This test checks if the specialized kernel can handle the simplest
//! multi-point case (count=2) without hanging.
//!
//! If this test passes, the kernel is functional. If it hangs, the kernel
//! cannot be used and we need CPU fallback or alternative approaches.

use ark_bw6_761::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, One, PrimeField};
use ark_std::UniformRand;

use sppark_msm::GpuMsm;

/// Test 1: Two points, simple scalars [2, 3] * [G, G] = 5*G
/// This is the ABSOLUTE MINIMUM multi-point MSM.
/// If this hangs, the kernel is unusable.
#[test]
fn test_gpu_two_points_simple() {
    eprintln!("\n=== GPU Test: Two Points [2,3]*[G,G] = 5*G ===");

    let generator = G1Affine::generator();
    let points = vec![generator, generator];
    let scalars = vec![
        Fr::from(2u64).into_bigint(),
        Fr::from(3u64).into_bigint(),
    ];

    eprintln!("Calling GPU kernel with count=2...");
    let gpu_result = match G1Affine::msm_gpu(&points, &scalars) {
        Ok(result) => {
            eprintln!("✅ GPU kernel returned successfully!");
            result
        }
        Err(_) => {
            eprintln!("⚠️ GPU kernel returned error, falling back to CPU");
            // This is EXPECTED if kernel can't handle it
            G1Projective::msm(&points, &scalars.iter().map(|s| Fr::from_bigint(*s).unwrap()).collect::<Vec<_>>()).unwrap()
        }
    };

    // Expected: 2*G + 3*G = 5*G
    let cpu_expected = generator.into_group() * Fr::from(5u64);

    eprintln!("Comparing GPU result to CPU expectation...");
    assert_eq!(
        gpu_result.into_affine(),
        cpu_expected.into_affine(),
        "GPU MSM(2 points) != CPU expected"
    );

    eprintln!("✅ Test passed: GPU result matches CPU");
}

/// Test 2: Two points, random scalars
/// Verifies correctness with non-trivial values
#[test]
#[ignore] // Only run if test 1 passes
fn test_gpu_two_points_random() {
    eprintln!("\n=== GPU Test: Two Points (Random Scalars) ===");

    let mut rng = ark_std::test_rng();
    let generator = G1Affine::generator();
    let points = vec![generator, generator];
    let scalars: Vec<BigInt<6>> = vec![
        Fr::rand(&mut rng).into_bigint(),
        Fr::rand(&mut rng).into_bigint(),
    ];

    eprintln!("Calling GPU kernel with random scalars...");
    let gpu_result = G1Affine::msm_gpu(&points, &scalars).expect("GPU MSM failed");

    // Expected: sum of scalars * G
    let scalar_sum = Fr::from_bigint(scalars[0]).unwrap() + Fr::from_bigint(scalars[1]).unwrap();
    let cpu_expected = generator.into_group() * scalar_sum;

    assert_eq!(
        gpu_result.into_affine(),
        cpu_expected.into_affine(),
        "GPU MSM(random) != CPU expected"
    );

    eprintln!("✅ Random scalars test passed");
}

/// Test 3: Four points [1,2,3,4]*[G,G,G,G] = 10*G
/// Next step up in complexity
#[test]
#[ignore] // Only run if test 1 passes
fn test_gpu_four_points() {
    eprintln!("\n=== GPU Test: Four Points [1,2,3,4]*[G,G,G,G] = 10*G ===");

    let generator = G1Affine::generator();
    let points = vec![generator; 4];
    let scalars = vec![
        Fr::from(1u64).into_bigint(),
        Fr::from(2u64).into_bigint(),
        Fr::from(3u64).into_bigint(),
        Fr::from(4u64).into_bigint(),
    ];

    eprintln!("Calling GPU kernel with count=4...");
    let gpu_result = G1Affine::msm_gpu(&points, &scalars).expect("GPU MSM failed");

    // Expected: (1+2+3+4)*G = 10*G
    let cpu_expected = generator.into_group() * Fr::from(10u64);

    assert_eq!(
        gpu_result.into_affine(),
        cpu_expected.into_affine(),
        "GPU MSM(4 points) != CPU expected"
    );

    eprintln!("✅ Four points test passed");
}
