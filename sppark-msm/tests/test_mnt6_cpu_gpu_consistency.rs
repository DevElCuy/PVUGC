// MNT6-298 CPU vs GPU Consistency Tests
// Custom test harness (harness = false) to avoid Rust test framework CUDA cleanup conflicts
//
// These tests verify that the CGBN GPU MSM produces identical results to arkworks CPU MSM.
// This is the critical validation needed before using MNT6-298 GPU MSM in production provers.
//
// Run with: cargo test --release --features gpu --test test_mnt6_cpu_gpu_consistency

#![cfg(all(feature = "gpu", mnt6_cgbn_available))]

use ark_mnt6_298::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, One, PrimeField, UniformRand, Zero};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use std::mem::ManuallyDrop;

/// Call CGBN MSM kernel via public wrapper
fn gpu_msm(points: &[G1Affine], scalars: &[Fr]) -> Result<G1Projective, i32> {
    use sppark_msm::msm_mnt6_298_gpu_cgbn;

    // Convert Fr to BigInt<5> for GPU
    let scalar_bigints: Vec<BigInt<5>> = scalars.iter().map(|s| s.into_bigint()).collect();

    match msm_mnt6_298_gpu_cgbn(points, &scalar_bigints) {
        Ok(result) => Ok(result),
        Err(_) => Err(-1),
    }
}

/// CPU MSM using arkworks
fn cpu_msm(points: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    G1Projective::msm(points, scalars).unwrap()
}

/// Compare two projective points for equality
fn points_equal(a: &G1Projective, b: &G1Projective) -> bool {
    a.into_affine() == b.into_affine()
}

// ============================================================================
// Consistency Test Functions
// ============================================================================

fn test_consistency_generator_simple() -> bool {
    println!("\n=== Test: Generator with Simple Scalars ===");

    let generator = G1Affine::generator();
    let points = vec![generator, generator];
    let scalars = vec![Fr::from(2u64), Fr::from(3u64)];

    // Expected: 2*G + 3*G = 5*G
    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                println!("  PASS: Results match");
                true
            } else {
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                println!("  FAIL: Results differ");
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_single_point() -> bool {
    println!("\n=== Test: Single Point Consistency ===");

    let generator = G1Affine::generator();
    let points = vec![generator];
    let scalars = vec![Fr::from(42u64)];

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: 42*G matches");
                true
            } else {
                println!("  FAIL: Results differ");
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_empty() -> bool {
    println!("\n=== Test: Empty Input Consistency ===");

    let points: Vec<G1Affine> = vec![];
    let scalars: Vec<Fr> = vec![];

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            // Both should return identity
            let cpu_is_zero = cpu_result.is_zero();
            let gpu_is_zero = gpu_result.is_zero();

            if cpu_is_zero && gpu_is_zero {
                println!("  PASS: Both return identity (zero)");
                true
            } else {
                println!("  FAIL: CPU zero={}, GPU zero={}", cpu_is_zero, gpu_is_zero);
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_zero_scalars() -> bool {
    println!("\n=== Test: All Zero Scalars Consistency ===");

    let generator = G1Affine::generator();
    let points = vec![generator, generator, generator];
    let scalars = vec![Fr::zero(), Fr::zero(), Fr::zero()];

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            let cpu_is_zero = cpu_result.is_zero();
            let gpu_is_zero = gpu_result.is_zero();

            if cpu_is_zero && gpu_is_zero {
                println!("  PASS: Both return identity for 0*G + 0*G + 0*G");
                true
            } else {
                println!("  FAIL: CPU zero={}, GPU zero={}", cpu_is_zero, gpu_is_zero);
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_identity_points() -> bool {
    println!("\n=== Test: Identity Points Consistency ===");

    let identity = G1Affine::identity();
    let points = vec![identity, identity];
    let scalars = vec![Fr::from(100u64), Fr::from(200u64)];

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            let cpu_is_zero = cpu_result.is_zero();
            let gpu_is_zero = gpu_result.is_zero();

            if cpu_is_zero && gpu_is_zero {
                println!("  PASS: Both return identity for scalar*infinity");
                true
            } else {
                println!("  FAIL: CPU zero={}, GPU zero={}", cpu_is_zero, gpu_is_zero);
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_cancellation() -> bool {
    println!("\n=== Test: Cancellation (a*G + (-a)*G = 0) Consistency ===");

    let generator = G1Affine::generator();
    let scalar_a = Fr::from(12345u64);
    let scalar_neg_a = -scalar_a;

    let points = vec![generator, generator];
    let scalars = vec![scalar_a, scalar_neg_a];

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            let cpu_is_zero = cpu_result.is_zero();
            let gpu_is_zero = gpu_result.is_zero();

            if cpu_is_zero && gpu_is_zero {
                println!("  PASS: Both return identity for cancellation");
                true
            } else {
                println!("  FAIL: CPU zero={}, GPU zero={}", cpu_is_zero, gpu_is_zero);
                if !gpu_is_zero {
                    println!("  GPU result: {:?}", gpu_result.into_affine());
                }
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_random_small() -> bool {
    println!("\n=== Test: Random Small MSM (n=4) Consistency ===");

    // Use fixed seed for reproducibility (different seed from MNT4 tests)
    let mut rng = StdRng::seed_from_u64(123456);

    let n = 4;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Random n=4 MSM matches");
                true
            } else {
                println!("  FAIL: Results differ");
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_random_medium() -> bool {
    println!("\n=== Test: Random Medium MSM (n=16) Consistency ===");

    let mut rng = StdRng::seed_from_u64(543210);

    let n = 16;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Random n=16 MSM matches");
                true
            } else {
                println!("  FAIL: Results differ");
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_random_larger() -> bool {
    println!("\n=== Test: Random Larger MSM (n=64) Consistency ===");

    let mut rng = StdRng::seed_from_u64(999990);

    let n = 64;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Random n=64 MSM matches");
                true
            } else {
                println!("  FAIL: Results differ");
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_mixed_edge_cases() -> bool {
    println!("\n=== Test: Mixed Edge Cases Consistency ===");

    let generator = G1Affine::generator();
    let identity = G1Affine::identity();

    // Mix of generator, identity, and various scalars including zero
    let points = vec![generator, identity, generator, generator, identity];
    let scalars = vec![
        Fr::from(5u64),
        Fr::from(100u64), // This multiplies identity, so contributes 0
        Fr::zero(),       // This is 0*G = 0
        Fr::from(7u64),
        Fr::from(50u64), // This multiplies identity, so contributes 0
    ];

    // Expected: 5*G + 0 + 0 + 7*G + 0 = 12*G
    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Mixed edge cases match (expected 12*G)");
                true
            } else {
                println!("  FAIL: Results differ");
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_large_scalars() -> bool {
    println!("\n=== Test: Large Scalar Values Consistency ===");

    let generator = G1Affine::generator();

    // Use scalars close to the field modulus
    let scalar_one = Fr::one();
    let scalar_neg_one = -Fr::one(); // This is p-1
    let scalar_two = Fr::from(2u64);

    let points = vec![generator, generator, generator];
    let scalars = vec![scalar_one, scalar_neg_one, scalar_two];

    // Expected: 1*G + (p-1)*G + 2*G = G + (-G) + 2G = 2*G
    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Large scalar values match");
                true
            } else {
                println!("  FAIL: Results differ");
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_negative_scalars() -> bool {
    println!("\n=== Test: Negative Scalars Consistency ===");

    // Critical for proving: random scalars can be "negative" (> p/2)
    let mut rng = StdRng::seed_from_u64(777770);

    let n = 16;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    // Mix of positive and negative scalars
    let scalars: Vec<Fr> = (0..n)
        .map(|i| {
            let s = Fr::rand(&mut rng);
            if i % 2 == 0 { s } else { -s }  // Alternate positive/negative
        })
        .collect();

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Negative scalars match");
                true
            } else {
                println!("  FAIL: Results differ");
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_multiple_cancellations() -> bool {
    println!("\n=== Test: Multiple Cancellations in Single MSM ===");

    // Simulate scenario where multiple point pairs cancel out
    let generator = G1Affine::generator();
    let g2 = (generator.into_group() + generator.into_group()).into_affine();
    let g3 = (g2.into_group() + generator.into_group()).into_affine();

    // Pattern: a*G - a*G + b*2G - b*2G + c*3G = c*3G
    let a = Fr::from(100u64);
    let b = Fr::from(200u64);
    let c = Fr::from(7u64);

    let points = vec![generator, generator, g2, g2, g3];
    let scalars = vec![a, -a, b, -b, c];

    let cpu_result = cpu_msm(&points, &scalars);

    // Verify CPU result is c*3G = 21*G
    let expected = (generator.into_group() * Fr::from(21u64)).into_affine();
    if cpu_result.into_affine() != expected {
        println!("  FAIL: CPU result unexpected");
        return false;
    }

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Multiple cancellations handled correctly");
                true
            } else {
                println!("  FAIL: Results differ");
                println!("  Expected: 21*G");
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_random_256() -> bool {
    println!("\n=== Test: Random Large MSM (n=256) Consistency ===");

    let mut rng = StdRng::seed_from_u64(2562560);

    let n = 256;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Random n=256 MSM matches");
                true
            } else {
                println!("  FAIL: Results differ");
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_random_512() -> bool {
    println!("\n=== Test: Random Large MSM (n=512) Consistency ===");

    let mut rng = StdRng::seed_from_u64(5125120);

    let n = 512;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Random n=512 MSM matches");
                true
            } else {
                println!("  FAIL: Results differ");
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_identical_points() -> bool {
    println!("\n=== Test: Identical Points with Different Scalars ===");

    let mut rng = StdRng::seed_from_u64(1111110);

    let point = G1Projective::rand(&mut rng).into_affine();
    let n = 32;
    let points = vec![point; n];
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let cpu_result = cpu_msm(&points, &scalars);

    // Verify: sum of scalars * point
    let scalar_sum: Fr = scalars.iter().sum();
    let expected = (point.into_group() * scalar_sum).into_affine();
    if cpu_result.into_affine() != expected {
        println!("  FAIL: CPU result doesn't match expected scalar sum");
        return false;
    }

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Identical points handled correctly");
                true
            } else {
                println!("  FAIL: Results differ");
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_sequential_partial_sums_to_zero() -> bool {
    println!("\n=== Test: Sequential Accumulation Hitting Zero ===");

    // This tests the edge case where during serial accumulation,
    // the running sum temporarily becomes zero before continuing.
    // Pattern: G, G, -2G, G, G, -2G, G = G (intermediate zeros at positions 3 and 6)

    let generator = G1Affine::generator();

    let points = vec![generator; 7];
    let scalars = vec![
        Fr::from(1u64),
        Fr::from(1u64),
        -Fr::from(2u64),  // Sum is now 0
        Fr::from(1u64),
        Fr::from(1u64),
        -Fr::from(2u64),  // Sum is now 0 again
        Fr::from(1u64),   // Final sum is 1
    ];

    let cpu_result = cpu_msm(&points, &scalars);

    // Expected: 1*G
    let expected = generator.into_group();
    if cpu_result.into_affine() != expected.into_affine() {
        println!("  FAIL: CPU result unexpected (expected 1*G)");
        return false;
    }

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Sequential accumulation through zero works");
                true
            } else {
                println!("  FAIL: Results differ");
                println!("  Expected: G");
                println!("  CPU: {:?}", cpu_result.into_affine());
                println!("  GPU: {:?}", gpu_result.into_affine());
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_groth16_like_pattern() -> bool {
    println!("\n=== Test: Groth16-like MSM Pattern ===");

    // Simulate a pattern similar to what Groth16 proving would use:
    // - Mix of random points (witness elements)
    // - Mix of structured points (generator multiples for public inputs)
    // - Various scalar magnitudes

    let mut rng = StdRng::seed_from_u64(1616160);
    let generator = G1Affine::generator();

    let n = 64;
    let mut points: Vec<G1Affine> = Vec::with_capacity(n);
    let mut scalars: Vec<Fr> = Vec::with_capacity(n);

    // First few: generator multiples (like gamma_abc_g1)
    for i in 0..4 {
        points.push((generator.into_group() * Fr::from((i + 1) as u64)).into_affine());
        scalars.push(Fr::rand(&mut rng));
    }

    // Rest: random points (like witness commitments)
    for _ in 4..n {
        points.push(G1Projective::rand(&mut rng).into_affine());
        scalars.push(Fr::rand(&mut rng));
    }

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Groth16-like pattern matches");
                true
            } else {
                println!("  FAIL: Results differ");
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_random_1024() -> bool {
    println!("\n=== Test: Random Very Large MSM (n=1024) Consistency ===");

    let mut rng = StdRng::seed_from_u64(102410240);

    let n = 1024;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Random n=1024 MSM matches");
                true
            } else {
                println!("  FAIL: Results differ");
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_max_u64_scalars() -> bool {
    println!("\n=== Test: Max u64 Scalar Values Consistency ===");

    let mut rng = StdRng::seed_from_u64(646464640);

    let n = 16;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    // Use u64::MAX as scalar (large but valid scalar value)
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::from(u64::MAX)).collect();

    let cpu_result = cpu_msm(&points, &scalars);

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    match gpu_msm(&points, &scalars) {
        Ok(gpu_result) => {
            if points_equal(&gpu_result, &cpu_result) {
                println!("  PASS: Max u64 scalar values match");
                true
            } else {
                println!("  FAIL: Results differ");
                false
            }
        }
        Err(code) => {
            println!("  FAIL: GPU error {}", code);
            false
        }
    }
}

fn test_consistency_deterministic() -> bool {
    println!("\n=== Test: Deterministic Behavior ===");

    let mut rng = StdRng::seed_from_u64(987650);

    let n = 64;
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    // Run GPU MSM twice with same inputs
    let points1 = ManuallyDrop::new(points.clone());
    let scalars1 = ManuallyDrop::new(scalars.clone());
    let result1 = gpu_msm(&points1, &scalars1);

    let points2 = ManuallyDrop::new(points.clone());
    let scalars2 = ManuallyDrop::new(scalars.clone());
    let result2 = gpu_msm(&points2, &scalars2);

    match (result1, result2) {
        (Ok(r1), Ok(r2)) => {
            if points_equal(&r1, &r2) {
                println!("  PASS: GPU MSM is deterministic");
                true
            } else {
                println!("  FAIL: Same inputs produced different results");
                false
            }
        }
        _ => {
            println!("  FAIL: GPU MSM failed");
            false
        }
    }
}

// ============================================================================
// Main Test Runner (Custom Harness)
// ============================================================================

fn main() {
    println!("========================================================");
    println!("  MNT6-298 CPU vs GPU Consistency Tests (CGBN Kernel)");
    println!("========================================================");
    println!("  Validating GPU results match arkworks CPU implementation");
    println!("");

    let mut passed = 0;
    let mut failed = 0;

    let tests: Vec<(&str, fn() -> bool)> = vec![
        // Basic tests
        ("generator_simple", test_consistency_generator_simple),
        ("single_point", test_consistency_single_point),
        ("empty", test_consistency_empty),
        ("zero_scalars", test_consistency_zero_scalars),
        ("identity_points", test_consistency_identity_points),
        // Cancellation tests (critical for correctness)
        ("cancellation", test_consistency_cancellation),
        ("multiple_cancellations", test_consistency_multiple_cancellations),
        ("sequential_partial_sums_to_zero", test_consistency_sequential_partial_sums_to_zero),
        // Random MSM tests (various sizes)
        ("random_small_n4", test_consistency_random_small),
        ("random_medium_n16", test_consistency_random_medium),
        ("random_larger_n64", test_consistency_random_larger),
        ("random_256", test_consistency_random_256),
        ("random_512", test_consistency_random_512),
        ("random_1024", test_consistency_random_1024),
        // Edge cases
        ("mixed_edge_cases", test_consistency_mixed_edge_cases),
        ("large_scalars", test_consistency_large_scalars),
        ("negative_scalars", test_consistency_negative_scalars),
        ("identical_points", test_consistency_identical_points),
        ("max_u64_scalars", test_consistency_max_u64_scalars),
        // Behavior tests
        ("deterministic", test_consistency_deterministic),
        // Groth16 pattern test
        ("groth16_like_pattern", test_consistency_groth16_like_pattern),
    ];

    for (name, test_fn) in &tests {
        if test_fn() {
            passed += 1;
        } else {
            failed += 1;
            eprintln!("\n  FAILED: {}", name);
        }
    }

    println!("\n========================================================");
    println!("  Test Summary");
    println!("========================================================");
    println!("  Total:  {}", tests.len());
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);

    if failed == 0 {
        println!("\n  ALL TESTS PASSED - MNT6-298 GPU matches CPU!");
        println!("  Ready for production use in provers.");
        std::process::exit(0);
    } else {
        println!("\n  SOME TESTS FAILED - GPU/CPU mismatch detected");
        std::process::exit(1);
    }
}
