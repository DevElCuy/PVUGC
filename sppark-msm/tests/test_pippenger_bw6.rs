// Test suite for BW6-761 Pippenger MSM implementation
// Compares Pippenger GPU results against CPU reference

#![allow(unused_imports)]

use ark_bw6_761::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, PrimeField, UniformRand, Zero};
use ark_std::rand::rngs::StdRng;
use ark_std::rand::SeedableRng;
use std::time::Instant;

#[cfg(feature = "gpu")]
use sppark_msm::{msm_bw6_761_gpu_cgbn, msm_bw6_761_gpu_pippenger};

fn generate_random_points_and_scalars(n: usize, seed: u64) -> (Vec<G1Affine>, Vec<BigInt<6>>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<6>> = (0..n)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    (points, scalars)
}

fn cpu_msm(points: &[G1Affine], scalars: &[BigInt<6>]) -> G1Projective {
    G1Projective::msm_bigint(points, scalars)
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_small() {
    let n = 100;
    let (points, scalars) = generate_random_points_and_scalars(n, 12345);

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed");

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match CPU for n={}", n
    );

    println!("PASS: Pippenger matches CPU for n={}", n);
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_medium() {
    let n = 1000;
    let (points, scalars) = generate_random_points_and_scalars(n, 54321);

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed");

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match CPU for n={}", n
    );

    println!("PASS: Pippenger matches CPU for n={}", n);
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_vs_serial() {
    let n = 256;
    let (points, scalars) = generate_random_points_and_scalars(n, 11111);

    let serial_result = msm_bw6_761_gpu_cgbn(&points, &scalars)
        .expect("Serial CGBN MSM failed");
    let pippenger_result = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed");

    assert_eq!(
        serial_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match serial CGBN for n={}", n
    );

    println!("PASS: Pippenger matches serial CGBN for n={}", n);
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_empty() {
    let points: Vec<G1Affine> = vec![];
    let scalars: Vec<BigInt<6>> = vec![];

    let result = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for empty input");

    assert!(
        result.is_zero(),
        "Empty input should return identity"
    );

    println!("PASS: Empty input returns identity");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_single_point() {
    let mut rng = StdRng::seed_from_u64(12345);
    let point = G1Projective::rand(&mut rng).into_affine();
    let scalar = Fr::from(5u64).into_bigint();

    let points = vec![point];
    let scalars = vec![scalar];

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for single point");

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match CPU for single point"
    );

    println!("PASS: Single point matches CPU");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_zero_scalars() {
    let n = 10;
    let mut rng = StdRng::seed_from_u64(12345);
    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<BigInt<6>> = vec![BigInt::<6>::zero(); n];

    let result = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for zero scalars");

    assert!(
        result.is_zero(),
        "All zero scalars should return identity"
    );

    println!("PASS: All zero scalars returns identity");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_deterministic() {
    let n = 100;
    let (points, scalars) = generate_random_points_and_scalars(n, 99999);

    let result1 = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed");
    let result2 = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed");
    let result3 = msm_bw6_761_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed");

    assert_eq!(result1, result2);
    assert_eq!(result2, result3);

    println!("PASS: Pippenger is deterministic");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_sweep_sizes() {
    for n in [16, 20, 32, 50, 64, 100, 128] {
        let (points, scalars) = generate_random_points_and_scalars(n, 77777 + n as u64);
        let cpu_result = cpu_msm(&points, &scalars);
        let pippenger_result = msm_bw6_761_gpu_pippenger(&points, &scalars)
            .expect(&format!("Pippenger MSM failed for n={}", n));

        let cpu_aff = cpu_result.into_affine();
        let pip_aff = pippenger_result.into_affine();

        if cpu_aff == pip_aff {
            println!("n={}: PASS", n);
        } else {
            println!("n={}: FAIL", n);
            println!("  CPU: {:?}", cpu_aff);
            println!("  Pip: {:?}", pip_aff);
            panic!("Failed at n={}", n);
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_bw6_identical_scalars() {
    // Test with identical scalars (triggers same-point addition)
    let mut rng = StdRng::seed_from_u64(42);
    let all_points: Vec<G1Affine> = (0..256)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    for n in [32, 64, 100, 128, 200, 256] {
        for scalar_val in [2u64, 127, 255, 1000] {
            let points = &all_points[0..n];
            let scalars: Vec<BigInt<6>> = vec![Fr::from(scalar_val).into_bigint(); n];

            let cpu_result = cpu_msm(points, &scalars);
            let pippenger_result = msm_bw6_761_gpu_pippenger(points, &scalars)
                .expect(&format!("Pippenger MSM failed for n={}", n));

            let cpu_aff = cpu_result.into_affine();
            let pip_aff = pippenger_result.into_affine();

            if cpu_aff == pip_aff {
                println!("n={}, scalar={}: PASS", n, scalar_val);
            } else {
                println!("n={}, scalar={}: FAIL", n, scalar_val);
                println!("  CPU: {:?}", cpu_aff);
                println!("  Pip: {:?}", pip_aff);
                panic!("Failed at n={}, scalar={}", n, scalar_val);
            }
        }
    }
}

#[test]
#[ignore = "benchmark test, run manually"]
#[cfg(feature = "gpu")]
fn benchmark_pippenger_vs_serial_bw6() {
    for n in [1000, 2000, 5000, 10000] {
        let (points, scalars) = generate_random_points_and_scalars(n, 12345);

        // Warmup
        let _ = msm_bw6_761_gpu_cgbn(&points, &scalars);
        let _ = msm_bw6_761_gpu_pippenger(&points, &scalars);

        // Benchmark serial
        let start = Instant::now();
        for _ in 0..5 {
            let _ = msm_bw6_761_gpu_cgbn(&points, &scalars);
        }
        let serial_time = start.elapsed() / 5;

        // Benchmark Pippenger
        let start = Instant::now();
        for _ in 0..5 {
            let _ = msm_bw6_761_gpu_pippenger(&points, &scalars);
        }
        let pippenger_time = start.elapsed() / 5;

        println!(
            "n={}: serial={:?}, pippenger={:?}, speedup={:.2}x",
            n,
            serial_time,
            pippenger_time,
            serial_time.as_secs_f64() / pippenger_time.as_secs_f64()
        );
    }
}
