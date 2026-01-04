// Test suite for MNT4-298 Pippenger MSM implementation
// Compares Pippenger GPU results against CPU reference and serial GPU

#![allow(unused_imports)]

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, PrimeField, UniformRand, Zero};
use ark_mnt4_298::{Fr, G1Affine, G1Projective};
use ark_std::rand::rngs::StdRng;
use ark_std::rand::SeedableRng;
use std::time::Instant;

#[cfg(feature = "gpu")]
use sppark_msm::{msm_mnt4_298_gpu_cgbn, msm_mnt4_298_gpu_pippenger};

fn generate_random_points_and_scalars(n: usize, seed: u64) -> (Vec<G1Affine>, Vec<BigInt<5>>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let scalars: Vec<BigInt<5>> = (0..n)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    (points, scalars)
}

fn cpu_msm(points: &[G1Affine], scalars: &[BigInt<5>]) -> G1Projective {
    G1Projective::msm_bigint(points, scalars)
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_two_points() {
    // Debug test with just 2 points
    let mut rng = StdRng::seed_from_u64(12345);
    let p1 = G1Projective::rand(&mut rng).into_affine();
    let p2 = G1Projective::rand(&mut rng).into_affine();
    let s1 = Fr::from(3u64).into_bigint();
    let s2 = Fr::from(5u64).into_bigint();

    let points = vec![p1, p2];
    let scalars = vec![s1, s2];

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for 2 points");

    println!("CPU result: {:?}", cpu_result.into_affine());
    println!("Pippenger result: {:?}", pippenger_result.into_affine());

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match CPU for 2 points"
    );

    println!("PASS: 2 points matches CPU");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_ten_points_small_scalars() {
    // Test 10 points with small scalars
    let mut rng = StdRng::seed_from_u64(12345);
    let points: Vec<G1Affine> = (0..10)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    // Small scalars that fit in wbits window
    let scalars: Vec<BigInt<5>> = (1..=10)
        .map(|i| Fr::from(i as u64).into_bigint())
        .collect();

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for 10 points");

    println!("CPU result: {:?}", cpu_result.into_affine());
    println!("Pippenger result: {:?}", pippenger_result.into_affine());

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match CPU for 10 points"
    );

    println!("PASS: 10 points matches CPU");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_ten_points_random_scalars() {
    // Test 10 points with random scalars
    let (points, scalars) = generate_random_points_and_scalars(10, 99999);

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for 10 random points");

    println!("CPU result: {:?}", cpu_result.into_affine());
    println!("Pippenger result: {:?}", pippenger_result.into_affine());

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match CPU for 10 random points"
    );

    println!("PASS: 10 random points matches CPU");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_sweep_sizes() {
    // Test various sizes to find failure threshold
    for n in [16, 20, 32, 50, 64] {
        let (points, scalars) = generate_random_points_and_scalars(n, 77777 + n as u64);
        let cpu_result = cpu_msm(&points, &scalars);
        let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
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
fn test_pippenger_64_all_ones() {
    // Test 64 points with all scalar=1
    let mut rng = StdRng::seed_from_u64(12345);
    let points: Vec<G1Affine> = (0..64)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<BigInt<5>> = vec![Fr::from(1u64).into_bigint(); 64];

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for 64 ones");

    println!("CPU result: {:?}", cpu_result.into_affine());
    println!("Pippenger result: {:?}", pippenger_result.into_affine());

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match CPU for 64 ones"
    );

    println!("PASS: 64 ones matches CPU");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_64_all_256() {
    // Test 64 points with all scalar=256 (tests window boundary)
    let mut rng = StdRng::seed_from_u64(12345);
    let points: Vec<G1Affine> = (0..64)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<BigInt<5>> = vec![Fr::from(256u64).into_bigint(); 64];

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for 64 x 256");

    println!("CPU result: {:?}", cpu_result.into_affine());
    println!("Pippenger result: {:?}", pippenger_result.into_affine());

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger doesn't match CPU for 64 x 256"
    );

    println!("PASS: 64 x 256 matches CPU");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_64_scalar_boundary() {
    // Test various boundary scalars with 64 points
    let mut rng = StdRng::seed_from_u64(12345);
    let points: Vec<G1Affine> = (0..64)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    for scalar_val in [127u64, 128, 129, 255, 256, 257, 512, 1000] {
        let scalars: Vec<BigInt<5>> = vec![Fr::from(scalar_val).into_bigint(); 64];

        let cpu_result = cpu_msm(&points, &scalars);
        let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
            .expect(&format!("Pippenger MSM failed for scalar={}", scalar_val));

        let cpu_aff = cpu_result.into_affine();
        let pip_aff = pippenger_result.into_affine();

        if cpu_aff == pip_aff {
            println!("scalar={}: PASS", scalar_val);
        } else {
            println!("scalar={}: FAIL", scalar_val);
            println!("  CPU: {:?}", cpu_aff);
            println!("  Pip: {:?}", pip_aff);
            panic!("Failed at scalar={}", scalar_val);
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_small() {
    // Small test (n=100) to verify correctness
    let n = 100;
    let (points, scalars) = generate_random_points_and_scalars(n, 12345);

    let cpu_result = cpu_msm(&points, &scalars);

    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed");

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger result doesn't match CPU for n={}", n
    );

    println!("PASS: Pippenger matches CPU for n={}", n);
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_vs_serial() {
    // Compare Pippenger against serial CGBN implementation
    let n = 256;
    let (points, scalars) = generate_random_points_and_scalars(n, 54321);

    let serial_result = msm_mnt4_298_gpu_cgbn(&points, &scalars)
        .expect("Serial CGBN MSM failed");

    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
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
fn test_pippenger_medium() {
    // Medium test (n=1000)
    let n = 1000;
    let (points, scalars) = generate_random_points_and_scalars(n, 99999);

    let cpu_result = cpu_msm(&points, &scalars);

    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed");

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Pippenger result doesn't match CPU for n={}", n
    );

    println!("PASS: Pippenger matches CPU for n={}", n);
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_empty() {
    let points: Vec<G1Affine> = vec![];
    let scalars: Vec<BigInt<5>> = vec![];

    let result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for empty input");

    assert!(
        result.is_zero(),
        "Empty MSM should return identity"
    );

    println!("PASS: Empty input returns identity");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_single_point() {
    let mut rng = StdRng::seed_from_u64(42);
    let point = G1Projective::rand(&mut rng).into_affine();
    let scalar = Fr::from(42u64).into_bigint();

    let cpu_result = G1Projective::msm_bigint(&[point], &[scalar]);

    let pippenger_result = msm_mnt4_298_gpu_pippenger(&[point], &[scalar])
        .expect("Pippenger MSM failed for single point");

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Single point MSM doesn't match"
    );

    println!("PASS: Single point matches CPU");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_single_point_scalar_127() {
    let mut rng = StdRng::seed_from_u64(42);
    let point = G1Projective::rand(&mut rng).into_affine();
    let scalar = Fr::from(127u64).into_bigint();

    let cpu_result = G1Projective::msm_bigint(&[point], &[scalar]);
    let pippenger_result = msm_mnt4_298_gpu_pippenger(&[point], &[scalar])
        .expect("Pippenger MSM failed for single point");

    println!("CPU result for 127*P: {:?}", cpu_result.into_affine());
    println!("Pip result for 127*P: {:?}", pippenger_result.into_affine());

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "Single point MSM doesn't match for scalar=127"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_two_points_scalar_127() {
    let mut rng = StdRng::seed_from_u64(42);
    let points: Vec<G1Affine> = (0..2)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<BigInt<5>> = vec![Fr::from(127u64).into_bigint(); 2];

    let cpu_result = cpu_msm(&points, &scalars);
    let pippenger_result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for 2 points scalar=127");

    println!("CPU result for 2x127*P: {:?}", cpu_result.into_affine());
    println!("Pip result for 2x127*P: {:?}", pippenger_result.into_affine());

    assert_eq!(
        cpu_result.into_affine(),
        pippenger_result.into_affine(),
        "2 point MSM doesn't match for scalar=127"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_sweep_n_scalar_127() {
    // Test various n values with identical scalars (triggers same-point addition)
    let mut rng = StdRng::seed_from_u64(42);
    let all_points: Vec<G1Affine> = (0..256)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    for n in [32, 64, 100, 128, 200, 256] {
        for scalar_val in [2u64, 127, 255, 1000] {
            let points = &all_points[0..n];
            let scalars: Vec<BigInt<5>> = vec![Fr::from(scalar_val).into_bigint(); n];

            let cpu_result = cpu_msm(points, &scalars);
            let pippenger_result = msm_mnt4_298_gpu_pippenger(points, &scalars)
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
#[cfg(feature = "gpu")]
fn test_pippenger_zero_scalars() {
    let mut rng = StdRng::seed_from_u64(42);
    let points: Vec<G1Affine> = (0..10)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<BigInt<5>> = vec![BigInt::<5>::from(0u64); 10];

    let result = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Pippenger MSM failed for zero scalars");

    assert!(
        result.is_zero(),
        "MSM with all zero scalars should return identity"
    );

    println!("PASS: All zero scalars returns identity");
}

#[test]
#[cfg(feature = "gpu")]
fn test_pippenger_deterministic() {
    let n = 500;
    let (points, scalars) = generate_random_points_and_scalars(n, 77777);

    let result1 = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("First Pippenger call failed");

    let result2 = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Second Pippenger call failed");

    let result3 = msm_mnt4_298_gpu_pippenger(&points, &scalars)
        .expect("Third Pippenger call failed");

    assert_eq!(
        result1.into_affine(),
        result2.into_affine(),
        "Pippenger not deterministic (run 1 vs 2)"
    );

    assert_eq!(
        result2.into_affine(),
        result3.into_affine(),
        "Pippenger not deterministic (run 2 vs 3)"
    );

    println!("PASS: Pippenger is deterministic");
}

// Benchmark test (run with --ignored)
#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn benchmark_pippenger_vs_serial() {
    println!("\n=== Pippenger vs Serial CGBN Benchmark ===\n");

    for &n in &[100, 256, 512, 1000, 2000, 4000] {
        let (points, scalars) = generate_random_points_and_scalars(n, 12345 + n as u64);

        // Warmup
        let _ = msm_mnt4_298_gpu_pippenger(&points, &scalars);

        // Benchmark Pippenger
        let start = Instant::now();
        for _ in 0..3 {
            let _ = msm_mnt4_298_gpu_pippenger(&points, &scalars);
        }
        let pippenger_time = start.elapsed() / 3;

        // Benchmark Serial (only for smaller sizes)
        let serial_time = if n <= 1000 {
            let start = Instant::now();
            for _ in 0..3 {
                let _ = msm_mnt4_298_gpu_cgbn(&points, &scalars);
            }
            Some(start.elapsed() / 3)
        } else {
            None
        };

        match serial_time {
            Some(serial) => {
                let speedup = serial.as_secs_f64() / pippenger_time.as_secs_f64();
                println!(
                    "n={:5}: Pippenger {:>8.2}ms, Serial {:>8.2}ms, Speedup: {:.2}x",
                    n,
                    pippenger_time.as_secs_f64() * 1000.0,
                    serial.as_secs_f64() * 1000.0,
                    speedup
                );
            }
            None => {
                println!(
                    "n={:5}: Pippenger {:>8.2}ms (serial too slow to benchmark)",
                    n,
                    pippenger_time.as_secs_f64() * 1000.0
                );
            }
        }
    }
}
