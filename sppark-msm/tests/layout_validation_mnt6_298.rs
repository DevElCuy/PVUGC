// MNT6-298 Layout Validation Tests
// Custom test harness (harness = false) to avoid Rust test framework CUDA cleanup conflicts
//
// Run with: cargo test --release --features gpu --test layout_validation_mnt6_298

#![cfg(all(feature = "gpu", mnt6_cgbn_available))]

use ark_mnt6_298::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInt, PrimeField};
use std::mem::ManuallyDrop;

fn call_msm(
    points: &[G1Affine],
    scalars: &[BigInt<5>],
) -> Result<G1Projective, ()> {
    use sppark_msm::msm_mnt6_298_gpu_cgbn;
    msm_mnt6_298_gpu_cgbn(points, scalars).map_err(|_| ())
}

fn test_g1affine_layout_with_generator() -> bool {
    println!("\n=== Test: G1Affine Layout with Generator ===");

    let generator = G1Affine::generator();
    let scalar_one = Fr::from(1u64).into_bigint();

    let points = ManuallyDrop::new(vec![generator]);
    let scalars = ManuallyDrop::new(vec![scalar_one]);

    let result = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ GPU MSM failed");
            return false;
        }
    };

    let result_affine = result.into_affine();

    if result_affine != generator {
        println!("  ❌ Generator MSM failed - CUDA may be reading fields in wrong order");
        return false;
    }

    println!("  ✅ PASS - G1Affine layout validated");
    true
}

fn test_bigint_layout_with_known_scalars() -> bool {
    println!("\n=== Test: BigInt<5> Layout with Known Scalars ===");

    let generator = G1Affine::generator();
    let scalar_2 = Fr::from(2u64).into_bigint();
    let points = ManuallyDrop::new(vec![generator]);
    let scalars = ManuallyDrop::new(vec![scalar_2]);

    let result_2 = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ GPU MSM failed for scalar=2");
            return false;
        }
    };

    let expected_2 = (generator * Fr::from(2u64)).into_affine();

    if result_2.into_affine() != expected_2 {
        println!("  ❌ Scalar=2 MSM failed - BigInt<5> limb ordering issue");
        return false;
    }

    println!("  ✅ PASS - BigInt<5> layout validated");
    true
}

fn test_infinity_flag_layout() -> bool {
    println!("\n=== Test: Infinity Flag Layout ===");

    let identity = G1Affine::identity();
    let generator = G1Affine::generator();

    let points = ManuallyDrop::new(vec![identity, generator, identity]);
    let scalars = ManuallyDrop::new(vec![
        Fr::from(5u64).into_bigint(),
        Fr::from(3u64).into_bigint(),
        Fr::from(7u64).into_bigint(),
    ]);

    let result = match call_msm(&points, &scalars) {
        Ok(r) => r,
        Err(_) => {
            println!("  ❌ Infinity flag test MSM failed");
            return false;
        }
    };

    let expected = (generator * Fr::from(3u64)).into_affine();

    if result.into_affine() != expected {
        println!("  ❌ Identity point MSM failed - infinity flag may be misread");
        return false;
    }

    println!("  ✅ PASS - Infinity flag layout validated");
    true
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║  MNT6-298 Layout Validation Tests (Custom Harness)   ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    let mut passed = 0;
    let mut failed = 0;

    let tests: Vec<(&str, fn() -> bool)> = vec![
        ("test_g1affine_layout_with_generator", test_g1affine_layout_with_generator),
        ("test_bigint_layout_with_known_scalars", test_bigint_layout_with_known_scalars),
        ("test_infinity_flag_layout", test_infinity_flag_layout),
    ];

    for (name, test_fn) in tests {
        if test_fn() {
            passed += 1;
        } else {
            failed += 1;
            eprintln!("\n❌ Test failed: {}", name);
        }
    }

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Test Summary                                         ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);

    if failed == 0 {
        println!("\n  ✅ ALL TESTS PASSED");
        std::process::exit(0);
    } else {
        println!("\n  ❌ SOME TESTS FAILED");
        std::process::exit(1);
    }
}
