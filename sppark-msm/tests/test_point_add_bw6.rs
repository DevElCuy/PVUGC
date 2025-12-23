#![cfg(feature = "gpu")]

//! Test BW6-761 point addition on GPU
//! This test directly tests point addition without Pippenger complexity

use sppark_msm::test_bw6_761_point_addition;
use ark_bw6_761::G1Affine;
use ark_ec::{AffineRepr, CurveGroup};

#[test]
fn test_bw6_point_add_two_generators() {
    eprintln!("\n=== BW6-761 Point Addition GPU Test ===");
    eprintln!("Testing: G + G = 2*G");

    let generator = G1Affine::generator();
    let points = [generator, generator];

    eprintln!("Calling GPU point addition...");
    match test_bw6_761_point_addition(&points) {
        Ok(result) => {
            eprintln!("✅ GPU point addition returned successfully!");

            // Compute expected result on CPU
            let expected = (generator + generator).into_affine();

            eprintln!("Verifying result...");
            if result.into_affine() == expected {
                eprintln!("✅ Result matches CPU: Point addition is correct!");
            } else {
                panic!("❌ Result mismatch! GPU point addition gave wrong answer");
            }
        }
        Err(_) => {
            panic!("❌ GPU point addition failed or timed out");
        }
    }
}
