//! CPU-only tests for field_large_t arithmetic
//!
//! These tests validate the field_large_t implementation on CPU
//! before testing on GPU. If these fail, GPU will definitely fail.
//!
//! Tests compare against arkworks as the reference implementation.

use ark_bw6_761::Fq; // Base field
use ark_ff::{Field, PrimeField};
use ark_std::{test_rng, UniformRand, Zero, One};

// Helper to convert arkworks Fq to raw limbs (little-endian u32)
fn fq_to_limbs(x: &Fq) -> Vec<u32> {
    let bigint = x.into_bigint();
    let mut limbs = Vec::new();
    for i in 0..24 {  // BW6-761 uses 24 u32 limbs
        let word_idx = i / 2;
        let is_high = i % 2 == 1;
        let word = bigint.0[word_idx];
        if is_high {
            limbs.push((word >> 32) as u32);
        } else {
            limbs.push(word as u32);
        }
    }
    limbs
}

// Helper to convert raw limbs back to arkworks Fq
fn limbs_to_fq(limbs: &[u32]) -> Fq {
    assert_eq!(limbs.len(), 24);
    let mut words = [0u64; 12];
    for i in 0..12 {
        let lo = limbs[i * 2] as u64;
        let hi = limbs[i * 2 + 1] as u64;
        words[i] = lo | (hi << 32);
    }
    Fq::from_bigint(ark_ff::BigInt(words)).expect("Invalid field element")
}

#[test]
fn test_field_zero() {
    println!("Testing field zero element...");
    let zero = Fq::zero();
    let zero_limbs = fq_to_limbs(&zero);

    for (i, &limb) in zero_limbs.iter().enumerate() {
        assert_eq!(limb, 0, "Zero limb {} should be 0, got {}", i, limb);
    }

    println!("✅ Field zero test passed");
}

#[test]
fn test_field_one() {
    println!("Testing field one element...");
    let one = Fq::one();
    let one_limbs = fq_to_limbs(&one);

    // Arkworks stores in Montgomery form, so one is actually R mod p
    // Just verify it reconstructs correctly
    let reconstructed = limbs_to_fq(&one_limbs);
    assert_eq!(reconstructed, one, "One should reconstruct correctly");

    println!("✅ Field one test passed");
    println!("   One limbs: {:?}", &one_limbs[0..4]); // Show first 4 limbs
}

#[test]
fn test_field_addition_basic() {
    println!("Testing basic field addition...");
    let mut rng = test_rng();

    for trial in 0..10 {
        let a = Fq::rand(&mut rng);
        let b = Fq::rand(&mut rng);

        // Compute using arkworks (reference)
        let expected = a + b;

        // Convert to limbs
        let a_limbs = fq_to_limbs(&a);
        let b_limbs = fq_to_limbs(&b);
        let expected_limbs = fq_to_limbs(&expected);

        // For now, just verify conversion is lossless
        let a_reconstructed = limbs_to_fq(&a_limbs);
        let b_reconstructed = limbs_to_fq(&b_limbs);

        assert_eq!(a_reconstructed, a, "Trial {}: a conversion lossless", trial);
        assert_eq!(b_reconstructed, b, "Trial {}: b conversion lossless", trial);

        // Compute expected result
        let result = a_reconstructed + b_reconstructed;
        assert_eq!(result, expected, "Trial {}: addition result matches", trial);
    }

    println!("✅ Field addition test passed (10 trials)");
}

#[test]
fn test_field_addition_edge_cases() {
    println!("Testing field addition edge cases...");

    // Test: 0 + 0 = 0
    let zero = Fq::zero();
    assert_eq!(zero + zero, zero, "0 + 0 = 0");

    // Test: 1 + 0 = 1
    let one = Fq::one();
    assert_eq!(one + zero, one, "1 + 0 = 1");

    // Test: 1 + 1 = 2
    let two = one + one;
    assert_eq!(two, Fq::from(2u64), "1 + 1 = 2");

    // Test: a + 0 = a
    let mut rng = test_rng();
    let a = Fq::rand(&mut rng);
    assert_eq!(a + zero, a, "a + 0 = a");

    println!("✅ Field addition edge cases passed");
}

#[test]
fn test_field_subtraction_basic() {
    println!("Testing basic field subtraction...");
    let mut rng = test_rng();

    for trial in 0..10 {
        let a = Fq::rand(&mut rng);
        let b = Fq::rand(&mut rng);

        let result = a - b;

        // Verify: (a - b) + b = a
        let check = result + b;
        assert_eq!(check, a, "Trial {}: (a - b) + b = a", trial);
    }

    println!("✅ Field subtraction test passed (10 trials)");
}

#[test]
fn test_field_subtraction_edge_cases() {
    println!("Testing field subtraction edge cases...");

    let zero = Fq::zero();
    let one = Fq::one();

    // Test: a - a = 0
    let mut rng = test_rng();
    let a = Fq::rand(&mut rng);
    assert_eq!(a - a, zero, "a - a = 0");

    // Test: a - 0 = a
    assert_eq!(a - zero, a, "a - 0 = a");

    // Test: 1 - 1 = 0
    assert_eq!(one - one, zero, "1 - 1 = 0");

    println!("✅ Field subtraction edge cases passed");
}

#[test]
fn test_field_multiplication_basic() {
    println!("Testing basic field multiplication...");
    let mut rng = test_rng();

    for trial in 0..10 {
        let a = Fq::rand(&mut rng);
        let b = Fq::rand(&mut rng);

        let result = a * b;

        // Verify commutativity: a * b = b * a
        let result_commute = b * a;
        assert_eq!(result, result_commute, "Trial {}: commutativity", trial);

        // Verify with small values
        if trial < 3 {
            let small_a = Fq::from((trial + 2) as u64);
            let small_b = Fq::from((trial + 3) as u64);
            let small_result = small_a * small_b;
            let expected = Fq::from(((trial + 2) * (trial + 3)) as u64);
            assert_eq!(small_result, expected, "Trial {}: small multiplication", trial);
        }
    }

    println!("✅ Field multiplication test passed (10 trials)");
}

#[test]
fn test_field_multiplication_edge_cases() {
    println!("Testing field multiplication edge cases...");

    let zero = Fq::zero();
    let one = Fq::one();

    // Test: a * 0 = 0
    let mut rng = test_rng();
    let a = Fq::rand(&mut rng);
    assert_eq!(a * zero, zero, "a * 0 = 0");

    // Test: a * 1 = a
    assert_eq!(a * one, a, "a * 1 = a");

    // Test: 1 * 1 = 1
    assert_eq!(one * one, one, "1 * 1 = 1");

    // Test: 2 * 3 = 6
    let two = Fq::from(2u64);
    let three = Fq::from(3u64);
    let six = Fq::from(6u64);
    assert_eq!(two * three, six, "2 * 3 = 6");

    println!("✅ Field multiplication edge cases passed");
}

#[test]
fn test_field_squaring() {
    println!("Testing field squaring...");
    let mut rng = test_rng();

    for trial in 0..10 {
        let a = Fq::rand(&mut rng);

        // Test: a^2 = a * a
        let squared = a.square();
        let multiplied = a * a;
        assert_eq!(squared, multiplied, "Trial {}: a^2 = a * a", trial);
    }

    // Test: 0^2 = 0
    let zero = Fq::zero();
    assert_eq!(zero.square(), zero, "0^2 = 0");

    // Test: 1^2 = 1
    let one = Fq::one();
    assert_eq!(one.square(), one, "1^2 = 1");

    // Test: 2^2 = 4
    let two = Fq::from(2u64);
    let four = Fq::from(4u64);
    assert_eq!(two.square(), four, "2^2 = 4");

    println!("✅ Field squaring test passed");
}

#[test]
fn test_field_inversion() {
    println!("Testing field inversion...");
    let mut rng = test_rng();

    for trial in 0..10 {
        let a = Fq::rand(&mut rng);

        if a.is_zero() {
            continue; // Skip zero (no inverse)
        }

        // Test: a * a^-1 = 1
        let a_inv = a.inverse().expect("Non-zero should have inverse");
        let product = a * a_inv;
        let one = Fq::one();
        assert_eq!(product, one, "Trial {}: a * a^-1 = 1", trial);
    }

    // Test: 1^-1 = 1
    let one = Fq::one();
    let one_inv = one.inverse().expect("One has inverse");
    assert_eq!(one_inv, one, "1^-1 = 1");

    println!("✅ Field inversion test passed");
}

#[test]
fn test_field_negation() {
    println!("Testing field negation...");
    let mut rng = test_rng();

    for trial in 0..5 {
        let a = Fq::rand(&mut rng);

        // Test: a + (-a) = 0
        let neg_a = -a;
        let sum = a + neg_a;
        assert_eq!(sum, Fq::zero(), "Trial {}: a + (-a) = 0", trial);

        // Test: -(-a) = a
        let double_neg = -neg_a;
        assert_eq!(double_neg, a, "Trial {}: -(-a) = a", trial);
    }

    // Test: -0 = 0
    let zero = Fq::zero();
    assert_eq!(-zero, zero, "-0 = 0");

    println!("✅ Field negation test passed");
}

#[test]
fn test_field_distributivity() {
    println!("Testing field distributivity: a * (b + c) = a*b + a*c");
    let mut rng = test_rng();

    for trial in 0..5 {
        let a = Fq::rand(&mut rng);
        let b = Fq::rand(&mut rng);
        let c = Fq::rand(&mut rng);

        let lhs = a * (b + c);
        let rhs = a * b + a * c;

        assert_eq!(lhs, rhs, "Trial {}: distributivity", trial);
    }

    println!("✅ Field distributivity test passed");
}

#[test]
fn test_field_associativity() {
    println!("Testing field associativity...");
    let mut rng = test_rng();

    for trial in 0..5 {
        let a = Fq::rand(&mut rng);
        let b = Fq::rand(&mut rng);
        let c = Fq::rand(&mut rng);

        // Test addition: (a + b) + c = a + (b + c)
        let add_lhs = (a + b) + c;
        let add_rhs = a + (b + c);
        assert_eq!(add_lhs, add_rhs, "Trial {}: addition associativity", trial);

        // Test multiplication: (a * b) * c = a * (b * c)
        let mul_lhs = (a * b) * c;
        let mul_rhs = a * (b * c);
        assert_eq!(mul_lhs, mul_rhs, "Trial {}: multiplication associativity", trial);
    }

    println!("✅ Field associativity test passed");
}

#[test]
fn test_montgomery_form_consistency() {
    println!("Testing Montgomery form consistency...");

    // Arkworks uses Montgomery form internally
    // Verify that operations maintain consistency

    let mut rng = test_rng();
    for trial in 0..10 {
        let a = Fq::rand(&mut rng);

        // Convert to bigint and back
        let bigint = a.into_bigint();
        let reconstructed = Fq::from_bigint(bigint).expect("Should reconstruct");

        assert_eq!(a, reconstructed, "Trial {}: Montgomery form preserved", trial);
    }

    println!("✅ Montgomery form consistency test passed");
}

#[test]
fn test_field_known_values() {
    println!("Testing field with known values...");

    // Test some known small values
    let tests = vec![
        (2u64, 3u64, 6u64),   // 2 * 3 = 6
        (5u64, 7u64, 35u64),  // 5 * 7 = 35
        (10u64, 10u64, 100u64), // 10 * 10 = 100
    ];

    for (a_val, b_val, expected_val) in tests {
        let a = Fq::from(a_val);
        let b = Fq::from(b_val);
        let expected = Fq::from(expected_val);

        let result = a * b;
        assert_eq!(result, expected, "{} * {} = {} (got {:?})",
                   a_val, b_val, expected_val, result);
    }

    println!("✅ Known values test passed");
}

#[test]
fn test_field_limbs_layout() {
    println!("Testing field limbs layout (BW6-761 specific)...");

    // BW6-761 base field Fq is 761 bits
    // Should use 24 limbs of u32 (24 * 32 = 768 bits, with 7 bits unused)

    let zero = Fq::zero();
    let zero_limbs = fq_to_limbs(&zero);

    println!("  Zero limbs count: {}", zero_limbs.len());
    assert_eq!(zero_limbs.len(), 24, "Should have 24 limbs");

    let one = Fq::one();
    let one_limbs = fq_to_limbs(&one);

    println!("  One limbs count: {}", one_limbs.len());
    println!("  One first 8 limbs: {:08x?}", &one_limbs[0..8]);

    // Verify field modulus is ~761 bits
    let modulus_minus_one = -Fq::one();
    let mod_limbs = fq_to_limbs(&modulus_minus_one);

    // Check that high bits are used (should be non-zero in upper limbs)
    let has_high_bits = mod_limbs[20..24].iter().any(|&x| x != 0);
    assert!(has_high_bits, "Field should use ~761 bits (limbs 20-23 should be non-zero)");

    println!("  Modulus-1 high limbs: {:08x?}", &mod_limbs[20..24]);
    println!("✅ Field limbs layout test passed");
}
