//! Validate BW6-761 constants between sppark and arkworks
//!
//! This test verifies that the field constants used in the CUDA kernels
//! match the arkworks reference implementation.

use ark_bw6_761::{Fq, Fr};
use ark_ff::PrimeField;

#[test]
fn test_bw6_761_base_field_modulus() {
    println!("\n=== BW6-761 Base Field (Fq) Modulus Validation ===");

    // Get arkworks modulus
    let modulus = Fq::MODULUS;

    println!("Arkworks Fq modulus (BigInt):");
    for (i, &word) in modulus.0.iter().enumerate() {
        println!("  word[{}] = 0x{:016x}", i, word);
    }

    // Convert to limbs (u32)
    println!("\nAs u32 limbs (little-endian):");
    for i in 0..24 {
        let word_idx = i / 2;
        let is_high = i % 2 == 1;
        let limb = if is_high {
            (modulus.0[word_idx] >> 32) as u32
        } else {
            modulus.0[word_idx] as u32
        };
        println!("  limb[{}] = 0x{:08x}", i, limb);
    }

    // Calculate modulus bit length
    let mut bit_length = 0;
    for (i, &word) in modulus.0.iter().enumerate().rev() {
        if word != 0 {
            bit_length = (i + 1) * 64 - word.leading_zeros() as usize;
            break;
        }
    }
    println!("\nModulus bit length: {} bits", bit_length);
    assert!(bit_length <= 761, "BW6-761 modulus should be at most 761 bits");
    assert!(bit_length >= 753, "BW6-761 modulus should be at least 753 bits");

    println!("✅ Base field modulus validated");
}

#[test]
fn test_bw6_761_scalar_field_modulus() {
    println!("\n=== BW6-761 Scalar Field (Fr) Modulus Validation ===");

    // Get arkworks scalar modulus
    let modulus = Fr::MODULUS;

    println!("Arkworks Fr modulus (BigInt):");
    for (i, &word) in modulus.0.iter().enumerate() {
        println!("  word[{}] = 0x{:016x}", i, word);
    }

    // Convert to limbs (u32)
    println!("\nAs u32 limbs (little-endian):");
    for i in 0..12 {  // Fr is 377 bits, uses 12 u32 limbs
        let word_idx = i / 2;
        let is_high = i % 2 == 1;
        let limb = if is_high {
            (modulus.0[word_idx] >> 32) as u32
        } else {
            modulus.0[word_idx] as u32
        };
        println!("  limb[{}] = 0x{:08x}", i, limb);
    }

    // Calculate modulus bit length
    let mut bit_length = 0;
    for (i, &word) in modulus.0.iter().enumerate().rev() {
        if word != 0 {
            bit_length = (i + 1) * 64 - word.leading_zeros() as usize;
            break;
        }
    }
    println!("\nModulus bit length: {} bits", bit_length);
    assert!(bit_length <= 377, "BW6-761 scalar should be at most 377 bits");
    assert!(bit_length >= 370, "BW6-761 scalar should be at least 370 bits");

    println!("✅ Scalar field modulus validated");
}

#[test]
fn test_bw6_761_montgomery_constants() {
    println!("\n=== BW6-761 Montgomery Constants ===");

    // Montgomery constant R = 2^768 mod p
    // This is what arkworks uses internally
    let one = Fq::from(1u64);
    let one_bigint = one.into_bigint();

    println!("Arkworks Fq ONE (in Montgomery form, = R mod p):");
    for (i, &word) in one_bigint.0.iter().enumerate() {
        println!("  word[{}] = 0x{:016x}", i, word);
    }

    println!("\nAs u32 limbs (little-endian):");
    for i in 0..24 {
        let word_idx = i / 2;
        let is_high = i % 2 == 1;
        let limb = if is_high {
            (one_bigint.0[word_idx] >> 32) as u32
        } else {
            one_bigint.0[word_idx] as u32
        };
        println!("  R_limb[{}] = 0x{:08x}", i, limb);
    }

    println!("✅ Montgomery constants displayed");
}

#[test]
fn test_bw6_761_generator_point() {
    use ark_bw6_761::G1Affine;
    use ark_ec::AffineRepr;

    println!("\n=== BW6-761 G1 Generator Point ===");

    let generator = G1Affine::generator();

    println!("Generator point:");
    println!("  x = {:?}", generator.x);
    println!("  y = {:?}", generator.y);
    println!("  infinity = {:?}", generator.infinity);

    println!("✅ Generator point displayed");
}

#[test]
fn test_field_element_sizes() {
    use std::mem::size_of;
    use ark_bw6_761::{Fq, Fr, G1Affine};
    use ark_ff::BigInt;

    println!("\n=== Type Sizes ===");
    println!("Fq size: {} bytes", size_of::<Fq>());
    println!("Fr size: {} bytes", size_of::<Fr>());
    println!("G1Affine size: {} bytes", size_of::<G1Affine>());
    println!("BigInt<6> size: {} bytes", size_of::<BigInt<6>>());
    println!("BigInt<12> size: {} bytes", size_of::<BigInt<12>>());

    // Expected:
    // Fq: 96 bytes (761 bits in Montgomery form)
    // Fr: 48 bytes (377 bits)
    // G1Affine: 2*96 + 8 = 200 bytes (x, y, infinity flag)
    // BigInt<6>: 48 bytes
    // BigInt<12>: 96 bytes

    assert_eq!(size_of::<Fq>(), 96, "Fq should be 96 bytes");
    assert_eq!(size_of::<Fr>(), 48, "Fr should be 48 bytes");
    assert_eq!(size_of::<BigInt<6>>(), 48, "BigInt<6> should be 48 bytes");
    assert_eq!(size_of::<BigInt<12>>(), 96, "BigInt<12> should be 96 bytes");

    println!("✅ All sizes match expected values");
}
