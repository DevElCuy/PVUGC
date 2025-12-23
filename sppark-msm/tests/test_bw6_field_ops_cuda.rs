#![cfg(feature = "gpu")]

//! Direct CUDA test of BW6-761 field operations
//!
//! This test validates basic field arithmetic on the GPU
//! to identify if the hang is in field operations themselves
//! or in the MSM algorithm.

use sppark_msm::test_bw6_761_field_operations;

#[test]
fn test_bw6_761_basic_field_ops() {
    eprintln!("\n=== BW6-761 Field Operations CUDA Test ===");
    eprintln!("This test runs basic field operations (add, sub, mul, sqr) on GPU");
    eprintln!("to identify where the hang occurs.\n");

    match test_bw6_761_field_operations() {
        Ok(()) => {
            eprintln!("✅ All field operations completed successfully!");
        }
        Err(()) => {
            panic!("❌ Field operations test failed or hung");
        }
    }
}
