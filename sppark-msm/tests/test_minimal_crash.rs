#![cfg(all(feature = "gpu", bw6_cgbn_available))]

use ark_bw6_761::{Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::{BigInt, PrimeField};
use sppark_msm::msm_bw6_761_gpu_cgbn;
use std::mem::ManuallyDrop;

#[test]
fn test_minimal_segfault_isolation() {
    println!("Step 1: Creating data");
    let generator = G1Affine::generator();
    let points = vec![generator, generator];

    // Create scalars explicitly to control lifetime
    let fr1 = Fr::from(2u64);
    let fr2 = Fr::from(3u64);
    let scalar1 = fr1.into_bigint();
    let scalar2 = fr2.into_bigint();
    let scalars = vec![scalar1, scalar2];

    // Prevent Drop of Fr and BigInt temporaries
    let fr1 = ManuallyDrop::new(fr1);
    let fr2 = ManuallyDrop::new(fr2);
    let scalar1 = ManuallyDrop::new(scalar1);
    let scalar2 = ManuallyDrop::new(scalar2);
    
    println!("Step 2: Wrapping in ManuallyDrop");
    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);
    
    println!("Step 3: Calling kernel");
    let result = msm_bw6_761_gpu_cgbn(&points, &scalars);
    
    println!("Step 4: Kernel returned");
    
    println!("Step 5: Wrapping result in ManuallyDrop");
    let result = ManuallyDrop::new(result);
    
    println!("Step 6: Matching result (by reference)");
    match &*result {
        Ok(_) => println!("Step 7: Match OK branch"),
        Err(_) => println!("Step 7: Match Err branch"),
    }
    
    println!("Step 8: After match block");
    
    println!("Step 9: Before cudaDeviceReset");
    #[link(name = "cudart")]
    extern "C" {
        fn cudaDeviceReset() -> i32;
    }
    unsafe {
        let err = cudaDeviceReset();
        println!("Step 10: cudaDeviceReset returned: {}", err);
    }
    
    println!("Step 11: Test function ending");
}
