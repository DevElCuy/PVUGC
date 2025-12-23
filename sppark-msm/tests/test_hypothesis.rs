// Test: Does accessing CUDA change cleanup behavior?
#![cfg(all(feature = "gpu", bw6_cgbn_available))]

use ark_bw6_761::{Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::{BigInt, PrimeField};
use sppark_msm::msm_bw6_761_gpu_cgbn;
use std::mem::ManuallyDrop;

#[test]
fn test_no_cuda_operations() {
    println!("=== Test: Result without any CUDA calls ===");
    
    // This test doesn't call the kernel at all
    let points: Vec<G1Affine> = vec![];
    let scalars: Vec<BigInt<6>> = vec![];
    
    let _points = ManuallyDrop::new(points);
    let _scalars = ManuallyDrop::new(scalars);

    // Don't even call the kernel
    println!("Test ending without calling kernel...");
}

#[test]
fn test_count_zero_calls_kernel() {
    println!("=== Test: count=0 (no CUDA operations internally) ===");
    
    let points: Vec<G1Affine> = vec![];
    let scalars: Vec<BigInt<6>> = vec![];
    
    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);
    
    // Call kernel with count=0 (returns immediately on CPU, no CUDA ops)
    let result = msm_bw6_761_gpu_cgbn(&points, &scalars);
    
    println!("Kernel returned: {:?}", result.is_ok());
    println!("Test ending...");
}

#[test]
fn test_count_two_calls_kernel() {
    println!("=== Test: count=2 (HAS CUDA operations) ===");

    let generator = G1Affine::generator();
    let points = vec![generator, generator];
    let scalars = vec![
        Fr::from(2u64).into_bigint(),
        Fr::from(3u64).into_bigint(),
    ];

    let points = ManuallyDrop::new(points);
    let scalars = ManuallyDrop::new(scalars);

    // Call kernel with count=2 (DOES allocate GPU memory, launch kernels)
    let result = msm_bw6_761_gpu_cgbn(&points, &scalars);

    println!("Kernel returned: {:?}", result.is_ok());

    // ADD: Explicit cudaDeviceReset
    #[link(name = "cudart")]
    extern "C" {
        fn cudaDeviceReset() -> i32;
    }
    unsafe {
        let err = cudaDeviceReset();
        println!("cudaDeviceReset returned: {}", err);
    }

    println!("Test ending...");
}
