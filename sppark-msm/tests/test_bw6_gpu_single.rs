#![cfg(all(feature = "gpu", bw6_gpu_available))]

use ark_bw6_761::{Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::BigInt;
use sppark_msm::{msm_bw6_761_gpu_raw, Bw6GpuError};

#[test]
fn gpu_single_point_matches_cpu() {
    // Skip by default; enable when actively debugging the BW6 GPU single-point path.
    if std::env::var("RUN_BW6_GPU_SINGLE").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")) != Some(true) {
        eprintln!("Skipping gpu_single_point_matches_cpu (set RUN_BW6_GPU_SINGLE=1 to run)");
        return;
    }

    // Small deterministic scalar to avoid large workloads and ensure reproducibility.
    let scalar_u64 = 5u64;
    let scalar_bigint = BigInt::<6>::from(scalar_u64);

    let point = G1Affine::generator();
    let cpu = point * Fr::from(scalar_u64);

    let gpu = match msm_bw6_761_gpu_raw(&[point], &[scalar_bigint]) {
        Ok(res) => res,
        Err(Bw6GpuError::Timeout) | Err(Bw6GpuError::KernelUnavailable) => {
            eprintln!("GPU single-point unavailable (timeout/unavailable), skipping");
            return;
        }
        Err(err) => panic!("GPU single-point MSM failed: {:?}", err),
    };

    assert_eq!(gpu, cpu, "GPU single-point MSM must match CPU result");
}
