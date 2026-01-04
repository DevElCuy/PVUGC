fn main() {
    // Declare the sppark_cuda_built cfg so it can be used in integration tests
    // This cfg is set by the sppark-msm dependency when CUDA builds successfully
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_built)");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_stub)");

    // Declare and set sparse quotient kernel availability cfgs
    // These mirror what sppark-msm sets, allowing integration tests to use them
    println!("cargo::rustc-check-cfg=cfg(sparse_quotient_mnt4_available)");
    println!("cargo::rustc-check-cfg=cfg(sparse_quotient_mnt6_available)");

    // When GPU feature is enabled, assume kernels are available
    // (sppark-msm will have compiled them as a dependency)
    if std::env::var("CARGO_FEATURE_GPU").is_ok() {
        println!("cargo:rustc-cfg=sparse_quotient_mnt4_available");
        println!("cargo:rustc-cfg=sparse_quotient_mnt6_available");
    }
}
