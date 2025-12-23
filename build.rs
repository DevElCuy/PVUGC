fn main() {
    // Declare the sppark_cuda_built cfg so it can be used in integration tests
    // This cfg is set by the sppark-msm dependency when CUDA builds successfully
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_built)");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_stub)");
}
