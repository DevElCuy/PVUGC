fn main() {
    println!("cargo:rerun-if-changed=src/msm_bls12_377.cu");
    println!("cargo:rerun-if-changed=src/msm_bw6_761.cu");
    println!("cargo:rerun-if-changed=sppark");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_built)");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_stub)");

    if std::env::var("CARGO_FEATURE_GPU").is_err() {
        println!("cargo:warning=CUDA build skipped (feature \"gpu\" not enabled)");
        return;
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_include = format!("{}/include", cuda_home);
    let cuda_lib = format!("{}/lib64", cuda_home);

    // Build CUDA kernels
    let mut cuda_build = cc::Build::new();
    cuda_build
        .cuda(true)
        .flag("-std=c++17")
        .flag("-allow-unsupported-compiler")
        .flag("-arch=sm_75")  // Turing GPUs (GTX 1660, etc.)
        .flag("-Xcompiler")
        .flag("-O3")
        .flag("--maxrregcount=128")
        .include("sppark")
        .include("sppark/blst/src")
        .include(&cuda_include)
        .file("src/msm_bls12_377.cu");
        // BW6-761 temporarily disabled - requires further investigation
        // .file("src/msm_bw6_761.cu");

    // Build C++ utility files needed by sppark (needs CUDA headers)
    let mut cpp_build = cc::Build::new();
    cpp_build
        .cuda(true)  // Compile as CUDA to get CUDA headers
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-x")
        .flag("cu")  // Treat as CUDA file
        .include("sppark")
        .include("sppark/blst/src")
        .include(&cuda_include)
        .file("sppark/util/all_gpus.cpp");

    // Compile blst's main assembly file which provides all operations
    let mut blst_build = cc::Build::new();
    blst_build
        .flag("-O3")
        .include("sppark/blst/src")
        .file("sppark/blst/build/assembly.S");

    match (cuda_build.try_compile("msm"), cpp_build.try_compile("sppark_util"), blst_build.try_compile("blst")) {
        (Ok(_), Ok(_), Ok(_)) => {
            println!("cargo:rustc-link-search=native={}", cuda_lib);
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-cfg=sppark_cuda_built");
        }
        (Err(err), _, _) | (_, Err(err), _) | (_, _, Err(err)) => {
            println!(
                "cargo:warning=Build disabled: {}",
                err
            );
            println!("cargo:rustc-cfg=sppark_cuda_stub");
        }
    }
}
