fn main() {
    println!("cargo:rerun-if-changed=src/msm_bls12_377.cu");
    println!("cargo:rerun-if-changed=src/msm_bw6_761_cgbn.cu");
    println!("cargo:rerun-if-changed=sppark");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_built)");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_stub)");
    println!("cargo::rustc-check-cfg=cfg(bw6_cgbn_available)");

    if std::env::var("CARGO_FEATURE_GPU").is_err() {
        println!("cargo:warning=CUDA build skipped (feature \"gpu\" not enabled)");
        return;
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_include = format!("{}/include", cuda_home);
    let cuda_lib = format!("{}/lib64", cuda_home);

    // Allow GPU architecture to be configured via environment variable
    // Default: sm_75 (Turing - GTX 1660, RTX 2060, etc.)
    // Common values: sm_60 (Pascal), sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere), sm_89 (Ada), sm_90 (Hopper)
    let cuda_arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());
    let arch_flag = format!("-arch={}", cuda_arch);

    // Build CUDA kernels
    let mut cuda_build = cc::Build::new();
    cuda_build
        .cuda(true)
        .flag("-std=c++17")
        .flag("-allow-unsupported-compiler")
        .flag(&arch_flag)
        .flag("-rdc=true")  // Enable relocatable device code for multiple .cu files
        .flag("-Xcompiler")
        .flag("-O3")
        .flag("-Xcompiler")
        .flag("-Wno-unknown-pragmas")  // Suppress #pragma unroll warnings
        .flag("-Xcompiler")
        .flag("-Wno-maybe-uninitialized")  // Suppress false-positive flow analysis warnings
        .flag("--maxrregcount=128");

    // Only enable verbose ptxas output if CUDA_VERBOSE=1
    if std::env::var("CUDA_VERBOSE").is_ok() {
        cuda_build.flag("--ptxas-options=-v");
    }

    cuda_build
        .include("sppark")
        .include("sppark/blst/src")
        .include(&cuda_include)
        .file("src/msm_bls12_377.cu");

    // Build CGBN-based BW6-761 kernel
    // Note: The sppark-based specialized kernel was removed because BW6-761's 761-bit
    // base field generates stack frames too large for GPU execution (~22KB/thread).
    // See docs/BW6_761_CGBN.md for historical context.
    let cgbn_include = "../cgbn-lib/include";
    let mut cgbn_build = cc::Build::new();

    cgbn_build
        .cuda(true)
        .flag("-std=c++17")
        .flag("-allow-unsupported-compiler")
        .flag(&arch_flag)
        .flag("-rdc=true")
        .flag("-Xcompiler")
        .flag("-O3")
        .flag("-Xcompiler")
        .flag("-Wno-unknown-pragmas")
        .flag("-Xcompiler")
        .flag("-Wno-maybe-uninitialized")
        .flag("--maxrregcount=128");

    // Only enable verbose ptxas output if CUDA_VERBOSE=1
    if std::env::var("CUDA_VERBOSE").is_ok() {
        cgbn_build.flag("--ptxas-options=-v");
    }

    // Enable debug logging if BW6_DEBUG env var is set
    if std::env::var("BW6_DEBUG").is_ok() {
        cgbn_build.define("BW6_DEBUG", None);
    }

    cgbn_build
        .include("sppark")
        .include("sppark/blst/src")
        .include(&cuda_include)
        .include(cgbn_include)
        .include("/usr/include")                    // For gmp.h
        .include("/usr/include/x86_64-linux-gnu")   // For gmp.h (arch-specific)
        .file("src/msm_bw6_761_cgbn.cu");

    // Try to compile CGBN kernel
    if let Err(e) = cgbn_build.try_compile("msm_bw6_cgbn") {
        println!("cargo:warning=CGBN BW6-761 kernel failed to build: {}", e);
        println!("cargo:warning=BW6-761 GPU MSM not available");
    } else {
        // Link against GMP (required by CGBN)
        println!("cargo:rustc-link-lib=gmp");
        println!("cargo:rustc-cfg=bw6_cgbn_available");
        println!("cargo:warning=BW6-761 CGBN: AVAILABLE (TPI=8, BITS=768)");
    }

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
            // Allow multiple definitions of CUDA kernel functions when linking BLS12-377 and BW6-761 CGBN
            // Both kernels include the same sppark headers, causing duplicate symbols
            println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
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
