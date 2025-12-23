fn main() {
    println!("cargo:rerun-if-changed=src/msm_bls12_377.cu");
    println!("cargo:rerun-if-changed=src/msm_bw6_761.cu");
    println!("cargo:rerun-if-changed=src/msm_bw6_761_specialized.cu");
    println!("cargo:rerun-if-changed=src/msm_bw6_761_cgbn.cu");
    println!("cargo:rerun-if-changed=sppark");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=BW6_WBITS");
    println!("cargo:rerun-if-env-changed=BW6_NTHREADS");
    println!("cargo:rerun-if-env-changed=BW6_NSTREAMS");
    println!("cargo:rerun-if-env-changed=BW6_MAXREG");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_built)");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_stub)");
    println!("cargo::rustc-check-cfg=cfg(bw6_cpu_fallback)");
    println!("cargo::rustc-check-cfg=cfg(bw6_gpu_available)");
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

    // Tunables for BW6-761 resource usage; fall back to conservative defaults if unset
    fn sanitize(val: u32, allowed: &[u32], fallback: u32) -> u32 {
        if allowed.contains(&val) {
            val
        } else {
            fallback
        }
    }

    let bw6_wbits_env = std::env::var("BW6_WBITS").ok().and_then(|v| v.parse::<u32>().ok());
    let bw6_nthreads_env = std::env::var("BW6_NTHREADS").ok().and_then(|v| v.parse::<u32>().ok());
    let bw6_nstreams_env = std::env::var("BW6_NSTREAMS").ok().and_then(|v| v.parse::<u32>().ok());
    let bw6_maxreg = std::env::var("BW6_MAXREG").unwrap_or_else(|_| "96".to_string());

    // Allowed sets based on pippenger.cuh constraints
    let bw6_wbits_raw = bw6_wbits_env.unwrap_or(6);
    let bw6_nthreads_raw = bw6_nthreads_env.unwrap_or(64);
    let bw6_nstreams_raw = bw6_nstreams_env.unwrap_or(4);

    let bw6_wbits = sanitize(bw6_wbits_raw, &[5, 6, 7, 8], 6);
    let bw6_nthreads = sanitize(bw6_nthreads_raw, &[32, 64, 128, 256], 64);
    let bw6_nstreams = sanitize(bw6_nstreams_raw, &[2, 4, 8], 4);

    if bw6_wbits != bw6_wbits_raw {
        println!(
            "cargo:warning=BW6_WBITS={} not allowed; clamped to {} (allowed: 5,6,7,8)",
            bw6_wbits_raw, bw6_wbits
        );
    }
    if bw6_nthreads != bw6_nthreads_raw {
        println!(
            "cargo:warning=BW6_NTHREADS={} not allowed; clamped to {} (allowed: 32,64,128,256)",
            bw6_nthreads_raw, bw6_nthreads
        );
    }
    if bw6_nstreams != bw6_nstreams_raw {
        println!(
            "cargo:warning=BW6_NSTREAMS={} not allowed; clamped to {} (allowed: 2,4,8)",
            bw6_nstreams_raw, bw6_nstreams
        );
    }

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

    // BW6-761 kernel selection:
    // - Generic kernel (bw6_generic feature): Known to hang due to register pressure
    // - Specialized kernel (default with gpu): Optimized for large fields
    if std::env::var("CARGO_FEATURE_BW6_GENERIC").is_ok() {
        println!("cargo:warning=Building generic BW6-761 kernel (known to hang)");
        cuda_build.file("src/msm_bw6_761.cu");
    } else {
        // Build specialized BW6-761 kernel with aggressive register management
        println!("cargo:warning=Building specialized BW6-761 kernel (experimental)");

        // Create a separate build for BW6-761 with stricter register limits
        let mut bw6_build = cc::Build::new();
        let wbits_str = bw6_wbits.to_string();
        let nthreads_str = bw6_nthreads.to_string();
        let nstreams_str = bw6_nstreams.to_string();

        bw6_build
            .cuda(true)
            .flag("-std=c++17")
            .flag("-allow-unsupported-compiler")
            .flag(&arch_flag)
            .flag("-rdc=true")
            .flag("-Xcompiler")
            .flag("-O3")
            .flag("-Xcompiler")
            .flag("-Wno-unknown-pragmas")  // Suppress #pragma unroll warnings
            .flag("-Xcompiler")
            .flag("-Wno-maybe-uninitialized")  // Suppress false-positive flow analysis warnings
            .flag(&format!("--maxrregcount={}", bw6_maxreg));

        // Only enable verbose ptxas output if CUDA_VERBOSE=1
        if std::env::var("CUDA_VERBOSE").is_ok() {
            bw6_build.flag("--ptxas-options=-v");
        }

        bw6_build
            .define("MSM_WBITS", Some(wbits_str.as_str()))
            .define("MSM_NTHREADS", Some(nthreads_str.as_str()))
            .define("MSM_NSTREAMS", Some(nstreams_str.as_str()));

        // Enable debug logging if BW6_DEBUG env var is set
        if std::env::var("BW6_DEBUG").is_ok() {
            bw6_build.define("BW6_DEBUG", None);
        }

        bw6_build
            .include("sppark")
            .include("sppark/blst/src")
            .include(&cuda_include)
            .file("src/msm_bw6_761_specialized.cu");

        // Try to compile specialized kernel
        if let Err(e) = bw6_build.try_compile("msm_bw6_specialized") {
            println!("cargo:warning=Specialized BW6-761 kernel failed to build: {}", e);
            println!("cargo:warning=BW6-761 will use CPU fallback");
            // Don't fail the whole build, just skip BW6-761 GPU support
        } else {
            // Successfully built specialized kernel, disable CPU fallback
            // Note: We'll set a cfg to indicate BW6 GPU is available
            println!("cargo:rustc-cfg=bw6_gpu_available");
            println!(
                "cargo:warning=BW6-761 GPU: AVAILABLE (WBITS={}, NTHREADS={}, NSTREAMS={}, MAXREG={})",
                bw6_wbits, bw6_nthreads, bw6_nstreams, bw6_maxreg
            );
        }

        // Build CGBN-based BW6-761 kernel (experimental)
        println!("cargo:warning=Building CGBN-based BW6-761 kernel (experimental)");

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
            .flag("--maxrregcount=128");  // CGBN uses fewer registers

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
            println!("cargo:warning=CGBN version not available");
        } else {
            // Link against GMP (required by CGBN)
            println!("cargo:rustc-link-lib=gmp");
            println!("cargo:rustc-cfg=bw6_cgbn_available");
            println!("cargo:warning=BW6-761 CGBN: AVAILABLE (TPI=8, BITS=768)");
        }
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
            // Allow multiple definitions of CUDA kernel functions when linking BLS12-377 and BW6-761
            // Both kernels include the same sppark headers, causing duplicate symbols
            println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
            // Note: bw6_gpu_available is set above if specialized kernel builds successfully
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
