use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

// Prefer sccache (wraps NVCC via cc-rs); fall back to ccache for host builds.
fn maybe_enable_compiler_cache() {
    if std::env::var_os("RUSTC_WRAPPER").is_some() {
        return;
    }

    if tool_in_path("sccache") {
        std::env::set_var("RUSTC_WRAPPER", "sccache");
        println!("cargo:warning=Using sccache to cache C/CUDA compilation");
        return;
    }

    if tool_in_path("ccache") {
        if std::env::var_os("CC").is_none() {
            std::env::set_var("CC", "ccache cc");
        }
        if std::env::var_os("CXX").is_none() {
            std::env::set_var("CXX", "ccache c++");
        }
        println!("cargo:warning=Using ccache to cache C/C++ compilation");
    }
}

fn tool_in_path(tool: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };

    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(tool);
        if candidate.is_file() {
            return true;
        }
        #[cfg(windows)]
        {
            for ext in ["exe", "cmd", "bat"] {
                if dir.join(format!("{tool}.{ext}")).is_file() {
                    return true;
                }
            }
        }
    }

    false
}

fn main() {
    maybe_enable_compiler_cache();
    println!("cargo:rerun-if-changed=src/msm_bls12_377.cu");
    println!("cargo:rerun-if-changed=src/msm_bw6_761_cgbn.cu");
    println!("cargo:rerun-if-changed=src/msm_mnt4_298_cgbn.cu");
    println!("cargo:rerun-if-changed=src/msm_mnt6_298_cgbn.cu");
    println!("cargo:rerun-if-changed=src/sparse_quotient_mnt4_298_cgbn.cu");
    println!("cargo:rerun-if-changed=src/sparse_quotient_mnt6_298_cgbn.cu");
    println!("cargo:rerun-if-changed=src/sparse_quotient_bw6_761_cgbn.cu");
    println!("cargo:rerun-if-changed=sppark");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=SKIP_BW6");
    println!("cargo:rerun-if-env-changed=CUDA_PARALLEL");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_built)");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_stub)");
    println!("cargo::rustc-check-cfg=cfg(bw6_cgbn_available)");
    println!("cargo::rustc-check-cfg=cfg(mnt4_cgbn_available)");
    println!("cargo::rustc-check-cfg=cfg(mnt6_cgbn_available)");
    println!("cargo::rustc-check-cfg=cfg(sparse_quotient_mnt4_available)");
    println!("cargo::rustc-check-cfg=cfg(sparse_quotient_mnt6_available)");
    println!("cargo::rustc-check-cfg=cfg(sparse_quotient_bw6_available)");

    if std::env::var("CARGO_FEATURE_GPU").is_err() {
        println!("cargo:warning=CUDA build skipped (feature \"gpu\" not enabled)");
        return;
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_include = format!("{}/include", cuda_home);
    let cuda_lib = format!("{}/lib64", cuda_home);

    // Allow GPU architecture to be configured via environment variable
    let cuda_arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());
    let arch_flag = format!("-arch={}", cuda_arch);

    let cuda_verbose = std::env::var("CUDA_VERBOSE").is_ok();
    let bw6_debug = std::env::var("BW6_DEBUG").is_ok();
    let skip_bw6 = std::env::var("SKIP_BW6").is_ok();
    let cgbn_include = "../cgbn-lib/include";

    // Atomic flags for tracking which kernels succeeded
    let bw6_available = AtomicBool::new(false);
    let mnt4_available = AtomicBool::new(false);
    let mnt6_available = AtomicBool::new(false);
    let sparse_mnt4_available = AtomicBool::new(false);
    let sparse_mnt6_available = AtomicBool::new(false);
    let sparse_bw6_available = AtomicBool::new(false);

    // By default, compile kernels sequentially to reduce memory pressure
    // Set CUDA_PARALLEL=1 to enable parallel compilation (requires more RAM)
    let cuda_parallel = std::env::var("CUDA_PARALLEL").is_ok();

    // Helper closures for building each kernel
    let build_bw6 = || {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .flag("-std=c++17")
            .flag("-allow-unsupported-compiler")
            .flag(&arch_flag)
            .flag("-rdc=true")
            .flag("-Xcompiler").flag("-O3")
            .flag("-Xcompiler").flag("-Wno-unknown-pragmas")
            .flag("-Xcompiler").flag("-Wno-maybe-uninitialized")
            .flag("-Xcompiler").flag("-fvisibility=hidden")  // Hide internal CGBN symbols
            .flag("--maxrregcount=128")
            // Force IMAD implementation to reduce code generation (less memory during compilation)
            .define("XMP_IMAD", None)
            // Lower ptxas optimization to reduce memory usage
            .flag("-Xptxas=-O1");

        if cuda_verbose {
            build.flag("--ptxas-options=-v");
        }
        if bw6_debug {
            build.define("BW6_DEBUG", None);
        }

        build
            .include("sppark")
            .include("sppark/blst/src")
            .include(&cuda_include)
            .include(cgbn_include)
            .include("/usr/include")
            .include("/usr/include/x86_64-linux-gnu")
            .file("src/msm_bw6_761_cgbn.cu");

        build.try_compile("msm_bw6_cgbn").is_ok()
    };

    let build_mnt4 = || {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .flag("-std=c++17")
            .flag("-allow-unsupported-compiler")
            .flag(&arch_flag)
            .flag("-rdc=true")
            .flag("-Xcompiler").flag("-O3")
            .flag("-Xcompiler").flag("-Wno-unknown-pragmas")
            .flag("-Xcompiler").flag("-Wno-maybe-uninitialized")
            .flag("-Xcompiler").flag("-fvisibility=hidden")  // Hide internal CGBN symbols
            .flag("--maxrregcount=128")
            // Memory optimization: force simpler IMAD and lower ptxas optimization
            .define("XMP_IMAD", None)
            .flag("-Xptxas=-O1");

        if cuda_verbose {
            build.flag("--ptxas-options=-v");
        }

        build
            .include("sppark")
            .include("sppark/blst/src")
            .include(&cuda_include)
            .include(cgbn_include)
            .include("/usr/include")
            .include("/usr/include/x86_64-linux-gnu")
            .file("src/msm_mnt4_298_cgbn.cu");

        build.try_compile("msm_mnt4_cgbn").is_ok()
    };

    let build_mnt6 = || {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .flag("-std=c++17")
            .flag("-allow-unsupported-compiler")
            .flag(&arch_flag)
            .flag("-rdc=true")
            .flag("-Xcompiler").flag("-O3")
            .flag("-Xcompiler").flag("-Wno-unknown-pragmas")
            .flag("-Xcompiler").flag("-Wno-maybe-uninitialized")
            .flag("-Xcompiler").flag("-fvisibility=hidden")  // Hide internal CGBN symbols
            .flag("--maxrregcount=128")
            // Memory optimization: force simpler IMAD and lower ptxas optimization
            .define("XMP_IMAD", None)
            .flag("-Xptxas=-O1");

        if cuda_verbose {
            build.flag("--ptxas-options=-v");
        }

        build
            .include("sppark")
            .include("sppark/blst/src")
            .include(&cuda_include)
            .include(cgbn_include)
            .include("/usr/include")
            .include("/usr/include/x86_64-linux-gnu")
            .file("src/msm_mnt6_298_cgbn.cu");

        build.try_compile("msm_mnt6_cgbn").is_ok()
    };

    let build_sparse_mnt4 = || {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .flag("-std=c++17")
            .flag("-allow-unsupported-compiler")
            .flag(&arch_flag)
            .flag("-rdc=true")
            .flag("-Xcompiler").flag("-O3")
            .flag("-Xcompiler").flag("-Wno-unknown-pragmas")
            .flag("-Xcompiler").flag("-Wno-maybe-uninitialized")
            .flag("-Xcompiler").flag("-fvisibility=hidden")  // Hide internal CGBN symbols
            .flag("--maxrregcount=128")
            // Memory optimization: force simpler IMAD and lower ptxas optimization
            .define("XMP_IMAD", None)
            .flag("-Xptxas=-O1");

        if cuda_verbose {
            build.flag("--ptxas-options=-v");
        }

        build
            .include("sppark")
            .include("sppark/blst/src")
            .include(&cuda_include)
            .include(cgbn_include)
            .include("/usr/include")
            .include("/usr/include/x86_64-linux-gnu")
            .file("src/sparse_quotient_mnt4_298_cgbn.cu");

        build.try_compile("sparse_quotient_mnt4_cgbn").is_ok()
    };

    let build_sparse_mnt6 = || {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .flag("-std=c++17")
            .flag("-allow-unsupported-compiler")
            .flag(&arch_flag)
            .flag("-rdc=true")
            .flag("-Xcompiler").flag("-O3")
            .flag("-Xcompiler").flag("-Wno-unknown-pragmas")
            .flag("-Xcompiler").flag("-Wno-maybe-uninitialized")
            .flag("-Xcompiler").flag("-fvisibility=hidden")  // Hide internal CGBN symbols
            .flag("--maxrregcount=128")
            // Memory optimization: force simpler IMAD and lower ptxas optimization
            .define("XMP_IMAD", None)
            .flag("-Xptxas=-O1");

        if cuda_verbose {
            build.flag("--ptxas-options=-v");
        }

        build
            .include("sppark")
            .include("sppark/blst/src")
            .include(&cuda_include)
            .include(cgbn_include)
            .include("/usr/include")
            .include("/usr/include/x86_64-linux-gnu")
            .file("src/sparse_quotient_mnt6_298_cgbn.cu");

        build.try_compile("sparse_quotient_mnt6_cgbn").is_ok()
    };

    let build_sparse_bw6 = || {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .flag("-std=c++17")
            .flag("-allow-unsupported-compiler")
            .flag(&arch_flag)
            .flag("-rdc=true")
            .flag("-Xcompiler").flag("-O3")
            .flag("-Xcompiler").flag("-Wno-unknown-pragmas")
            .flag("-Xcompiler").flag("-Wno-maybe-uninitialized")
            .flag("-Xcompiler").flag("-fvisibility=hidden")
            .flag("--maxrregcount=128")
            .define("XMP_IMAD", None)
            .flag("-Xptxas=-O1");

        if cuda_verbose {
            build.flag("--ptxas-options=-v");
        }

        build
            .include("sppark")
            .include("sppark/blst/src")
            .include(&cuda_include)
            .include(cgbn_include)
            .include("/usr/include")
            .include("/usr/include/x86_64-linux-gnu")
            .file("src/sparse_quotient_bw6_761_cgbn.cu");

        build.try_compile("sparse_quotient_bw6_cgbn").is_ok()
    };

    if cuda_parallel {
        // Parallel compilation (requires more RAM - 64GB+ recommended)
        println!("cargo:warning=Starting PARALLEL CUDA kernel compilation (CUDA_PARALLEL=1)...");

        thread::scope(|s| {
            let bw6_handle = if skip_bw6 {
                println!("cargo:warning=Skipping BW6-761 CGBN kernel (SKIP_BW6=1)");
                None
            } else {
                Some(s.spawn(build_bw6))
            };
            let mnt4_handle = s.spawn(build_mnt4);
            let mnt6_handle = s.spawn(build_mnt6);
            let sparse_mnt4_handle = s.spawn(build_sparse_mnt4);
            let sparse_mnt6_handle = s.spawn(build_sparse_mnt6);
            let sparse_bw6_handle = if skip_bw6 {
                None
            } else {
                Some(s.spawn(build_sparse_bw6))
            };

            // Wait for all threads and store results
            if let Some(handle) = bw6_handle {
                bw6_available.store(handle.join().unwrap_or(false), Ordering::SeqCst);
            }
            mnt4_available.store(mnt4_handle.join().unwrap_or(false), Ordering::SeqCst);
            mnt6_available.store(mnt6_handle.join().unwrap_or(false), Ordering::SeqCst);
            sparse_mnt4_available.store(sparse_mnt4_handle.join().unwrap_or(false), Ordering::SeqCst);
            sparse_mnt6_available.store(sparse_mnt6_handle.join().unwrap_or(false), Ordering::SeqCst);
            if let Some(handle) = sparse_bw6_handle {
                sparse_bw6_available.store(handle.join().unwrap_or(false), Ordering::SeqCst);
            }
        });
    } else {
        // Sequential compilation (default - reduces memory pressure)
        println!("cargo:warning=Starting SEQUENTIAL CUDA kernel compilation (set CUDA_PARALLEL=1 for parallel)...");

        if skip_bw6 {
            println!("cargo:warning=Skipping BW6-761 CGBN kernel (SKIP_BW6=1)");
        } else {
            println!("cargo:warning=Compiling BW6-761 CGBN kernel...");
            bw6_available.store(build_bw6(), Ordering::SeqCst);
        }

        println!("cargo:warning=Compiling MNT4-298 CGBN kernel...");
        mnt4_available.store(build_mnt4(), Ordering::SeqCst);

        println!("cargo:warning=Compiling MNT6-298 CGBN kernel...");
        mnt6_available.store(build_mnt6(), Ordering::SeqCst);

        println!("cargo:warning=Compiling Sparse Quotient MNT4-298 kernel...");
        sparse_mnt4_available.store(build_sparse_mnt4(), Ordering::SeqCst);

        println!("cargo:warning=Compiling Sparse Quotient MNT6-298 kernel...");
        sparse_mnt6_available.store(build_sparse_mnt6(), Ordering::SeqCst);

        if skip_bw6 {
            println!("cargo:warning=Skipping Sparse Quotient BW6-761 kernel (SKIP_BW6=1)");
        } else {
            println!("cargo:warning=Compiling Sparse Quotient BW6-761 kernel...");
            sparse_bw6_available.store(build_sparse_bw6(), Ordering::SeqCst);
        }
    }

    // Report CGBN kernel availability
    if bw6_available.load(Ordering::SeqCst) {
        println!("cargo:rustc-link-lib=gmp");
        println!("cargo:rustc-cfg=bw6_cgbn_available");
        println!("cargo:warning=BW6-761 CGBN: AVAILABLE (TPI=8, BITS=768)");
    } else {
        println!("cargo:warning=BW6-761 CGBN: NOT AVAILABLE");
    }

    if mnt4_available.load(Ordering::SeqCst) {
        println!("cargo:rustc-cfg=mnt4_cgbn_available");
        println!("cargo:warning=MNT4-298 CGBN: AVAILABLE (TPI=8, BITS=320)");
    } else {
        println!("cargo:warning=MNT4-298 CGBN: NOT AVAILABLE");
    }

    if mnt6_available.load(Ordering::SeqCst) {
        println!("cargo:rustc-cfg=mnt6_cgbn_available");
        println!("cargo:warning=MNT6-298 CGBN: AVAILABLE (TPI=8, BITS=320)");
    } else {
        println!("cargo:warning=MNT6-298 CGBN: NOT AVAILABLE");
    }

    if sparse_mnt4_available.load(Ordering::SeqCst) {
        println!("cargo:rustc-cfg=sparse_quotient_mnt4_available");
        println!("cargo:warning=Sparse Quotient MNT4-298: AVAILABLE");
    } else {
        println!("cargo:warning=Sparse Quotient MNT4-298: NOT AVAILABLE");
    }

    if sparse_mnt6_available.load(Ordering::SeqCst) {
        println!("cargo:rustc-cfg=sparse_quotient_mnt6_available");
        println!("cargo:warning=Sparse Quotient MNT6-298: AVAILABLE");
    } else {
        println!("cargo:warning=Sparse Quotient MNT6-298: NOT AVAILABLE");
    }

    if sparse_bw6_available.load(Ordering::SeqCst) {
        println!("cargo:rustc-cfg=sparse_quotient_bw6_available");
        println!("cargo:warning=Sparse Quotient BW6-761: AVAILABLE (TPI=8, BITS=384)");
    } else {
        println!("cargo:warning=Sparse Quotient BW6-761: NOT AVAILABLE");
    }

    // Build BLS12-377 kernel (required for base functionality)
    let mut cuda_build = cc::Build::new();
    cuda_build
        .cuda(true)
        .flag("-std=c++17")
        .flag("-allow-unsupported-compiler")
        .flag(&arch_flag)
        .flag("-rdc=true")
        .flag("-Xcompiler").flag("-O3")
        .flag("-Xcompiler").flag("-Wno-unknown-pragmas")
        .flag("-Xcompiler").flag("-Wno-maybe-uninitialized")
        .flag("--maxrregcount=128");

    if cuda_verbose {
        cuda_build.flag("--ptxas-options=-v");
    }

    cuda_build
        .include("sppark")
        .include("sppark/blst/src")
        .include(&cuda_include)
        .file("src/msm_bls12_377.cu");

    // Build C++ utility files needed by sppark (needs CUDA headers)
    let mut cpp_build = cc::Build::new();
    cpp_build
        .cuda(true)
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-x").flag("cu")
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
            println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
        }
        (Err(err), _, _) | (_, Err(err), _) | (_, _, Err(err)) => {
            println!("cargo:warning=Build disabled: {}", err);
            println!("cargo:rustc-cfg=sppark_cuda_stub");
        }
    }
}
