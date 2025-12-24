use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

fn main() {
    println!("cargo:rerun-if-changed=src/msm_bls12_377.cu");
    println!("cargo:rerun-if-changed=src/msm_bw6_761_cgbn.cu");
    println!("cargo:rerun-if-changed=src/msm_mnt4_298_cgbn.cu");
    println!("cargo:rerun-if-changed=src/msm_mnt6_298_cgbn.cu");
    println!("cargo:rerun-if-changed=src/sparse_quotient_mnt4_298_cgbn.cu");
    println!("cargo:rerun-if-changed=src/sparse_quotient_mnt6_298_cgbn.cu");
    println!("cargo:rerun-if-changed=sppark");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_built)");
    println!("cargo::rustc-check-cfg=cfg(sppark_cuda_stub)");
    println!("cargo::rustc-check-cfg=cfg(bw6_cgbn_available)");
    println!("cargo::rustc-check-cfg=cfg(mnt4_cgbn_available)");
    println!("cargo::rustc-check-cfg=cfg(mnt6_cgbn_available)");
    println!("cargo::rustc-check-cfg=cfg(sparse_quotient_mnt4_available)");
    println!("cargo::rustc-check-cfg=cfg(sparse_quotient_mnt6_available)");

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
    let cgbn_include = "../cgbn-lib/include";

    // Atomic flags for tracking which kernels succeeded
    let bw6_available = AtomicBool::new(false);
    let mnt4_available = AtomicBool::new(false);
    let mnt6_available = AtomicBool::new(false);
    let sparse_mnt4_available = AtomicBool::new(false);
    let sparse_mnt6_available = AtomicBool::new(false);

    // Use thread::scope for parallel compilation of CGBN kernels
    println!("cargo:warning=Starting parallel CUDA kernel compilation...");

    thread::scope(|s| {
        // BW6-761 CGBN kernel
        let bw6_handle = s.spawn(|| {
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
                .flag("--maxrregcount=128");

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
        });

        // MNT4-298 CGBN kernel
        let mnt4_handle = s.spawn(|| {
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
                .flag("--maxrregcount=128");

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
        });

        // MNT6-298 CGBN kernel
        let mnt6_handle = s.spawn(|| {
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
                .flag("--maxrregcount=128");

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
        });

        // Sparse quotient MNT4-298 kernel
        let sparse_mnt4_handle = s.spawn(|| {
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
                .flag("--maxrregcount=128");

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
        });

        // Sparse quotient MNT6-298 kernel
        let sparse_mnt6_handle = s.spawn(|| {
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
                .flag("--maxrregcount=128");

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
        });

        // Wait for all threads and store results
        bw6_available.store(bw6_handle.join().unwrap_or(false), Ordering::SeqCst);
        mnt4_available.store(mnt4_handle.join().unwrap_or(false), Ordering::SeqCst);
        mnt6_available.store(mnt6_handle.join().unwrap_or(false), Ordering::SeqCst);
        sparse_mnt4_available.store(sparse_mnt4_handle.join().unwrap_or(false), Ordering::SeqCst);
        sparse_mnt6_available.store(sparse_mnt6_handle.join().unwrap_or(false), Ordering::SeqCst);
    });

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
