#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

#[cfg(feature = "gpu")]
pub use ark_ec::AffineRepr;
#[cfg(feature = "gpu")]
pub use ark_ff::BigInt;

#[cfg(feature = "gpu")]
use core::mem::{align_of, size_of};
#[cfg(feature = "gpu")]
use ark_ff::PrimeField;

/// Compile-time layout validation for G1Affine
///
/// CRITICAL: This validates the memory layout assumptions that CUDA code relies on.
/// If arkworks changes G1Affine layout (field ordering, padding), these checks
/// will fail at compile time, preventing silent data corruption.
#[cfg(feature = "gpu")]
const _: () = {
    use ark_bls12_377::{G1Affine, Fq};

    // G1Affine should be exactly 2 Fq fields + infinity flag
    // Expected layout: { x: Fq, y: Fq, infinity: bool }
    // Total size: 2 * sizeof(Fq) + sizeof(bool) + padding

    const FQ_SIZE: usize = size_of::<Fq>();
    const EXPECTED_MIN_SIZE: usize = 2 * FQ_SIZE + 1; // Two fields + bool
    const ACTUAL_SIZE: usize = size_of::<G1Affine>();

    // Size must be at least 2 Fq + 1 bool
    assert!(ACTUAL_SIZE >= EXPECTED_MIN_SIZE,
        "G1Affine size smaller than expected - layout changed");

    // Size should not be unreasonably large (catches major layout changes)
    const MAX_REASONABLE_SIZE: usize = 2 * FQ_SIZE + 32; // Allow 32 bytes padding
    assert!(ACTUAL_SIZE <= MAX_REASONABLE_SIZE,
        "G1Affine size unexpectedly large - layout may have changed");

    // Alignment should be reasonable
    const ACTUAL_ALIGN: usize = align_of::<G1Affine>();
    assert!(ACTUAL_ALIGN <= 64, "G1Affine alignment unexpectedly large");
};

/// Compile-time layout validation for BigInt<4>
#[cfg(feature = "gpu")]
const _: () = {
    use ark_ff::BigInt;

    // BigInt<4> should be exactly 4 u64 limbs
    const EXPECTED_SIZE: usize = 4 * size_of::<u64>();
    const ACTUAL_SIZE: usize = size_of::<BigInt<4>>();

    assert!(ACTUAL_SIZE == EXPECTED_SIZE,
        "BigInt<4> size mismatch - should be 4 * u64");

    // Alignment should be u64
    const ACTUAL_ALIGN: usize = align_of::<BigInt<4>>();
    assert!(ACTUAL_ALIGN == align_of::<u64>(),
        "BigInt<4> alignment mismatch - should align to u64");
};

/// Compile-time layout validation for BW6-761 G1Affine
///
/// CRITICAL: BW6-761 has a 761-bit base field (much larger than BLS12-377).
/// These checks validate memory layout assumptions for the CUDA code.
#[cfg(feature = "gpu")]
const _: () = {
    use ark_bw6_761::{G1Affine, Fq};

    // G1Affine should be exactly 2 Fq fields + infinity flag
    // Expected layout: { x: Fq, y: Fq, infinity: bool }
    // BW6-761 Fq is 761 bits (96 bytes), much larger than BLS12-377

    const FQ_SIZE: usize = size_of::<Fq>();
    const EXPECTED_MIN_SIZE: usize = 2 * FQ_SIZE + 1; // Two fields + bool
    const ACTUAL_SIZE: usize = size_of::<G1Affine>();

    // Size must be at least 2 Fq + 1 bool
    assert!(ACTUAL_SIZE >= EXPECTED_MIN_SIZE,
        "BW6-761 G1Affine size smaller than expected - layout changed");

    // Size should not be unreasonably large (catches major layout changes)
    const MAX_REASONABLE_SIZE: usize = 2 * FQ_SIZE + 64; // Allow 64 bytes padding
    assert!(ACTUAL_SIZE <= MAX_REASONABLE_SIZE,
        "BW6-761 G1Affine size unexpectedly large - layout may have changed");

    // Alignment should be reasonable
    const ACTUAL_ALIGN: usize = align_of::<G1Affine>();
    assert!(ACTUAL_ALIGN <= 64, "BW6-761 G1Affine alignment unexpectedly large");
};

/// Compile-time layout validation for BigInt<6>
/// BW6-761 scalar field is 377 bits (same as BLS12-377 base field)
#[cfg(feature = "gpu")]
const _: () = {
    use ark_ff::BigInt;

    // BigInt<6> should be exactly 6 u64 limbs
    const EXPECTED_SIZE: usize = 6 * size_of::<u64>();
    const ACTUAL_SIZE: usize = size_of::<BigInt<6>>();

    assert!(ACTUAL_SIZE == EXPECTED_SIZE,
        "BigInt<6> size mismatch - should be 6 * u64");

    // Alignment should be u64
    const ACTUAL_ALIGN: usize = align_of::<BigInt<6>>();
    assert!(ACTUAL_ALIGN == align_of::<u64>(),
        "BigInt<6> alignment mismatch - should align to u64");
};

#[cfg(all(feature = "gpu", not(sppark_cuda_stub)))]
#[allow(improper_ctypes)]
extern "C" {
    fn msm_bls12_377_g1(
        points: *const ark_bls12_377::G1Affine,
        scalars: *const BigInt<4>,
        count: usize,
        result: *mut ark_bls12_377::G1Projective,
        ffi_affine_sz: usize,
        ffi_scalar_sz: usize,
    ) -> i32;

    #[cfg(bw6_gpu_available)]
    fn msm_bw6_761_g1(
        points: *const ark_bw6_761::G1Affine,
        scalars: *const BigInt<6>,
        count: usize,
        result: *mut ark_bw6_761::G1Projective,
        ffi_affine_sz: usize,
        ffi_scalar_sz: usize,
    ) -> i32;

    #[cfg(bw6_cgbn_available)]
    fn msm_bw6_761_g1_cgbn(
        points: *const ark_bw6_761::G1Affine,
        scalars: *const BigInt<6>,
        count: usize,
        result: *mut ark_bw6_761::G1Projective,
        ffi_affine_sz: usize,
        ffi_scalar_sz: usize,
    ) -> i32;

    // Field operations test for BW6-761
    fn test_bw6_field_ops() -> i32;

    // Point addition test for BW6-761
    fn test_bw6_point_add(points: *const ark_bw6_761::G1Affine, result: *mut ark_bw6_761::G1Projective) -> i32;
}

#[cfg(all(feature = "gpu", sppark_cuda_stub))]
unsafe fn msm_bls12_377_g1(
    _points: *const ark_bls12_377::G1Affine,
    _scalars: *const BigInt<4>,
    _count: usize,
    _result: *mut ark_bls12_377::G1Projective,
    _ffi_affine_sz: usize,
    _ffi_scalar_sz: usize,
) -> i32 {
    // Return non-zero to signal fallback to CPU.
    1
}

#[cfg(all(feature = "gpu", sppark_cuda_stub))]
unsafe fn msm_bw6_761_g1(
    _points: *const ark_bw6_761::G1Affine,
    _scalars: *const BigInt<6>,
    _count: usize,
    _result: *mut ark_bw6_761::G1Projective,
    _ffi_affine_sz: usize,
    _ffi_scalar_sz: usize,
) -> i32 {
    // Return non-zero to signal fallback to CPU.
    1
}

#[cfg(all(feature = "gpu", sppark_cuda_stub))]
unsafe fn test_bw6_field_ops() -> i32 {
    // Return non-zero when CUDA is not available
    1
}

#[cfg(all(feature = "gpu", sppark_cuda_stub))]
unsafe fn test_bw6_point_add(_points: *const ark_bw6_761::G1Affine, _result: *mut ark_bw6_761::G1Projective) -> i32 {
    // Return non-zero when CUDA is not available
    1
}

// CPU fallback stub for BW6-761 (used when specialized kernel is not available)
#[cfg(all(feature = "gpu", not(bw6_gpu_available), not(sppark_cuda_stub)))]
unsafe fn msm_bw6_761_g1(
    _points: *const ark_bw6_761::G1Affine,
    _scalars: *const BigInt<6>,
    _count: usize,
    _result: *mut ark_bw6_761::G1Projective,
    _ffi_affine_sz: usize,
    _ffi_scalar_sz: usize,
) -> i32 {
    // Return non-zero to signal fallback to CPU (specialized kernel not available).
    1
}

#[cfg(feature = "gpu")]
pub trait GpuMsm<G: AffineRepr, S> {
    fn msm_gpu(points: &[G], scalars: &[S]) -> Result<G::Group, ()>;
}

#[cfg(feature = "gpu")]
impl GpuMsm<ark_bls12_377::G1Affine, BigInt<4>> for ark_bls12_377::G1Affine {
    fn msm_gpu(
        points: &[ark_bls12_377::G1Affine],
        scalars: &[BigInt<4>],
    ) -> Result<ark_bls12_377::G1Projective, ()> {
        if points.len() != scalars.len() {
            return Err(());
        }
        let count = points.len();
        let mut result = ark_bls12_377::G1Projective::default();
        unsafe {
            let status = msm_bls12_377_g1(
                points.as_ptr(),
                scalars.as_ptr(),
                count,
                &mut result,
                core::mem::size_of::<ark_bls12_377::G1Affine>(),
                core::mem::size_of::<BigInt<4>>(),
            );
            if status != 0 {
                return Err(());
            }
        }
        Ok(result)
    }
}

#[cfg(feature = "gpu")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Bw6GpuError {
    AffineLayoutMismatch,
    ScalarLayoutMismatch,
    CudaRuntime,
    Timeout,
    KernelUnavailable,
    Unknown(i32),
}

impl Bw6GpuError {
    fn from_status(code: i32) -> Self {
        match code {
            0 => unreachable!(),
            -1 => Bw6GpuError::AffineLayoutMismatch,
            -2 => Bw6GpuError::ScalarLayoutMismatch,
            -3 => Bw6GpuError::CudaRuntime,
            -4 => Bw6GpuError::Timeout,
            -5 => Bw6GpuError::KernelUnavailable,
            other => Bw6GpuError::Unknown(other),
        }
    }
}

impl GpuMsm<ark_bw6_761::G1Affine, BigInt<6>> for ark_bw6_761::G1Affine {
    fn msm_gpu(
        points: &[ark_bw6_761::G1Affine],
        scalars: &[BigInt<6>],
    ) -> Result<ark_bw6_761::G1Projective, ()> {
        if points.len() != scalars.len() {
            return Err(());
        }

        // For single-scalar MSM, do it on CPU to avoid GPU launch overhead
        if points.len() == 1 {
            let scalar = ark_bw6_761::Fr::from_bigint(scalars[0]).ok_or(())?;
            return Ok(points[0] * scalar);
        }

        let count = points.len();
        let mut result = ark_bw6_761::G1Projective::default();
        unsafe {
            let status = msm_bw6_761_g1(
                points.as_ptr(),
                scalars.as_ptr(),
                count,
                &mut result,
                core::mem::size_of::<ark_bw6_761::G1Affine>(),
                core::mem::size_of::<BigInt<6>>(),
            );
            if status != 0 {
                eprintln!("BW6-761 GPU MSM failed with status code: {}", status);
                eprintln!("  Affine size: {} bytes", core::mem::size_of::<ark_bw6_761::G1Affine>());
                eprintln!("  Scalar size: {} bytes", core::mem::size_of::<BigInt<6>>());
                return Err(());
            }
        }
        Ok(result)
    }
}

#[cfg(feature = "gpu")]
pub fn test_bw6_761_field_operations() -> Result<(), ()> {
    unsafe {
        let status = test_bw6_field_ops();
        if status != 0 {
            eprintln!("BW6-761 field operations test failed with status: {}", status);
            return Err(());
        }
    }
    Ok(())
}

#[cfg(not(feature = "gpu"))]
pub fn test_bw6_761_field_operations() -> Result<(), ()> {
    eprintln!("GPU feature not enabled");
    Err(())
}

#[cfg(feature = "gpu")]
pub fn test_bw6_761_point_addition(points: &[ark_bw6_761::G1Affine; 2]) -> Result<ark_bw6_761::G1Projective, ()> {
    let mut result = ark_bw6_761::G1Projective::default();
    unsafe {
        let status = test_bw6_point_add(points.as_ptr(), &mut result);
        if status != 0 {
            eprintln!("BW6-761 point addition test failed with status: {}", status);
            return Err(());
        }
    }
    Ok(result)
}

#[cfg(not(feature = "gpu"))]
pub fn test_bw6_761_point_addition(_points: &[ark_bw6_761::G1Affine; 2]) -> Result<ark_bw6_761::G1Projective, ()> {
    eprintln!("GPU feature not enabled");
    Err(())
}

// Internal helper for tests: force the GPU BW6-761 MSM path (no CPU fast-path).
#[cfg(all(feature = "gpu", bw6_gpu_available))]
pub fn msm_bw6_761_gpu_raw(
    points: &[ark_bw6_761::G1Affine],
    scalars: &[BigInt<6>],
) -> Result<ark_bw6_761::G1Projective, Bw6GpuError> {
    if points.len() != scalars.len() {
        return Err(Bw6GpuError::Unknown(-99)); // argument mismatch
    }
    let mut result = ark_bw6_761::G1Projective::default();
    let status = unsafe {
        msm_bw6_761_g1(
            points.as_ptr(),
            scalars.as_ptr(),
            points.len(),
            &mut result,
            core::mem::size_of::<ark_bw6_761::G1Affine>(),
            core::mem::size_of::<BigInt<6>>(),
        )
    };
    if status == 0 {
        Ok(result)
    } else {
        Err(Bw6GpuError::from_status(status))
    }
}

#[cfg(not(all(feature = "gpu", bw6_gpu_available)))]
pub fn msm_bw6_761_gpu_raw(
    _points: &[ark_bw6_761::G1Affine],
    _scalars: &[BigInt<6>],
) -> Result<ark_bw6_761::G1Projective, Bw6GpuError> {
    Err(Bw6GpuError::KernelUnavailable)
}

// Internal helper for tests: CGBN-based BW6-761 MSM kernel.
//
// WARNING: This kernel is an MVP STUB that does NOT compute correct MSM results.
// It ignores scalars, has no scalar multiplication, and returns placeholder output.
// DO NOT USE IN PRODUCTION. Enabled only for experimental/development purposes.
#[cfg(all(feature = "gpu", bw6_cgbn_available))]
pub fn msm_bw6_761_gpu_cgbn(
    points: &[ark_bw6_761::G1Affine],
    scalars: &[BigInt<6>],
) -> Result<ark_bw6_761::G1Projective, Bw6GpuError> {
    use ark_ff::PrimeField;

    // Gate behind environment variable to prevent accidental misuse
    if std::env::var("ENABLE_CGBN_STUB").is_err() {
        return Err(Bw6GpuError::KernelUnavailable);
    }

    if points.len() != scalars.len() {
        return Err(Bw6GpuError::Unknown(-99)); // argument mismatch
    }

    // Convert points from Montgomery form to plain form
    // arkworks stores Fq elements in Montgomery representation: x_mont = x * R mod p
    // into_bigint() performs Montgomery reduction: returns x (plain form)
    //
    // CRITICAL: CUDA expects uint32_t[24], but arkworks BigInt<12> is u64[12]
    // We need to convert from u64[12] to u32[24] (split each u64 into two u32s)
    #[repr(C)]
    struct PlainG1Affine {
        x: [u32; 24],  // CUDA expects uint32_t[24] (96 bytes)
        y: [u32; 24],  // CUDA expects uint32_t[24] (96 bytes)
        infinity: bool,
        _padding: [u8; 7],  // Match CUDA alignment (200 bytes total)
    }

    // Helper: convert BigInt<12> (u64[12]) to [u32; 24]
    fn bigint_to_u32_array(bigint: ark_ff::BigInt<12>) -> [u32; 24] {
        let mut result = [0u32; 24];
        for (i, &limb_u64) in bigint.0.iter().enumerate() {
            // Split each u64 into two u32s (little-endian)
            result[i * 2] = limb_u64 as u32;           // Lower 32 bits
            result[i * 2 + 1] = (limb_u64 >> 32) as u32; // Upper 32 bits
        }
        result
    }

    let plain_points: Vec<PlainG1Affine> = points.iter().map(|p| {
        if p.infinity {
            PlainG1Affine {
                x: [0u32; 24],
                y: [0u32; 24],
                infinity: true,
                _padding: [0u8; 7],
            }
        } else {
            // into_bigint() does Montgomery reduction: (x * R) * R^(-1) mod p = x
            let x_bigint = p.x.into_bigint();
            let y_bigint = p.y.into_bigint();
            PlainG1Affine {
                x: bigint_to_u32_array(x_bigint),
                y: bigint_to_u32_array(y_bigint),
                infinity: false,
                _padding: [0u8; 7],
            }
        }
    }).collect();

    let mut result = ark_bw6_761::G1Projective::default();
    let status = unsafe {
        msm_bw6_761_g1_cgbn(
            plain_points.as_ptr() as *const ark_bw6_761::G1Affine,
            scalars.as_ptr(),
            points.len(),
            &mut result,
            core::mem::size_of::<PlainG1Affine>(),
            core::mem::size_of::<BigInt<6>>(),
        )
    };
    if status == 0 {
        Ok(result)
    } else {
        Err(Bw6GpuError::from_status(status))
    }
}

#[cfg(not(all(feature = "gpu", bw6_cgbn_available)))]
pub fn msm_bw6_761_gpu_cgbn(
    _points: &[ark_bw6_761::G1Affine],
    _scalars: &[BigInt<6>],
) -> Result<ark_bw6_761::G1Projective, Bw6GpuError> {
    Err(Bw6GpuError::KernelUnavailable)
}
