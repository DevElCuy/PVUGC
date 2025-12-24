#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

#[cfg(feature = "gpu")]
pub use ark_ec::AffineRepr;
#[cfg(feature = "gpu")]
pub use ark_ff::BigInt;

#[cfg(feature = "gpu")]
use core::mem::{align_of, size_of};

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

/// Compile-time layout validation for MNT4-298 G1Affine
#[cfg(feature = "gpu")]
const _: () = {
    use ark_mnt4_298::{G1Affine, Fq};

    // G1Affine should be exactly 2 Fq fields + infinity flag
    // MNT4-298 Fq is 298 bits (40 bytes)
    const FQ_SIZE: usize = size_of::<Fq>();
    const EXPECTED_MIN_SIZE: usize = 2 * FQ_SIZE + 1;
    const ACTUAL_SIZE: usize = size_of::<G1Affine>();

    assert!(ACTUAL_SIZE >= EXPECTED_MIN_SIZE,
        "MNT4-298 G1Affine size smaller than expected - layout changed");

    const MAX_REASONABLE_SIZE: usize = 2 * FQ_SIZE + 32;
    assert!(ACTUAL_SIZE <= MAX_REASONABLE_SIZE,
        "MNT4-298 G1Affine size unexpectedly large - layout may have changed");

    const ACTUAL_ALIGN: usize = align_of::<G1Affine>();
    assert!(ACTUAL_ALIGN <= 64, "MNT4-298 G1Affine alignment unexpectedly large");
};

/// Compile-time layout validation for MNT6-298 G1Affine
#[cfg(feature = "gpu")]
const _: () = {
    use ark_mnt6_298::{G1Affine, Fq};

    // G1Affine should be exactly 2 Fq fields + infinity flag
    // MNT6-298 Fq is 298 bits (40 bytes)
    const FQ_SIZE: usize = size_of::<Fq>();
    const EXPECTED_MIN_SIZE: usize = 2 * FQ_SIZE + 1;
    const ACTUAL_SIZE: usize = size_of::<G1Affine>();

    assert!(ACTUAL_SIZE >= EXPECTED_MIN_SIZE,
        "MNT6-298 G1Affine size smaller than expected - layout changed");

    const MAX_REASONABLE_SIZE: usize = 2 * FQ_SIZE + 32;
    assert!(ACTUAL_SIZE <= MAX_REASONABLE_SIZE,
        "MNT6-298 G1Affine size unexpectedly large - layout may have changed");

    const ACTUAL_ALIGN: usize = align_of::<G1Affine>();
    assert!(ACTUAL_ALIGN <= 64, "MNT6-298 G1Affine alignment unexpectedly large");
};

/// Compile-time layout validation for BigInt<5>
/// MNT4-298 and MNT6-298 fields are 298 bits (5 u64 limbs)
#[cfg(feature = "gpu")]
const _: () = {
    use ark_ff::BigInt;

    // BigInt<5> should be exactly 5 u64 limbs
    const EXPECTED_SIZE: usize = 5 * size_of::<u64>();
    const ACTUAL_SIZE: usize = size_of::<BigInt<5>>();

    assert!(ACTUAL_SIZE == EXPECTED_SIZE,
        "BigInt<5> size mismatch - should be 5 * u64");

    // Alignment should be u64
    const ACTUAL_ALIGN: usize = align_of::<BigInt<5>>();
    assert!(ACTUAL_ALIGN == align_of::<u64>(),
        "BigInt<5> alignment mismatch - should align to u64");
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

    #[cfg(bw6_cgbn_available)]
    fn msm_bw6_761_g1_cgbn(
        points: *const ark_bw6_761::G1Affine,
        scalars: *const BigInt<6>,
        count: usize,
        result: *mut ark_bw6_761::G1Projective,
        ffi_affine_sz: usize,
        ffi_scalar_sz: usize,
    ) -> i32;

    #[cfg(mnt4_cgbn_available)]
    fn msm_mnt4_298_g1_cgbn(
        points: *const ark_mnt4_298::G1Affine,
        scalars: *const BigInt<5>,
        count: usize,
        result: *mut ark_mnt4_298::G1Projective,
        ffi_affine_sz: usize,
        ffi_scalar_sz: usize,
    ) -> i32;

    #[cfg(mnt6_cgbn_available)]
    fn msm_mnt6_298_g1_cgbn(
        points: *const ark_mnt6_298::G1Affine,
        scalars: *const BigInt<5>,
        count: usize,
        result: *mut ark_mnt6_298::G1Projective,
        ffi_affine_sz: usize,
        ffi_scalar_sz: usize,
    ) -> i32;
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

#[cfg(feature = "gpu")]
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

// BW6-761 GPU MSM implementation using CGBN kernel
//
// Note: The sppark-based specialized kernel was removed because BW6-761's 761-bit
// base field generates stack frames too large for GPU execution (~22KB/thread).
// See docs/BW6_761_CGBN.md for historical context.
#[cfg(all(feature = "gpu", bw6_cgbn_available))]
impl GpuMsm<ark_bw6_761::G1Affine, BigInt<6>> for ark_bw6_761::G1Affine {
    fn msm_gpu(
        points: &[ark_bw6_761::G1Affine],
        scalars: &[BigInt<6>],
    ) -> Result<ark_bw6_761::G1Projective, ()> {
        msm_bw6_761_gpu_cgbn(points, scalars).map_err(|_| ())
    }
}

// Fallback when CGBN is not available - return error to signal CPU fallback
#[cfg(all(feature = "gpu", not(bw6_cgbn_available)))]
impl GpuMsm<ark_bw6_761::G1Affine, BigInt<6>> for ark_bw6_761::G1Affine {
    fn msm_gpu(
        _points: &[ark_bw6_761::G1Affine],
        _scalars: &[BigInt<6>],
    ) -> Result<ark_bw6_761::G1Projective, ()> {
        Err(()) // CGBN kernel not available, use CPU fallback
    }
}

// CGBN-based BW6-761 MSM kernel.
//
// This is the primary GPU MSM implementation for BW6-761 using NVIDIA's CGBN library.
// CGBN distributes field element limbs across cooperating threads (TPI=8), avoiding
// the per-thread stack overflow issues that plague sppark's template-based approach.
#[cfg(all(feature = "gpu", bw6_cgbn_available))]
pub fn msm_bw6_761_gpu_cgbn(
    points: &[ark_bw6_761::G1Affine],
    scalars: &[BigInt<6>],
) -> Result<ark_bw6_761::G1Projective, Bw6GpuError> {
    use ark_ff::PrimeField;
    use ark_bw6_761::Fq;

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

    // Plain form Jacobian/Projective point (output from CUDA)
    // CUDA returns normalized affine coordinates as Jacobian with Z=1
    #[repr(C)]
    struct PlainG1Projective {
        x: [u32; 24],  // CUDA returns uint32_t[24] (96 bytes)
        y: [u32; 24],  // CUDA returns uint32_t[24] (96 bytes)
        z: [u32; 24],  // CUDA returns uint32_t[24] (96 bytes) - should be 1 for normalized
        infinity: bool,
        _padding: [u8; 7],  // Match CUDA alignment
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

    // Helper: convert [u32; 24] back to BigInt<12> (u64[12])
    fn u32_array_to_bigint(arr: &[u32; 24]) -> ark_ff::BigInt<12> {
        let mut limbs = [0u64; 12];
        for i in 0..12 {
            // Combine two u32s into one u64 (little-endian)
            limbs[i] = (arr[i * 2] as u64) | ((arr[i * 2 + 1] as u64) << 32);
        }
        ark_ff::BigInt(limbs)
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

    // Use PlainG1Projective to receive CUDA output (plain form coordinates)
    let mut plain_result = PlainG1Projective {
        x: [0u32; 24],
        y: [0u32; 24],
        z: [0u32; 24],
        infinity: false,
        _padding: [0u8; 7],
    };

    let status = unsafe {
        msm_bw6_761_g1_cgbn(
            plain_points.as_ptr() as *const ark_bw6_761::G1Affine,
            scalars.as_ptr(),
            points.len(),
            &mut plain_result as *mut PlainG1Projective as *mut ark_bw6_761::G1Projective,
            core::mem::size_of::<PlainG1Affine>(),
            core::mem::size_of::<BigInt<6>>(),
        )
    };

    if status != 0 {
        return Err(Bw6GpuError::from_status(status));
    }

    // Handle infinity case
    if plain_result.infinity {
        return Ok(ark_bw6_761::G1Projective::default());
    }

    // Convert plain form coordinates back to Montgomery form
    // from_bigint() converts plain → Montgomery: x → x * R mod p
    let x_bigint = u32_array_to_bigint(&plain_result.x);
    let y_bigint = u32_array_to_bigint(&plain_result.y);
    let z_bigint = u32_array_to_bigint(&plain_result.z);

    // Convert BigInt to Fq (Montgomery form)
    let x_fq = Fq::from_bigint(x_bigint).ok_or(Bw6GpuError::Unknown(-100))?;
    let y_fq = Fq::from_bigint(y_bigint).ok_or(Bw6GpuError::Unknown(-101))?;
    let z_fq = Fq::from_bigint(z_bigint).ok_or(Bw6GpuError::Unknown(-102))?;

    // Construct G1Projective with Montgomery form coordinates
    Ok(ark_bw6_761::G1Projective::new_unchecked(x_fq, y_fq, z_fq))
}

#[cfg(all(feature = "gpu", not(bw6_cgbn_available)))]
pub fn msm_bw6_761_gpu_cgbn(
    _points: &[ark_bw6_761::G1Affine],
    _scalars: &[BigInt<6>],
) -> Result<ark_bw6_761::G1Projective, Bw6GpuError> {
    Err(Bw6GpuError::KernelUnavailable)
}

// ========== MNT4-298 GPU MSM Implementation ==========

#[cfg(feature = "gpu")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MntGpuError {
    AffineLayoutMismatch,
    ScalarLayoutMismatch,
    CudaRuntime,
    Timeout,
    KernelUnavailable,
    Unknown(i32),
}

#[cfg(feature = "gpu")]
impl MntGpuError {
    fn from_status(code: i32) -> Self {
        match code {
            0 => unreachable!(),
            -1 => MntGpuError::AffineLayoutMismatch,
            -2 => MntGpuError::ScalarLayoutMismatch,
            -3 => MntGpuError::CudaRuntime,
            -4 => MntGpuError::Timeout,
            -5 => MntGpuError::KernelUnavailable,
            other => MntGpuError::Unknown(other),
        }
    }
}

// Note: GpuMsm trait is NOT implemented for MNT4-298 because MNT4 and MNT6
// share the same underlying G1Affine type (they form a cycle pair), causing
// Rust trait coherence conflicts. Use msm_mnt4_298_gpu_cgbn() directly instead.

// CGBN-based MNT4-298 MSM kernel
#[cfg(all(feature = "gpu", mnt4_cgbn_available))]
pub fn msm_mnt4_298_gpu_cgbn(
    points: &[ark_mnt4_298::G1Affine],
    scalars: &[BigInt<5>],
) -> Result<ark_mnt4_298::G1Projective, MntGpuError> {
    use ark_ff::PrimeField;
    use ark_mnt4_298::Fq;

    if points.len() != scalars.len() {
        return Err(MntGpuError::Unknown(-99));
    }

    // CUDA expects uint32_t[10], but arkworks BigInt<5> is u64[5]
    // Convert from u64[5] to u32[10]
    #[repr(C)]
    struct PlainG1Affine {
        x: [u32; 10],
        y: [u32; 10],
        infinity: bool,
        _padding: [u8; 7],
    }

    #[repr(C)]
    struct PlainG1Projective {
        x: [u32; 10],
        y: [u32; 10],
        z: [u32; 10],
        infinity: bool,
        _padding: [u8; 7],
    }

    // Helper: convert BigInt<5> (u64[5]) to [u32; 10]
    fn bigint_to_u32_array(bigint: ark_ff::BigInt<5>) -> [u32; 10] {
        let mut result = [0u32; 10];
        for (i, &limb_u64) in bigint.0.iter().enumerate() {
            result[i * 2] = limb_u64 as u32;
            result[i * 2 + 1] = (limb_u64 >> 32) as u32;
        }
        result
    }

    // Helper: convert [u32; 10] back to BigInt<5>
    fn u32_array_to_bigint(arr: &[u32; 10]) -> ark_ff::BigInt<5> {
        let mut limbs = [0u64; 5];
        for i in 0..5 {
            limbs[i] = (arr[i * 2] as u64) | ((arr[i * 2 + 1] as u64) << 32);
        }
        ark_ff::BigInt(limbs)
    }

    let plain_points: Vec<PlainG1Affine> = points.iter().map(|p| {
        if p.infinity {
            PlainG1Affine {
                x: [0u32; 10],
                y: [0u32; 10],
                infinity: true,
                _padding: [0u8; 7],
            }
        } else {
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

    let mut plain_result = PlainG1Projective {
        x: [0u32; 10],
        y: [0u32; 10],
        z: [0u32; 10],
        infinity: false,
        _padding: [0u8; 7],
    };

    let status = unsafe {
        msm_mnt4_298_g1_cgbn(
            plain_points.as_ptr() as *const ark_mnt4_298::G1Affine,
            scalars.as_ptr(),
            points.len(),
            &mut plain_result as *mut PlainG1Projective as *mut ark_mnt4_298::G1Projective,
            core::mem::size_of::<PlainG1Affine>(),
            core::mem::size_of::<BigInt<5>>(),
        )
    };

    if status != 0 {
        return Err(MntGpuError::from_status(status));
    }

    if plain_result.infinity {
        return Ok(ark_mnt4_298::G1Projective::default());
    }

    let x_bigint = u32_array_to_bigint(&plain_result.x);
    let y_bigint = u32_array_to_bigint(&plain_result.y);
    let z_bigint = u32_array_to_bigint(&plain_result.z);

    let x_fq = Fq::from_bigint(x_bigint).ok_or(MntGpuError::Unknown(-100))?;
    let y_fq = Fq::from_bigint(y_bigint).ok_or(MntGpuError::Unknown(-101))?;
    let z_fq = Fq::from_bigint(z_bigint).ok_or(MntGpuError::Unknown(-102))?;

    Ok(ark_mnt4_298::G1Projective::new_unchecked(x_fq, y_fq, z_fq))
}

#[cfg(all(feature = "gpu", not(mnt4_cgbn_available)))]
pub fn msm_mnt4_298_gpu_cgbn(
    _points: &[ark_mnt4_298::G1Affine],
    _scalars: &[BigInt<5>],
) -> Result<ark_mnt4_298::G1Projective, MntGpuError> {
    Err(MntGpuError::KernelUnavailable)
}

// ========== MNT6-298 GPU MSM Implementation ==========

// Note: GpuMsm trait is NOT implemented for MNT6-298 because MNT4 and MNT6
// share the same underlying G1Affine type (they form a cycle pair), causing
// Rust trait coherence conflicts. Use msm_mnt6_298_gpu_cgbn() directly instead.

// CGBN-based MNT6-298 MSM kernel
#[cfg(all(feature = "gpu", mnt6_cgbn_available))]
pub fn msm_mnt6_298_gpu_cgbn(
    points: &[ark_mnt6_298::G1Affine],
    scalars: &[BigInt<5>],
) -> Result<ark_mnt6_298::G1Projective, MntGpuError> {
    use ark_ff::PrimeField;
    use ark_mnt6_298::Fq;

    if points.len() != scalars.len() {
        return Err(MntGpuError::Unknown(-99));
    }

    // CUDA expects uint32_t[10], but arkworks BigInt<5> is u64[5]
    #[repr(C)]
    struct PlainG1Affine {
        x: [u32; 10],
        y: [u32; 10],
        infinity: bool,
        _padding: [u8; 7],
    }

    #[repr(C)]
    struct PlainG1Projective {
        x: [u32; 10],
        y: [u32; 10],
        z: [u32; 10],
        infinity: bool,
        _padding: [u8; 7],
    }

    // Helper: convert BigInt<5> (u64[5]) to [u32; 10]
    fn bigint_to_u32_array(bigint: ark_ff::BigInt<5>) -> [u32; 10] {
        let mut result = [0u32; 10];
        for (i, &limb_u64) in bigint.0.iter().enumerate() {
            result[i * 2] = limb_u64 as u32;
            result[i * 2 + 1] = (limb_u64 >> 32) as u32;
        }
        result
    }

    // Helper: convert [u32; 10] back to BigInt<5>
    fn u32_array_to_bigint(arr: &[u32; 10]) -> ark_ff::BigInt<5> {
        let mut limbs = [0u64; 5];
        for i in 0..5 {
            limbs[i] = (arr[i * 2] as u64) | ((arr[i * 2 + 1] as u64) << 32);
        }
        ark_ff::BigInt(limbs)
    }

    let plain_points: Vec<PlainG1Affine> = points.iter().map(|p| {
        if p.infinity {
            PlainG1Affine {
                x: [0u32; 10],
                y: [0u32; 10],
                infinity: true,
                _padding: [0u8; 7],
            }
        } else {
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

    let mut plain_result = PlainG1Projective {
        x: [0u32; 10],
        y: [0u32; 10],
        z: [0u32; 10],
        infinity: false,
        _padding: [0u8; 7],
    };

    let status = unsafe {
        msm_mnt6_298_g1_cgbn(
            plain_points.as_ptr() as *const ark_mnt6_298::G1Affine,
            scalars.as_ptr(),
            points.len(),
            &mut plain_result as *mut PlainG1Projective as *mut ark_mnt6_298::G1Projective,
            core::mem::size_of::<PlainG1Affine>(),
            core::mem::size_of::<BigInt<5>>(),
        )
    };

    if status != 0 {
        return Err(MntGpuError::from_status(status));
    }

    if plain_result.infinity {
        return Ok(ark_mnt6_298::G1Projective::default());
    }

    let x_bigint = u32_array_to_bigint(&plain_result.x);
    let y_bigint = u32_array_to_bigint(&plain_result.y);
    let z_bigint = u32_array_to_bigint(&plain_result.z);

    let x_fq = Fq::from_bigint(x_bigint).ok_or(MntGpuError::Unknown(-100))?;
    let y_fq = Fq::from_bigint(y_bigint).ok_or(MntGpuError::Unknown(-101))?;
    let z_fq = Fq::from_bigint(z_bigint).ok_or(MntGpuError::Unknown(-102))?;

    Ok(ark_mnt6_298::G1Projective::new_unchecked(x_fq, y_fq, z_fq))
}

#[cfg(all(feature = "gpu", not(mnt6_cgbn_available)))]
pub fn msm_mnt6_298_gpu_cgbn(
    _points: &[ark_mnt6_298::G1Affine],
    _scalars: &[BigInt<5>],
) -> Result<ark_mnt6_298::G1Projective, MntGpuError> {
    Err(MntGpuError::KernelUnavailable)
}

// ========== Sparse Quotient GPU Coefficient Computation ==========

/// Error type for sparse quotient GPU operations
#[cfg(feature = "gpu")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseQuotientGpuError {
    CudaRuntime,
    InvalidInput,
    KernelUnavailable,
    Unknown(i32),
}

#[cfg(feature = "gpu")]
impl SparseQuotientGpuError {
    fn from_status(code: i32) -> Self {
        match code {
            0 => unreachable!(),
            -3 => SparseQuotientGpuError::CudaRuntime,
            -6 => SparseQuotientGpuError::InvalidInput,
            other => SparseQuotientGpuError::Unknown(other),
        }
    }
}

/// Sparse matrix in CSR format for GPU sparse quotient computation
#[cfg(feature = "gpu")]
pub struct SparseMatrixCsr {
    /// Column pointers: col_ptr[i] is the start index of column i in row_idx/values
    pub col_ptr: Vec<u32>,
    /// Row indices for each non-zero entry
    pub row_idx: Vec<u32>,
    /// Scalar values (as u32[10] arrays, 40 bytes each)
    pub values: Vec<[u32; 10]>,
}

/// Output from sparse quotient coefficient computation for one pair
#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct SparseQuotientPairOutput {
    /// Accumulated coefficients for col_a entries (indexed by local position in column)
    pub acc_u: Vec<[u32; 10]>,
    /// Accumulated coefficients for col_b entries (indexed by local position in column)
    pub acc_v: Vec<[u32; 10]>,
    /// Diagonal terms: (row_index, scalar_value)
    pub diag_terms: Vec<(u32, [u32; 10])>,
}

#[cfg(all(feature = "gpu", not(sppark_cuda_stub)))]
#[allow(improper_ctypes)]
extern "C" {
    fn sparse_quotient_coeffs_mnt4_298_gpu(
        // Sparse matrix A (CSR)
        col_a_ptr: *const u32,
        col_a_idx: *const u32,
        col_a_val: *const [u32; 10],
        num_cols_a: u32,
        nnz_a: u32,
        // Sparse matrix B (CSR)
        col_b_ptr: *const u32,
        col_b_idx: *const u32,
        col_b_val: *const [u32; 10],
        num_cols_b: u32,
        nnz_b: u32,
        // Domain element tables
        domain_elements: *const [u32; 10],
        inv_domain_elements: *const [u32; 10],
        inv_n_one_minus_omega: *const [u32; 10],
        domain_size: u32,
        // Pair batch
        pairs_i: *const u32,
        pairs_j: *const u32,
        num_pairs: u32,
        // Output sizing
        max_col_a: u32,
        max_col_b: u32,
        max_diag_per_pair: u32,
        // Output arrays
        out_acc_u: *mut [u32; 10],
        out_acc_v: *mut [u32; 10],
        out_diag_k: *mut u32,
        out_diag_val: *mut [u32; 10],
        out_num_diag: *mut u32,
    ) -> i32;

    fn sparse_quotient_mnt4_298_gpu_available() -> i32;
}

/// Check if sparse quotient GPU kernel is available
#[cfg(all(feature = "gpu", not(sppark_cuda_stub)))]
pub fn sparse_quotient_gpu_available() -> bool {
    unsafe { sparse_quotient_mnt4_298_gpu_available() != 0 }
}

#[cfg(any(not(feature = "gpu"), sppark_cuda_stub))]
pub fn sparse_quotient_gpu_available() -> bool {
    false
}

/// Compute sparse quotient coefficients on GPU for MNT4-298
///
/// This function computes the coefficient accumulation phase of sparse quotient
/// computation for a batch of (i,j) pairs. The coefficients are used to build
/// MSM tasks for computing H_{ij} bases.
///
/// # Arguments
/// * `col_a` - Sparse matrix A in CSR format (column i has entries col_a.values[col_a.col_ptr[i]..col_a.col_ptr[i+1]])
/// * `col_b` - Sparse matrix B in CSR format
/// * `domain_elements` - ω^k for k = 0..domain_size-1 (as u32[10] arrays)
/// * `inv_domain_elements` - ω^{-k} for k = 0..domain_size-1
/// * `inv_n_one_minus_omega` - precomputed inv(n * (1 - ω^d)) for d = 0..domain_size-1
/// * `domain_size` - size of evaluation domain
/// * `pairs` - batch of (i, j) pairs to process
/// * `max_col_a` - maximum entries per column in A (for output sizing)
/// * `max_col_b` - maximum entries per column in B (for output sizing)
/// * `max_diag_per_pair` - maximum diagonal terms per pair (typically min(max_col_a, max_col_b))
///
/// # Returns
/// Vector of SparseQuotientPairOutput, one per input pair
#[cfg(all(feature = "gpu", not(sppark_cuda_stub)))]
pub fn compute_sparse_quotient_coeffs_mnt4_298_gpu(
    col_a: &SparseMatrixCsr,
    col_b: &SparseMatrixCsr,
    domain_elements: &[[u32; 10]],
    inv_domain_elements: &[[u32; 10]],
    inv_n_one_minus_omega: &[[u32; 10]],
    domain_size: u32,
    pairs: &[(u32, u32)],
    max_col_a: u32,
    max_col_b: u32,
    max_diag_per_pair: u32,
) -> Result<Vec<SparseQuotientPairOutput>, SparseQuotientGpuError> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    let num_pairs = pairs.len() as u32;

    // Split pairs into separate arrays
    let pairs_i: Vec<u32> = pairs.iter().map(|(i, _)| *i).collect();
    let pairs_j: Vec<u32> = pairs.iter().map(|(_, j)| *j).collect();

    // Allocate output buffers
    let mut out_acc_u = vec![[0u32; 10]; (num_pairs as usize) * (max_col_a as usize)];
    let mut out_acc_v = vec![[0u32; 10]; (num_pairs as usize) * (max_col_b as usize)];
    let mut out_diag_k = vec![0u32; (num_pairs as usize) * (max_diag_per_pair as usize)];
    let mut out_diag_val = vec![[0u32; 10]; (num_pairs as usize) * (max_diag_per_pair as usize)];
    let mut out_num_diag = vec![0u32; num_pairs as usize];

    let status = unsafe {
        sparse_quotient_coeffs_mnt4_298_gpu(
            col_a.col_ptr.as_ptr(),
            col_a.row_idx.as_ptr(),
            col_a.values.as_ptr(),
            (col_a.col_ptr.len() - 1) as u32,
            col_a.values.len() as u32,
            col_b.col_ptr.as_ptr(),
            col_b.row_idx.as_ptr(),
            col_b.values.as_ptr(),
            (col_b.col_ptr.len() - 1) as u32,
            col_b.values.len() as u32,
            domain_elements.as_ptr(),
            inv_domain_elements.as_ptr(),
            inv_n_one_minus_omega.as_ptr(),
            domain_size,
            pairs_i.as_ptr(),
            pairs_j.as_ptr(),
            num_pairs,
            max_col_a,
            max_col_b,
            max_diag_per_pair,
            out_acc_u.as_mut_ptr(),
            out_acc_v.as_mut_ptr(),
            out_diag_k.as_mut_ptr(),
            out_diag_val.as_mut_ptr(),
            out_num_diag.as_mut_ptr(),
        )
    };

    if status != 0 {
        return Err(SparseQuotientGpuError::from_status(status));
    }

    // Convert output buffers to per-pair results
    let mut results = Vec::with_capacity(pairs.len());

    for pair_idx in 0..pairs.len() {
        let (i, j) = pairs[pair_idx];

        // Get column sizes from CSR structure
        let n_u = (col_a.col_ptr[i as usize + 1] - col_a.col_ptr[i as usize]) as usize;
        let n_v = (col_b.col_ptr[j as usize + 1] - col_b.col_ptr[j as usize]) as usize;

        // Extract acc_u for this pair
        let acc_u_start = pair_idx * (max_col_a as usize);
        let acc_u: Vec<[u32; 10]> = out_acc_u[acc_u_start..acc_u_start + n_u].to_vec();

        // Extract acc_v for this pair
        let acc_v_start = pair_idx * (max_col_b as usize);
        let acc_v: Vec<[u32; 10]> = out_acc_v[acc_v_start..acc_v_start + n_v].to_vec();

        // Extract diagonal terms for this pair
        let num_diag = out_num_diag[pair_idx] as usize;
        let diag_start = pair_idx * (max_diag_per_pair as usize);
        let diag_terms: Vec<(u32, [u32; 10])> = (0..num_diag)
            .map(|d| (out_diag_k[diag_start + d], out_diag_val[diag_start + d]))
            .collect();

        results.push(SparseQuotientPairOutput {
            acc_u,
            acc_v,
            diag_terms,
        });
    }

    Ok(results)
}

#[cfg(all(feature = "gpu", sppark_cuda_stub))]
pub fn compute_sparse_quotient_coeffs_mnt4_298_gpu(
    _col_a: &SparseMatrixCsr,
    _col_b: &SparseMatrixCsr,
    _domain_elements: &[[u32; 10]],
    _inv_domain_elements: &[[u32; 10]],
    _inv_n_one_minus_omega: &[[u32; 10]],
    _domain_size: u32,
    _pairs: &[(u32, u32)],
    _max_col_a: u32,
    _max_col_b: u32,
    _max_diag_per_pair: u32,
) -> Result<Vec<SparseQuotientPairOutput>, SparseQuotientGpuError> {
    Err(SparseQuotientGpuError::KernelUnavailable)
}
