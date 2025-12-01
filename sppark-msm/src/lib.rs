#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

#[cfg(feature = "gpu")]
pub use ark_ec::AffineRepr;
#[cfg(feature = "gpu")]
pub use ark_ff::BigInt;

#[cfg(all(feature = "gpu", not(sppark_cuda_stub)))]
#[allow(improper_ctypes)]
extern "C" {
    fn msm_bls12_377_g1(
        points: *const ark_bls12_377::G1Affine,
        scalars: *const BigInt<4>,
        count: usize,
        result: *mut ark_bls12_377::G1Projective,
        ffi_affine_sz: usize,
    ) -> i32;
}

#[cfg(all(feature = "gpu", sppark_cuda_stub))]
unsafe fn msm_bls12_377_g1(
    _points: *const ark_bls12_377::G1Affine,
    _scalars: *const BigInt<4>,
    _count: usize,
    _result: *mut ark_bls12_377::G1Projective,
    _ffi_affine_sz: usize,
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
            );
            if status != 0 {
                return Err(());
            }
        }
        Ok(result)
    }
}
