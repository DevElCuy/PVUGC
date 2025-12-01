//! MSM Backend switcher
use ark_ec::{AffineRepr, VariableBaseMSM};
use ark_ff::PrimeField;
#[cfg(feature = "gpu")]
use std::any::TypeId;

#[cfg(feature = "gpu")]
use sppark_msm::GpuMsm;

#[cfg(feature = "gpu")]
fn try_gpu_msm<G>(
    bases: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> Option<G::Group>
where
    G: AffineRepr + 'static,
    G::Group: 'static,
    G::ScalarField: 'static,
    <G::ScalarField as PrimeField>::BigInt: 'static,
{
    use ark_bls12_377::G1Affine as Bls12_377_G1;
    use ark_ff::BigInt;
    use std::mem;

    // Cast a slice between concrete curve types once the TypeId check passes.
    unsafe fn cast_slice<T, U>(slice: &[T]) -> &[U] {
        debug_assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
        std::slice::from_raw_parts(slice.as_ptr() as *const U, slice.len())
    }

    // Move a concrete projective value into the generic group type.
    unsafe fn cast_group<T, U>(value: T) -> U {
        debug_assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
        std::ptr::read(&value as *const T as *const U)
    }

    let g_id = TypeId::of::<G>();
    let scalar_id = TypeId::of::<<G::ScalarField as PrimeField>::BigInt>();

    if g_id == TypeId::of::<Bls12_377_G1>() && scalar_id == TypeId::of::<BigInt<4>>() {
        let result =
            Bls12_377_G1::msm_gpu(unsafe { cast_slice(bases) }, unsafe { cast_slice(scalars) })
                .ok()?;
        return Some(unsafe { cast_group(result) });
    }

    None
}

pub fn msm_g1<G>(bases: &[G], scalars: &[<G::ScalarField as PrimeField>::BigInt]) -> G::Group
where
    G: AffineRepr + 'static,
    G::Group: 'static,
    G::ScalarField: 'static,
    <G::ScalarField as PrimeField>::BigInt: 'static,
{
    #[cfg(feature = "gpu")]
    {
        if let Some(result) = try_gpu_msm::<G>(bases, scalars) {
            return result;
        }
    }

    // Fallback to CPU
    G::Group::msm_bigint(bases, scalars)
}
