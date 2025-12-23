// BW6-761 MSM implementation (generic - disabled by default)
// NOTE: This generic kernel is disabled by default due to register pressure issues.
// Use the specialized kernel (msm_bw6_761_specialized.cu) instead.
#include <stddef.h>

// Prevent sppark from instantiating templates - we'll do it ourselves
#define SPPARK_DONT_INSTANTIATE_TEMPLATES

#include "ff/bw6-761.hpp"
#include "ec/jacobian_t.hpp"
#include "ec/xyzz_t.hpp"
#include "msm/pippenger.cuh"

// BW6-761 G1 MSM
extern "C" int msm_bw6_761_g1(
    const void* points,
    const void* scalars,
    size_t count,
    void* result,
    size_t ffi_affine_sz,
    size_t ffi_scalar_sz
) {
    using namespace bw6_761;
    using point_t = jacobian_t<fp_t>;
    using bucket_t = xyzz_t<fp_t>;
    using affine_t = bucket_t::affine_inf_t;
    using scalar_t = fr_t;

    // CRITICAL SAFETY CHECKS: Verify arkworks layout matches sppark expectations
    // If arkworks changes G1Affine or BigInt<6> layout, this catches the mismatch
    if (ffi_affine_sz != sizeof(affine_t)) {
        // Size mismatch - arkworks G1Affine doesn't match sppark affine_inf_t
        static_cast<point_t*>(result)->inf();
        return -1;  // Error: affine layout mismatch
    }

    if (ffi_scalar_sz != sizeof(scalar_t)) {
        // Size mismatch - arkworks BigInt<6> doesn't match sppark fr_t
        static_cast<point_t*>(result)->inf();
        return -2;  // Error: scalar layout mismatch
    }

    RustError err = mult_pippenger<bucket_t>(
        static_cast<point_t*>(result),
        reinterpret_cast<const affine_t*>(points),
        count,
        reinterpret_cast<const scalar_t*>(scalars),
        false,
        ffi_affine_sz
    );

    if (err.code != 0) {
        static_cast<point_t*>(result)->inf();
    }

    return err.code;
}
