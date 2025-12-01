// BW6-761 MSM implementation
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
    size_t ffi_affine_sz
) {
    using namespace bw6_761;
    using point_t = jacobian_t<fp_t>;
    using bucket_t = xyzz_t<fp_t>;
    using affine_t = bucket_t::affine_inf_t;
    using scalar_t = fr_t;

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
