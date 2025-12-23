// BW6-761 Specialized MSM Implementation
// Optimized for large (761-bit) fields with aggressive register management
#include <stddef.h>
#include <cuda_runtime.h>

// Prevent sppark from instantiating templates
#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#define TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE

// Resource configuration for 761-bit field (96-byte elements)
// NOTE: defaults can be overridden via build.rs/env defines.
#ifndef MSM_WBITS
#define MSM_WBITS 6           // Window size: 2^6 = 64 buckets (conservative)
#endif
#ifndef MSM_NTHREADS
#define MSM_NTHREADS 32       // Threads per block (conservative)
#endif
#ifndef MSM_NSTREAMS
#define MSM_NSTREAMS 2        // Minimal parallel streams
#endif

// Error codes for explicit propagation
#define BW6_MSM_SUCCESS 0
#define BW6_MSM_ERROR_AFFINE_LAYOUT -1
#define BW6_MSM_ERROR_SCALAR_LAYOUT -2
#define BW6_MSM_ERROR_CUDA_RUNTIME -3
#define BW6_MSM_ERROR_TIMEOUT -4

#include "ff/bw6-761.hpp"
#include "ec/jacobian_t.hpp"
#include "ec/xyzz_t.hpp"

// Custom includes for large-field optimization
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "msm/pippenger.cuh"

// BW6-761 G1 MSM - Specialized for large fields
// Note: Resource limits (MSM_WBITS, MSM_NTHREADS, --maxrregcount) affect
// the kernel functions inside pippenger.cuh
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

    // Diagnostic logging (enable with BW6_DEBUG)
#ifdef BW6_DEBUG
    printf("[BW6-761] Entering msm_bw6_761_g1, count=%zu\n", count);
    printf("[BW6-761] FFI sizes: affine=%zu, scalar=%zu\n", ffi_affine_sz, ffi_scalar_sz);
    printf("[BW6-761] Expected sizes: affine=%zu, scalar=%zu\n", sizeof(affine_t), sizeof(scalar_t));
#endif

    // CRITICAL SAFETY CHECKS: Verify arkworks layout matches sppark expectations
    if (ffi_affine_sz != sizeof(affine_t)) {
        return BW6_MSM_ERROR_AFFINE_LAYOUT;  // Error: affine layout mismatch
    }

    if (ffi_scalar_sz != sizeof(scalar_t)) {
        return BW6_MSM_ERROR_SCALAR_LAYOUT;  // Error: scalar layout mismatch
    }

#ifdef BW6_DEBUG
    printf("[BW6-761] FFI layout checks passed\n");
#endif

    // Handle edge cases
    if (count == 0) {
        static_cast<point_t*>(result)->inf();
        return 0;
    }

    // Known hang on single-point path for BW6-761; return explicit timeout so
    // the host can fall back to a CPU implementation without blocking here.
    if (count == 1) {
        static_cast<point_t*>(result)->inf();
        return BW6_MSM_ERROR_TIMEOUT;
    }

#ifdef BW6_DEBUG
    printf("[BW6-761] Calling mult_pippenger with count=%zu\n", count);
    printf("[BW6-761] Configuration: MSM_WBITS=%d, MSM_NTHREADS=%d, MSM_NSTREAMS=%d\n", MSM_WBITS, MSM_NTHREADS, MSM_NSTREAMS);
#endif

    // CRITICAL: Increase CUDA stack size to handle large stack frames in BW6-761 point arithmetic
    // Point addition functions require 10-25KB stack frames due to 761-bit field operations
    // Default CUDA stack is ~1KB/thread, we need much more for deep call stacks
    cudaError_t stack_err = cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024);
    if (stack_err != cudaSuccess) {
#ifdef BW6_DEBUG
        printf("[BW6-761] WARNING: Failed to set stack size: %s\n", cudaGetErrorString(stack_err));
#endif
        return BW6_MSM_ERROR_CUDA_RUNTIME;
    }

    // Force a CUDA device synchronization to ensure any previous errors are cleared
    cudaError_t sync_before = cudaDeviceSynchronize();
    if (sync_before != cudaSuccess) {
#ifdef BW6_DEBUG
        printf("[BW6-761] CUDA sync before mult_pippenger failed: %s\n", cudaGetErrorString(sync_before));
#endif
        return BW6_MSM_ERROR_CUDA_RUNTIME;
    }

    // Call Pippenger MSM with specialized configuration
    // The MSM_WBITS, MSM_NTHREADS, MSM_NSTREAMS defines above will be used
    RustError err = mult_pippenger<bucket_t>(
        static_cast<point_t*>(result),
        reinterpret_cast<const affine_t*>(points),
        count,
        reinterpret_cast<const scalar_t*>(scalars),
        false,  // not Montgomery form (arkworks uses standard form)
        ffi_affine_sz
    );

    // Check for any CUDA errors after mult_pippenger
    cudaError_t cuda_err = cudaGetLastError();

#ifdef BW6_DEBUG
    printf("[BW6-761] mult_pippenger returned, err.code=%d, CUDA error: %s\n",
           err.code, cudaGetErrorString(cuda_err));
#endif

    if (cuda_err != cudaSuccess) {
        return BW6_MSM_ERROR_CUDA_RUNTIME;
    }

    if (err.code != 0) {
        return BW6_MSM_ERROR_CUDA_RUNTIME;
    }

    // Success
    return BW6_MSM_SUCCESS;
}
