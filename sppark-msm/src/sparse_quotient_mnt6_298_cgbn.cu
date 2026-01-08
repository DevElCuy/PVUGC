// MNT6-298 CGBN-Based Sparse Quotient Coefficient Computation
//
// This kernel accelerates the coefficient accumulation phase of sparse quotient
// computation in the PVUGC setup for MNT6-298 outer curve.
//
// This is essentially identical to the MNT4-298 version, but uses MNT6-298's
// scalar field modulus (which equals MNT4-298's base field modulus).

#include <stddef.h>
#include <cuda_runtime.h>

// Include gmp.h BEFORE cgbn.h to avoid cgbn_cpu.h stub
#include <gmp.h>
#include <cgbn/cgbn.h>

// Error codes
#define SPARSE_QUOTIENT_SUCCESS 0
#define SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME -3
#define SPARSE_QUOTIENT_ERROR_INVALID_INPUT -6

// CGBN parameters for MNT6-298 scalar field (298 bits, padded to 320)
class sparse_quotient_mnt6_cgbn_params_t {
public:
    static const uint32_t TPB = 0;
    static const uint32_t MAX_ROTATION = 4;
    static const uint32_t SHM_LIMIT = 0;
    static const bool CONSTANT_TIME = false;
    static const uint32_t TPI = 8;
    static const uint32_t BITS = 320;
};

// MNT6-298 scalar field modulus (Fr, 298 bits) == MNT4-298 base field (Fq)
// Little-endian u32[10] representation
__device__ __constant__ uint32_t MNT6_298_FR_DEVICE[10] = {
    0x71660001, 0xc90cd65a, 0x51200e12, 0x41a9e35e, 0x5d1330ea,
    0xcaeec963, 0xa7b0548e, 0xa266249d, 0xf7bcd473, 0x000003bc
};

// Scalar type (Fr element, 298 bits = 40 bytes in 10 u32 limbs)
typedef struct {
    uint32_t limbs[10];
} __align__(8) scalar_mnt6_t;

/**
 * Sparse Quotient Coefficient Kernel Class for MNT6-298
 */
template<class params>
class sparse_quotient_mnt6_kernel_t {
public:
    typedef cgbn_context_t<params::TPI, params> context_t;
    typedef cgbn_env_t<context_t, params::BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;

    context_t _context;
    env_t     _env;
    int32_t   _instance;

    __device__ __forceinline__ sparse_quotient_mnt6_kernel_t(
        cgbn_monitor_t monitor,
        cgbn_error_report_t* report,
        int32_t instance
    ) : _context(monitor, report, (uint32_t)instance),
        _env(_context),
        _instance(instance) {}

    __device__ __forceinline__ void field_add(bn_t& r, const bn_t& a,
                                               const bn_t& b, const bn_t& Fr) {
        cgbn_add(_env, r, a, b);
        if (cgbn_compare(_env, r, Fr) >= 0) {
            cgbn_sub(_env, r, r, Fr);
        }
    }

    __device__ __forceinline__ void field_sub(bn_t& r, const bn_t& a,
                                               const bn_t& b, const bn_t& Fr) {
        int32_t borrow = cgbn_sub(_env, r, a, b);
        if (borrow != 0) {
            cgbn_add(_env, r, r, Fr);
        }
    }

    __device__ __forceinline__ void field_mul(bn_t& r, const bn_t& a,
                                               const bn_t& b, const bn_t& Fr) {
        typedef typename env_t::cgbn_wide_t wide_t;
        wide_t product;
        cgbn_mul_wide(_env, product, a, b);
        cgbn_rem_wide(_env, r, product, Fr);
    }

    __device__ __forceinline__ void field_neg(bn_t& r, const bn_t& a, const bn_t& Fr) {
        if (cgbn_equals_ui32(_env, a, 0)) {
            cgbn_set_ui32(_env, r, 0);
        } else {
            cgbn_sub(_env, r, Fr, a);
        }
    }

    __device__ __forceinline__ void load_scalar(bn_t& r, const scalar_mnt6_t* s) {
        cgbn_load(_env, r, (cgbn_mem_t<320>*)s->limbs);
    }

    __device__ __forceinline__ void store_scalar(scalar_mnt6_t* s, const bn_t& r) {
        cgbn_store(_env, (cgbn_mem_t<320>*)s->limbs, r);
    }

    __device__ __forceinline__ void load_modulus(bn_t& Fr) {
        cgbn_load(_env, Fr, (cgbn_mem_t<320>*)MNT6_298_FR_DEVICE);
    }
};

/**
 * Compute coefficients for a batch of (i, j) pairs - MNT6-298 version
 *
 * Two-phase algorithm to avoid race conditions:
 *   Phase 1: Each instance owns idx_u values, loops over all idx_v to compute acc_u
 *   Phase 2: Each instance owns idx_v values, loops over all idx_u to compute acc_v
 */
__global__ void compute_quotient_coeffs_mnt6_kernel(
    const uint32_t* col_a_ptr,
    const uint32_t* col_a_idx,
    const scalar_mnt6_t* col_a_val,
    const uint32_t* col_b_ptr,
    const uint32_t* col_b_idx,
    const scalar_mnt6_t* col_b_val,
    const scalar_mnt6_t* domain_elements,      // omega^d * inv(n * (1 - omega^d)) for d = 0..n-1 (precomputed)
    const scalar_mnt6_t* inv_domain_elements,  // unused (kept for ABI)
    const scalar_mnt6_t* inv_n_one_minus_omega, // inv(n * (1 - omega^d)) for d = 0..n-1
    uint32_t domain_size,
    const uint32_t* pairs_i,
    const uint32_t* pairs_j,
    uint32_t num_pairs,
    scalar_mnt6_t* out_acc_u,
    scalar_mnt6_t* out_acc_v,
    uint32_t max_col_a,
    uint32_t max_col_b,
    uint32_t* out_diag_k,
    scalar_mnt6_t* out_diag_val,
    uint32_t* out_num_diag,
    uint32_t max_diag_per_pair,
    cgbn_error_report_t* report
) {
    uint32_t pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    constexpr uint32_t TPI = sparse_quotient_mnt6_cgbn_params_t::TPI;
    (void)inv_domain_elements;

    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;

    sparse_quotient_mnt6_kernel_t<sparse_quotient_mnt6_cgbn_params_t> kernel(
        cgbn_no_checks, report, instance
    );

    typedef sparse_quotient_mnt6_kernel_t<sparse_quotient_mnt6_cgbn_params_t>::bn_t bn_t;

    bn_t Fr;
    kernel.load_modulus(Fr);

    uint32_t i = pairs_i[pair_idx];
    uint32_t j = pairs_j[pair_idx];

    uint32_t u_start = col_a_ptr[i];
    uint32_t u_end = col_a_ptr[i + 1];
    uint32_t v_start = col_b_ptr[j];
    uint32_t v_end = col_b_ptr[j + 1];

    uint32_t n_u = u_end - u_start;
    uint32_t n_v = v_end - v_start;

    if (n_u == 0 || n_v == 0) {
        if (threadIdx.x == 0) {
            out_num_diag[pair_idx] = 0;
        }
        return;
    }

    uint32_t instances_per_block = blockDim.x / TPI;
    uint32_t local_instance = threadIdx.x / TPI;
    uint32_t lane = threadIdx.x % TPI;

    // Initialize outputs to zero
    if (lane == 0) {
        for (uint32_t idx = local_instance; idx < n_u; idx += instances_per_block) {
            scalar_mnt6_t* out = &out_acc_u[pair_idx * max_col_a + idx];
            for (int l = 0; l < 10; l++) out->limbs[l] = 0;
        }
        for (uint32_t idx = local_instance; idx < n_v; idx += instances_per_block) {
            scalar_mnt6_t* out = &out_acc_v[pair_idx * max_col_b + idx];
            for (int l = 0; l < 10; l++) out->limbs[l] = 0;
        }
    }
    __syncthreads();

    __shared__ uint32_t diag_count;
    if (threadIdx.x == 0) {
        diag_count = 0;
    }
    __syncthreads();

    // Helper variables for instance-scoped shuffle
    uint32_t lane_in_warp = threadIdx.x & 31;
    uint32_t group_thread = threadIdx.x % TPI;
    uint32_t instance_mask = ((1u << TPI) - 1u) << (lane_in_warp - group_thread);

    // ========================================================================
    // PHASE 1: Compute acc_u (and collect diagonal terms)
    // Each instance owns specific idx_u values, loops over ALL idx_v
    // ========================================================================
    for (uint32_t idx_u = local_instance; idx_u < n_u; idx_u += instances_per_block) {
        uint32_t k = col_a_idx[u_start + idx_u];
        bn_t val_u;
        kernel.load_scalar(val_u, &col_a_val[u_start + idx_u]);

        // Accumulator for this idx_u (local to this instance, no races)
        bn_t acc_u_local;
        cgbn_set_ui32(kernel._env, acc_u_local, 0);

        // Loop over all idx_v
        for (uint32_t idx_v = 0; idx_v < n_v; idx_v++) {
            uint32_t m = col_b_idx[v_start + idx_v];
            bn_t val_v;
            kernel.load_scalar(val_v, &col_b_val[v_start + idx_v]);

            bn_t prod;
            kernel.field_mul(prod, val_u, val_v, Fr);

            if (k == m) {
                // Diagonal term: store (k, prod)
                uint32_t diag_slot;
                if (lane == 0) {
                    diag_slot = atomicAdd(&diag_count, 1);
                }
                diag_slot = __shfl_sync(instance_mask, diag_slot, 0, TPI);

                if (diag_slot < max_diag_per_pair) {
                    if (lane == 0) {
                        out_diag_k[pair_idx * max_diag_per_pair + diag_slot] = k;
                    }
                    kernel.store_scalar(&out_diag_val[pair_idx * max_diag_per_pair + diag_slot], prod);
                }
            } else {
                // Off-diagonal term: accumulate contribution to acc_u.
                // Simplified: inv_wm * wm cancels, so acc_u -= prod * inv_n_one_minus_omega[d].
                uint32_t d = (k >= m) ? (k - m) : (k + domain_size - m);

                bn_t inv_n_term, tmp;

                kernel.load_scalar(inv_n_term, &inv_n_one_minus_omega[d]);

                kernel.field_mul(tmp, prod, inv_n_term, Fr);
                kernel.field_sub(acc_u_local, acc_u_local, tmp, Fr);
            }
        }

        // Store accumulated result for this idx_u
        kernel.store_scalar(&out_acc_u[pair_idx * max_col_a + idx_u], acc_u_local);
    }

    __syncthreads();

    // ========================================================================
    // PHASE 2: Compute acc_v
    // Each instance owns specific idx_v values, loops over ALL idx_u
    // ========================================================================
    for (uint32_t idx_v = local_instance; idx_v < n_v; idx_v += instances_per_block) {
        uint32_t m = col_b_idx[v_start + idx_v];
        bn_t val_v;
        kernel.load_scalar(val_v, &col_b_val[v_start + idx_v]);

        // Accumulator for this idx_v (local to this instance, no races)
        bn_t acc_v_local;
        cgbn_set_ui32(kernel._env, acc_v_local, 0);

        // Loop over all idx_u
        for (uint32_t idx_u = 0; idx_u < n_u; idx_u++) {
            uint32_t k = col_a_idx[u_start + idx_u];

            // Skip diagonal terms (already handled in phase 1)
            if (k == m) continue;

            bn_t val_u;
            kernel.load_scalar(val_u, &col_a_val[u_start + idx_u]);

            bn_t prod;
            kernel.field_mul(prod, val_u, val_v, Fr);

            // Off-diagonal term: accumulate contribution to acc_v
            uint32_t d = (k >= m) ? (k - m) : (k + domain_size - m);

            bn_t coeff, contrib;

            // domain_elements[d] contains precomputed omega^d * inv_n_one_minus_omega[d]
            kernel.load_scalar(coeff, &domain_elements[d]);
            kernel.field_mul(contrib, prod, coeff, Fr);
            kernel.field_add(acc_v_local, acc_v_local, contrib, Fr);
        }

        // Store accumulated result for this idx_v
        kernel.store_scalar(&out_acc_v[pair_idx * max_col_b + idx_v], acc_v_local);
    }

    __syncthreads();

    // Write diagonal count
    if (threadIdx.x == 0) {
        out_num_diag[pair_idx] = diag_count < max_diag_per_pair ? diag_count : max_diag_per_pair;
    }
}

// C interface for FFI
extern "C" {

int sparse_quotient_coeffs_mnt6_298_gpu(
    const uint32_t* col_a_ptr,
    const uint32_t* col_a_idx,
    const void* col_a_val,
    uint32_t num_cols_a,
    uint32_t nnz_a,
    const uint32_t* col_b_ptr,
    const uint32_t* col_b_idx,
    const void* col_b_val,
    uint32_t num_cols_b,
    uint32_t nnz_b,
    const void* domain_elements,
    const void* inv_domain_elements,
    const void* inv_n_one_minus_omega,
    uint32_t domain_size,
    const uint32_t* pairs_i,
    const uint32_t* pairs_j,
    uint32_t num_pairs,
    uint32_t max_col_a,
    uint32_t max_col_b,
    uint32_t max_diag_per_pair,
    void* out_acc_u,
    void* out_acc_v,
    uint32_t* out_diag_k,
    void* out_diag_val,
    uint32_t* out_num_diag
) {
    if (num_pairs == 0) return SPARSE_QUOTIENT_SUCCESS;

    cudaError_t err;
    uint32_t threads_per_block = 0;
    uint32_t num_blocks = 0;

    // Allocate device memory
    uint32_t* d_col_a_ptr;
    uint32_t* d_col_a_idx;
    scalar_mnt6_t* d_col_a_val;
    uint32_t* d_col_b_ptr;
    uint32_t* d_col_b_idx;
    scalar_mnt6_t* d_col_b_val;

    err = cudaMalloc(&d_col_a_ptr, (num_cols_a + 1) * sizeof(uint32_t));
    if (err != cudaSuccess) return SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME;
    err = cudaMalloc(&d_col_a_idx, nnz_a * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_col_a_ptr); return SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME; }
    err = cudaMalloc(&d_col_a_val, nnz_a * sizeof(scalar_mnt6_t));
    if (err != cudaSuccess) { cudaFree(d_col_a_ptr); cudaFree(d_col_a_idx); return SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME; }

    err = cudaMalloc(&d_col_b_ptr, (num_cols_b + 1) * sizeof(uint32_t));
    if (err != cudaSuccess) goto cleanup_a;
    err = cudaMalloc(&d_col_b_idx, nnz_b * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_col_b_ptr); goto cleanup_a; }
    err = cudaMalloc(&d_col_b_val, nnz_b * sizeof(scalar_mnt6_t));
    if (err != cudaSuccess) { cudaFree(d_col_b_ptr); cudaFree(d_col_b_idx); goto cleanup_a; }

    scalar_mnt6_t* d_domain_elements;
    scalar_mnt6_t* d_inv_domain_elements;
    scalar_mnt6_t* d_inv_n_one_minus_omega;

    err = cudaMalloc(&d_domain_elements, domain_size * sizeof(scalar_mnt6_t));
    if (err != cudaSuccess) goto cleanup_b;
    err = cudaMalloc(&d_inv_domain_elements, domain_size * sizeof(scalar_mnt6_t));
    if (err != cudaSuccess) { cudaFree(d_domain_elements); goto cleanup_b; }
    err = cudaMalloc(&d_inv_n_one_minus_omega, domain_size * sizeof(scalar_mnt6_t));
    if (err != cudaSuccess) { cudaFree(d_domain_elements); cudaFree(d_inv_domain_elements); goto cleanup_b; }

    uint32_t* d_pairs_i;
    uint32_t* d_pairs_j;
    err = cudaMalloc(&d_pairs_i, num_pairs * sizeof(uint32_t));
    if (err != cudaSuccess) goto cleanup_domain;
    err = cudaMalloc(&d_pairs_j, num_pairs * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_pairs_i); goto cleanup_domain; }

    scalar_mnt6_t* d_out_acc_u;
    scalar_mnt6_t* d_out_acc_v;
    uint32_t* d_out_diag_k;
    scalar_mnt6_t* d_out_diag_val;
    uint32_t* d_out_num_diag;

    err = cudaMalloc(&d_out_acc_u, num_pairs * max_col_a * sizeof(scalar_mnt6_t));
    if (err != cudaSuccess) goto cleanup_pairs;
    err = cudaMalloc(&d_out_acc_v, num_pairs * max_col_b * sizeof(scalar_mnt6_t));
    if (err != cudaSuccess) { cudaFree(d_out_acc_u); goto cleanup_pairs; }
    err = cudaMalloc(&d_out_diag_k, num_pairs * max_diag_per_pair * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_out_acc_u); cudaFree(d_out_acc_v); goto cleanup_pairs; }
    err = cudaMalloc(&d_out_diag_val, num_pairs * max_diag_per_pair * sizeof(scalar_mnt6_t));
    if (err != cudaSuccess) { cudaFree(d_out_acc_u); cudaFree(d_out_acc_v); cudaFree(d_out_diag_k); goto cleanup_pairs; }
    err = cudaMalloc(&d_out_num_diag, num_pairs * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_out_acc_u); cudaFree(d_out_acc_v); cudaFree(d_out_diag_k); cudaFree(d_out_diag_val); goto cleanup_pairs; }

    cgbn_error_report_t* d_report;
    err = cudaMalloc(&d_report, sizeof(cgbn_error_report_t));
    if (err != cudaSuccess) goto cleanup_output;

    // Copy input data
    cudaMemcpy(d_col_a_ptr, col_a_ptr, (num_cols_a + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_a_idx, col_a_idx, nnz_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_a_val, col_a_val, nnz_a * sizeof(scalar_mnt6_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_b_ptr, col_b_ptr, (num_cols_b + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_b_idx, col_b_idx, nnz_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_b_val, col_b_val, nnz_b * sizeof(scalar_mnt6_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domain_elements, domain_elements, domain_size * sizeof(scalar_mnt6_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_domain_elements, inv_domain_elements, domain_size * sizeof(scalar_mnt6_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_n_one_minus_omega, inv_n_one_minus_omega, domain_size * sizeof(scalar_mnt6_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs_i, pairs_i, num_pairs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs_j, pairs_j, num_pairs * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMemset(d_report, 0, sizeof(cgbn_error_report_t));

    // Launch kernel
    threads_per_block = 256;
    num_blocks = num_pairs;

    compute_quotient_coeffs_mnt6_kernel<<<num_blocks, threads_per_block>>>(
        d_col_a_ptr, d_col_a_idx, d_col_a_val,
        d_col_b_ptr, d_col_b_idx, d_col_b_val,
        d_domain_elements, d_inv_domain_elements, d_inv_n_one_minus_omega,
        domain_size,
        d_pairs_i, d_pairs_j, num_pairs,
        d_out_acc_u, d_out_acc_v, max_col_a, max_col_b,
        d_out_diag_k, d_out_diag_val, d_out_num_diag, max_diag_per_pair,
        d_report
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup_all;

    // Copy results back
    cudaMemcpy(out_acc_u, d_out_acc_u, num_pairs * max_col_a * sizeof(scalar_mnt6_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_acc_v, d_out_acc_v, num_pairs * max_col_b * sizeof(scalar_mnt6_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_diag_k, d_out_diag_k, num_pairs * max_diag_per_pair * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_diag_val, d_out_diag_val, num_pairs * max_diag_per_pair * sizeof(scalar_mnt6_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_num_diag, d_out_num_diag, num_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_report);
    cudaFree(d_out_num_diag);
    cudaFree(d_out_diag_val);
    cudaFree(d_out_diag_k);
    cudaFree(d_out_acc_v);
    cudaFree(d_out_acc_u);
    cudaFree(d_pairs_j);
    cudaFree(d_pairs_i);
    cudaFree(d_inv_n_one_minus_omega);
    cudaFree(d_inv_domain_elements);
    cudaFree(d_domain_elements);
    cudaFree(d_col_b_val);
    cudaFree(d_col_b_idx);
    cudaFree(d_col_b_ptr);
    cudaFree(d_col_a_val);
    cudaFree(d_col_a_idx);
    cudaFree(d_col_a_ptr);

    return SPARSE_QUOTIENT_SUCCESS;

cleanup_all:
    cudaFree(d_report);
cleanup_output:
    cudaFree(d_out_num_diag);
    cudaFree(d_out_diag_val);
    cudaFree(d_out_diag_k);
    cudaFree(d_out_acc_v);
    cudaFree(d_out_acc_u);
cleanup_pairs:
    cudaFree(d_pairs_j);
    cudaFree(d_pairs_i);
cleanup_domain:
    cudaFree(d_inv_n_one_minus_omega);
    cudaFree(d_inv_domain_elements);
    cudaFree(d_domain_elements);
cleanup_b:
    cudaFree(d_col_b_val);
    cudaFree(d_col_b_idx);
    cudaFree(d_col_b_ptr);
cleanup_a:
    cudaFree(d_col_a_val);
    cudaFree(d_col_a_idx);
    cudaFree(d_col_a_ptr);

    return SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME;
}

int sparse_quotient_mnt6_298_gpu_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return 0;
    }
    return 1;
}

} // extern "C"
