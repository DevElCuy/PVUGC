// BW6-761 CGBN-Based Sparse Quotient Coefficient Computation
//
// This kernel accelerates the coefficient accumulation phase of sparse quotient
// computation in the PVUGC setup for the BW6-761 curve.
//
// BW6-761 scalar field (Fr) is 377 bits, using 12 u32 limbs (384-bit padded).
// This is larger than MNT4/MNT6-298's 298-bit scalar field (10 u32 limbs).

#include <stddef.h>
#include <cuda_runtime.h>

// Include gmp.h BEFORE cgbn.h to avoid cgbn_cpu.h stub
#include <gmp.h>
#include <cgbn/cgbn.h>

// Error codes
#define SPARSE_QUOTIENT_SUCCESS 0
#define SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME -3
#define SPARSE_QUOTIENT_ERROR_INVALID_INPUT -6

// CGBN parameters for BW6-761 scalar field (377 bits, padded to 384)
class sparse_quotient_bw6_cgbn_params_t {
public:
    static const uint32_t TPB = 0;            // Get from blockDim.x
    static const uint32_t MAX_ROTATION = 4;
    static const uint32_t SHM_LIMIT = 0;
    static const bool CONSTANT_TIME = false;
    static const uint32_t TPI = 8;            // 8 threads per big number (safe for 384-bit)
    static const uint32_t BITS = 384;         // BW6-761 scalar field (377-bit, padded)
};

// BW6-761 scalar field order (Fr, 377 bits)
// This is the same as BLS12-377's scalar field
// Little-endian u32[12] representation
__device__ __constant__ uint32_t BW6_761_FR_DEVICE[12] = {
    0x00000001, 0x8508c000, 0x30000000, 0x170b5d44,
    0xba094800, 0x1ef3622f, 0x00f5138f, 0x1a22d9f3,
    0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46
};

// Scalar type (Fr element, 377 bits = 48 bytes in 12 u32 limbs)
typedef struct {
    uint32_t limbs[12];
} __align__(8) scalar_t;

// CSR format for sparse column data
typedef struct {
    const uint32_t* col_ptr;
    const uint32_t* row_idx;
    const scalar_t* values;
    uint32_t num_cols;
    uint32_t nnz;
} sparse_matrix_csr_t;

// Per-pair output
typedef struct {
    scalar_t* acc_u;
    scalar_t* acc_v;
    uint32_t* diag_k;
    scalar_t* diag_val;
    uint32_t num_diag;
} pair_output_t;

/**
 * Sparse Quotient Coefficient Kernel Class for BW6-761
 */
template<class params>
class sparse_quotient_bw6_kernel_t {
public:
    typedef cgbn_context_t<params::TPI, params> context_t;
    typedef cgbn_env_t<context_t, params::BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;

    context_t _context;
    env_t     _env;
    int32_t   _instance;

    __device__ __forceinline__ sparse_quotient_bw6_kernel_t(
        cgbn_monitor_t monitor,
        cgbn_error_report_t* report,
        int32_t instance
    ) : _context(monitor, report, (uint32_t)instance),
        _env(_context),
        _instance(instance) {}

    // Field addition: r = (a + b) mod Fr
    __device__ __forceinline__ void field_add(bn_t& r, const bn_t& a,
                                               const bn_t& b, const bn_t& Fr) {
        cgbn_add(_env, r, a, b);
        if (cgbn_compare(_env, r, Fr) >= 0) {
            cgbn_sub(_env, r, r, Fr);
        }
    }

    // Field subtraction: r = (a - b) mod Fr
    __device__ __forceinline__ void field_sub(bn_t& r, const bn_t& a,
                                               const bn_t& b, const bn_t& Fr) {
        int32_t borrow = cgbn_sub(_env, r, a, b);
        if (borrow != 0) {
            cgbn_add(_env, r, r, Fr);
        }
    }

    // Field multiplication: r = (a * b) mod Fr
    __device__ __forceinline__ void field_mul(bn_t& r, const bn_t& a,
                                               const bn_t& b, const bn_t& Fr) {
        typedef typename env_t::cgbn_wide_t wide_t;
        wide_t product;
        cgbn_mul_wide(_env, product, a, b);
        cgbn_rem_wide(_env, r, product, Fr);
    }

    // Field negation: r = -a mod Fr
    __device__ __forceinline__ void field_neg(bn_t& r, const bn_t& a, const bn_t& Fr) {
        if (cgbn_equals_ui32(_env, a, 0)) {
            cgbn_set_ui32(_env, r, 0);
        } else {
            cgbn_sub(_env, r, Fr, a);
        }
    }

    // Load scalar from memory
    __device__ __forceinline__ void load_scalar(bn_t& r, const scalar_t* s) {
        cgbn_load(_env, r, (cgbn_mem_t<384>*)s->limbs);
    }

    // Store scalar to memory
    __device__ __forceinline__ void store_scalar(scalar_t* s, const bn_t& r) {
        cgbn_store(_env, (cgbn_mem_t<384>*)s->limbs, r);
    }

    // Load field modulus
    __device__ __forceinline__ void load_modulus(bn_t& Fr) {
        cgbn_load(_env, Fr, (cgbn_mem_t<384>*)BW6_761_FR_DEVICE);
    }
};

/**
 * Compute coefficients for a single (i, j) pair - Two-Phase Algorithm
 */
__global__ void compute_quotient_coeffs_bw6_kernel(
    // Sparse matrix data (CSR format)
    const uint32_t* col_a_ptr,
    const uint32_t* col_a_idx,
    const scalar_t* col_a_val,
    const uint32_t* col_b_ptr,
    const uint32_t* col_b_idx,
    const scalar_t* col_b_val,
    // Precomputed domain element tables
    const scalar_t* domain_elements,
    const scalar_t* inv_domain_elements,
    const scalar_t* inv_n_one_minus_omega,
    uint32_t domain_size,
    // Pair assignments
    const uint32_t* pairs_i,
    const uint32_t* pairs_j,
    uint32_t num_pairs,
    // Output arrays
    scalar_t* out_acc_u,
    scalar_t* out_acc_v,
    uint32_t max_col_a,
    uint32_t max_col_b,
    // Diagonal output
    uint32_t* out_diag_k,
    scalar_t* out_diag_val,
    uint32_t* out_num_diag,
    uint32_t max_diag_per_pair,
    // Error reporting
    cgbn_error_report_t* report
) {
    uint32_t pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    constexpr uint32_t TPI = sparse_quotient_bw6_cgbn_params_t::TPI;

    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;

    sparse_quotient_bw6_kernel_t<sparse_quotient_bw6_cgbn_params_t> kernel(
        cgbn_no_checks, report, instance
    );

    typedef sparse_quotient_bw6_kernel_t<sparse_quotient_bw6_cgbn_params_t>::bn_t bn_t;

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

    // Initialize output accumulators to zero
    if (lane == 0) {
        for (uint32_t idx = local_instance; idx < n_u; idx += instances_per_block) {
            scalar_t* out = &out_acc_u[pair_idx * max_col_a + idx];
            for (int l = 0; l < 12; l++) out->limbs[l] = 0;
        }
        for (uint32_t idx = local_instance; idx < n_v; idx += instances_per_block) {
            scalar_t* out = &out_acc_v[pair_idx * max_col_b + idx];
            for (int l = 0; l < 12; l++) out->limbs[l] = 0;
        }
    }
    __syncthreads();

    __shared__ uint32_t diag_count;
    if (threadIdx.x == 0) {
        diag_count = 0;
    }
    __syncthreads();

    uint32_t lane_in_warp = threadIdx.x & 31;
    uint32_t group_thread = threadIdx.x % TPI;
    uint32_t instance_mask = ((1u << TPI) - 1u) << (lane_in_warp - group_thread);

    // ========================================================================
    // PHASE 1: Compute acc_u (and collect diagonal terms)
    // ========================================================================
    for (uint32_t idx_u = local_instance; idx_u < n_u; idx_u += instances_per_block) {
        uint32_t k = col_a_idx[u_start + idx_u];
        bn_t val_u;
        kernel.load_scalar(val_u, &col_a_val[u_start + idx_u]);

        bn_t acc_u_local;
        cgbn_set_ui32(kernel._env, acc_u_local, 0);

        for (uint32_t idx_v = 0; idx_v < n_v; idx_v++) {
            uint32_t m = col_b_idx[v_start + idx_v];
            bn_t val_v;
            kernel.load_scalar(val_v, &col_b_val[v_start + idx_v]);

            bn_t prod;
            kernel.field_mul(prod, val_u, val_v, Fr);

            if (k == m) {
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
                uint32_t d = (k >= m) ? (k - m) : (k + domain_size - m);

                bn_t wm, inv_wm, inv_n_term, inv_denom, common, tmp;

                kernel.load_scalar(wm, &domain_elements[m]);
                kernel.load_scalar(inv_wm, &inv_domain_elements[m]);
                kernel.load_scalar(inv_n_term, &inv_n_one_minus_omega[d]);

                kernel.field_mul(inv_denom, inv_wm, inv_n_term, Fr);
                kernel.field_neg(inv_denom, inv_denom, Fr);

                kernel.field_mul(common, prod, inv_denom, Fr);

                kernel.field_mul(tmp, common, wm, Fr);
                kernel.field_add(acc_u_local, acc_u_local, tmp, Fr);
            }
        }

        kernel.store_scalar(&out_acc_u[pair_idx * max_col_a + idx_u], acc_u_local);
    }

    __syncthreads();

    // ========================================================================
    // PHASE 2: Compute acc_v
    // ========================================================================
    for (uint32_t idx_v = local_instance; idx_v < n_v; idx_v += instances_per_block) {
        uint32_t m = col_b_idx[v_start + idx_v];
        bn_t val_v;
        kernel.load_scalar(val_v, &col_b_val[v_start + idx_v]);

        bn_t acc_v_local;
        cgbn_set_ui32(kernel._env, acc_v_local, 0);

        for (uint32_t idx_u = 0; idx_u < n_u; idx_u++) {
            uint32_t k = col_a_idx[u_start + idx_u];

            if (k == m) continue;

            bn_t val_u;
            kernel.load_scalar(val_u, &col_a_val[u_start + idx_u]);

            bn_t prod;
            kernel.field_mul(prod, val_u, val_v, Fr);

            uint32_t d = (k >= m) ? (k - m) : (k + domain_size - m);

            bn_t wk, inv_wm, inv_n_term, inv_denom, common, tmp;

            kernel.load_scalar(wk, &domain_elements[k]);
            kernel.load_scalar(inv_wm, &inv_domain_elements[m]);
            kernel.load_scalar(inv_n_term, &inv_n_one_minus_omega[d]);

            kernel.field_mul(inv_denom, inv_wm, inv_n_term, Fr);
            kernel.field_neg(inv_denom, inv_denom, Fr);

            kernel.field_mul(common, prod, inv_denom, Fr);

            kernel.field_mul(tmp, common, wk, Fr);
            kernel.field_sub(acc_v_local, acc_v_local, tmp, Fr);
        }

        kernel.store_scalar(&out_acc_v[pair_idx * max_col_b + idx_v], acc_v_local);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        out_num_diag[pair_idx] = diag_count < max_diag_per_pair ? diag_count : max_diag_per_pair;
    }
}

// C interface for FFI
extern "C" {

int sparse_quotient_coeffs_bw6_761_gpu(
    // Sparse matrix A (CSR)
    const uint32_t* col_a_ptr,
    const uint32_t* col_a_idx,
    const void* col_a_val,
    uint32_t num_cols_a,
    uint32_t nnz_a,
    // Sparse matrix B (CSR)
    const uint32_t* col_b_ptr,
    const uint32_t* col_b_idx,
    const void* col_b_val,
    uint32_t num_cols_b,
    uint32_t nnz_b,
    // Domain element tables
    const void* domain_elements,
    const void* inv_domain_elements,
    const void* inv_n_one_minus_omega,
    uint32_t domain_size,
    // Pair batch
    const uint32_t* pairs_i,
    const uint32_t* pairs_j,
    uint32_t num_pairs,
    // Output sizing
    uint32_t max_col_a,
    uint32_t max_col_b,
    uint32_t max_diag_per_pair,
    // Output arrays
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

    size_t acc_u_size = (size_t)num_pairs * max_col_a * sizeof(scalar_t);
    size_t acc_v_size = (size_t)num_pairs * max_col_b * sizeof(scalar_t);

    fprintf(stderr, "[sparse_quotient BW6] pairs=%u, max_col_a=%u, max_col_b=%u, acc_u=%.1fMB, acc_v=%.1fMB\n",
            num_pairs, max_col_a, max_col_b,
            acc_u_size / (1024.0 * 1024.0), acc_v_size / (1024.0 * 1024.0));

    // Allocate device memory for sparse matrices
    uint32_t* d_col_a_ptr;
    uint32_t* d_col_a_idx;
    scalar_t* d_col_a_val;
    uint32_t* d_col_b_ptr;
    uint32_t* d_col_b_idx;
    scalar_t* d_col_b_val;

    err = cudaMalloc(&d_col_a_ptr, (num_cols_a + 1) * sizeof(uint32_t));
    if (err != cudaSuccess) return SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME;
    err = cudaMalloc(&d_col_a_idx, nnz_a * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_col_a_ptr); return SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME; }
    err = cudaMalloc(&d_col_a_val, nnz_a * sizeof(scalar_t));
    if (err != cudaSuccess) { cudaFree(d_col_a_ptr); cudaFree(d_col_a_idx); return SPARSE_QUOTIENT_ERROR_CUDA_RUNTIME; }

    err = cudaMalloc(&d_col_b_ptr, (num_cols_b + 1) * sizeof(uint32_t));
    if (err != cudaSuccess) goto cleanup_a;
    err = cudaMalloc(&d_col_b_idx, nnz_b * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_col_b_ptr); goto cleanup_a; }
    err = cudaMalloc(&d_col_b_val, nnz_b * sizeof(scalar_t));
    if (err != cudaSuccess) { cudaFree(d_col_b_ptr); cudaFree(d_col_b_idx); goto cleanup_a; }

    // Allocate device memory for domain element tables
    scalar_t* d_domain_elements;
    scalar_t* d_inv_domain_elements;
    scalar_t* d_inv_n_one_minus_omega;

    err = cudaMalloc(&d_domain_elements, domain_size * sizeof(scalar_t));
    if (err != cudaSuccess) goto cleanup_b;
    err = cudaMalloc(&d_inv_domain_elements, domain_size * sizeof(scalar_t));
    if (err != cudaSuccess) { cudaFree(d_domain_elements); goto cleanup_b; }
    err = cudaMalloc(&d_inv_n_one_minus_omega, domain_size * sizeof(scalar_t));
    if (err != cudaSuccess) { cudaFree(d_domain_elements); cudaFree(d_inv_domain_elements); goto cleanup_b; }

    // Allocate device memory for pairs
    uint32_t* d_pairs_i;
    uint32_t* d_pairs_j;
    err = cudaMalloc(&d_pairs_i, num_pairs * sizeof(uint32_t));
    if (err != cudaSuccess) goto cleanup_domain;
    err = cudaMalloc(&d_pairs_j, num_pairs * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_pairs_i); goto cleanup_domain; }

    // Allocate device memory for outputs
    scalar_t* d_out_acc_u;
    scalar_t* d_out_acc_v;
    uint32_t* d_out_diag_k;
    scalar_t* d_out_diag_val;
    uint32_t* d_out_num_diag;

    err = cudaMalloc(&d_out_acc_u, acc_u_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[sparse_quotient BW6] cudaMalloc acc_u failed: %s (requested %.2f MB)\n",
                cudaGetErrorString(err), acc_u_size / (1024.0 * 1024.0));
        goto cleanup_pairs;
    }
    err = cudaMalloc(&d_out_acc_v, acc_v_size);
    if (err != cudaSuccess) { cudaFree(d_out_acc_u); goto cleanup_pairs; }
    err = cudaMalloc(&d_out_diag_k, num_pairs * max_diag_per_pair * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_out_acc_u); cudaFree(d_out_acc_v); goto cleanup_pairs; }
    err = cudaMalloc(&d_out_diag_val, num_pairs * max_diag_per_pair * sizeof(scalar_t));
    if (err != cudaSuccess) { cudaFree(d_out_acc_u); cudaFree(d_out_acc_v); cudaFree(d_out_diag_k); goto cleanup_pairs; }
    err = cudaMalloc(&d_out_num_diag, num_pairs * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_out_acc_u); cudaFree(d_out_acc_v); cudaFree(d_out_diag_k); cudaFree(d_out_diag_val); goto cleanup_pairs; }

    // CGBN error report
    cgbn_error_report_t* d_report;
    err = cudaMalloc(&d_report, sizeof(cgbn_error_report_t));
    if (err != cudaSuccess) goto cleanup_output;

    // Copy input data to device
    cudaMemcpy(d_col_a_ptr, col_a_ptr, (num_cols_a + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_a_idx, col_a_idx, nnz_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_a_val, col_a_val, nnz_a * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_b_ptr, col_b_ptr, (num_cols_b + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_b_idx, col_b_idx, nnz_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_b_val, col_b_val, nnz_b * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domain_elements, domain_elements, domain_size * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_domain_elements, inv_domain_elements, domain_size * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_n_one_minus_omega, inv_n_one_minus_omega, domain_size * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs_i, pairs_i, num_pairs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs_j, pairs_j, num_pairs * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Initialize CGBN error report
    cudaMemset(d_report, 0, sizeof(cgbn_error_report_t));

    // Launch kernel
    threads_per_block = 256;
    num_blocks = num_pairs;

    compute_quotient_coeffs_bw6_kernel<<<num_blocks, threads_per_block>>>(
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

    // Copy results back to host
    cudaMemcpy(out_acc_u, d_out_acc_u, num_pairs * max_col_a * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_acc_v, d_out_acc_v, num_pairs * max_col_b * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_diag_k, d_out_diag_k, num_pairs * max_diag_per_pair * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_diag_val, d_out_diag_val, num_pairs * max_diag_per_pair * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_num_diag, d_out_num_diag, num_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Cleanup and return success
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

int sparse_quotient_bw6_761_gpu_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return 0;
    }
    return 1;
}

} // extern "C"
