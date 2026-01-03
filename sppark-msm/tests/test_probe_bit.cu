/***
 * Probe-Bit Test for BW6-761 Scalar Multiplication Bug
 *
 * This test automatically finds the first bit where GPU diverges from CPU
 * by stopping the double-and-add loop at a specified bit position and
 * comparing intermediate accumulator states.
 *
 * Compile: nvcc -o test_probe_bit test_probe_bit.cu \
 *              -I../src -I../../cgbn-lib/include -lgmp -arch=sm_75 -O2
 * Run: ./test_probe_bit
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"

// CGBN configuration
#define TPI 8
#define BITS 768

// BW6-761 constants
__device__ __constant__ uint32_t BW6_761_P_DEVICE[24] = {
    0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68,
    0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
    0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a,
    0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824
};

__device__ __constant__ uint32_t BW6_761_NP0 = 0x8fa798dd;

__device__ __constant__ uint32_t BW6_761_R2_DEVICE[24] = {
    0x2d1fa659, 0xc686392d, 0xf79484ab, 0x7b14c9b2,
    0xc1d2b459, 0x7fa1e825, 0x48329d88, 0xd6ec28f8,
    0x73a1ed40, 0x4afb427b, 0x0d5930ae, 0x972c6940,
    0x8c995976, 0x2c7a26bf, 0xc6e57af9, 0xac52e458,
    0x0c536dfe, 0xac731bfa, 0x0b103f50, 0x121e5c63,
    0xb886cda4, 0x8f1b0953, 0x2da8d807, 0x00ad253c
};

uint32_t BW6_761_P_HOST[24] = {
    0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68,
    0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
    0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a,
    0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824
};

// Point 131 coordinates
uint32_t POINT131_X[24] = {
    0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
    0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
    0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
    0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
    0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
    0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
};

uint32_t POINT131_Y[24] = {
    0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
    0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
    0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
    0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
    0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
    0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
};

// The failing scalar (limbs 3-10,11)
uint32_t FAILING_SCALAR[12] = {
    0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b,
    0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712,
    0xa562b722, 0x0004ee43
};

// Test data for probe kernel
typedef struct {
    cgbn_mem_t<BITS> px;
    cgbn_mem_t<BITS> py;
    uint32_t scalar[12];
    int stop_at_bit;           // Stop after processing this bit (from MSB)
    int capture_iteration;     // Iteration to capture (1-based), 0 = disabled
    int capture_phase;         // 0 = after double (pre-add), 1 = after add
    uint32_t capture_valid;    // 1 if capture occurred
    uint32_t capture_bit_set;  // 1 if bit was set at capture iteration
    uint32_t capture_pdiff_zero;
    uint32_t capture_r_zero;
    cgbn_mem_t<BITS> capture_x;
    cgbn_mem_t<BITS> capture_y;
    cgbn_mem_t<BITS> capture_zz;
    cgbn_mem_t<BITS> capture_zzz;
    cgbn_mem_t<BITS> capture_q_minus_x3;
    cgbn_mem_t<BITS> capture_y1_ppp;
    cgbn_mem_t<BITS> capture_r_times_qmx_pre;
    cgbn_mem_t<BITS> capture_r_times_qmx;
    cgbn_mem_t<BITS> result_x; // Intermediate X (affine)
    cgbn_mem_t<BITS> result_y; // Intermediate Y (affine)
    uint32_t result_infinity; // 1 if result is infinity
} probe_test_t;

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

//-----------------------------------------------------------------------------
// GMP utilities
//-----------------------------------------------------------------------------

void mpz_to_limbs(mpz_t n, uint32_t *limbs, int count) {
    memset(limbs, 0, count * sizeof(uint32_t));
    size_t actual_count;
    mpz_export(limbs, &actual_count, -1, sizeof(uint32_t), -1, 0, n);
}

void limbs_to_mpz(mpz_t n, const uint32_t *limbs, int count) {
    mpz_import(n, count, -1, sizeof(uint32_t), -1, 0, limbs);
}

//-----------------------------------------------------------------------------
// GMP reference - scalar mul stopping at a specific bit
//-----------------------------------------------------------------------------

void gmp_mod_inverse(mpz_t result, mpz_t a, mpz_t p) {
    mpz_invert(result, a, p);
}

bool gmp_point_double(mpz_t x3, mpz_t y3, mpz_t x1, mpz_t y1, mpz_t p) {
    if (mpz_cmp_ui(y1, 0) == 0) {
        mpz_set_ui(x3, 0);
        mpz_set_ui(y3, 0);
        return true;
    }

    mpz_t lambda, temp, temp2, x1_sq;
    mpz_init(lambda); mpz_init(temp); mpz_init(temp2); mpz_init(x1_sq);

    mpz_mul(x1_sq, x1, x1); mpz_mod(x1_sq, x1_sq, p);
    mpz_mul_ui(temp, x1_sq, 3); mpz_mod(temp, temp, p);
    mpz_mul_ui(temp2, y1, 2); mpz_mod(temp2, temp2, p);
    if (mpz_invert(temp2, temp2, p) == 0) {
        mpz_set_ui(x3, 0);
        mpz_set_ui(y3, 0);
        mpz_clear(lambda); mpz_clear(temp); mpz_clear(temp2); mpz_clear(x1_sq);
        return true;
    }
    mpz_mul(lambda, temp, temp2); mpz_mod(lambda, lambda, p);

    mpz_mul(x3, lambda, lambda); mpz_mod(x3, x3, p);
    mpz_mul_ui(temp, x1, 2); mpz_mod(temp, temp, p);
    mpz_sub(x3, x3, temp); mpz_mod(x3, x3, p);

    mpz_sub(temp, x1, x3); mpz_mod(temp, temp, p);
    mpz_mul(y3, lambda, temp); mpz_mod(y3, y3, p);
    mpz_sub(y3, y3, y1); mpz_mod(y3, y3, p);

    mpz_clear(lambda); mpz_clear(temp); mpz_clear(temp2); mpz_clear(x1_sq);
    return false;
}

bool gmp_point_add(mpz_t x3, mpz_t y3, mpz_t x1, mpz_t y1, mpz_t x2, mpz_t y2, mpz_t p) {
    if (mpz_cmp(x1, x2) == 0) {
        if (mpz_cmp(y1, y2) == 0) {
            return gmp_point_double(x3, y3, x1, y1, p);
        }
        mpz_set_ui(x3, 0);
        mpz_set_ui(y3, 0);
        return true;
    }
    mpz_t lambda, temp, temp2;
    mpz_init(lambda); mpz_init(temp); mpz_init(temp2);

    mpz_sub(temp, y2, y1); mpz_mod(temp, temp, p);
    mpz_sub(temp2, x2, x1); mpz_mod(temp2, temp2, p);
    if (mpz_invert(temp2, temp2, p) == 0) {
        mpz_set_ui(x3, 0);
        mpz_set_ui(y3, 0);
        mpz_clear(lambda); mpz_clear(temp); mpz_clear(temp2);
        return true;
    }
    mpz_mul(lambda, temp, temp2); mpz_mod(lambda, lambda, p);

    mpz_mul(x3, lambda, lambda); mpz_mod(x3, x3, p);
    mpz_sub(x3, x3, x1); mpz_mod(x3, x3, p);
    mpz_sub(x3, x3, x2); mpz_mod(x3, x3, p);

    mpz_sub(temp, x1, x3); mpz_mod(temp, temp, p);
    mpz_mul(y3, lambda, temp); mpz_mod(y3, y3, p);
    mpz_sub(y3, y3, y1); mpz_mod(y3, y3, p);

    mpz_clear(lambda); mpz_clear(temp); mpz_clear(temp2);
    return false;
}

// Scalar mul that stops after processing stop_at_bit (counting from MSB)
bool gmp_scalar_mul_stop_at(mpz_t rx, mpz_t ry,
                             mpz_t px, mpz_t py,
                             mpz_t k, mpz_t p,
                             int stop_at_bit) {
    if (mpz_cmp_ui(k, 0) == 0) {
        mpz_set_ui(rx, 0);
        mpz_set_ui(ry, 0);
        return true;
    }

    int highest_bit = mpz_sizeinbase(k, 2) - 1;

    mpz_t acc_x, acc_y, temp_x, temp_y;
    mpz_init_set(acc_x, px);
    mpz_init_set(acc_y, py);
    mpz_init(temp_x);
    mpz_init(temp_y);

    bool acc_inf = false;
    int bits_processed = 0;
    for (int i = highest_bit - 1; i >= 0 && bits_processed < stop_at_bit; i--) {
        if (!acc_inf) {
            acc_inf = gmp_point_double(temp_x, temp_y, acc_x, acc_y, p);
            if (!acc_inf) {
                mpz_set(acc_x, temp_x);
                mpz_set(acc_y, temp_y);
            }
        }

        if (mpz_tstbit(k, i)) {
            if (acc_inf) {
                mpz_set(acc_x, px);
                mpz_set(acc_y, py);
                acc_inf = false;
            } else {
                acc_inf = gmp_point_add(temp_x, temp_y, acc_x, acc_y, px, py, p);
                if (!acc_inf) {
                    mpz_set(acc_x, temp_x);
                    mpz_set(acc_y, temp_y);
                }
            }
        }
        bits_processed++;
    }

    if (acc_inf) {
        mpz_set_ui(rx, 0);
        mpz_set_ui(ry, 0);
    } else {
        mpz_set(rx, acc_x);
        mpz_set(ry, acc_y);
    }

    mpz_clear(acc_x); mpz_clear(acc_y);
    mpz_clear(temp_x); mpz_clear(temp_y);
    return acc_inf;
}

//-----------------------------------------------------------------------------
// CUDA Kernel - Probe at specific bit
//-----------------------------------------------------------------------------

__global__ void probe_kernel(cgbn_error_report_t *report,
                              probe_test_t *test) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (instance != 0) return;

    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t bn_env(bn_context.env<env_t>());
    typedef env_t::cgbn_t bn_t;

    bn_t P, R2, one, one_m;
    cgbn_load(bn_env, P, (cgbn_mem_t<BITS>*)BW6_761_P_DEVICE);
    cgbn_load(bn_env, R2, (cgbn_mem_t<BITS>*)BW6_761_R2_DEVICE);
    cgbn_set_ui32(bn_env, one, 1);
    cgbn_mont_mul(bn_env, one_m, one, R2, P, BW6_761_NP0);

    bn_t px, py;
    cgbn_load(bn_env, px, &(test->px));
    cgbn_load(bn_env, py, &(test->py));

    bn_t px_m, py_m;
    cgbn_mont_mul(bn_env, px_m, px, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, py_m, py, R2, P, BW6_761_NP0);

    uint32_t *scalar = test->scalar;
    int stop_at_bit = test->stop_at_bit;
    int capture_iteration = test->capture_iteration;
    int capture_phase = test->capture_phase;

    if (threadIdx.x == 0) {
        test->result_infinity = 0;
        test->capture_valid = 0;
        test->capture_bit_set = 0;
        test->capture_pdiff_zero = 0;
        test->capture_r_zero = 0;
    }
    __syncthreads();

    // Find highest bit
    int highest_bit = -1;
    for (int limb_idx = 11; limb_idx >= 0; limb_idx--) {
        uint32_t limb = scalar[limb_idx];
        if (limb != 0) {
            for (int bit_idx = 31; bit_idx >= 0; bit_idx--) {
                if (limb & (1u << bit_idx)) {
                    highest_bit = limb_idx * 32 + bit_idx;
                    break;
                }
            }
            if (highest_bit >= 0) break;
        }
    }

    // XYZZ accumulator
    bn_t acc_x, acc_y, acc_zz, acc_zzz;
    cgbn_set(bn_env, acc_x, px_m);
    cgbn_set(bn_env, acc_y, py_m);
    cgbn_set(bn_env, acc_zz, one_m);
    cgbn_set(bn_env, acc_zzz, one_m);

    bn_t temp_x, temp_y, temp_zz, temp_zzz;

    #define field_add(r, a, b) do { cgbn_add(bn_env, r, a, b); cgbn_rem(bn_env, r, r, P); } while(0)
    #define field_sub(r, a, b) do { \
        bn_t __temp_result; \
        if (cgbn_compare(bn_env, a, b) >= 0) { \
            cgbn_sub(bn_env, __temp_result, a, b); \
        } else { \
            bn_t __p_minus_b; \
            cgbn_sub(bn_env, __p_minus_b, P, b); \
            cgbn_add(bn_env, __temp_result, a, __p_minus_b); \
        } \
        cgbn_set(bn_env, r, __temp_result); \
    } while(0)
    #define field_mul(r, a, b) cgbn_mont_mul(bn_env, r, a, b, P, BW6_761_NP0)

    bn_t zero;
    cgbn_set_ui32(bn_env, zero, 0);

    int bits_processed = 0;
    for (int bit_pos = highest_bit - 1; bit_pos >= 0 && bits_processed < stop_at_bit; bit_pos--) {
        // Point Doubling (XYZZ)
        bn_t A, V, U, W, S, M, temp;
        field_mul(A, acc_y, acc_y);
        field_add(U, acc_y, acc_y);
        field_add(V, A, A); field_add(V, V, V);
        field_mul(W, U, V);
        field_mul(S, acc_x, V);
        field_mul(M, acc_x, acc_x);
        field_add(temp, M, M); field_add(M, M, temp);
        field_mul(temp_x, M, M);
        field_sub(temp_x, temp_x, S); field_sub(temp_x, temp_x, S);
        field_sub(temp, S, temp_x);
        field_mul(temp_y, M, temp);
        field_mul(temp, A, V); field_add(temp, temp, temp);
        field_sub(temp_y, temp_y, temp);
        field_mul(temp_zz, V, acc_zz);
        field_mul(temp_zzz, W, acc_zzz);

        cgbn_set(bn_env, acc_x, temp_x);
        cgbn_set(bn_env, acc_y, temp_y);
        cgbn_set(bn_env, acc_zz, temp_zz);
        cgbn_set(bn_env, acc_zzz, temp_zzz);

        int limb_idx = bit_pos / 32;
        int bit_idx = bit_pos % 32;
        bool bit_set = (scalar[limb_idx] & (1u << bit_idx)) != 0;

        bool capture_here = (capture_iteration > 0) && ((bits_processed + 1) == capture_iteration);
        if (capture_here && (capture_phase == 0 || (capture_phase == 1 && !bit_set))) {
            cgbn_store(bn_env, &(test->capture_x), acc_x);
            cgbn_store(bn_env, &(test->capture_y), acc_y);
            cgbn_store(bn_env, &(test->capture_zz), acc_zz);
            cgbn_store(bn_env, &(test->capture_zzz), acc_zzz);
            if (threadIdx.x == 0) {
                test->capture_valid = 1;
                test->capture_bit_set = bit_set ? 1u : 0u;
            }
            break;
        }

        if (bit_set) {
            // Point Addition (XYZZ + Affine)
            bn_t U2, S2, Pdiff, R, PP, PPP, Q, temp2;
            field_mul(U2, px_m, acc_zz);
            field_mul(S2, py_m, acc_zzz);
            field_sub(Pdiff, U2, acc_x);
            field_sub(R, S2, acc_y);
            field_mul(PP, Pdiff, Pdiff);
            field_mul(PPP, Pdiff, PP);
            field_mul(Q, acc_x, PP);
            bool pdiff_zero = (cgbn_compare(bn_env, Pdiff, zero) == 0);
            bool r_zero = (cgbn_compare(bn_env, R, zero) == 0);

            field_mul(temp_x, R, R);
            field_sub(temp_x, temp_x, PPP);
            field_sub(temp_x, temp_x, Q); field_sub(temp_x, temp_x, Q);
            field_sub(temp, Q, temp_x);
            field_mul(temp2, acc_y, PPP);
            field_mul(temp_y, R, temp);
            if (capture_here && capture_phase == 1) {
                cgbn_store(bn_env, &(test->capture_r_times_qmx_pre), temp_y);
            }
            field_sub(temp_y, temp_y, temp2);
            field_mul(temp_zz, acc_zz, PP);
            field_mul(temp_zzz, acc_zzz, PPP);

            cgbn_set(bn_env, acc_x, temp_x);
            cgbn_set(bn_env, acc_y, temp_y);
            cgbn_set(bn_env, acc_zz, temp_zz);
            cgbn_set(bn_env, acc_zzz, temp_zzz);

            if (capture_here && capture_phase == 1) {
                cgbn_store(bn_env, &(test->capture_x), acc_x);
                cgbn_store(bn_env, &(test->capture_y), acc_y);
                cgbn_store(bn_env, &(test->capture_zz), acc_zz);
                cgbn_store(bn_env, &(test->capture_zzz), acc_zzz);
                cgbn_store(bn_env, &(test->capture_q_minus_x3), temp);
                cgbn_store(bn_env, &(test->capture_y1_ppp), temp2);
                cgbn_store(bn_env, &(test->capture_r_times_qmx), temp_y);
                if (threadIdx.x == 0) {
                    test->capture_valid = 1;
                    test->capture_bit_set = 1u;
                    test->capture_pdiff_zero = pdiff_zero ? 1u : 0u;
                    test->capture_r_zero = r_zero ? 1u : 0u;
                }
                break;
            }
        }
        bits_processed++;
    }

    // Convert from XYZZ to affine
    bn_t ZZ_plain, ZZZ_plain;
    cgbn_mont_mul(bn_env, ZZ_plain, acc_zz, one, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, ZZZ_plain, acc_zzz, one, P, BW6_761_NP0);

    if (cgbn_compare(bn_env, ZZ_plain, zero) == 0 ||
        cgbn_compare(bn_env, ZZZ_plain, zero) == 0) {
        cgbn_store(bn_env, &(test->result_x), zero);
        cgbn_store(bn_env, &(test->result_y), zero);
        if (threadIdx.x == 0) {
            test->result_infinity = 1;
        }
        return;
    }

    bn_t ZZ_inv, ZZZ_inv;
    cgbn_modular_inverse(bn_env, ZZ_inv, ZZ_plain, P);
    cgbn_modular_inverse(bn_env, ZZZ_inv, ZZZ_plain, P);

    bn_t ZZ_inv_m, ZZZ_inv_m;
    cgbn_mont_mul(bn_env, ZZ_inv_m, ZZ_inv, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, ZZZ_inv_m, ZZZ_inv, R2, P, BW6_761_NP0);

    bn_t x_aff_m, y_aff_m;
    field_mul(x_aff_m, acc_x, ZZ_inv_m);
    field_mul(y_aff_m, acc_y, ZZZ_inv_m);

    bn_t x_aff, y_aff;
    cgbn_mont_mul(bn_env, x_aff, x_aff_m, one, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, y_aff, y_aff_m, one, P, BW6_761_NP0);

    cgbn_store(bn_env, &(test->result_x), x_aff);
    cgbn_store(bn_env, &(test->result_y), y_aff);

    #undef field_add
    #undef field_sub
    #undef field_mul
}

//-----------------------------------------------------------------------------
// Compare GPU vs CPU at a given bit
//-----------------------------------------------------------------------------

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)
#define CGBN_CHECK(report) do { if (cgbn_error_report_check(report)) { fprintf(stderr, "CGBN error\n"); exit(1); } } while(0)

bool compare_at_bit(int stop_at_bit, probe_test_t *gpu_test, cgbn_error_report_t *report,
                     uint32_t *px_limbs, uint32_t *py_limbs, uint32_t *scalar_limbs) {
    probe_test_t test;
    mpz_t P, px, py, k, rx_cpu, ry_cpu;
    mpz_init(P); mpz_init(px); mpz_init(py); mpz_init(k);
    mpz_init(rx_cpu); mpz_init(ry_cpu);

    limbs_to_mpz(P, BW6_761_P_HOST, 24);
    limbs_to_mpz(px, px_limbs, 24);
    limbs_to_mpz(py, py_limbs, 24);
    limbs_to_mpz(k, scalar_limbs, 12);

    // CPU reference
    bool cpu_inf = gmp_scalar_mul_stop_at(rx_cpu, ry_cpu, px, py, k, P, stop_at_bit);
    uint32_t cpu_x[24], cpu_y[24];
    mpz_to_limbs(rx_cpu, cpu_x, 24);
    mpz_to_limbs(ry_cpu, cpu_y, 24);

    // GPU
    memcpy(test.px._limbs, px_limbs, 24 * sizeof(uint32_t));
    memcpy(test.py._limbs, py_limbs, 24 * sizeof(uint32_t));
    memcpy(test.scalar, scalar_limbs, 12 * sizeof(uint32_t));
    test.stop_at_bit = stop_at_bit;
    test.capture_iteration = 0;
    test.capture_phase = 0;
    test.capture_valid = 0;
    test.capture_bit_set = 0;
    test.capture_pdiff_zero = 0;
    test.capture_r_zero = 0;
    memset(test.result_x._limbs, 0, sizeof(test.result_x._limbs));
    memset(test.result_y._limbs, 0, sizeof(test.result_y._limbs));
    test.result_infinity = 0;

    CUDA_CHECK(cudaMemcpy(gpu_test, &test, sizeof(probe_test_t), cudaMemcpyHostToDevice));
    probe_kernel<<<1, TPI>>>(report, gpu_test);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);
    CUDA_CHECK(cudaMemcpy(&test, gpu_test, sizeof(probe_test_t), cudaMemcpyDeviceToHost));

    bool gpu_inf = test.result_infinity != 0;
    if (cpu_inf != gpu_inf) {
        return false;
    }
    if (cpu_inf) {
        return true;
    }

    bool match = true;
    for (int i = 0; i < 24; i++) {
        if (test.result_x._limbs[i] != cpu_x[i] || test.result_y._limbs[i] != cpu_y[i]) {
            match = false;
            break;
        }
    }

    mpz_clear(P); mpz_clear(px); mpz_clear(py); mpz_clear(k);
    mpz_clear(rx_cpu); mpz_clear(ry_cpu);

    return match;
}

void print_limbs_full(const char *label, const uint32_t *limbs, int count) {
    printf("%s:", label);
    for (int i = count - 1; i >= 0; i--) {
        if ((count - 1 - i) % 4 == 0) {
            printf("\n  ");
        }
        printf("0x%08x ", limbs[i]);
    }
    printf("\n");
}

int find_highest_bit(const uint32_t *scalar) {
    int highest_bit = -1;
    for (int limb_idx = 11; limb_idx >= 0; limb_idx--) {
        if (scalar[limb_idx] != 0) {
            for (int bit_idx = 31; bit_idx >= 0; bit_idx--) {
                if (scalar[limb_idx] & (1u << bit_idx)) {
                    highest_bit = limb_idx * 32 + bit_idx;
                    break;
                }
            }
            if (highest_bit >= 0) break;
        }
    }
    return highest_bit;
}

void clear_scalar_bit(uint32_t *scalar, int bit_pos) {
    if (bit_pos < 0) return;
    int limb_idx = bit_pos / 32;
    int bit_idx = bit_pos % 32;
    scalar[limb_idx] &= ~(1u << bit_idx);
}

bool compare_phase_at_bit(int stop_at_bit, int phase,
                          probe_test_t *gpu_test, cgbn_error_report_t *report,
                          uint32_t *px_limbs, uint32_t *py_limbs, uint32_t *scalar_limbs,
                          const char *dump_label, bool dump,
                          uint32_t *out_bit_set, uint32_t *out_pdiff_zero, uint32_t *out_r_zero) {
    probe_test_t test;
    mpz_t P, px, py, k, rx_cpu, ry_cpu;
    mpz_init(P); mpz_init(px); mpz_init(py); mpz_init(k);
    mpz_init(rx_cpu); mpz_init(ry_cpu);

    uint32_t scalar_copy[12];
    memcpy(scalar_copy, scalar_limbs, sizeof(scalar_copy));

    int highest_bit = find_highest_bit(scalar_limbs);
    int target_bit_pos = highest_bit - stop_at_bit;
    if (phase == 0) {
        clear_scalar_bit(scalar_copy, target_bit_pos);
    }

    limbs_to_mpz(P, BW6_761_P_HOST, 24);
    limbs_to_mpz(px, px_limbs, 24);
    limbs_to_mpz(py, py_limbs, 24);
    limbs_to_mpz(k, scalar_copy, 12);

    bool cpu_inf = gmp_scalar_mul_stop_at(rx_cpu, ry_cpu, px, py, k, P, stop_at_bit);
    uint32_t cpu_x[24], cpu_y[24];
    mpz_to_limbs(rx_cpu, cpu_x, 24);
    mpz_to_limbs(ry_cpu, cpu_y, 24);

    memcpy(test.px._limbs, px_limbs, 24 * sizeof(uint32_t));
    memcpy(test.py._limbs, py_limbs, 24 * sizeof(uint32_t));
    memcpy(test.scalar, scalar_limbs, 12 * sizeof(uint32_t));
    test.stop_at_bit = stop_at_bit;
    test.capture_iteration = stop_at_bit;
    test.capture_phase = phase;
    test.capture_valid = 0;
    test.capture_bit_set = 0;
    test.capture_pdiff_zero = 0;
    test.capture_r_zero = 0;
    memset(test.result_x._limbs, 0, sizeof(test.result_x._limbs));
    memset(test.result_y._limbs, 0, sizeof(test.result_y._limbs));
    test.result_infinity = 0;

    CUDA_CHECK(cudaMemcpy(gpu_test, &test, sizeof(probe_test_t), cudaMemcpyHostToDevice));
    probe_kernel<<<1, TPI>>>(report, gpu_test);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);
    CUDA_CHECK(cudaMemcpy(&test, gpu_test, sizeof(probe_test_t), cudaMemcpyDeviceToHost));

    bool gpu_inf = test.result_infinity != 0;
    bool match = true;

    if (cpu_inf != gpu_inf) {
        match = false;
    } else if (!cpu_inf) {
        for (int i = 0; i < 24; i++) {
            if (test.result_x._limbs[i] != cpu_x[i] || test.result_y._limbs[i] != cpu_y[i]) {
                match = false;
                break;
            }
        }
    }

    if (dump) {
        printf("\n--- %s ---\n", dump_label);
        printf("capture_valid=%u, bit_set=%u, Pdiff_zero=%u, R_zero=%u, gpu_inf=%u, cpu_inf=%u\n",
               test.capture_valid, test.capture_bit_set, test.capture_pdiff_zero,
               test.capture_r_zero, gpu_inf ? 1u : 0u, cpu_inf ? 1u : 0u);
        if (test.capture_valid) {
            print_limbs_full("XYZZ.X (mont)", test.capture_x._limbs, 24);
            print_limbs_full("XYZZ.Y (mont)", test.capture_y._limbs, 24);
            print_limbs_full("XYZZ.ZZ (mont)", test.capture_zz._limbs, 24);
            print_limbs_full("XYZZ.ZZZ (mont)", test.capture_zzz._limbs, 24);
            print_limbs_full("GPU q_minus_x3 (mont)", test.capture_q_minus_x3._limbs, 24);
            print_limbs_full("GPU y1*PPP (mont)", test.capture_y1_ppp._limbs, 24);
            print_limbs_full("GPU R*(Q-X3) pre-sub (mont)", test.capture_r_times_qmx_pre._limbs, 24);
            print_limbs_full("GPU Y3 post-sub (mont)", test.capture_r_times_qmx._limbs, 24);
        } else {
            printf("No capture data recorded.\n");
        }
        print_limbs_full("GPU affine X", test.result_x._limbs, 24);
        print_limbs_full("GPU affine Y", test.result_y._limbs, 24);
        print_limbs_full("CPU affine X", cpu_x, 24);
        print_limbs_full("CPU affine Y", cpu_y, 24);

        if (test.capture_valid) {
            mpz_t p_mod, r_pre, y1_ppp, y3_post, tmp;
            mpz_init(p_mod);
            mpz_init(r_pre);
            mpz_init(y1_ppp);
            mpz_init(y3_post);
            mpz_init(tmp);

            limbs_to_mpz(p_mod, BW6_761_P_HOST, 24);
            limbs_to_mpz(r_pre, test.capture_r_times_qmx_pre._limbs, 24);
            limbs_to_mpz(y1_ppp, test.capture_y1_ppp._limbs, 24);
            limbs_to_mpz(y3_post, test.capture_r_times_qmx._limbs, 24);

            mpz_sub(tmp, r_pre, y1_ppp);
            mpz_mod(tmp, tmp, p_mod);
            bool sub_ok = (mpz_cmp(tmp, y3_post) == 0);

            mpz_add(tmp, r_pre, y1_ppp);
            mpz_mod(tmp, tmp, p_mod);
            bool add_ok = (mpz_cmp(tmp, y3_post) == 0);

            printf("Check mont relation: Y3 == R*(Q-X3) - Y1*PPP ? %s\n", sub_ok ? "YES" : "NO");
            printf("Check mont relation: Y3 == R*(Q-X3) + Y1*PPP ? %s\n", add_ok ? "YES" : "NO");

            mpz_clear(p_mod);
            mpz_clear(r_pre);
            mpz_clear(y1_ppp);
            mpz_clear(y3_post);
            mpz_clear(tmp);
        }
        printf("--- end %s ---\n", dump_label);
    }

    if (out_bit_set) *out_bit_set = test.capture_bit_set;
    if (out_pdiff_zero) *out_pdiff_zero = test.capture_pdiff_zero;
    if (out_r_zero) *out_r_zero = test.capture_r_zero;

    mpz_clear(P); mpz_clear(px); mpz_clear(py); mpz_clear(k);
    mpz_clear(rx_cpu); mpz_clear(ry_cpu);
    return match;
}

//-----------------------------------------------------------------------------
// Binary search for first divergent bit
//-----------------------------------------------------------------------------

int find_first_divergent_bit(probe_test_t *gpu_test, cgbn_error_report_t *report,
                              uint32_t *px, uint32_t *py, uint32_t *scalar) {
    int highest_bit = find_highest_bit(scalar);

    int max_bits = highest_bit; // number of iterations = highest_bit (from MSB-1 to 0)
    printf("Scalar has %d bits, max iterations = %d\n", highest_bit + 1, max_bits);

    // First verify that full scalar fails
    if (compare_at_bit(max_bits, gpu_test, report, px, py, scalar)) {
        printf("UNEXPECTED: Full scalar passes! No divergence.\n");
        return -1;
    }
    printf("Confirmed: Full scalar fails (GPU != CPU)\n\n");

    printf("Scanning for first divergent bit...\n");
    for (int i = 1; i <= max_bits; i++) {
        bool match = compare_at_bit(i, gpu_test, report, px, py, scalar);
        printf("  stop_at_bit=%3d: %s\n", i, match ? "MATCH" : "DIVERGE");
        if (!match) {
            return i;
        }
    }

    return -1;
}

//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------

int main() {
    printf("=== BW6-761 Probe-Bit Test ===\n");
    printf("Built: %s %s\n\n", __DATE__, __TIME__);
    printf("Finding the first bit where GPU diverges from CPU...\n\n");

    probe_test_t *gpu_test;
    cgbn_error_report_t *report;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void **)&gpu_test, sizeof(probe_test_t)));
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    printf("=== Testing with FAILING scalar (limbs 3-10,11) ===\n");
    int first_div = find_first_divergent_bit(gpu_test, report, POINT131_X, POINT131_Y, FAILING_SCALAR);

    if (first_div > 0) {
        printf("\n=== RESULT ===\n");
        printf("First divergence at iteration %d (after processing %d bits from MSB)\n", first_div, first_div);

        // Calculate which scalar bit this corresponds to
        int highest_bit = -1;
        for (int limb_idx = 11; limb_idx >= 0; limb_idx--) {
            if (FAILING_SCALAR[limb_idx] != 0) {
                for (int bit_idx = 31; bit_idx >= 0; bit_idx--) {
                    if (FAILING_SCALAR[limb_idx] & (1u << bit_idx)) {
                        highest_bit = limb_idx * 32 + bit_idx;
                        break;
                    }
                    if (highest_bit >= 0) break;
                }
                if (highest_bit >= 0) break;
            }
        }

        int divergent_bit_pos = highest_bit - first_div;
        int divergent_limb = divergent_bit_pos / 32;
        int divergent_bit_in_limb = divergent_bit_pos % 32;

        printf("This corresponds to scalar bit %d (limb %d, bit %d)\n",
               divergent_bit_pos, divergent_limb, divergent_bit_in_limb);
        printf("Scalar limb %d value: 0x%08x\n", divergent_limb, FAILING_SCALAR[divergent_limb]);

        // Check if this bit is set in the scalar
        bool bit_set = (FAILING_SCALAR[divergent_limb] & (1u << divergent_bit_in_limb)) != 0;
        printf("This bit is %s in the scalar\n", bit_set ? "SET (add operation)" : "CLEAR (no add)");

        // Verify by checking iteration-1
        printf("\nVerifying: iteration %d should MATCH, iteration %d should DIVERGE\n", first_div - 1, first_div);
        bool prev_match = compare_at_bit(first_div - 1, gpu_test, report, POINT131_X, POINT131_Y, FAILING_SCALAR);
        bool curr_match = compare_at_bit(first_div, gpu_test, report, POINT131_X, POINT131_Y, FAILING_SCALAR);
        printf("  iteration %d: %s\n", first_div - 1, prev_match ? "MATCH (correct)" : "DIVERGE (unexpected!)");
        printf("  iteration %d: %s\n", first_div, curr_match ? "MATCH (unexpected!)" : "DIVERGE (correct)");

        printf("\n=== Probe at iteration %d ===\n", first_div);
        uint32_t capture_bit_set = 0;
        uint32_t pdiff_zero = 0;
        uint32_t r_zero = 0;
        bool pre_match = compare_phase_at_bit(first_div, 0, gpu_test, report,
                                              POINT131_X, POINT131_Y, FAILING_SCALAR,
                                              "PRE-ADD (after double)", true,
                                              &capture_bit_set, &pdiff_zero, &r_zero);
        printf("Pre-add (after double, before add): %s\n", pre_match ? "MATCH" : "DIVERGE");
        bool post_match = compare_phase_at_bit(first_div, 1, gpu_test, report,
                                               POINT131_X, POINT131_Y, FAILING_SCALAR,
                                               "POST-ADD (after add)", true,
                                               &capture_bit_set, &pdiff_zero, &r_zero);
        printf("Post-add (after add): %s\n", post_match ? "MATCH" : "DIVERGE");
        printf("GPU flags at iteration %d: bit_set=%u, Pdiff_zero=%u, R_zero=%u\n",
               first_div, capture_bit_set, pdiff_zero, r_zero);
    } else {
        printf("\nNo divergence detected (all iterations matched).\n");
    }

    CUDA_CHECK(cudaFree(gpu_test));
    CUDA_CHECK(cgbn_error_report_free(report));

    return 0;
}
