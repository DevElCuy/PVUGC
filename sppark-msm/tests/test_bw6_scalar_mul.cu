/***
 * Standalone CUDA Scalar Multiplication Test for BW6-761
 *
 * Phase 3 of BW6-761 scalar multiplication bug investigation.
 * Tests scalar multiplication with step-by-step tracing to find
 * the exact bit position where GPU diverges from reference.
 *
 * Compile: nvcc -o test_bw6_scalar_mul test_bw6_scalar_mul.cu \
 *              -I../src -I../../cgbn-lib/include -lgmp -arch=sm_75
 * Run: ./test_bw6_scalar_mul
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
#define SCALAR_BITS 384  // Fr is 377 bits, padded to 384

// BW6-761 base field modulus P
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

// BW6-761 G1 Generator
uint32_t BW6_761_GEN_X_HOST[24] = {
    0x66e5b43d, 0x4088f3af, 0xa6af603f, 0x055928ac,
    0x56133e82, 0x6750dd03, 0x280ca27f, 0x03758f9a,
    0xc9ea0971, 0x5bd71fa0, 0x47729b90, 0xa17a54ce,
    0x94c2e746, 0x11dbfcd2, 0xc15520ac, 0x79017ffa,
    0x85f56fc7, 0xee05c54b, 0x551b27f0, 0xe6a0cfb7,
    0xa477beae, 0xb277ce98, 0x0ea190c8, 0x01075b02
};

uint32_t BW6_761_GEN_Y_HOST[24] = {
    0xb4e95363, 0xbafc8f2d, 0x0b20d2a1, 0xad1cb2be,
    0xcad0fb93, 0xb2b08119, 0xb3053253, 0x9f9df141,
    0x6fc2cdd4, 0xbe3fb90b, 0x717a4c55, 0xcc685d31,
    0x71b5b806, 0xc5b8fa17, 0xaf7e0dba, 0x265909f1,
    0xa2e573a3, 0x1a7348d2, 0x884c9ec6, 0x0f952589,
    0x45cc2a42, 0xe6fd637b, 0x0a6fc574, 0x0058b84e
};

// Test data structure for scalar mul
typedef struct {
    cgbn_mem_t<BITS> px;           // Point X (affine)
    cgbn_mem_t<BITS> py;           // Point Y (affine)
    uint32_t scalar[12];           // Scalar (377 bits = 12 u32)
    cgbn_mem_t<BITS> expected_x;   // Expected result X (affine)
    cgbn_mem_t<BITS> expected_y;   // Expected result Y
    cgbn_mem_t<BITS> result_x;     // GPU result X
    cgbn_mem_t<BITS> result_y;     // GPU result Y
} scalar_mul_test_t;

// CGBN types
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
// GMP elliptic curve scalar multiplication (reference)
//-----------------------------------------------------------------------------

void gmp_mod_inverse(mpz_t result, mpz_t a, mpz_t p) {
    mpz_invert(result, a, p);
}

// Point doubling: 2P for y^2 = x^3 + b (a=0)
void gmp_point_double(mpz_t x3, mpz_t y3,
                       mpz_t x1, mpz_t y1,
                       mpz_t p) {
    mpz_t lambda, temp, temp2, x1_sq;
    mpz_init(lambda);
    mpz_init(temp);
    mpz_init(temp2);
    mpz_init(x1_sq);

    // lambda = 3*x1^2 / (2*y1)
    mpz_mul(x1_sq, x1, x1);
    mpz_mod(x1_sq, x1_sq, p);
    mpz_mul_ui(temp, x1_sq, 3);
    mpz_mod(temp, temp, p);

    mpz_mul_ui(temp2, y1, 2);
    mpz_mod(temp2, temp2, p);
    gmp_mod_inverse(temp2, temp2, p);
    mpz_mul(lambda, temp, temp2);
    mpz_mod(lambda, lambda, p);

    // x3 = lambda^2 - 2*x1
    mpz_mul(x3, lambda, lambda);
    mpz_mod(x3, x3, p);
    mpz_mul_ui(temp, x1, 2);
    mpz_mod(temp, temp, p);
    mpz_sub(x3, x3, temp);
    mpz_mod(x3, x3, p);

    // y3 = lambda*(x1 - x3) - y1
    mpz_sub(temp, x1, x3);
    mpz_mod(temp, temp, p);
    mpz_mul(y3, lambda, temp);
    mpz_mod(y3, y3, p);
    mpz_sub(y3, y3, y1);
    mpz_mod(y3, y3, p);

    mpz_clear(lambda);
    mpz_clear(temp);
    mpz_clear(temp2);
    mpz_clear(x1_sq);
}

// Point addition: P1 + P2
void gmp_point_add(mpz_t x3, mpz_t y3,
                    mpz_t x1, mpz_t y1,
                    mpz_t x2, mpz_t y2,
                    mpz_t p) {
    // Check if same point (use doubling)
    if (mpz_cmp(x1, x2) == 0 && mpz_cmp(y1, y2) == 0) {
        gmp_point_double(x3, y3, x1, y1, p);
        return;
    }

    mpz_t lambda, temp, temp2;
    mpz_init(lambda);
    mpz_init(temp);
    mpz_init(temp2);

    // lambda = (y2 - y1) / (x2 - x1)
    mpz_sub(temp, y2, y1);
    mpz_mod(temp, temp, p);
    mpz_sub(temp2, x2, x1);
    mpz_mod(temp2, temp2, p);
    gmp_mod_inverse(temp2, temp2, p);
    mpz_mul(lambda, temp, temp2);
    mpz_mod(lambda, lambda, p);

    // x3 = lambda^2 - x1 - x2
    mpz_mul(x3, lambda, lambda);
    mpz_mod(x3, x3, p);
    mpz_sub(x3, x3, x1);
    mpz_mod(x3, x3, p);
    mpz_sub(x3, x3, x2);
    mpz_mod(x3, x3, p);

    // y3 = lambda*(x1 - x3) - y1
    mpz_sub(temp, x1, x3);
    mpz_mod(temp, temp, p);
    mpz_mul(y3, lambda, temp);
    mpz_mod(y3, y3, p);
    mpz_sub(y3, y3, y1);
    mpz_mod(y3, y3, p);

    mpz_clear(lambda);
    mpz_clear(temp);
    mpz_clear(temp2);
}

// Scalar multiplication: k * P using double-and-add
void gmp_scalar_mul(mpz_t rx, mpz_t ry,
                     mpz_t px, mpz_t py,
                     mpz_t k, mpz_t p) {
    if (mpz_cmp_ui(k, 0) == 0) {
        mpz_set_ui(rx, 0);
        mpz_set_ui(ry, 0);
        return;
    }

    // Find highest bit
    int highest_bit = mpz_sizeinbase(k, 2) - 1;

    // Initialize with P
    mpz_t acc_x, acc_y, temp_x, temp_y;
    mpz_init_set(acc_x, px);
    mpz_init_set(acc_y, py);
    mpz_init(temp_x);
    mpz_init(temp_y);

    // Double-and-add from MSB-1 down to 0
    for (int i = highest_bit - 1; i >= 0; i--) {
        // Double
        gmp_point_double(temp_x, temp_y, acc_x, acc_y, p);
        mpz_set(acc_x, temp_x);
        mpz_set(acc_y, temp_y);

        // Add if bit is set
        if (mpz_tstbit(k, i)) {
            gmp_point_add(temp_x, temp_y, acc_x, acc_y, px, py, p);
            mpz_set(acc_x, temp_x);
            mpz_set(acc_y, temp_y);
        }
    }

    mpz_set(rx, acc_x);
    mpz_set(ry, acc_y);

    mpz_clear(acc_x);
    mpz_clear(acc_y);
    mpz_clear(temp_x);
    mpz_clear(temp_y);
}

//-----------------------------------------------------------------------------
// CUDA Kernel - Scalar Multiplication (matching msm_bw6_761_cgbn.cu)
//-----------------------------------------------------------------------------

__global__ void test_scalar_mul_kernel(cgbn_error_report_t *report,
                                        scalar_mul_test_t *tests,
                                        uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (instance >= count) return;

    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t bn_env(bn_context.env<env_t>());
    typedef env_t::cgbn_t bn_t;

    bn_t P, R2, one, one_m;
    cgbn_load(bn_env, P, (cgbn_mem_t<BITS>*)BW6_761_P_DEVICE);
    cgbn_load(bn_env, R2, (cgbn_mem_t<BITS>*)BW6_761_R2_DEVICE);
    cgbn_set_ui32(bn_env, one, 1);
    cgbn_mont_mul(bn_env, one_m, one, R2, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, one_m, P) >= 0) {
        cgbn_sub(bn_env, one_m, one_m, P);
    }

    // Load point (affine, plain form)
    bn_t px, py;
    cgbn_load(bn_env, px, &(tests[instance].px));
    cgbn_load(bn_env, py, &(tests[instance].py));

    // Convert to Montgomery
    bn_t px_m, py_m;
    cgbn_mont_mul(bn_env, px_m, px, R2, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, px_m, P) >= 0) {
        cgbn_sub(bn_env, px_m, px_m, P);
    }
    cgbn_mont_mul(bn_env, py_m, py, R2, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, py_m, P) >= 0) {
        cgbn_sub(bn_env, py_m, py_m, P);
    }

    // Get scalar
    uint32_t *scalar = tests[instance].scalar;

    // Check for zero scalar
    bool scalar_zero = true;
    for (int i = 0; i < 12; i++) {
        if (scalar[i] != 0) { scalar_zero = false; break; }
    }
    if (scalar_zero) {
        // Return identity (store zeros)
        bn_t zero;
        cgbn_set_ui32(bn_env, zero, 0);
        cgbn_store(bn_env, &(tests[instance].result_x), zero);
        cgbn_store(bn_env, &(tests[instance].result_y), zero);
        return;
    }

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

    // Initialize accumulator with point (XYZZ with ZZ=1, ZZZ=1 in Montgomery)
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

    // CRITICAL FIX: CGBN's cgbn_mont_mul can return values in [0, 2P) instead
    // of strictly [0, P). We must add explicit reduction after each Montgomery
    // multiplication. See CGBN issue #15.
    #define field_mul(r, a, b) do { \
        cgbn_mont_mul(bn_env, r, a, b, P, BW6_761_NP0); \
        if (cgbn_compare(bn_env, r, P) >= 0) { \
            cgbn_sub(bn_env, r, r, P); \
        } \
    } while(0)

    // Double-and-add from MSB-1 down to 0
    for (int bit_pos = highest_bit - 1; bit_pos >= 0; bit_pos--) {
        //---------------------------------------------------------------------
        // Point Doubling (XYZZ)
        //---------------------------------------------------------------------
        bn_t A, V, U, W, S, M, temp;

        // A = Y1^2
        field_mul(A, acc_y, acc_y);

        // U = 2*Y1
        field_add(U, acc_y, acc_y);

        // V = 4*A
        field_add(V, A, A);
        field_add(V, V, V);

        // W = U*V
        field_mul(W, U, V);

        // S = X1*V
        field_mul(S, acc_x, V);

        // M = 3*X1^2
        field_mul(M, acc_x, acc_x);
        field_add(temp, M, M);
        field_add(M, M, temp);

        // X3 = M^2 - 2*S
        field_mul(temp_x, M, M);
        field_sub(temp_x, temp_x, S);
        field_sub(temp_x, temp_x, S);

        // Y3 = M*(S - X3) - 2*A*V
        field_sub(temp, S, temp_x);
        field_mul(temp_y, M, temp);
        field_mul(temp, A, V);
        field_add(temp, temp, temp);
        field_sub(temp_y, temp_y, temp);

        // ZZ3 = V*ZZ1
        field_mul(temp_zz, V, acc_zz);

        // ZZZ3 = W*ZZZ1
        field_mul(temp_zzz, W, acc_zzz);

        cgbn_set(bn_env, acc_x, temp_x);
        cgbn_set(bn_env, acc_y, temp_y);
        cgbn_set(bn_env, acc_zz, temp_zz);
        cgbn_set(bn_env, acc_zzz, temp_zzz);

        // Check if bit is set
        int limb_idx = bit_pos / 32;
        int bit_idx = bit_pos % 32;
        bool bit_set = (scalar[limb_idx] & (1u << bit_idx)) != 0;

        if (bit_set) {
            //---------------------------------------------------------------------
            // Point Addition (XYZZ + Affine)
            //---------------------------------------------------------------------
            bn_t U2, S2, Pdiff, R, PP, PPP, Q, temp2;

            // U2 = X2 * ZZ1
            field_mul(U2, px_m, acc_zz);

            // S2 = Y2 * ZZZ1
            field_mul(S2, py_m, acc_zzz);

            // Pdiff = U2 - X1
            field_sub(Pdiff, U2, acc_x);

            // R = S2 - Y1
            field_sub(R, S2, acc_y);

            // PP = Pdiff^2
            field_mul(PP, Pdiff, Pdiff);

            // PPP = Pdiff * PP
            field_mul(PPP, Pdiff, PP);

            // Q = X1 * PP
            field_mul(Q, acc_x, PP);

            // X3 = R^2 - PPP - 2*Q
            field_mul(temp_x, R, R);
            field_sub(temp_x, temp_x, PPP);
            field_sub(temp_x, temp_x, Q);
            field_sub(temp_x, temp_x, Q);

            // Y3 = R*(Q - X3) - Y1*PPP
            field_sub(temp, Q, temp_x);
            field_mul(temp2, acc_y, PPP);
            field_mul(temp_y, R, temp);
            field_sub(temp_y, temp_y, temp2);

            // ZZ3 = ZZ1 * PP
            field_mul(temp_zz, acc_zz, PP);

            // ZZZ3 = ZZZ1 * PPP
            field_mul(temp_zzz, acc_zzz, PPP);

            cgbn_set(bn_env, acc_x, temp_x);
            cgbn_set(bn_env, acc_y, temp_y);
            cgbn_set(bn_env, acc_zz, temp_zz);
            cgbn_set(bn_env, acc_zzz, temp_zzz);
        }
    }

    //-------------------------------------------------------------------------
    // Convert from XYZZ to affine
    //-------------------------------------------------------------------------
    bn_t ZZ_plain, ZZZ_plain;
    cgbn_mont_mul(bn_env, ZZ_plain, acc_zz, one, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, ZZ_plain, P) >= 0) {
        cgbn_sub(bn_env, ZZ_plain, ZZ_plain, P);
    }
    cgbn_mont_mul(bn_env, ZZZ_plain, acc_zzz, one, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, ZZZ_plain, P) >= 0) {
        cgbn_sub(bn_env, ZZZ_plain, ZZZ_plain, P);
    }

    bn_t ZZ_inv, ZZZ_inv;
    cgbn_modular_inverse(bn_env, ZZ_inv, ZZ_plain, P);
    cgbn_modular_inverse(bn_env, ZZZ_inv, ZZZ_plain, P);

    bn_t ZZ_inv_m, ZZZ_inv_m;
    cgbn_mont_mul(bn_env, ZZ_inv_m, ZZ_inv, R2, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, ZZ_inv_m, P) >= 0) {
        cgbn_sub(bn_env, ZZ_inv_m, ZZ_inv_m, P);
    }
    cgbn_mont_mul(bn_env, ZZZ_inv_m, ZZZ_inv, R2, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, ZZZ_inv_m, P) >= 0) {
        cgbn_sub(bn_env, ZZZ_inv_m, ZZZ_inv_m, P);
    }

    bn_t x_aff_m, y_aff_m;
    field_mul(x_aff_m, acc_x, ZZ_inv_m);
    field_mul(y_aff_m, acc_y, ZZZ_inv_m);

    bn_t x_aff, y_aff;
    cgbn_mont_mul(bn_env, x_aff, x_aff_m, one, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, x_aff, P) >= 0) {
        cgbn_sub(bn_env, x_aff, x_aff, P);
    }
    cgbn_mont_mul(bn_env, y_aff, y_aff_m, one, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(bn_env, y_aff, P) >= 0) {
        cgbn_sub(bn_env, y_aff, y_aff, P);
    }

    cgbn_store(bn_env, &(tests[instance].result_x), x_aff);
    cgbn_store(bn_env, &(tests[instance].result_y), y_aff);

    #undef field_add
    #undef field_sub
    #undef field_mul
}

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------

bool compare_limbs(const uint32_t *a, const uint32_t *b, int count) {
    for (int i = 0; i < count; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void print_limbs_short(const char *label, const uint32_t *limbs, int count) {
    printf("%s: %08x%08x%08x...%08x%08x%08x\n", label,
           limbs[count-1], limbs[count-2], limbs[count-3],
           limbs[2], limbs[1], limbs[0]);
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CGBN_CHECK(report) \
    do { \
        if (cgbn_error_report_check(report)) { \
            fprintf(stderr, "CGBN error detected\n"); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//-----------------------------------------------------------------------------
// Run a single scalar mul test
//-----------------------------------------------------------------------------

bool run_scalar_mul_test(const char *name,
                          uint32_t *px_limbs, uint32_t *py_limbs,
                          uint32_t *scalar_limbs,
                          scalar_mul_test_t *gpu_test,
                          cgbn_error_report_t *report) {
    printf("\n=== Test: %s ===\n", name);

    scalar_mul_test_t test;
    mpz_t P, px, py, k, rx, ry;

    mpz_init(P);
    mpz_init(px);
    mpz_init(py);
    mpz_init(k);
    mpz_init(rx);
    mpz_init(ry);

    limbs_to_mpz(P, BW6_761_P_HOST, 24);
    limbs_to_mpz(px, px_limbs, 24);
    limbs_to_mpz(py, py_limbs, 24);
    limbs_to_mpz(k, scalar_limbs, 12);

    printf("Scalar bits: %zu\n", mpz_sizeinbase(k, 2));

    // Compute reference with GMP
    gmp_scalar_mul(rx, ry, px, py, k, P);

    // Prepare test data
    memcpy(test.px._limbs, px_limbs, 24 * sizeof(uint32_t));
    memcpy(test.py._limbs, py_limbs, 24 * sizeof(uint32_t));
    memcpy(test.scalar, scalar_limbs, 12 * sizeof(uint32_t));
    mpz_to_limbs(rx, test.expected_x._limbs, 24);
    mpz_to_limbs(ry, test.expected_y._limbs, 24);
    memset(test.result_x._limbs, 0, sizeof(test.result_x._limbs));
    memset(test.result_y._limbs, 0, sizeof(test.result_y._limbs));

    // Run GPU kernel
    CUDA_CHECK(cudaMemcpy(gpu_test, &test, sizeof(scalar_mul_test_t), cudaMemcpyHostToDevice));
    test_scalar_mul_kernel<<<1, TPI>>>(report, gpu_test, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);
    CUDA_CHECK(cudaMemcpy(&test, gpu_test, sizeof(scalar_mul_test_t), cudaMemcpyDeviceToHost));

    bool x_ok = compare_limbs(test.result_x._limbs, test.expected_x._limbs, 24);
    bool y_ok = compare_limbs(test.result_y._limbs, test.expected_y._limbs, 24);

    if (x_ok && y_ok) {
        printf("Result: PASS\n");
    } else {
        printf("Result: FAIL\n");
        print_limbs_short("Expected X", test.expected_x._limbs, 24);
        print_limbs_short("Got X     ", test.result_x._limbs, 24);
        print_limbs_short("Expected Y", test.expected_y._limbs, 24);
        print_limbs_short("Got Y     ", test.result_y._limbs, 24);
    }

    mpz_clear(P);
    mpz_clear(px);
    mpz_clear(py);
    mpz_clear(k);
    mpz_clear(rx);
    mpz_clear(ry);

    return x_ok && y_ok;
}

//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------

int main() {
    printf("=== BW6-761 CUDA Scalar Multiplication Test (Phase 3) ===\n");

    // Setup CUDA
    scalar_mul_test_t *gpu_test;
    cgbn_error_report_t *report;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void **)&gpu_test, sizeof(scalar_mul_test_t)));
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    int pass_count = 0;
    int fail_count = 0;

    //=========================================================================
    // Test 1: Simple scalar k=2 (single double)
    //=========================================================================
    {
        uint32_t scalar[12] = {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("k=2 (single double)",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 2: k=3 (double + add)
    //=========================================================================
    {
        uint32_t scalar[12] = {3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("k=3 (double + add)",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 3: k=5 (101 binary: 2 doubles, 1 add)
    //=========================================================================
    {
        uint32_t scalar[12] = {5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("k=5",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 4: k=7 (111 binary: many adds)
    //=========================================================================
    {
        uint32_t scalar[12] = {7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("k=7",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 5: k=255 (8 bits, all 1s)
    //=========================================================================
    {
        uint32_t scalar[12] = {255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("k=255 (8 bits all 1s)",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 6: k=2^64 (power of 2, only doublings)
    //=========================================================================
    {
        uint32_t scalar[12] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // 2^64
        if (run_scalar_mul_test("k=2^64 (only doublings)",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 7: k=2^128
    //=========================================================================
    {
        uint32_t scalar[12] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};  // 2^128
        if (run_scalar_mul_test("k=2^128",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 8: k=2^256
    //=========================================================================
    {
        uint32_t scalar[12] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0};  // 2^256
        if (run_scalar_mul_test("k=2^256",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 9: k=2^370 (close to scalar field size)
    //=========================================================================
    {
        uint32_t scalar[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000400};  // 2^370
        if (run_scalar_mul_test("k=2^370",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 10: Large scalar with many 1 bits (alternating pattern)
    //=========================================================================
    {
        uint32_t scalar[12] = {0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA,
                               0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA,
                               0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0x0000000A};
        if (run_scalar_mul_test("alternating bits (0xAAA...)",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 11: Large scalar - all 1s in first 64 bits
    //=========================================================================
    {
        uint32_t scalar[12] = {0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("64 bits all 1s",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 12: Large scalar - consecutive 1s pattern (known to stress add)
    //=========================================================================
    {
        uint32_t scalar[12] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                               0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                               0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00001FFF}; // 373 bits all 1s
        if (run_scalar_mul_test("373 bits all 1s",
                                 BW6_761_GEN_X_HOST, BW6_761_GEN_Y_HOST,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 13: Point[131] with k=2 (simple double)
    //=========================================================================
    {
        // Point 131 X coordinate
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15,
            0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f,
            0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737,
            0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc,
            0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c,
            0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1,
            0x092cc535, 0x00082ce3
        };

        // Point 131 Y coordinate
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac,
            0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6,
            0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8,
            0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a,
            0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d,
            0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a,
            0xc2434bee, 0x00b9bcba
        };

        uint32_t scalar_2[12] = {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("Point[131] * 2",
                                 point131_x, point131_y,
                                 scalar_2, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 14: Point[131] with k=3
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15,
            0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f,
            0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737,
            0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc,
            0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c,
            0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1,
            0x092cc535, 0x00082ce3
        };

        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac,
            0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6,
            0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8,
            0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a,
            0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d,
            0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a,
            0xc2434bee, 0x00b9bcba
        };

        uint32_t scalar_3[12] = {3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("Point[131] * 3",
                                 point131_x, point131_y,
                                 scalar_3, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 15: Point[131] with k=255
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15,
            0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f,
            0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737,
            0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc,
            0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c,
            0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1,
            0x092cc535, 0x00082ce3
        };

        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac,
            0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6,
            0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8,
            0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a,
            0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d,
            0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a,
            0xc2434bee, 0x00b9bcba
        };

        uint32_t scalar_255[12] = {255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("Point[131] * 255",
                                 point131_x, point131_y,
                                 scalar_255, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 16: Point[131] * scalar masked to 256 bits
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // 256 bits
        uint32_t scalar_256[12] = {0x0b7bc8d4, 0x1e71ee0d, 0x50a8aa7b, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
        if (run_scalar_mul_test("Point[131] * scalar (256 bits)",
                                 point131_x, point131_y,
                                 scalar_256, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 17: Point[131] * scalar masked to 320 bits
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // 319 bits
        uint32_t scalar_320[12] = {0x0b7bc8d4, 0x1e71ee0d, 0x50a8aa7b, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0x00000000, 0x00000000};
        if (run_scalar_mul_test("Point[131] * scalar (319 bits)",
                                 point131_x, point131_y,
                                 scalar_320, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 18: Point[131] * scalar masked to 352 bits
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // 352 bits
        uint32_t scalar_352[12] = {0x0b7bc8d4, 0x1e71ee0d, 0x50a8aa7b, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x00000000};
        if (run_scalar_mul_test("Point[131] * scalar (352 bits)",
                                 point131_x, point131_y,
                                 scalar_352, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 19: Point[131] * scalar masked to 368 bits
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // 368 bits
        uint32_t scalar_368[12] = {0x0b7bc8d4, 0x1e71ee0d, 0x50a8aa7b, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0000ee43};
        if (run_scalar_mul_test("Point[131] * scalar (368 bits)",
                                 point131_x, point131_y,
                                 scalar_368, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 20: Point[131] * 2^368 (pure doublings, 369 iterations, bit 368 set only)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // 2^368 = only bit 368 set in limb 11 = 0x00010000
        uint32_t scalar_pow368[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00010000};
        if (run_scalar_mul_test("Point[131] * 2^368 (369 iterations)",
                                 point131_x, point131_y,
                                 scalar_pow368, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 21: Point[131] * 2^369 (pure doublings, 370 iterations, bit 369 set only)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // 2^369 = only bit 369 set in limb 11 = 0x00020000
        uint32_t scalar_pow369[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00020000};
        if (run_scalar_mul_test("Point[131] * 2^369 (370 iterations)",
                                 point131_x, point131_y,
                                 scalar_pow369, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 22: Point[131] * 2^370 (pure doublings, 371 iterations, bit 370 set only)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // 2^370 = only bit 370 set in limb 11 = 0x00040000
        uint32_t scalar_pow370[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00040000};
        if (run_scalar_mul_test("Point[131] * 2^370 (371 iterations)",
                                 point131_x, point131_y,
                                 scalar_pow370, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 23: Point[131] * (2^370 + 1) - 371 iterations with an add at the end
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // 2^370 + 1
        uint32_t scalar_pow370_plus1[12] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00040000};
        if (run_scalar_mul_test("Point[131] * (2^370 + 1)",
                                 point131_x, point131_y,
                                 scalar_pow370_plus1, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 24: Scalar with pattern 1,0,0,1,1,1 at top (bits 370-365)
    // This is the same pattern as the failing scalar's top bits
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // Pattern 1,0,0,1,1,1 = 0x27 << 13 in limb 11 = 0x0004e000
        // bit 370=1, 369=0, 368=0, 367=1, 366=1, 365=1
        uint32_t scalar_pattern[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0004e000};
        if (run_scalar_mul_test("Point[131] * (pattern 100111 at bits 370-365)",
                                 point131_x, point131_y,
                                 scalar_pattern, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 25: Incrementally add lower bits from the failing scalar
    // Start with just the high part
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // Just limb 11 from the failing scalar: 0x0004ee43
        uint32_t scalar_high_only[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (only top limb 0x0004ee43)",
                                 point131_x, point131_y,
                                 scalar_high_only, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 26: Add limb 10
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limb 10 + 11
        uint32_t scalar_two_limbs[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 10-11)",
                                 point131_x, point131_y,
                                 scalar_two_limbs, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 27: Add limbs 9-11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 9-11
        uint32_t scalar_3limbs[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 9-11)",
                                 point131_x, point131_y,
                                 scalar_3limbs, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 28: Add limbs 8-11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 8-11
        uint32_t scalar_4limbs[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 8-11)",
                                 point131_x, point131_y,
                                 scalar_4limbs, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 29: Add limbs 6-11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 6-11
        uint32_t scalar_6limbs[12] = {0, 0, 0, 0, 0, 0, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 6-11)",
                                 point131_x, point131_y,
                                 scalar_6limbs, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 30: Add limbs 4-11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 4-11
        uint32_t scalar_8limbs[12] = {0, 0, 0, 0, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 4-11)",
                                 point131_x, point131_y,
                                 scalar_8limbs, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31a: ONLY limb 3 (isolate if limb 3 is the issue)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // ONLY limb 3 = 0xfdffb2b6 (bits 96-127)
        uint32_t scalar_limb3_only[12] = {0, 0, 0, 0xfdffb2b6, 0, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("Point[131] * (only limb 3)",
                                 point131_x, point131_y,
                                 scalar_limb3_only, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31b: Limbs 3 and 11 (bottom vs top interaction)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3 and 11 (gap in between)
        uint32_t scalar_3_11[12] = {0, 0, 0, 0xfdffb2b6, 0, 0, 0, 0, 0, 0, 0, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3 and 11)",
                                 point131_x, point131_y,
                                 scalar_3_11, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31c: Limbs 3,4 and 11 - is limb 4 the key interaction?
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3, 4, 11
        uint32_t scalar_3_4_11[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0, 0, 0, 0, 0, 0, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3,4,11)",
                                 point131_x, point131_y,
                                 scalar_3_4_11, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31d: Limbs 3 and 4 only (no limb 11)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3 and 4 only
        uint32_t scalar_3_4[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0, 0, 0, 0, 0, 0, 0};
        if (run_scalar_mul_test("Point[131] * (limbs 3,4 only)",
                                 point131_x, point131_y,
                                 scalar_3_4, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31e: limbs 3,4,5,11 - gradual adding
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3,4,5,11
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0, 0, 0, 0, 0, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3,4,5,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31f: limbs 3-6,11 - more gradual
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3-6,11
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0, 0, 0, 0, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3-6,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31g: limbs 3-7,11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3-7,11
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0, 0, 0, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3-7,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31h: limbs 3-8,11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3-8,11
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0, 0, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3-8,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31i: limbs 3-9,11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3-9,11
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3-9,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31j: limbs 3-10,11 (same as 3-11, just for clarity)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3-10,11 (same as 3-11)
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3-10,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 31: Add limbs 3-11 (the FAILING combo)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 3-11
        uint32_t scalar_9limbs[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 3-11)",
                                 point131_x, point131_y,
                                 scalar_9limbs, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 33: scalar/3 (should PASS according to Rust test)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // scalar/3 computed from Python
        uint32_t scalar_div3[12] = {0xae54982c, 0xb9d09f58, 0xc5e33880, 0xaa550e3d,
                                     0x392a52a0, 0xa9a252e3, 0x4b8b05e8, 0xf7e8be3e,
                                     0x6c18e9f0, 0x20dba209, 0x3720e7b6, 0x0001a4c1};
        if (run_scalar_mul_test("Point[131] * (scalar/3)",
                                 point131_x, point131_y,
                                 scalar_div3, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 34: Just limbs 10-11 of the ORIGINAL failing scalar
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // Just limbs 10 and 11 from the failing scalar
        uint32_t scalar_10_11[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (only limbs 10,11)",
                                 point131_x, point131_y,
                                 scalar_10_11, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 35: Just limb 10 of the failing scalar (no limb 11)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // Just limb 10
        uint32_t scalar_10_only[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xa562b722, 0};
        if (run_scalar_mul_test("Point[131] * (only limb 10)",
                                 point131_x, point131_y,
                                 scalar_10_only, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 36: limbs 4-10,11 (without limb 3)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 4-10,11 (no limb 3)
        uint32_t scalar_test[12] = {0, 0, 0, 0, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 4-10,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 37: limbs 2-11 (adding limb 2 to the failing combo)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limbs 2-11
        uint32_t scalar_10limbs[12] = {0, 0, 0x50a8aa7b, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limbs 2-11)",
                                 point131_x, point131_y,
                                 scalar_10limbs, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 38: Narrow limb 3 - just HIGH 16 bits + limbs 4-10,11
    // limb 3 = 0xfdffb2b6, high 16 bits = 0xfdff, low 16 = 0xb2b6
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limb 3 = HIGH 16 bits only (0xfdff0000) + limbs 4-10,11
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdff0000, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limb3 high16 + limbs 4-10,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 39: Narrow limb 3 - just LOW 16 bits + limbs 4-10,11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limb 3 = LOW 16 bits only (0x0000b2b6) + limbs 4-10,11
        uint32_t scalar_test[12] = {0, 0, 0, 0x0000b2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limb3 low16 + limbs 4-10,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 40: Narrow limb 10 - just HIGH 16 bits + limbs 3-9,11
    // limb 10 = 0xa562b722, high 16 = 0xa562, low 16 = 0xb722
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limb 10 = HIGH 16 bits only (0xa5620000) + limbs 3-9,11
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa5620000, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limb10 high16 + limbs 3-9,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 41: Narrow limb 10 - just LOW 16 bits + limbs 3-9,11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limb 10 = LOW 16 bits only (0x0000b722) + limbs 3-9,11
        uint32_t scalar_test[12] = {0, 0, 0, 0xfdffb2b6, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0x0000b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (limb10 low16 + limbs 3-9,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 42: Just bit 127 (top of limb 3) + limbs 4-10,11
    // limb 3 = 0xfdffb2b6 has bit 127 set (0x80000000 in limb 3 = bit 127)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // Just bit 127 (0x80000000 in limb 3) + limbs 4-10,11
        uint32_t scalar_test[12] = {0, 0, 0, 0x80000000, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (bit127 + limbs 4-10,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 43: Minimal failing case - just bit 96 (lsb of limb 3) + limbs 4-10,11
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // Just bit 96 (0x00000001 in limb 3) + limbs 4-10,11
        uint32_t scalar_test[12] = {0, 0, 0, 0x00000001, 0xad67f760, 0x7de78a8b, 0xe2a111b9, 0xe6ba3abb, 0x444b8bd1, 0x61c96712, 0xa562b722, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (bit96 + limbs 4-10,11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 44: limb3=1 + limb10 high bit + limb11 (minimal combo)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limb3=1, limb10=0x80000000 (just bit 351), limb11=0x0004ee43
        uint32_t scalar_test[12] = {0, 0, 0, 0x00000001, 0, 0, 0, 0, 0, 0, 0x80000000, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (bit96 + bit351 + limb11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 45: limb3=0x80000000 + limb10=0x80000000 + limb11 (high bits only)
    //=========================================================================
    {
        uint32_t point131_x[24] = {
            0x2967f6ff, 0xe5080f15, 0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f, 0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737, 0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc, 0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c, 0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1, 0x092cc535, 0x00082ce3
        };
        uint32_t point131_y[24] = {
            0x53ce4432, 0xeb08c1ac, 0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6, 0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8, 0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a, 0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d, 0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a, 0xc2434bee, 0x00b9bcba
        };
        // limb3=0x80000000 (bit 127), limb10=0x80000000 (bit 351), limb11=0x0004ee43
        uint32_t scalar_test[12] = {0, 0, 0, 0x80000000, 0, 0, 0, 0, 0, 0, 0x80000000, 0x0004ee43};
        if (run_scalar_mul_test("Point[131] * (bit127 + bit351 + limb11)",
                                 point131_x, point131_y,
                                 scalar_test, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Test 32: Full scalar for Point 131 (THE FAILING CASE)
    //=========================================================================
    {
        // Point 131 X coordinate
        uint32_t point_x[24] = {
            0x2967f6ff, 0xe5080f15,
            0x8b60ca1c, 0x5082f40d,
            0x24f3066f, 0x63c0b83f,
            0x77cada7a, 0xcb59a8e2,
            0xd326e822, 0x0d821737,
            0x4f96fc19, 0x3efbc544,
            0x7f65cfb8, 0x61b704dc,
            0x124b3a1b, 0x85cbc08b,
            0xfaf6458c, 0x5a96846c,
            0x5ade3520, 0x0cc7b85b,
            0xd9a556c0, 0x7f39cff1,
            0x092cc535, 0x00082ce3
        };

        // Point 131 Y coordinate
        uint32_t point_y[24] = {
            0x53ce4432, 0xeb08c1ac,
            0xe977f938, 0xa6b1c5b1,
            0xefae186a, 0xc01a47a6,
            0xc0f45c19, 0x0d0f5d36,
            0x204bd4fd, 0x6d07e8a8,
            0x5b78ea29, 0x9c7e60eb,
            0x6cdbd3b6, 0x3bf8036a,
            0xcc055f17, 0xe7682a84,
            0xaed1d4ed, 0xc8991c4d,
            0x7f9c8295, 0x5a9c4373,
            0xf2c60eaf, 0x5db4731a,
            0xc2434bee, 0x00b9bcba
        };

        // Scalar for point 131 - full 371 bits: 0x0004ee43
        uint32_t scalar[12] = {
            0x0b7bc8d4, 0x1e71ee0d,
            0x50a8aa7b, 0xfdffb2b6,
            0xad67f760, 0x7de78a8b,
            0xe2a111b9, 0xe6ba3abb,
            0x444b8bd1, 0x61c96712,
            0xa562b722, 0x0004ee43
        };

        if (run_scalar_mul_test("Point[131] * scalar[131] (371 bits - THE FAILING CASE)",
                                 point_x, point_y,
                                 scalar, gpu_test, report)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    //=========================================================================
    // Summary
    //=========================================================================
    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", pass_count);
    printf("Failed: %d\n", fail_count);

    if (fail_count == 0) {
        printf("\n[SUCCESS] All Phase 3 scalar multiplication tests passed!\n");
        printf("Basic scalar mul is correct. If bugs exist, they may be in:\n");
        printf("  - Specific point coordinates from arkworks random generation\n");
        printf("  - FFI/layout issues between Rust and CUDA\n");
    } else {
        printf("\n[FAILURE] Some tests failed. Scalar multiplication has bugs.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(gpu_test));
    CUDA_CHECK(cgbn_error_report_free(report));

    return fail_count > 0 ? 1 : 0;
}
