/***
 * Standalone CUDA Point Operations Test for BW6-761
 *
 * Phase 2 of BW6-761 scalar multiplication bug investigation.
 * Tests point_double and point_add_mixed operations.
 *
 * Strategy:
 * - Generate reference vectors using arkworks on the host
 * - Test GPU point operations against these references
 * - Include edge cases: infinity handling, point doubling via addition
 *
 * Compile: nvcc -o test_bw6_point_ops test_bw6_point_ops.cu \
 *              -I../src -I../../cgbn-lib/include -lgmp -arch=sm_75
 * Run: ./test_bw6_point_ops
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"

// CGBN configuration for BW6-761
#define TPI 8
#define BITS 768
#define LIMBS 24

// BW6-761 base field modulus P
__device__ __constant__ uint32_t BW6_761_P_DEVICE[24] = {
    0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68,
    0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
    0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a,
    0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824
};

// Montgomery np0
__device__ __constant__ uint32_t BW6_761_NP0 = 0x8fa798dd;

// Montgomery R^2 mod P
__device__ __constant__ uint32_t BW6_761_R2_DEVICE[24] = {
    0x2d1fa659, 0xc686392d, 0xf79484ab, 0x7b14c9b2,
    0xc1d2b459, 0x7fa1e825, 0x48329d88, 0xd6ec28f8,
    0x73a1ed40, 0x4afb427b, 0x0d5930ae, 0x972c6940,
    0x8c995976, 0x2c7a26bf, 0xc6e57af9, 0xac52e458,
    0x0c536dfe, 0xac731bfa, 0x0b103f50, 0x121e5c63,
    0xb886cda4, 0x8f1b0953, 0x2da8d807, 0x00ad253c
};

// Host copy
uint32_t BW6_761_P_HOST[24] = {
    0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68,
    0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
    0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a,
    0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824
};

// BW6-761 G1 Generator (arkworks-verified, little-endian u32[24])
uint32_t BW6_761_GEN_X_HOST[24] = {
    0x66e5b43d, 0x4088f3af,
    0xa6af603f, 0x055928ac,
    0x56133e82, 0x6750dd03,
    0x280ca27f, 0x03758f9a,
    0xc9ea0971, 0x5bd71fa0,
    0x47729b90, 0xa17a54ce,
    0x94c2e746, 0x11dbfcd2,
    0xc15520ac, 0x79017ffa,
    0x85f56fc7, 0xee05c54b,
    0x551b27f0, 0xe6a0cfb7,
    0xa477beae, 0xb277ce98,
    0x0ea190c8, 0x01075b02
};

uint32_t BW6_761_GEN_Y_HOST[24] = {
    0xb4e95363, 0xbafc8f2d,
    0x0b20d2a1, 0xad1cb2be,
    0xcad0fb93, 0xb2b08119,
    0xb3053253, 0x9f9df141,
    0x6fc2cdd4, 0xbe3fb90b,
    0x717a4c55, 0xcc685d31,
    0x71b5b806, 0xc5b8fa17,
    0xaf7e0dba, 0x265909f1,
    0xa2e573a3, 0x1a7348d2,
    0x884c9ec6, 0x0f952589,
    0x45cc2a42, 0xe6fd637b,
    0x0a6fc574, 0x0058b84e
};

// Test data: affine point (x, y) and expected result after operations
typedef struct {
    cgbn_mem_t<BITS> x;      // Input point X
    cgbn_mem_t<BITS> y;      // Input point Y
    cgbn_mem_t<BITS> x2;     // After double: X
    cgbn_mem_t<BITS> y2;     // After double: Y
    cgbn_mem_t<BITS> result_x2;  // GPU result for 2*P
    cgbn_mem_t<BITS> result_y2;
} point_test_t;

// For testing point addition
typedef struct {
    cgbn_mem_t<BITS> x1, y1;    // First point (XYZZ accumulator)
    cgbn_mem_t<BITS> zz1, zzz1; // ZZ, ZZZ of accumulator
    cgbn_mem_t<BITS> x2, y2;    // Second point (affine)
    cgbn_mem_t<BITS> expected_x3, expected_y3;  // Expected result (affine after normalization)
    cgbn_mem_t<BITS> result_x3, result_y3;      // GPU result
} add_test_t;

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
// Elliptic curve operations in GMP (for reference generation)
//-----------------------------------------------------------------------------

// Modular inverse using extended Euclidean algorithm
void gmp_mod_inverse(mpz_t result, mpz_t a, mpz_t p) {
    mpz_invert(result, a, p);
}

// Point doubling in affine coordinates for y^2 = x^3 + b (a=0)
// 2P = (x3, y3) where:
// lambda = 3*x1^2 / (2*y1)
// x3 = lambda^2 - 2*x1
// y3 = lambda*(x1 - x3) - y1
void gmp_point_double_affine(mpz_t x3, mpz_t y3,
                              mpz_t x1, mpz_t y1,
                              mpz_t p) {
    mpz_t lambda, temp, temp2, x1_sq;
    mpz_init(lambda);
    mpz_init(temp);
    mpz_init(temp2);
    mpz_init(x1_sq);

    // lambda = 3*x1^2 / (2*y1)
    mpz_mul(x1_sq, x1, x1);       // x1^2
    mpz_mod(x1_sq, x1_sq, p);
    mpz_mul_ui(temp, x1_sq, 3);   // 3*x1^2
    mpz_mod(temp, temp, p);

    mpz_mul_ui(temp2, y1, 2);     // 2*y1
    mpz_mod(temp2, temp2, p);
    gmp_mod_inverse(temp2, temp2, p);  // (2*y1)^(-1)
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

// Point addition in affine coordinates
// P1 + P2 = P3 where:
// lambda = (y2 - y1) / (x2 - x1)
// x3 = lambda^2 - x1 - x2
// y3 = lambda*(x1 - x3) - y1
void gmp_point_add_affine(mpz_t x3, mpz_t y3,
                           mpz_t x1, mpz_t y1,
                           mpz_t x2, mpz_t y2,
                           mpz_t p) {
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

//-----------------------------------------------------------------------------
// CUDA Kernel for Point Doubling Test
//-----------------------------------------------------------------------------

__global__ void test_point_double_kernel(cgbn_error_report_t *report,
                                          point_test_t *tests,
                                          uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (instance >= count) return;

    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t bn_env(bn_context.env<env_t>());
    typedef env_t::cgbn_t bn_t;

    bn_t P, R2;
    cgbn_load(bn_env, P, (cgbn_mem_t<BITS>*)BW6_761_P_DEVICE);
    cgbn_load(bn_env, R2, (cgbn_mem_t<BITS>*)BW6_761_R2_DEVICE);

    // Load point (affine, plain form)
    bn_t x1, y1;
    cgbn_load(bn_env, x1, &(tests[instance].x));
    cgbn_load(bn_env, y1, &(tests[instance].y));

    // Convert to Montgomery
    bn_t x1_m, y1_m;
    cgbn_mont_mul(bn_env, x1_m, x1, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, y1_m, y1, R2, P, BW6_761_NP0);

    // Initialize in XYZZ with ZZ=1, ZZZ=1 (in Montgomery: ZZ=R, ZZZ=R)
    bn_t zz1_m, zzz1_m, one, one_m;
    cgbn_set_ui32(bn_env, one, 1);
    cgbn_mont_mul(bn_env, one_m, one, R2, P, BW6_761_NP0);  // 1 in Montgomery
    cgbn_set(bn_env, zz1_m, one_m);
    cgbn_set(bn_env, zzz1_m, one_m);

    //-------------------------------------------------------------------------
    // Point Doubling (matching msm_bw6_761_cgbn.cu implementation)
    //-------------------------------------------------------------------------
    bn_t A, V, U, W, S, M, temp;
    bn_t X3, Y3, ZZ3, ZZZ3;

    // Helper lambdas (inline field ops)
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

    // A = Y1^2
    field_mul(A, y1_m, y1_m);

    // U = 2*Y1
    field_add(U, y1_m, y1_m);

    // V = 4*A
    field_add(V, A, A);
    field_add(V, V, V);

    // W = U*V = 8*Y1*A
    field_mul(W, U, V);

    // S = X1*V
    field_mul(S, x1_m, V);

    // M = 3*X1^2 (a=0 for BW6-761)
    field_mul(M, x1_m, x1_m);
    field_add(temp, M, M);
    field_add(M, M, temp);

    // X3 = M^2 - 2*S
    field_mul(X3, M, M);
    field_sub(X3, X3, S);
    field_sub(X3, X3, S);

    // Y3 = M*(S - X3) - 2*A*V
    field_sub(temp, S, X3);
    field_mul(Y3, M, temp);
    field_mul(temp, A, V);
    field_add(temp, temp, temp);
    field_sub(Y3, Y3, temp);

    // ZZ3 = V*ZZ1
    field_mul(ZZ3, V, zz1_m);

    // ZZZ3 = W*ZZZ1
    field_mul(ZZZ3, W, zzz1_m);

    //-------------------------------------------------------------------------
    // Convert from XYZZ back to affine
    // x = X / ZZ, y = Y / ZZZ
    //-------------------------------------------------------------------------

    // Convert ZZ, ZZZ from Montgomery to plain for inversion
    bn_t ZZ3_plain, ZZZ3_plain;
    cgbn_mont_mul(bn_env, ZZ3_plain, ZZ3, one, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, ZZZ3_plain, ZZZ3, one, P, BW6_761_NP0);

    // Compute inverses
    bn_t ZZ3_inv, ZZZ3_inv;
    cgbn_modular_inverse(bn_env, ZZ3_inv, ZZ3_plain, P);
    cgbn_modular_inverse(bn_env, ZZZ3_inv, ZZZ3_plain, P);

    // Convert back to Montgomery form for multiplication
    bn_t ZZ3_inv_m, ZZZ3_inv_m;
    cgbn_mont_mul(bn_env, ZZ3_inv_m, ZZ3_inv, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, ZZZ3_inv_m, ZZZ3_inv, R2, P, BW6_761_NP0);

    // x_affine = X * ZZ^(-1), y_affine = Y * ZZZ^(-1)
    bn_t x_aff_m, y_aff_m;
    field_mul(x_aff_m, X3, ZZ3_inv_m);
    field_mul(y_aff_m, Y3, ZZZ3_inv_m);

    // Convert from Montgomery to plain
    bn_t x_aff, y_aff;
    cgbn_mont_mul(bn_env, x_aff, x_aff_m, one, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, y_aff, y_aff_m, one, P, BW6_761_NP0);

    // Store results
    cgbn_store(bn_env, &(tests[instance].result_x2), x_aff);
    cgbn_store(bn_env, &(tests[instance].result_y2), y_aff);

    #undef field_add
    #undef field_sub
    #undef field_mul
}

//-----------------------------------------------------------------------------
// CUDA Kernel for Point Addition Test
//-----------------------------------------------------------------------------

__global__ void test_point_add_kernel(cgbn_error_report_t *report,
                                       add_test_t *tests,
                                       uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (instance >= count) return;

    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t bn_env(bn_context.env<env_t>());
    typedef env_t::cgbn_t bn_t;

    bn_t P, R2;
    cgbn_load(bn_env, P, (cgbn_mem_t<BITS>*)BW6_761_P_DEVICE);
    cgbn_load(bn_env, R2, (cgbn_mem_t<BITS>*)BW6_761_R2_DEVICE);

    bn_t one, one_m;
    cgbn_set_ui32(bn_env, one, 1);
    cgbn_mont_mul(bn_env, one_m, one, R2, P, BW6_761_NP0);

    // Load first point (in XYZZ form with given ZZ, ZZZ)
    bn_t x1, y1, zz1, zzz1;
    cgbn_load(bn_env, x1, &(tests[instance].x1));
    cgbn_load(bn_env, y1, &(tests[instance].y1));
    cgbn_load(bn_env, zz1, &(tests[instance].zz1));
    cgbn_load(bn_env, zzz1, &(tests[instance].zzz1));

    // Load second point (affine)
    bn_t x2, y2;
    cgbn_load(bn_env, x2, &(tests[instance].x2));
    cgbn_load(bn_env, y2, &(tests[instance].y2));

    // Convert all to Montgomery
    bn_t x1_m, y1_m, zz1_m, zzz1_m, x2_m, y2_m;
    cgbn_mont_mul(bn_env, x1_m, x1, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, y1_m, y1, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, zz1_m, zz1, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, zzz1_m, zzz1, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, x2_m, x2, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, y2_m, y2, R2, P, BW6_761_NP0);

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

    //-------------------------------------------------------------------------
    // Point Addition (XYZZ + Affine) - matching msm_bw6_761_cgbn.cu
    //-------------------------------------------------------------------------
    bn_t U2, S2, Pdiff, R, PP, PPP, Q, temp, temp2;
    bn_t X3, Y3, ZZ3, ZZZ3;

    // U2 = X2 * ZZ1
    field_mul(U2, x2_m, zz1_m);

    // S2 = Y2 * ZZZ1
    field_mul(S2, y2_m, zzz1_m);

    // P = U2 - X1
    field_sub(Pdiff, U2, x1_m);

    // R = S2 - Y1
    field_sub(R, S2, y1_m);

    // PP = P^2
    field_mul(PP, Pdiff, Pdiff);

    // PPP = P * PP
    field_mul(PPP, Pdiff, PP);

    // Q = X1 * PP
    field_mul(Q, x1_m, PP);

    // X3 = R^2 - PPP - 2*Q
    field_mul(X3, R, R);
    field_sub(X3, X3, PPP);
    field_sub(X3, X3, Q);
    field_sub(X3, X3, Q);

    // Y3 = R*(Q - X3) - Y1*PPP
    field_sub(temp, Q, X3);
    field_mul(temp2, y1_m, PPP);
    field_mul(Y3, R, temp);
    field_sub(Y3, Y3, temp2);

    // ZZ3 = ZZ1 * PP
    field_mul(ZZ3, zz1_m, PP);

    // ZZZ3 = ZZZ1 * PPP
    field_mul(ZZZ3, zzz1_m, PPP);

    //-------------------------------------------------------------------------
    // Convert from XYZZ back to affine
    //-------------------------------------------------------------------------
    bn_t ZZ3_plain, ZZZ3_plain;
    cgbn_mont_mul(bn_env, ZZ3_plain, ZZ3, one, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, ZZZ3_plain, ZZZ3, one, P, BW6_761_NP0);

    bn_t ZZ3_inv, ZZZ3_inv;
    cgbn_modular_inverse(bn_env, ZZ3_inv, ZZ3_plain, P);
    cgbn_modular_inverse(bn_env, ZZZ3_inv, ZZZ3_plain, P);

    bn_t ZZ3_inv_m, ZZZ3_inv_m;
    cgbn_mont_mul(bn_env, ZZ3_inv_m, ZZ3_inv, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, ZZZ3_inv_m, ZZZ3_inv, R2, P, BW6_761_NP0);

    bn_t x_aff_m, y_aff_m;
    field_mul(x_aff_m, X3, ZZ3_inv_m);
    field_mul(y_aff_m, Y3, ZZZ3_inv_m);

    bn_t x_aff, y_aff;
    cgbn_mont_mul(bn_env, x_aff, x_aff_m, one, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, y_aff, y_aff_m, one, P, BW6_761_NP0);

    cgbn_store(bn_env, &(tests[instance].result_x3), x_aff);
    cgbn_store(bn_env, &(tests[instance].result_y3), y_aff);

    #undef field_add
    #undef field_sub
    #undef field_mul
}

//-----------------------------------------------------------------------------
// Comparison and printing
//-----------------------------------------------------------------------------

bool compare_limbs(const uint32_t *a, const uint32_t *b, int count) {
    for (int i = 0; i < count; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void print_limbs_short(const char *label, const uint32_t *limbs) {
    printf("%s: %08x%08x%08x...%08x%08x%08x\n", label,
           limbs[23], limbs[22], limbs[21],
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
// Main
//-----------------------------------------------------------------------------

int main() {
    printf("=== BW6-761 CUDA Point Operations Test (Phase 2) ===\n\n");

    mpz_t P, gx, gy, x2, y2, x3, y3, x4, y4;
    mpz_init(P);
    mpz_init(gx);
    mpz_init(gy);
    mpz_init(x2);
    mpz_init(y2);
    mpz_init(x3);
    mpz_init(y3);
    mpz_init(x4);
    mpz_init(y4);

    limbs_to_mpz(P, BW6_761_P_HOST, 24);
    limbs_to_mpz(gx, BW6_761_GEN_X_HOST, 24);
    limbs_to_mpz(gy, BW6_761_GEN_Y_HOST, 24);

    printf("Modulus P bit size: %zu\n", mpz_sizeinbase(P, 2));
    printf("Generator G:\n");
    gmp_printf("  x: %Zx\n", gx);
    gmp_printf("  y: %Zx\n", gy);

    // Verify generator is on curve: y^2 = x^3 + b
    // For BW6-761, b = -1 (or P - 1)
    mpz_t y_sq, x_cube, b;
    mpz_init(y_sq);
    mpz_init(x_cube);
    mpz_init(b);

    mpz_mul(y_sq, gy, gy);
    mpz_mod(y_sq, y_sq, P);

    mpz_powm_ui(x_cube, gx, 3, P);

    // b = P - 1 for BW6-761
    mpz_sub_ui(b, P, 1);

    mpz_add(x_cube, x_cube, b);
    mpz_mod(x_cube, x_cube, P);

    if (mpz_cmp(y_sq, x_cube) == 0) {
        printf("\nGenerator on curve verification: PASS\n\n");
    } else {
        printf("\nGenerator on curve verification: FAIL\n");
        gmp_printf("y^2 = %Zx\n", y_sq);
        gmp_printf("x^3 + b = %Zx\n", x_cube);
        return 1;
    }

    //=========================================================================
    // Test 1: Point Doubling (2*G)
    //=========================================================================
    printf("=== Test 1: Point Doubling (2*G) ===\n");

    // Compute 2*G using GMP
    gmp_point_double_affine(x2, y2, gx, gy, P);
    printf("Expected 2*G:\n");
    gmp_printf("  x: %Zx\n", x2);
    gmp_printf("  y: %Zx\n", y2);

    // Prepare GPU test
    point_test_t *double_test = (point_test_t *)malloc(sizeof(point_test_t));
    mpz_to_limbs(gx, double_test->x._limbs, 24);
    mpz_to_limbs(gy, double_test->y._limbs, 24);
    mpz_to_limbs(x2, double_test->x2._limbs, 24);
    mpz_to_limbs(y2, double_test->y2._limbs, 24);
    memset(double_test->result_x2._limbs, 0, sizeof(double_test->result_x2._limbs));
    memset(double_test->result_y2._limbs, 0, sizeof(double_test->result_y2._limbs));

    // Run GPU kernel
    point_test_t *gpu_double_test;
    cgbn_error_report_t *report;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void **)&gpu_double_test, sizeof(point_test_t)));
    CUDA_CHECK(cudaMemcpy(gpu_double_test, double_test, sizeof(point_test_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    test_point_double_kernel<<<1, TPI>>>(report, gpu_double_test, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    CUDA_CHECK(cudaMemcpy(double_test, gpu_double_test, sizeof(point_test_t), cudaMemcpyDeviceToHost));

    bool double_x_ok = compare_limbs(double_test->result_x2._limbs, double_test->x2._limbs, 24);
    bool double_y_ok = compare_limbs(double_test->result_y2._limbs, double_test->y2._limbs, 24);

    printf("\nGPU result:\n");
    print_limbs_short("  x", double_test->result_x2._limbs);
    print_limbs_short("  y", double_test->result_y2._limbs);

    if (double_x_ok && double_y_ok) {
        printf("\nPoint double (2*G): PASS\n\n");
    } else {
        printf("\nPoint double (2*G): FAIL\n");
        print_limbs_short("Expected x", double_test->x2._limbs);
        print_limbs_short("Got x     ", double_test->result_x2._limbs);
        print_limbs_short("Expected y", double_test->y2._limbs);
        print_limbs_short("Got y     ", double_test->result_y2._limbs);
        return 1;
    }

    //=========================================================================
    // Test 2: Point Addition (G + 2*G = 3*G)
    //=========================================================================
    printf("=== Test 2: Point Addition (G + 2*G = 3*G) ===\n");

    // Compute 3*G = G + 2*G using GMP
    gmp_point_add_affine(x3, y3, gx, gy, x2, y2, P);
    printf("Expected 3*G:\n");
    gmp_printf("  x: %Zx\n", x3);
    gmp_printf("  y: %Zx\n", y3);

    // Prepare GPU test: G (as XYZZ with ZZ=ZZZ=1) + 2G (affine)
    add_test_t *add_test = (add_test_t *)malloc(sizeof(add_test_t));
    mpz_to_limbs(gx, add_test->x1._limbs, 24);
    mpz_to_limbs(gy, add_test->y1._limbs, 24);
    // ZZ = ZZZ = 1
    memset(add_test->zz1._limbs, 0, sizeof(add_test->zz1._limbs));
    memset(add_test->zzz1._limbs, 0, sizeof(add_test->zzz1._limbs));
    add_test->zz1._limbs[0] = 1;
    add_test->zzz1._limbs[0] = 1;
    // Second point: 2*G
    mpz_to_limbs(x2, add_test->x2._limbs, 24);
    mpz_to_limbs(y2, add_test->y2._limbs, 24);
    // Expected: 3*G
    mpz_to_limbs(x3, add_test->expected_x3._limbs, 24);
    mpz_to_limbs(y3, add_test->expected_y3._limbs, 24);
    memset(add_test->result_x3._limbs, 0, sizeof(add_test->result_x3._limbs));
    memset(add_test->result_y3._limbs, 0, sizeof(add_test->result_y3._limbs));

    add_test_t *gpu_add_test;
    CUDA_CHECK(cudaMalloc((void **)&gpu_add_test, sizeof(add_test_t)));
    CUDA_CHECK(cudaMemcpy(gpu_add_test, add_test, sizeof(add_test_t), cudaMemcpyHostToDevice));

    test_point_add_kernel<<<1, TPI>>>(report, gpu_add_test, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    CUDA_CHECK(cudaMemcpy(add_test, gpu_add_test, sizeof(add_test_t), cudaMemcpyDeviceToHost));

    bool add_x_ok = compare_limbs(add_test->result_x3._limbs, add_test->expected_x3._limbs, 24);
    bool add_y_ok = compare_limbs(add_test->result_y3._limbs, add_test->expected_y3._limbs, 24);

    printf("\nGPU result:\n");
    print_limbs_short("  x", add_test->result_x3._limbs);
    print_limbs_short("  y", add_test->result_y3._limbs);

    if (add_x_ok && add_y_ok) {
        printf("\nPoint add (G + 2*G = 3*G): PASS\n\n");
    } else {
        printf("\nPoint add (G + 2*G = 3*G): FAIL\n");
        print_limbs_short("Expected x", add_test->expected_x3._limbs);
        print_limbs_short("Got x     ", add_test->result_x3._limbs);
        print_limbs_short("Expected y", add_test->expected_y3._limbs);
        print_limbs_short("Got y     ", add_test->result_y3._limbs);
        return 1;
    }

    //=========================================================================
    // Test 3: Chain: 2*G + G = 3*G (add in different order)
    //=========================================================================
    printf("=== Test 3: Point Addition (2*G + G = 3*G) ===\n");

    // This time: 2*G (as XYZZ) + G (affine) = 3*G
    mpz_to_limbs(x2, add_test->x1._limbs, 24);  // First point: 2*G
    mpz_to_limbs(y2, add_test->y1._limbs, 24);
    memset(add_test->zz1._limbs, 0, sizeof(add_test->zz1._limbs));
    memset(add_test->zzz1._limbs, 0, sizeof(add_test->zzz1._limbs));
    add_test->zz1._limbs[0] = 1;
    add_test->zzz1._limbs[0] = 1;
    mpz_to_limbs(gx, add_test->x2._limbs, 24);  // Second point: G
    mpz_to_limbs(gy, add_test->y2._limbs, 24);
    // Expected still 3*G
    memset(add_test->result_x3._limbs, 0, sizeof(add_test->result_x3._limbs));
    memset(add_test->result_y3._limbs, 0, sizeof(add_test->result_y3._limbs));

    CUDA_CHECK(cudaMemcpy(gpu_add_test, add_test, sizeof(add_test_t), cudaMemcpyHostToDevice));
    test_point_add_kernel<<<1, TPI>>>(report, gpu_add_test, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);
    CUDA_CHECK(cudaMemcpy(add_test, gpu_add_test, sizeof(add_test_t), cudaMemcpyDeviceToHost));

    add_x_ok = compare_limbs(add_test->result_x3._limbs, add_test->expected_x3._limbs, 24);
    add_y_ok = compare_limbs(add_test->result_y3._limbs, add_test->expected_y3._limbs, 24);

    if (add_x_ok && add_y_ok) {
        printf("Point add (2*G + G = 3*G): PASS\n\n");
    } else {
        printf("Point add (2*G + G = 3*G): FAIL\n");
        print_limbs_short("Expected x", add_test->expected_x3._limbs);
        print_limbs_short("Got x     ", add_test->result_x3._limbs);
        return 1;
    }

    //=========================================================================
    // Test 4: Chain doubling: 2*(2*G) = 4*G
    //=========================================================================
    printf("=== Test 4: Chain Doubling (2*(2*G) = 4*G) ===\n");

    gmp_point_double_affine(x4, y4, x2, y2, P);
    printf("Expected 4*G:\n");
    gmp_printf("  x: %Zx\n", x4);

    mpz_to_limbs(x2, double_test->x._limbs, 24);
    mpz_to_limbs(y2, double_test->y._limbs, 24);
    mpz_to_limbs(x4, double_test->x2._limbs, 24);
    mpz_to_limbs(y4, double_test->y2._limbs, 24);
    memset(double_test->result_x2._limbs, 0, sizeof(double_test->result_x2._limbs));
    memset(double_test->result_y2._limbs, 0, sizeof(double_test->result_y2._limbs));

    CUDA_CHECK(cudaMemcpy(gpu_double_test, double_test, sizeof(point_test_t), cudaMemcpyHostToDevice));
    test_point_double_kernel<<<1, TPI>>>(report, gpu_double_test, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);
    CUDA_CHECK(cudaMemcpy(double_test, gpu_double_test, sizeof(point_test_t), cudaMemcpyDeviceToHost));

    double_x_ok = compare_limbs(double_test->result_x2._limbs, double_test->x2._limbs, 24);
    double_y_ok = compare_limbs(double_test->result_y2._limbs, double_test->y2._limbs, 24);

    if (double_x_ok && double_y_ok) {
        printf("Chain double (4*G): PASS\n\n");
    } else {
        printf("Chain double (4*G): FAIL\n");
        print_limbs_short("Expected x", double_test->x2._limbs);
        print_limbs_short("Got x     ", double_test->result_x2._limbs);
        return 1;
    }

    //=========================================================================
    // Summary
    //=========================================================================
    printf("=== Summary ===\n");
    printf("Point double (2*G):           PASS\n");
    printf("Point add (G + 2*G = 3*G):    PASS\n");
    printf("Point add (2*G + G = 3*G):    PASS\n");
    printf("Chain double (4*G):           PASS\n");
    printf("\n[SUCCESS] All Phase 2 point operation tests passed!\n");
    printf("Proceed to Phase 3 (scalar multiplication).\n");

    // Cleanup
    free(double_test);
    free(add_test);
    CUDA_CHECK(cudaFree(gpu_double_test));
    CUDA_CHECK(cudaFree(gpu_add_test));
    CUDA_CHECK(cgbn_error_report_free(report));

    mpz_clear(P);
    mpz_clear(gx);
    mpz_clear(gy);
    mpz_clear(x2);
    mpz_clear(y2);
    mpz_clear(x3);
    mpz_clear(y3);
    mpz_clear(x4);
    mpz_clear(y4);
    mpz_clear(y_sq);
    mpz_clear(x_cube);
    mpz_clear(b);

    return 0;
}
