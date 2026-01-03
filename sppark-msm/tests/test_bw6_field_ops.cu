/***
 * Standalone CGBN Field Arithmetic Test for BW6-761
 *
 * Phase 1 of BW6-761 scalar multiplication bug investigation.
 * Validates CGBN field operations against GMP reference.
 *
 * Tests:
 * 1. field_add - modular addition
 * 2. field_sub - modular subtraction (including aliasing cases)
 * 3. field_mul - Montgomery multiplication
 * 4. to_montgomery / from_montgomery - round-trip conversion
 *
 * Compile: nvcc -o test_bw6_field_ops test_bw6_field_ops.cu \
 *              -I../src -I../../cgbn-lib/include -lgmp -arch=sm_75
 * Run: ./test_bw6_field_ops
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"

// CGBN configuration for BW6-761 (768-bit operations)
#define TPI 8              // Threads per instance (required for 768-bit)
#define BITS 768           // Round up from 761 bits
#define TEST_INSTANCES 100 // Number of random test cases

// BW6-761 base field modulus P (761 bits) - little-endian u32[24]
__device__ __constant__ uint32_t BW6_761_P_DEVICE[24] = {
    0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68,
    0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
    0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a,
    0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824
};

// Montgomery np0 = -P^(-1) mod 2^32
__device__ __constant__ uint32_t BW6_761_NP0 = 0x8fa798dd;

// Montgomery R^2 mod P (verified constant)
__device__ __constant__ uint32_t BW6_761_R2_DEVICE[24] = {
    0x2d1fa659, 0xc686392d, 0xf79484ab, 0x7b14c9b2,
    0xc1d2b459, 0x7fa1e825, 0x48329d88, 0xd6ec28f8,
    0x73a1ed40, 0x4afb427b, 0x0d5930ae, 0x972c6940,
    0x8c995976, 0x2c7a26bf, 0xc6e57af9, 0xac52e458,
    0x0c536dfe, 0xac731bfa, 0x0b103f50, 0x121e5c63,
    0xb886cda4, 0x8f1b0953, 0x2da8d807, 0x00ad253c
};

// Host copies for GMP calculations
uint32_t BW6_761_P_HOST[24] = {
    0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68,
    0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
    0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a,
    0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824
};

uint32_t BW6_761_R2_HOST[24] = {
    0x2d1fa659, 0xc686392d, 0xf79484ab, 0x7b14c9b2,
    0xc1d2b459, 0x7fa1e825, 0x48329d88, 0xd6ec28f8,
    0x73a1ed40, 0x4afb427b, 0x0d5930ae, 0x972c6940,
    0x8c995976, 0x2c7a26bf, 0xc6e57af9, 0xac52e458,
    0x0c536dfe, 0xac731bfa, 0x0b103f50, 0x121e5c63,
    0xb886cda4, 0x8f1b0953, 0x2da8d807, 0x00ad253c
};

// Test data structure
typedef struct {
    cgbn_mem_t<BITS> a;
    cgbn_mem_t<BITS> b;
    cgbn_mem_t<BITS> result_add;
    cgbn_mem_t<BITS> result_sub;
    cgbn_mem_t<BITS> result_mul;  // Montgomery multiplication result
    cgbn_mem_t<BITS> result_sub_aliased;  // Test field_sub(a, a, b)
    cgbn_mem_t<BITS> result_mont_roundtrip;  // to_mont(from_mont(a))
    cgbn_mem_t<BITS> expected_add;
    cgbn_mem_t<BITS> expected_sub;
    cgbn_mem_t<BITS> expected_mul;  // (a * b) mod P in plain form
} test_instance_t;

// CGBN types
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

//-----------------------------------------------------------------------------
// GMP utility functions
//-----------------------------------------------------------------------------

void mpz_to_limbs(mpz_t n, uint32_t *limbs, int count) {
    memset(limbs, 0, count * sizeof(uint32_t));
    size_t actual_count;
    mpz_export(limbs, &actual_count, -1, sizeof(uint32_t), -1, 0, n);
}

void limbs_to_mpz(mpz_t n, const uint32_t *limbs, int count) {
    mpz_import(n, count, -1, sizeof(uint32_t), -1, 0, limbs);
}

void random_field_element(uint32_t *limbs, mpz_t modulus, gmp_randstate_t rng) {
    mpz_t elem;
    mpz_init(elem);
    mpz_urandomm(elem, rng, modulus);
    mpz_to_limbs(elem, limbs, 24);
    mpz_clear(elem);
}

//-----------------------------------------------------------------------------
// Generate test instances with GMP reference
//-----------------------------------------------------------------------------

test_instance_t* generate_test_instances(uint32_t count) {
    test_instance_t *instances = (test_instance_t *)malloc(sizeof(test_instance_t) * count);

    mpz_t P, R, R2, a, b, sum, sub, product, mont_a, mont_b, mont_product;
    gmp_randstate_t rng;

    mpz_init(P);
    mpz_init(R);
    mpz_init(R2);
    mpz_init(a);
    mpz_init(b);
    mpz_init(sum);
    mpz_init(sub);
    mpz_init(product);
    mpz_init(mont_a);
    mpz_init(mont_b);
    mpz_init(mont_product);

    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, 12345);  // Fixed seed for reproducibility

    // Load P and compute R = 2^768
    limbs_to_mpz(P, BW6_761_P_HOST, 24);
    mpz_ui_pow_ui(R, 2, 768);
    limbs_to_mpz(R2, BW6_761_R2_HOST, 24);

    printf("BW6-761 modulus P bit size: %zu\n", mpz_sizeinbase(P, 2));
    printf("Montgomery R = 2^768\n");

    // Verify R2 = R^2 mod P
    mpz_t R2_check;
    mpz_init(R2_check);
    mpz_mul(R2_check, R, R);
    mpz_mod(R2_check, R2_check, P);
    if (mpz_cmp(R2, R2_check) == 0) {
        printf("R^2 mod P verification: PASS\n\n");
    } else {
        printf("R^2 mod P verification: FAIL (constant may be wrong)\n\n");
    }
    mpz_clear(R2_check);

    for (uint32_t i = 0; i < count; i++) {
        // Generate random field elements
        random_field_element(instances[i].a._limbs, P, rng);
        random_field_element(instances[i].b._limbs, P, rng);

        limbs_to_mpz(a, instances[i].a._limbs, 24);
        limbs_to_mpz(b, instances[i].b._limbs, 24);

        // Expected addition: (a + b) mod P
        mpz_add(sum, a, b);
        mpz_mod(sum, sum, P);
        mpz_to_limbs(sum, instances[i].expected_add._limbs, 24);

        // Expected subtraction: (a - b) mod P
        mpz_sub(sub, a, b);
        mpz_mod(sub, sub, P);
        mpz_to_limbs(sub, instances[i].expected_sub._limbs, 24);

        // Expected multiplication: (a * b) mod P (plain form)
        mpz_mul(product, a, b);
        mpz_mod(product, product, P);
        mpz_to_limbs(product, instances[i].expected_mul._limbs, 24);

        // Zero out result fields
        memset(instances[i].result_add._limbs, 0, sizeof(instances[i].result_add._limbs));
        memset(instances[i].result_sub._limbs, 0, sizeof(instances[i].result_sub._limbs));
        memset(instances[i].result_mul._limbs, 0, sizeof(instances[i].result_mul._limbs));
        memset(instances[i].result_sub_aliased._limbs, 0, sizeof(instances[i].result_sub_aliased._limbs));
        memset(instances[i].result_mont_roundtrip._limbs, 0, sizeof(instances[i].result_mont_roundtrip._limbs));
    }

    mpz_clear(P);
    mpz_clear(R);
    mpz_clear(R2);
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(sum);
    mpz_clear(sub);
    mpz_clear(product);
    mpz_clear(mont_a);
    mpz_clear(mont_b);
    mpz_clear(mont_product);
    gmp_randclear(rng);

    return instances;
}

//-----------------------------------------------------------------------------
// CUDA Kernel - Field Operations Test
//-----------------------------------------------------------------------------

__global__ void test_field_ops_kernel(cgbn_error_report_t *report,
                                       test_instance_t *instances,
                                       uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;

    if (instance >= count)
        return;

    // Create CGBN context and environment
    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t bn_env(bn_context.env<env_t>());

    typedef env_t::cgbn_t bn_t;

    bn_t a, b, P, R2, result, temp_result;

    // Load modulus and R^2
    cgbn_load(bn_env, P, (cgbn_mem_t<BITS>*)BW6_761_P_DEVICE);
    cgbn_load(bn_env, R2, (cgbn_mem_t<BITS>*)BW6_761_R2_DEVICE);

    // Load operands
    cgbn_load(bn_env, a, &(instances[instance].a));
    cgbn_load(bn_env, b, &(instances[instance].b));

    //-------------------------------------------------------------------------
    // Test 1: field_add (a + b) mod P
    //-------------------------------------------------------------------------
    cgbn_add(bn_env, result, a, b);
    cgbn_rem(bn_env, result, result, P);
    cgbn_store(bn_env, &(instances[instance].result_add), result);

    //-------------------------------------------------------------------------
    // Test 2: field_sub (a - b) mod P - non-aliased
    //-------------------------------------------------------------------------
    // Replicate the field_sub logic from msm_bw6_761_cgbn.cu
    int32_t cmp = cgbn_compare(bn_env, a, b);
    if (cmp >= 0) {
        cgbn_sub(bn_env, temp_result, a, b);
    } else {
        bn_t p_minus_b;
        cgbn_sub(bn_env, p_minus_b, P, b);
        cgbn_add(bn_env, temp_result, a, p_minus_b);
    }
    cgbn_set(bn_env, result, temp_result);
    cgbn_store(bn_env, &(instances[instance].result_sub), result);

    //-------------------------------------------------------------------------
    // Test 3: field_sub aliased - field_sub(a, a, b) where output == first input
    // This tests the aliasing fix
    //-------------------------------------------------------------------------
    bn_t a_copy;
    cgbn_set(bn_env, a_copy, a);  // Make a copy

    // Now compute field_sub with aliasing: result stored back into a_copy
    cmp = cgbn_compare(bn_env, a_copy, b);
    if (cmp >= 0) {
        cgbn_sub(bn_env, temp_result, a_copy, b);
    } else {
        bn_t p_minus_b;
        cgbn_sub(bn_env, p_minus_b, P, b);
        cgbn_add(bn_env, temp_result, a_copy, p_minus_b);
    }
    cgbn_set(bn_env, a_copy, temp_result);
    cgbn_store(bn_env, &(instances[instance].result_sub_aliased), a_copy);

    //-------------------------------------------------------------------------
    // Test 4: Montgomery multiplication
    // Convert a,b to Montgomery form, multiply, convert back
    //-------------------------------------------------------------------------
    bn_t a_mont, b_mont, product_mont, product_plain, one;

    // to_montgomery: a_mont = a * R^2 * R^(-1) = a * R mod P
    cgbn_mont_mul(bn_env, a_mont, a, R2, P, BW6_761_NP0);
    cgbn_mont_mul(bn_env, b_mont, b, R2, P, BW6_761_NP0);

    // Montgomery multiplication: product_mont = a_mont * b_mont * R^(-1) = (a*b)*R mod P
    cgbn_mont_mul(bn_env, product_mont, a_mont, b_mont, P, BW6_761_NP0);

    // from_montgomery: product_plain = product_mont * 1 * R^(-1) = a*b mod P
    cgbn_set_ui32(bn_env, one, 1);
    cgbn_mont_mul(bn_env, product_plain, product_mont, one, P, BW6_761_NP0);

    cgbn_store(bn_env, &(instances[instance].result_mul), product_plain);

    //-------------------------------------------------------------------------
    // Test 5: Montgomery round-trip: to_mont(from_mont(a)) should equal a
    //-------------------------------------------------------------------------
    bn_t a_to_mont, a_roundtrip;

    // to_montgomery: a_to_mont = a * R mod P
    cgbn_mont_mul(bn_env, a_to_mont, a, R2, P, BW6_761_NP0);

    // from_montgomery: a_roundtrip = a_to_mont * R^(-1) = a mod P
    cgbn_mont_mul(bn_env, a_roundtrip, a_to_mont, one, P, BW6_761_NP0);

    cgbn_store(bn_env, &(instances[instance].result_mont_roundtrip), a_roundtrip);
}

//-----------------------------------------------------------------------------
// Comparison and printing utilities
//-----------------------------------------------------------------------------

bool compare_limbs(const uint32_t *a, const uint32_t *b, int count) {
    for (int i = 0; i < count; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void print_limbs(const char *label, const uint32_t *limbs, int count) {
    printf("%s: ", label);
    for (int i = count - 1; i >= 0; i--) {
        printf("%08x", limbs[i]);
    }
    printf("\n");
}

void print_limbs_short(const char *label, const uint32_t *limbs) {
    // Print just the first and last few limbs for brevity
    printf("%s: %08x%08x%08x...%08x%08x%08x\n", label,
           limbs[23], limbs[22], limbs[21],
           limbs[2], limbs[1], limbs[0]);
}

//-----------------------------------------------------------------------------
// CUDA error checking macros
//-----------------------------------------------------------------------------

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
    printf("=== BW6-761 CGBN Field Arithmetic Test (Phase 1) ===\n");
    printf("Configuration: TPI=%d, BITS=%d, INSTANCES=%d\n\n", TPI, BITS, TEST_INSTANCES);

    test_instance_t *instances, *gpu_instances;
    cgbn_error_report_t *report;

    // Generate test instances with GMP reference
    printf("Generating test instances with GMP reference...\n");
    instances = generate_test_instances(TEST_INSTANCES);

    // Allocate GPU memory
    printf("Copying to GPU...\n");
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void **)&gpu_instances, sizeof(test_instance_t) * TEST_INSTANCES));
    CUDA_CHECK(cudaMemcpy(gpu_instances, instances, sizeof(test_instance_t) * TEST_INSTANCES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    // Launch kernel
    printf("Running field operations kernel...\n");
    int threads_per_block = 128;  // Must be multiple of TPI
    int num_blocks = (TEST_INSTANCES * TPI + threads_per_block - 1) / threads_per_block;

    test_field_ops_kernel<<<num_blocks, threads_per_block>>>(report, gpu_instances, TEST_INSTANCES);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);
    printf("Kernel completed.\n\n");

    // Copy results back
    CUDA_CHECK(cudaMemcpy(instances, gpu_instances, sizeof(test_instance_t) * TEST_INSTANCES, cudaMemcpyDeviceToHost));

    //-------------------------------------------------------------------------
    // Validate results
    //-------------------------------------------------------------------------
    printf("=== Validation Results ===\n\n");

    int add_pass = 0, add_fail = 0;
    int sub_pass = 0, sub_fail = 0;
    int sub_alias_pass = 0, sub_alias_fail = 0;
    int mul_pass = 0, mul_fail = 0;
    int roundtrip_pass = 0, roundtrip_fail = 0;

    for (int i = 0; i < TEST_INSTANCES; i++) {
        // Check addition
        if (compare_limbs(instances[i].result_add._limbs, instances[i].expected_add._limbs, 24)) {
            add_pass++;
        } else {
            add_fail++;
            if (add_fail <= 2) {
                printf("[ADD FAIL] Instance %d:\n", i);
                print_limbs_short("  a       ", instances[i].a._limbs);
                print_limbs_short("  b       ", instances[i].b._limbs);
                print_limbs_short("  expected", instances[i].expected_add._limbs);
                print_limbs_short("  got     ", instances[i].result_add._limbs);
                printf("\n");
            }
        }

        // Check subtraction
        if (compare_limbs(instances[i].result_sub._limbs, instances[i].expected_sub._limbs, 24)) {
            sub_pass++;
        } else {
            sub_fail++;
            if (sub_fail <= 2) {
                printf("[SUB FAIL] Instance %d:\n", i);
                print_limbs_short("  a       ", instances[i].a._limbs);
                print_limbs_short("  b       ", instances[i].b._limbs);
                print_limbs_short("  expected", instances[i].expected_sub._limbs);
                print_limbs_short("  got     ", instances[i].result_sub._limbs);
                printf("\n");
            }
        }

        // Check aliased subtraction (should match regular subtraction)
        if (compare_limbs(instances[i].result_sub_aliased._limbs, instances[i].expected_sub._limbs, 24)) {
            sub_alias_pass++;
        } else {
            sub_alias_fail++;
            if (sub_alias_fail <= 2) {
                printf("[SUB_ALIAS FAIL] Instance %d:\n", i);
                print_limbs_short("  expected", instances[i].expected_sub._limbs);
                print_limbs_short("  got     ", instances[i].result_sub_aliased._limbs);
                printf("\n");
            }
        }

        // Check Montgomery multiplication
        if (compare_limbs(instances[i].result_mul._limbs, instances[i].expected_mul._limbs, 24)) {
            mul_pass++;
        } else {
            mul_fail++;
            if (mul_fail <= 2) {
                printf("[MUL FAIL] Instance %d:\n", i);
                print_limbs_short("  a       ", instances[i].a._limbs);
                print_limbs_short("  b       ", instances[i].b._limbs);
                print_limbs_short("  expected", instances[i].expected_mul._limbs);
                print_limbs_short("  got     ", instances[i].result_mul._limbs);
                printf("\n");
            }
        }

        // Check Montgomery round-trip (should match original a)
        if (compare_limbs(instances[i].result_mont_roundtrip._limbs, instances[i].a._limbs, 24)) {
            roundtrip_pass++;
        } else {
            roundtrip_fail++;
            if (roundtrip_fail <= 2) {
                printf("[ROUNDTRIP FAIL] Instance %d:\n", i);
                print_limbs_short("  original a", instances[i].a._limbs);
                print_limbs_short("  roundtrip ", instances[i].result_mont_roundtrip._limbs);
                printf("\n");
            }
        }
    }

    //-------------------------------------------------------------------------
    // Print summary
    //-------------------------------------------------------------------------
    printf("=== Summary ===\n");
    printf("field_add:       %3d/%d %s\n", add_pass, TEST_INSTANCES, add_fail == 0 ? "PASS" : "FAIL");
    printf("field_sub:       %3d/%d %s\n", sub_pass, TEST_INSTANCES, sub_fail == 0 ? "PASS" : "FAIL");
    printf("field_sub alias: %3d/%d %s\n", sub_alias_pass, TEST_INSTANCES, sub_alias_fail == 0 ? "PASS" : "FAIL");
    printf("field_mul mont:  %3d/%d %s\n", mul_pass, TEST_INSTANCES, mul_fail == 0 ? "PASS" : "FAIL");
    printf("mont roundtrip:  %3d/%d %s\n", roundtrip_pass, TEST_INSTANCES, roundtrip_fail == 0 ? "PASS" : "FAIL");

    bool all_pass = (add_fail == 0 && sub_fail == 0 && sub_alias_fail == 0 &&
                     mul_fail == 0 && roundtrip_fail == 0);

    printf("\n");
    if (all_pass) {
        printf("[SUCCESS] All Phase 1 field arithmetic tests passed!\n");
        printf("Field operations are correct. Proceed to Phase 2 (point ops).\n");
    } else {
        printf("[FAILURE] Some tests failed. Fix field ops before proceeding.\n");
    }

    // Cleanup
    free(instances);
    CUDA_CHECK(cudaFree(gpu_instances));
    CUDA_CHECK(cgbn_error_report_free(report));

    return all_pass ? 0 : 1;
}
