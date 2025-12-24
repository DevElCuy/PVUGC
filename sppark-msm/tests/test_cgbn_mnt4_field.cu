/***
 * Standalone CGBN Field Arithmetic Test for MNT4-298
 *
 * Purpose: Validate that CGBN can correctly perform field operations
 *          for MNT4-298's 298-bit base field before attempting integration.
 *
 * Strategy:
 * - Use TPI=8 (8 threads cooperate per field element)
 * - Use BITS=320 (round up from 298 to nearest 32-bit boundary)
 * - Test modular addition, subtraction, multiplication
 * - Cross-validate against GMP reference calculations
 *
 * Compile: nvcc -o test_cgbn_mnt4_field test_cgbn_mnt4_field.cu -I../../cgbn-lib/include -lgmp -arch=sm_75
 * Run: ./test_cgbn_mnt4_field
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"

// CGBN configuration for MNT4-298
#define TPI 8              // Threads per instance (8 threads cooperate)
#define BITS 320           // Round up from 298 bits
#define INSTANCES 100      // Test 100 field operations

// MNT4-298 base field modulus (Fq, 298 bits)
// Little-endian u32[10] representation
// From msm_mnt4_298_cgbn.cu
__device__ __constant__ uint32_t MNT4_298_MODULUS[10] = {
    0x71660001, 0xc90cd65a, 0x51200e12, 0x41a9e35e, 0x5d1330ea,
    0xcaeec963, 0xa7b0548e, 0xa266249d, 0xf7bcd473, 0x000003bc
};

// Host copy of modulus for GMP calculations
uint32_t MNT4_298_MODULUS_HOST[10] = {
    0x71660001, 0xc90cd65a, 0x51200e12, 0x41a9e35e, 0x5d1330ea,
    0xcaeec963, 0xa7b0548e, 0xa266249d, 0xf7bcd473, 0x000003bc
};

// Test instance: two field elements and results
typedef struct {
    cgbn_mem_t<BITS> a;
    cgbn_mem_t<BITS> b;
    cgbn_mem_t<BITS> result_sum;
    cgbn_mem_t<BITS> result_sub;
    cgbn_mem_t<BITS> result_product;
    cgbn_mem_t<BITS> expected_sum;
    cgbn_mem_t<BITS> expected_sub;
    cgbn_mem_t<BITS> expected_product;
} test_instance_t;

// CGBN context and environment types
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// Convert GMP mpz_t to limb array (little-endian)
void mpz_to_limbs(mpz_t n, uint32_t *limbs, int count) {
    memset(limbs, 0, count * sizeof(uint32_t));
    size_t actual_count;
    mpz_export(limbs, &actual_count, -1, sizeof(uint32_t), -1, 0, n);
}

// Convert limb array to GMP mpz_t (little-endian)
void limbs_to_mpz(mpz_t n, const uint32_t *limbs, int count) {
    mpz_import(n, count, -1, sizeof(uint32_t), -1, 0, limbs);
}

// Generate random field element less than modulus
void random_field_element(uint32_t *limbs, mpz_t modulus, gmp_randstate_t rng) {
    mpz_t elem;
    mpz_init(elem);
    mpz_urandomm(elem, rng, modulus);
    mpz_to_limbs(elem, limbs, BITS/32);
    mpz_clear(elem);
}

// Generate test instances with GMP reference values
test_instance_t* generate_test_instances(uint32_t count) {
    test_instance_t *instances = (test_instance_t *)malloc(sizeof(test_instance_t) * count);

    // Initialize GMP
    mpz_t modulus, a, b, sum, sub, product;
    gmp_randstate_t rng;

    mpz_init(modulus);
    mpz_init(a);
    mpz_init(b);
    mpz_init(sum);
    mpz_init(sub);
    mpz_init(product);
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, 12345);  // Fixed seed for reproducibility

    // Load modulus
    limbs_to_mpz(modulus, MNT4_298_MODULUS_HOST, 10);

    printf("MNT4-298 modulus (hex): ");
    gmp_printf("%Zx\n", modulus);
    printf("Modulus bit size: %zu\n\n", mpz_sizeinbase(modulus, 2));

    for(uint32_t i = 0; i < count; i++) {
        // Generate random field elements
        random_field_element(instances[i].a._limbs, modulus, rng);
        random_field_element(instances[i].b._limbs, modulus, rng);

        // Compute expected results using GMP
        limbs_to_mpz(a, instances[i].a._limbs, 10);
        limbs_to_mpz(b, instances[i].b._limbs, 10);

        // sum = (a + b) mod P
        mpz_add(sum, a, b);
        mpz_mod(sum, sum, modulus);
        mpz_to_limbs(sum, instances[i].expected_sum._limbs, BITS/32);

        // sub = (a - b) mod P
        mpz_sub(sub, a, b);
        mpz_mod(sub, sub, modulus);
        mpz_to_limbs(sub, instances[i].expected_sub._limbs, BITS/32);

        // product = (a * b) mod P
        mpz_mul(product, a, b);
        mpz_mod(product, product, modulus);
        mpz_to_limbs(product, instances[i].expected_product._limbs, BITS/32);

        // Initialize result arrays to zero
        memset(instances[i].result_sum._limbs, 0, sizeof(instances[i].result_sum._limbs));
        memset(instances[i].result_sub._limbs, 0, sizeof(instances[i].result_sub._limbs));
        memset(instances[i].result_product._limbs, 0, sizeof(instances[i].result_product._limbs));
    }

    mpz_clear(modulus);
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(sum);
    mpz_clear(sub);
    mpz_clear(product);
    gmp_randclear(rng);

    return instances;
}

// Kernel: Test all field operations
__global__ void test_field_ops(cgbn_error_report_t *report, test_instance_t *instances, uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;

    if(instance >= count)
        return;

    // Create CGBN context and environment
    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t bn_env(bn_context.env<env_t>());

    // Declare field elements
    env_t::cgbn_t a, b, sum, sub, product, modulus;
    typedef env_t::cgbn_wide_t wide_t;
    wide_t wide_product;

    // Load modulus
    cgbn_load(bn_env, modulus, (cgbn_mem_t<BITS>*)MNT4_298_MODULUS);

    // Load operands from memory
    cgbn_load(bn_env, a, &(instances[instance].a));
    cgbn_load(bn_env, b, &(instances[instance].b));

    // Addition: (a + b) mod P
    cgbn_add(bn_env, sum, a, b);
    cgbn_rem(bn_env, sum, sum, modulus);
    cgbn_store(bn_env, &(instances[instance].result_sum), sum);

    // Subtraction: (a - b) mod P
    // CGBN uses unsigned arithmetic, so we need to handle underflow
    int32_t borrow = cgbn_sub(bn_env, sub, a, b);
    if (borrow != 0) {
        cgbn_add(bn_env, sub, sub, modulus);
    }
    cgbn_store(bn_env, &(instances[instance].result_sub), sub);

    // Multiplication: (a * b) mod P using wide multiplication
    cgbn_mul_wide(bn_env, wide_product, a, b);
    cgbn_rem_wide(bn_env, product, wide_product, modulus);
    cgbn_store(bn_env, &(instances[instance].result_product), product);
}

// Compare two limb arrays
bool compare_limbs(const uint32_t *a, const uint32_t *b, int count) {
    for(int i = 0; i < count; i++) {
        if(a[i] != b[i]) return false;
    }
    return true;
}

// Print limb array as hex
void print_limbs(const char *label, const uint32_t *limbs, int count) {
    printf("%s: ", label);
    for(int i = count - 1; i >= 0; i--) {
        printf("%08x", limbs[i]);
    }
    printf("\n");
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CGBN_CHECK(report) \
    do { \
        if(cgbn_error_report_check(report)) { \
            fprintf(stderr, "CGBN error detected\n"); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    test_instance_t *instances, *gpu_instances;
    cgbn_error_report_t *report;

    printf("=== CGBN MNT4-298 Field Arithmetic Test ===\n");
    printf("Configuration: TPI=%d, BITS=%d, INSTANCES=%d\n\n", TPI, BITS, INSTANCES);

    // Generate test instances with GMP reference values
    printf("Generating %d random field elements with GMP reference...\n", INSTANCES);
    instances = generate_test_instances(INSTANCES);

    // Allocate GPU memory
    printf("Copying to GPU...\n");
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void **)&gpu_instances, sizeof(test_instance_t) * INSTANCES));
    CUDA_CHECK(cudaMemcpy(gpu_instances, instances, sizeof(test_instance_t) * INSTANCES, cudaMemcpyHostToDevice));

    // Create error report
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    // Run field operations kernel
    printf("\nRunning field operations kernel...\n");
    int threads_per_block = 128;  // Must be multiple of TPI
    int num_blocks = (INSTANCES * TPI + threads_per_block - 1) / threads_per_block;
    printf("  Launching kernel: %d blocks x %d threads = %d total threads\n",
           num_blocks, threads_per_block, num_blocks * threads_per_block);
    printf("  Processing %d instances (%d threads per instance)\n", INSTANCES, TPI);

    test_field_ops<<<num_blocks, threads_per_block>>>(report, gpu_instances, INSTANCES);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);
    printf("  Kernel completed without errors\n");

    // Copy results back
    printf("\nCopying results back to CPU...\n");
    CUDA_CHECK(cudaMemcpy(instances, gpu_instances, sizeof(test_instance_t) * INSTANCES, cudaMemcpyDeviceToHost));

    // Validate against GMP reference
    printf("\n=== Validation Against GMP Reference ===\n");
    int add_pass = 0, add_fail = 0;
    int sub_pass = 0, sub_fail = 0;
    int mul_pass = 0, mul_fail = 0;

    for(int i = 0; i < INSTANCES; i++) {
        // Check addition
        if(compare_limbs(instances[i].result_sum._limbs, instances[i].expected_sum._limbs, 10)) {
            add_pass++;
        } else {
            add_fail++;
            if(add_fail <= 3) {
                printf("\n[ADD MISMATCH] Instance %d:\n", i);
                print_limbs("  a       ", instances[i].a._limbs, 10);
                print_limbs("  b       ", instances[i].b._limbs, 10);
                print_limbs("  expected", instances[i].expected_sum._limbs, 10);
                print_limbs("  got     ", instances[i].result_sum._limbs, 10);
            }
        }

        // Check subtraction
        if(compare_limbs(instances[i].result_sub._limbs, instances[i].expected_sub._limbs, 10)) {
            sub_pass++;
        } else {
            sub_fail++;
            if(sub_fail <= 3) {
                printf("\n[SUB MISMATCH] Instance %d:\n", i);
                print_limbs("  a       ", instances[i].a._limbs, 10);
                print_limbs("  b       ", instances[i].b._limbs, 10);
                print_limbs("  expected", instances[i].expected_sub._limbs, 10);
                print_limbs("  got     ", instances[i].result_sub._limbs, 10);
            }
        }

        // Check multiplication
        if(compare_limbs(instances[i].result_product._limbs, instances[i].expected_product._limbs, 10)) {
            mul_pass++;
        } else {
            mul_fail++;
            if(mul_fail <= 3) {
                printf("\n[MUL MISMATCH] Instance %d:\n", i);
                print_limbs("  a       ", instances[i].a._limbs, 10);
                print_limbs("  b       ", instances[i].b._limbs, 10);
                print_limbs("  expected", instances[i].expected_product._limbs, 10);
                print_limbs("  got     ", instances[i].result_product._limbs, 10);
            }
        }
    }

    printf("\n=== Results ===\n");
    printf("Addition:       %d/%d passed", add_pass, INSTANCES);
    if(add_fail == 0) printf(" [OK]\n"); else printf(" [%d FAILED]\n", add_fail);

    printf("Subtraction:    %d/%d passed", sub_pass, INSTANCES);
    if(sub_fail == 0) printf(" [OK]\n"); else printf(" [%d FAILED]\n", sub_fail);

    printf("Multiplication: %d/%d passed", mul_pass, INSTANCES);
    if(mul_fail == 0) printf(" [OK]\n"); else printf(" [%d FAILED]\n", mul_fail);

    // Overall result
    bool all_passed = (add_fail == 0 && sub_fail == 0 && mul_fail == 0);

    if(all_passed) {
        printf("\n[SUCCESS] All MNT4-298 field arithmetic tests passed!\n");
        printf("CGBN is correctly computing modular add/sub/mul for the 298-bit field.\n");
    } else {
        printf("\n[FAILURE] Some tests failed - GPU/GMP mismatch detected.\n");
    }

    // Cleanup
    free(instances);
    CUDA_CHECK(cudaFree(gpu_instances));
    CUDA_CHECK(cgbn_error_report_free(report));

    printf("\n=== Test Complete ===\n");

    return all_passed ? 0 : 1;
}
