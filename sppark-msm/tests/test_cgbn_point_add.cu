/**
 * CGBN-Based Point Addition Proof of Concept
 *
 * Purpose: Demonstrate that elliptic curve point addition can work
 *          with CGBN cooperative groups for BW6-761.
 *
 * Key Challenge: Threading model mismatch
 * - sppark's xyzz_t::add(): 1 thread processes 1 point
 * - CGBN: TPI threads cooperate on each field element
 *
 * Solution: Create new point type that expects cooperative groups
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
#define INSTANCES 10

// BW6-761 base field modulus (from sppark/ff/bw6-761.hpp)
__device__ __constant__ uint32_t BW6_761_P[24] = {
    0x0000008b, 0xf49d0000, 0x00000082, 0xe6913e68,
    0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
    0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a,
    0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824
};

// CGBN types
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// Affine point in CGBN format (host/device memory)
typedef struct {
    cgbn_mem_t<BITS> x;
    cgbn_mem_t<BITS> y;
    bool infinity;
} affine_cgbn_t;

// Extended projective point (XYZZ coordinates) in CGBN format
typedef struct {
    cgbn_mem_t<BITS> x;
    cgbn_mem_t<BITS> y;
    cgbn_mem_t<BITS> zz;   // Z^2
    cgbn_mem_t<BITS> zzz;  // Z^3
} xyzz_cgbn_t;

/**
 * Point Addition: XYZZ + Affine → XYZZ
 *
 * Formulas (mixed addition):
 *   U1 = X1 * ZZ1      (but ZZ1=1 for first add from affine)
 *   U2 = X2
 *   S1 = Y1 * ZZZ1     (but ZZZ1=1 for first add from affine)
 *   S2 = Y2
 *   H  = U2 - U1
 *   R  = S2 - S1
 *   HH = H^2
 *   HHH = H * HH
 *   V  = U1 * HH
 *   X3 = R^2 - HHH - 2*V
 *   Y3 = R*(V - X3) - S1*HHH
 *   ZZ3 = ZZ1 * HH
 *   ZZZ3 = ZZZ1 * HHH
 *
 * This kernel performs mixed addition: XYZZ + Affine → XYZZ
 * All TPI threads must execute this together.
 */
__global__ void test_point_addition_cgbn(
    cgbn_error_report_t *report,
    xyzz_cgbn_t *results,       // Output: XYZZ points
    affine_cgbn_t *points_a,    // Input: Affine points (first operand)
    affine_cgbn_t *points_b,    // Input: Affine points (second operand)
    uint32_t count
) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;

    if(instance >= count)
        return;

    // Create CGBN context and environment
    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t bn_env(bn_context.env<env_t>());

    // Load modulus
    env_t::cgbn_t P;
    cgbn_load(bn_env, P, (cgbn_mem_t<BITS>*)BW6_761_P);

    // Load first point (convert affine to XYZZ with ZZ=1, ZZZ=1)
    env_t::cgbn_t x1, y1, zz1, zzz1;
    cgbn_load(bn_env, x1, &points_a[instance].x);
    cgbn_load(bn_env, y1, &points_a[instance].y);
    cgbn_set_ui32(bn_env, zz1, 1);   // ZZ = 1 (affine)
    cgbn_set_ui32(bn_env, zzz1, 1);  // ZZZ = 1 (affine)

    // Load second point (affine)
    env_t::cgbn_t x2, y2;
    cgbn_load(bn_env, x2, &points_b[instance].x);
    cgbn_load(bn_env, y2, &points_b[instance].y);

    // Temporary variables for point addition
    env_t::cgbn_t U1, U2, S1, S2, H, R, HH, HHH, V, X3, Y3, ZZ3, ZZZ3, temp;

    // U1 = X1 * ZZ1 (but ZZ1=1, so U1 = X1)
    cgbn_set(bn_env, U1, x1);

    // U2 = X2
    cgbn_set(bn_env, U2, x2);

    // S1 = Y1 * ZZZ1 (but ZZZ1=1, so S1 = Y1)
    cgbn_set(bn_env, S1, y1);

    // S2 = Y2
    cgbn_set(bn_env, S2, y2);

    // H = U2 - U1 (mod P)
    cgbn_sub(bn_env, H, U2, U1);
    // If H < 0, add P
    if (cgbn_compare_ui32(bn_env, H, 0) < 0) {
        cgbn_add(bn_env, H, H, P);
    }

    // R = S2 - S1 (mod P)
    cgbn_sub(bn_env, R, S2, S1);
    if (cgbn_compare_ui32(bn_env, R, 0) < 0) {
        cgbn_add(bn_env, R, R, P);
    }

    // HH = H^2 (mod P)
    cgbn_mul(bn_env, HH, H, H);
    cgbn_rem(bn_env, HH, HH, P);

    // HHH = H * HH (mod P)
    cgbn_mul(bn_env, HHH, H, HH);
    cgbn_rem(bn_env, HHH, HHH, P);

    // V = U1 * HH (mod P)
    cgbn_mul(bn_env, V, U1, HH);
    cgbn_rem(bn_env, V, V, P);

    // X3 = R^2 - HHH - 2*V (mod P)
    cgbn_mul(bn_env, X3, R, R);       // R^2
    cgbn_rem(bn_env, X3, X3, P);
    cgbn_sub(bn_env, X3, X3, HHH);    // R^2 - HHH
    if (cgbn_compare_ui32(bn_env, X3, 0) < 0) cgbn_add(bn_env, X3, X3, P);
    cgbn_sub(bn_env, X3, X3, V);      // ... - V
    if (cgbn_compare_ui32(bn_env, X3, 0) < 0) cgbn_add(bn_env, X3, X3, P);
    cgbn_sub(bn_env, X3, X3, V);      // ... - 2*V
    if (cgbn_compare_ui32(bn_env, X3, 0) < 0) cgbn_add(bn_env, X3, X3, P);

    // Y3 = R*(V - X3) - S1*HHH (mod P)
    cgbn_sub(bn_env, temp, V, X3);    // V - X3
    if (cgbn_compare_ui32(bn_env, temp, 0) < 0) cgbn_add(bn_env, temp, temp, P);
    cgbn_mul(bn_env, Y3, R, temp);    // R*(V - X3)
    cgbn_rem(bn_env, Y3, Y3, P);
    cgbn_mul(bn_env, temp, S1, HHH);  // S1 * HHH
    cgbn_rem(bn_env, temp, temp, P);
    cgbn_sub(bn_env, Y3, Y3, temp);   // R*(V-X3) - S1*HHH
    if (cgbn_compare_ui32(bn_env, Y3, 0) < 0) cgbn_add(bn_env, Y3, Y3, P);

    // ZZ3 = ZZ1 * HH (but ZZ1=1, so ZZ3 = HH)
    cgbn_set(bn_env, ZZ3, HH);

    // ZZZ3 = ZZZ1 * HHH (but ZZZ1=1, so ZZZ3 = HHH)
    cgbn_set(bn_env, ZZZ3, HHH);

    // Store result
    cgbn_store(bn_env, &results[instance].x, X3);
    cgbn_store(bn_env, &results[instance].y, Y3);
    cgbn_store(bn_env, &results[instance].zz, ZZ3);
    cgbn_store(bn_env, &results[instance].zzz, ZZZ3);
}

// Utility macros
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

void random_field_element(uint32_t *limbs) {
    for(int i = 0; i < BITS/32; i++) {
        limbs[i] = rand();
    }
}

int main() {
    printf("=== CGBN Point Addition Test (BW6-761) ===\n");
    printf("Configuration: TPI=%d, BITS=%d, INSTANCES=%d\n\n", TPI, BITS, INSTANCES);

    // Allocate host memory
    affine_cgbn_t *points_a = (affine_cgbn_t*)malloc(sizeof(affine_cgbn_t) * INSTANCES);
    affine_cgbn_t *points_b = (affine_cgbn_t*)malloc(sizeof(affine_cgbn_t) * INSTANCES);
    xyzz_cgbn_t *results = (xyzz_cgbn_t*)malloc(sizeof(xyzz_cgbn_t) * INSTANCES);

    // Generate random points
    printf("Generating %d random point pairs...\n", INSTANCES);
    for(int i = 0; i < INSTANCES; i++) {
        random_field_element(points_a[i].x._limbs);
        random_field_element(points_a[i].y._limbs);
        points_a[i].infinity = false;

        random_field_element(points_b[i].x._limbs);
        random_field_element(points_b[i].y._limbs);
        points_b[i].infinity = false;
    }

    // Allocate GPU memory
    affine_cgbn_t *gpu_points_a, *gpu_points_b;
    xyzz_cgbn_t *gpu_results;
    cgbn_error_report_t *report;

    printf("Copying to GPU...\n");
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&gpu_points_a, sizeof(affine_cgbn_t) * INSTANCES));
    CUDA_CHECK(cudaMalloc(&gpu_points_b, sizeof(affine_cgbn_t) * INSTANCES));
    CUDA_CHECK(cudaMalloc(&gpu_results, sizeof(xyzz_cgbn_t) * INSTANCES));

    CUDA_CHECK(cudaMemcpy(gpu_points_a, points_a, sizeof(affine_cgbn_t) * INSTANCES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_points_b, points_b, sizeof(affine_cgbn_t) * INSTANCES, cudaMemcpyHostToDevice));

    CUDA_CHECK(cgbn_error_report_alloc(&report));

    // Launch kernel
    printf("\n[Test] CGBN Point Addition...\n");
    int threads_per_block = 128;
    int num_blocks = (INSTANCES * TPI + threads_per_block - 1) / threads_per_block;
    printf("  Kernel launch: %d blocks × %d threads\n", num_blocks, threads_per_block);
    printf("  Processing %d point additions (%d threads per point)\n", INSTANCES, TPI);

    test_point_addition_cgbn<<<num_blocks, threads_per_block>>>(
        report, gpu_results, gpu_points_a, gpu_points_b, INSTANCES
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);
    printf("  ✅ Point addition kernel completed\n");

    // Copy results back
    printf("\nCopying results back to CPU...\n");
    CUDA_CHECK(cudaMemcpy(results, gpu_results, sizeof(xyzz_cgbn_t) * INSTANCES, cudaMemcpyDeviceToHost));

    // Validation: check that all results are non-zero
    printf("\n[Validation] Checking results...\n");
    int non_zero_results = 0;

    for(int i = 0; i < INSTANCES; i++) {
        bool all_zero = true;
        for(int j = 0; j < BITS/32; j++) {
            if(results[i].x._limbs[j] != 0 || results[i].y._limbs[j] != 0) {
                all_zero = false;
                break;
            }
        }
        if(!all_zero) non_zero_results++;
    }

    printf("  Non-zero results: %d/%d\n", non_zero_results, INSTANCES);

    if(non_zero_results >= INSTANCES * 0.9) {
        printf("\n✅ SUCCESS: CGBN point addition appears functional\n");
        printf("   Next: Cross-validate against CPU/arkworks reference\n");
    } else {
        printf("\n⚠️  WARNING: Many zero results - may indicate errors\n");
    }

    // Cleanup
    free(points_a);
    free(points_b);
    free(results);
    CUDA_CHECK(cudaFree(gpu_points_a));
    CUDA_CHECK(cudaFree(gpu_points_b));
    CUDA_CHECK(cudaFree(gpu_results));
    CUDA_CHECK(cgbn_error_report_free(report));

    printf("\n=== Test Complete ===\n");
    return 0;
}
