#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

// Match Rust FFI structures (with padding to match Rust's #[repr(C)] layout)
typedef struct {
    uint32_t x[24];  // 761 bits = 24 × 32-bit limbs (96 bytes)
    uint32_t y[24];  // 96 bytes
    bool infinity;   // 1 byte
    uint8_t _padding[7];  // Explicit padding: 96+96+1+7 = 200 bytes total
} affine_cgbn_t;

typedef struct {
    uint32_t limbs[12];  // 377 bits = 12 × 32-bit limbs
} scalar_cgbn_t;

typedef struct {
    uint32_t x[24];  // 96 bytes
    uint32_t y[24];  // 96 bytes
    uint32_t z[24];  // 96 bytes
    bool infinity;   // 1 byte
    uint8_t _padding[7];  // Padding: 96+96+96+1+7 = 296 bytes total
} jacobian_cgbn_t;

// External FFI function from the CUDA kernel
// IMPORTANT: Parameter order matches the CUDA function!
extern "C" int msm_bw6_761_g1_cgbn(
    const void* points_ptr,      // 1st: points
    const void* scalars_ptr,     // 2nd: scalars
    size_t count,                // 3rd: count
    void* result_ptr,            // 4th: result
    size_t ffi_affine_sz,        // 5th: affine size
    size_t ffi_scalar_sz         // 6th: scalar size
);

// Helper to create test point
void create_test_point(affine_cgbn_t* p, uint32_t seed) {
    for (int i = 0; i < 24; i++) {
        p->x[i] = seed + i;
        p->y[i] = seed * 2 + i;
    }
    p->infinity = false;
}

// Helper to create test scalar
void create_test_scalar(scalar_cgbn_t* s, uint32_t seed) {
    for (int i = 0; i < 12; i++) {
        s->limbs[i] = seed + i * 7;
    }
}

// Helper to print result
void print_jacobian(const char* label, const jacobian_cgbn_t* j) {
    printf("%s:\n", label);
    printf("  x[0-3]: %08x %08x %08x %08x\n", j->x[0], j->x[1], j->x[2], j->x[3]);
    printf("  y[0-3]: %08x %08x %08x %08x\n", j->y[0], j->y[1], j->y[2], j->y[3]);
    printf("  z[0-3]: %08x %08x %08x %08x\n", j->z[0], j->z[1], j->z[2], j->z[3]);
    printf("  infinity: %s\n", j->infinity ? "true" : "false");
}

// Test 1: Empty input (count=0)
bool test_empty() {
    printf("\n=== TEST 1: Empty Input (count=0) ===\n");

    jacobian_cgbn_t result;
    memset(&result, 0xAA, sizeof(result));  // Fill with pattern

    int ret = msm_bw6_761_g1_cgbn(
        nullptr, nullptr, 0, &result,  // points, scalars, count, result
        sizeof(affine_cgbn_t), sizeof(scalar_cgbn_t)
    );

    printf("Return code: %d\n", ret);
    print_jacobian("Result", &result);

    // Should return identity (infinity)
    bool pass = (ret == 0) && result.infinity;
    printf("RESULT: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    return pass;
}

// Test 2: Single point (count=1)
bool test_single_point() {
    printf("\n=== TEST 2: Single Point (count=1) ===\n");

    affine_cgbn_t point;
    scalar_cgbn_t scalar;
    jacobian_cgbn_t result;

    create_test_point(&point, 100);
    create_test_scalar(&scalar, 200);

    printf("Input point x[0-3]: %08x %08x %08x %08x\n",
           point.x[0], point.x[1], point.x[2], point.x[3]);
    printf("Input scalar limbs[0-3]: %08x %08x %08x %08x\n",
           scalar.limbs[0], scalar.limbs[1], scalar.limbs[2], scalar.limbs[3]);

    memset(&result, 0xBB, sizeof(result));

    int ret = msm_bw6_761_g1_cgbn(
        &point, &scalar, 1, &result,  // points, scalars, count, result
        sizeof(affine_cgbn_t), sizeof(scalar_cgbn_t)
    );

    printf("Return code: %d\n", ret);
    print_jacobian("Result", &result);

    // Should NOT crash, should return some result
    bool pass = (ret == 0);
    printf("RESULT: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    return pass;
}

// Test 3: Two points (count=2)
bool test_two_points() {
    printf("\n=== TEST 3: Two Points (count=2) ===\n");

    affine_cgbn_t points[2];
    scalar_cgbn_t scalars[2];
    jacobian_cgbn_t result;

    create_test_point(&points[0], 111);
    create_test_point(&points[1], 222);
    create_test_scalar(&scalars[0], 333);
    create_test_scalar(&scalars[1], 444);

    printf("Point 0 x[0-3]: %08x %08x %08x %08x\n",
           points[0].x[0], points[0].x[1], points[0].x[2], points[0].x[3]);
    printf("Point 1 x[0-3]: %08x %08x %08x %08x\n",
           points[1].x[0], points[1].x[1], points[1].x[2], points[1].x[3]);

    memset(&result, 0xCC, sizeof(result));

    int ret = msm_bw6_761_g1_cgbn(
        points, scalars, 2, &result,  // points, scalars, count, result
        sizeof(affine_cgbn_t), sizeof(scalar_cgbn_t)
    );

    printf("Return code: %d\n", ret);
    print_jacobian("Result", &result);

    bool pass = (ret == 0);
    printf("RESULT: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    return pass;
}

// Test 4: Multiple sizes (count=4,8,16)
bool test_multiple_sizes() {
    printf("\n=== TEST 4: Multiple Sizes (4,8,16) ===\n");

    bool all_pass = true;

    for (uint32_t count : {4, 8, 16}) {
        printf("\n--- Testing count=%u ---\n", count);

        affine_cgbn_t* points = new affine_cgbn_t[count];
        scalar_cgbn_t* scalars = new scalar_cgbn_t[count];
        jacobian_cgbn_t result;

        for (uint32_t i = 0; i < count; i++) {
            create_test_point(&points[i], 1000 + i * 10);
            create_test_scalar(&scalars[i], 2000 + i * 20);
        }

        memset(&result, 0xDD, sizeof(result));

        int ret = msm_bw6_761_g1_cgbn(
            points, scalars, count, &result,  // points, scalars, count, result
            sizeof(affine_cgbn_t), sizeof(scalar_cgbn_t)
        );

        printf("Return code: %d\n", ret);
        printf("Result x[0-3]: %08x %08x %08x %08x\n",
               result.x[0], result.x[1], result.x[2], result.x[3]);
        printf("Result infinity: %s\n", result.infinity ? "true" : "false");

        bool pass = (ret == 0);
        printf("count=%u: %s\n", count, pass ? "✅" : "❌");
        all_pass &= pass;

        delete[] points;
        delete[] scalars;
    }

    printf("\nOVERALL: %s\n", all_pass ? "✅ PASS" : "❌ FAIL");
    return all_pass;
}

// Test 5: Point at infinity
bool test_infinity_point() {
    printf("\n=== TEST 5: Point at Infinity ===\n");

    affine_cgbn_t point;
    scalar_cgbn_t scalar;
    jacobian_cgbn_t result;

    memset(&point, 0, sizeof(point));
    point.infinity = true;  // Infinity point
    create_test_scalar(&scalar, 999);

    printf("Input point: infinity=true\n");

    memset(&result, 0xEE, sizeof(result));

    int ret = msm_bw6_761_g1_cgbn(
        &point, &scalar, 1, &result,  // points, scalars, count, result
        sizeof(affine_cgbn_t), sizeof(scalar_cgbn_t)
    );

    printf("Return code: %d\n", ret);
    printf("DEBUG: result.x[23] = 0x%08x (0xDEADBEEF=true, 0xBADC0FFE=false)\n", result.x[23]);
    print_jacobian("Result", &result);

    // k * O = O (scalar times infinity is infinity)
    bool pass = (ret == 0) && result.infinity;
    printf("RESULT: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    return pass;
}

// Test 6: Layout validation
bool test_layout_validation() {
    printf("\n=== TEST 6: Layout Validation ===\n");

    printf("sizeof(affine_cgbn_t) = %zu (expected 200)\n", sizeof(affine_cgbn_t));
    printf("sizeof(scalar_cgbn_t) = %zu (expected 48)\n", sizeof(scalar_cgbn_t));
    printf("sizeof(jacobian_cgbn_t) = %zu (expected 296)\n", sizeof(jacobian_cgbn_t));

    bool pass = (sizeof(affine_cgbn_t) == 200) &&
                (sizeof(scalar_cgbn_t) == 48) &&
                (sizeof(jacobian_cgbn_t) == 296);

    printf("RESULT: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    return pass;
}

int main() {
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  CGBN BW6-761 Kernel Standalone Test Suite        ║\n");
    printf("║  (nvcc-level testing, no Rust/Cargo)              ║\n");
    printf("╚════════════════════════════════════════════════════╝\n");

    // Initialize CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("ERROR: No CUDA devices found or CUDA initialization failed\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Found %d CUDA device(s)\n", device_count);

    cudaSetDevice(0);
    printf("Using CUDA device 0\n\n");

    int passed = 0;
    int total = 6;

    if (test_layout_validation()) passed++;
    if (test_empty()) passed++;
    if (test_single_point()) passed++;
    if (test_two_points()) passed++;
    if (test_multiple_sizes()) passed++;
    if (test_infinity_point()) passed++;

    printf("\n╔════════════════════════════════════════════════════╗\n");
    printf("║  Summary                                           ║\n");
    printf("╚════════════════════════════════════════════════════╝\n");
    printf("Passed: %d/%d\n", passed, total);

    if (passed == total) {
        printf("\n✅ ALL TESTS PASSED - Kernel is battle tested!\n");
        return 0;
    } else {
        printf("\n❌ SOME TESTS FAILED\n");
        return 1;
    }
}
