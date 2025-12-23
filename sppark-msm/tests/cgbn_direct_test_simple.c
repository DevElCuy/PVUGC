// Simpler C test - link against precompiled library
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

// Minimal structures
typedef struct __attribute__((aligned(8))) {
    uint32_t x[24];
    uint32_t y[24];
    bool infinity;
    char padding[7];
} test_affine_t;

typedef struct __attribute__((aligned(8))) {
    uint64_t limbs[6];
} test_scalar_t;

typedef struct __attribute__((aligned(8))) {
    uint32_t x[24];
    uint32_t y[24];
    uint32_t z[24];
    bool infinity;
} test_projective_t;

// Forward declare FFI
extern int msm_bw6_761_g1_cgbn(
    const void* points,
    const void* scalars,
    size_t count,
    void* result,
    size_t affine_size,
    size_t scalar_size
);

int main() {
    printf("=== Direct C Test (No Rust) ===\n");
    printf("Affine size: %zu, Scalar size: %zu\n", 
           sizeof(test_affine_t), sizeof(test_scalar_t));
    
    test_affine_t points[2];
    test_scalar_t scalars[2];
    test_projective_t result;
    
    memset(points, 0, sizeof(points));
    memset(scalars, 0, sizeof(scalars));
    memset(&result, 0, sizeof(result));
    
    // Set minimal valid data
    points[0].infinity = false;
    points[1].infinity = false;
    scalars[0].limbs[0] = 2;
    scalars[1].limbs[0] = 3;
    
    printf("Calling kernel...\n");
    int status = msm_bw6_761_g1_cgbn(
        points, scalars, 2, &result,
        sizeof(test_affine_t), sizeof(test_scalar_t)
    );
    
    printf("Status: %d\n", status);
    printf("Main ending cleanly...\n");
    return 0;
}
