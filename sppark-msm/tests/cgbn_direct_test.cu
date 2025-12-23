// Direct CUDA test - no Rust, no test framework
// Compile: nvcc -o cgbn_direct_test tests/cgbn_direct_test.cu -I../cgbn-lib/include -I. -lgmp -std=c++17
// Run: ./cgbn_direct_test

#include <cuda_runtime.h>
#include <stdio.h>
#include <gmp.h>
#include <cgbn/cgbn.h>

// Forward declare the FFI function
extern "C" int msm_bw6_761_g1_cgbn(
    const void* points_ptr,
    const void* scalars_ptr,
    size_t count,
    void* result_ptr,
    size_t ffi_affine_sz,
    size_t ffi_scalar_sz
);

// Minimal test structures (matching Rust layout)
struct alignas(8) TestAffine {
    uint32_t x[24];  // 96 bytes
    uint32_t y[24];  // 96 bytes
    bool infinity;   // 1 byte + 7 padding = 8 bytes
    // Total: 200 bytes
};

struct alignas(8) TestScalar {
    uint64_t limbs[6];  // 48 bytes
};

struct alignas(8) TestProjective {
    uint32_t x[24];
    uint32_t y[24];
    uint32_t z[24];
    bool infinity;
};

int main() {
    printf("=== Direct CUDA CGBN Test (No Rust) ===\n");
    printf("TestAffine size: %zu, align: %zu\n", sizeof(TestAffine), alignof(TestAffine));
    printf("TestScalar size: %zu, align: %zu\n", sizeof(TestScalar), alignof(TestScalar));
    
    // Create test inputs
    TestAffine points[2];
    TestScalar scalars[2];
    TestProjective result;
    
    // Initialize with dummy data
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 24; j++) {
            points[i].x[j] = 1;
            points[i].y[j] = 1;
        }
        points[i].infinity = false;
        
        for (int j = 0; j < 6; j++) {
            scalars[i].limbs[j] = i + 1;
        }
    }
    
    printf("Calling CGBN kernel...\n");
    
    int status = msm_bw6_761_g1_cgbn(
        points,
        scalars,
        2,
        &result,
        sizeof(TestAffine),
        sizeof(TestScalar)
    );
    
    printf("Kernel returned status: %d\n", status);
    
    if (status == 0) {
        printf("✅ Kernel succeeded\n");
        printf("Result infinity: %d\n", result.infinity);
    } else {
        printf("❌ Kernel failed\n");
    }
    
    printf("Main function ending...\n");
    
    // Explicit cleanup
    cudaDeviceReset();
    
    printf("After cudaDeviceReset, exiting...\n");
    return 0;
}
