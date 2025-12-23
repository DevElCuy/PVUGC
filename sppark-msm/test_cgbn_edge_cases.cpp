// Standalone C++ test for CGBN BW6-761 MSM edge cases
// Tests: count=0, count=1, count=2 to isolate crashes
//
// Compile:
//   nvcc -o test_cgbn_edge_cases test_cgbn_edge_cases.cpp src/msm_bw6_761_cgbn.cu \
//        -I./sppark -I./sppark/util -DCGBN_TPI=8 -DCGBN_BITS=768 --std=c++17 \
//        -I/usr/include/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lgmp

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

// Match CUDA kernel expectations
struct alignas(8) TestAffine {
    uint32_t x[24];
    uint32_t y[24];
    bool infinity;
    uint8_t _padding[7];
};

struct alignas(8) TestScalar {
    uint64_t limbs[6];
};

struct alignas(8) TestProjective {
    uint32_t x[24];
    uint32_t y[24];
    uint32_t z[24];
    bool infinity;
};

// Forward declare CGBN kernel FFI
extern "C" int msm_bw6_761_g1_cgbn(
    const void* points_ptr,
    const void* scalars_ptr,
    size_t count,
    void* result_ptr,
    size_t ffi_affine_sz,
    size_t ffi_scalar_sz
);

void print_result(const TestProjective& result, const char* label) {
    printf("  %s:\n", label);
    printf("    infinity: %d\n", result.infinity);
    printf("    x[0]: 0x%08x\n", result.x[0]);
    printf("    y[0]: 0x%08x\n", result.y[0]);
    printf("    z[0]: 0x%08x\n", result.z[0]);
}

bool test_count_0() {
    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║  TEST 1: count=0 (empty input)            ║\n");
    printf("╚═══════════════════════════════════════════╝\n");

    TestProjective result;
    memset(&result, 0xAA, sizeof(result)); // Fill with pattern to detect uninitialized

    printf("  Calling kernel with count=0...\n");
    int status = msm_bw6_761_g1_cgbn(
        nullptr,
        nullptr,
        0,
        &result,
        sizeof(TestAffine),
        sizeof(TestScalar)
    );

    printf("  Status: %d\n", status);
    print_result(result, "Result");

    if (status == 0 && result.infinity) {
        printf("  ✅ PASS: count=0 returns identity\n");
        return true;
    } else {
        printf("  ❌ FAIL: count=0 expected status=0 and infinity=true\n");
        return false;
    }
}

bool test_count_1() {
    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║  TEST 2: count=1 (single point)           ║\n");
    printf("╚═══════════════════════════════════════════╝\n");

    // Create generator point (all zeros represents generator in some conventions)
    TestAffine point;
    memset(&point, 0, sizeof(point));
    point.infinity = false;

    TestScalar scalar;
    memset(&scalar, 0, sizeof(scalar));
    scalar.limbs[0] = 5; // 5*G

    TestProjective result;
    memset(&result, 0xBB, sizeof(result));

    printf("  Calling kernel with count=1...\n");
    int status = msm_bw6_761_g1_cgbn(
        &point,
        &scalar,
        1,
        &result,
        sizeof(TestAffine),
        sizeof(TestScalar)
    );

    printf("  Status: %d\n", status);
    print_result(result, "Result");

    if (status == 0) {
        printf("  ✅ PASS: count=1 completed successfully\n");
        return true;
    } else {
        printf("  ❌ FAIL: count=1 returned error status %d\n", status);
        return false;
    }
}

bool test_count_2() {
    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║  TEST 3: count=2 (normal case)            ║\n");
    printf("╚═══════════════════════════════════════════╝\n");

    TestAffine points[2];
    memset(points, 0, sizeof(points));
    points[0].infinity = false;
    points[1].infinity = false;

    TestScalar scalars[2];
    memset(scalars, 0, sizeof(scalars));
    scalars[0].limbs[0] = 2;
    scalars[1].limbs[0] = 3;

    TestProjective result;
    memset(&result, 0xCC, sizeof(result));

    printf("  Calling kernel with count=2...\n");
    int status = msm_bw6_761_g1_cgbn(
        points,
        scalars,
        2,
        &result,
        sizeof(TestAffine),
        sizeof(TestScalar)
    );

    printf("  Status: %d\n", status);
    print_result(result, "Result");

    if (status == 0) {
        printf("  ✅ PASS: count=2 completed successfully\n");
        return true;
    } else {
        printf("  ❌ FAIL: count=2 returned error status %d\n", status);
        return false;
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║  CGBN BW6-761 Edge Cases Test (Pure C++/CUDA)    ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n");

    printf("\nSize validation:\n");
    printf("  TestAffine:     %zu bytes (expected 200)\n", sizeof(TestAffine));
    printf("  TestScalar:     %zu bytes (expected 48)\n", sizeof(TestScalar));
    printf("  TestProjective: %zu bytes\n", sizeof(TestProjective));

    if (sizeof(TestAffine) != 200) {
        printf("  ❌ FATAL: TestAffine size mismatch!\n");
        return 1;
    }

    int passed = 0;
    int failed = 0;

    // Run all tests
    if (test_count_0()) passed++; else failed++;
    if (test_count_1()) passed++; else failed++;
    if (test_count_2()) passed++; else failed++;

    printf("\n╔═══════════════════════════════════════════════════╗\n");
    printf("║  SUMMARY                                          ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n");
    printf("  Passed: %d\n", passed);
    printf("  Failed: %d\n", failed);

    if (failed == 0) {
        printf("\n  ✅ ALL TESTS PASSED\n");
        printf("  If this exits cleanly, the kernel is fine.\n");
        printf("  If Rust tests crash, it's a Rust/CUDA interaction issue.\n");
    } else {
        printf("\n  ❌ SOME TESTS FAILED\n");
        printf("  Kernel has bugs that need fixing.\n");
    }

    printf("\nExiting main()...\n");
    return (failed == 0) ? 0 : 1;
}
