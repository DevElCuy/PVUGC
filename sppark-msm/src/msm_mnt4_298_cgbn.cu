// MNT4-298 CGBN-Based MSM Implementation
// Uses NVIDIA CGBN library for cooperative-group field arithmetic
// Follows BW6-761 CGBN pattern with 320-bit field (10 x 32-bit limbs)

#include <stddef.h>
#include <cuda_runtime.h>

// IMPORTANT: Include gmp.h BEFORE cgbn.h to avoid cgbn_cpu.h stub
// CGBN checks for __GMP_H__ and uses cgbn_mpz.h instead of cgbn_cpu.h
#include <gmp.h>
#include <cgbn/cgbn.h>

// Error codes
#define MNT4_MSM_SUCCESS 0
#define MNT4_MSM_ERROR_AFFINE_LAYOUT -1
#define MNT4_MSM_ERROR_SCALAR_LAYOUT -2
#define MNT4_MSM_ERROR_CUDA_RUNTIME -3
#define MNT4_MSM_ERROR_TIMEOUT -4

// CGBN parameters class (following CGBN sample pattern)
class mnt4_cgbn_params_t {
public:
  static const uint32_t TPB = 0;            // Get TPB from blockDim.x
  static const uint32_t MAX_ROTATION = 4;   // Good default value
  static const uint32_t SHM_LIMIT = 0;      // No shared memory
  static const bool CONSTANT_TIME = false;  // Not available yet
  static const uint32_t TPI = 8;            // 8 threads cooperate per big number
  static const uint32_t BITS = 320;         // Round up from 298 bits
};

// MNT4-298 base field modulus (Fq, 298 bits)
// Little-endian u32[10] representation
__device__ __constant__ uint32_t MNT4_298_P_DEVICE[10] = {
    0x71660001, 0xc90cd65a, 0x51200e12, 0x41a9e35e, 0x5d1330ea,
    0xcaeec963, 0xa7b0548e, 0xa266249d, 0xf7bcd473, 0x000003bc
};

// MNT4-298 scalar field order (Fr, 298 bits) == MNT6-298 Fq
__device__ __constant__ uint32_t MNT4_298_R_DEVICE[10] = {
    0x00000001, 0xbb4334a4, 0x925d6ad3, 0xfb494c07, 0x5cf44194,
    0xcaeec963, 0xa7b0548e, 0xa266249d, 0xf7bcd473, 0x000003bc
};

// Curve parameter a for MNT4-298 G1: a = 2
#define MNT4_CURVE_A 2

// Affine point (input format - matches arkworks G1Affine)
typedef struct {
    uint32_t x[10];  // 40 bytes (298 bits + padding)
    uint32_t y[10];  // 40 bytes
    bool infinity;   // 1 byte
    // Note: arkworks uses 8-byte alignment
} __align__(8) affine_cgbn_t;

// Projective point (XYZZ coordinates) for computation
typedef struct {
    cgbn_mem_t<320> x;
    cgbn_mem_t<320> y;
    cgbn_mem_t<320> zz;   // Z^2
    cgbn_mem_t<320> zzz;  // Z^3
} xyzz_cgbn_t;

// Jacobian point (output format - matches arkworks G1Projective)
typedef struct {
    uint32_t x[10];  // 40 bytes
    uint32_t y[10];  // 40 bytes
    uint32_t z[10];  // 40 bytes
    bool infinity;   // 1 byte
    // Note: arkworks uses 8-byte alignment
} __align__(8) jacobian_cgbn_t;

// Scalar type (Fr element, 298 bits = 40 bytes)
typedef struct {
    uint32_t limbs[10];  // 10 × 32-bit = 320 bits (padded from 298)
    // Note: arkworks BigInt<5> uses 8-byte alignment
} __align__(8) scalar_cgbn_t;

/**
 * MNT4-298 MSM Class (following CGBN sample pattern)
 *
 * This class encapsulates CGBN context and environment as member variables
 * to avoid parameter passing overhead that causes stack overflow.
 *
 * Pattern from: NVlabs/CGBN sample_03_powm/powm_odd.cu
 */
template<class params>
class mnt4_msm_t {
public:
  // CGBN type definitions
  typedef cgbn_context_t<params::TPI, params> context_t;
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  // Member variables (not passed as parameters!)
  context_t _context;
  env_t     _env;
  int32_t   _instance;

  // Constructor
  __device__ __forceinline__ mnt4_msm_t(cgbn_monitor_t monitor,
                                        cgbn_error_report_t *report,
                                        int32_t instance)
      : _context(monitor, report, (uint32_t)instance),
        _env(_context),
        _instance(instance) {}

  /**
   * Field addition with modular reduction: r = (a + b) mod P
   */
  __device__ __forceinline__ void field_add(bn_t& r, const bn_t& a,
                                             const bn_t& b, const bn_t& P) {
    cgbn_add(_env, r, a, b);
    cgbn_rem(_env, r, r, P);
  }

  /**
   * Field subtraction with modular reduction: r = (a - b) mod P
   *
   * CRITICAL: CGBN uses unsigned arithmetic. cgbn_sub returns a borrow flag
   * indicating underflow, NOT a negative number. We must use the return value
   * to detect when we need to add the modulus.
   */
  __device__ __forceinline__ void field_sub(bn_t& r, const bn_t& a,
                                             const bn_t& b, const bn_t& P) {
    // cgbn_sub returns 1 if there was a borrow (underflow), 0 otherwise
    int32_t borrow = cgbn_sub(_env, r, a, b);
    if (borrow != 0) {
      cgbn_add(_env, r, r, P);
    }
  }

  /**
   * Field multiplication with modular reduction: r = (a * b) mod P
   *
   * CRITICAL: Must use wide multiplication (cgbn_mul_wide) because:
   * - a, b are up to 298 bits each
   * - a * b is up to 596 bits (doesn't fit in 320-bit bn_t)
   * - cgbn_mul only keeps the low 320 bits, losing upper bits!
   * - Must use cgbn_rem_wide to compute (full_product) mod P
   */
  __device__ __forceinline__ void field_mul(bn_t& r, const bn_t& a,
                                             const bn_t& b, const bn_t& P) {
    // Wide type holds 2*BITS = 640 bits, enough for 298*2 = 596 bit product
    typedef typename env_t::cgbn_wide_t wide_t;
    wide_t product;

    // Full multiplication: product = a * b (up to 596 bits)
    cgbn_mul_wide(_env, product, a, b);

    // Wide remainder: r = product mod P
    cgbn_rem_wide(_env, r, product, P);
  }

  /**
   * Point Addition: XYZZ + Affine → XYZZ (Mixed Addition)
   *
   * XYZZ Mixed Addition Formulas from hyperelliptic.org "madd-2008-s":
   * https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
   *
   * In XYZZ form: point (X, Y, ZZ, ZZZ) represents affine (X/ZZ, Y/ZZZ)
   * where ZZ = Z^2, ZZZ = Z^3
   *
   * Formulas (mixed addition with affine point where Z2=1, ZZ2=1, ZZZ2=1):
   *   U2 = X2*ZZ1
   *   S2 = Y2*ZZZ1
   *   P = U2-X1
   *   R = S2-Y1
   *   PP = P^2
   *   PPP = P*PP
   *   Q = X1*PP
   *   X3 = R^2-PPP-2*Q
   *   Y3 = R*(Q-X3)-Y1*PPP
   *   ZZ3 = ZZ1*PP
   *   ZZZ3 = ZZZ1*PPP
   *
   * All TPI threads must call this together
   */
  __device__ __forceinline__ void point_add_mixed(
      const bn_t& P_mod,
      // Result (XYZZ)
      bn_t& X3, bn_t& Y3, bn_t& ZZ3, bn_t& ZZZ3,
      // First operand (XYZZ): represents affine (X1/ZZ1, Y1/ZZZ1)
      const bn_t& X1, const bn_t& Y1,
      const bn_t& ZZ1, const bn_t& ZZZ1,
      // Second operand (Affine): (X2, Y2) with implicit Z2=1
      const bn_t& X2, const bn_t& Y2
  ) {
    bn_t U2, S2, P, R, PP, PPP, Q, temp;

    // U2 = X2 * ZZ1 (scale affine X2 to match XYZZ coordinate system)
    field_mul(U2, X2, ZZ1, P_mod);

    // S2 = Y2 * ZZZ1 (scale affine Y2 to match XYZZ coordinate system)
    field_mul(S2, Y2, ZZZ1, P_mod);

    // P = U2 - X1 (difference in scaled x-coordinates)
    field_sub(P, U2, X1, P_mod);

    // R = S2 - Y1 (difference in scaled y-coordinates)
    field_sub(R, S2, Y1, P_mod);

    // PP = P^2
    field_mul(PP, P, P, P_mod);

    // PPP = P * PP = P^3
    field_mul(PPP, P, PP, P_mod);

    // Q = X1 * PP
    field_mul(Q, X1, PP, P_mod);

    // X3 = R^2 - PPP - 2*Q
    field_mul(X3, R, R, P_mod);
    field_sub(X3, X3, PPP, P_mod);
    field_sub(X3, X3, Q, P_mod);
    field_sub(X3, X3, Q, P_mod);

    // Y3 = R*(Q - X3) - Y1*PPP
    field_sub(temp, Q, X3, P_mod);
    field_mul(Y3, R, temp, P_mod);
    field_mul(temp, Y1, PPP, P_mod);
    field_sub(Y3, Y3, temp, P_mod);

    // ZZ3 = ZZ1 * PP
    field_mul(ZZ3, ZZ1, PP, P_mod);

    // ZZZ3 = ZZZ1 * PPP
    field_mul(ZZZ3, ZZZ1, PPP, P_mod);
  }

  /**
   * Point Doubling: XYZZ → XYZZ (2P)
   *
   * XYZZ Doubling Formulas for y^2 = x^3 + ax + b (MNT4-298 has a=2):
   * A = Y1^2
   * V = 4*A
   * U = 2*Y1
   * W = U*V = 8*Y1*A
   * S = X1*V
   * M = 3*X1^2 + a*ZZ1^2  (for a=2: M = 3*X1^2 + 2*ZZ1^2)
   *
   * X3 = M^2 - 2*S
   * Y3 = M*(S - X3) - 2*A*V  (= M*(S - X3) - 8*Y1^4)
   * ZZ3 = V*ZZ1
   * ZZZ3 = W*ZZZ1
   *
   * All TPI threads must call this together
   */
  __device__ __forceinline__ void point_double(
      const bn_t& P,
      bn_t& X3, bn_t& Y3, bn_t& ZZ3, bn_t& ZZZ3,
      const bn_t& X1, const bn_t& Y1,
      const bn_t& ZZ1, const bn_t& ZZZ1
  ) {
    bn_t A, V, U, W, S, M, temp, a_term;

    // A = Y1^2
    field_mul(A, Y1, Y1, P);

    // U = 2*Y1
    field_add(U, Y1, Y1, P);

    // V = 4*A
    field_add(V, A, A, P);
    field_add(V, V, V, P);

    // W = U*V = 8*Y1*A
    field_mul(W, U, V, P);

    // S = X1*V
    field_mul(S, X1, V, P);

    // M = 3*X1^2 + a*ZZ1^2  (MNT4-298 has a=2)
    field_mul(M, X1, X1, P);
    field_add(temp, M, M, P);
    field_add(M, M, temp, P);  // M = 3*X1^2

    // Add a*ZZ1^2 term (a=2 for MNT4)
    field_mul(a_term, ZZ1, ZZ1, P);  // ZZ1^2
    field_add(a_term, a_term, a_term, P);  // 2*ZZ1^2 (since a=2)
    field_add(M, M, a_term, P);  // M = 3*X1^2 + 2*ZZ1^2

    // X3 = M^2 - 2*S
    field_mul(X3, M, M, P);
    field_sub(X3, X3, S, P);
    field_sub(X3, X3, S, P);

    // Y3 = M*(S - X3) - 2*A*V
    field_sub(temp, S, X3, P);
    field_mul(Y3, M, temp, P);
    field_mul(temp, A, V, P);
    field_add(temp, temp, temp, P);
    field_sub(Y3, Y3, temp, P);

    // ZZ3 = V*ZZ1
    field_mul(ZZ3, V, ZZ1, P);

    // ZZZ3 = W*ZZZ1
    field_mul(ZZZ3, W, ZZZ1, P);
  }

  /**
   * Scalar Multiplication: k * P using double-and-add algorithm
   *
   * Algorithm:
   * 1. Handle special cases (k=0, k=1, point at infinity)
   * 2. Find highest set bit in scalar
   * 3. Initialize accumulator with point (for MSB)
   * 4. For each bit from MSB-1 down to 0:
   *    a. Double accumulator
   *    b. If bit is 1, add point to accumulator
   *
   * Scalar field Fr is 298 bits (10 × 32-bit limbs)
   * All TPI threads must call this together
   */
  __device__ __forceinline__ void scalar_mul(
      const bn_t& P,
      // Result (XYZZ) - will be in projective form
      bn_t& rx, bn_t& ry, bn_t& rzz, bn_t& rzzz,
      // Point (Affine)
      const bn_t& px, const bn_t& py,
      bool point_is_infinity,
      // Scalar (Fr element, 298 bits = 10 limbs)
      const uint32_t scalar_limbs[10],
      // Output: whether result is infinity
      bool& result_infinity
  ) {
    int lane = threadIdx.x % params::TPI;

    // Handle point at infinity
    if (point_is_infinity) {
      result_infinity = true;
      return;
    }

    // Check if scalar is zero
    bool scalar_is_zero = true;
    for (int i = 0; i < 10; i++) {
      if (scalar_limbs[i] != 0) {
        scalar_is_zero = false;
        break;
      }
    }

    if (scalar_is_zero) {
      result_infinity = true;
      return;
    }

    // Find the highest set bit in the scalar (scan from MSB)
    // Scalar is 298 bits = 10 × 32-bit limbs
    int highest_bit = -1;
    for (int limb_idx = 9; limb_idx >= 0; limb_idx--) {
      uint32_t limb = scalar_limbs[limb_idx];
      if (limb != 0) {
        // Find highest bit in this limb (31 down to 0)
        for (int bit_idx = 31; bit_idx >= 0; bit_idx--) {
          if (limb & (1u << bit_idx)) {
            highest_bit = limb_idx * 32 + bit_idx;
            break;
          }
        }
        if (highest_bit >= 0) break;
      }
    }

    // If no bits set (shouldn't happen after zero check, but be safe)
    if (highest_bit < 0) {
      result_infinity = true;
      return;
    }

    // Initialize accumulator with the point (for the MSB)
    // Start in XYZZ coordinates: (px, py, 1, 1)
    bn_t acc_x, acc_y, acc_zz, acc_zzz;
    cgbn_set(_env, acc_x, px);
    cgbn_set(_env, acc_y, py);
    cgbn_set_ui32(_env, acc_zz, 1);
    cgbn_set_ui32(_env, acc_zzz, 1);

    // Declare temps ONCE (not inside loop!) to avoid register explosion
    bn_t temp_x, temp_y, temp_zz, temp_zzz;

    // Double-and-add from MSB-1 down to bit 0
    for (int bit_pos = highest_bit - 1; bit_pos >= 0; bit_pos--) {
      // Double: acc = 2 * acc
      point_double(P, temp_x, temp_y, temp_zz, temp_zzz,
                   acc_x, acc_y, acc_zz, acc_zzz);

      cgbn_set(_env, acc_x, temp_x);
      cgbn_set(_env, acc_y, temp_y);
      cgbn_set(_env, acc_zz, temp_zz);
      cgbn_set(_env, acc_zzz, temp_zzz);

      // Check if bit is set
      int limb_idx = bit_pos / 32;
      int bit_idx = bit_pos % 32;
      bool bit_is_set = (scalar_limbs[limb_idx] & (1u << bit_idx)) != 0;

      // Add: acc = acc + P (if bit is 1)
      if (bit_is_set) {
        point_add_mixed(P, temp_x, temp_y, temp_zz, temp_zzz,
                        acc_x, acc_y, acc_zz, acc_zzz,
                        px, py);

        cgbn_set(_env, acc_x, temp_x);
        cgbn_set(_env, acc_y, temp_y);
        cgbn_set(_env, acc_zz, temp_zz);
        cgbn_set(_env, acc_zzz, temp_zzz);
      }
    }

    // Store result
    cgbn_set(_env, rx, acc_x);
    cgbn_set(_env, ry, acc_y);
    cgbn_set(_env, rzz, acc_zz);
    cgbn_set(_env, rzzz, acc_zzz);
    result_infinity = false;
  }

  /**
   * Point Addition: XYZZ + XYZZ → XYZZ (Full Addition)
   *
   * XYZZ Full Addition Formulas from hyperelliptic.org "add-2008-s":
   * https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
   *
   * Formulas:
   *   U1 = X1*ZZ2
   *   U2 = X2*ZZ1
   *   S1 = Y1*ZZZ2
   *   S2 = Y2*ZZZ1
   *   P = U2-U1
   *   R = S2-S1
   *   PP = P^2
   *   PPP = P*PP
   *   Q = U1*PP
   *   X3 = R^2 - PPP - 2*Q
   *   Y3 = R*(Q-X3) - S1*PPP
   *   ZZ3 = ZZ1*ZZ2*PP
   *   ZZZ3 = ZZZ1*ZZZ2*PPP
   *
   * All TPI threads must call this together
   */
  __device__ __forceinline__ void point_add(
      const bn_t& P_mod,
      // Result (XYZZ)
      bn_t& X3, bn_t& Y3, bn_t& ZZ3, bn_t& ZZZ3,
      // First operand (XYZZ)
      const bn_t& X1, const bn_t& Y1,
      const bn_t& ZZ1, const bn_t& ZZZ1,
      // Second operand (XYZZ)
      const bn_t& X2, const bn_t& Y2,
      const bn_t& ZZ2, const bn_t& ZZZ2
  ) {
    bn_t U1, U2, S1, S2, P, R, PP, PPP, Q, temp;

    // U1 = X1 * ZZ2
    field_mul(U1, X1, ZZ2, P_mod);

    // U2 = X2 * ZZ1
    field_mul(U2, X2, ZZ1, P_mod);

    // S1 = Y1 * ZZZ2
    field_mul(S1, Y1, ZZZ2, P_mod);

    // S2 = Y2 * ZZZ1
    field_mul(S2, Y2, ZZZ1, P_mod);

    // P = U2 - U1
    field_sub(P, U2, U1, P_mod);

    // R = S2 - S1
    field_sub(R, S2, S1, P_mod);

    // PP = P^2
    field_mul(PP, P, P, P_mod);

    // PPP = P * PP = P^3
    field_mul(PPP, P, PP, P_mod);

    // Q = U1 * PP
    field_mul(Q, U1, PP, P_mod);

    // X3 = R^2 - PPP - 2*Q
    field_mul(X3, R, R, P_mod);
    field_sub(X3, X3, PPP, P_mod);
    field_sub(X3, X3, Q, P_mod);
    field_sub(X3, X3, Q, P_mod);

    // Y3 = R*(Q - X3) - S1*PPP
    field_sub(temp, Q, X3, P_mod);
    field_mul(Y3, R, temp, P_mod);
    field_mul(temp, S1, PPP, P_mod);
    field_sub(Y3, Y3, temp, P_mod);

    // ZZ3 = ZZ1 * ZZ2 * PP
    field_mul(ZZ3, ZZ1, ZZ2, P_mod);
    field_mul(ZZ3, ZZ3, PP, P_mod);

    // ZZZ3 = ZZZ1 * ZZZ2 * PPP
    field_mul(ZZZ3, ZZZ1, ZZZ2, P_mod);
    field_mul(ZZZ3, ZZZ3, PPP, P_mod);
  }

  /**
   * Convert XYZZ to Jacobian coordinates
   *
   * Normalize to affine by computing x = X/ZZ, y = Y/ZZZ
   * Then output as Jacobian with Z=1: (x, y, 1)
   *
   * Uses CGBN modular inverse for division
   */
  __device__ __forceinline__ void xyzz_to_jacobian(
      const bn_t& P,
      jacobian_cgbn_t* result,
      const bn_t& X, const bn_t& Y,
      const bn_t& ZZ, const bn_t& ZZZ
  ) {
    bn_t x_norm, y_norm;
    bn_t zz_inv, zzz_inv;

    // Compute ZZ^(-1) mod P
    cgbn_modular_inverse(_env, zz_inv, ZZ, P);

    // Compute ZZZ^(-1) mod P
    cgbn_modular_inverse(_env, zzz_inv, ZZZ, P);

    // x = X * ZZ^(-1) mod P
    field_mul(x_norm, X, zz_inv, P);

    // y = Y * ZZZ^(-1) mod P
    field_mul(y_norm, Y, zzz_inv, P);

    // Store normalized affine coordinates
    cgbn_store(_env, (cgbn_mem_t<params::BITS>*)result->x, x_norm);
    cgbn_store(_env, (cgbn_mem_t<params::BITS>*)result->y, y_norm);

    // Store Z = 1 (affine coordinates in Jacobian form)
    bn_t one;
    cgbn_set_ui32(_env, one, 1);
    cgbn_store(_env, (cgbn_mem_t<params::BITS>*)result->z, one);

    // ALL threads write the same value (CGBN pattern)
    result->infinity = false;
  }
};  // End of mnt4_msm_t class

/**
 * MSM Kernel (CGBN-based) - Serial Accumulation MVP
 *
 * Strategy: Simple serial algorithm
 * 1. Single thread group (TPI=8 threads) processes all points
 * 2. For each point: compute k_i * P_i
 * 3. Accumulate: result = result + (k_i * P_i)
 * 4. Return final accumulated result
 *
 * This is slower than parallel approaches but simpler for MVP.
 * Future optimization: Parallel scalar muls + tree reduction.
 *
 * Threading: One thread group (mnt4_cgbn_params_t::TPI threads cooperate)
 */
template<class params>
__global__ void msm_naive_cgbn_kernel(
    cgbn_error_report_t *report,
    jacobian_cgbn_t *result_out,
    const affine_cgbn_t *points,
    const scalar_cgbn_t *scalars,
    uint32_t count
) {
    // Only thread group 0 does the work (serial accumulation)
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
    if (instance != 0)
        return;

    // Instantiate MNT4 MSM class
    mnt4_msm_t<params> msm(cgbn_report_monitor, report, instance);
    typename mnt4_msm_t<params>::bn_t P, px, py;
    typename mnt4_msm_t<params>::bn_t acc_x, acc_y, acc_zz, acc_zzz;
    typename mnt4_msm_t<params>::bn_t term_x, term_y, term_zz, term_zzz;

    // Load modulus
    cgbn_load(msm._env, P, (cgbn_mem_t<params::BITS>*)MNT4_298_P_DEVICE);

    // Accumulator (starts at infinity)
    __shared__ bool shared_acc_is_infinity;
    if (threadIdx.x == 0) {
        shared_acc_is_infinity = true;
    }
    __syncthreads();

    bool acc_is_infinity = shared_acc_is_infinity;

    // Process each point
    for (uint32_t i = 0; i < count; i++) {
        // Load point
        cgbn_load(msm._env, px, (cgbn_mem_t<params::BITS>*)points[i].x);
        cgbn_load(msm._env, py, (cgbn_mem_t<params::BITS>*)points[i].y);
        bool point_is_infinity = points[i].infinity;

        // Load scalar
        const uint32_t* scalar_limbs = scalars[i].limbs;

        // Compute k_i * P_i using scalar multiplication (INLINED to avoid stack overflow)
        bool term_is_infinity;

        // Handle point at infinity
        if (point_is_infinity) {
            term_is_infinity = true;
        } else {
            // Check if scalar is zero
            bool scalar_is_zero = true;
            for (int j = 0; j < 10; j++) {
                if (scalar_limbs[j] != 0) {
                    scalar_is_zero = false;
                    break;
                }
            }

            if (scalar_is_zero) {
                term_is_infinity = true;
            } else {
                // Find highest set bit in scalar
                int highest_bit = -1;
                for (int limb_idx = 9; limb_idx >= 0; limb_idx--) {
                    uint32_t limb = scalar_limbs[limb_idx];
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

                if (highest_bit < 0) {
                    term_is_infinity = true;
                } else {
                    // Initialize accumulator with point (for MSB)
                    cgbn_set(msm._env, term_x, px);
                    cgbn_set(msm._env, term_y, py);
                    cgbn_set_ui32(msm._env, term_zz, 1);
                    cgbn_set_ui32(msm._env, term_zzz, 1);

                    // Declare temps ONCE (not in loop!)
                    typename mnt4_msm_t<params>::bn_t dbl_x, dbl_y, dbl_zz, dbl_zzz;
                    typename mnt4_msm_t<params>::bn_t add_x, add_y, add_zz, add_zzz;

                    // Double-and-add from MSB-1 down to bit 0
                    for (int bit_pos = highest_bit - 1; bit_pos >= 0; bit_pos--) {
                        // Double: term = 2 * term
                        msm.point_double(P, dbl_x, dbl_y, dbl_zz, dbl_zzz,
                                        term_x, term_y, term_zz, term_zzz);
                        cgbn_set(msm._env, term_x, dbl_x);
                        cgbn_set(msm._env, term_y, dbl_y);
                        cgbn_set(msm._env, term_zz, dbl_zz);
                        cgbn_set(msm._env, term_zzz, dbl_zzz);

                        // Check if bit is set
                        int limb_idx = bit_pos / 32;
                        int bit_idx = bit_pos % 32;
                        bool bit_is_set = (scalar_limbs[limb_idx] & (1u << bit_idx)) != 0;

                        // Add: term = term + P (if bit is 1)
                        if (bit_is_set) {
                            msm.point_add_mixed(P, add_x, add_y, add_zz, add_zzz,
                                               term_x, term_y, term_zz, term_zzz,
                                               px, py);
                            cgbn_set(msm._env, term_x, add_x);
                            cgbn_set(msm._env, term_y, add_y);
                            cgbn_set(msm._env, term_zz, add_zz);
                            cgbn_set(msm._env, term_zzz, add_zzz);
                        }
                    }

                    term_is_infinity = false;
                }
            }
        }

        // Accumulate: acc = acc + term
        if (!term_is_infinity) {
            if (acc_is_infinity) {
                // First non-infinity term becomes the accumulator
                cgbn_set(msm._env, acc_x, term_x);
                cgbn_set(msm._env, acc_y, term_y);
                cgbn_set(msm._env, acc_zz, term_zz);
                cgbn_set(msm._env, acc_zzz, term_zzz);
                acc_is_infinity = false;
                if (threadIdx.x == 0) {
                    shared_acc_is_infinity = false;
                }
                __syncthreads();
            } else {
                // Add to accumulator: acc = acc + term
                // CRITICAL: Must handle three cases:
                // 1. acc == term (same point) → use doubling
                // 2. acc == -term (inverse points) → result is infinity
                // 3. acc != term and acc != -term → use standard addition
                typename mnt4_msm_t<params>::bn_t new_x, new_y, new_zz, new_zzz;

                // Check if points are equal or inverse by comparing affine coordinates:
                // acc represents (acc_x/acc_zz, acc_y/acc_zzz)
                // term represents (term_x/term_zz, term_y/term_zzz)
                //
                // X-coords equal if: acc_x * term_zz == term_x * acc_zz
                // Y-coords equal if: acc_y * term_zzz == term_y * acc_zzz
                // Y-coords are negatives if: acc_y * term_zzz + term_y * acc_zzz == 0 (mod P)
                //   equivalently: acc_y * term_zzz == P - (term_y * acc_zzz)
                typename mnt4_msm_t<params>::bn_t cross1, cross2, cross3, cross4;
                msm.field_mul(cross1, acc_x, term_zz, P);
                msm.field_mul(cross2, term_x, acc_zz, P);
                msm.field_mul(cross3, acc_y, term_zzz, P);
                msm.field_mul(cross4, term_y, acc_zzz, P);

                // Compare x-coordinates
                bool x_equal = (cgbn_compare(msm._env, cross1, cross2) == 0);

                // Compare y-coordinates (both same and opposite)
                bool y_equal = (cgbn_compare(msm._env, cross3, cross4) == 0);

                // Check if y-coords are negatives: cross3 + cross4 == 0 mod P
                // i.e., cross3 + cross4 == P (since both are already reduced mod P)
                typename mnt4_msm_t<params>::bn_t y_sum;
                cgbn_add(msm._env, y_sum, cross3, cross4);
                cgbn_rem(msm._env, y_sum, y_sum, P);
                bool y_opposite = cgbn_equals_ui32(msm._env, y_sum, 0);

                bool points_equal = x_equal && y_equal;
                bool points_inverse = x_equal && y_opposite && !y_equal;

                if (points_inverse) {
                    // Points are inverses: P + (-P) = O (point at infinity)
                    acc_is_infinity = true;
                    if (threadIdx.x == 0) {
                        shared_acc_is_infinity = true;
                    }
                    __syncthreads();
                } else if (points_equal) {
                    // Points are equal: use doubling formula (2*acc)
                    msm.point_double(P, new_x, new_y, new_zz, new_zzz,
                                    acc_x, acc_y, acc_zz, acc_zzz);
                    cgbn_set(msm._env, acc_x, new_x);
                    cgbn_set(msm._env, acc_y, new_y);
                    cgbn_set(msm._env, acc_zz, new_zz);
                    cgbn_set(msm._env, acc_zzz, new_zzz);
                } else {
                    // Points are different: use standard addition
                    msm.point_add(P, new_x, new_y, new_zz, new_zzz,
                                 acc_x, acc_y, acc_zz, acc_zzz,
                                 term_x, term_y, term_zz, term_zzz);

                    // After addition, check if result is infinity (ZZ == 0)
                    // This can happen if points were inverses but our check above missed it
                    // (e.g., due to different XYZZ representations of the same affine point)
                    bool result_is_zero = cgbn_equals_ui32(msm._env, new_zz, 0);
                    if (result_is_zero) {
                        acc_is_infinity = true;
                        if (threadIdx.x == 0) {
                            shared_acc_is_infinity = true;
                        }
                        __syncthreads();
                    } else {
                        cgbn_set(msm._env, acc_x, new_x);
                        cgbn_set(msm._env, acc_y, new_y);
                        cgbn_set(msm._env, acc_zz, new_zz);
                        cgbn_set(msm._env, acc_zzz, new_zzz);
                    }
                }
            }
        }
    }

    // Synchronize and store result
    __syncthreads();
    acc_is_infinity = shared_acc_is_infinity;

    if (acc_is_infinity) {
        // Result is infinity
        for (int i = 0; i < 10; i++) {
            result_out->x[i] = 0;
            result_out->y[i] = 0;
            result_out->z[i] = 0;
        }
        result_out->infinity = true;
    } else {
        // Convert XYZZ to Jacobian and store
        msm.xyzz_to_jacobian(P, result_out, acc_x, acc_y, acc_zz, acc_zzz);
    }
}

/**
 * FFI Entry Point: CGBN-based MSM for MNT4-298 G1
 *
 * This is a minimal viable implementation to prove CGBN integration works.
 * Future optimizations: Pippenger algorithm, proper scalar multiplication, etc.
 */
extern "C" int msm_mnt4_298_g1_cgbn(
    const void* points_ptr,
    const void* scalars_ptr,
    size_t count,
    void* result_ptr,
    size_t ffi_affine_sz,
    size_t ffi_scalar_sz
) {
    // Validate FFI layout (errors only)
    if (ffi_affine_sz != sizeof(affine_cgbn_t)) {
        fprintf(stderr, "[MNT4-298 CGBN] ERROR: Affine size mismatch: CUDA expects %zu, Rust provided %zu\n",
               sizeof(affine_cgbn_t), ffi_affine_sz);
        return MNT4_MSM_ERROR_AFFINE_LAYOUT;
    }

    if (ffi_scalar_sz != sizeof(scalar_cgbn_t)) {
        fprintf(stderr, "[MNT4-298 CGBN] ERROR: Scalar size mismatch: CUDA expects %zu, Rust provided %zu\n",
               sizeof(scalar_cgbn_t), ffi_scalar_sz);
        return MNT4_MSM_ERROR_SCALAR_LAYOUT;
    }

    // Handle edge cases
    if (count == 0) {
        jacobian_cgbn_t* result = static_cast<jacobian_cgbn_t*>(result_ptr);
        // Initialize all fields to zero
        memset(result->x, 0, sizeof(result->x));
        memset(result->y, 0, sizeof(result->y));
        memset(result->z, 0, sizeof(result->z));
        result->infinity = true;
        return MNT4_MSM_SUCCESS;
    }

    // Cast pointers
    const affine_cgbn_t* points = static_cast<const affine_cgbn_t*>(points_ptr);
    const scalar_cgbn_t* scalars = static_cast<const scalar_cgbn_t*>(scalars_ptr);
    jacobian_cgbn_t* result = static_cast<jacobian_cgbn_t*>(result_ptr);

    // Allocate GPU memory
    affine_cgbn_t* d_points = nullptr;
    scalar_cgbn_t* d_scalars = nullptr;
    jacobian_cgbn_t* d_result = nullptr;
    cgbn_error_report_t* d_report = nullptr;

    // Declare all variables before any goto statements (C++ restriction)
    cudaError_t err;
    int threads_per_block;
    int num_blocks;

    err = cudaMalloc(&d_points, sizeof(affine_cgbn_t) * count);
    if (err != cudaSuccess) return MNT4_MSM_ERROR_CUDA_RUNTIME;

    err = cudaMalloc(&d_scalars, sizeof(scalar_cgbn_t) * count);
    if (err != cudaSuccess) {
        cudaFree(d_points);
        return MNT4_MSM_ERROR_CUDA_RUNTIME;
    }

    err = cudaMalloc(&d_result, sizeof(jacobian_cgbn_t));
    if (err != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_scalars);
        return MNT4_MSM_ERROR_CUDA_RUNTIME;
    }

    err = cgbn_error_report_alloc(&d_report);
    if (err != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_scalars);
        cudaFree(d_result);
        return MNT4_MSM_ERROR_CUDA_RUNTIME;
    }

    // Copy inputs to GPU
    err = cudaMemcpy(d_points, points, sizeof(affine_cgbn_t) * count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_error;

    err = cudaMemcpy(d_scalars, scalars, sizeof(scalar_cgbn_t) * count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_error;

    // Launch kernel (serial accumulation: single thread group)
    threads_per_block = mnt4_cgbn_params_t::TPI;
    num_blocks = 1;

    msm_naive_cgbn_kernel<mnt4_cgbn_params_t><<<num_blocks, threads_per_block>>>(
        d_report, d_result, d_points, d_scalars, count
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[MNT4-298 CGBN] ERROR: Kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup_error;
    }

    // Check for CGBN errors
    if (cgbn_error_report_check(d_report)) {
        fprintf(stderr, "[MNT4-298 CGBN] ERROR: CGBN error detected\n");
        goto cleanup_error;
    }

    // Copy result back
    err = cudaMemcpy(result, d_result, sizeof(jacobian_cgbn_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[MNT4-298 CGBN] ERROR: Failed to copy result: %s\n", cudaGetErrorString(err));
        goto cleanup_error;
    }

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_result);
    cgbn_error_report_free(d_report);

    return MNT4_MSM_SUCCESS;

cleanup_error:
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_result);
    cgbn_error_report_free(d_report);
    return MNT4_MSM_ERROR_CUDA_RUNTIME;
}

// ============================================================================
// PIPPENGER MSM IMPLEMENTATION
// ============================================================================
//
// Pippenger's bucket method for MSM with CGBN field arithmetic.
// Algorithm phases:
// 1. Breakdown: Partition scalars into signed wbits-wide digits
// 2. Histogram: Count points per bucket for memory allocation
// 3. Sort: Group point indices by bucket assignment
// 4. Accumulate: Add points to buckets (parallel, one CGBN instance per bucket)
// 5. Integrate: Horner-like bucket sum within each window
// 6. Combine: Merge window results with doublings
//
// Complexity: O(n + nwins * 2^(wbits-1)) vs O(n * scalar_bits) for serial
// ============================================================================

// Pippenger configuration
#define PIPPENGER_MIN_WBITS 8
#define PIPPENGER_MAX_WBITS 16
#define PIPPENGER_SCALAR_BITS 298

// Helper: Compute optimal window size based on point count
__host__ __device__ __forceinline__
uint32_t compute_optimal_wbits(uint32_t n) {
    if (n == 0) return PIPPENGER_MIN_WBITS;

    // wbits ~ log2(n) - 1, clamped to [MIN_WBITS, MAX_WBITS]
    uint32_t log2_n = 0;
    uint32_t temp = n;
    while (temp > 1) { temp >>= 1; log2_n++; }

    int32_t wbits = (int32_t)log2_n - 1;
    if (wbits < PIPPENGER_MIN_WBITS) wbits = PIPPENGER_MIN_WBITS;
    if (wbits > PIPPENGER_MAX_WBITS) wbits = PIPPENGER_MAX_WBITS;

    return (uint32_t)wbits;
}

// Helper: Compute number of windows for given scalar bits and window size
__host__ __device__ __forceinline__
uint32_t compute_nwins(uint32_t scalar_bits, uint32_t wbits) {
    return (scalar_bits + wbits - 1) / wbits;
}

// XYZZ bucket with infinity flag for Pippenger
typedef struct {
    uint32_t x[10];
    uint32_t y[10];
    uint32_t zz[10];
    uint32_t zzz[10];
    bool is_infinity;
    uint8_t _padding[7];  // Align to 8 bytes
} __align__(8) bucket_xyzz_t;

// ============================================================================
// KERNEL 1: Scalar Breakdown (extract signed digits)
// ============================================================================
//
// Each thread processes one scalar, extracting nwins digits of wbits width.
// Uses Booth encoding to produce signed digits in range [-2^(wbits-1), 2^(wbits-1)]
// which halves the number of buckets needed.
//
// Output format per digit (32-bit):
//   bits [0:15]  = bucket index (unsigned, 0 means skip)
//   bit  [31]    = sign (0=positive, 1=negative)
//
__global__ void breakdown_scalars_mnt4_298(
    uint32_t* digits,           // Output: [n * nwins] packed digits
    const scalar_cgbn_t* scalars,
    uint32_t n,
    uint32_t nwins,
    uint32_t wbits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint32_t* scalar = scalars[idx].limbs;
    const uint32_t wmask = (1u << wbits) - 1;
    const uint32_t half_buckets = 1u << (wbits - 1);

    // Track carry for Booth encoding across windows
    uint32_t carry = 0;

    for (uint32_t win = 0; win < nwins; win++) {
        uint32_t bit_offset = win * wbits;
        uint32_t limb_idx = bit_offset / 32;
        uint32_t bit_idx = bit_offset % 32;

        // Extract wbits from scalar (may span two limbs)
        uint64_t window_val = 0;
        if (limb_idx < 10) {
            window_val = scalar[limb_idx];
        }
        if (limb_idx + 1 < 10 && bit_idx + wbits > 32) {
            window_val |= ((uint64_t)scalar[limb_idx + 1]) << 32;
        }
        window_val = (window_val >> bit_idx) & wmask;

        // Add carry from previous window's Booth encoding
        window_val += carry;
        carry = 0;

        // Booth encoding: if value > half_buckets, subtract 2^wbits and carry 1
        // Note: we use > (not >=) because half_buckets can be represented directly
        // For wbits=8: values 0-128 stay positive, 129-255 become negative
        uint32_t sign = 0;
        if (window_val > half_buckets) {
            // Negative representation: bucket = 2^wbits - window_val
            window_val = (1u << wbits) - window_val;
            sign = 1;
            carry = 1;  // Carry to next window
        }

        // Store packed digit: bucket_id in low 16 bits, sign in bit 31
        // Note: bucket_id 0 means identity (skip this point for this window)
        uint32_t packed = (uint32_t)window_val | (sign << 31);
        digits[win * n + idx] = packed;
    }
}

// ============================================================================
// KERNEL 2: Histogram (count points per bucket)
// ============================================================================
//
// Count how many points go to each bucket for memory allocation and sorting.
// Uses atomicAdd for thread-safe counting.
//
__global__ void histogram_buckets_mnt4_298(
    uint32_t* histogram,        // Output: [nwins * num_buckets] counts
    const uint32_t* digits,     // Input: [n * nwins] packed digits
    uint32_t n,
    uint32_t nwins,
    uint32_t wbits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // +1 for bucket 2^(wbits-1) which can occur in Booth encoding
    uint32_t num_buckets = (1u << (wbits - 1)) + 1;

    for (uint32_t win = 0; win < nwins; win++) {
        uint32_t packed = digits[win * n + idx];
        // Bucket_id is in low 16 bits (sign is in bit 31)
        uint32_t bucket_id = packed & 0xFFFF;

        // Skip bucket 0 (identity, no contribution)
        if (bucket_id > 0) {
            uint32_t hist_idx = win * num_buckets + bucket_id;
            atomicAdd(&histogram[hist_idx], 1);
        }
    }
}

// ============================================================================
// KERNEL 3: Prefix Sum (compute bucket offsets from histogram)
// ============================================================================
//
// Convert histogram counts to cumulative offsets for scatter phase.
// Simple serial prefix sum per window (efficient for moderate bucket counts).
//
__global__ void prefix_sum_histogram_mnt4_298(
    uint32_t* offsets,          // Output: [nwins * num_buckets] offsets
    const uint32_t* histogram,  // Input: [nwins * num_buckets] counts
    uint32_t nwins,
    uint32_t num_buckets
) {
    uint32_t win = blockIdx.x * blockDim.x + threadIdx.x;
    if (win >= nwins) return;

    uint32_t base = win * num_buckets;
    uint32_t sum = 0;

    for (uint32_t b = 0; b < num_buckets; b++) {
        uint32_t count = histogram[base + b];
        offsets[base + b] = sum;
        sum += count;
    }
}

// ============================================================================
// KERNEL 4: Scatter (sort point indices into bucket order)
// ============================================================================
//
// Place point indices into their bucket positions using atomicAdd for
// thread-safe scatter. Each point is placed once per window.
//
__global__ void scatter_to_buckets_mnt4_298(
    uint32_t* sorted_indices,   // Output: [nwins * n] sorted point indices
    uint32_t* bucket_counters,  // Working: [nwins * num_buckets] current counts
    const uint32_t* offsets,    // Input: [nwins * num_buckets] bucket offsets
    const uint32_t* digits,     // Input: [n * nwins] packed digits
    uint32_t n,
    uint32_t nwins,
    uint32_t wbits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // +1 for bucket 2^(wbits-1) which can occur in Booth encoding
    uint32_t num_buckets = (1u << (wbits - 1)) + 1;

    for (uint32_t win = 0; win < nwins; win++) {
        uint32_t packed = digits[win * n + idx];
        // Bucket_id is in low 16 bits, sign is in bit 31
        uint32_t bucket_id = packed & 0xFFFF;
        uint32_t sign = (packed >> 31) & 1;

        // Skip bucket 0 (identity)
        if (bucket_id > 0) {
            uint32_t hist_idx = win * num_buckets + bucket_id;
            uint32_t pos = offsets[hist_idx] + atomicAdd(&bucket_counters[hist_idx], 1);

            // Pack point index with sign: idx | (sign << 31)
            sorted_indices[win * n + pos] = idx | (sign << 31);
        }
    }
}

// ============================================================================
// KERNEL 5: Bucket Accumulation (parallel bucket addition with CGBN)
// ============================================================================
//
// Each CGBN instance (TPI=8 threads) processes one bucket.
// Iterates through all points assigned to the bucket and accumulates them.
//
template<class params>
__global__ void accumulate_buckets_mnt4_298(
    cgbn_error_report_t* report,
    bucket_xyzz_t* buckets,         // Output: [nwins * num_buckets]
    const affine_cgbn_t* points,    // Input: [n] points
    const uint32_t* sorted_indices, // Input: [nwins * n] sorted indices
    const uint32_t* offsets,        // Input: [nwins * num_buckets] offsets
    const uint32_t* histogram,      // Input: [nwins * num_buckets] counts
    uint32_t n,
    uint32_t nwins,
    uint32_t wbits
) {
    // Each CGBN instance handles one bucket
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    // +1 for bucket 2^(wbits-1) which can occur in Booth encoding
    uint32_t num_buckets = (1u << (wbits - 1)) + 1;
    uint32_t total_buckets = nwins * num_buckets;

    if ((uint32_t)instance >= total_buckets) return;

    uint32_t win = instance / num_buckets;
    uint32_t bucket_id = instance % num_buckets;

    // Skip bucket 0 (identity bucket, always empty by design)
    if (bucket_id == 0) {
        buckets[instance].is_infinity = true;
        return;
    }

    // Get bucket range from histogram
    uint32_t hist_idx = win * num_buckets + bucket_id;
    uint32_t start = offsets[hist_idx];
    uint32_t count = histogram[hist_idx];

    // Initialize CGBN
    mnt4_msm_t<params> msm(cgbn_report_monitor, report, instance);
    typename mnt4_msm_t<params>::bn_t P, px, py;
    typename mnt4_msm_t<params>::bn_t acc_x, acc_y, acc_zz, acc_zzz;
    typename mnt4_msm_t<params>::bn_t tmp_x, tmp_y, tmp_zz, tmp_zzz;

    // Load modulus
    cgbn_load(msm._env, P, (cgbn_mem_t<params::BITS>*)MNT4_298_P_DEVICE);

    bool acc_is_infinity = true;

    // Accumulate all points in this bucket
    for (uint32_t i = 0; i < count; i++) {
        uint32_t packed_idx = sorted_indices[win * n + start + i];
        uint32_t point_idx = packed_idx & 0x7FFFFFFF;
        bool negate = (packed_idx >> 31) != 0;

        // Load point
        cgbn_load(msm._env, px, (cgbn_mem_t<params::BITS>*)points[point_idx].x);
        cgbn_load(msm._env, py, (cgbn_mem_t<params::BITS>*)points[point_idx].y);

        // Apply negation if needed (negate y coordinate)
        if (negate) {
            // py = P - py
            typename mnt4_msm_t<params>::bn_t neg_y;
            cgbn_sub(msm._env, neg_y, P, py);
            cgbn_set(msm._env, py, neg_y);
        }

        // Skip points at infinity
        if (points[point_idx].infinity) continue;

        if (acc_is_infinity) {
            // First point: initialize accumulator as XYZZ with Z=1
            cgbn_set(msm._env, acc_x, px);
            cgbn_set(msm._env, acc_y, py);
            cgbn_set_ui32(msm._env, acc_zz, 1);
            cgbn_set_ui32(msm._env, acc_zzz, 1);
            acc_is_infinity = false;
        } else {
            // Add point using mixed addition (XYZZ + Affine)
            msm.point_add_mixed(P, tmp_x, tmp_y, tmp_zz, tmp_zzz,
                               acc_x, acc_y, acc_zz, acc_zzz,
                               px, py);

            // Check if result is zero (happens when adding inverse points)
            bool result_is_zero = cgbn_equals_ui32(msm._env, tmp_zz, 0);
            if (result_is_zero) {
                acc_is_infinity = true;
            } else {
                cgbn_set(msm._env, acc_x, tmp_x);
                cgbn_set(msm._env, acc_y, tmp_y);
                cgbn_set(msm._env, acc_zz, tmp_zz);
                cgbn_set(msm._env, acc_zzz, tmp_zzz);
            }
        }
    }

    // Store bucket result
    buckets[instance].is_infinity = acc_is_infinity;
    if (!acc_is_infinity) {
        cgbn_store(msm._env, (cgbn_mem_t<params::BITS>*)buckets[instance].x, acc_x);
        cgbn_store(msm._env, (cgbn_mem_t<params::BITS>*)buckets[instance].y, acc_y);
        cgbn_store(msm._env, (cgbn_mem_t<params::BITS>*)buckets[instance].zz, acc_zz);
        cgbn_store(msm._env, (cgbn_mem_t<params::BITS>*)buckets[instance].zzz, acc_zzz);
    }
}

// ============================================================================
// KERNEL 6: Bucket Integration (Horner-like sum within each window)
// ============================================================================
//
// For each window, compute: sum = Σ (b * bucket[b]) for b = 1..num_buckets-1
// Using Horner method: acc = bucket[N-1], sum = 0
//   for b = N-1 down to 1: sum += acc; acc += bucket[b-1]
//   return sum
//
template<class params>
__global__ void integrate_buckets_mnt4_298(
    cgbn_error_report_t* report,
    bucket_xyzz_t* window_sums,     // Output: [nwins]
    const bucket_xyzz_t* buckets,   // Input: [nwins * num_buckets]
    uint32_t nwins,
    uint32_t wbits
) {
    // Each CGBN instance handles one window
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    if ((uint32_t)instance >= nwins) return;

    uint32_t win = instance;
    // +1 for bucket 2^(wbits-1) which can occur in Booth encoding
    uint32_t num_buckets = (1u << (wbits - 1)) + 1;
    uint32_t bucket_base = win * num_buckets;

    // Initialize CGBN
    mnt4_msm_t<params> msm(cgbn_report_monitor, report, instance);
    typename mnt4_msm_t<params>::bn_t P;
    typename mnt4_msm_t<params>::bn_t acc_x, acc_y, acc_zz, acc_zzz;
    typename mnt4_msm_t<params>::bn_t sum_x, sum_y, sum_zz, sum_zzz;
    typename mnt4_msm_t<params>::bn_t tmp_x, tmp_y, tmp_zz, tmp_zzz;
    typename mnt4_msm_t<params>::bn_t bkt_x, bkt_y, bkt_zz, bkt_zzz;

    // Load modulus
    cgbn_load(msm._env, P, (cgbn_mem_t<params::BITS>*)MNT4_298_P_DEVICE);

    bool acc_is_infinity = true;
    bool sum_is_infinity = true;

    // Horner-like integration: start from highest bucket
    // Formula: S = Σ(b * bucket[b]) for b = 1 to num_buckets-1
    // Algorithm: for b = num_buckets-1 down to 1:
    //   acc += bucket[b]
    //   sum += acc
    for (int32_t b = num_buckets - 1; b >= 1; b--) {
        uint32_t bkt_idx = bucket_base + b;

        // First: acc += bucket[b]
        if (!buckets[bkt_idx].is_infinity) {
            cgbn_load(msm._env, bkt_x, (cgbn_mem_t<params::BITS>*)buckets[bkt_idx].x);
            cgbn_load(msm._env, bkt_y, (cgbn_mem_t<params::BITS>*)buckets[bkt_idx].y);
            cgbn_load(msm._env, bkt_zz, (cgbn_mem_t<params::BITS>*)buckets[bkt_idx].zz);
            cgbn_load(msm._env, bkt_zzz, (cgbn_mem_t<params::BITS>*)buckets[bkt_idx].zzz);

            if (acc_is_infinity) {
                cgbn_set(msm._env, acc_x, bkt_x);
                cgbn_set(msm._env, acc_y, bkt_y);
                cgbn_set(msm._env, acc_zz, bkt_zz);
                cgbn_set(msm._env, acc_zzz, bkt_zzz);
                acc_is_infinity = false;
            } else {
                // Check if acc and bucket[b] are the same point (need doubling)
                typename mnt4_msm_t<params>::bn_t lhs, rhs;
                msm.field_mul(lhs, acc_x, bkt_zz, P);
                msm.field_mul(rhs, bkt_x, acc_zz, P);
                bool same_x = cgbn_compare(msm._env, lhs, rhs) == 0;

                msm.field_mul(lhs, acc_y, bkt_zzz, P);
                msm.field_mul(rhs, bkt_y, acc_zzz, P);
                bool same_y = cgbn_compare(msm._env, lhs, rhs) == 0;

                if (same_x && same_y) {
                    // Same point: use doubling
                    msm.point_double(P, tmp_x, tmp_y, tmp_zz, tmp_zzz,
                                    acc_x, acc_y, acc_zz, acc_zzz);
                    cgbn_set(msm._env, acc_x, tmp_x);
                    cgbn_set(msm._env, acc_y, tmp_y);
                    cgbn_set(msm._env, acc_zz, tmp_zz);
                    cgbn_set(msm._env, acc_zzz, tmp_zzz);
                } else if (same_x) {
                    // Inverse points -> result is infinity
                    acc_is_infinity = true;
                } else {
                    // Different points: use standard addition
                    msm.point_add(P, tmp_x, tmp_y, tmp_zz, tmp_zzz,
                                 acc_x, acc_y, acc_zz, acc_zzz,
                                 bkt_x, bkt_y, bkt_zz, bkt_zzz);

                    bool result_is_zero = cgbn_equals_ui32(msm._env, tmp_zz, 0);
                    if (result_is_zero) {
                        acc_is_infinity = true;
                    } else {
                        cgbn_set(msm._env, acc_x, tmp_x);
                        cgbn_set(msm._env, acc_y, tmp_y);
                        cgbn_set(msm._env, acc_zz, tmp_zz);
                        cgbn_set(msm._env, acc_zzz, tmp_zzz);
                    }
                }
            }
        }

        // Second: sum += acc
        if (!acc_is_infinity) {
            if (sum_is_infinity) {
                cgbn_set(msm._env, sum_x, acc_x);
                cgbn_set(msm._env, sum_y, acc_y);
                cgbn_set(msm._env, sum_zz, acc_zz);
                cgbn_set(msm._env, sum_zzz, acc_zzz);
                sum_is_infinity = false;
            } else {
                // Check if sum and acc are the same point (need doubling)
                // Two XYZZ points (X1,Y1,ZZ1,ZZZ1) and (X2,Y2,ZZ2,ZZZ2) are equal if:
                // X1*ZZ2 = X2*ZZ1 and Y1*ZZZ2 = Y2*ZZZ1
                typename mnt4_msm_t<params>::bn_t lhs, rhs;
                msm.field_mul(lhs, sum_x, acc_zz, P);
                msm.field_mul(rhs, acc_x, sum_zz, P);
                bool same_x = cgbn_compare(msm._env, lhs, rhs) == 0;

                msm.field_mul(lhs, sum_y, acc_zzz, P);
                msm.field_mul(rhs, acc_y, sum_zzz, P);
                bool same_y = cgbn_compare(msm._env, lhs, rhs) == 0;

                if (same_x && same_y) {
                    // Same point: use doubling
                    msm.point_double(P, tmp_x, tmp_y, tmp_zz, tmp_zzz,
                                    sum_x, sum_y, sum_zz, sum_zzz);
                    cgbn_set(msm._env, sum_x, tmp_x);
                    cgbn_set(msm._env, sum_y, tmp_y);
                    cgbn_set(msm._env, sum_zz, tmp_zz);
                    cgbn_set(msm._env, sum_zzz, tmp_zzz);
                } else if (same_x) {
                    // Same x but different y means inverse points -> result is infinity
                    sum_is_infinity = true;
                } else {
                    // Different points: use standard addition
                    msm.point_add(P, tmp_x, tmp_y, tmp_zz, tmp_zzz,
                                 sum_x, sum_y, sum_zz, sum_zzz,
                                 acc_x, acc_y, acc_zz, acc_zzz);

                    bool result_is_zero = cgbn_equals_ui32(msm._env, tmp_zz, 0);
                    if (result_is_zero) {
                        sum_is_infinity = true;
                    } else {
                        cgbn_set(msm._env, sum_x, tmp_x);
                        cgbn_set(msm._env, sum_y, tmp_y);
                        cgbn_set(msm._env, sum_zz, tmp_zz);
                        cgbn_set(msm._env, sum_zzz, tmp_zzz);
                    }
                }
            }
        }
    }

    // NOTE: No final addition needed. The Horner loop already gives the correct result.
    // The formula is: S = Σ(b * bucket[b]) for b = 1 to num_buckets-1
    // After the loop, sum contains this result.

    // Store window sum
    window_sums[win].is_infinity = sum_is_infinity;
    if (!sum_is_infinity) {
        cgbn_store(msm._env, (cgbn_mem_t<params::BITS>*)window_sums[win].x, sum_x);
        cgbn_store(msm._env, (cgbn_mem_t<params::BITS>*)window_sums[win].y, sum_y);
        cgbn_store(msm._env, (cgbn_mem_t<params::BITS>*)window_sums[win].zz, sum_zz);
        cgbn_store(msm._env, (cgbn_mem_t<params::BITS>*)window_sums[win].zzz, sum_zzz);
    }
}

// ============================================================================
// KERNEL 7: Window Combination (merge window sums with doublings)
// ============================================================================
//
// Combine window sums: result = Σ window_sum[w] * 2^(w*wbits)
// Using: result = window_sum[nwins-1]
//   for w = nwins-2 down to 0:
//     result = 2^wbits * result + window_sum[w]
//
template<class params>
__global__ void combine_windows_mnt4_298(
    cgbn_error_report_t* report,
    jacobian_cgbn_t* result_out,
    const bucket_xyzz_t* window_sums,
    uint32_t nwins,
    uint32_t wbits
) {
    // Single CGBN instance does the final combination
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
    if (instance != 0) return;

    mnt4_msm_t<params> msm(cgbn_report_monitor, report, instance);
    typename mnt4_msm_t<params>::bn_t P;
    typename mnt4_msm_t<params>::bn_t res_x, res_y, res_zz, res_zzz;
    typename mnt4_msm_t<params>::bn_t tmp_x, tmp_y, tmp_zz, tmp_zzz;
    typename mnt4_msm_t<params>::bn_t win_x, win_y, win_zz, win_zzz;

    cgbn_load(msm._env, P, (cgbn_mem_t<params::BITS>*)MNT4_298_P_DEVICE);

    bool result_is_infinity = true;

    // Start with highest window
    for (int32_t w = nwins - 1; w >= 0; w--) {
        // Double result wbits times (if not first iteration)
        if (w < (int32_t)nwins - 1 && !result_is_infinity) {
            for (uint32_t d = 0; d < wbits; d++) {
                msm.point_double(P, tmp_x, tmp_y, tmp_zz, tmp_zzz,
                                res_x, res_y, res_zz, res_zzz);
                cgbn_set(msm._env, res_x, tmp_x);
                cgbn_set(msm._env, res_y, tmp_y);
                cgbn_set(msm._env, res_zz, tmp_zz);
                cgbn_set(msm._env, res_zzz, tmp_zzz);
            }
        }

        // Add window sum
        if (!window_sums[w].is_infinity) {
            cgbn_load(msm._env, win_x, (cgbn_mem_t<params::BITS>*)window_sums[w].x);
            cgbn_load(msm._env, win_y, (cgbn_mem_t<params::BITS>*)window_sums[w].y);
            cgbn_load(msm._env, win_zz, (cgbn_mem_t<params::BITS>*)window_sums[w].zz);
            cgbn_load(msm._env, win_zzz, (cgbn_mem_t<params::BITS>*)window_sums[w].zzz);

            if (result_is_infinity) {
                cgbn_set(msm._env, res_x, win_x);
                cgbn_set(msm._env, res_y, win_y);
                cgbn_set(msm._env, res_zz, win_zz);
                cgbn_set(msm._env, res_zzz, win_zzz);
                result_is_infinity = false;
            } else {
                // Check if result and window_sum are the same point (need doubling)
                typename mnt4_msm_t<params>::bn_t lhs, rhs;
                msm.field_mul(lhs, res_x, win_zz, P);
                msm.field_mul(rhs, win_x, res_zz, P);
                bool same_x = cgbn_compare(msm._env, lhs, rhs) == 0;

                msm.field_mul(lhs, res_y, win_zzz, P);
                msm.field_mul(rhs, win_y, res_zzz, P);
                bool same_y = cgbn_compare(msm._env, lhs, rhs) == 0;

                if (same_x && same_y) {
                    // Same point: use doubling
                    msm.point_double(P, tmp_x, tmp_y, tmp_zz, tmp_zzz,
                                    res_x, res_y, res_zz, res_zzz);
                    cgbn_set(msm._env, res_x, tmp_x);
                    cgbn_set(msm._env, res_y, tmp_y);
                    cgbn_set(msm._env, res_zz, tmp_zz);
                    cgbn_set(msm._env, res_zzz, tmp_zzz);
                } else if (same_x) {
                    // Inverse points -> result is infinity
                    result_is_infinity = true;
                } else {
                    // Different points: use standard addition
                    msm.point_add(P, tmp_x, tmp_y, tmp_zz, tmp_zzz,
                                 res_x, res_y, res_zz, res_zzz,
                                 win_x, win_y, win_zz, win_zzz);

                    bool is_zero = cgbn_equals_ui32(msm._env, tmp_zz, 0);
                    if (is_zero) {
                        result_is_infinity = true;
                    } else {
                        cgbn_set(msm._env, res_x, tmp_x);
                        cgbn_set(msm._env, res_y, tmp_y);
                        cgbn_set(msm._env, res_zz, tmp_zz);
                        cgbn_set(msm._env, res_zzz, tmp_zzz);
                    }
                }
            }
        }
    }

    // Convert XYZZ to Jacobian
    if (result_is_infinity) {
        for (int i = 0; i < 10; i++) {
            result_out->x[i] = 0;
            result_out->y[i] = 0;
            result_out->z[i] = 0;
        }
        result_out->infinity = true;
    } else {
        msm.xyzz_to_jacobian(P, result_out, res_x, res_y, res_zz, res_zzz);
    }
}

// ============================================================================
// FFI Entry Point: Pippenger MSM for MNT4-298 G1
// ============================================================================
extern "C" int msm_mnt4_298_g1_cgbn_pippenger(
    const void* points_ptr,
    const void* scalars_ptr,
    size_t count,
    void* result_ptr,
    size_t ffi_affine_sz,
    size_t ffi_scalar_sz
) {
    // Validate FFI layout
    if (ffi_affine_sz != sizeof(affine_cgbn_t)) {
        fprintf(stderr, "[MNT4-298 Pippenger] ERROR: Affine size mismatch\n");
        return MNT4_MSM_ERROR_AFFINE_LAYOUT;
    }
    if (ffi_scalar_sz != sizeof(scalar_cgbn_t)) {
        fprintf(stderr, "[MNT4-298 Pippenger] ERROR: Scalar size mismatch\n");
        return MNT4_MSM_ERROR_SCALAR_LAYOUT;
    }

    // Handle edge case
    if (count == 0) {
        jacobian_cgbn_t* result = static_cast<jacobian_cgbn_t*>(result_ptr);
        memset(result, 0, sizeof(jacobian_cgbn_t));
        result->infinity = true;
        return MNT4_MSM_SUCCESS;
    }

    // For very small inputs, use serial algorithm (less overhead)
    if (count < 64) {
        return msm_mnt4_298_g1_cgbn(points_ptr, scalars_ptr, count,
                                    result_ptr, ffi_affine_sz, ffi_scalar_sz);
    }

    const affine_cgbn_t* points = static_cast<const affine_cgbn_t*>(points_ptr);
    const scalar_cgbn_t* scalars = static_cast<const scalar_cgbn_t*>(scalars_ptr);
    jacobian_cgbn_t* result = static_cast<jacobian_cgbn_t*>(result_ptr);

    // Compute optimal parameters
    uint32_t n = (uint32_t)count;
    uint32_t wbits = compute_optimal_wbits(n);
    uint32_t nwins = compute_nwins(PIPPENGER_SCALAR_BITS, wbits);
    // +1 for bucket 2^(wbits-1) which can occur in Booth encoding
    uint32_t num_buckets = (1u << (wbits - 1)) + 1;
    uint32_t total_buckets = nwins * num_buckets;

    // Allocate device memory
    cudaError_t err;
    affine_cgbn_t* d_points = nullptr;
    scalar_cgbn_t* d_scalars = nullptr;
    uint32_t* d_digits = nullptr;
    uint32_t* d_histogram = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_bucket_counters = nullptr;
    uint32_t* d_sorted_indices = nullptr;
    bucket_xyzz_t* d_buckets = nullptr;
    bucket_xyzz_t* d_window_sums = nullptr;
    jacobian_cgbn_t* d_result = nullptr;
    cgbn_error_report_t* d_report = nullptr;

    // Allocation
    err = cudaMalloc(&d_points, sizeof(affine_cgbn_t) * n);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_scalars, sizeof(scalar_cgbn_t) * n);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_digits, sizeof(uint32_t) * nwins * n);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_histogram, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_offsets, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_bucket_counters, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_sorted_indices, sizeof(uint32_t) * nwins * n);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_buckets, sizeof(bucket_xyzz_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_window_sums, sizeof(bucket_xyzz_t) * nwins);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMalloc(&d_result, sizeof(jacobian_cgbn_t));
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cgbn_error_report_alloc(&d_report);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    // Copy inputs to device
    err = cudaMemcpy(d_points, points, sizeof(affine_cgbn_t) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMemcpy(d_scalars, scalars, sizeof(scalar_cgbn_t) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    // Zero histograms and counters
    err = cudaMemset(d_histogram, 0, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    err = cudaMemset(d_bucket_counters, 0, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    {
        // Kernel launches
        int threads = 256;
        int blocks_n = (n + threads - 1) / threads;
        int blocks_nwins = (nwins + threads - 1) / threads;

        // 1. Breakdown scalars
        breakdown_scalars_mnt4_298<<<blocks_n, threads>>>(
            d_digits, d_scalars, n, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        // 2. Build histogram
        histogram_buckets_mnt4_298<<<blocks_n, threads>>>(
            d_histogram, d_digits, n, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        // 3. Prefix sum for offsets
        prefix_sum_histogram_mnt4_298<<<blocks_nwins, threads>>>(
            d_offsets, d_histogram, nwins, num_buckets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        // 4. Scatter to buckets
        scatter_to_buckets_mnt4_298<<<blocks_n, threads>>>(
            d_sorted_indices, d_bucket_counters, d_offsets, d_digits,
            n, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        // 5. Accumulate buckets (CGBN kernel)
        int tpi = mnt4_cgbn_params_t::TPI;
        int threads_accumulate = 32 * tpi;  // 32 CGBN instances per block
        int instances_needed = total_buckets;
        int blocks_accumulate = (instances_needed * tpi + threads_accumulate - 1) / threads_accumulate;

        accumulate_buckets_mnt4_298<mnt4_cgbn_params_t><<<blocks_accumulate, threads_accumulate>>>(
            d_report, d_buckets, d_points, d_sorted_indices, d_offsets, d_histogram,
            n, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        // 6. Integrate buckets (CGBN kernel)
        int instances_integrate = nwins;
        int blocks_integrate = (instances_integrate * tpi + threads_accumulate - 1) / threads_accumulate;

        integrate_buckets_mnt4_298<mnt4_cgbn_params_t><<<blocks_integrate, threads_accumulate>>>(
            d_report, d_window_sums, d_buckets, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        // 7. Combine windows (single CGBN instance)
        combine_windows_mnt4_298<mnt4_cgbn_params_t><<<1, tpi>>>(
            d_report, d_result, d_window_sums, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pippenger_cleanup_error;
    }

    // Check CGBN errors
    if (cgbn_error_report_check(d_report)) {
        fprintf(stderr, "[MNT4-298 Pippenger] ERROR: CGBN error detected\n");
        goto pippenger_cleanup_error;
    }

    // Copy result back
    err = cudaMemcpy(result, d_result, sizeof(jacobian_cgbn_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto pippenger_cleanup_error;

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_digits);
    cudaFree(d_histogram);
    cudaFree(d_offsets);
    cudaFree(d_bucket_counters);
    cudaFree(d_sorted_indices);
    cudaFree(d_buckets);
    cudaFree(d_window_sums);
    cudaFree(d_result);
    cgbn_error_report_free(d_report);

    return MNT4_MSM_SUCCESS;

pippenger_cleanup_error:
    if (d_points) cudaFree(d_points);
    if (d_scalars) cudaFree(d_scalars);
    if (d_digits) cudaFree(d_digits);
    if (d_histogram) cudaFree(d_histogram);
    if (d_offsets) cudaFree(d_offsets);
    if (d_bucket_counters) cudaFree(d_bucket_counters);
    if (d_sorted_indices) cudaFree(d_sorted_indices);
    if (d_buckets) cudaFree(d_buckets);
    if (d_window_sums) cudaFree(d_window_sums);
    if (d_result) cudaFree(d_result);
    if (d_report) cgbn_error_report_free(d_report);
    return MNT4_MSM_ERROR_CUDA_RUNTIME;
}
