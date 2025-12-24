// BW6-761 CGBN-Based MSM Implementation
// Uses NVIDIA CGBN library for cooperative-group field arithmetic
// Avoids stack overflow by distributing field element limbs across TPI threads

#include <stddef.h>
#include <cuda_runtime.h>

// IMPORTANT: Include gmp.h BEFORE cgbn.h to avoid cgbn_cpu.h stub
// CGBN checks for __GMP_H__ and uses cgbn_mpz.h instead of cgbn_cpu.h
#include <gmp.h>
#include <cgbn/cgbn.h>

// Error codes
#define BW6_MSM_SUCCESS 0
#define BW6_MSM_ERROR_AFFINE_LAYOUT -1
#define BW6_MSM_ERROR_SCALAR_LAYOUT -2
#define BW6_MSM_ERROR_CUDA_RUNTIME -3
#define BW6_MSM_ERROR_TIMEOUT -4

// CGBN parameters class (following CGBN sample pattern)
class bw6_cgbn_params_t {
public:
  static const uint32_t TPB = 0;            // Get TPB from blockDim.x
  static const uint32_t MAX_ROTATION = 4;   // Good default value
  static const uint32_t SHM_LIMIT = 0;      // No shared memory
  static const bool CONSTANT_TIME = false;  // Not available yet
  static const uint32_t TPI = 8;            // 8 threads cooperate per big number
  static const uint32_t BITS = 768;         // Round up from 761 bits
};

// BW6-761 base field modulus (verified against arkworks Fq::MODULUS)
// Note: This is BigInt<12> converted to u32[24] in little-endian order
__device__ __constant__ uint32_t BW6_761_P_DEVICE[24] = {
    0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68,
    0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
    0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a,
    0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824
};

// BW6-761 scalar field order (Fr, 377 bits)
__device__ __constant__ uint32_t BW6_761_R_DEVICE[12] = {
    0x00000001, 0x8508c000, 0x30000000, 0x170b5d44,
    0xba094800, 0x1ef3622f, 0x00f5138f, 0x1a22d9f3,
    0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46
};

// Affine point (input format - matches arkworks G1Affine)
typedef struct {
    uint32_t x[24];  // 96 bytes (761 bits + padding)
    uint32_t y[24];  // 96 bytes
    bool infinity;   // 1 byte
    // Note: arkworks uses 8-byte alignment → 200 bytes total
} __align__(8) affine_cgbn_t;

// Projective point (XYZZ coordinates) for computation
typedef struct {
    cgbn_mem_t<768> x;
    cgbn_mem_t<768> y;
    cgbn_mem_t<768> zz;   // Z^2
    cgbn_mem_t<768> zzz;  // Z^3
} xyzz_cgbn_t;

// Jacobian point (output format - matches arkworks G1Projective)
typedef struct {
    uint32_t x[24];  // 96 bytes
    uint32_t y[24];  // 96 bytes
    uint32_t z[24];  // 96 bytes
    bool infinity;   // 1 byte
    // Note: arkworks uses 8-byte alignment
} __align__(8) jacobian_cgbn_t;

// Scalar type (Fr element, 377 bits = 48 bytes)
typedef struct {
    uint32_t limbs[12];  // 12 × 32-bit = 384 bits (padded from 377)
    // Note: arkworks BigInt<6> uses 8-byte alignment → 48 bytes total
} __align__(8) scalar_cgbn_t;

/**
 * BW6-761 MSM Class (following CGBN sample pattern)
 *
 * This class encapsulates CGBN context and environment as member variables
 * to avoid parameter passing overhead that causes stack overflow.
 *
 * Pattern from: NVlabs/CGBN sample_03_powm/powm_odd.cu
 */
template<class params>
class bw6_msm_t {
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
  __device__ __forceinline__ bw6_msm_t(cgbn_monitor_t monitor,
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
   * - a, b are up to 761 bits each
   * - a * b is up to 1522 bits (doesn't fit in 768-bit bn_t)
   * - cgbn_mul only keeps the low 768 bits, losing upper bits!
   * - Must use cgbn_rem_wide to compute (full_product) mod P
   */
  __device__ __forceinline__ void field_mul(bn_t& r, const bn_t& a,
                                             const bn_t& b, const bn_t& P) {
    // Wide type holds 2*BITS = 1536 bits, enough for 761*2 = 1522 bit product
    typedef typename env_t::cgbn_wide_t wide_t;
    wide_t product;

    // Full multiplication: product = a * b (up to 1522 bits)
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
   * XYZZ Doubling Formulas for y^2 = x^3 + ax + b (BW6-761 has a=0):
   * A = Y1^2
   * V = 4*A
   * U = 2*Y1
   * W = U*V = 8*Y1*A
   * S = X1*V
   * M = 3*X1^2 + a*ZZ1^2  (for a=0: M = 3*X1^2)
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
    bn_t A, V, U, W, S, M, temp;

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

    // M = 3*X1^2  (BW6-761 has curve parameter a=0)
    field_mul(M, X1, X1, P);
    field_add(temp, M, M, P);
    field_add(M, M, temp, P);

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
   * Scalar field Fr is 377 bits (12 × 32-bit limbs)
   * All TPI threads must call this together
   */
  __device__ __forceinline__ void scalar_mul(
      const bn_t& P,
      // Result (XYZZ) - will be in projective form
      bn_t& rx, bn_t& ry, bn_t& rzz, bn_t& rzzz,
      // Point (Affine)
      const bn_t& px, const bn_t& py,
      bool point_is_infinity,
      // Scalar (Fr element, 377 bits = 12 limbs)
      const uint32_t scalar_limbs[12],
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
    for (int i = 0; i < 12; i++) {
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
    // Scalar is 377 bits = 12 × 32-bit limbs
    int highest_bit = -1;
    for (int limb_idx = 11; limb_idx >= 0; limb_idx--) {
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
};  // End of bw6_msm_t class

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
 * Threading: One thread group (bw6_cgbn_params_t::TPI threads cooperate)
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

    // Instantiate BW6 MSM class
    bw6_msm_t<params> msm(cgbn_report_monitor, report, instance);
    typename bw6_msm_t<params>::bn_t P, px, py;
    typename bw6_msm_t<params>::bn_t acc_x, acc_y, acc_zz, acc_zzz;
    typename bw6_msm_t<params>::bn_t term_x, term_y, term_zz, term_zzz;

    // Load modulus
    cgbn_load(msm._env, P, (cgbn_mem_t<params::BITS>*)BW6_761_P_DEVICE);

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
            for (int j = 0; j < 12; j++) {
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
                for (int limb_idx = 11; limb_idx >= 0; limb_idx--) {
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
                    typename bw6_msm_t<params>::bn_t dbl_x, dbl_y, dbl_zz, dbl_zzz;
                    typename bw6_msm_t<params>::bn_t add_x, add_y, add_zz, add_zzz;

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
                typename bw6_msm_t<params>::bn_t new_x, new_y, new_zz, new_zzz;

                // Check if points are equal or inverse by comparing affine coordinates:
                // acc represents (acc_x/acc_zz, acc_y/acc_zzz)
                // term represents (term_x/term_zz, term_y/term_zzz)
                //
                // X-coords equal if: acc_x * term_zz == term_x * acc_zz
                // Y-coords equal if: acc_y * term_zzz == term_y * acc_zzz
                // Y-coords are negatives if: acc_y * term_zzz + term_y * acc_zzz == 0 (mod P)
                //   equivalently: acc_y * term_zzz == P - (term_y * acc_zzz)
                typename bw6_msm_t<params>::bn_t cross1, cross2, cross3, cross4;
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
                typename bw6_msm_t<params>::bn_t y_sum;
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
        for (int i = 0; i < 24; i++) {
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

// Debug: Print first few limbs of a bn_t (only from lane 0)
template<class params>
__device__ void debug_print_bn(bw6_msm_t<params>& msm, const char* name, const typename bw6_msm_t<params>::bn_t& x) {
#ifdef BW6_DEBUG
    // Store to temp memory and print
    cgbn_mem_t<params::BITS> temp;
    cgbn_store(msm._env, &temp, x);
    if (threadIdx.x == 0) {
        printf("  %s: [0x%08x, 0x%08x, 0x%08x, 0x%08x, ...]\n",
               name, temp._limbs[0], temp._limbs[1], temp._limbs[2], temp._limbs[3]);
    }
    __syncthreads();
#endif
}

/**
 * FFI Entry Point: CGBN-based MSM for BW6-761 G1
 *
 * This is a minimal viable implementation to prove CGBN integration works.
 * Future optimizations: Pippenger algorithm, proper scalar multiplication, etc.
 */
extern "C" int msm_bw6_761_g1_cgbn(
    const void* points_ptr,
    const void* scalars_ptr,
    size_t count,
    void* result_ptr,
    size_t ffi_affine_sz,
    size_t ffi_scalar_sz
) {
    // Validate FFI layout (errors only)
    if (ffi_affine_sz != sizeof(affine_cgbn_t)) {
        fprintf(stderr, "[BW6-761 CGBN] ERROR: Affine size mismatch: CUDA expects %zu, Rust provided %zu\n",
               sizeof(affine_cgbn_t), ffi_affine_sz);
        return BW6_MSM_ERROR_AFFINE_LAYOUT;
    }

    if (ffi_scalar_sz != sizeof(scalar_cgbn_t)) {
        fprintf(stderr, "[BW6-761 CGBN] ERROR: Scalar size mismatch: CUDA expects %zu, Rust provided %zu\n",
               sizeof(scalar_cgbn_t), ffi_scalar_sz);
        return BW6_MSM_ERROR_SCALAR_LAYOUT;
    }

#ifdef BW6_DEBUG
    printf("[BW6-761 CGBN] MSM: count=%zu, affine=%zu, scalar=%zu\n",
           count, ffi_affine_sz, ffi_scalar_sz);
#endif

    // Handle edge cases
    if (count == 0) {
        jacobian_cgbn_t* result = static_cast<jacobian_cgbn_t*>(result_ptr);
        // Initialize all fields to zero
        memset(result->x, 0, sizeof(result->x));
        memset(result->y, 0, sizeof(result->y));
        memset(result->z, 0, sizeof(result->z));
        result->infinity = true;
        return BW6_MSM_SUCCESS;
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
    if (err != cudaSuccess) return BW6_MSM_ERROR_CUDA_RUNTIME;

    err = cudaMalloc(&d_scalars, sizeof(scalar_cgbn_t) * count);
    if (err != cudaSuccess) {
        cudaFree(d_points);
        return BW6_MSM_ERROR_CUDA_RUNTIME;
    }

    err = cudaMalloc(&d_result, sizeof(jacobian_cgbn_t));
    if (err != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_scalars);
        return BW6_MSM_ERROR_CUDA_RUNTIME;
    }

    err = cgbn_error_report_alloc(&d_report);
    if (err != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_scalars);
        cudaFree(d_result);
        return BW6_MSM_ERROR_CUDA_RUNTIME;
    }

    // Copy inputs to GPU
    err = cudaMemcpy(d_points, points, sizeof(affine_cgbn_t) * count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_error;

    err = cudaMemcpy(d_scalars, scalars, sizeof(scalar_cgbn_t) * count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_error;

    // Launch kernel (serial accumulation: single thread group)
    threads_per_block = bw6_cgbn_params_t::TPI;
    num_blocks = 1;

    msm_naive_cgbn_kernel<bw6_cgbn_params_t><<<num_blocks, threads_per_block>>>(
        d_report, d_result, d_points, d_scalars, count
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[BW6-761 CGBN] ERROR: Kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup_error;
    }

    // Check for CGBN errors
    if (cgbn_error_report_check(d_report)) {
        fprintf(stderr, "[BW6-761 CGBN] ERROR: CGBN error detected\n");
        goto cleanup_error;
    }

    // Copy result back
    err = cudaMemcpy(result, d_result, sizeof(jacobian_cgbn_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[BW6-761 CGBN] ERROR: Failed to copy result: %s\n", cudaGetErrorString(err));
        goto cleanup_error;
    }

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_result);
    cgbn_error_report_free(d_report);

    return BW6_MSM_SUCCESS;

cleanup_error:
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_result);
    cgbn_error_report_free(d_report);
    return BW6_MSM_ERROR_CUDA_RUNTIME;
}
