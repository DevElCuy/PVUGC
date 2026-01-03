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
// Note: TPI=8 is required for 768-bit - CGBN doesn't support TPI=4 at this bit width
class bw6_cgbn_params_t {
public:
  static const uint32_t TPB = 0;            // Get TPB from blockDim.x
  static const uint32_t MAX_ROTATION = 4;   // Good default value
  static const uint32_t SHM_LIMIT = 0;      // No shared memory
  static const bool CONSTANT_TIME = false;  // Not available yet
  static const uint32_t TPI = 8;            // 8 threads required for 768-bit (CGBN constraint)
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

// Montgomery constant: np0 = -P^(-1) mod 2^32
// Satisfies: np0 * P[0] ≡ 0xFFFFFFFF (mod 2^32)
__device__ __constant__ uint32_t BW6_761_NP0 = 0x8fa798dd;

// Montgomery R^2 mod P (R = 2^768)
// Used to convert to Montgomery form: mont(a) = a * R^2 * R^(-1) = a * R mod P
__device__ __constant__ uint32_t BW6_761_R2_DEVICE[24] = {
    0x2d1fa659, 0xc686392d, 0xf79484ab, 0x7b14c9b2,
    0xc1d2b459, 0x7fa1e825, 0x48329d88, 0xd6ec28f8,
    0x73a1ed40, 0x4afb427b, 0x0d5930ae, 0x972c6940,
    0x8c995976, 0x2c7a26bf, 0xc6e57af9, 0xac52e458,
    0x0c536dfe, 0xac731bfa, 0x0b103f50, 0x121e5c63,
    0xb886cda4, 0x8f1b0953, 0x2da8d807, 0x00ad253c
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
   * CRITICAL: Uses temp variables to avoid CGBN aliasing bugs.
   * CGBN's cgbn_sub/cgbn_add may not be alias-safe when output == input.
   *
   * Algorithm:
   *   if a >= b: r = a - b (result is in [0, P-1])
   *   if a < b:  r = a + (P - b) = a - b + P (result is in [0, P-1])
   */
  __device__ __forceinline__ void field_sub(bn_t& r, const bn_t& a,
                                             const bn_t& b, const bn_t& P) {
    bn_t temp_result;  // Use temp to avoid aliasing issues

    // Compare a and b
    int32_t cmp = cgbn_compare(_env, a, b);
    if (cmp >= 0) {
      // a >= b: simple subtraction, result is already in [0, P-1]
      cgbn_sub(_env, temp_result, a, b);
    } else {
      // a < b: compute (P - b) + a = a - b + P
      bn_t p_minus_b;
      cgbn_sub(_env, p_minus_b, P, b);  // P - b, no borrow since P > b
      cgbn_add(_env, temp_result, a, p_minus_b);  // a + (P - b) = a - b + P
    }

    // Copy result to output
    cgbn_set(_env, r, temp_result);
  }

  /**
   * Field multiplication with modular reduction using Montgomery form.
   *
   * Uses CGBN's built-in mont_mul which is highly optimized and doesn't
   * require wide multiplication (which causes massive code generation).
   *
   * IMPORTANT: Inputs must already be in Montgomery form!
   * mont_mul(aR, bR) = (aR * bR * R^(-1)) mod P = (ab)R mod P
   *
   * np0 = -P^(-1) mod 2^32 (Montgomery constant)
   *
   * CRITICAL FIX: CGBN's cgbn_mont_mul can return values in [0, 2P) instead
   * of strictly [0, P). This is documented in CGBN issue #15. We must add
   * explicit reduction after each Montgomery multiplication to ensure the
   * result is fully reduced to [0, P). Without this, subsequent operations
   * may produce incorrect results for certain input combinations.
   */
  __device__ __forceinline__ void field_mul(bn_t& r, const bn_t& a,
                                             const bn_t& b, const bn_t& P) {
    cgbn_mont_mul(_env, r, a, b, P, BW6_761_NP0);
    // Ensure result is fully reduced to [0, P)
    // CGBN mont_mul may return r in [P, 2P-1], so subtract P if needed
    if (cgbn_compare(_env, r, P) >= 0) {
      cgbn_sub(_env, r, r, P);
    }
  }

  /**
   * Convert a value to Montgomery form: mont(a) = a * R mod P
   * Done by: mont(a) = mont_mul(a, R^2) = a * R^2 * R^(-1) = a * R
   *
   * Note: Includes reduction to handle CGBN weak reduction (issue #15)
   */
  __device__ __forceinline__ void to_montgomery(bn_t& r, const bn_t& a, const bn_t& P) {
    bn_t R2;
    cgbn_load(_env, R2, (cgbn_mem_t<params::BITS>*)BW6_761_R2_DEVICE);
    cgbn_mont_mul(_env, r, a, R2, P, BW6_761_NP0);
    // Reduce if result >= P (CGBN weak reduction fix)
    if (cgbn_compare(_env, r, P) >= 0) {
      cgbn_sub(_env, r, r, P);
    }
  }

  /**
   * Convert from Montgomery form to normal: a = mont(a) * R^(-1) mod P
   * Done by: mont_mul(aR, 1) = aR * 1 * R^(-1) = a
   *
   * Note: Includes reduction to handle CGBN weak reduction (issue #15)
   */
  __device__ __forceinline__ void from_montgomery(bn_t& r, const bn_t& a, const bn_t& P) {
    bn_t one;
    cgbn_set_ui32(_env, one, 1);
    cgbn_mont_mul(_env, r, a, one, P, BW6_761_NP0);
    // Reduce if result >= P (CGBN weak reduction fix)
    if (cgbn_compare(_env, r, P) >= 0) {
      cgbn_sub(_env, r, r, P);
    }
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
      const bn_t& X2, const bn_t& Y2,
      int debug_bit_pos = -1  // For debug output (unused now)
  ) {
    bn_t U2, S2, P, R, PP, PPP, Q, temp, temp2;

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
    // Use dedicated temps to avoid aliasing in CGBN ops when r overlaps inputs.
    bn_t q_minus_x3, y3_tmp;
    field_sub(q_minus_x3, Q, X3, P_mod);    // q_minus_x3 = Q - X3
    field_mul(temp2, Y1, PPP, P_mod);       // temp2 = Y1*PPP
    field_mul(y3_tmp, R, q_minus_x3, P_mod);// y3_tmp = R * (Q-X3)
    field_sub(Y3, y3_tmp, temp2, P_mod);    // Y3 = y3_tmp - temp2

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
   * Convert XYZZ (in Montgomery form) to Jacobian coordinates (plain form)
   *
   * Normalize to affine by computing x = X/ZZ, y = Y/ZZZ
   * Then output as Jacobian with Z=1: (x, y, 1)
   *
   * IMPORTANT: Input coordinates are in Montgomery form.
   * Output must be in plain form for arkworks compatibility.
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

    // Compute ZZ^(-1) mod P (in Montgomery form)
    // Note: modular_inverse of aR gives (aR)^(-1) = a^(-1) * R^(-1)
    // To get inverse in Montgomery form, we need: (a^(-1))R
    // So we compute: inv = modular_inverse(ZZ) then convert to mont
    // But actually for division: X/ZZ = X * ZZ^(-1)
    // In Montgomery: (XR) * (ZZ^(-1) in mont) = X/ZZ in Montgomery
    // Let's think carefully:
    // We have XR, ZZR. We want (X/ZZ)R = XR * (ZZR)^(-1) * R
    // modular_inverse(ZZR) = (ZZR)^(-1) = ZZ^(-1) * R^(-1) (plain form!)
    // So: XR * ZZ^(-1) * R^(-1) * R (if we do mont_mul with R2 after inverse)
    // = XR * ZZ^(-1) = X/ZZ * R (Montgomery form) -- wrong
    // Actually: we want plain form output, so:
    // x_plain = X/ZZ = (XR) * (ZZR)^(-1) in plain
    // = (XR) * (ZZ^(-1) * R^(-1)) = X * ZZ^(-1) * R * R^(-1) = X * ZZ^(-1) = X/ZZ ✓
    // So cgbn_modular_inverse on Montgomery values gives plain form inverse!
    // Then multiplying Montgomery value by plain gives us:
    // (XR) * (ZZ^(-1)) via mont_mul = XR * ZZ^(-1) * R^(-1) = X * ZZ^(-1) * R * R^(-1) = X/ZZ (plain!)
    // So we should use regular multiply, not mont_mul for this...
    // Actually: cgbn_mul + cgbn_rem would overflow. Let's use a different approach.

    // Alternative approach: convert to plain first, then divide
    // X_plain = X / R, ZZ_plain = ZZ / R, then X_plain / ZZ_plain
    // But this is complex. Let's use the insight that:
    // If we want x = X/ZZ (in plain), and X,ZZ are in Montgomery:
    // x_plain = (XR)/(ZZR) = X/ZZ (the R's cancel!)
    // So we can convert X,ZZ to plain, then do plain division.

    // Convert ZZ, ZZZ from Montgomery to plain for computing inverses
    bn_t ZZ_plain, ZZZ_plain, one;
    cgbn_set_ui32(_env, one, 1);
    cgbn_mont_mul(_env, ZZ_plain, ZZ, one, P, BW6_761_NP0);     // ZZ_plain = ZZ/R
    // CGBN weak reduction fix
    if (cgbn_compare(_env, ZZ_plain, P) >= 0) {
      cgbn_sub(_env, ZZ_plain, ZZ_plain, P);
    }
    cgbn_mont_mul(_env, ZZZ_plain, ZZZ, one, P, BW6_761_NP0);   // ZZZ_plain = ZZZ/R
    // CGBN weak reduction fix
    if (cgbn_compare(_env, ZZZ_plain, P) >= 0) {
      cgbn_sub(_env, ZZZ_plain, ZZZ_plain, P);
    }

    // Compute inverses in plain form
    cgbn_modular_inverse(_env, zz_inv, ZZ_plain, P);            // zz_inv = ZZ^(-1) (plain)
    cgbn_modular_inverse(_env, zzz_inv, ZZZ_plain, P);          // zzz_inv = ZZZ^(-1) (plain)

    // Convert inverses to Montgomery form
    bn_t zz_inv_mont, zzz_inv_mont, R2;
    cgbn_load(_env, R2, (cgbn_mem_t<params::BITS>*)BW6_761_R2_DEVICE);
    cgbn_mont_mul(_env, zz_inv_mont, zz_inv, R2, P, BW6_761_NP0);    // zz_inv_mont = ZZ^(-1) * R
    // CGBN weak reduction fix
    if (cgbn_compare(_env, zz_inv_mont, P) >= 0) {
      cgbn_sub(_env, zz_inv_mont, zz_inv_mont, P);
    }
    cgbn_mont_mul(_env, zzz_inv_mont, zzz_inv, R2, P, BW6_761_NP0);  // zzz_inv_mont = ZZZ^(-1) * R
    // CGBN weak reduction fix
    if (cgbn_compare(_env, zzz_inv_mont, P) >= 0) {
      cgbn_sub(_env, zzz_inv_mont, zzz_inv_mont, P);
    }

    // Multiply in Montgomery form: X and Y are already in Montgomery form
    // x_norm_mont = X * zz_inv_mont = (X*R) * (ZZ^(-1)*R) * R^(-1) = X * ZZ^(-1) * R
    // y_norm_mont = Y * zzz_inv_mont = (Y*R) * (ZZZ^(-1)*R) * R^(-1) = Y * ZZZ^(-1) * R
    bn_t x_mont, y_mont;
    cgbn_mont_mul(_env, x_mont, X, zz_inv_mont, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(_env, x_mont, P) >= 0) {
      cgbn_sub(_env, x_mont, x_mont, P);
    }
    cgbn_mont_mul(_env, y_mont, Y, zzz_inv_mont, P, BW6_761_NP0);
    // CGBN weak reduction fix
    if (cgbn_compare(_env, y_mont, P) >= 0) {
      cgbn_sub(_env, y_mont, y_mont, P);
    }

    // Convert result from Montgomery to plain form for output
    cgbn_mont_mul(_env, x_norm, x_mont, one, P, BW6_761_NP0);   // x_norm = X/ZZ (plain)
    // CGBN weak reduction fix
    if (cgbn_compare(_env, x_norm, P) >= 0) {
      cgbn_sub(_env, x_norm, x_norm, P);
    }
    cgbn_mont_mul(_env, y_norm, y_mont, one, P, BW6_761_NP0);   // y_norm = Y/ZZZ (plain)
    // CGBN weak reduction fix
    if (cgbn_compare(_env, y_norm, P) >= 0) {
      cgbn_sub(_env, y_norm, y_norm, P);
    }

    // Store normalized affine coordinates (in plain form)
    cgbn_store(_env, (cgbn_mem_t<params::BITS>*)result->x, x_norm);
    cgbn_store(_env, (cgbn_mem_t<params::BITS>*)result->y, y_norm);

    // Store Z = 1 (plain form)
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

    // Load R^2 for Montgomery conversion (once, outside loop)
    typename bw6_msm_t<params>::bn_t R2;
    cgbn_load(msm._env, R2, (cgbn_mem_t<params::BITS>*)BW6_761_R2_DEVICE);

    // Montgomery form of 1: 1*R mod P
    typename bw6_msm_t<params>::bn_t one_mont;
    {
        typename bw6_msm_t<params>::bn_t one;
        cgbn_set_ui32(msm._env, one, 1);
        cgbn_mont_mul(msm._env, one_mont, one, R2, P, BW6_761_NP0);
        // CGBN weak reduction fix: ensure one_mont < P
        if (cgbn_compare(msm._env, one_mont, P) >= 0) {
            cgbn_sub(msm._env, one_mont, one_mont, P);
        }
    }

    // Process each point
    for (uint32_t i = 0; i < count; i++) {
        // Load point and convert to Montgomery form
        typename bw6_msm_t<params>::bn_t px_plain, py_plain;
        cgbn_load(msm._env, px_plain, (cgbn_mem_t<params::BITS>*)points[i].x);
        cgbn_load(msm._env, py_plain, (cgbn_mem_t<params::BITS>*)points[i].y);
        // Convert to Montgomery: px = px_plain * R mod P
        cgbn_mont_mul(msm._env, px, px_plain, R2, P, BW6_761_NP0);
        // CGBN weak reduction fix
        if (cgbn_compare(msm._env, px, P) >= 0) {
            cgbn_sub(msm._env, px, px, P);
        }
        cgbn_mont_mul(msm._env, py, py_plain, R2, P, BW6_761_NP0);
        // CGBN weak reduction fix
        if (cgbn_compare(msm._env, py, P) >= 0) {
            cgbn_sub(msm._env, py, py, P);
        }
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
                    // Point coords are already in Montgomery form
                    // ZZ=1 and ZZZ=1 must also be in Montgomery form
                    cgbn_set(msm._env, term_x, px);
                    cgbn_set(msm._env, term_y, py);
                    cgbn_set(msm._env, term_zz, one_mont);
                    cgbn_set(msm._env, term_zzz, one_mont);

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

// Pippenger configuration for BW6-761
#define PIPPENGER_MIN_WBITS 8
#define PIPPENGER_MAX_WBITS 16
#define PIPPENGER_SCALAR_BITS_BW6 377  // BW6-761 Fr has 377 bits

// Helper: Compute optimal window size based on point count
__host__ __device__ __forceinline__
uint32_t compute_optimal_wbits_bw6(uint32_t n) {
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
uint32_t compute_nwins_bw6(uint32_t scalar_bits, uint32_t wbits) {
    return (scalar_bits + wbits - 1) / wbits;
}

// XYZZ bucket with infinity flag for Pippenger (768-bit fields)
typedef struct {
    uint32_t x[24];
    uint32_t y[24];
    uint32_t zz[24];
    uint32_t zzz[24];
    bool is_infinity;
    uint8_t _padding[7];  // Align to 8 bytes
} __align__(8) bucket_xyzz_bw6_t;

// ============================================================================
// KERNEL 1: Scalar Breakdown (extract signed digits) - BW6-761
// ============================================================================
__global__ void breakdown_scalars_bw6_761(
    uint32_t* digits,           // Output: [n * nwins] packed digits
    const scalar_cgbn_t* scalars,  // Input: 12 x u32 limbs per scalar
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
        // BW6-761 scalars are 12 limbs
        uint64_t window_val = 0;
        if (limb_idx < 12) {
            window_val = scalar[limb_idx];
        }
        if (limb_idx + 1 < 12 && bit_idx + wbits > 32) {
            window_val |= ((uint64_t)scalar[limb_idx + 1]) << 32;
        }
        window_val = (window_val >> bit_idx) & wmask;

        // Add carry from previous window's Booth encoding
        window_val += carry;
        carry = 0;

        // Booth encoding: if value > half_buckets, subtract 2^wbits and carry 1
        uint32_t sign = 0;
        if (window_val > half_buckets) {
            window_val = (1u << wbits) - window_val;
            sign = 1;
            carry = 1;
        }

        // Store packed digit: bucket_id in low 16 bits, sign in bit 31
        uint32_t packed = (uint32_t)window_val | (sign << 31);
        digits[win * n + idx] = packed;
    }
}

// ============================================================================
// KERNEL 2: Histogram (count points per bucket)
// ============================================================================
__global__ void histogram_buckets_bw6_761(
    uint32_t* histogram,        // Output: [nwins * num_buckets] counts
    const uint32_t* digits,     // Input: [n * nwins] packed digits
    uint32_t n,
    uint32_t nwins,
    uint32_t wbits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t num_buckets = (1u << (wbits - 1)) + 1;

    for (uint32_t win = 0; win < nwins; win++) {
        uint32_t packed = digits[win * n + idx];
        uint32_t bucket_id = packed & 0xFFFF;

        if (bucket_id > 0) {
            uint32_t hist_idx = win * num_buckets + bucket_id;
            atomicAdd(&histogram[hist_idx], 1);
        }
    }
}

// ============================================================================
// KERNEL 3: Prefix Sum (compute bucket offsets from histogram)
// ============================================================================
__global__ void prefix_sum_histogram_bw6_761(
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
__global__ void scatter_to_buckets_bw6_761(
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

    uint32_t num_buckets = (1u << (wbits - 1)) + 1;

    for (uint32_t win = 0; win < nwins; win++) {
        uint32_t packed = digits[win * n + idx];
        uint32_t bucket_id = packed & 0xFFFF;
        uint32_t sign = (packed >> 31) & 1;

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
template<class params>
__global__ void accumulate_buckets_bw6_761(
    cgbn_error_report_t* report,
    bucket_xyzz_bw6_t* buckets,     // Output: [nwins * num_buckets]
    const affine_cgbn_t* points,    // Input: [n] points
    const uint32_t* sorted_indices, // Input: [nwins * n] sorted indices
    const uint32_t* offsets,        // Input: [nwins * num_buckets] offsets
    const uint32_t* histogram,      // Input: [nwins * num_buckets] counts
    uint32_t n,
    uint32_t nwins,
    uint32_t wbits
) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    uint32_t num_buckets = (1u << (wbits - 1)) + 1;
    uint32_t total_buckets = nwins * num_buckets;

    if ((uint32_t)instance >= total_buckets) return;

    uint32_t win = instance / num_buckets;
    uint32_t bucket_id = instance % num_buckets;

    // Skip bucket 0 (identity bucket)
    if (bucket_id == 0) {
        buckets[instance].is_infinity = true;
        return;
    }

    // Get bucket range from histogram
    uint32_t hist_idx = win * num_buckets + bucket_id;
    uint32_t start = offsets[hist_idx];
    uint32_t count = histogram[hist_idx];

    // Initialize CGBN
    bw6_msm_t<params> msm(cgbn_report_monitor, report, instance);
    typename bw6_msm_t<params>::bn_t P, px, py, px_plain, py_plain;
    typename bw6_msm_t<params>::bn_t acc_x, acc_y, acc_zz, acc_zzz;
    typename bw6_msm_t<params>::bn_t tmp_x, tmp_y, tmp_zz, tmp_zzz;

    // Load modulus and R^2 for Montgomery conversion
    cgbn_load(msm._env, P, (cgbn_mem_t<params::BITS>*)BW6_761_P_DEVICE);
    typename bw6_msm_t<params>::bn_t R2;
    cgbn_load(msm._env, R2, (cgbn_mem_t<params::BITS>*)BW6_761_R2_DEVICE);

    // Montgomery form of 1: 1*R mod P
    typename bw6_msm_t<params>::bn_t one_mont;
    {
        typename bw6_msm_t<params>::bn_t one;
        cgbn_set_ui32(msm._env, one, 1);
        cgbn_mont_mul(msm._env, one_mont, one, R2, P, BW6_761_NP0);
        // CGBN weak reduction fix
        if (cgbn_compare(msm._env, one_mont, P) >= 0) {
            cgbn_sub(msm._env, one_mont, one_mont, P);
        }
    }

    bool acc_is_infinity = true;

    // Accumulate all points in this bucket
    for (uint32_t i = 0; i < count; i++) {
        uint32_t packed_idx = sorted_indices[win * n + start + i];
        uint32_t point_idx = packed_idx & 0x7FFFFFFF;
        bool negate = (packed_idx >> 31) != 0;

        // Skip points at infinity
        if (points[point_idx].infinity) continue;

        // Load point in plain form and convert to Montgomery form
        cgbn_load(msm._env, px_plain, (cgbn_mem_t<params::BITS>*)points[point_idx].x);
        cgbn_load(msm._env, py_plain, (cgbn_mem_t<params::BITS>*)points[point_idx].y);
        cgbn_mont_mul(msm._env, px, px_plain, R2, P, BW6_761_NP0);
        // CGBN weak reduction fix
        if (cgbn_compare(msm._env, px, P) >= 0) {
            cgbn_sub(msm._env, px, px, P);
        }
        cgbn_mont_mul(msm._env, py, py_plain, R2, P, BW6_761_NP0);
        // CGBN weak reduction fix
        if (cgbn_compare(msm._env, py, P) >= 0) {
            cgbn_sub(msm._env, py, py, P);
        }

        // Apply negation if needed (negate y coordinate in Montgomery form)
        if (negate) {
            typename bw6_msm_t<params>::bn_t neg_y;
            cgbn_sub(msm._env, neg_y, P, py);
            cgbn_set(msm._env, py, neg_y);
        }

        if (acc_is_infinity) {
            // First point: initialize accumulator as XYZZ with Z=1 (in Montgomery form)
            cgbn_set(msm._env, acc_x, px);
            cgbn_set(msm._env, acc_y, py);
            cgbn_set(msm._env, acc_zz, one_mont);
            cgbn_set(msm._env, acc_zzz, one_mont);
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
template<class params>
__global__ void integrate_buckets_bw6_761(
    cgbn_error_report_t* report,
    bucket_xyzz_bw6_t* window_sums,     // Output: [nwins]
    const bucket_xyzz_bw6_t* buckets,   // Input: [nwins * num_buckets]
    uint32_t nwins,
    uint32_t wbits
) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    if ((uint32_t)instance >= nwins) return;

    uint32_t win = instance;
    uint32_t num_buckets = (1u << (wbits - 1)) + 1;
    uint32_t bucket_base = win * num_buckets;

    // Initialize CGBN
    bw6_msm_t<params> msm(cgbn_report_monitor, report, instance);
    typename bw6_msm_t<params>::bn_t P;
    typename bw6_msm_t<params>::bn_t acc_x, acc_y, acc_zz, acc_zzz;
    typename bw6_msm_t<params>::bn_t sum_x, sum_y, sum_zz, sum_zzz;
    typename bw6_msm_t<params>::bn_t tmp_x, tmp_y, tmp_zz, tmp_zzz;
    typename bw6_msm_t<params>::bn_t bkt_x, bkt_y, bkt_zz, bkt_zzz;

    // Load modulus
    cgbn_load(msm._env, P, (cgbn_mem_t<params::BITS>*)BW6_761_P_DEVICE);

    bool acc_is_infinity = true;
    bool sum_is_infinity = true;

    // Horner-like integration: for b = num_buckets-1 down to 1:
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
                // Check if acc and bucket[b] are the same point
                typename bw6_msm_t<params>::bn_t lhs, rhs;
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
                // Check if sum and acc are the same point
                typename bw6_msm_t<params>::bn_t lhs, rhs;
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
                    // Inverse points -> result is infinity
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
template<class params>
__global__ void combine_windows_bw6_761(
    cgbn_error_report_t* report,
    jacobian_cgbn_t* result_out,
    const bucket_xyzz_bw6_t* window_sums,
    uint32_t nwins,
    uint32_t wbits
) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
    if (instance != 0) return;

    bw6_msm_t<params> msm(cgbn_report_monitor, report, instance);
    typename bw6_msm_t<params>::bn_t P;
    typename bw6_msm_t<params>::bn_t res_x, res_y, res_zz, res_zzz;
    typename bw6_msm_t<params>::bn_t tmp_x, tmp_y, tmp_zz, tmp_zzz;
    typename bw6_msm_t<params>::bn_t win_x, win_y, win_zz, win_zzz;

    cgbn_load(msm._env, P, (cgbn_mem_t<params::BITS>*)BW6_761_P_DEVICE);

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
                // Check if result and window_sum are the same point
                typename bw6_msm_t<params>::bn_t lhs, rhs;
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
        for (int i = 0; i < 24; i++) {
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
// FFI Entry Point: Pippenger MSM for BW6-761 G1
// ============================================================================
extern "C" int msm_bw6_761_g1_cgbn_pippenger(
    const void* points_ptr,
    const void* scalars_ptr,
    size_t count,
    void* result_ptr,
    size_t ffi_affine_sz,
    size_t ffi_scalar_sz
) {
    // Validate FFI layout
    if (ffi_affine_sz != sizeof(affine_cgbn_t)) {
        fprintf(stderr, "[BW6-761 Pippenger] ERROR: Affine size mismatch\n");
        return BW6_MSM_ERROR_AFFINE_LAYOUT;
    }
    if (ffi_scalar_sz != sizeof(scalar_cgbn_t)) {
        fprintf(stderr, "[BW6-761 Pippenger] ERROR: Scalar size mismatch\n");
        return BW6_MSM_ERROR_SCALAR_LAYOUT;
    }

    // Handle edge case
    if (count == 0) {
        jacobian_cgbn_t* result = static_cast<jacobian_cgbn_t*>(result_ptr);
        memset(result, 0, sizeof(jacobian_cgbn_t));
        result->infinity = true;
        return BW6_MSM_SUCCESS;
    }

    // NOTE: Serial fallback disabled - serial BW6 MSM has a pre-existing bug
    // For very small inputs, use Pippenger anyway (until serial is fixed)
    // if (count < 64) {
    //     return msm_bw6_761_g1_cgbn(points_ptr, scalars_ptr, count,
    //                                result_ptr, ffi_affine_sz, ffi_scalar_sz);
    // }

    const affine_cgbn_t* points = static_cast<const affine_cgbn_t*>(points_ptr);
    const scalar_cgbn_t* scalars = static_cast<const scalar_cgbn_t*>(scalars_ptr);
    jacobian_cgbn_t* result = static_cast<jacobian_cgbn_t*>(result_ptr);

    // Compute optimal parameters
    uint32_t n = (uint32_t)count;
    uint32_t wbits = compute_optimal_wbits_bw6(n);
    uint32_t nwins = compute_nwins_bw6(PIPPENGER_SCALAR_BITS_BW6, wbits);
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
    bucket_xyzz_bw6_t* d_buckets = nullptr;
    bucket_xyzz_bw6_t* d_window_sums = nullptr;
    jacobian_cgbn_t* d_result = nullptr;
    cgbn_error_report_t* d_report = nullptr;

    // Allocation
    err = cudaMalloc(&d_points, sizeof(affine_cgbn_t) * n);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_scalars, sizeof(scalar_cgbn_t) * n);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_digits, sizeof(uint32_t) * nwins * n);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_histogram, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_offsets, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_bucket_counters, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_sorted_indices, sizeof(uint32_t) * nwins * n);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_buckets, sizeof(bucket_xyzz_bw6_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_window_sums, sizeof(bucket_xyzz_bw6_t) * nwins);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMalloc(&d_result, sizeof(jacobian_cgbn_t));
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cgbn_error_report_alloc(&d_report);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    // Copy inputs to device
    err = cudaMemcpy(d_points, points, sizeof(affine_cgbn_t) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMemcpy(d_scalars, scalars, sizeof(scalar_cgbn_t) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    // Zero histograms and counters
    err = cudaMemset(d_histogram, 0, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    err = cudaMemset(d_bucket_counters, 0, sizeof(uint32_t) * total_buckets);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

    {
        // Kernel launches
        int threads = 256;
        int blocks_n = (n + threads - 1) / threads;
        int blocks_nwins = (nwins + threads - 1) / threads;

        // 1. Breakdown scalars
        breakdown_scalars_bw6_761<<<blocks_n, threads>>>(
            d_digits, d_scalars, n, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        // 2. Build histogram
        histogram_buckets_bw6_761<<<blocks_n, threads>>>(
            d_histogram, d_digits, n, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        // 3. Prefix sum for offsets
        prefix_sum_histogram_bw6_761<<<blocks_nwins, threads>>>(
            d_offsets, d_histogram, nwins, num_buckets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        // 4. Scatter to buckets
        scatter_to_buckets_bw6_761<<<blocks_n, threads>>>(
            d_sorted_indices, d_bucket_counters, d_offsets, d_digits,
            n, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        // 5. Accumulate buckets (CGBN kernel)
        int tpi = bw6_cgbn_params_t::TPI;
        int threads_accumulate = 32 * tpi;  // 32 CGBN instances per block
        int instances_needed = total_buckets;
        int blocks_accumulate = (instances_needed * tpi + threads_accumulate - 1) / threads_accumulate;

        accumulate_buckets_bw6_761<bw6_cgbn_params_t><<<blocks_accumulate, threads_accumulate>>>(
            d_report, d_buckets, d_points, d_sorted_indices, d_offsets, d_histogram,
            n, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        // 6. Integrate buckets (CGBN kernel)
        int instances_integrate = nwins;
        int blocks_integrate = (instances_integrate * tpi + threads_accumulate - 1) / threads_accumulate;

        integrate_buckets_bw6_761<bw6_cgbn_params_t><<<blocks_integrate, threads_accumulate>>>(
            d_report, d_window_sums, d_buckets, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        // 7. Combine windows (single CGBN instance)
        combine_windows_bw6_761<bw6_cgbn_params_t><<<1, tpi>>>(
            d_report, d_result, d_window_sums, nwins, wbits
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;
    }

    // Check CGBN errors
    if (cgbn_error_report_check(d_report)) {
        fprintf(stderr, "[BW6-761 Pippenger] ERROR: CGBN error detected\n");
        goto pippenger_cleanup_error_bw6;
    }

    // Copy result back
    err = cudaMemcpy(result, d_result, sizeof(jacobian_cgbn_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto pippenger_cleanup_error_bw6;

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

    return BW6_MSM_SUCCESS;

pippenger_cleanup_error_bw6:
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
    return BW6_MSM_ERROR_CUDA_RUNTIME;
}
