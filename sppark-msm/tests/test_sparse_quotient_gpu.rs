// Sparse Quotient GPU Kernel Tests
// Custom test harness (harness = false) to avoid Rust test framework CUDA cleanup conflicts
//
// These tests verify that the sparse quotient coefficient computation GPU kernel
// produces correct results matching the CPU reference implementation.
//
// Run with: cargo test --release --features gpu --test test_sparse_quotient_gpu
//
// FIXED: The __shfl_sync bug has been fixed by using instance-scoped masks and width=TPI.
// GPU tests are now enabled.

#![cfg(all(feature = "gpu", sparse_quotient_mnt4_available))]

use ark_ff::{BigInt, FftField, One, PrimeField, UniformRand, Zero};
use ark_mnt4_298::Fr as MNT4Fr;
use ark_std::rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use sppark_msm::{compute_sparse_quotient_coeffs_mnt4_298_gpu, SparseMatrixCsr};

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert an arkworks Fr element to u32[10] limbs (little-endian)
fn fr_to_limbs(f: &MNT4Fr) -> [u32; 10] {
    let bigint: BigInt<5> = f.into_bigint();
    let mut limbs = [0u32; 10];
    for (i, &limb64) in bigint.0.iter().enumerate() {
        limbs[2 * i] = limb64 as u32;
        limbs[2 * i + 1] = (limb64 >> 32) as u32;
    }
    limbs
}

/// Convert u32[10] limbs (little-endian) back to arkworks Fr
fn limbs_to_fr(limbs: &[u32; 10]) -> MNT4Fr {
    let mut bigint_limbs = [0u64; 5];
    for i in 0..5 {
        bigint_limbs[i] = (limbs[2 * i] as u64) | ((limbs[2 * i + 1] as u64) << 32);
    }
    MNT4Fr::from_bigint(BigInt(bigint_limbs)).unwrap_or(MNT4Fr::zero())
}

/// CPU reference implementation for sparse quotient coefficient computation
/// This matches the algorithm in src/pvugc_outer.rs lines 731-754
fn cpu_sparse_quotient_coeffs(
    col_a_rows: &[(usize, MNT4Fr)], // (row_idx, value) pairs for column i
    col_b_rows: &[(usize, MNT4Fr)], // (row_idx, value) pairs for column j
    domain_elements: &[MNT4Fr],     // ω^k for k = 0..n-1
    inv_domain_elements: &[MNT4Fr], // ω^{-k} for k = 0..n-1
    inv_n_one_minus_omega: &[MNT4Fr], // inv(n * (1 - ω^d)) for d = 0..n-1
    domain_size: usize,
) -> (Vec<MNT4Fr>, Vec<MNT4Fr>, Vec<(usize, MNT4Fr)>) {
    let n_u = col_a_rows.len();
    let n_v = col_b_rows.len();

    let mut acc_u = vec![MNT4Fr::zero(); n_u];
    let mut acc_v = vec![MNT4Fr::zero(); n_v];
    let mut diag_terms: Vec<(usize, MNT4Fr)> = Vec::new();

    for (idx_u, &(k, val_u)) in col_a_rows.iter().enumerate() {
        for (idx_v, &(m, val_v)) in col_b_rows.iter().enumerate() {
            let prod = val_u * val_v;

            if k == m {
                // Diagonal term
                diag_terms.push((k, prod));
            } else {
                // Off-diagonal term
                let wm = domain_elements[m];
                let wk = domain_elements[k];
                // inv(n*(wk-wm)) = -ω^{-m} * inv_n_one_minus_omega[(k-m) mod n]
                let d = if k >= m { k - m } else { k + domain_size - m };
                let inv_denom = -(inv_domain_elements[m] * inv_n_one_minus_omega[d]);
                let common = prod * inv_denom;
                acc_u[idx_u] += common * wm;
                acc_v[idx_v] -= common * wk;
            }
        }
    }

    (acc_u, acc_v, diag_terms)
}

/// Build a simple evaluation domain (multiplicative subgroup)
/// Returns (domain_elements, inv_domain_elements, inv_n_one_minus_omega)
fn build_domain_tables(domain_size: usize) -> (Vec<MNT4Fr>, Vec<MNT4Fr>, Vec<MNT4Fr>) {
    // Find a primitive n-th root of unity
    let omega = MNT4Fr::get_root_of_unity(domain_size as u64).expect("Domain size must be power of 2");
    let n_field = MNT4Fr::from(domain_size as u64);

    // domain_elements[k] = ω^k
    let mut domain_elements = Vec::with_capacity(domain_size);
    let mut current = MNT4Fr::one();
    for _ in 0..domain_size {
        domain_elements.push(current);
        current *= omega;
    }

    // inv_domain_elements[k] = ω^{-k} = ω^{n-k}
    let mut inv_domain_elements = vec![MNT4Fr::one(); domain_size];
    for i in 1..domain_size {
        inv_domain_elements[i] = domain_elements[domain_size - i];
    }

    // inv_n_one_minus_omega[d] = 1/(n * (1 - ω^d))
    // For d=0: 1 - ω^0 = 0, so we set a placeholder (not used for d=0)
    let mut inv_n_one_minus_omega = vec![MNT4Fr::zero(); domain_size];
    let mut denoms = Vec::with_capacity(domain_size - 1);
    let mut indices = Vec::with_capacity(domain_size - 1);

    for d in 1..domain_size {
        let denom = n_field * (MNT4Fr::one() - domain_elements[d]);
        denoms.push(denom);
        indices.push(d);
    }

    ark_ff::batch_inversion(&mut denoms);
    for (i, &d) in indices.iter().enumerate() {
        inv_n_one_minus_omega[d] = denoms[i];
    }

    (domain_elements, inv_domain_elements, inv_n_one_minus_omega)
}

/// Build CSR matrix from list of columns (each column is a list of (row_idx, value) pairs)
fn build_csr_matrix(columns: &[Vec<(usize, MNT4Fr)>]) -> SparseMatrixCsr {
    let num_cols = columns.len();
    let mut col_ptr = Vec::with_capacity(num_cols + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    col_ptr.push(0u32);
    for col in columns {
        for &(r, v) in col {
            row_idx.push(r as u32);
            values.push(fr_to_limbs(&v));
        }
        col_ptr.push(row_idx.len() as u32);
    }

    SparseMatrixCsr {
        col_ptr,
        row_idx,
        values,
    }
}

/// Compare two Fr vectors for equality with tolerance for numerical precision
fn vecs_equal(a: &[MNT4Fr], b: &[MNT4Fr]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// Compare diagonal terms (order may differ between CPU and GPU)
fn diag_terms_equal(
    cpu_terms: &[(usize, MNT4Fr)],
    gpu_terms: &[(u32, [u32; 10])],
) -> bool {
    if cpu_terms.len() != gpu_terms.len() {
        return false;
    }

    // Sort both by row index for comparison
    let mut cpu_sorted: Vec<_> = cpu_terms.to_vec();
    cpu_sorted.sort_by_key(|(k, _)| *k);

    let mut gpu_sorted: Vec<_> = gpu_terms
        .iter()
        .map(|&(k, v)| (k as usize, limbs_to_fr(&v)))
        .collect::<Vec<_>>();
    gpu_sorted.sort_by_key(|(k, _)| *k);

    // Compare aggregated by key (CPU aggregates, GPU may not)
    // Actually, check if values match for same keys
    for (cpu_k, cpu_v) in &cpu_sorted {
        let matching: Vec<_> = gpu_sorted.iter().filter(|(k, _)| k == cpu_k).collect();
        if matching.is_empty() {
            return false;
        }
        // GPU may have multiple entries for same k, sum them
        let gpu_sum: MNT4Fr = matching.iter().map(|(_, v)| *v).sum();
        if gpu_sum != *cpu_v {
            // Check if there are multiple CPU entries for same k
            let cpu_sum: MNT4Fr = cpu_sorted.iter().filter(|(k, _)| k == cpu_k).map(|(_, v)| *v).sum();
            if gpu_sum != cpu_sum {
                return false;
            }
        }
    }

    true
}

// ============================================================================
// Test Functions
// ============================================================================

fn test_gpu_available() -> bool {
    println!("\n=== Test: GPU Kernel Available ===");

    if sppark_msm::sparse_quotient_gpu_available() {
        println!("  PASS: Sparse quotient GPU kernel is available");
        true
    } else {
        println!("  SKIP: Sparse quotient GPU kernel not available");
        false
    }
}

fn test_empty_pairs() -> bool {
    println!("\n=== Test: Empty Pairs ===");

    let domain_size = 8usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Convert to limbs for GPU
    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    // Empty columns
    let columns_a: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![]];
    let columns_b: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![]];
    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let pairs: Vec<(u32, u32)> = vec![];

    match compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a,
        &col_b,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        1,
        1,
        1,
    ) {
        Ok(results) => {
            if results.is_empty() {
                println!("  PASS: Empty pairs returns empty results");
                true
            } else {
                println!("  FAIL: Expected empty results, got {}", results.len());
                false
            }
        }
        Err(e) => {
            println!("  FAIL: GPU error {:?}", e);
            false
        }
    }
}

fn test_single_pair_diagonal_only() -> bool {
    println!("\n=== Test: Single Pair with Diagonal Only ===");

    let domain_size = 8usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Create columns with matching row indices (diagonal case)
    // Column 0: row 2 with value 3
    // Column 0: row 2 with value 5
    let columns_a: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![(2, MNT4Fr::from(3u64))]];
    let columns_b: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![(2, MNT4Fr::from(5u64))]];

    // CPU reference
    let (cpu_acc_u, cpu_acc_v, cpu_diag) = cpu_sparse_quotient_coeffs(
        &columns_a[0],
        &columns_b[0],
        &domain_elements,
        &inv_domain_elements,
        &inv_n_one_minus_omega,
        domain_size,
    );

    // GPU computation
    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let pairs = vec![(0u32, 0u32)];

    match compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a,
        &col_b,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        1,
        1,
        1,
    ) {
        Ok(results) => {
            if results.len() != 1 {
                println!("  FAIL: Expected 1 result, got {}", results.len());
                return false;
            }

            let result = &results[0];

            // Check acc_u and acc_v are zero (no off-diagonal terms)
            let gpu_acc_u: Vec<MNT4Fr> = result.acc_u.iter().map(limbs_to_fr).collect();
            let gpu_acc_v: Vec<MNT4Fr> = result.acc_v.iter().map(limbs_to_fr).collect();

            if !vecs_equal(&cpu_acc_u, &gpu_acc_u) {
                println!("  FAIL: acc_u mismatch");
                println!("    CPU: {:?}", cpu_acc_u);
                println!("    GPU: {:?}", gpu_acc_u);
                return false;
            }

            if !vecs_equal(&cpu_acc_v, &gpu_acc_v) {
                println!("  FAIL: acc_v mismatch");
                println!("    CPU: {:?}", cpu_acc_v);
                println!("    GPU: {:?}", gpu_acc_v);
                return false;
            }

            // Check diagonal terms
            if !diag_terms_equal(&cpu_diag, &result.diag_terms) {
                println!("  FAIL: diagonal terms mismatch");
                println!("    CPU: {:?}", cpu_diag);
                println!("    GPU: {:?}", result.diag_terms);
                return false;
            }

            // Verify expected diagonal: k=2, value=3*5=15
            if cpu_diag.len() != 1 || cpu_diag[0].0 != 2 || cpu_diag[0].1 != MNT4Fr::from(15u64) {
                println!("  FAIL: Unexpected CPU diagonal");
                return false;
            }

            println!("  PASS: Diagonal term k=2, value=15 matches");
            true
        }
        Err(e) => {
            println!("  FAIL: GPU error {:?}", e);
            false
        }
    }
}

fn test_single_pair_off_diagonal_only() -> bool {
    println!("\n=== Test: Single Pair with Off-Diagonal Only ===");

    let domain_size = 8usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Create columns with different row indices (off-diagonal case)
    // Column 0: row 1 with value 2
    // Column 0: row 3 with value 4
    let columns_a: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![(1, MNT4Fr::from(2u64))]];
    let columns_b: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![(3, MNT4Fr::from(4u64))]];

    // CPU reference
    let (cpu_acc_u, cpu_acc_v, cpu_diag) = cpu_sparse_quotient_coeffs(
        &columns_a[0],
        &columns_b[0],
        &domain_elements,
        &inv_domain_elements,
        &inv_n_one_minus_omega,
        domain_size,
    );

    // GPU computation
    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let pairs = vec![(0u32, 0u32)];

    match compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a,
        &col_b,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        1,
        1,
        1,
    ) {
        Ok(results) => {
            if results.len() != 1 {
                println!("  FAIL: Expected 1 result, got {}", results.len());
                return false;
            }

            let result = &results[0];

            let gpu_acc_u: Vec<MNT4Fr> = result.acc_u.iter().map(limbs_to_fr).collect();
            let gpu_acc_v: Vec<MNT4Fr> = result.acc_v.iter().map(limbs_to_fr).collect();

            if !vecs_equal(&cpu_acc_u, &gpu_acc_u) {
                println!("  FAIL: acc_u mismatch");
                println!("    CPU: {:?}", cpu_acc_u);
                println!("    GPU: {:?}", gpu_acc_u);
                return false;
            }

            if !vecs_equal(&cpu_acc_v, &gpu_acc_v) {
                println!("  FAIL: acc_v mismatch");
                println!("    CPU: {:?}", cpu_acc_v);
                println!("    GPU: {:?}", gpu_acc_v);
                return false;
            }

            // No diagonal terms expected
            if !cpu_diag.is_empty() || !result.diag_terms.is_empty() {
                println!("  FAIL: Expected no diagonal terms");
                return false;
            }

            println!("  PASS: Off-diagonal coefficients match (no diagonals)");
            true
        }
        Err(e) => {
            println!("  FAIL: GPU error {:?}", e);
            false
        }
    }
}

fn test_single_pair_mixed() -> bool {
    println!("\n=== Test: Single Pair with Mixed Diagonal and Off-Diagonal ===");

    let domain_size = 8usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Column 0: rows [1, 2, 3] with values [2, 3, 5]
    // Column 0: rows [2, 4] with values [7, 11]
    // This gives:
    //   Diagonal: (1,2)*(...) none, (2,2) = 3*7, (3,2) none
    //   Off-diagonal: all other combinations
    let columns_a: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![
        (1, MNT4Fr::from(2u64)),
        (2, MNT4Fr::from(3u64)),
        (3, MNT4Fr::from(5u64)),
    ]];
    let columns_b: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![
        (2, MNT4Fr::from(7u64)),
        (4, MNT4Fr::from(11u64)),
    ]];

    // CPU reference
    let (cpu_acc_u, cpu_acc_v, cpu_diag) = cpu_sparse_quotient_coeffs(
        &columns_a[0],
        &columns_b[0],
        &domain_elements,
        &inv_domain_elements,
        &inv_n_one_minus_omega,
        domain_size,
    );

    // GPU computation
    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let pairs = vec![(0u32, 0u32)];

    match compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a,
        &col_b,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        3,
        2,
        2,
    ) {
        Ok(results) => {
            if results.len() != 1 {
                println!("  FAIL: Expected 1 result, got {}", results.len());
                return false;
            }

            let result = &results[0];

            let gpu_acc_u: Vec<MNT4Fr> = result.acc_u.iter().map(limbs_to_fr).collect();
            let gpu_acc_v: Vec<MNT4Fr> = result.acc_v.iter().map(limbs_to_fr).collect();

            if !vecs_equal(&cpu_acc_u, &gpu_acc_u) {
                println!("  FAIL: acc_u mismatch");
                println!("    CPU: {:?}", cpu_acc_u);
                println!("    GPU: {:?}", gpu_acc_u);
                return false;
            }

            if !vecs_equal(&cpu_acc_v, &gpu_acc_v) {
                println!("  FAIL: acc_v mismatch");
                println!("    CPU: {:?}", cpu_acc_v);
                println!("    GPU: {:?}", gpu_acc_v);
                return false;
            }

            // Check diagonal terms
            if !diag_terms_equal(&cpu_diag, &result.diag_terms) {
                println!("  FAIL: diagonal terms mismatch");
                println!("    CPU: {:?}", cpu_diag);
                println!(
                    "    GPU: {:?}",
                    result.diag_terms
                        .iter()
                        .map(|&(k, v)| (k, limbs_to_fr(&v)))
                        .collect::<Vec<_>>()
                );
                return false;
            }

            // Should have 1 diagonal: k=2, value=3*7=21
            if cpu_diag.len() != 1 {
                println!("  FAIL: Expected 1 diagonal term, got {}", cpu_diag.len());
                return false;
            }

            println!("  PASS: Mixed diagonal/off-diagonal coefficients match");
            true
        }
        Err(e) => {
            println!("  FAIL: GPU error {:?}", e);
            false
        }
    }
}

fn test_multiple_pairs_batch() -> bool {
    println!("\n=== Test: Multiple Pairs Batch Processing ===");

    let domain_size = 16usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Create multiple columns
    let columns_a: Vec<Vec<(usize, MNT4Fr)>> = vec![
        vec![(0, MNT4Fr::from(1u64)), (2, MNT4Fr::from(2u64))],
        vec![(1, MNT4Fr::from(3u64)), (3, MNT4Fr::from(4u64)), (5, MNT4Fr::from(5u64))],
        vec![(4, MNT4Fr::from(6u64))],
    ];
    let columns_b: Vec<Vec<(usize, MNT4Fr)>> = vec![
        vec![(0, MNT4Fr::from(7u64)), (1, MNT4Fr::from(8u64))],
        vec![(3, MNT4Fr::from(9u64)), (5, MNT4Fr::from(10u64))],
    ];

    // Test pairs
    let pairs = vec![(0u32, 0u32), (0u32, 1u32), (1u32, 0u32), (1u32, 1u32), (2u32, 0u32)];

    // Compute CPU references for each pair
    let mut cpu_results: Vec<(Vec<MNT4Fr>, Vec<MNT4Fr>, Vec<(usize, MNT4Fr)>)> = Vec::new();
    for &(i, j) in &pairs {
        let (acc_u, acc_v, diag) = cpu_sparse_quotient_coeffs(
            &columns_a[i as usize],
            &columns_b[j as usize],
            &domain_elements,
            &inv_domain_elements,
            &inv_n_one_minus_omega,
            domain_size,
        );
        cpu_results.push((acc_u, acc_v, diag));
    }

    // GPU computation
    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let max_col_a = columns_a.iter().map(|c| c.len()).max().unwrap_or(0) as u32;
    let max_col_b = columns_b.iter().map(|c| c.len()).max().unwrap_or(0) as u32;
    let max_diag = std::cmp::min(max_col_a, max_col_b);

    match compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a,
        &col_b,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        max_col_a,
        max_col_b,
        max_diag,
    ) {
        Ok(results) => {
            if results.len() != pairs.len() {
                println!("  FAIL: Expected {} results, got {}", pairs.len(), results.len());
                return false;
            }

            for (pair_idx, ((cpu_acc_u, cpu_acc_v, cpu_diag), gpu_result)) in
                cpu_results.iter().zip(results.iter()).enumerate()
            {
                let (i, j) = pairs[pair_idx];

                let gpu_acc_u: Vec<MNT4Fr> = gpu_result.acc_u.iter().map(limbs_to_fr).collect();
                let gpu_acc_v: Vec<MNT4Fr> = gpu_result.acc_v.iter().map(limbs_to_fr).collect();

                if !vecs_equal(cpu_acc_u, &gpu_acc_u) {
                    println!("  FAIL: Pair ({}, {}) acc_u mismatch", i, j);
                    println!("    CPU: {:?}", cpu_acc_u);
                    println!("    GPU: {:?}", gpu_acc_u);
                    return false;
                }

                if !vecs_equal(cpu_acc_v, &gpu_acc_v) {
                    println!("  FAIL: Pair ({}, {}) acc_v mismatch", i, j);
                    println!("    CPU: {:?}", cpu_acc_v);
                    println!("    GPU: {:?}", gpu_acc_v);
                    return false;
                }

                if !diag_terms_equal(cpu_diag, &gpu_result.diag_terms) {
                    println!("  FAIL: Pair ({}, {}) diagonal terms mismatch", i, j);
                    return false;
                }
            }

            println!("  PASS: All {} pairs match CPU reference", pairs.len());
            true
        }
        Err(e) => {
            println!("  FAIL: GPU error {:?}", e);
            false
        }
    }
}

fn test_random_sparse_small(seed: u64) -> bool {
    println!("\n=== Test: Random Sparse (Small, seed={}) ===", seed);

    let mut rng = StdRng::seed_from_u64(seed);
    let domain_size = 32usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Generate random sparse columns
    let num_cols_a = 4;
    let num_cols_b = 3;
    let max_nnz_per_col = 5;

    let columns_a: Vec<Vec<(usize, MNT4Fr)>> = (0..num_cols_a)
        .map(|_| {
            let nnz = rng.gen_range(1..=max_nnz_per_col);
            let mut rows: Vec<usize> = (0..domain_size).collect();
            rows.shuffle(&mut rng);
            rows.truncate(nnz);
            rows.sort();
            rows.iter()
                .map(|&r| (r, MNT4Fr::rand(&mut rng)))
                .collect()
        })
        .collect();

    let columns_b: Vec<Vec<(usize, MNT4Fr)>> = (0..num_cols_b)
        .map(|_| {
            let nnz = rng.gen_range(1..=max_nnz_per_col);
            let mut rows: Vec<usize> = (0..domain_size).collect();
            rows.shuffle(&mut rng);
            rows.truncate(nnz);
            rows.sort();
            rows.iter()
                .map(|&r| (r, MNT4Fr::rand(&mut rng)))
                .collect()
        })
        .collect();

    // Generate some random pairs
    let num_pairs = 6;
    let pairs: Vec<(u32, u32)> = (0..num_pairs)
        .map(|_| {
            (
                rng.gen_range(0..num_cols_a) as u32,
                rng.gen_range(0..num_cols_b) as u32,
            )
        })
        .collect();

    // Compute CPU references
    let cpu_results: Vec<_> = pairs
        .iter()
        .map(|&(i, j)| {
            cpu_sparse_quotient_coeffs(
                &columns_a[i as usize],
                &columns_b[j as usize],
                &domain_elements,
                &inv_domain_elements,
                &inv_n_one_minus_omega,
                domain_size,
            )
        })
        .collect();

    // GPU computation
    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let max_col_a = columns_a.iter().map(|c| c.len()).max().unwrap_or(1) as u32;
    let max_col_b = columns_b.iter().map(|c| c.len()).max().unwrap_or(1) as u32;
    let max_diag = std::cmp::max(max_col_a, max_col_b);

    match compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a,
        &col_b,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        max_col_a,
        max_col_b,
        max_diag,
    ) {
        Ok(results) => {
            for (pair_idx, ((cpu_acc_u, cpu_acc_v, cpu_diag), gpu_result)) in
                cpu_results.iter().zip(results.iter()).enumerate()
            {
                let (i, j) = pairs[pair_idx];

                let gpu_acc_u: Vec<MNT4Fr> = gpu_result.acc_u.iter().map(limbs_to_fr).collect();
                let gpu_acc_v: Vec<MNT4Fr> = gpu_result.acc_v.iter().map(limbs_to_fr).collect();

                if !vecs_equal(cpu_acc_u, &gpu_acc_u) {
                    println!("  FAIL: Pair ({}, {}) acc_u mismatch", i, j);
                    return false;
                }

                if !vecs_equal(cpu_acc_v, &gpu_acc_v) {
                    println!("  FAIL: Pair ({}, {}) acc_v mismatch", i, j);
                    return false;
                }

                if !diag_terms_equal(cpu_diag, &gpu_result.diag_terms) {
                    println!("  FAIL: Pair ({}, {}) diagonal terms mismatch", i, j);
                    return false;
                }
            }

            println!("  PASS: {} random pairs match CPU reference", pairs.len());
            true
        }
        Err(e) => {
            println!("  FAIL: GPU error {:?}", e);
            false
        }
    }
}

fn test_random_sparse_medium(seed: u64) -> bool {
    println!("\n=== Test: Random Sparse (Medium, seed={}) ===", seed);

    let mut rng = StdRng::seed_from_u64(seed);
    let domain_size = 64usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Generate random sparse columns with more density
    let num_cols_a = 8;
    let num_cols_b = 6;
    let max_nnz_per_col = 12;

    let columns_a: Vec<Vec<(usize, MNT4Fr)>> = (0..num_cols_a)
        .map(|_| {
            let nnz = rng.gen_range(2..=max_nnz_per_col);
            let mut rows: Vec<usize> = (0..domain_size).collect();
            rows.shuffle(&mut rng);
            rows.truncate(nnz);
            rows.sort();
            rows.iter()
                .map(|&r| (r, MNT4Fr::rand(&mut rng)))
                .collect()
        })
        .collect();

    let columns_b: Vec<Vec<(usize, MNT4Fr)>> = (0..num_cols_b)
        .map(|_| {
            let nnz = rng.gen_range(2..=max_nnz_per_col);
            let mut rows: Vec<usize> = (0..domain_size).collect();
            rows.shuffle(&mut rng);
            rows.truncate(nnz);
            rows.sort();
            rows.iter()
                .map(|&r| (r, MNT4Fr::rand(&mut rng)))
                .collect()
        })
        .collect();

    // Generate pairs
    let num_pairs = 16;
    let pairs: Vec<(u32, u32)> = (0..num_pairs)
        .map(|_| {
            (
                rng.gen_range(0..num_cols_a) as u32,
                rng.gen_range(0..num_cols_b) as u32,
            )
        })
        .collect();

    // Compute CPU references
    let cpu_results: Vec<_> = pairs
        .iter()
        .map(|&(i, j)| {
            cpu_sparse_quotient_coeffs(
                &columns_a[i as usize],
                &columns_b[j as usize],
                &domain_elements,
                &inv_domain_elements,
                &inv_n_one_minus_omega,
                domain_size,
            )
        })
        .collect();

    // GPU computation
    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let max_col_a = columns_a.iter().map(|c| c.len()).max().unwrap_or(1) as u32;
    let max_col_b = columns_b.iter().map(|c| c.len()).max().unwrap_or(1) as u32;
    let max_diag = std::cmp::max(max_col_a, max_col_b);

    match compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a,
        &col_b,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        max_col_a,
        max_col_b,
        max_diag,
    ) {
        Ok(results) => {
            let mut mismatches = 0;
            for (pair_idx, ((cpu_acc_u, cpu_acc_v, cpu_diag), gpu_result)) in
                cpu_results.iter().zip(results.iter()).enumerate()
            {
                let (i, j) = pairs[pair_idx];

                let gpu_acc_u: Vec<MNT4Fr> = gpu_result.acc_u.iter().map(limbs_to_fr).collect();
                let gpu_acc_v: Vec<MNT4Fr> = gpu_result.acc_v.iter().map(limbs_to_fr).collect();

                if !vecs_equal(cpu_acc_u, &gpu_acc_u)
                    || !vecs_equal(cpu_acc_v, &gpu_acc_v)
                    || !diag_terms_equal(cpu_diag, &gpu_result.diag_terms)
                {
                    mismatches += 1;
                    if mismatches <= 3 {
                        println!("  Pair ({}, {}) mismatch", i, j);
                    }
                }
            }

            if mismatches == 0 {
                println!("  PASS: {} random pairs match CPU reference", pairs.len());
                true
            } else {
                println!("  FAIL: {} of {} pairs had mismatches", mismatches, pairs.len());
                false
            }
        }
        Err(e) => {
            println!("  FAIL: GPU error {:?}", e);
            false
        }
    }
}

fn test_large_sparse_columns(seed: u64) -> bool {
    println!("\n=== Test: Large Sparse Columns (seed={}) ===", seed);

    let mut rng = StdRng::seed_from_u64(seed);
    let domain_size = 128usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Create columns with many entries (stress test for accumulation)
    let nnz_a = 24;
    let nnz_b = 20;

    let mut rows_a: Vec<usize> = (0..domain_size).collect();
    rows_a.shuffle(&mut rng);
    rows_a.truncate(nnz_a);
    rows_a.sort();

    let mut rows_b: Vec<usize> = (0..domain_size).collect();
    rows_b.shuffle(&mut rng);
    rows_b.truncate(nnz_b);
    rows_b.sort();

    let col_a_data: Vec<(usize, MNT4Fr)> = rows_a
        .iter()
        .map(|&r| (r, MNT4Fr::rand(&mut rng)))
        .collect();
    let col_b_data: Vec<(usize, MNT4Fr)> = rows_b
        .iter()
        .map(|&r| (r, MNT4Fr::rand(&mut rng)))
        .collect();

    let columns_a = vec![col_a_data.clone()];
    let columns_b = vec![col_b_data.clone()];

    // CPU reference
    let (cpu_acc_u, cpu_acc_v, cpu_diag) = cpu_sparse_quotient_coeffs(
        &col_a_data,
        &col_b_data,
        &domain_elements,
        &inv_domain_elements,
        &inv_n_one_minus_omega,
        domain_size,
    );

    // GPU computation
    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let pairs = vec![(0u32, 0u32)];
    let max_diag = std::cmp::min(nnz_a, nnz_b) as u32;

    match compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a,
        &col_b,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        nnz_a as u32,
        nnz_b as u32,
        max_diag,
    ) {
        Ok(results) => {
            if results.len() != 1 {
                println!("  FAIL: Expected 1 result");
                return false;
            }

            let result = &results[0];
            let gpu_acc_u: Vec<MNT4Fr> = result.acc_u.iter().map(limbs_to_fr).collect();
            let gpu_acc_v: Vec<MNT4Fr> = result.acc_v.iter().map(limbs_to_fr).collect();

            if !vecs_equal(&cpu_acc_u, &gpu_acc_u) {
                println!("  FAIL: acc_u mismatch ({}x{} combinations)", nnz_a, nnz_b);
                return false;
            }

            if !vecs_equal(&cpu_acc_v, &gpu_acc_v) {
                println!("  FAIL: acc_v mismatch");
                return false;
            }

            if !diag_terms_equal(&cpu_diag, &result.diag_terms) {
                println!("  FAIL: diagonal terms mismatch");
                return false;
            }

            println!(
                "  PASS: Large columns ({}x{}={} combinations) match",
                nnz_a,
                nnz_b,
                nnz_a * nnz_b
            );
            true
        }
        Err(e) => {
            println!("  FAIL: GPU error {:?}", e);
            false
        }
    }
}

fn test_deterministic() -> bool {
    println!("\n=== Test: Deterministic Results ===");

    let domain_size = 32usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Fixed columns
    let columns_a: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![
        (1, MNT4Fr::from(100u64)),
        (5, MNT4Fr::from(200u64)),
        (10, MNT4Fr::from(300u64)),
    ]];
    let columns_b: Vec<Vec<(usize, MNT4Fr)>> = vec![vec![
        (1, MNT4Fr::from(50u64)),
        (7, MNT4Fr::from(150u64)),
    ]];

    let domain_limbs: Vec<_> = domain_elements.iter().map(fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(fr_to_limbs).collect();

    let col_a = build_csr_matrix(&columns_a);
    let col_b = build_csr_matrix(&columns_b);

    let pairs = vec![(0u32, 0u32)];

    // Run multiple times and check results are identical
    let mut first_result: Option<(Vec<MNT4Fr>, Vec<MNT4Fr>, Vec<(u32, [u32; 10])>)> = None;

    for run in 0..3 {
        match compute_sparse_quotient_coeffs_mnt4_298_gpu(
            &col_a,
            &col_b,
            &domain_limbs,
            &inv_domain_limbs,
            &inv_n_limbs,
            domain_size as u32,
            &pairs,
            3,
            2,
            2,
        ) {
            Ok(results) => {
                let result = &results[0];
                let gpu_acc_u: Vec<MNT4Fr> = result.acc_u.iter().map(limbs_to_fr).collect();
                let gpu_acc_v: Vec<MNT4Fr> = result.acc_v.iter().map(limbs_to_fr).collect();

                match &first_result {
                    None => {
                        first_result = Some((gpu_acc_u, gpu_acc_v, result.diag_terms.clone()));
                    }
                    Some((ref first_u, ref first_v, ref first_diag)) => {
                        if !vecs_equal(first_u, &gpu_acc_u)
                            || !vecs_equal(first_v, &gpu_acc_v)
                            || first_diag != &result.diag_terms
                        {
                            println!("  FAIL: Run {} differs from run 0", run);
                            return false;
                        }
                    }
                }
            }
            Err(e) => {
                println!("  FAIL: GPU error on run {}: {:?}", run, e);
                return false;
            }
        }
    }

    println!("  PASS: Results are deterministic across 3 runs");
    true
}

// ============================================================================
// CPU Reference Tests (Always Run)
// ============================================================================

fn test_cpu_reference_diagonal() -> bool {
    println!("\n=== Test: CPU Reference - Diagonal Case ===");

    let domain_size = 8usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Both columns have row 2, so k==m diagonal case
    let col_a = vec![(2, MNT4Fr::from(3u64))];
    let col_b = vec![(2, MNT4Fr::from(5u64))];

    let (acc_u, acc_v, diag) = cpu_sparse_quotient_coeffs(
        &col_a,
        &col_b,
        &domain_elements,
        &inv_domain_elements,
        &inv_n_one_minus_omega,
        domain_size,
    );

    // Check acc_u and acc_v are zero (only diagonal, no off-diagonal)
    if !acc_u.iter().all(|x| x.is_zero()) {
        println!("  FAIL: acc_u should be zero");
        return false;
    }
    if !acc_v.iter().all(|x| x.is_zero()) {
        println!("  FAIL: acc_v should be zero");
        return false;
    }

    // Check diagonal: k=2, value=3*5=15
    if diag.len() != 1 || diag[0].0 != 2 || diag[0].1 != MNT4Fr::from(15u64) {
        println!("  FAIL: Expected diagonal (2, 15), got {:?}", diag);
        return false;
    }

    println!("  PASS: CPU reference produces correct diagonal term");
    true
}

fn test_cpu_reference_off_diagonal() -> bool {
    println!("\n=== Test: CPU Reference - Off-Diagonal Case ===");

    let domain_size = 8usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Different rows, so k!=m off-diagonal case
    let col_a = vec![(1, MNT4Fr::from(2u64))];
    let col_b = vec![(3, MNT4Fr::from(4u64))];

    let (acc_u, acc_v, diag) = cpu_sparse_quotient_coeffs(
        &col_a,
        &col_b,
        &domain_elements,
        &inv_domain_elements,
        &inv_n_one_minus_omega,
        domain_size,
    );

    // No diagonal terms expected
    if !diag.is_empty() {
        println!("  FAIL: Expected no diagonal terms, got {:?}", diag);
        return false;
    }

    // acc_u and acc_v should have non-zero values
    if acc_u.iter().all(|x| x.is_zero()) && acc_v.iter().all(|x| x.is_zero()) {
        println!("  FAIL: Expected non-zero accumulator values");
        return false;
    }

    println!("  PASS: CPU reference produces correct off-diagonal coefficients");
    true
}

fn test_cpu_reference_mixed() -> bool {
    println!("\n=== Test: CPU Reference - Mixed Case ===");

    let domain_size = 16usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Mix of diagonal and off-diagonal
    let col_a = vec![
        (1, MNT4Fr::from(2u64)),
        (3, MNT4Fr::from(3u64)),  // will match with col_b's row 3
        (5, MNT4Fr::from(5u64)),
    ];
    let col_b = vec![
        (3, MNT4Fr::from(7u64)),  // diagonal with col_a's row 3
        (7, MNT4Fr::from(11u64)),
    ];

    let (acc_u, acc_v, diag) = cpu_sparse_quotient_coeffs(
        &col_a,
        &col_b,
        &domain_elements,
        &inv_domain_elements,
        &inv_n_one_minus_omega,
        domain_size,
    );

    // Should have 1 diagonal: k=3, value=3*7=21
    if diag.len() != 1 {
        println!("  FAIL: Expected 1 diagonal term, got {}", diag.len());
        return false;
    }
    if diag[0].0 != 3 || diag[0].1 != MNT4Fr::from(21u64) {
        println!("  FAIL: Expected diagonal (3, 21), got {:?}", diag);
        return false;
    }

    // Should have non-zero accumulators from off-diagonal terms
    // (1,3), (1,7), (3,7), (5,3), (5,7) are off-diagonal
    let non_zero_u: usize = acc_u.iter().filter(|x| !x.is_zero()).count();
    let non_zero_v: usize = acc_v.iter().filter(|x| !x.is_zero()).count();

    if non_zero_u == 0 && non_zero_v == 0 {
        println!("  FAIL: Expected some non-zero accumulator values");
        return false;
    }

    println!("  PASS: CPU reference handles mixed case correctly");
    println!("    Diagonal terms: {}", diag.len());
    println!("    Non-zero acc_u: {}, Non-zero acc_v: {}", non_zero_u, non_zero_v);
    true
}

fn test_cpu_reference_associativity() -> bool {
    println!("\n=== Test: CPU Reference - Coefficient Accumulation ===");

    let domain_size = 8usize;
    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_domain_tables(domain_size);

    // Multiple entries hitting the same accumulator indices
    let col_a = vec![
        (1, MNT4Fr::from(1u64)),
        (2, MNT4Fr::from(1u64)),
    ];
    let col_b = vec![
        (3, MNT4Fr::from(1u64)),
        (4, MNT4Fr::from(1u64)),
    ];

    let (acc_u, acc_v, diag) = cpu_sparse_quotient_coeffs(
        &col_a,
        &col_b,
        &domain_elements,
        &inv_domain_elements,
        &inv_n_one_minus_omega,
        domain_size,
    );

    // No diagonal (all k != m)
    if !diag.is_empty() {
        println!("  FAIL: Expected no diagonal terms");
        return false;
    }

    // Each acc_u[i] should have contributions from 2 off-diagonal pairs
    // Each acc_v[j] should have contributions from 2 off-diagonal pairs
    // Verify they accumulate correctly (non-zero in general)
    let has_accumulation = acc_u.iter().any(|x| !x.is_zero()) || acc_v.iter().any(|x| !x.is_zero());
    if !has_accumulation {
        println!("  FAIL: Expected accumulated coefficients");
        return false;
    }

    println!("  PASS: CPU reference correctly accumulates coefficients");
    true
}

// ============================================================================
// Main Test Runner
// ============================================================================

fn main() {
    println!("==============================================");
    println!("Sparse Quotient GPU Kernel Tests (MNT4-298)");
    println!("==============================================");
    println!("");
    println!("Testing GPU kernel with fixed __shfl_sync");
    println!("(now uses instance-scoped masks with width=TPI)");
    println!("==============================================");

    let mut passed = 0;
    let mut failed = 0;

    // Run CPU reference tests first (these don't call the GPU kernel)
    println!("\n--- CPU Reference Implementation Tests ---");

    let cpu_tests: Vec<(&str, fn() -> bool)> = vec![
        ("CPU Reference Diagonal", test_cpu_reference_diagonal),
        ("CPU Reference Off-Diagonal", test_cpu_reference_off_diagonal),
        ("CPU Reference Mixed", test_cpu_reference_mixed),
        ("CPU Reference Accumulation", test_cpu_reference_associativity),
    ];

    for (name, test_fn) in cpu_tests {
        if test_fn() {
            passed += 1;
        } else {
            failed += 1;
            println!("  FAILED: {}", name);
        }
    }

    // Check GPU availability
    println!("\n--- GPU Kernel Tests ---");
    if !test_gpu_available() {
        println!("\nGPU kernel not available - skipping GPU tests");
    } else {
        passed += 1; // GPU available check passed

        // Basic functionality tests
        let tests: Vec<(&str, fn() -> bool)> = vec![
            ("Empty Pairs", test_empty_pairs),
            ("Single Pair Diagonal Only", test_single_pair_diagonal_only),
            ("Single Pair Off-Diagonal Only", test_single_pair_off_diagonal_only),
            ("Single Pair Mixed", test_single_pair_mixed),
            ("Multiple Pairs Batch", test_multiple_pairs_batch),
            ("Deterministic", test_deterministic),
        ];

        for (name, test_fn) in tests {
            if test_fn() {
                passed += 1;
            } else {
                failed += 1;
                println!("  FAILED: {}", name);
            }
        }

        // Random tests with different seeds
        let random_tests: Vec<(&str, fn(u64) -> bool, Vec<u64>)> = vec![
            ("Random Small", test_random_sparse_small, vec![42, 123, 456]),
            ("Random Medium", test_random_sparse_medium, vec![789, 1011, 1213]),
            ("Large Columns", test_large_sparse_columns, vec![2024, 2025]),
        ];

        for (name, test_fn, seeds) in random_tests {
            for seed in seeds {
                if test_fn(seed) {
                    passed += 1;
                } else {
                    failed += 1;
                    println!("  FAILED: {} (seed={})", name, seed);
                }
            }
        }
    }

    println!("\n==============================================");
    println!("Results: {} passed, {} failed", passed, failed);
    println!("==============================================");

    if failed > 0 {
        std::process::exit(1);
    }
}
