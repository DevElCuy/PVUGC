//! GPU Scalability Benchmark Test
//!
//! This benchmark generates structured output for tracking GPU performance
//! across different scales. All curves run sequentially at each scale level
//! (no parallel execution).
//!
//! ## Curve Support Status
//!
//! | Curve | Sparse Quotient GPU | MSM GPU | Notes |
//! |-------|---------------------|---------|-------|
//! | MNT4-298 | Yes | Yes | Full support |
//! | MNT6-298 | Yes | Yes | Full support |
//! | BW6-761 | Yes | Yes | Full support (377-bit scalar field) |
//!
//! ## Usage
//!
//! ```bash
//! # Run with default scales (100, 1K, 10K pairs) - all curves at each scale
//! cargo test --features gpu --release --test benchmark_gpu_scalability -- --nocapture
//!
//! # Run specific scale only
//! BENCHMARK_PAIRS=50000 cargo test --features gpu --release --test benchmark_gpu_scalability -- --nocapture
//!
//! # Run specific curve only (at all scales)
//! BENCHMARK_CURVE=mnt4 cargo test --features gpu --release --test benchmark_gpu_scalability -- --nocapture
//! BENCHMARK_CURVE=mnt6 cargo test --features gpu --release --test benchmark_gpu_scalability -- --nocapture
//! BENCHMARK_CURVE=bw6 cargo test --features gpu --release --test benchmark_gpu_scalability -- --nocapture
//! ```
//!
//! ## Output Order
//!
//! For each scale level (e.g., 100 pairs), the benchmark runs:
//! 1. MNT4-298
//! 2. MNT6-298
//! 3. BW6-761
//!
//! Then moves to the next scale level. This ensures clean, sequential output.

#![cfg(all(feature = "gpu", sparse_quotient_mnt4_available))]

use ark_ff::{AdditiveGroup, BigInt, FftField, One, PrimeField, UniformRand};
use ark_mnt4_298::Fr as MNT4Fr;
use ark_mnt6_298::Fr as MNT6Fr;
use ark_bw6_761::Fr as BW6Fr;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use sppark_msm::{
    compute_sparse_quotient_coeffs_mnt4_298_gpu,
    compute_sparse_quotient_coeffs_mnt6_298_gpu,
    compute_sparse_quotient_coeffs_bw6_761_gpu,
    get_gpu_available_memory, get_gpu_target_memory,
    get_gpu_total_memory,
    sparse_quotient_gpu_available,
    sparse_quotient_mnt6_gpu_available,
    sparse_quotient_bw6_gpu_available,
    SparseMatrixCsr,
    SparseMatrixCsrBw6,
};
use std::time::Instant;

// ============================================================================
// MNT4/MNT6 Helper Functions (298-bit, 10 limbs)
// ============================================================================

/// Convert an arkworks MNT4 Fr element to u32[10] limbs (little-endian)
fn mnt4_fr_to_limbs(f: &MNT4Fr) -> [u32; 10] {
    let bigint: BigInt<5> = f.into_bigint();
    let mut limbs = [0u32; 10];
    for (i, &limb64) in bigint.0.iter().enumerate() {
        limbs[2 * i] = limb64 as u32;
        limbs[2 * i + 1] = (limb64 >> 32) as u32;
    }
    limbs
}

/// Convert an arkworks MNT6 Fr element to u32[10] limbs (little-endian)
fn mnt6_fr_to_limbs(f: &MNT6Fr) -> [u32; 10] {
    let bigint: BigInt<5> = f.into_bigint();
    let mut limbs = [0u32; 10];
    for (i, &limb64) in bigint.0.iter().enumerate() {
        limbs[2 * i] = limb64 as u32;
        limbs[2 * i + 1] = (limb64 >> 32) as u32;
    }
    limbs
}

// ============================================================================
// BW6-761 Helper Functions (377-bit scalar field, 12 limbs)
// ============================================================================

/// Convert an arkworks BW6 Fr element to u32[12] limbs (little-endian)
fn bw6_fr_to_limbs(f: &BW6Fr) -> [u32; 12] {
    let bigint: BigInt<6> = f.into_bigint();
    let mut limbs = [0u32; 12];
    for (i, &limb64) in bigint.0.iter().enumerate() {
        limbs[2 * i] = limb64 as u32;
        limbs[2 * i + 1] = (limb64 >> 32) as u32;
    }
    limbs
}

// ============================================================================
// Domain Table Builders
// ============================================================================

/// Build MNT4 evaluation domain tables
fn build_mnt4_domain_tables(domain_size: usize) -> (Vec<MNT4Fr>, Vec<MNT4Fr>, Vec<MNT4Fr>) {
    let omega = MNT4Fr::get_root_of_unity(domain_size as u64).expect("Domain size must be power of 2");
    let n_field = MNT4Fr::from(domain_size as u64);

    let mut domain_elements = Vec::with_capacity(domain_size);
    let mut current = MNT4Fr::one();
    for _ in 0..domain_size {
        domain_elements.push(current);
        current *= omega;
    }

    let mut inv_domain_elements = vec![MNT4Fr::one(); domain_size];
    for i in 1..domain_size {
        inv_domain_elements[i] = domain_elements[domain_size - i];
    }

    let mut inv_n_one_minus_omega = vec![MNT4Fr::ZERO; domain_size];
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

/// Build MNT6 evaluation domain tables
fn build_mnt6_domain_tables(domain_size: usize) -> (Vec<MNT6Fr>, Vec<MNT6Fr>, Vec<MNT6Fr>) {
    let omega = MNT6Fr::get_root_of_unity(domain_size as u64).expect("Domain size must be power of 2");
    let n_field = MNT6Fr::from(domain_size as u64);

    let mut domain_elements = Vec::with_capacity(domain_size);
    let mut current = MNT6Fr::one();
    for _ in 0..domain_size {
        domain_elements.push(current);
        current *= omega;
    }

    let mut inv_domain_elements = vec![MNT6Fr::one(); domain_size];
    for i in 1..domain_size {
        inv_domain_elements[i] = domain_elements[domain_size - i];
    }

    let mut inv_n_one_minus_omega = vec![MNT6Fr::ZERO; domain_size];
    let mut denoms = Vec::with_capacity(domain_size - 1);
    let mut indices = Vec::with_capacity(domain_size - 1);

    for d in 1..domain_size {
        let denom = n_field * (MNT6Fr::one() - domain_elements[d]);
        denoms.push(denom);
        indices.push(d);
    }

    ark_ff::batch_inversion(&mut denoms);
    for (i, &d) in indices.iter().enumerate() {
        inv_n_one_minus_omega[d] = denoms[i];
    }

    (domain_elements, inv_domain_elements, inv_n_one_minus_omega)
}

/// Build BW6-761 evaluation domain tables
fn build_bw6_domain_tables(domain_size: usize) -> (Vec<BW6Fr>, Vec<BW6Fr>, Vec<BW6Fr>) {
    let omega = BW6Fr::get_root_of_unity(domain_size as u64).expect("Domain size must be power of 2");
    let n_field = BW6Fr::from(domain_size as u64);

    let mut domain_elements = Vec::with_capacity(domain_size);
    let mut current = BW6Fr::one();
    for _ in 0..domain_size {
        domain_elements.push(current);
        current *= omega;
    }

    let mut inv_domain_elements = vec![BW6Fr::one(); domain_size];
    for i in 1..domain_size {
        inv_domain_elements[i] = domain_elements[domain_size - i];
    }

    let mut inv_n_one_minus_omega = vec![BW6Fr::ZERO; domain_size];
    let mut denoms = Vec::with_capacity(domain_size - 1);
    let mut indices = Vec::with_capacity(domain_size - 1);

    for d in 1..domain_size {
        let denom = n_field * (BW6Fr::one() - domain_elements[d]);
        denoms.push(denom);
        indices.push(d);
    }

    ark_ff::batch_inversion(&mut denoms);
    for (i, &d) in indices.iter().enumerate() {
        inv_n_one_minus_omega[d] = denoms[i];
    }

    (domain_elements, inv_domain_elements, inv_n_one_minus_omega)
}

// ============================================================================
// CSR Matrix Builders
// ============================================================================

/// Build CSR matrix for MNT4
fn build_mnt4_csr_matrix(columns: &[Vec<(usize, MNT4Fr)>]) -> SparseMatrixCsr {
    let num_cols = columns.len();
    let mut col_ptr = Vec::with_capacity(num_cols + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    col_ptr.push(0u32);
    for col in columns {
        for &(r, v) in col {
            row_idx.push(r as u32);
            values.push(mnt4_fr_to_limbs(&v));
        }
        col_ptr.push(row_idx.len() as u32);
    }

    SparseMatrixCsr { col_ptr, row_idx, values }
}

/// Build CSR matrix for MNT6
fn build_mnt6_csr_matrix(columns: &[Vec<(usize, MNT6Fr)>]) -> SparseMatrixCsr {
    let num_cols = columns.len();
    let mut col_ptr = Vec::with_capacity(num_cols + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    col_ptr.push(0u32);
    for col in columns {
        for &(r, v) in col {
            row_idx.push(r as u32);
            values.push(mnt6_fr_to_limbs(&v));
        }
        col_ptr.push(row_idx.len() as u32);
    }

    SparseMatrixCsr { col_ptr, row_idx, values }
}

/// Build CSR matrix for BW6-761
fn build_bw6_csr_matrix(columns: &[Vec<(usize, BW6Fr)>]) -> SparseMatrixCsrBw6 {
    let num_cols = columns.len();
    let mut col_ptr = Vec::with_capacity(num_cols + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    col_ptr.push(0u32);
    for col in columns {
        for &(r, v) in col {
            row_idx.push(r as u32);
            values.push(bw6_fr_to_limbs(&v));
        }
        col_ptr.push(row_idx.len() as u32);
    }

    SparseMatrixCsrBw6 { col_ptr, row_idx, values }
}

// ============================================================================
// Benchmark Result Structure
// ============================================================================

#[derive(Debug, Clone)]
struct BenchmarkResult {
    curve: String,
    num_pairs: usize,
    num_columns_a: usize,
    num_columns_b: usize,
    nnz_per_col: usize,
    domain_size: usize,
    bytes_per_pair: usize,
    gpu_memory_total_mb: usize,
    gpu_memory_available_mb: usize,
    time_ms: u128,
    throughput_pairs_per_sec: f64,
}

impl BenchmarkResult {
    fn print_structured(&self) {
        println!("\n=== GPU Scalability Benchmark Result ===");
        println!("curve: {}", self.curve);
        println!("num_pairs: {}", self.num_pairs);
        println!("num_columns_a: {}", self.num_columns_a);
        println!("num_columns_b: {}", self.num_columns_b);
        println!("nnz_per_col: {}", self.nnz_per_col);
        println!("domain_size: {}", self.domain_size);
        println!("bytes_per_pair: {}", self.bytes_per_pair);
        println!("gpu_memory_total_mb: {}", self.gpu_memory_total_mb);
        println!("gpu_memory_available_mb: {}", self.gpu_memory_available_mb);
        println!("time_ms: {}", self.time_ms);
        println!("throughput_pairs_per_sec: {:.2}", self.throughput_pairs_per_sec);
        println!("=========================================\n");
    }

    fn print_json(&self) {
        println!(
            r#"{{"curve":"{}","num_pairs":{},"num_columns_a":{},"num_columns_b":{},"nnz_per_col":{},"domain_size":{},"bytes_per_pair":{},"gpu_memory_total_mb":{},"gpu_memory_available_mb":{},"time_ms":{},"throughput_pairs_per_sec":{:.2}}}"#,
            self.curve,
            self.num_pairs,
            self.num_columns_a,
            self.num_columns_b,
            self.nnz_per_col,
            self.domain_size,
            self.bytes_per_pair,
            self.gpu_memory_total_mb,
            self.gpu_memory_available_mb,
            self.time_ms,
            self.throughput_pairs_per_sec
        );
    }
}

// ============================================================================
// Projection Helper
// ============================================================================

fn print_projections(results: &[BenchmarkResult]) {
    if results.is_empty() {
        return;
    }

    // Group results by curve
    let mnt_results: Vec<_> = results.iter().filter(|r| r.curve.starts_with("MNT")).collect();
    let bw6_results: Vec<_> = results.iter().filter(|r| r.curve.starts_with("BW6")).collect();

    let mnt_throughput = mnt_results.iter()
        .map(|r| r.throughput_pairs_per_sec)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let bw6_throughput = bw6_results.iter()
        .map(|r| r.throughput_pairs_per_sec)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(mnt_throughput / 4.0);  // Estimate if not measured

    println!("\n=== Projections for Full Circuits ===");
    if mnt_throughput > 0.0 {
        println!("Measured MNT4/MNT6 throughput: {:.0} pairs/sec", mnt_throughput);
    }
    if !bw6_results.is_empty() {
        println!("Measured BW6-761 throughput: {:.0} pairs/sec", bw6_throughput);
    } else {
        println!("Estimated BW6-761 throughput: {:.0} pairs/sec (4x slowdown from MNT)", bw6_throughput);
    }
    println!();

    let scenarios: [(&str, u64, bool); 5] = [
        ("MNT4/MNT6 Full (680M pairs)", 680_000_000, false),
        ("MNT4/MNT6 Skip Verifier (160M pairs)", 160_000_000, false),
        ("BW6-761 Full (~2.5B pairs)", 2_500_000_000, true),
        ("BW6-761 Skip Verifier (~600M pairs)", 600_000_000, true),
        ("Benchmark scale (1M pairs)", 1_000_000, false),
    ];

    println!(
        "{:<42} {:>12} {:>15} {:>10}",
        "Scenario", "Est. Time", "Pairs", "Curve"
    );
    println!("{}", "-".repeat(82));

    for (name, pairs, is_bw6) in scenarios {
        let throughput = if is_bw6 { bw6_throughput } else { mnt_throughput };
        if throughput <= 0.0 { continue; }
        let curve = if is_bw6 { "BW6" } else { "MNT" };
        let seconds = pairs as f64 / throughput;
        let time_str = if seconds < 60.0 {
            format!("{:.1} sec", seconds)
        } else if seconds < 3600.0 {
            format!("{:.1} min", seconds / 60.0)
        } else {
            format!("{:.1} hours", seconds / 3600.0)
        };
        println!("{:<42} {:>12} {:>15} {:>10}", name, time_str, pairs, curve);
    }

    println!();
    println!("Notes:");
    println!("  - MNT projections use measured throughput directly");
    println!("  - BW6 uses measured throughput if available, else 4x slowdown estimate");
    println!("  - Actual times may vary due to memory bandwidth and thermal throttling");
}

// ============================================================================
// Benchmark Runners
// ============================================================================

fn run_mnt4_benchmark(num_pairs: usize, nnz_per_col: usize, domain_size: usize) -> Option<BenchmarkResult> {
    if !sparse_quotient_gpu_available() {
        println!("  MNT4 GPU sparse quotient kernel not available");
        return None;
    }

    let mut rng = StdRng::seed_from_u64(0xBEEF);

    let num_columns = ((num_pairs as f64).sqrt().ceil() as usize).max(1);
    let num_columns_a = num_columns;
    let num_columns_b = num_columns;

    println!(
        "  [MNT4-298] Setting up {} pairs with {}x{} columns, {} entries/col, domain={}...",
        num_pairs, num_columns_a, num_columns_b, nnz_per_col, domain_size
    );

    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_mnt4_domain_tables(domain_size);

    let mut columns_a: Vec<Vec<(usize, MNT4Fr)>> = Vec::with_capacity(num_columns_a);
    for _ in 0..num_columns_a {
        let mut col = Vec::with_capacity(nnz_per_col);
        for row in 0..nnz_per_col {
            let row_idx = row % domain_size;
            let val = MNT4Fr::rand(&mut rng);
            col.push((row_idx, val));
        }
        columns_a.push(col);
    }

    let mut columns_b: Vec<Vec<(usize, MNT4Fr)>> = Vec::with_capacity(num_columns_b);
    for _ in 0..num_columns_b {
        let mut col = Vec::with_capacity(nnz_per_col);
        for row in 0..nnz_per_col {
            let row_idx = row % domain_size;
            let val = MNT4Fr::rand(&mut rng);
            col.push((row_idx, val));
        }
        columns_b.push(col);
    }

    let domain_limbs: Vec<_> = domain_elements.iter().map(mnt4_fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(mnt4_fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(mnt4_fr_to_limbs).collect();

    let col_a_csr = build_mnt4_csr_matrix(&columns_a);
    let col_b_csr = build_mnt4_csr_matrix(&columns_b);

    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(num_pairs);
    'outer: for i in 0..num_columns_a {
        for j in 0..num_columns_b {
            pairs.push((i as u32, j as u32));
            if pairs.len() >= num_pairs {
                break 'outer;
            }
        }
    }
    while pairs.len() < num_pairs {
        pairs.push((0, 0));
    }
    pairs.truncate(num_pairs);

    let max_diag = nnz_per_col;
    let bytes_per_pair = (nnz_per_col * 40) + (nnz_per_col * 40) + (max_diag * 44);
    let gpu_total = get_gpu_total_memory();
    let gpu_available = get_gpu_available_memory();

    println!("  Running GPU kernel...");
    let start = Instant::now();

    let result = compute_sparse_quotient_coeffs_mnt4_298_gpu(
        &col_a_csr,
        &col_b_csr,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        nnz_per_col as u32,
        nnz_per_col as u32,
        max_diag as u32,
    );

    let elapsed = start.elapsed();
    let time_ms = elapsed.as_millis();

    match result {
        Ok(_outputs) => {
            let throughput = if time_ms > 0 {
                (num_pairs as f64) / (time_ms as f64 / 1000.0)
            } else {
                f64::INFINITY
            };

            Some(BenchmarkResult {
                curve: "MNT4-298".to_string(),
                num_pairs,
                num_columns_a,
                num_columns_b,
                nnz_per_col,
                domain_size,
                bytes_per_pair,
                gpu_memory_total_mb: gpu_total / (1024 * 1024),
                gpu_memory_available_mb: gpu_available / (1024 * 1024),
                time_ms,
                throughput_pairs_per_sec: throughput,
            })
        }
        Err(e) => {
            println!("  GPU kernel error: {:?}", e);
            None
        }
    }
}

fn run_mnt6_benchmark(num_pairs: usize, nnz_per_col: usize, domain_size: usize) -> Option<BenchmarkResult> {
    if !sparse_quotient_mnt6_gpu_available() {
        println!("  MNT6 GPU sparse quotient kernel not available");
        return None;
    }

    let mut rng = StdRng::seed_from_u64(0xCAFE);

    let num_columns = ((num_pairs as f64).sqrt().ceil() as usize).max(1);
    let num_columns_a = num_columns;
    let num_columns_b = num_columns;

    println!(
        "  [MNT6-298] Setting up {} pairs with {}x{} columns, {} entries/col, domain={}...",
        num_pairs, num_columns_a, num_columns_b, nnz_per_col, domain_size
    );

    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_mnt6_domain_tables(domain_size);

    let mut columns_a: Vec<Vec<(usize, MNT6Fr)>> = Vec::with_capacity(num_columns_a);
    for _ in 0..num_columns_a {
        let mut col = Vec::with_capacity(nnz_per_col);
        for row in 0..nnz_per_col {
            let row_idx = row % domain_size;
            let val = MNT6Fr::rand(&mut rng);
            col.push((row_idx, val));
        }
        columns_a.push(col);
    }

    let mut columns_b: Vec<Vec<(usize, MNT6Fr)>> = Vec::with_capacity(num_columns_b);
    for _ in 0..num_columns_b {
        let mut col = Vec::with_capacity(nnz_per_col);
        for row in 0..nnz_per_col {
            let row_idx = row % domain_size;
            let val = MNT6Fr::rand(&mut rng);
            col.push((row_idx, val));
        }
        columns_b.push(col);
    }

    let domain_limbs: Vec<_> = domain_elements.iter().map(mnt6_fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(mnt6_fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(mnt6_fr_to_limbs).collect();

    let col_a_csr = build_mnt6_csr_matrix(&columns_a);
    let col_b_csr = build_mnt6_csr_matrix(&columns_b);

    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(num_pairs);
    'outer: for i in 0..num_columns_a {
        for j in 0..num_columns_b {
            pairs.push((i as u32, j as u32));
            if pairs.len() >= num_pairs {
                break 'outer;
            }
        }
    }
    while pairs.len() < num_pairs {
        pairs.push((0, 0));
    }
    pairs.truncate(num_pairs);

    let max_diag = nnz_per_col;
    let bytes_per_pair = (nnz_per_col * 40) + (nnz_per_col * 40) + (max_diag * 44);
    let gpu_total = get_gpu_total_memory();
    let gpu_available = get_gpu_available_memory();

    println!("  Running GPU kernel...");
    let start = Instant::now();

    let result = compute_sparse_quotient_coeffs_mnt6_298_gpu(
        &col_a_csr,
        &col_b_csr,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        nnz_per_col as u32,
        nnz_per_col as u32,
        max_diag as u32,
    );

    let elapsed = start.elapsed();
    let time_ms = elapsed.as_millis();

    match result {
        Ok(_outputs) => {
            let throughput = if time_ms > 0 {
                (num_pairs as f64) / (time_ms as f64 / 1000.0)
            } else {
                f64::INFINITY
            };

            Some(BenchmarkResult {
                curve: "MNT6-298".to_string(),
                num_pairs,
                num_columns_a,
                num_columns_b,
                nnz_per_col,
                domain_size,
                bytes_per_pair,
                gpu_memory_total_mb: gpu_total / (1024 * 1024),
                gpu_memory_available_mb: gpu_available / (1024 * 1024),
                time_ms,
                throughput_pairs_per_sec: throughput,
            })
        }
        Err(e) => {
            println!("  GPU kernel error: {:?}", e);
            None
        }
    }
}

fn run_bw6_benchmark(num_pairs: usize, nnz_per_col: usize, domain_size: usize) -> Option<BenchmarkResult> {
    if !sparse_quotient_bw6_gpu_available() {
        println!("  BW6-761 GPU sparse quotient kernel not available");
        return None;
    }

    let mut rng = StdRng::seed_from_u64(0xFACE);

    let num_columns = ((num_pairs as f64).sqrt().ceil() as usize).max(1);
    let num_columns_a = num_columns;
    let num_columns_b = num_columns;

    println!(
        "  [BW6-761] Setting up {} pairs with {}x{} columns, {} entries/col, domain={}...",
        num_pairs, num_columns_a, num_columns_b, nnz_per_col, domain_size
    );

    let (domain_elements, inv_domain_elements, inv_n_one_minus_omega) =
        build_bw6_domain_tables(domain_size);

    let mut columns_a: Vec<Vec<(usize, BW6Fr)>> = Vec::with_capacity(num_columns_a);
    for _ in 0..num_columns_a {
        let mut col = Vec::with_capacity(nnz_per_col);
        for row in 0..nnz_per_col {
            let row_idx = row % domain_size;
            let val = BW6Fr::rand(&mut rng);
            col.push((row_idx, val));
        }
        columns_a.push(col);
    }

    let mut columns_b: Vec<Vec<(usize, BW6Fr)>> = Vec::with_capacity(num_columns_b);
    for _ in 0..num_columns_b {
        let mut col = Vec::with_capacity(nnz_per_col);
        for row in 0..nnz_per_col {
            let row_idx = row % domain_size;
            let val = BW6Fr::rand(&mut rng);
            col.push((row_idx, val));
        }
        columns_b.push(col);
    }

    let domain_limbs: Vec<_> = domain_elements.iter().map(bw6_fr_to_limbs).collect();
    let inv_domain_limbs: Vec<_> = inv_domain_elements.iter().map(bw6_fr_to_limbs).collect();
    let inv_n_limbs: Vec<_> = inv_n_one_minus_omega.iter().map(bw6_fr_to_limbs).collect();

    let col_a_csr = build_bw6_csr_matrix(&columns_a);
    let col_b_csr = build_bw6_csr_matrix(&columns_b);

    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(num_pairs);
    'outer: for i in 0..num_columns_a {
        for j in 0..num_columns_b {
            pairs.push((i as u32, j as u32));
            if pairs.len() >= num_pairs {
                break 'outer;
            }
        }
    }
    while pairs.len() < num_pairs {
        pairs.push((0, 0));
    }
    pairs.truncate(num_pairs);

    // BW6-761 uses 48 bytes per scalar (12 limbs) instead of 40 bytes (10 limbs)
    let max_diag = nnz_per_col;
    let bytes_per_pair = (nnz_per_col * 48) + (nnz_per_col * 48) + (max_diag * 52);
    let gpu_total = get_gpu_total_memory();
    let gpu_available = get_gpu_available_memory();

    println!("  Running GPU kernel...");
    let start = Instant::now();

    let result = compute_sparse_quotient_coeffs_bw6_761_gpu(
        &col_a_csr,
        &col_b_csr,
        &domain_limbs,
        &inv_domain_limbs,
        &inv_n_limbs,
        domain_size as u32,
        &pairs,
        nnz_per_col as u32,
        nnz_per_col as u32,
        max_diag as u32,
    );

    let elapsed = start.elapsed();
    let time_ms = elapsed.as_millis();

    match result {
        Ok(_outputs) => {
            let throughput = if time_ms > 0 {
                (num_pairs as f64) / (time_ms as f64 / 1000.0)
            } else {
                f64::INFINITY
            };

            Some(BenchmarkResult {
                curve: "BW6-761".to_string(),
                num_pairs,
                num_columns_a,
                num_columns_b,
                nnz_per_col,
                domain_size,
                bytes_per_pair,
                gpu_memory_total_mb: gpu_total / (1024 * 1024),
                gpu_memory_available_mb: gpu_available / (1024 * 1024),
                time_ms,
                throughput_pairs_per_sec: throughput,
            })
        }
        Err(e) => {
            println!("  GPU kernel error: {:?}", e);
            None
        }
    }
}

// ============================================================================
// Single Benchmark Test (runs sequentially, scale-first ordering)
// ============================================================================

/// Main benchmark - runs ALL curves at each scale level sequentially.
///
/// Environment variables:
///   BENCHMARK_PAIRS=N         - Run only N pairs (default: 100, 1K, 10K)
///   BENCHMARK_DOMAIN_SIZE=N   - Set domain size (default: 1024)
///   BENCHMARK_NNZ_PER_COL=N   - Set entries per column (default: 100)
///   BENCHMARK_CURVE=mnt4|mnt6|bw6 - Run only specific curve
///
/// Output order: For each scale level, runs MNT4 -> MNT6 -> BW6, then next scale.
#[test]
fn benchmark_gpu() {
    println!();
    println!("{}", "=".repeat(72));
    println!("  GPU SCALABILITY BENCHMARK");
    println!("  Sequential execution: all curves at each scale, one at a time");
    println!("{}", "=".repeat(72));
    println!();

    // Print GPU info
    let total = get_gpu_total_memory();
    let available = get_gpu_available_memory();
    let target = get_gpu_target_memory();

    println!("GPU Memory:");
    println!("  Total:     {} MB", total / (1024 * 1024));
    println!("  Available: {} MB", available / (1024 * 1024));
    println!("  Target:    {} MB (for batch operations)", target / (1024 * 1024));
    println!();

    println!("Kernel Availability:");
    println!("  MNT4-298: {}", if sparse_quotient_gpu_available() { "AVAILABLE" } else { "NOT AVAILABLE" });
    println!("  MNT6-298: {}", if sparse_quotient_mnt6_gpu_available() { "AVAILABLE" } else { "NOT AVAILABLE" });
    println!("  BW6-761:  {}", if sparse_quotient_bw6_gpu_available() { "AVAILABLE" } else { "NOT AVAILABLE" });
    println!();

    // Determine which curves to test
    let curve_filter = std::env::var("BENCHMARK_CURVE").ok();
    let run_mnt4 = curve_filter.as_ref().map_or(true, |c| c == "mnt4");
    let run_mnt6 = curve_filter.as_ref().map_or(true, |c| c == "mnt6");
    let run_bw6 = curve_filter.as_ref().map_or(true, |c| c == "bw6");

    // Determine scales to test
    let scales: Vec<usize> = if let Ok(pairs_str) = std::env::var("BENCHMARK_PAIRS") {
        vec![pairs_str.parse().unwrap_or(10000)]
    } else {
        vec![100, 1_000, 10_000]
    };

    let domain_size: usize = std::env::var("BENCHMARK_DOMAIN_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let nnz_per_col: usize = std::env::var("BENCHMARK_NNZ_PER_COL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("Test Configuration:");
    println!("  Scales: {:?} pairs", scales);
    println!("  Domain size: {} (power of 2)", domain_size);
    println!("  NNZ per column: {}", nnz_per_col);
    if let Some(ref c) = curve_filter {
        println!("  Curve filter: {}", c);
    }
    println!();

    let mut all_results = Vec::new();

    // Test all curves at each scale before moving to the next scale
    for &num_pairs in &scales {
        println!();
        println!("{}", "=".repeat(72));
        println!("  SCALE: {} pairs", num_pairs);
        println!("{}", "=".repeat(72));

        // MNT4-298
        if run_mnt4 && sparse_quotient_gpu_available() {
            println!();
            println!("--- MNT4-298 @ {} pairs ---", num_pairs);
            if let Some(result) = run_mnt4_benchmark(num_pairs, nnz_per_col, domain_size) {
                result.print_structured();
                all_results.push(result);
            }
        }

        // MNT6-298
        if run_mnt6 && sparse_quotient_mnt6_gpu_available() {
            println!();
            println!("--- MNT6-298 @ {} pairs ---", num_pairs);
            if let Some(result) = run_mnt6_benchmark(num_pairs, nnz_per_col, domain_size) {
                result.print_structured();
                all_results.push(result);
            }
        }

        // BW6-761
        if run_bw6 && sparse_quotient_bw6_gpu_available() {
            println!();
            println!("--- BW6-761 @ {} pairs ---", num_pairs);
            if let Some(result) = run_bw6_benchmark(num_pairs, nnz_per_col, domain_size) {
                result.print_structured();
                all_results.push(result);
            }
        }
    }

    // Print summary
    if !all_results.is_empty() {
        println!();
        println!("{}", "=".repeat(72));
        println!("  SUMMARY");
        println!("{}", "=".repeat(72));
        println!();

        println!(
            "{:<12} {:>12} {:>12} {:>15}",
            "Curve", "Pairs", "Time (ms)", "Pairs/sec"
        );
        println!("{}", "-".repeat(55));

        for r in &all_results {
            println!(
                "{:<12} {:>12} {:>12} {:>15.0}",
                r.curve,
                r.num_pairs,
                r.time_ms,
                r.throughput_pairs_per_sec
            );
        }

        print_projections(&all_results);

        println!("\n--- Machine-Readable Results (for tracking) ---");
        for r in &all_results {
            r.print_json();
        }
    }
}
