//! Test GPU memory detection functionality

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_memory_detection() {
    let available = sppark_msm::get_gpu_available_memory();
    let total = sppark_msm::get_gpu_total_memory();
    let percent = sppark_msm::get_gpu_memory_percent();
    let target = sppark_msm::get_gpu_target_memory();

    println!("\n=== GPU Memory Detection Test ===");
    println!("  Total:     {} MB ({} bytes)", total / (1024 * 1024), total);
    println!("  Available: {} MB ({} bytes)", available / (1024 * 1024), available);
    println!("  Percent:   {}%", percent);
    println!("  Target:    {} MB ({} bytes)", target / (1024 * 1024), target);

    // Basic sanity checks
    assert!(total > 0, "Total GPU memory should be > 0");
    assert!(available > 0, "Available GPU memory should be > 0");
    assert!(available <= total, "Available should be <= total");
    assert!(percent >= 10 && percent <= 95, "Percent should be in [10, 95]");
    assert!(target > 0, "Target memory should be > 0");

    // Target should be roughly percent% of available
    let expected_target = (available * percent as usize) / 100;
    let tolerance = expected_target / 10; // 10% tolerance
    assert!(
        (target as i64 - expected_target as i64).abs() < tolerance as i64,
        "Target ({}) should be close to {}% of available ({})",
        target, percent, expected_target
    );

    println!("  All checks passed!");
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_memory_percent_env_var() {
    // Test default (should be 80)
    let default_percent = sppark_msm::get_gpu_memory_percent();
    println!("\n=== GPU Memory Percent Env Var Test ===");
    println!("  Default percent: {}%", default_percent);

    // Default should be 80 if env var not set
    // Note: This may fail if GPU_MEMORY_PERCENT is set in the environment
    if std::env::var("GPU_MEMORY_PERCENT").is_err() {
        assert_eq!(default_percent, 80, "Default should be 80%");
    }

    println!("  Env var test passed!");
}

#[test]
#[cfg(not(feature = "gpu"))]
fn test_gpu_memory_detection_no_gpu() {
    let available = sppark_msm::get_gpu_available_memory();
    let total = sppark_msm::get_gpu_total_memory();
    let target = sppark_msm::get_gpu_target_memory();

    println!("\n=== GPU Memory Detection (No GPU) Test ===");
    println!("  Total:     {} (expected 0)", total);
    println!("  Available: {} (expected 0)", available);
    println!("  Target:    {} (expected fallback 500MB)", target);

    assert_eq!(available, 0, "Available should be 0 without GPU");
    assert_eq!(total, 0, "Total should be 0 without GPU");
    assert_eq!(target, 500_000_000, "Target should fallback to 500MB");

    println!("  No-GPU checks passed!");
}
