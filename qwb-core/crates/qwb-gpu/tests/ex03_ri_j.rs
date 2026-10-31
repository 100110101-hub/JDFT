//! EX-03 验收：RI-J 核与朴素四重循环一致（< 1e-12），并实测加速比。
use std::time::Instant;

use qwb_gpu::ri_j::{ri_j_gemm, ri_j_naive};

fn setup(n: usize, n_aux: usize) -> (Vec<f64>, Vec<f64>) {
    let mut seed = 555_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
    };
    let b: Vec<f64> = (0..n * n * n_aux).map(|_| next()).collect();
    let density: Vec<f64> = (0..n * n).map(|_| next()).collect();
    (b, density)
}

#[test]
fn ri_j_matches_the_naive_contraction() {
    for (n, n_aux) in [(8_usize, 24_usize), (16, 48)] {
        let (b, density) = setup(n, n_aux);
        let reference = ri_j_naive(n, n_aux, &b, &density);
        let result = ri_j_gemm(n, n_aux, &b, &density).expect("ri_j");
        let scale = reference
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
            .max(1.0);
        let worst = reference
            .iter()
            .zip(&result.j)
            .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()));
        println!(
            "n={n} n_aux={n_aux}: used_gpu={} worst |gemm - naive| = {worst:.3e}（scale {scale:.3e}）{}",
            result.used_gpu,
            match &result.fallback_reason {
                Some(why) => format!("，退回 CPU：{why}"),
                None => String::new(),
            }
        );
        assert!(
            worst / scale < 1.0e-12,
            "RI-J 两种收缩不一致：{worst:.3e}（相对 {:.3e}）",
            worst / scale
        );
    }
}

#[test]
fn ri_j_rejects_bad_shapes() {
    assert!(ri_j_gemm(4, 8, &[0.0; 3], &[0.0; 16]).is_err());
    assert!(ri_j_gemm(4, 8, &[0.0; 128], &[0.0; 3]).is_err());
}

#[test]
fn ri_j_reports_a_measured_speedup() {
    let (n, n_aux) = (60_usize, 180_usize);
    let (b, density) = setup(n, n_aux);
    let started = Instant::now();
    let naive = ri_j_naive(n, n_aux, &b, &density);
    let naive_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let kernel = ri_j_gemm(n, n_aux, &b, &density).expect("ri_j");
    let kernel_seconds = started.elapsed().as_secs_f64();
    let scale = naive
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    let worst = naive
        .iter()
        .zip(&kernel.j)
        .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()));
    println!(
        "n={n} n_aux={n_aux}: naive {naive_seconds:.3} s, kernel {kernel_seconds:.3} s, 加速 {:.1}x, worst {worst:.2e}（相对 {:.1e}）",
        naive_seconds / kernel_seconds,
        worst / scale
    );
    assert!(worst / scale < 1.0e-12, "不一致：{:.3e}", worst / scale);
}
