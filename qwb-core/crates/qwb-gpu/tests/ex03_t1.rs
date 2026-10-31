//! EX-03 验收：T1 核与朴素四重循环一致（< 1e-12），并报告实际路径与耗时。
use std::time::Instant;

use qwb_gpu::t1::{t1_gemm, t1_naive};

fn setup(n: usize, n_occ: usize) -> (Vec<f64>, Vec<f64>) {
    let mut seed = 987_654_321_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
    };
    // 正交归一的 C 不必精确：这里只比较「同一输入下两种收缩是否同值」
    let c: Vec<f64> = (0..n * n).map(|_| next()).collect();
    let eri: Vec<f64> = (0..n * n * n * n).map(|_| next() * 0.1).collect();
    let _ = n_occ;
    (c, eri)
}

#[test]
fn t1_gemm_matches_the_naive_contraction() {
    for (n, n_occ) in [(6_usize, 2_usize), (8, 3)] {
        let (c, eri) = setup(n, n_occ);
        let reference = t1_naive(n, n_occ, &c, &eri);
        let result = t1_gemm(n, n_occ, &c, &eri).expect("t1");
        let scale = reference
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
            .max(1.0);
        let worst = reference
            .iter()
            .zip(&result.values)
            .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()));
        println!(
            "n={n} n_occ={n_occ}: path {:?}, worst |gemm - naive| = {worst:.3e} (scale {scale:.3e}){}",
            result.path,
            match &result.fallback_reason {
                Some(why) => format!(", 退回 CPU：{why}"),
                None => String::new(),
            }
        );
        assert!(
            worst / scale < 1.0e-12,
            "T1 两种收缩不一致：{worst:.3e}（相对 {:.3e}）",
            worst / scale
        );
    }
}

#[test]
fn t1_reports_unavailable_instead_of_panicking_on_bad_shapes() {
    assert!(t1_gemm(4, 2, &[0.0; 3], &[0.0; 16]).is_err());
    let c = [0.0_f64; 16];
    let eri = [0.0_f64; 256];
    assert!(t1_gemm(4, 0, &c, &eri).is_err());
}

/// 加速比：同一 T1 用「GPU gemm」与「朴素四重循环」跑，尺寸选到 O(n^4) 有意义的大小。
#[test]
fn t1_reports_a_measured_speedup() {
    let (n, n_occ) = (24_usize, 8_usize);
    let (c, eri) = setup(n, n_occ);
    let started = Instant::now();
    let naive = t1_naive(n, n_occ, &c, &eri);
    let naive_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let kernel = t1_gemm(n, n_occ, &c, &eri).expect("t1");
    let kernel_seconds = started.elapsed().as_secs_f64();
    let scale = naive
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    let worst = naive
        .iter()
        .zip(&kernel.values)
        .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()));
    println!(
        "n={n}: naive {naive_seconds:.3} s, kernel {kernel_seconds:.3} s, 加速 {:.1}x, worst {worst:.2e}（相对 {:.1e}）",
        naive_seconds / kernel_seconds,
        worst / scale
    );
    assert!(worst / scale < 1.0e-12, "不一致：{:.3e}", worst / scale);
}
