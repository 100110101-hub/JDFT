//! EX-03：cuBLAS 的实测吞吐（`cargo run --release --example bench`）。
//!
//! 只报**实测**数字：GFLOPS = 2·n³ / t。CPU 参考用同一台机器上的 f64 三重循环
//! （单线程、`--release`），因为它是最保守的下界；真实 SCF 里的 CPU 路径用的是
//! `qwb-linalg::matmul`，那是另一个量级的对照，不在这里重复造。
use std::time::Instant;

use qwb_gpu::{available, gemm_f64};

fn naive(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

fn main() {
    match available() {
        Ok(()) => println!("CUDA 可用"),
        Err(err) => {
            println!("CUDA 不可用（退化为 CPU，这是允许的）：{err}");
            return;
        }
    }
    for size in [512_usize, 1024, 2048] {
        let a: Vec<f64> = (0..size * size).map(|i| (i % 17) as f64 * 0.01).collect();
        let b: Vec<f64> = (0..size * size).map(|i| (i % 13) as f64 * 0.01).collect();
        let started = Instant::now();
        let out = gemm_f64(size, size, size, &a, &b).expect("gemm");
        let gpu_seconds = started.elapsed().as_secs_f64();
        let gflops = 2.0 * (size as f64).powi(3) / gpu_seconds / 1e9;
        println!("n = {size:5}: GPU {gpu_seconds:8.4} s  {gflops:7.2} GFLOPS");
        if size <= 512 {
            let started = Instant::now();
            let cpu = naive(size, size, size, &a, &b);
            let cpu_seconds = started.elapsed().as_secs_f64();
            let worst = cpu
                .iter()
                .zip(&out)
                .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()));
            println!(
                "               CPU(1 线程朴素) {cpu_seconds:8.4} s  {:.2} GFLOPS  相对差 {:.2e}  加速 {:.1}x",
                2.0 * (size as f64).powi(3) / cpu_seconds / 1e9,
                worst,
                cpu_seconds / gpu_seconds
            );
        }
    }
}
