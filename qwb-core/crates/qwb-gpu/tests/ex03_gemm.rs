//! EX-03 验收：cuBLAS 与 CPU 逐元素一致（1e-12 相对），且缺 GPU 时优雅退化。
use qwb_gpu::{GpuError, available, gemm_f64};

fn naive_gemm(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            if a_ip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

#[test]
fn gemm_matches_the_cpu_reference_or_reports_unavailable() {
    let (m, k, n) = (64_usize, 48_usize, 40_usize);
    // 确定性伪随机（不引 rand）
    let mut seed = 12345_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
    };
    let a: Vec<f64> = (0..m * k).map(|_| next()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| next()).collect();

    match gemm_f64(m, k, n, &a, &b) {
        Ok(gpu) => {
            let cpu = naive_gemm(m, k, n, &a, &b);
            let scale = cpu.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())).max(1.0);
            let worst = cpu
                .iter()
                .zip(&gpu)
                .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()));
            println!("gemm {m}x{k}x{n}: worst |gpu - cpu| = {worst:.3e} (scale {scale:.3e})");
            assert!(
                worst / scale < 1.0e-12,
                "GPU 与 CPU 不一致：{worst:.3e}（相对 {:.3e}）",
                worst / scale
            );
        }
        Err(GpuError::Unavailable(why)) => {
            // 允许：没有 CUDA 的机器上本来就不该有 GPU 路径。**不许** panic。
            println!("CUDA 不可用，跳过（这是允许的退化）：{why}");
        }
        Err(other) => panic!("意外错误：{other}"),
    }
}

#[test]
fn gemm_rejects_bad_shapes() {
    let err = gemm_f64(2, 3, 4, &[0.0; 3], &[0.0; 12]).unwrap_err();
    assert!(matches!(err, GpuError::Dimension { .. }), "得到 {err:?}");
}

#[test]
fn availability_probe_never_panics() {
    match available() {
        Ok(()) => println!("CUDA 可用"),
        Err(GpuError::Unavailable(why)) => println!("CUDA 不可用：{why}"),
        Err(other) => panic!("探测不该返回 {other}"),
    }
}
