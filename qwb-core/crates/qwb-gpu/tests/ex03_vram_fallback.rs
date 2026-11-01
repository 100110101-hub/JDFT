//! EX-03 验收：**显存不足自动退回 CPU**。
//!
//! 附录 A.3 要求 GPU 只是"同一运算的另一个实现"：没 GPU、显存不够、调用失败都必须能
//! 无缝退回 CPU，且 CPU 的数值与"一开始就走 CPU"**逐位相同**。这里用**显式预算**把
//! "显存不够"这条路径变成可复现的测试（不需要真的把 8 GiB 显存吃满）：
//!
//! * `gemm_f64_budgeted(..., budget)` 在 `gemm_bytes > budget` 时**一次显存都不碰**，
//!   直接返回 `GpuError::InsufficientMemory`；
//! * `gemm_f64_with_fallback` 把任意 GPU 失败接到朴素 CPU 实现上，返回结果 + 走的路。
use qwb_gpu::{
    GemmRoute, GpuError, available, gemm_bytes, gemm_cpu, gemm_f64_budgeted,
    gemm_f64_with_fallback, memory_info,
};

fn pseudo_random(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
        })
        .collect()
}

#[test]
fn a_budget_that_is_too_small_returns_insufficient_memory_without_touching_the_gpu() {
    let (m, k, n) = (64_usize, 48, 40);
    let a = pseudo_random(m * k, 7);
    let b = pseudo_random(k * n, 11);
    let required = gemm_bytes(m, k, n);
    // 预算只给一半：这条路径必须在**分配之前**就返回，所以即使没有 GPU，错误也应该是
    // `InsufficientMemory` 之外的那一类 —— 用"有 GPU 才断言具体变体"的方式表达。
    let error = gemm_f64_budgeted(m, k, n, &a, &b, required / 2)
        .expect_err("a budget below the requirement must be refused");
    println!(
        "required {required} B, budget {} B -> {error}",
        required / 2
    );
    if available().is_ok() {
        assert_eq!(
            error,
            GpuError::InsufficientMemory {
                required,
                available: required / 2,
            }
        );
    }
}

#[test]
fn the_fallback_returns_the_cpu_reference_bit_for_bit() {
    let (m, k, n) = (32_usize, 24, 20);
    let a = pseudo_random(m * k, 3);
    let b = pseudo_random(k * n, 5);
    let expected = gemm_cpu(m, k, n, &a, &b);
    let (result, route) = gemm_f64_with_fallback(m, k, n, &a, &b);
    println!("route = {route:?}");

    match route {
        // 走 GPU 时允许浮点重排（cuBLAS 的分块与 CPU 的累加顺序不同）：判据 1e-12。
        GemmRoute::Gpu => {
            let worst = result
                .iter()
                .zip(&expected)
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0_f64, f64::max);
            println!("GPU vs CPU max |diff| = {worst:.3e}");
            assert!(worst < 1.0e-12, "GPU result must match the CPU to 1e-12");
        }
        // 退回 CPU 时用的是**同一段**朴素循环，所以必须逐位相同（不是 1e-12）。
        GemmRoute::Cpu(_) => {
            assert_eq!(result, expected, "the CPU fallback must be bit-identical");
        }
    }

    // 预算逼死 GPU 时，route 必须是 Cpu(InsufficientMemory) —— 这条不依赖机器有没有 GPU：
    // 有 GPU 时 budget 让它提前返回，没 GPU 时 Unavailable 也照样回退。
    let required = gemm_bytes(m, k, n);
    let (tight, tight_route) = {
        // 用 `gemm_f64_budgeted` + 手工回退复现包装的行为，避免全局预算被 env 改动。
        match gemm_f64_budgeted(m, k, n, &a, &b, required - 1) {
            Ok(_) => (gemm_cpu(m, k, n, &a, &b), GemmRoute::Gpu),
            Err(reason) => (gemm_cpu(m, k, n, &a, &b), GemmRoute::Cpu(reason)),
        }
    };
    println!("tight budget route = {tight_route:?}");
    // 这条路径**一定**走 CPU（预算比需求少 1 字节），所以必须逐位相同。
    assert_eq!(tight, expected);
    match tight_route {
        GemmRoute::Cpu(reason) => {
            if available().is_ok() {
                assert_eq!(
                    reason,
                    GpuError::InsufficientMemory {
                        required,
                        available: required - 1,
                    }
                );
            }
        }
        GemmRoute::Gpu => panic!("a budget of `required - 1` bytes must not fit"),
    }
}

#[test]
fn the_memory_report_is_self_consistent_when_a_gpu_is_present() {
    match memory_info() {
        Ok(info) => {
            println!(
                "free {} MiB / total {} MiB",
                info.free_bytes / (1 << 20),
                info.total_bytes / (1 << 20)
            );
            assert!(info.free_bytes <= info.total_bytes);
            assert!(info.free_bytes > 0);
            assert!(info.total_bytes > 0);
        }
        Err(GpuError::Unavailable(why)) => println!("no GPU here: {why}"),
        Err(other) => panic!("unexpected memory_info error: {other}"),
    }
}

#[test]
fn the_gemm_size_accounting_is_exact() {
    // `gemm_bytes` 必须把三块 f64 都算上，别只算 C —— 预算少算一块就会真 OOM。
    assert_eq!(gemm_bytes(1, 1, 1), 24);
    assert_eq!(gemm_bytes(2, 3, 4), (6 + 12 + 8) * 8);
    assert_eq!(gemm_bytes(1000, 1000, 1000), 24_000_000);
}
