//! IN-04 的 **GPU 侧补充测量**：RI-J 全管线（setup 仍然是 CPU）与 4 索引 J 在墙钟、同值上的对照。
//!
//! 为什么这份测量放在 `qwb-gpu` 而不是 `qwb-scf`：包图方向是 `qwb-gpu → qwb-scf`（见
//! `qwb-scf/src/xc_device.rs`），`qwb-scf` 的测试拿不到 `qwb_gpu::ri_j`。而 IN-04 要回答的问题是
//! 「GPU 的 RI-J 核有没有把全管线推进到判据的 ≥3×」，所以必须把**同一条管线**里的两段放在一起量：
//! setup（三中心 + 度量 + `M^{-1/2}` + `B'` 变换，纯 CPU，没有 GPU 实现）与收缩（`t = B'ᵀ·P`、
//! `J = B'·t`，CPU 朴素循环 vs `qwb_gpu::ri_j::ri_j_gemm` 的 cuBLAS 路径）。
//!
//! 协议（写死在源码里，便于复现）：**release** 构建、每个体系先跑 1 次**热身**（丢弃），
//! 再取 `reps` 次测量、报告最小值与中位数；每次测量都从零重建（张量/三中心不跨次复用）。
//! 密度是 `docs/choices.md` 九十六 起的测量关用的同一个伪随机密度（同一个 LCG 种子），
//! 所以相对差与 `qwb-scf/tests/in04_ri_j_measurement.rs` 的数可以直接对齐。
//!
//! 手动跑：
//!
//! ```text
//! cargo test --release -p qwb-gpu --test ex03_ri_j_pipeline_measurement -- --ignored --nocapture
//! ```
use std::time::Instant;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_gpu::ri_j::{ri_j_gemm, ri_j_naive};
use qwb_integrals::ri::{inverse_sqrt, three_center, two_center_coulomb, whiten_three_center};
use qwb_registry::Registry;

fn data_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    )
}

/// 苯（与 `qwb-scf/tests/in04_ri_j_measurement.rs` 同一个几何与同一套键长）。
fn benzene() -> Molecule {
    let c = 1.397_f64;
    let h = 2.481_f64;
    let mut atoms = Vec::new();
    for k in 0..6 {
        let angle = std::f64::consts::PI * 2.0 * k as f64 / 6.0;
        atoms.push(Atom::new("C", [c * angle.cos(), c * angle.sin(), 0.0]));
    }
    for k in 0..6 {
        let angle = std::f64::consts::PI * 2.0 * k as f64 / 6.0;
        atoms.push(Atom::new("H", [h * angle.cos(), h * angle.sin(), 0.0]));
    }
    Molecule::new(atoms, 0, 1)
}

/// 与 IN-04 的测量关同一个伪随机密度（同一个 LCG 种子、同一个 `-0.5` 平移），行主序 `n×n`。
fn pseudo_density(n: usize) -> Vec<f64> {
    let mut seed = 1_234_u64;
    let mut density = vec![0.0_f64; n * n];
    for slot in density.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5;
    }
    density
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn min_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

/// 4 索引 J（张量建立 + 朴素 `n⁴` 收缩），返回 `(J, 总秒数, 张量建立秒数)`。
///
/// `threads = 0` 是产品默认（rayon 默认线程数），`threads = 1` 是**单线程** —— 后者用来做
/// 「同等的并行度条件下」的对照：`two_electron` 早已并行（`DeterministicReduce`），
/// IN-04 第二轮起 `qwb_integrals::ri` 的三中心 / 度量 / `B'` 变换也并行，
/// 所以 `threads = 1` 这一列仍然是「RI 侧一行并行都没有」的那个世界。
fn exact_j(
    basis: &InternalBasis,
    molecule: &Molecule,
    n: usize,
    density: &[f64],
    threads: usize,
) -> (Vec<f64>, f64, f64) {
    let started = Instant::now();
    let tensor = qwb_integrals::two_electron::two_electron_with_threads(basis, molecule, threads)
        .expect("two_electron");
    let tensor_seconds = started.elapsed().as_secs_f64();
    let mut j = vec![0.0_f64; n * n];
    for i in 0..n {
        for jj in 0..n {
            let mut sum = 0.0_f64;
            for k in 0..n {
                for l in 0..n {
                    sum += tensor.get(i, jj, k, l) * density[k * n + l];
                }
            }
            j[i * n + jj] = sum;
        }
    }
    (j, started.elapsed().as_secs_f64(), tensor_seconds)
}

/// RI setup 的耗时分解（三中心 / 度量逆平方根 / `B'` 变换）。分开计时是必要的：
/// 上一轮的结论「装配不是瓶颈」是对的，但「setup 里哪一段是瓶颈」当时没有拆开量。
struct Setup {
    b_fit: Vec<f64>,
    total: f64,
    three_center: f64,
    metric: f64,
    b_prime: f64,
}

/// RI-J 的**共享 setup**：三中心 + 2 中心度量 + `M^{-1/2}` + `B' = M^{-1/2}·B`（`B'` 行主序
/// `(μν) × P`）。这一段没有 GPU 实现（`qwb-gpu` 只有收缩核），任何 RI-J 变体都要付这笔钱。
///
/// 三个积分/变换步骤都走 `qwb_integrals::ri` 的**默认线程数**入口（IN-04 第二轮把它们并行化了：
/// 3 中心、度量、`B'` 各自 `DeterministicReduce::map_ordered`，逐位与单线程相同）。
fn ri_setup(basis: &InternalBasis, aux: &InternalBasis, molecule: &Molecule) -> Setup {
    let started = Instant::now();
    let b = three_center(basis, aux, molecule).expect("three_center");
    let three_center_seconds = started.elapsed().as_secs_f64();

    let metric_started = Instant::now();
    let metric = two_center_coulomb(aux, molecule).expect("two_center_coulomb");
    let m_inv_sqrt = inverse_sqrt(&metric, 1.0e-10).expect("inverse_sqrt");
    let metric_seconds = metric_started.elapsed().as_secs_f64();

    let b_prime_started = Instant::now();
    let b_fit = whiten_three_center(&b, &m_inv_sqrt).expect("whiten_three_center");
    Setup {
        b_fit,
        total: started.elapsed().as_secs_f64(),
        three_center: three_center_seconds,
        metric: metric_seconds,
        b_prime: b_prime_started.elapsed().as_secs_f64(),
    }
}

/// 最大绝对差、按「4 索引 J 的最大元素」归一化的相对差，以及最大元素的绝对值。
fn differences(exact: &[f64], approx: &[f64]) -> (f64, f64, f64) {
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (a, b) in exact.iter().zip(approx) {
        worst = worst.max((a - b).abs());
        scale = scale.max(a.abs());
    }
    (worst, worst / scale.max(1e-30), scale)
}

fn measure(label: &str, registry: &Registry, molecule: &Molecule, reps: usize) {
    let basis =
        InternalBasis::build(registry.get_basis("def2-svp").unwrap(), molecule).expect("basis");
    let aux = InternalBasis::build(
        registry.get_basis("def2-universal-jkfit").unwrap(),
        molecule,
    )
    .expect("aux basis");
    let n = basis.n_functions();
    let n_aux = aux.n_functions();
    let density = pseudo_density(n);

    // ── 4 索引 J ────────────────────────────────────────────────────────────
    // 产品默认路径（`two_electron` 用 rayon 默认线程数）与**单线程**版本各量一遍：
    // 前者是与台账/判据对齐的「现状」，后者是「两边都串行」的对照（RI 侧这一轮也并行化了）。
    let mut exact_runs = Vec::new();
    let mut exact_tensor_runs = Vec::new();
    let mut serial_exact_runs = Vec::new();
    let mut j_exact: Vec<f64> = Vec::new();
    for run in 0..=reps {
        let (j, seconds, tensor_seconds) = exact_j(&basis, molecule, n, &density, 0);
        if run == 0 {
            j_exact = j;
        } else {
            exact_runs.push(seconds);
            exact_tensor_runs.push(tensor_seconds);
        }
        let (_, serial_seconds, _) = exact_j(&basis, molecule, n, &density, 1);
        if run > 0 {
            serial_exact_runs.push(serial_seconds);
        }
    }

    // ── RI-J：setup（CPU）+ 收缩（CPU naive / GPU cuBLAS 各一次）──────────
    let mut setup_runs = Vec::new();
    let mut three_center_runs = Vec::new();
    let mut metric_runs = Vec::new();
    let mut b_prime_runs = Vec::new();
    let mut cpu_runs = Vec::new();
    let mut gpu_runs = Vec::new();
    let mut j_ri_naive: Vec<f64> = Vec::new();
    let mut j_ri_gemm: Vec<f64> = Vec::new();
    let mut used_gpu = false;
    let mut fallback_reason: Option<String> = None;
    let mut repeat_bits_equal = (false, false);
    for run in 0..=reps {
        let setup = ri_setup(&basis, &aux, molecule);
        let b_fit = &setup.b_fit;

        let started = Instant::now();
        let j_naive = ri_j_naive(n, n_aux, b_fit, &density);
        let cpu_seconds = started.elapsed().as_secs_f64();

        let started = Instant::now();
        let result = ri_j_gemm(n, n_aux, b_fit, &density).expect("ri_j_gemm");
        let gpu_seconds = started.elapsed().as_secs_f64();

        // 确定性：同一条路径连算两次必须逐位相同（K10 第 ③ 条）。
        let naive_again = ri_j_naive(n, n_aux, b_fit, &density);
        let gemm_again = ri_j_gemm(n, n_aux, b_fit, &density).expect("ri_j_gemm again");
        repeat_bits_equal = (
            j_naive
                .iter()
                .zip(&naive_again)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            result
                .j
                .iter()
                .zip(&gemm_again.j)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        );

        if run == 0 {
            j_ri_naive = j_naive;
            used_gpu = result.used_gpu;
            fallback_reason = result.fallback_reason.clone();
            j_ri_gemm = result.j;
        } else {
            setup_runs.push(setup.total);
            three_center_runs.push(setup.three_center);
            metric_runs.push(setup.metric);
            b_prime_runs.push(setup.b_prime);
            cpu_runs.push(cpu_seconds);
            gpu_runs.push(gpu_seconds);
        }
    }

    let (worst_naive, rel_naive, scale) = differences(&j_exact, &j_ri_naive);
    let (worst_gemm, rel_gemm, _) = differences(&j_exact, &j_ri_gemm);
    let (worst_paths, rel_paths, _) = differences(&j_ri_naive, &j_ri_gemm);

    let exact_min = min_of(&exact_runs);
    let exact_med = median(&mut exact_runs.clone());
    let tensor_min = min_of(&exact_tensor_runs);
    let serial_exact_min = min_of(&serial_exact_runs);
    let serial_exact_med = median(&mut serial_exact_runs.clone());
    let setup_min = min_of(&setup_runs);
    let setup_med = median(&mut setup_runs.clone());
    let three_center_min = min_of(&three_center_runs);
    let metric_min = min_of(&metric_runs);
    let b_prime_min = min_of(&b_prime_runs);
    let cpu_min = min_of(&cpu_runs);
    let cpu_med = median(&mut cpu_runs.clone());
    let gpu_min = min_of(&gpu_runs);
    let gpu_med = median(&mut gpu_runs.clone());

    let full_cpu_min = setup_min + cpu_min;
    let full_cpu_med = setup_med + cpu_med;
    let full_gpu_min = setup_min + gpu_min;
    let full_gpu_med = setup_med + gpu_med;

    println!("\n==== {label}: n={n}, n_aux={n_aux}, reps={reps}（另热身 1 次，已丢弃）====");
    println!(
        "4 索引 J        min {exact_min:.3} s / median {exact_med:.3} s（其中张量建立 min {tensor_min:.3} s；rayon 默认线程）"
    );
    println!(
        "4 索引 J 单线程  min {serial_exact_min:.3} s / median {serial_exact_med:.3} s（张量建立是同一条并行路径的 threads=1）→ 并行提速 {:.2}x（min）",
        serial_exact_min / exact_min
    );
    println!(
        "RI setup        min {setup_min:.3} s / median {setup_med:.3} s = 三中心 {three_center_min:.3} + 度量+逆平方根 {metric_min:.3} + B' 变换 {b_prime_min:.3}（纯 CPU，无 GPU 实现）"
    );
    println!("RI 收缩 CPU     min {cpu_min:.3} s / median {cpu_med:.3} s");
    println!(
        "RI 收缩 GPU     min {gpu_min:.3} s / median {gpu_med:.3} s（used_gpu={used_gpu}，fallback={fallback_reason:?}）"
    );
    println!(
        "RI 全管线 CPU   min {full_cpu_min:.3} s / median {full_cpu_med:.3} s → 加速 {:.2}x（min）/ {:.2}x（median）",
        exact_min / full_cpu_min,
        exact_med / full_cpu_med
    );
    println!(
        "RI 全管线 GPU   min {full_gpu_min:.3} s / median {full_gpu_med:.3} s → 加速 {:.2}x（min）/ {:.2}x（median）",
        exact_min / full_gpu_min,
        exact_med / full_gpu_med
    );
    println!(
        "同等的串行条件下（4 索引 J 用 threads=1）：RI 全管线 CPU 加速 {:.2}x（min）/ {:.2}x（median）；GPU 收缩 {:.2}x（min）",
        serial_exact_min / full_cpu_min,
        serial_exact_med / full_cpu_med,
        serial_exact_min / full_gpu_min
    );
    println!(
        "占 RI 全管线（CPU 收缩侧）：setup {:.2}%（三中心 {:.2}% / 度量 {:.2}% / B' {:.2}%）、收缩 {:.2}%",
        100.0 * setup_min / full_cpu_min,
        100.0 * three_center_min / full_cpu_min,
        100.0 * metric_min / full_cpu_min,
        100.0 * b_prime_min / full_cpu_min,
        100.0 * cpu_min / full_cpu_min
    );
    println!(
        "占 RI 全管线（GPU 收缩侧）：setup {:.2}%、收缩 {:.4}%",
        100.0 * setup_min / full_gpu_min,
        100.0 * gpu_min / full_gpu_min
    );
    println!(
        "同值：max|J_RI_CPU − J_exact| = {worst_naive:.3e} Ha（相对 {rel_naive:.2e}，max|J_exact| = {scale:.3e} Ha）"
    );
    println!("同值：max|J_RI_GPU − J_exact| = {worst_gemm:.3e} Ha（相对 {rel_gemm:.2e}）");
    println!("CPU vs GPU 两条收缩：max|Δ| = {worst_paths:.3e} Ha（相对 {rel_paths:.2e}）");
    println!(
        "确定性：CPU 连算两次逐位相同 {}/GPU 连算两次逐位相同 {}",
        repeat_bits_equal.0, repeat_bits_equal.1
    );
}

#[test]
#[ignore = "slow: IN-04 的 GPU/CPU RI-J 全管线测量（苯/def2-SVP 一次 setup 约 70 s，热身+重复后约 10 min）"]
fn ri_j_pipeline_cpu_versus_gpu_measurement() {
    match qwb_gpu::available() {
        Ok(()) => println!(
            "CUDA 可用；显存 {:?}",
            qwb_gpu::memory_info().map(|info| (info.free_bytes, info.total_bytes))
        ),
        Err(why) => println!("CUDA 不可用：{why} —— GPU 收缩会退回 CPU，下面会如实打印 fallback"),
    }
    println!("RUST 构建：release（cargo test --release）；opt-level 由 [profile.release] 决定");
    let registry = Registry::open(data_root()).expect("qwb-data");
    // 水：热身 1 + 3 次测量；苯：热身 1 + 2 次（一次 setup 就要 ~70 s，重复次数按墙钟预算定，
    // 但**体系与路径一个都没少**）。
    measure("water", &registry, &water(), 3);
    measure("benzene", &registry, &benzene(), 2);
}
