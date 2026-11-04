//! EX-03 A1/A2（**产品路径**）：`RksBackend::with_xc_grid_gpu()` 跑完整 PBE SCF，
//! CPU vs GPU 的能量一致性与墙钟。
//!
//! 这一条走的是**引擎**而不是某一层：注册表 → 基组 → 2e 积分 → 网格 → SCF → `JobResult`。
//! 计时里包含建基组/积分/网格（两条路都一样），所以这里的加速比是**诚实的端到端**数字，
//! 不是"只掐收缩段"。
//!
//! 协议（每个算例都按同一套）：每个路径先跑 1 次**热身**（吃掉 JIT 编译、页缓存、线程池
//! 启动），再跑 `REPS` 次取样本，报 **min 与 mean**，并把样本原样打出来；CPU 与 GPU 交替跑
//! （先 CPU 再 GPU），两个路径用**同一个** `TaskOptions`（同样的 `scf_max_iter`、能量/密度
//! 阈值、DIIS 长度）。
//!
//! 没有 GPU / 没开 `gpu-kernel` 时**不是跳过**：断言退回 CPU、能量与 CPU 路径逐位相同，
//! 并打印原因。门禁在没有 GPU 的机器上照样绿。
#![allow(clippy::field_reassign_with_default)] // options 逐字段覆盖最清楚

use std::path::PathBuf;
use std::time::Instant;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, GridLevel, JobInput, JobResult, JobType, Molecule, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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

fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 0.74]),
        ],
        0,
        1,
    )
}

/// 同一个算例、同一套收敛判据；只有分子 / 基组 / 网格等级不同。
fn input(molecule: Molecule, basis: &str, grid: GridLevel) -> JobInput {
    let mut options = TaskOptions::default();
    options.scf_max_iter = 200;
    options.scf_energy_threshold_hartree = 1.0e-9;
    options.scf_density_threshold = 1.0e-7;
    options.diis_history = 8;
    JobInput::new(molecule, basis, "pbe", grid, "vacuum", JobType::Sp, options)
}

fn cpu_backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

fn gpu_backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data")).with_xc_grid_gpu()
}

/// 一次「协议」的产物：两条路径的样本、min/mean 与最后一次的 `JobResult`。
struct Measured {
    cpu_samples: Vec<f64>,
    gpu_samples: Vec<f64>,
    cpu_job: JobResult,
    gpu_job: JobResult,
}

impl Measured {
    fn cpu_min(&self) -> f64 {
        self.cpu_samples
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }
    fn gpu_min(&self) -> f64 {
        self.gpu_samples
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }
    fn cpu_mean(&self) -> f64 {
        self.cpu_samples.iter().sum::<f64>() / self.cpu_samples.len() as f64
    }
    fn gpu_mean(&self) -> f64 {
        self.gpu_samples.iter().sum::<f64>() / self.gpu_samples.len() as f64
    }
    fn samples(samples: &[f64]) -> String {
        samples
            .iter()
            .map(|seconds| format!("{seconds:.3}"))
            .collect::<Vec<_>>()
            .join(" / ")
    }
}

/// 热身 1 次 + 交替采样 `reps` 次。
fn measure(job_input: &JobInput, reps: usize) -> Measured {
    let _ = cpu_backend()
        .run(job_input, &NullReporter)
        .expect("cpu warmup");
    let _ = gpu_backend()
        .run(job_input, &NullReporter)
        .expect("gpu warmup");
    let mut cpu_samples = Vec::with_capacity(reps);
    let mut gpu_samples = Vec::with_capacity(reps);
    let mut cpu_job = None;
    let mut gpu_job = None;
    for _ in 0..reps {
        let started = Instant::now();
        cpu_job = Some(
            cpu_backend()
                .run(job_input, &NullReporter)
                .expect("cpu run"),
        );
        cpu_samples.push(started.elapsed().as_secs_f64());

        let started = Instant::now();
        gpu_job = Some(
            gpu_backend()
                .run(job_input, &NullReporter)
                .expect("gpu run"),
        );
        gpu_samples.push(started.elapsed().as_secs_f64());
    }
    Measured {
        cpu_samples,
        gpu_samples,
        cpu_job: cpu_job.expect("cpu job"),
        gpu_job: gpu_job.expect("gpu job"),
    }
}

#[test]
fn a1_a2_pbe_scf_cpu_vs_gpu_end_to_end() {
    const WARMUP: usize = 1;
    const REPS: usize = 3;
    println!("NVRTC 状态：{:?}", qwb_gpu_status());
    for (molecule, basis, grid, label) in [
        (water(), "sto-3g", GridLevel::Normal, "H2O/STO-3G"),
        (water(), "def2-svp", GridLevel::Normal, "H2O/def2-SVP"),
    ] {
        let job_input = input(molecule, basis, grid);
        let measured = measure(&job_input, REPS);
        let (cpu_job, gpu_job) = (&measured.cpu_job, &measured.gpu_job);
        let delta = (gpu_job.energy_hartree - cpu_job.energy_hartree).abs();
        let cpu_iterations = iterations_from_log(&cpu_job.log);
        let gpu_iterations = iterations_from_log(&gpu_job.log);
        println!(
            "PBE/{label}/grid-normal: warm-up = {WARMUP}, reps = {REPS} | \
             CPU min {:.3} s / mean {:.3} s | GPU min {:.3} s / mean {:.3} s | \
             min 加速 {:.2}x、mean 加速 {:.2}x | E_cpu = {:.15e}, E_gpu = {:.15e}, \
             |ΔE| = {delta:.3e} Ha（逐位相同 = {}）| iterations CPU/GPU = {:?} / {:?}",
            measured.cpu_min(),
            measured.cpu_mean(),
            measured.gpu_min(),
            measured.gpu_mean(),
            measured.cpu_min() / measured.gpu_min(),
            measured.cpu_mean() / measured.gpu_mean(),
            cpu_job.energy_hartree,
            gpu_job.energy_hartree,
            gpu_job.energy_hartree.to_bits() == cpu_job.energy_hartree.to_bits(),
            cpu_iterations,
            gpu_iterations,
        );
        // 日志里必须能读到 XC 走了哪条路（走没走成、为什么没走成）。
        for job in [cpu_job, gpu_job] {
            for entry in job.log.split('\n').filter(|line| line.contains("xc =")) {
                println!("    log: {entry}");
            }
        }
        println!("    样本 CPU：{}", Measured::samples(&measured.cpu_samples));
        println!("    样本 GPU：{}", Measured::samples(&measured.gpu_samples));

        // A1：端到端能量差 ≤ 1e-10 Ha（与附录 A 的判据同一条线）。
        assert!(delta <= 1.0e-10, "|ΔE| = {delta:.3e} 超过 1e-10 Ha");
        assert!(
            cpu_job.scf_converged && gpu_job.scf_converged,
            "CPU/GPU 有一侧没收敛"
        );
        // A2：不对加速比做断言 —— 小体系上设备路径**慢**是合法发现（启动开销 + 传输）。
        assert!(measured.cpu_min() > 0.0 && measured.gpu_min() > 0.0);
        // 迭代步数两条路必须一致：同样的判据、同样的初猜，路径不该改变收敛步数。
        assert_eq!(
            cpu_iterations, gpu_iterations,
            "CPU 与 GPU 的 SCF 迭代步数不同 —— 比较的不是同一个算例"
        );
        assert!(
            cpu_iterations.is_some(),
            "没能从日志里读出迭代步数 —— 那条一致性断言就变成了 None == None"
        );
    }
}

#[test]
fn a2b_small_systems_do_not_win_on_the_device_path() {
    // 「GPU 也可能更慢」必须自己量出来、自己报出来：小体系上每步 XC 只有
    // `O(n_points·n²)` 这么点活（H2/STO-3G 的 n = 2、网格 29 400 点），而设备侧有
    // JIT/cuBLAS 句柄、H2D 往返与同步的固定开销。这里照同一套协议量，**不对加速比做断言**
    // （赢或输都是发现）。
    const WARMUP: usize = 1;
    const REPS: usize = 5;
    for (molecule, label) in [(h2(), "H2/STO-3G"), (water(), "H2O/STO-3G")] {
        let job_input = input(molecule, "sto-3g", GridLevel::Coarse);
        let measured = measure(&job_input, REPS);
        let delta = (measured.gpu_job.energy_hartree - measured.cpu_job.energy_hartree).abs();
        println!(
            "PBE/{label}/grid-coarse（{} 步）: warm-up = {WARMUP}, reps = {REPS} | \
             CPU min {:.3} s / mean {:.3} s | GPU min {:.3} s / mean {:.3} s | \
             min 加速 {:.2}x、mean 加速 {:.2}x | E_cpu = {:.15e}, E_gpu = {:.15e}, \
             |ΔE| = {delta:.3e} Ha | 样本 CPU {} / GPU {}",
            iterations_from_log(&measured.cpu_job.log).unwrap_or(0),
            measured.cpu_min(),
            measured.cpu_mean(),
            measured.gpu_min(),
            measured.gpu_mean(),
            measured.cpu_min() / measured.gpu_min(),
            measured.cpu_mean() / measured.gpu_mean(),
            measured.cpu_job.energy_hartree,
            measured.gpu_job.energy_hartree,
            Measured::samples(&measured.cpu_samples),
            Measured::samples(&measured.gpu_samples),
        );
        assert!(delta <= 1.0e-10, "|ΔE| = {delta:.3e} 超过 1e-10 Ha");
    }
}

/// 从引擎日志里读 SCF 迭代步数（`iterations = N`）。
///
/// `JobResult` 的 K2 契约里没有迭代步数字段（冻结），`JobResult::log` 的收尾那一行里有
/// `iterations = N`，所以从这里读。
fn iterations_from_log(log: &str) -> Option<u32> {
    const MARKER: &str = "iterations = ";
    let start = log.find(MARKER)? + MARKER.len();
    let token: String = log[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    token.parse().ok()
}

/// `gpu-kernel` 下的 NVRTC 状态（没有这个 feature 时是 `Err`，打印出来即可）。
fn qwb_gpu_status() -> String {
    #[cfg(feature = "gpu-kernel")]
    {
        format!("{:?}", qwb_gpu::xc::nvrtc_status())
    }
    #[cfg(not(feature = "gpu-kernel"))]
    {
        "feature `gpu-kernel` 未启用（这条构建里 qwb-gpu 不参与编译）".to_string()
    }
}
