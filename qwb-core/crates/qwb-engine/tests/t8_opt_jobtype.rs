//! T8：`JobType::Opt` 真的接到 `EngineBackend` 上了（这是 T8 判据里
//! "选 PBE/def2-SVP/opt → 提交 → 结果" 的引擎侧一半）。
//!
//! 三条断言：
//! 1. `opt` 作业返回的是**优化后几何**上的结果（轨道/偶极/能量都是终点的）；
//! 2. 力判据（最大分量 < 4.5e-4 Ha/Bohr）真的满足，并且优化后的键长落在合理区间；
//! 3. 梯度走的是**解析**路径（`used_analytic_gradient = true`，`analytic_fallback = None`）
//!    —— 数值梯度会让每一步付出 `6N` 次 SCF，T8 的 def2-SVP 慢档就靠解析梯度才跑得动。
//!
//! 另一个测试盖住 K2 的 `method_name = "hf"`：`qwb-data/functional/` 里没有 `hf.yaml`
//! （HF 不是泛函数据），引擎必须自己把它翻成"空 terms + α_hf = 1"，否则 K2 文档里
//! 写的 `"hf"` 用法根本走不通。H2/STO-3G 的 HF 能量有外部参考值
//! （`qwb-scf/tests/reference/h2_sto3g.csv`：−1.116759 Ha，容差 1e-3）。

use std::path::PathBuf;

use qwb_engine::OptSink;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn registry() -> Registry {
    Registry::open(data_root()).expect("qwb-data 必须能加载")
}

fn h2(stretch: f64) -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -stretch]),
            Atom::new("H", [0.0, 0.0, stretch]),
        ],
        0,
        1,
    )
}

/// 不打印任何东西的 `Reporter`（引擎要它来打进度/日志）。
struct Silent;

impl qwb_engine_api::Reporter for Silent {
    fn progress(&self, _stage: &str, _pct: f64) {}
    fn log(&self, _line: &str) {}
    fn is_cancelled(&self) -> bool {
        false
    }
}

fn opt_input(method: &str, stretch: f64) -> JobInput {
    let mut options = TaskOptions::default();
    options.max_opt_steps = 30;
    // T3 的力判据（K2 的默认值也是它，这里写出来是为了让断言与输入对上）
    options.opt_gradient_threshold_hartree_per_bohr = 4.5e-4;
    JobInput::new(
        h2(stretch),
        "sto-3g",
        method,
        GridLevel::Normal,
        "vacuum",
        JobType::Opt,
        options,
    )
}

fn bond_length(input: &Molecule) -> f64 {
    let a = input.atoms[0].position_angstrom;
    let b = input.atoms[1].position_angstrom;
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn max_abs_gradient(values: &[[f64; 3]]) -> f64 {
    values
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

#[test]
fn opt_jobtype_optimises_and_reports_the_final_geometry() {
    let sink = OptSink::new();
    let backend = qwb_engine::RksBackend::new(registry()).with_opt_sink(sink.clone());

    // 从一个明显拉长的几何出发（0.80 Å 的 H–H），要真的把键长压回平衡附近
    let input = opt_input("pbe", 0.40);
    let result = backend.run(&input, &Silent).expect("opt job");

    let report = sink.take().expect("opt 作业必须留下 OptReport");
    assert!(
        report.converged,
        "力判据没满足：{:?}",
        report.trajectory.last()
    );
    assert!(
        report.iterations >= 1 && report.iterations <= 30,
        "步数不合法：{}",
        report.iterations
    );
    let force = max_abs_gradient(&report.gradient);
    assert!(force < 4.5e-4, "最大力 {force:.3e} Ha/Bohr 超过 T3 判据");

    // 优化后的键长：H2/STO-3G/PBE 约 0.735 Å（PBE 略长于实验 0.741？不，STO-3G 会短）
    let length = bond_length(&report.molecule);
    assert!(
        (0.70..0.78).contains(&length),
        "优化后的 H–H 键长 {length:.4} Å 不在 0.70–0.78 之间"
    );

    // `JobResult` 是**终点**上的量：能量与 report 一致，轨道/偶极都在
    assert!(
        (result.energy_hartree - report.energy_hartree).abs() < 1.0e-9,
        "结果能量 {} 与优化终点 {} 不一致",
        result.energy_hartree,
        report.energy_hartree
    );
    assert!(result.scf_converged);
    assert!(result.mo_coeffs.is_some(), "opt 结果必须带轨道系数");
    assert!(result.mo_energies_hartree.is_some());
    assert!(result.dipole_debye.is_some());
    // 收尾单点是在**优化后**的几何上算的：能量必须比初值低
    let initial = opt_input("pbe", 0.40);
    let mut sp_input = initial.clone();
    sp_input.jobtype = JobType::Sp;
    let start = backend.run(&sp_input, &Silent).expect("initial sp");
    assert!(
        result.energy_hartree < start.energy_hartree,
        "优化后的能量 {} 没有比初值 {} 低",
        result.energy_hartree,
        start.energy_hartree
    );
}

#[test]
fn opt_uses_the_analytic_gradient() {
    let sink = OptSink::new();
    let backend = qwb_engine::RksBackend::new(registry()).with_opt_sink(sink.clone());
    backend
        .run(&opt_input("pbe", 0.38), &Silent)
        .expect("opt job");
    let report = sink.take().expect("OptReport");
    assert!(
        report.used_analytic_gradient,
        "KS 解析梯度本该可用，实际退到了数值梯度：{:?}",
        report.analytic_fallback
    );
    assert!(
        report.analytic_fallback.is_none(),
        "解析梯度失败原因：{:?}",
        report.analytic_fallback
    );
}

#[test]
fn hf_method_name_works_through_the_data_layer() {
    // K2 规定 HF 用 `method_name = "hf"`；qwb-data 里没有 hf.yaml，引擎自己做翻译。
    let backend = qwb_engine::RksBackend::new(registry());
    let input = JobInput::new(
        h2(0.37),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        TaskOptions::default(),
    );
    let result = backend.run(&input, &Silent).expect("hf sp");
    // 外部参考：qwb-scf/tests/reference/h2_sto3g.csv 的 hf_total_energy = -1.116759（容差 1e-3）
    assert!(
        (result.energy_hartree - (-1.116_759)).abs() < 1.0e-3,
        "HF/STO-3G 的 H2 能量 {} 与外部参考 -1.116759 差超过 1e-3",
        result.energy_hartree
    );

    // 大写也认（K2 的数值/枚举拼写只对 jobtype 冻结，方法名不区分大小写更稳）
    let upper = JobInput::new(
        h2(0.37),
        "sto-3g",
        "HF",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        TaskOptions::default(),
    );
    let again = backend.run(&upper, &Silent).expect("hf sp (upper)");
    assert_eq!(result.energy_hartree, again.energy_hartree);
}

#[test]
fn freq_is_refused_explicitly() {
    let backend = qwb_engine::RksBackend::new(registry());
    let input = JobInput::new(
        h2(0.37),
        "sto-3g",
        "pbe",
        GridLevel::Coarse,
        "vacuum",
        JobType::Freq,
        TaskOptions::default(),
    );
    let error = backend.run(&input, &Silent).expect_err("freq 还没接线");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("freq"), "错误信息要指名是 freq：{message}");
        }
        other => panic!("freq 应该报 Unsupported，实际是 {other:?}"),
    }
}
