//! T5 的引擎侧验收（ECP + 重元素）：CuH / LANL2DZ 单点能跑通，且走的是 ECP 那条路。
//!
//! # 判据来源与两处更正
//!
//! T5 原文冻结了 CuH / LANL2DZ 与 ZnO / def2-SVP+def2-ECP，并写「Cu `n_core_electrons == 28`、
//! `z_eff == 1.0`」。两处与真实数据不符，2026-10-26 经用户批准更正（详见
//! `docs/status.yaml` 的 D-07/T5 与 `qwb-registry/tests/d07_lanl2dz.rs` 的文件头）：
//!
//! * Cu 的 LANL2DZ 是 **10 芯电子**（`[Ne]` 芯，3s3p 留在价层）→ `z_eff = 19`；
//! * Zn 的 def2-ECP **不存在**（BSE 的 def2-ecp 从 Rb 开始，def2 对 3d 是全电子），
//!   所以 ZnO 改用同族 LANL2DZ（Zn：18 芯，`z_eff = 12`）。
//!
//! # 门禁里跑什么（为什么用 grid-coarse）
//!
//! CuH/LANL2DZ 有 24 个基函数，ECP 矩阵元的求积是 O(点数 × n_ao²)；debug 构建下
//! grid-normal 的一次完整单点要 **1643 s**（实测），放不进门禁。所以门禁版用
//! **grid-coarse**，把 grid-normal 的完整数字放到 `#[ignore]` 的慢档测试里，两地都钉能量：
//!
//! | 网格 | E (Ha) | 说明 |
//! | --- | --- | --- |
//! | grid-coarse | −213.9067554378 | 门禁版 |
//! | grid-normal | −213.9069054186 | 慢档（`#[ignore]`） |
//!
//! 两个值都是 **internal-independent-path**（§7.0 的 fallback）：PBE/LANL2DZ 的 CuH
//! 在可引用表里查不到，它们是回归钉 —— 钉的是"ECP 接线不会被后续改动悄悄改掉"。
//!
//! # ZnO 的状态（如实记）
//!
//! ZnO/LANL2DZ/PBE 的 SCF **不收敛**（400/500 步、多初猜、阻尼、能级移动、四个网格、
//! PBE/SVWN5 都不行）。定位到的根因不是 ECP 矩阵元（I-05 已验到 1e-14），而是**闭壳层受限的
//! O 原子本身是伪解**（O 的 ³P 有两个单占 p 轨道）：core guess 从一个伪组态出发、在几个组态
//! 之间跳（`QWB_SCF_TRACE=1` 实测 max|dP| ≈ 1、能量在 −119…−121 Ha 之间振荡）。
//! 记录在 `docs/status.yaml` 的 T5（未通过项）里，不在这个测试里假装通过。
use std::path::PathBuf;

use qwb_engine::RksBackend;
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

struct Silent;

impl qwb_engine_api::Reporter for Silent {
    fn progress(&self, _stage: &str, _pct: f64) {}
    fn log(&self, _line: &str) {}
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// 收集引擎日志（用来断言 ECP 那一行）。
struct Collecting {
    lines: std::sync::Mutex<Vec<String>>,
}

impl Collecting {
    fn new() -> Self {
        Self {
            lines: std::sync::Mutex::new(Vec::new()),
        }
    }
    fn joined(&self) -> String {
        self.lines.lock().expect("lock").join("\n")
    }
}

impl qwb_engine_api::Reporter for Collecting {
    fn progress(&self, _stage: &str, _pct: f64) {}
    fn log(&self, line: &str) {
        if let Ok(mut lines) = self.lines.lock() {
            lines.push(line.to_string());
        }
    }
    fn is_cancelled(&self) -> bool {
        false
    }
}

fn cuh(distance_angstrom: f64) -> Molecule {
    Molecule::new(
        vec![
            Atom::new("Cu", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, distance_angstrom]),
        ],
        0,
        1,
    )
}

fn job(molecule: Molecule, ecp: &str, grid: GridLevel) -> JobInput {
    let mut options = TaskOptions::default();
    options.scf_max_iter = 200;
    options.ecp_name = Some(ecp.to_string());
    JobInput::new(
        molecule,
        "lanl2dz",
        "pbe",
        grid,
        "vacuum",
        JobType::Sp,
        options,
    )
}

#[test]
fn cuh_lanl2dz_single_point_converges() {
    let backend = RksBackend::new(registry());
    let reporter = Collecting::new();
    let result = backend
        .run(&job(cuh(1.46), "lanl2dz-cu", GridLevel::Coarse), &reporter)
        .expect("CuH/LANL2DZ/PBE 单点");
    println!(
        "CuH/LANL2DZ/PBE (grid-coarse): E = {:.10} Ha",
        result.energy_hartree
    );
    assert!(result.scf_converged);
    // 回归钉（internal-independent-path）：见文件头。
    assert!(
        (result.energy_hartree - (-213.906_755_437_8)).abs() < 1.0e-6,
        "CuH/LANL2DZ/PBE 能量 {:.10} 与钉住值 -213.9067554378 不一致",
        result.energy_hartree
    );

    // ① ECP 真的进了哈密顿量：日志里要有 n_core/z_eff 那一行。
    let log = reporter.joined();
    let ecp_line = log
        .lines()
        .find(|line| line.contains("ecp = lanl2dz-cu"))
        .unwrap_or_else(|| panic!("引擎日志里没有 ECP 那一行：{log}"));
    assert!(
        ecp_line.contains("n_core = 10") && ecp_line.contains("z_eff = 19"),
        "ECP 日志没写清 n_core/z_eff：{ecp_line}"
    );

    // ② G5：结果的数据快照里必须**同时**有基组与 ECP 文件。
    let paths: Vec<(&str, &str)> = result.data_snapshot.iter().collect();
    assert!(
        paths
            .iter()
            .any(|(path, _)| path.ends_with("basis/lanl2dz.yaml")),
        "data_snapshot 缺 lanl2dz 基组：{paths:?}"
    );
    assert!(
        paths
            .iter()
            .any(|(path, _)| path.ends_with("ecp/lanl2dz-cu.yaml")),
        "data_snapshot 缺 lanl2dz-cu 的 ECP：{paths:?}"
    );
    // 轨道/偶极照常给出（ECP 不影响 K2 的字段）。
    assert!(result.mo_coeffs.is_some());
    assert!(result.mo_energies_hartree.is_some());
}

#[test]
#[ignore = "慢档（debug 下约 27 分钟）：CuH/LANL2DZ 在 grid-normal 上的完整单点；数字已实测 E = -213.9069054186 Ha"]
fn cuh_lanl2dz_grid_normal_is_pinned() {
    let backend = RksBackend::new(registry());
    let first = backend
        .run(&job(cuh(1.46), "lanl2dz-cu", GridLevel::Normal), &Silent)
        .expect("grid-normal 单点");
    assert!(
        (first.energy_hartree - (-213.906_905_418_6)).abs() < 1.0e-6,
        "grid-normal 能量 {:.10} 与钉住值 -213.9069054186 不一致",
        first.energy_hartree
    );
    // ECP 矩阵元是确定性求积：同一输入两次跑必须逐位相同。
    let second = backend
        .run(&job(cuh(1.46), "lanl2dz-cu", GridLevel::Normal), &Silent)
        .expect("grid-normal 单点（第二次）");
    assert_eq!(
        first.energy_hartree.to_bits(),
        second.energy_hartree.to_bits(),
        "同一输入的两次 ECP 计算能量不是逐位相同"
    );
}

#[test]
fn ecp_element_must_match_the_molecule() {
    let backend = RksBackend::new(registry());
    // CuH 里没有 Zn：lanl2dz-zn 必须报数据错误（这一步在 SCF 之前，很便宜）。
    let error = backend
        .run(&job(cuh(1.46), "lanl2dz-zn", GridLevel::Coarse), &Silent)
        .expect_err("ECP 元素不匹配应当失败");
    match error {
        EngineError::Data(message) => {
            assert!(message.contains("Zn"), "错误信息要指名元素：{message}");
        }
        other => panic!("期望 Data 错误，实际 {other:?}"),
    }
}

#[test]
fn optimisation_with_ecp_is_refused() {
    let backend = RksBackend::new(registry());
    let mut input = job(cuh(1.46), "lanl2dz-cu", GridLevel::Coarse);
    input.jobtype = JobType::Opt;
    let error = backend
        .run(&input, &Silent)
        .expect_err("带 ECP 的 opt 应被拒绝");
    match error {
        EngineError::Unsupported(message) => {
            assert!(
                message.contains("ECP") && message.contains("derivative"),
                "错误信息要说明缺的是 ECP 的导数：{message}"
            );
        }
        other => panic!("期望 Unsupported，实际 {other:?}"),
    }
}
