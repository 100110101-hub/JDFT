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
//! | grid-coarse | −196.6931697289 | 门禁版 |
//! | grid-normal | −196.6933745514 | 慢档（`#[ignore]`） |
//!
//! 两个数值是 **internal-independent-path**（§7.0 的 fallback）：PBE/LANL2DZ 的 CuH
//! 在可引用表里查不到，它们是回归钉 —— 钉的是"ECP 接线不会被后续改动悄悄改掉"。
//! 2026-11 修掉 ECP 的 `r` 幂次约定与通道语义之后它们整体移动（旧值
//! `−213.9067719351` / `−213.9069218045`），重钉之后**每一条都对着外部标尺**：
//! `tests/reference/cuh_lanl2dz_pbe_ecp.csv` 是 PySCF 2.14.0 在同一几何/基组/ECP/泛函下
//! 的复算值（`−196.6933563155` / `−196.6933577633`），两条测试都断言与它差 < 5 mHa
//! （实测 1.87e-4 / 1.68e-5 Ha）。回归钉是本项目自己的输出（1e-6），外部标尺是 PySCF
//! （5 mHa），两件事分开记 —— 这样"改小钉子"不会顺手把外部对照一起改没。
//!
//! # ZnO：这一节的旧结论（"退火之后收敛"）**已被推翻**，证据在下一节
//!
//! 旧记录是：单初猜 DIIS 不收敛（400/500 步、多初猜、阻尼、能级移动、四个网格、
//! PBE/SVWN5 都不行），根因是**闭壳层受限的 O 原子本身是伪解**（O 的 ³P 有两个单占 p
//! 轨道），core guess 从一个伪组态出发、在几个组态之间跳（`QWB_SCF_TRACE=1` 实测
//! 能量在 −119…−121 Ha 之间振荡）；修法是费米展宽退火，23 步"收敛"，且结果与展宽温度
//! 无关。其中"收敛"和"温度无关"两句是 RP-02 缺陷的症状，不是解的性质。
//! 另外，旧记录写「根因不是 ECP 矩阵元（I-05 已验到 1e-14）」—— 那句也不成立：I-05 验的是
//! 本项目两条实现（解析 vs 求积）互相一致，**没有**和外部实现比过；与 PySCF 比，
//! Zn 的 LANL2DZ ECP 的 l=2 通道被弱化了约 8 倍（见下一节）。
//!
//! 能量钉在 `tests/reference/zno_lanl2dz_pbe_ecp.csv`（5 mHa 容差是 T5 判据给的）。
//!
//! # 2026-11（RP-02）：旧的三行参考值是**初猜能量**，不是 ZnO 的能量
//!
//! 原先这里写「费米展宽退火 23 步收敛；且结果与展宽温度/步数无关（T = 0.1–0.5、
//! 步数 10–80 给同一个能量，逐位相同）」。那正是 RP-02 缺陷的**签名**而不是证据：
//! `fermi_occupations` 的化学势二分把两个分支写反，占据数全部塌成 0，密度变成零矩阵，
//! 展宽结束后 SCF 从零密度出发落回 `H_core` 初猜，「能量不再变、密度不再动」两条旧判据
//! 于是全过 —— 初猜能量与展宽温度**无关**，所以"温度无关"恰恰说明它没在解 SCF。
//!
//! 实测（RP-02 的证据链）：旧的三行参考值就是初猜能量（`-103.7068819011` /
//! `-103.7069781304` / `-103.7779550527`，与旧行差 2.5e-5 Ha），而 PySCF 2.14.0 的
//! PBE/LANL2DZ 是 `-140.9635394365` / `-140.9656054282`（r = 1.70 / 1.77 Å）。
//!
//! 修掉二分方向之后，本项目的 SCF 在 ZnO/LANL2DZ 上**不收敛**（四级退路全都不行，
//! 见 CSV 的 note），根因是一个独立于 RP-02 的既有缺陷：**Zn 的 LANL2DZ ECP 的 l=2（d）
//! 非局域通道被弱化了约 8 倍**（同一套参数：`Tr(S⁻¹V_ecp)` 本项目 +1.331 vs PySCF -21.701；
//! `S⁻¹V_ecp` 最负的 5 个本征值 -0.78…-0.14 vs -6.98…-6.27）。3d¹⁰ 没被束缚住，总能比
//! oracle 高约 20 Ha。
//!
//! 所以：参考 CSV 装的是**PySCF 的答案**（`scripts/pyscf_zno_lanl2dz_reference.py`
//! 生成、`--check` 可重建）。另：oracle 的位能曲线最低点在
//! **1.77 Å**（不是原来的 1.85 Å），1.85 Å 与「1.85 是平衡点」同样是伪解时代的结论。
//!
//! # 2026-11（ECP）：那个 ECP 缺陷已经修好，ZnO 两个慢档测试**转绿**
//!
//! 根因是两层，都在 `qwb_scf::ecp` + 数据层：
//!
//! 1. **`r` 幂次约定**：BSE 的 `r_exponents` 是 `r^(n-2)`，数据里却声明成 `r^n`，
//!    加载器也没有归一 —— 每一项被整体抬 `r²`，局域通道的 `-Z_core/r` 变成 `-Z_core·r`；
//! 2. **通道语义**：数据里的 `l < L` 块**已经是** `ΔV_l = V_l − V_L`（PySCF/libcint 的
//!    `ECPscalar` = `V_local + Σ_l V_l P_l`，实测差 8.9e-16），实现里又减了一次 `V_L`。
//!
//! 两条加起来把 Zn 的 d 通道压成 1/18（`S⁻¹V_ecp` 的 d 子块对角元 `-0.283` 而 PySCF
//! 是 `-5.166`），3d¹⁰ 没被束缚住。修好之后：ZnO/LANL2DZ/PBE 在**单初猜 DIIS 第 28 步**
//! 就收敛（不再需要退火退路），r = 1.70 Å 得 `-140.9636049221`（oracle `-140.9635394365`，
//! 差 6.6e-5 Ha），r = 1.77 Å 得 `-140.9655952885`（oracle `-140.9656054282`，差 1.0e-5）。
//! 逐通道的对外核对钉在 `qwb-scf/tests/i05_ecp_oracle.rs`（Cu/Zn 各通道 × 1460 个矩阵元，
//! 最大差 4.6e-13）。
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
    // 2026-11-06 重钉：把 `pbe_mu` 从 6 位的 0.21951 换成精确的 βπ²/3
    // （docs/choices.md 五十一），PBE 总能量整体移动 −1.6e-5 Ha。
    // 2026-11 ECP 重钉：ECP 的 `r` 幂次约定与通道语义修好之后，CuH 的能量从
    // −213.9067719351 移到 −196.6931697289（旧钉是"ECP 只有一半"时代的数）。
    // 这次重钉有**外部标尺**：PySCF 2.14.0 在同一几何/基组/ECP/PBE 下是
    // −196.6933563155（`tests/reference/cuh_lanl2dz_pbe_ecp.csv`，下面第 ③ 条断言 5 mHa）。
    assert!(
        (result.energy_hartree - (-196.693_169_728_9)).abs() < 1.0e-6,
        "CuH/LANL2DZ/PBE 能量 {:.10} 与钉住值 -196.6931697289 不一致",
        result.energy_hartree
    );

    // ③ 外部标尺（T5 的 5 mHa）：同一个数必须落在 PySCF 上。
    let (oracle, tolerance) =
        reference_in("cuh_lanl2dz_pbe_ecp.csv", "cuh_total_energy_grid_coarse");
    let difference = (result.energy_hartree - oracle).abs();
    println!("CuH grid-coarse：PySCF {oracle:.10}，差 {difference:.2e}（容差 {tolerance}）");
    assert!(
        difference < tolerance,
        "CuH 能量 {:.10} 与 PySCF {oracle:.10} 差 {difference:.2e}（容差 {tolerance}）",
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
#[ignore = "慢档：CuH/LANL2DZ 在 grid-normal 上的完整单点（跑两次验逐位可重复）；数字已实测 E = -196.6933745514 Ha"]
fn cuh_lanl2dz_grid_normal_is_pinned() {
    let backend = RksBackend::new(registry());
    let first = backend
        .run(&job(cuh(1.46), "lanl2dz-cu", GridLevel::Normal), &Silent)
        .expect("grid-normal 单点");
    assert!(
        // 第 421 轮：旧钉值 -213.9069054186 是 `pbe_mu = 0.21951`（5 位）时代的数。
        // `d852bb2`（SC-01b）把 `pbe_mu` 换成精确值 0.21951497276451704（= beta*pi^2/3,
        // beta = 0.066725），相对变化 2.27e-5，PBE 梯度项随之把 CuH 的总能量推了 1.64e-5 Ha。
        // **这不是回归**：把数据改回 0.21951 重跑，本断言复现旧钉值（< 1e-6），
        // 所以因果关系是量出来的而不是猜的（choices 二百〇九）。
        // 2026-11 ECP 重钉：旧值 -213.9069218045 → -196.6933745514（ECP 的 `r` 幂次约定
        // 与通道语义同时修好）。外部标尺：PySCF 2.14.0 / PBE / grids.level = 2 是
        // -196.6933577633，见下面第 ③ 条 5 mHa 断言。
        (first.energy_hartree - (-196.693_374_551_4)).abs() < 1.0e-6,
        "grid-normal 能量 {:.10} 与钉住值 -196.6933745514 不一致",
        first.energy_hartree
    );
    println!(
        "CuH/LANL2DZ/PBE (grid-normal): E = {:.10} Ha",
        first.energy_hartree
    );

    // ③ 外部标尺（T5 的 5 mHa）。
    let (oracle, tolerance) =
        reference_in("cuh_lanl2dz_pbe_ecp.csv", "cuh_total_energy_grid_normal");
    let difference = (first.energy_hartree - oracle).abs();
    println!("CuH grid-normal：PySCF {oracle:.10}，差 {difference:.2e}（容差 {tolerance}）");
    assert!(
        difference < tolerance,
        "grid-normal 能量 {:.10} 与 PySCF {oracle:.10} 差 {difference:.2e}（容差 {tolerance}）",
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

// ---------------------------------------------------------------------------
// ZnO：T5 的第二个体系。判据是"与 reference csv 差 < 5 mHa"。
// ---------------------------------------------------------------------------

fn zno(distance_angstrom: f64) -> Molecule {
    Molecule::new(
        vec![
            Atom::new("Zn", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, distance_angstrom]),
        ],
        0,
        1,
    )
}

fn zno_job(distance_angstrom: f64, grid: GridLevel) -> JobInput {
    let mut options = TaskOptions::default();
    // 预算 200 步。这个数不是从"退火 23 步收敛"那个伪解定的：RP-02 修好之后，
    // ECP 还没修时实测（max_iter = 200）四级退路全都不收敛（单初猜 200 步在 -119…-121 Ha
    // 之间振荡、多初猜 5×60 不行、费米展宽 200 步不行、阻尼 Roothaan 进极限环）。
    // ECP 修好之后单初猜 DIIS **28 步**收敛，200 步的预算只是余量。
    options.scf_max_iter = 200;
    options.ecp_name = Some("lanl2dz-zn".to_string());
    JobInput::new(
        zno(distance_angstrom),
        "lanl2dz",
        "pbe",
        grid,
        "vacuum",
        JobType::Sp,
        options,
    )
}

/// 读 `tests/reference/<file>` 里某个量的 `(值, 容差)`。
///
/// CSV 带 `source_doi,source_url` 两列（DA-08 的来源约定），所以按**列名**取值，
/// 多余列不参与比较。
fn reference_in(file: &str, quantity: &str) -> (f64, f64) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference")
        .join(file);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("读不到 {}: {err}", path.display()));
    let mut value_index = None;
    let mut tolerance_index = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if value_index.is_none() {
            value_index = fields.iter().position(|field| *field == "value_hartree");
            tolerance_index = fields
                .iter()
                .position(|field| *field == "tolerance_hartree");
            assert!(
                value_index.is_some() && tolerance_index.is_some() && fields[0] == "quantity",
                "参考 csv 的表头不对：{line}"
            );
            continue;
        }
        if fields[0] == quantity {
            return (
                fields[value_index.expect("value 列")]
                    .parse()
                    .expect("value"),
                fields[tolerance_index.expect("tolerance 列")]
                    .parse()
                    .expect("tolerance"),
            );
        }
    }
    panic!("{quantity} 不在 {file} 里");
}

/// `tests/reference/zno_lanl2dz_pbe_ecp.csv` 里某个量的 `(值, 容差)`。
fn reference(quantity: &str) -> (f64, f64) {
    reference_in("zno_lanl2dz_pbe_ecp.csv", quantity)
}

#[test]
#[ignore = "慢档（ZnO/LANL2DZ 的 SCF）：ZnO 单点 vs PySCF 参考 csv（ECP 修好后已转绿）"]
fn zno_lanl2dz_single_point_matches_the_reference() {
    // 判据是 T5 的：与 `zno_lanl2dz_pbe_ecp.csv` 的 PySCF 值差 < 5 mHa。
    // 2026-11 之前这个判据不通过：Zn 的 LANL2DZ ECP 的 d 通道被压成 1/18，3d¹⁰ 没被
    // 束缚住，总能比 oracle 高约 20 Ha，SCF 停在伪态上。ECP 修好之后单初猜 DIIS 28 步
    // 收敛，实测 -140.9636049221（oracle -140.9635394365，差 6.6e-5 Ha）。
    let backend = RksBackend::new(registry());
    let reporter = Collecting::new();
    let result = backend
        .run(&zno_job(1.70, GridLevel::Coarse), &reporter)
        .expect("ZnO/LANL2DZ/PBE 单点");
    println!(
        "ZnO/LANL2DZ/PBE (r = 1.70 Å, grid-coarse): E = {:.10} Ha",
        result.energy_hartree
    );
    let log = reporter.joined();
    // 旧断言是"应该走费米展宽退火那条退路"—— 那是 RP-02 + ECP 双缺陷时代的诊断，
    // 不是判据。ECP 修好之后第一级（单初猜 DIIS）就收敛，退火反而不该被触发；
    // 所以这里改成断言**更强**的事实：收敛了、而且没有落到退路上。
    assert!(result.scf_converged, "ZnO 没有收敛：{log}");
    assert!(
        log.contains("scf converged") && !log.contains("annealed smearing"),
        "ZnO 应该在单初猜 DIIS 上直接收敛（ECP 修好后不再需要退火退路）：{log}"
    );

    let (expected, tolerance) = reference("zno_total_energy_r1p70_coarse");
    let difference = (result.energy_hartree - expected).abs();
    println!("参考 {expected:.10} ± {tolerance}，差 {difference:.2e}");
    assert!(
        difference < tolerance,
        "ZnO 能量 {:.10} 与参考 {expected:.10} 差 {difference:.2e}（容差 {tolerance}）",
        result.energy_hartree
    );
}

#[test]
#[ignore = "慢档（ZnO 平衡点附近再跑两次 SCF）：位能曲线的最低点在 1.77 Å（PySCF 口径）；ECP 修好后已转绿"]
fn zno_equilibrium_is_at_one_point_seven_seven() {
    // 这条测试原来叫 `..._at_one_point_eight_five`，断言"1.85 Å 的能量低于 1.70 Å"。
    // 那个结论和旧的三行参考值一样是 RP-02 伪解的产物：PySCF 2.14.0 的 PBE/LANL2DZ
    // 位能曲线（`scripts/rp02_oracle.py --part zno`，1.60–1.95 Å 扫描 + 1.74–1.81 细扫）
    // 的最低点在 **1.77 Å**：
    //     1.76 = -140.9656042058、1.77 = -140.9656054282、1.78 = -140.9655225639，
    //     而 1.70 = -140.9635394365 **低于** 1.85 = -140.9629552417。
    // 所以断言改成"1.77 Å 的能量低于 1.70 Å"，几何也换成 1.77 Å（参考 csv 的
    // `zno_total_energy_equilibrium` 行就是这个点）。
    // 实验值 1.704 Å；oracle 的 1.77 Å 与 LANL2DZ 对 ZnO 的已知系统偏差同量级。
    // 2026-11（ECP 修好后实测）：r=1.70 → -140.9635194044、r=1.77 → -140.9655952885，
    // 与 oracle 的 -140.9656054282 差 1.0e-5 Ha。
    let backend = RksBackend::new(registry());
    let loose = backend
        .run(&zno_job(1.70, GridLevel::Normal), &Silent)
        .expect("1.70 Å");
    let tight = backend
        .run(&zno_job(1.77, GridLevel::Normal), &Silent)
        .expect("1.77 Å");
    println!(
        "ZnO: r=1.70 → {:.10} Ha，r=1.77 → {:.10} Ha",
        loose.energy_hartree, tight.energy_hartree
    );
    assert!(
        loose.scf_converged && tight.scf_converged,
        "ZnO 的两个点都必须收敛（1.70: {}, 1.77: {}）",
        loose.scf_converged,
        tight.scf_converged
    );
    assert!(
        tight.energy_hartree < loose.energy_hartree,
        "1.77 Å 的能量没有低于 1.70 Å（{:.10} vs {:.10}）",
        tight.energy_hartree,
        loose.energy_hartree
    );
    let (expected, tolerance) = reference("zno_total_energy_equilibrium");
    assert!(
        (tight.energy_hartree - expected).abs() < tolerance,
        "平衡点能量 {:.10} 与参考 {expected:.10} 差超过 {tolerance}",
        tight.energy_hartree
    );
}
