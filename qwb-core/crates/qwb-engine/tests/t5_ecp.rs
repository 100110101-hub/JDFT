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
//! 所以：参考 CSV 现在装的是**PySCF 的答案**（`scripts/pyscf_zno_lanl2dz_reference.py`
//! 生成、`--check` 可重建），而下面两个 `#[ignore]` 的 ZnO 测试**现在是红的** ——
//! 它们是真正的验收判据，要等那个 ECP 缺陷被单独修掉。另：oracle 的位能曲线最低点在
//! **1.77 Å**（不是原来的 1.85 Å），1.85 Å 与「1.85 是平衡点」同样是伪解时代的结论。
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
    // （docs/choices.md 五十一），PBE 总能量整体移动 −1.6e-5 Ha，旧钉
    // -213.9067554378 作废。
    assert!(
        (result.energy_hartree - (-213.906_771_935_1)).abs() < 1.0e-6,
        "CuH/LANL2DZ/PBE 能量 {:.10} 与钉住值 -213.9067719351 不一致",
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
        // 第 421 轮：旧钉值 -213.9069054186 是 `pbe_mu = 0.21951`（5 位）时代的数。
        // `d852bb2`（SC-01b）把 `pbe_mu` 换成精确值 0.21951497276451704（= beta*pi^2/3,
        // beta = 0.066725），相对变化 2.27e-5，PBE 梯度项随之把 CuH 的总能量推了 1.64e-5 Ha。
        // **这不是回归**：把数据改回 0.21951 重跑，本断言复现旧钉值（< 1e-6），
        // 所以因果关系是量出来的而不是猜的（choices 二百〇九）。
        (first.energy_hartree - (-213.906_921_804_5)).abs() < 1.0e-6,
        "grid-normal 能量 {:.10} 与钉住值 -213.9069218045 不一致",
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
    // 预算 200 步（原来是 25）：25 步是照着"退火 23 步收敛"那个**伪解**定的。
    // RP-02 修好之后这个体系在 25 步内连一级退路都走不完。改成 200 是为了让"不是预算
    // 的问题"可以被实测证明 —— 实测（max_iter = 200）四级退路全都不收敛：
    // 单初猜 200 步在 -119…-121 Ha 之间振荡、多初猜 5×60 不行、费米展宽 200 步不行、
    // 阻尼 Roothaan 进极限环（dE → 7e-9 而 max|dP| 卡在 4.445e-1）。
    // 根因是 Zn 的 LANL2DZ ECP 的 l = 2 通道（见文件头），不是迭代预算。
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

/// 读 `tests/reference/zno_lanl2dz_pbe_ecp.csv` 里某个量的 `(值, 容差)`。
///
/// CSV 现在带 `source_doi,source_url` 两列（DA-08 的来源约定，见
/// `scripts/pyscf_zno_lanl2dz_reference.py`），所以按**前四列的列名**取值，
/// 多余列不参与比较。
fn reference(quantity: &str) -> (f64, f64) {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/zno_lanl2dz_pbe_ecp.csv");
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
    panic!("{quantity} 不在参考 csv 里");
}

#[test]
#[ignore = "慢档（ZnO/LANL2DZ 的 SCF 跑满 200 步退路；当前**红**，等 Zn 的 LANL2DZ ECP l=2 通道修掉）：ZnO 单点 vs PySCF 参考 csv"]
fn zno_lanl2dz_single_point_matches_the_reference() {
    // 判据是 T5 的：与 `zno_lanl2dz_pbe_ecp.csv` 的 PySCF 值差 < 5 mHa。
    // 现在这个判据**不通过**，而且不是容差问题：RP-02 修好二分方向之后，ZnO/LANL2DZ
    // 的四级 SCF 退路全都不收敛（见文件头与 CSV 的 note），根因是 Zn 的 LANL2DZ ECP
    // 的 l=2 通道被弱化 ~8 倍（3d¹⁰ 没被束缚住，总能比 oracle 高约 20 Ha）。
    // 保留这个测试原样（不放松容差、不改成"预期失败"）——它就是要红到 ECP 被修好。
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
    assert!(
        log.contains("annealed smearing"),
        "ZnO 应该走退火这条退路，日志里没有：{log}"
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
#[ignore = "慢档（ZnO 平衡点附近再跑两次 SCF）：位能曲线的最低点在 1.77 Å（PySCF 口径）；当前**红**，同上"]
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
