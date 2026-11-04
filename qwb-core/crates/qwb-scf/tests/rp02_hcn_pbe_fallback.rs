//! RP-02 回归：退火（费米展宽）这一级退路**要么给出真解，要么诚实地拒绝**。
//!
//! # 被钉住的失败
//!
//! HCN / PBE / def2-SVP / `grid-coarse`，∠H–C–N = 90°（C–N 1.17 Å、C–H 1.19 Å）曾经报
//!
//! ```text
//! scf: annealing converged in 23 iterations: E = -81.7122547541 Ha
//! scf converged in 23 iterations: E = -81.7122547541 Ha
//! ```
//!
//! 并带着 `scf_converged = true` 返回。那个数不是"差一点的 SCF 解"，它是**初始 `H_core`
//! 初猜密度的能量**（同一个基组/网格上独立算得 -81.712254449435 Ha，见
//! `rp02_stale_diis_guard.rs`）。机制是：`fermi_occupations` 的化学势二分把两个分支写反了，
//! 占据数全部塌成 0（`QWB_SCF_TRACE=1` 实测 step 1–20 都是 `sum(f) = 0.000000`、
//! `E = +23.883886438`），密度变成零矩阵；展宽结束后 SCF 从零密度出发，一步落回初猜，
//! 之后「能量不再变、密度不再动」两条旧判据全过 —— 于是把初猜的能量当成收敛解返回。
//!
//! # 现在钉什么
//!
//! 同一个 rung（`smearing_steps = 20`、`T = 0.3`，就是引擎第三级退路用的参数）：
//!
//! 1. **直线几何**（180°，真值已知）：必须收敛，而且必须落在 PySCF 的 PBE 值
//!    `-93.2319416351 Ha` 的 1 mHa 之内。修复前后实测：`-80.9548467168` →
//!    `-93.231930057882`（40 步）。
//! 2. **90° 弯几何**（缺陷现场）：**要么**同样落在 PySCF 的 `-93.1721319955 Ha` 的 1 mHa
//!    之内，**要么**返回 `ScfNotConverged` —— 绝不许返回初猜能量。修复前它在第 23 步
//!    "收敛"到 -81.71，这里会直接判失败；修复后 40 步内不收敛（诚实拒绝），如实通过。
//!
//! oracle 的两行由 `scripts/rp02_oracle.py --part hcn`（WSL + PySCF 2.14.0，
//! `grids.level = 3`、`conv_tol = 1e-11`）量得：
//!
//! ```text
//! geometry   90.00  pbe/def2-svp L3 E = -93.1721319955  conv=True  nao=33 ne=14
//! geometry  180.00  pbe/def2-svp L3 E = -93.2319416351  conv=True  nao=33 ne=14
//! ```
//!
//! 容差 1 mHa 不是"刚好够"：`grid-coarse` 与 `grids.level = 3` 的求积差实测 ~1e-4 Ha，
//! 而初猜与真解差 **11.46 Ha**（弯）／**12.28 Ha**（直）。
#![allow(clippy::field_reassign_with_default)]

use std::path::PathBuf;

use qwb_engine_api::Molecule;
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

/// PySCF 2.14.0 `pbe`/`def2-svp`、`grids.level = 3`、`conv_tol = 1e-11`。
const PYSCF_BENT_90_HARTREE: f64 = -93.172_131_995_5;
/// 同上，直线几何（∠H–C–N = 180°）。
const PYSCF_LINEAR_180_HARTREE: f64 = -93.231_941_635_1;
/// 90° 弯几何上 `H_core` 初猜密度的能量（缺陷报的就是它）。
const GUESS_ENERGY_BENT_90_HARTREE: f64 = -81.712_254_449_435;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// C 在原点、N 沿 +z 1.17 Å、H 在 x–z 平面里 1.19 Å 处；`angle` 是 ∠H–C–N。
fn hcn(angle_deg: f64) -> Molecule {
    use qwb_engine_api::Atom;
    let radians = angle_deg.to_radians();
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("N", [0.0, 0.0, 1.17]),
            Atom::new("H", [1.19 * radians.sin(), 0.0, 1.19 * radians.cos()]),
        ],
        0,
        1,
    )
}

/// 引擎第三级退路的那套参数（`qwb-engine` 的 annealed 分支）。
fn annealed_options(max_iter: u32) -> RksOptions {
    let mut options = RksOptions::default();
    options.max_iter = max_iter;
    options.energy_threshold = 1.0e-9;
    options.density_threshold = 1.0e-7;
    options.smearing_steps = 20;
    options.smearing_temperature = 0.3;
    options
}

fn annealed_scf(molecule: &Molecule, max_iter: u32) -> Result<qwb_scf::rks::RksResult, String> {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp").expect("def2-svp"),
        molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), molecule)
        .expect("grid");
    let functional = registry.get_functional("pbe").expect("pbe");
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    rks_scf(
        &basis,
        molecule,
        &tensor,
        &grid,
        &terms,
        annealed_options(max_iter),
    )
    .map_err(|err| err.to_string())
}

#[test]
fn annealed_smearing_reaches_the_pyscf_pbe_energy_at_the_linear_geometry() {
    let result = annealed_scf(&hcn(180.0), 400).expect("直线几何上退火必须收敛");
    println!(
        "HCN 直线 / PBE / def2-SVP / grid-coarse（退火）: E = {:.12} Ha, E_xc = {:.12}, {} 步",
        result.energy_hartree, result.exchange_correlation_hartree, result.iterations
    );
    assert!(result.scf_converged);
    assert!(
        (result.energy_hartree - PYSCF_LINEAR_180_HARTREE).abs() < 1.0e-3,
        "退火 SCF 给了 {:.10} Ha，距 PySCF 的 PBE/def2-SVP 值 {:.10} Ha 有 {:.4} Ha；\
         -80.9548 意味着退路停在初始 H_core 初猜密度上（费米占据数全零）",
        result.energy_hartree,
        PYSCF_LINEAR_180_HARTREE,
        result.energy_hartree - PYSCF_LINEAR_180_HARTREE
    );
}

#[test]
fn the_bent_geometry_is_never_reported_as_a_converged_guess_energy() {
    // 预算 40 步：够走完 20 步展宽 + 20 步整数占据。修复前的缺陷在第 **23** 步"收敛"，
    // 所以这个预算足以把旧行为暴露出来；修复后它在 40 步内不收敛，于是诚实拒绝。
    let outcome = annealed_scf(&hcn(90.0), 40);
    match outcome {
        Ok(result) => {
            println!(
                "HCN 90° / PBE / def2-SVP / grid-coarse（退火）: E = {:.12} Ha, {} 步",
                result.energy_hartree, result.iterations
            );
            assert!(result.scf_converged);
            assert!(
                (result.energy_hartree - GUESS_ENERGY_BENT_90_HARTREE).abs() > 1.0e-6,
                "SCF 报的 {:.10} Ha 就是 H_core 初猜密度的能量 {:.10} Ha —— \
                 密度从来没离开初猜，这不是收敛解",
                result.energy_hartree,
                GUESS_ENERGY_BENT_90_HARTREE
            );
            assert!(
                (result.energy_hartree - PYSCF_BENT_90_HARTREE).abs() < 1.0e-3,
                "退火 SCF 收了，但给的是 {:.10} Ha，距 PySCF 的 {:.10} Ha 有 {:.4} Ha",
                result.energy_hartree,
                PYSCF_BENT_90_HARTREE,
                result.energy_hartree - PYSCF_BENT_90_HARTREE
            );
        }
        Err(message) => {
            println!("HCN 90° / 退火在 40 步内没有收敛，拒绝返回能量：{message}");
            assert!(
                message.contains("not converged"),
                "拒绝的理由必须是「没收敛」（引擎据此换下一级退路），实际：{message}"
            );
        }
    }
}
