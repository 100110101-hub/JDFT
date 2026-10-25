//! T6 判据 3：**B3LYP 的解析梯度与有限差分一致（< 1e-6 Ha/Bohr）**。
//!
//! 这是 `docs/status.yaml` 里 T6 最后一条未通过的判据。B3LYP 同时压着解析梯度里所有
//! 容易写错的环节：
//!
//! * `α_hf = 0.20` 的缩放（2e 交换骨架项 + `F` 里的 `−½αK`）—— 纯泛函必须当它是 0；
//! * `v_xc` 进 Pulay 的 `W = ½P F P`；
//! * `∂E_xc/∂R|_P` 的 σ 通道（AO 二阶导数）；
//! * LYP 的 `∇²ρ` 通道（AO 拉普拉斯量再对中心求一次导）。
//!
//! 体系用 H2/STO-3G（2 个基函数）：判据是"公式对不对"，不是"体系够不够大"。
//! 网格用 `grid-normal`（输入文件的默认值）—— `grid-coarse` 上残留的 2.3e-6 是**没算网格
//! 权重导数**造成的，`b3lyp_gradient_error_per_grid_level`（`#[ignore]` 慢档）把四个等级的
//! 实测数都记下来了：coarse 2.26e-6 / normal 7.3e-8 / fine 2.6e-8 / veryfine 2.3e-8。
//!
//! 注意 B3LYP 是**程序化构造**的（数据层不收 `b3lyp.yaml`：LYP 要 `∇²ρ`，
//! 而 K1 的 phase-1 语义门禁冻结了这一点），权重与 `d12_b3lyp.rs` 完全一致。
#![allow(clippy::field_reassign_with_default)]
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_analytic_gradient, rks_scf};

/// B3LYP 的 HF 混合系数（`TaskOptions.hf_exchange` 的默认含义）。
const B3LYP_HF: f64 = 0.20;
const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn h2_stretched() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.62]),
            Atom::new("H", [0.0, 0.0, 0.62]),
        ],
        0,
        1,
    )
}

fn b3lyp_terms() -> Vec<Box<dyn qwb_xc::FunctionalTerm>> {
    let mut p: BTreeMap<String, f64> = BTreeMap::new();
    p.insert("slater_alpha".to_string(), 2.0 / 3.0);
    p.insert("b88_beta".to_string(), 0.0042);
    p.insert("lyp_a".to_string(), 0.049_18);
    p.insert("lyp_b".to_string(), 0.132);
    p.insert("lyp_c".to_string(), 0.2533);
    p.insert("lyp_d".to_string(), 0.349);
    p.insert("vwn5_p_A".to_string(), 0.0310907);
    p.insert("vwn5_p_x0".to_string(), -0.409286);
    p.insert("vwn5_p_b".to_string(), 13.072);
    p.insert("vwn5_p_c".to_string(), 42.7198);
    vec![
        qwb_xc::scale_term(qwb_xc::build_term("LdaSlaterX", &p).expect("slater"), 0.08),
        qwb_xc::scale_term(qwb_xc::build_term("GgaB88X", &p).expect("b88"), 0.72),
        qwb_xc::scale_term(qwb_xc::build_term("GgaLypC", &p).expect("lyp"), 0.81),
        qwb_xc::scale_term(qwb_xc::build_term("LdaSvwn5", &p).expect("vwn5"), 0.19),
    ]
}

/// 解析梯度与手工中心差分的（最大差, 数值参照自身散布）。
fn analytic_and_numeric(delta: f64, grid_level: &str) -> (f64, f64) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2_stretched();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_level).expect("grid"), &molecule)
        .expect("grid");
    let terms = b3lyp_terms();
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.hf_exchange = B3LYP_HF;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");

    let analytic = rks_analytic_gradient(
        &basis,
        &molecule,
        &grid,
        &one.core_hamiltonian,
        &tensor,
        &result.density,
        &terms,
        B3LYP_HF,
    )
    .expect("analytic");

    // 数值参照：B3LYP 不在数据层（LYP 要 ∇²ρ），所以用同一条 SCF 参数手工做中心差分，
    // 保证与解析侧是**同一个能量泛函**。两个步长各算一次，用来估计数值参照自身的散布。
    let fine_delta = delta * 0.8;
    let mut worst = 0.0_f64;
    let mut scatter = 0.0_f64;
    #[allow(clippy::needless_range_loop)] // 下面同时要 atom/axis 与 analytic 的下标
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let mut numeric = [0.0_f64; 2];
            for (index, step) in [delta, fine_delta].iter().enumerate() {
                let mut energies = [0.0_f64; 2];
                for (slot, sign) in [-1.0_f64, 1.0].iter().enumerate() {
                    let mut moved = molecule.clone();
                    moved.atoms[atom].position_angstrom[axis] += sign * step * ANGSTROM_PER_BOHR;
                    let moved_basis = qwb_basis::InternalBasis::build(
                        registry.get_basis("sto-3g").expect("sto-3g"),
                        &moved,
                    )
                    .expect("basis");
                    let moved_tensor =
                        qwb_integrals::two_electron::two_electron(&moved_basis, &moved)
                            .expect("2e");
                    let moved_grid = qwb_grid::molecular_grid(
                        registry.get_grid(grid_level).expect("grid"),
                        &moved,
                    )
                    .expect("grid");
                    let mut scf_options = RksOptions::default();
                    scf_options.max_iter = 400;
                    scf_options.energy_threshold = 1.0e-10;
                    scf_options.hf_exchange = B3LYP_HF;
                    let outcome = rks_scf(
                        &moved_basis,
                        &moved,
                        &moved_tensor,
                        &moved_grid,
                        &terms,
                        scf_options,
                    )
                    .expect("scf moved");
                    energies[slot] = outcome.energy_hartree;
                }
                numeric[index] = (energies[1] - energies[0]) / (2.0 * step);
            }
            println!(
                "{grid_level} atom {atom} axis {axis}: 解析 {:>13.9}  数值(δ={delta}) {:>13.9}  数值(δ={fine_delta}) {:>13.9}  差 {:.2e} / {:.2e}",
                analytic[atom][axis],
                numeric[0],
                numeric[1],
                (analytic[atom][axis] - numeric[0]).abs(),
                (analytic[atom][axis] - numeric[1]).abs()
            );
            worst = worst.max(
                (analytic[atom][axis] - numeric[0])
                    .abs()
                    .max((analytic[atom][axis] - numeric[1]).abs()),
            );
            scatter = scatter.max((numeric[0] - numeric[1]).abs());
        }
    }
    (worst, scatter)
}

#[test]
fn b3lyp_analytic_gradient_matches_finite_differences() {
    let (worst, scatter) = analytic_and_numeric(1.0e-3, "grid-normal");
    println!("B3LYP/grid-normal：解析 vs 数值最大差 {worst:.2e}；数值参照自身散布 {scatter:.2e}");
    assert!(
        worst < 1.0e-6,
        "B3LYP 解析与有限差分最大差 {worst:.2e}（T6 判据要求 < 1e-6）"
    );
}

/// 慢档（约 3 分钟）：四个网格等级下的解析–数值差，用来量化「没算网格权重导数」的残留。
///
/// 实测（H2/STO-3G/B3LYP，δ = 1e-3 Bohr）：
/// coarse **2.26e-6** / normal **7.29e-8** / fine **2.64e-8** / veryfine **2.25e-8**。
/// 网格越细残留越小 —— 这是"权重导数项"的特征；真正的公式漏项不会随网格收敛。
#[test]
#[ignore = "慢档（约 3 分钟）：四个网格等级的 B3LYP 梯度对照，用来量化网格权重导数的残留"]
fn b3lyp_gradient_error_per_grid_level() {
    for level in ["grid-coarse", "grid-normal", "grid-fine", "grid-veryfine"] {
        let (worst, _) = analytic_and_numeric(1.0e-3, level);
        println!("{level}: B3LYP 解析 vs 数值最大差 {worst:.2e}");
    }
}
