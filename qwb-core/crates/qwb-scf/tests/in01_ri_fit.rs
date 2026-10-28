//! IN-01 的验收判据：RI-J 能量的**实现正确性**与**辅助基质量**分开量。
//!
//! # 为什么把原文的判据改了（`project-phase2.md` §6.3）
//!
//! IN-01 原文写「拟合误差对 H2O/def2-SVP < 1e-6 Ha（能量）」。实测发现这条把
//! 两件事混在一起了：
//!
//! | 量 | 实测 | 谁决定的 |
//! |---|---|---|
//! | `E_J^RI` 与**精确 4 中心** `E_J` 的差（def2-SVP-RI） | **6.14e-3 Ha** | 辅助基不完备（`def2-SVP-RI` 只 76 个函数） |
//! | RI-J 的 **SCF 总能量**与精确 SCF 的差 | 2.27e-3 Ha | 同上（有变分抵消） |
//! | 我们的 `E_J^RI` 与 **PySCF 同配对** `E_J^RI` 的差 | **1.4e-7 Ha** | 只剩实现的数值误差（K2 的 Å→bohr 常数） |
//!
//! 也就是说 6.14e-3 Ha 是**辅助基的性质**，不是实现错误 —— PySCF 用同一个
//! `def2-SVP`/`def2-SVP-RI` 配对给出同一个数（`scripts/pyscf_rij_reference.py`
//! 就是生成这份参考的脚本）。要求"与精确 4 中心差 < 1e-6"等于要求 `def2-SVP-RI`
//! 完备到 2e-8 的相对精度，任何实现都做不到。
//!
//! 所以判据改成两条，各自可执行、各自有外部尺子：
//!
//! 1. **实现正确性**：我们的 `E_J^RI` 与 PySCF 的同配对结果差 < 1e-6 Ha；
//! 2. **辅助基质量如实记账**：`E_J^RI − E_J` 的实测值落盘（本文件的打印与
//!    `docs/status-v2.yaml` 的 IN-01 都写 6.14e-3），并断言它**没有超过**一个
//!    宽松上界（否则说明积分或拟合真的坏了）。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::ri::{inverse_sqrt, three_center, two_center_coulomb};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("rij_energy.csv")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    )
}

struct Reference {
    aux_basis: String,
    n_aux: usize,
    e_j_exact: f64,
    e_j_ri: f64,
}

fn load_reference() -> Vec<Reference> {
    let text = std::fs::read_to_string(reference_path()).expect("rij_energy.csv");
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 7,
            "row {lineno} has {} fields",
            fields.len()
        );
        out.push(Reference {
            aux_basis: fields[2].to_string(),
            n_aux: fields[3].parse().expect("n_aux"),
            e_j_exact: fields[4].parse().expect("e_j_exact"),
            e_j_ri: fields[5].parse().expect("e_j_ri"),
        });
    }
    assert!(!out.is_empty(), "empty RI-J reference table");
    out
}

/// 直接 4 中心的 Coulomb 能量 `E_J = ½ Σ P_ij P_kl (ij|kl)`。
fn exact_coulomb_energy(
    tensor: &qwb_integrals::TwoElectronIntegrals,
    density: &qwb_linalg::Matrix,
) -> f64 {
    let n = tensor.n_functions;
    let mut energy = 0.0;
    for i in 0..n {
        for j in 0..n {
            let pij = density[(i, j)];
            if pij == 0.0 {
                continue;
            }
            for k in 0..n {
                for l in 0..n {
                    energy += 0.5 * pij * density[(k, l)] * tensor.get(i, j, k, l);
                }
            }
        }
    }
    energy
}

/// RI 的 Coulomb 能量 `E_J^RI = ½ dᵀ V⁻¹ d`，`d_P = Σ_ij P_ij (ij|P)`。
fn ri_coulomb_energy(
    aux: &qwb_basis::InternalBasis,
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
    density: &qwb_linalg::Matrix,
) -> f64 {
    let three = three_center(basis, aux, molecule).expect("3-center");
    let v = two_center_coulomb(aux, molecule).expect("(P|Q)");
    let w = inverse_sqrt(&v, 1.0e-10).expect("V^{-1/2}");
    let n = three.n_orbital;
    let n_aux = three.n_auxiliary;
    let mut d = vec![0.0_f64; n_aux];
    for (p, slot) in d.iter_mut().enumerate() {
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                sum += density[(i, j)] * three.get(i, j, p);
            }
        }
        *slot = sum;
    }
    let mut wd = vec![0.0_f64; n_aux];
    for p in 0..n_aux {
        let mut sum = 0.0;
        for q in 0..n_aux {
            sum += w[(p, q)] * d[q];
        }
        wd[p] = sum;
    }
    0.5 * wd.iter().map(|value| value * value).sum::<f64>()
}

#[test]
fn ri_coulomb_energy_matches_pyscf_and_the_basis_error_is_recorded() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let reference = load_reference();
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp").expect("def2-svp"),
        &molecule,
    )
    .expect("orbital basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.density_threshold = 1.0e-9;
    let scf = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("HF");
    println!(
        "参考 SCF：E = {:.10} Ha，{} 步",
        scf.energy_hartree, scf.iterations
    );

    let exact = exact_coulomb_energy(&tensor, &scf.density);
    let exact_reference = reference[0].e_j_exact;
    println!("E_J（4 中心）= {exact:.10}，PySCF {exact_reference:.10}");
    // 1e-6 而不是 1e-9：K2 的 `BOHR_PER_ANGSTROM` 与 PySCF 用的常数差 1.8e-8
    // （docs/choices.md 五十六），对 46 Ha 的 E_J 就是 ~1e-7 的地板。
    assert!(
        (exact - exact_reference).abs() < 1.0e-6,
        "精确 E_J 与 PySCF 不一致：{exact:.10} vs {exact_reference:.10}"
    );

    for row in &reference {
        let aux = qwb_basis::InternalBasis::build(
            registry.get_basis(&row.aux_basis).expect(&row.aux_basis),
            &molecule,
        )
        .expect("aux basis");
        assert_eq!(aux.n_functions(), row.n_aux, "辅助基函数个数不同");
        let ri = ri_coulomb_energy(&aux, &basis, &molecule, &scf.density);
        let versus_pyscf = (ri - row.e_j_ri).abs();
        let incompleteness = ri - exact;
        println!(
            "aux = {:>12}（{} 个）：E_J^RI = {ri:.10}；vs PySCF {versus_pyscf:.3e}；\
             E_J^RI − E_J = {incompleteness:+.3e}",
            row.aux_basis, row.n_aux
        );
        // 判据 1：实现正确性（外部尺子）。
        assert!(
            versus_pyscf < 1.0e-6,
            "{}：E_J^RI {ri:.10} 与 PySCF {:.10} 差 {versus_pyscf:.3e}",
            row.aux_basis,
            row.e_j_ri
        );
    }

    // def2-SVP-RI 这一行是"官方伴随辅助基"：误差量级 6e-3 Ha，必须落在宽松上界内
    // （真坏掉时会差到 1 Ha 量级，见对照组）。
    let ri_aux = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp-ri").expect("def2-svp-ri"),
        &molecule,
    )
    .expect("aux basis");
    let ri = ri_coulomb_energy(&ri_aux, &basis, &molecule, &scf.density);
    let gap = ri - exact;
    assert!(
        gap.abs() < 2.0e-2,
        "def2-SVP-RI 的拟合误差 {gap:+.3e} Ha 超出宽上界 2e-2（实测应为 6.1e-3）"
    );
    // 对照组：辅助基不对口时误差要大得多，否则说明这条判据没在起作用。
    let sto =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("aux basis");
    let sto_ri = ri_coulomb_energy(&sto, &basis, &molecule, &scf.density);
    let sto_gap = sto_ri - exact;
    println!("对照：STO-3G 当辅助基时 E_J^RI − E_J = {sto_gap:+.3e}");
    assert!(
        sto_gap.abs() > 1.0e-2,
        "STO-3G 当辅助基的误差 {sto_gap:.3e} 太小，判据失去对照意义"
    );
}
