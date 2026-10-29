//! RS-08 的结构验收：无迹四极矩的**原点变换律**、μ=0 的原点不变性、迹为零。
//!
//! 判据（§6.7 RS-08）要求「原点多选一致性」与「H2O 的四极矩与文献差 < 5%」。
//! 这里先钉前两条：traceless 四极矩在原点上不是不变的（除非 μ=0），正确的性质是
//!
//! ```text
//! Θ(O+R) − Θ(O) = −½[ 3(R_i μ_j + R_j μ_i) − 2(R·μ) δ_ij ]     （中性体系）
//! ```
//!
//! 对 μ=0 的体系（CH4）这个式子退化为**严格不变**——那正是判据想说的东西。
#![allow(clippy::needless_range_loop)]
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_props::{dipole_debye, quadrupole_traceless_debye_angstrom};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

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

fn carbon_dioxide() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.160]),
            Atom::new("O", [0.0, 0.0, -1.160]),
        ],
        0,
        1,
    )
}

fn methane() -> Molecule {
    // 正四面体，μ = 0
    let d = 0.629_0_f64;
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [d, d, d]),
            Atom::new("H", [-d, -d, d]),
            Atom::new("H", [-d, d, -d]),
            Atom::new("H", [d, -d, -d]),
        ],
        0,
        1,
    )
}

fn density(registry: &Registry, molecule: &Molecule) -> (InternalBasis, qwb_linalg::Matrix) {
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), molecule).unwrap();
    let tensor = two_electron(&basis, molecule).unwrap();
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), molecule).unwrap();
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options).unwrap();
    (basis, result.density)
}

#[test]
fn rs08_quadrupole_trace_and_symmetry() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let (basis, p) = density(&registry, &molecule);
    let theta =
        quadrupole_traceless_debye_angstrom(&basis, &molecule, &p, [0.0, 0.0, 0.0]).unwrap();
    let trace = theta[0][0] + theta[1][1] + theta[2][2];
    println!("Θ (原点 = 全零):");
    for row in &theta {
        println!("  [{:+.9}, {:+.9}, {:+.9}]", row[0], row[1], row[2]);
    }
    println!("trace = {trace:+.3e}");
    assert!(
        trace.abs() < 1.0e-12,
        "traceless quadrupole has trace {trace:.3e}"
    );
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (theta[i][j] - theta[j][i]).abs() < 1.0e-12,
                "quadrupole not symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn rs08_quadrupole_follows_the_origin_shift_law() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let (basis, p) = density(&registry, &molecule);
    let origin = [0.0, 0.0, 0.0];
    let shift = [0.3_f64, 0.2, 0.1];
    let theta = quadrupole_traceless_debye_angstrom(&basis, &molecule, &p, origin).unwrap();
    let shifted = quadrupole_traceless_debye_angstrom(&basis, &molecule, &p, shift).unwrap();
    let mu = dipole_debye(&basis, &molecule, &p).unwrap();
    let mu_dot_r = mu[0] * shift[0] + mu[1] * shift[1] + mu[2] * shift[2];
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let predicted = theta[i][j]
                - 0.5
                    * (3.0 * (shift[i] * mu[j] + shift[j] * mu[i])
                        - 2.0 * if i == j { mu_dot_r } else { 0.0 });
            let difference = (shifted[i][j] - predicted).abs();
            worst = worst.max(difference);
            println!(
                "  ({i},{j}): 平移后 {:+.9}  变换律预测 {:+.9}  差 {difference:.2e}",
                shifted[i][j], predicted
            );
        }
    }
    println!(
        "μ = [{:+.6}, {:+.6}, {:+.6}] D，最大差 {worst:.3e}",
        mu[0], mu[1], mu[2]
    );
    assert!(
        worst < 1.0e-7,
        "origin shift law violated by {worst:.3e} D·Å"
    );
}

/// T_d 对称性 ⇒ CH4 的无迹四极矩**恒为零**（这是解析期望，不是拟合）。
/// 这也顺带说明：判据里「μ=0 时原点不变」在 CH4 上是平凡的，真正有内容的是 CO2。
#[test]
fn rs08_methane_quadrupole_vanishes_by_symmetry() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = methane();
    let (basis, p) = density(&registry, &molecule);
    let mu = dipole_debye(&basis, &molecule, &p).unwrap();
    let mu_norm = (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]).sqrt();
    let theta = quadrupole_traceless_debye_angstrom(&basis, &molecule, &p, [0.0, 0.0, 0.0])
        .expect("quadrupole");
    let mut largest = 0.0_f64;
    for row in &theta {
        for value in row {
            largest = largest.max(value.abs());
        }
    }
    println!("CH4 μ = {mu_norm:.3e} D，|Θ_max| = {largest:.3e} D·Å");
    assert!(
        mu_norm < 1.0e-9,
        "methane dipole must vanish, got {mu_norm:.3e}"
    );
    assert!(
        largest < 1.0e-10,
        "T_d symmetry requires a vanishing traceless quadrupole, got {largest:.3e}"
    );
}

/// **判据本体**：μ = 0 且有四极矩的体系（CO2，D∞h）在原点平移下不变。
#[test]
fn rs08_quadrupole_is_origin_invariant_for_co2() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.160]),
            Atom::new("O", [0.0, 0.0, -1.160]),
        ],
        0,
        1,
    );
    let (basis, p) = density(&registry, &molecule);
    let mu = dipole_debye(&basis, &molecule, &p).unwrap();
    let mu_norm = (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]).sqrt();
    let theta = quadrupole_traceless_debye_angstrom(&basis, &molecule, &p, [0.0, 0.0, 0.0])
        .expect("quadrupole");
    let shifted = quadrupole_traceless_debye_angstrom(&basis, &molecule, &p, [0.31, -0.27, 0.18])
        .expect("quadrupole");
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((theta[i][j] - shifted[i][j]).abs());
            scale = scale.max(theta[i][j].abs());
        }
    }
    println!(
        "CO2 μ = {mu_norm:.3e} D，Θ_zz = {:+.9} D·Å，Θ_xx = {:+.9}，原点平移差 {worst:.3e}（尺度 {scale:.3e}）",
        theta[2][2], theta[0][0]
    );
    assert!(mu_norm < 1.0e-9, "CO2 dipole must vanish");
    assert!(
        worst / scale < 1.0e-6,
        "traceless quadrupole must be origin invariant for μ = 0, relative {:.3e}",
        worst / scale
    );
}

/// 判据第二条：**与外部基准的差 < 5%**。
///
/// 基准用 PySCF 的 `int1e_rr`（`scripts/pyscf_quadrupole_reference.py` 生成
/// `tests/reference/quadrupole.csv`，原子单位 e·bohr²）。为什么不用文献值：
/// 不同文献的 traceless 约定/单位/原点各不相同，本仓库没有逐条核对过的表格；
/// 独立实现是同一物理量更可靠的基准（理由记在 `docs/choices.md` 六十九）。
#[test]
fn rs08_quadrupole_matches_the_pyscf_reference() {
    const DEBYE_PER_BOHR: f64 = 2.541_746_632_031_1;
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecules = [("water", water()), ("co2", carbon_dioxide()), ("methane", methane())];
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference/quadrupole.csv");
    let text = std::fs::read_to_string(&path).expect("quadrupole.csv");
    let mut checked = 0usize;
    let mut worst = 0.0_f64;
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let name = fields[0].trim();
        let component = fields[1].trim();
        let reference: f64 = fields[2].trim().parse().expect("reference value");
        let molecule = molecules
            .iter()
            .find(|(label, _)| *label == name)
            .map(|(_, molecule)| molecule)
            .unwrap_or_else(|| panic!("unknown system {name} in the reference csv"));
        let (basis, density) = density(&registry, molecule);
        let theta =
            quadrupole_traceless_debye_angstrom(&basis, molecule, &density, [0.0, 0.0, 0.0])
                .expect("quadrupole");
        let i = component.as_bytes()[0] as usize - b'0' as usize;
        let j = component.as_bytes()[1] as usize - b'0' as usize;
        let ours = theta[i][j] / (DEBYE_PER_BOHR * ANGSTROM_PER_BOHR);
        let scale = reference.abs().max(1.0e-6);
        let relative = (ours - reference).abs() / scale;
        worst = worst.max(relative);
        checked += 1;
        println!(
            "  {name} ({i}{j}): ours {ours:+.10} a.u.  PySCF {reference:+.10}  相对差 {relative:.2e}"
        );
    }
    println!("checked {checked} components, worst relative difference {worst:.2e}");
    assert!(checked >= 12, "expected the full reference set, checked {checked}");
    assert!(
        worst < 5.0e-2,
        "quadrupole differs from the PySCF reference by {worst:.3e} (criterion 5%)"
    );
}
