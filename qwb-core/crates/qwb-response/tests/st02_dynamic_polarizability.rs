//! ST-02 的验收（第一半）：**动态极化率** `α(ω)`。
//!
//! 判据（`project-phase2.md` §6.7 ST-02）：「与文献（静态/动态）差 < 10%；转动不变量协变 < 1e-6」。
//!
//! 四条检查，前两条是**不需要外部数据**的强判据：
//!
//! 1. **静态极限必须等于 CPHF 的静态极化率**（RS-01）。两条路径完全不同——CPHF 解线性
//!    方程组 `(A+B)X = d`，本模块先解 RPA 本征问题再态求和——在完整单激发空间里数学恒等。
//!    实测差在 1e-9 量级；这条同时验证 `(A+B)`、`(A−B)`、归一化与跃迁偶极矩。
//! 2. **转动不变量协变**：各向同性 `α_iso(ω)` 在旋转分子后不变（判据 < 1e-6）。
//! 3. **频率依赖方向**：`ω` 增大时 `α(ω)` 单调增大（远离共振时），并在触及最低激发能时
//!    显式报错（态求和在那里有极点）。
//! 4. 与方法匹配的参考（PySCF 的 TDHF 态 + 同一态求和公式）比对（见第二个测试）。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::dipole::dipole_integrals;
use qwb_registry::Registry;
use qwb_response::cphf::CphfOptions;
use qwb_response::dynamic_polarizability;
use qwb_response::mo_integrals::build_cphf_integrals;
use qwb_response::static_polarizability;
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

/// 绕 `(1,1,1)/√3` 转 30°（与 RS-04 的抗磁屏蔽测试同一套 Rodrigues 写法）。
fn rotate(molecule: &Molecule, angle: f64) -> Molecule {
    let n = 1.0 / 3.0_f64.sqrt();
    let k = [n, n, n];
    let (s, c) = angle.sin_cos();
    let mut r = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = c * f64::from(u8::from(i == j)) + (1.0 - c) * k[i] * k[j];
        }
    }
    let e = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] += s * e[i][j];
        }
    }
    let atoms: Vec<Atom> = molecule
        .atoms
        .iter()
        .map(|atom| {
            let p = atom.position_bohr();
            let mut q = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    q[i] += r[i][j] * p[j];
                }
                q[i] /= qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
            }
            Atom::new(&atom.symbol, q)
        })
        .collect();
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

fn solve_inputs(
    basis_name: &str,
    molecule: &Molecule,
) -> (qwb_response::mo_integrals::CphfIntegrals, Vec<f64>) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-11;
    options.density_threshold = 1.0e-9;
    let scf = rks_scf(&basis, molecule, &tensor, &grid, &[], options).expect("HF");
    let n_occ = 5;
    let energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();
    let dipole_ao = dipole_integrals(&basis, molecule).expect("dipole");
    let integrals =
        build_cphf_integrals(&tensor, &scf.coefficients, &dipole_ao, n_occ).expect("CPHF");
    (integrals, energies)
}

#[test]
fn the_static_limit_reproduces_the_cphf_polarizability() {
    for basis_name in ["sto-3g", "def2-svp"] {
        let molecule = water();
        let (integrals, energies) = solve_inputs(basis_name, &molecule);
        let (cphf, _) = static_polarizability(&integrals, &energies, CphfOptions::default())
            .expect("static polarizability");
        let dynamic = dynamic_polarizability(&integrals, &energies, &[0.0]).expect("dynamic");
        let sos = dynamic[0].isotropic_au;
        let relative = (sos - cphf.isotropic_au).abs() / cphf.isotropic_au;
        println!(
            "H2O/{basis_name}: alpha_iso(0) SOS = {sos:.10} vs CPHF = {:.10} (rel {relative:.2e}), \
             {} RPA states, lowest excitation {:.6} Ha",
            cphf.isotropic_au, dynamic[0].n_states, dynamic[0].lowest_excitation_hartree
        );
        assert!(
            relative < 1.0e-6,
            "{basis_name}: the sum-over-states static limit must equal the CPHF value, \
             got {relative:.3e}"
        );
        // 张量的每个分量都要对上（不只是迹）。
        for i in 0..3 {
            for j in 0..3 {
                let delta = (dynamic[0].tensor_au[i][j] - cphf.tensor_au[i][j]).abs();
                assert!(
                    delta < 1.0e-6 * cphf.tensor_au[i][j].abs().max(1.0),
                    "{basis_name}: tensor[{i}][{j}] {delta:.3e} vs CPHF"
                );
            }
        }
    }
}

#[test]
fn the_isotropic_response_is_rotation_invariant() {
    let molecule = water();
    let (integrals, energies) = solve_inputs("sto-3g", &molecule);
    let frequencies = [0.0, 0.05, 0.1, 0.2];
    let base = dynamic_polarizability(&integrals, &energies, &frequencies).expect("dynamic");

    let turned = rotate(&molecule, std::f64::consts::PI / 6.0);
    let (integrals2, energies2) = solve_inputs("sto-3g", &turned);
    let moved = dynamic_polarizability(&integrals2, &energies2, &frequencies).expect("dynamic");

    let mut worst = 0.0_f64;
    for (a, b) in base.iter().zip(moved.iter()) {
        let relative = (a.isotropic_au - b.isotropic_au).abs() / a.isotropic_au;
        worst = worst.max(relative);
        println!(
            "  omega = {:.3}: alpha_iso {:.8} vs rotated {:.8} (rel {relative:.2e})",
            a.omega_au, a.isotropic_au, b.isotropic_au
        );
    }
    assert!(
        worst < 1.0e-6,
        "alpha_iso(omega) must be rotation invariant, worst {worst:.3e}"
    );
}

#[test]
fn the_response_grows_with_frequency_and_stops_at_the_pole() {
    let molecule = water();
    let (integrals, energies) = solve_inputs("sto-3g", &molecule);
    let frequencies = [0.0, 0.1, 0.2, 0.3];
    let values = dynamic_polarizability(&integrals, &energies, &frequencies).expect("dynamic");
    for (index, state) in values.iter().enumerate() {
        println!(
            "  omega = {:.3}: alpha_iso = {:.8}, anisotropy = {:.8}",
            state.omega_au, state.isotropic_au, state.anisotropy_au
        );
        if index > 0 {
            assert!(
                state.isotropic_au > values[index - 1].isotropic_au,
                "alpha must grow towards the resonance: {} vs {}",
                state.isotropic_au,
                values[index - 1].isotropic_au
            );
        }
    }
    // 触到极点必须报错（不是返回一个巨大的数）。
    let lowest = values[0].lowest_excitation_hartree;
    let error = dynamic_polarizability(&integrals, &energies, &[lowest])
        .expect_err("omega at the resonance must be rejected");
    println!("  omega = {lowest:.6} Ha 被拒绝：{error}");
    match error {
        qwb_engine_api::EngineError::InvalidInput(message) => {
            assert!(
                message.contains("pole") || message.contains("excitation"),
                "{message}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

/// 方法匹配的参考：PySCF 2.14 的 TDHF 全部态 + 同一条态求和公式
/// （`scripts/pyscf_dynamic_polarizability_reference.py`）。
struct ReferenceRow {
    basis: String,
    omega_au: f64,
    alpha_iso_au: f64,
}

fn reference_rows() -> Vec<ReferenceRow> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("dynamic_polarizability.csv");
    let text = std::fs::read_to_string(&path).expect("dynamic_polarizability.csv");
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
        out.push(ReferenceRow {
            basis: fields[1].to_string(),
            omega_au: fields[2].parse().expect("omega"),
            alpha_iso_au: fields[3].parse().expect("alpha_iso"),
        });
    }
    out
}

#[test]
fn the_dynamic_response_matches_the_tdhf_reference() {
    let reference = reference_rows();
    assert!(!reference.is_empty());
    let mut worst = 0.0_f64;
    for basis_name in ["sto-3g", "def2-svp"] {
        let molecule = water();
        let (integrals, energies) = solve_inputs(basis_name, &molecule);
        let frequencies: Vec<f64> = reference
            .iter()
            .filter(|row| row.basis == basis_name)
            .map(|row| row.omega_au)
            .collect();
        let ours = dynamic_polarizability(&integrals, &energies, &frequencies).expect("dynamic");
        for (state, row) in ours
            .iter()
            .zip(reference.iter().filter(|row| row.basis == basis_name))
        {
            let relative = (state.isotropic_au - row.alpha_iso_au).abs() / row.alpha_iso_au;
            worst = worst.max(relative);
            println!(
                "  {basis_name} omega = {:.2}: ours {:.10} vs PySCF {:.10} (rel {relative:.2e})",
                state.omega_au, state.isotropic_au, row.alpha_iso_au
            );
        }
    }
    // 判据是「与文献差 < 10%」；与**方法匹配**的参考实测在 1e-7 量级。
    assert!(
        worst < 1.0e-6,
        "dynamic polarizability differs from the TDHF reference by {worst:.3e}"
    );
    println!("worst relative difference vs the TDHF reference: {worst:.2e}");
}
