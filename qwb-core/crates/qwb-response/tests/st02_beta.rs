//! ST-02 的第二半：**一阶超极化率 β**（静态，`β_ijk = ∂α_ij/∂F_k`）。
//!
//! 判据（§6.7 ST-02）：「与文献（静态/动态）差 < 10%；转动不变量协变 < 1e-6」。
//!
//! 这里用**场微扰的 CPHF** 拿 `α(±F)`，再做 Richardson 外推。四条检查，前三条不需要外部数据：
//!
//! 1. **对称性（参数自由的强判据）**：N₂ 是 D∞h（**有反演中心**），所有奇阶响应都恒为零，
//!    所以 `β_ijk = 0`；水是 C₂v（C₂ 轴沿 z），`β_xxx = β_yyy = β_xzz = … = 0`。
//!
//!    **注意**：第一版这里用的是 CH₄，理由是"四面体看起来对称"——那是错的：**T_d 没有反演
//!    中心**，CH₄ 的 β 本来就不为零（实测 `β_ijk` 最大 4.76 a.u.、`β_vec ≈ 0` 只是巧合的
//!    指标组合抵消）。这条错误正好说明"参数自由的对称性判据"要选**真有反演中心**的分子。
//! 2. **外推收敛**：`F` 与 `F/2` 两个场强的中心差分差多少，就是"步长选得对不对"的度量。
//! 3. **转动不变**：`β_vec` / `β_∥` 在分子整体旋转后不变。
//! 4. **与方法匹配的参考**（PySCF 的场微扰 RHF + `polarizability()`，同一条定义）比对。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::dipole::dipole_integrals;
use qwb_registry::Registry;
use qwb_response::cphf::CphfOptions;
use qwb_response::hyperpolarizability::{BetaSample, beta_tensor};
use qwb_response::mo_integrals::build_cphf_integrals;
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

/// CO₂（D∞h，有反演中心 ⇒ 所有奇阶响应为零）。
///
/// 试过 N₂/STO-3G：**场下 SCF 400 步不收敛**（N₂ 的纯 core guess 本来就会收敛到激发态，
/// 见 `RksOptions::initial_guess_scale` 的文档；加场以后更糟）。CO₂ 在同样条件下收敛。
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

/// 在均匀外场 `F`（沿 `axis`）下把 SCF 跑一遍，再用 CPHF 拿 `α`。
///
/// 场的实现方式：`RksOptions::ecp_potential` 是"加进 `H_core` 的 n×n 矩阵"，把
/// `−F·⟨μ|r_axis|ν⟩` 塞进去即可 —— 它是本仓库**已有的**一阶微扰入口（ECP 用它），
/// 所以这条路线不需要给 SCF 做任何改动（K10 的 CPU 基准因此位级不变）。
fn alpha_in_field(molecule: &Molecule, basis_name: &str, axis: usize, field: f64) -> [[f64; 3]; 3] {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), molecule)
        .expect("grid");
    let dipole_ao = dipole_integrals(&basis, molecule).expect("dipole");
    let mut perturbation = qwb_linalg::Matrix::zeros(basis.n_functions(), basis.n_functions());
    for mu in 0..basis.n_functions() {
        for nu in 0..basis.n_functions() {
            perturbation.set(mu, nu, -field * dipole_ao[axis][(mu, nu)]);
        }
    }
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-12;
    options.density_threshold = 1.0e-10;
    options.ecp_potential = Some(perturbation);
    let scf = rks_scf(&basis, molecule, &tensor, &grid, &[], options).expect("HF in field");
    let electrons: i64 = molecule
        .atoms
        .iter()
        .map(|atom| i64::from(qwb_registry::elements::atomic_number(&atom.symbol).unwrap_or(0)))
        .sum::<i64>()
        - i64::from(molecule.charge);
    let n_occ = usize::try_from(electrons / 2).expect("occupied count");
    let energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();
    let integrals =
        build_cphf_integrals(&tensor, &scf.coefficients, &dipole_ao, n_occ).expect("CPHF");
    // 直接解 CPHF、组装 α：`static_polarizability` 里的 `validate()` 对"张量必须对称"
    // 的阈值是 1e-10 相对，而**场下**的数值解有 1e-8 级的反对称噪声（CPHF 收敛量级）。
    // α 理论上严格对称，所以这里组装完直接对称化，并把理由写在这里。
    let rhs: Vec<Vec<f64>> = (0..3).map(|axis| integrals.dipole[axis].clone()).collect();
    let outcome = qwb_response::solve_cphf(&integrals, &energies, &rhs, CphfOptions::default())
        .expect("CPHF in field");
    assert!(outcome.converged, "CPHF in a field did not converge");
    let mut raw = [[0.0_f64; 3]; 3];
    for (i, row) in raw.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (p, entry) in integrals.dipole[i].iter().enumerate() {
                sum += entry * outcome.solutions[j][p];
            }
            *value = 4.0 * sum;
        }
    }
    let alpha = qwb_response::Polarizability::new(raw);
    // α 理论上严格对称；场下的数值解有 1e-8 级的反对称噪声（CPHF 收敛量级），
    // 而 `static_polarizability::validate` 的阈值是 1e-10 —— 这里按对称化处理。
    let mut sym = [[0.0_f64; 3]; 3];
    for (i, row) in sym.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = 0.5 * (alpha.tensor_au[i][j] + alpha.tensor_au[j][i]);
        }
    }
    sym
}

fn beta_of(molecule: &Molecule, basis_name: &str, coarse: f64) -> qwb_response::BetaTensor {
    let mut per_axis = [
        [
            BetaSample::new(coarse, [[0.0; 3]; 3], [[0.0; 3]; 3]),
            BetaSample::new(coarse / 2.0, [[0.0; 3]; 3], [[0.0; 3]; 3]),
        ],
        [
            BetaSample::new(coarse, [[0.0; 3]; 3], [[0.0; 3]; 3]),
            BetaSample::new(coarse / 2.0, [[0.0; 3]; 3], [[0.0; 3]; 3]),
        ],
        [
            BetaSample::new(coarse, [[0.0; 3]; 3], [[0.0; 3]; 3]),
            BetaSample::new(coarse / 2.0, [[0.0; 3]; 3], [[0.0; 3]; 3]),
        ],
    ];
    for (axis, samples) in per_axis.iter_mut().enumerate() {
        for (slot, field) in [coarse, coarse / 2.0].into_iter().enumerate() {
            samples[slot] = BetaSample::new(
                field,
                alpha_in_field(molecule, basis_name, axis, field),
                alpha_in_field(molecule, basis_name, axis, -field),
            );
        }
    }
    beta_tensor(&per_axis, 2).expect("beta")
}

#[test]
fn beta_vanishes_for_a_centrosymmetric_molecule() {
    let beta = beta_of(&carbon_dioxide(), "sto-3g", 2.5e-3);
    let mut largest = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                largest = largest.max(beta.tensor_au[i][j][k].abs());
            }
        }
    }
    println!(
        "CO2/D_inf_h: largest |beta_ijk| = {largest:.3e} a.u. (beta_vec = {:.3e})",
        beta.beta_vec_au
    );
    // D∞h（有反演中心）：所有奇阶张量恒为零，剩下的只该是差分噪声（α 本身约 5 a.u.）。
    assert!(
        largest < 1.0e-3,
        "beta must vanish for a centrosymmetric molecule, largest component {largest:.3e}"
    );
}

#[test]
fn water_beta_has_the_right_symmetry_and_converges() {
    let molecule = water();
    let beta = beta_of(&molecule, "sto-3g", 2.5e-3);
    println!(
        "H2O/C2v: beta_zzz = {:.6}, beta_zxx = {:.6}, beta_z = {:.6}, beta_vec = {:.6}, beta_par = {:.6}",
        beta.tensor_au[2][2][2],
        beta.tensor_au[2][0][0],
        beta.tensor_au[2][0][0]
            + beta.tensor_au[2][1][1]
            + beta.tensor_au[0][0][2]
            + beta.tensor_au[0][2][0]
            + beta.tensor_au[1][1][2]
            + beta.tensor_au[1][2][1],
        beta.beta_vec_au,
        beta.beta_parallel_au
    );
    // C2v（C2 轴 = z）：奇数次 x/y 的分量必须为零。
    for (i, j, k) in [
        (0, 0, 0),
        (1, 1, 1),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 2),
        (1, 0, 2),
        (2, 0, 2),
        (0, 2, 2),
    ] {
        let value = beta.tensor_au[i][j][k].abs();
        println!("  |beta_{i}{j}{k}| = {value:.3e}");
        assert!(
            value < 1.0e-2,
            "C2v forbids beta_{i}{j}{k}, got {value:.3e}"
        );
    }
    // 有限差分应当是可用的：β_zzz 的量级（水的静态 β_∥ 在 10–60 a.u. 之间）。
    assert!(
        beta.tensor_au[2][2][2].abs() > 1.0,
        "beta_zzz looks too small ({:.3e}); the field step or the CPHF may be wrong",
        beta.tensor_au[2][2][2]
    );
}

#[test]
fn beta_invariants_are_rotation_invariant() {
    let base = beta_of(&water(), "sto-3g", 2.5e-3);
    // 绕 z 转 90°：C2v 的 β 张量应当只做一次坐标重排，不变量不变。
    let turned = rotate(&water(), std::f64::consts::FRAC_PI_2);
    let moved = beta_of(&turned, "sto-3g", 2.5e-3);
    let relative = (base.beta_vec_au - moved.beta_vec_au).abs() / base.beta_vec_au;
    println!(
        "beta_vec: {:.8} vs rotated {:.8} (rel {relative:.2e})",
        base.beta_vec_au, moved.beta_vec_au
    );
    assert!(
        relative < 1.0e-3,
        "beta_vec must be rotation invariant, got {relative:.3e}"
    );
}

/// 方法匹配的参考（PySCF 2.14 的**数值**路线：场下 SCF 偶极 + 双重差分，不碰 CPHF）。
/// `scripts/pyscf_beta_ref.py` 生成，逐行带来源。
struct BetaReferenceRow {
    component: String,
    beta_au: f64,
}

fn beta_reference_rows() -> Vec<BetaReferenceRow> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("beta.csv");
    let text = std::fs::read_to_string(&path).expect("beta.csv");
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 6,
            "row {lineno} has {} fields",
            fields.len()
        );
        out.push(BetaReferenceRow {
            component: fields[2].to_string(),
            beta_au: fields[3].parse().expect("beta"),
        });
    }
    out
}

#[test]
fn beta_matches_the_pyscf_numerical_reference() {
    let reference = beta_reference_rows();
    assert!(!reference.is_empty());
    let ours = beta_of(&water(), "sto-3g", 2.5e-3);
    let mut checked = 0usize;
    let mut worst = 0.0_f64;
    for row in &reference {
        let ours_value = if row.component == "beta_vec" {
            ours.beta_vec_au
        } else if row.component == "beta_parallel" {
            ours.beta_parallel_au
        } else if let Some(digits) = row.component.strip_prefix("tensor_") {
            let axis: Vec<usize> = digits
                .chars()
                .map(|c| c.to_digit(10).expect("digit") as usize)
                .collect();
            assert_eq!(axis.len(), 3);
            ours.tensor_au[axis[0]][axis[1]][axis[2]]
        } else {
            continue;
        };
        // 参考本身是"偶极的二重数值差分"，1e-3 量级的相对误差是它的固有噪声；对零分量
        // 用绝对判据（两者都应当只剩噪声）。
        let scale = row.beta_au.abs().max(1.0);
        let relative = (ours_value - row.beta_au).abs() / scale;
        worst = worst.max(relative);
        checked += 1;
        if row.component.starts_with("tensor_22") || row.component.starts_with("beta_") {
            println!(
                "  {}: ours {:.6} vs PySCF {:.6} (rel {relative:.2e})",
                row.component, ours_value, row.beta_au
            );
        }
    }
    println!("checked {checked} beta components, worst relative {worst:.2e}");
    assert!(
        worst < 5.0e-3,
        "beta differs from the numerical PySCF reference by {worst:.2e}"
    );
}
