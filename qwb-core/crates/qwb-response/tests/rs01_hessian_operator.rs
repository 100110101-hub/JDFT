//! 算子本身对不对：`(A+B)` 必须与**能量沿一次精确轨道旋转的二阶导**对上。
//!
//! 只用一个（占据 i，虚 a）对的**精确旋转** `C_i(λ) = cosλ·C_i + sinλ·C_a`，
//! 因为这样每个 λ 上的轨道都严格正交归一，有限差分没有"正交归一被破坏"的
//! `O(λ²)` 污染：
//!
//! ```text
//! d²E/dλ² |₀ = 2 (A+B)_{(ia),(ia)}
//! ```
//!
//! 这条测试只依赖"能量是什么"，不依赖 CPHF 的推导文本 —— 它同时钉住 `(A+B)` 的
//! 因子（4 vs 2 那类错误会在这里直接露出来）。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_response::cphf::orbital_hessian_apply;
use qwb_response::mo_integrals::{build_cphf_integrals, pair_index};
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

struct Reference {
    basis: qwb_basis::InternalBasis,
    molecule: Molecule,
    tensor: qwb_integrals::TwoElectronIntegrals,
    kinetic: qwb_linalg::Matrix,
    nuclear: qwb_linalg::Matrix,
    coefficients: qwb_linalg::Matrix,
    orbital_energies: Vec<f64>,
    electronic_energy: f64,
}

fn setup() -> Reference {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-12;
    options.density_threshold = 1.0e-10;
    let scf = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("HF");
    let orbital_energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();
    Reference {
        basis,
        molecule,
        tensor,
        kinetic: one.kinetic,
        nuclear: one.nuclear,
        coefficients: scf.coefficients,
        orbital_energies,
        electronic_energy: scf.electronic_hartree,
    }
}

fn energy(reference: &Reference, density: &qwb_linalg::Matrix) -> f64 {
    let n = reference.basis.n_functions();
    let mut e = 0.0;
    for i in 0..n {
        for j in 0..n {
            e += density[(i, j)] * (reference.kinetic[(i, j)] + reference.nuclear[(i, j)]);
        }
    }
    let mut j = 0.0;
    let mut k = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            let p_mu_nu = density[(mu, nu)];
            if p_mu_nu == 0.0 {
                continue;
            }
            for lambda in 0..n {
                for sigma in 0..n {
                    let p_lambda_sigma = density[(lambda, sigma)];
                    if p_lambda_sigma == 0.0 {
                        continue;
                    }
                    j += p_mu_nu * p_lambda_sigma * reference.tensor.get(mu, nu, lambda, sigma);
                    k += p_mu_nu * p_lambda_sigma * reference.tensor.get(mu, lambda, nu, sigma);
                }
            }
        }
    }
    0.5 * j - 0.25 * k + e
}

/// 只旋转 (i, a) 这一对（精确旋转，保持正交归一）。
fn rotated_density(reference: &Reference, i: usize, a: usize, lambda: f64) -> qwb_linalg::Matrix {
    let n = reference.basis.n_functions();
    let n_occ = 5;
    let (c, s) = (lambda.cos(), lambda.sin());
    let mut d = qwb_linalg::Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut value = 0.0;
            for k in 0..n_occ {
                let (phi_mu, phi_nu) = if k == i {
                    (
                        c * reference.coefficients[(mu, i)] + s * reference.coefficients[(mu, a)],
                        c * reference.coefficients[(nu, i)] + s * reference.coefficients[(nu, a)],
                    )
                } else {
                    (
                        reference.coefficients[(mu, k)],
                        reference.coefficients[(nu, k)],
                    )
                };
                value += 2.0 * phi_mu * phi_nu;
            }
            d[(mu, nu)] = value;
        }
    }
    d
}

// 卡点（第 19 轮）：同一个对 (i=0, a=5) 的能量二阶导，在两份测试里给出 37.96（本条）
// 与 80.55（同一轮的一次性诊断）两个不同的数，说明**至少有一份**的差分/密度构造有 bug。
// 在把这个矛盾解开之前不把 RS-01 标 done —— 见 docs/status-v2.yaml 的 RS-01。
#[test]
#[ignore = "卡点：与本轮另一份诊断给出的 FD 值矛盾（37.96 vs 80.55），先修再用"]
fn orbital_hessian_diagonal_matches_the_energy_second_derivative() {
    let reference = setup();
    let n_occ = 5;
    let n_vir = reference.basis.n_functions() - n_occ;
    let dipole_ao = qwb_integrals::dipole::dipole_integrals(&reference.basis, &reference.molecule)
        .expect("dipole");
    let integrals = build_cphf_integrals(
        &reference.tensor,
        &reference.coefficients,
        &dipole_ao,
        n_occ,
    )
    .expect("CPHF integrals");

    // 先核对能量泛函本身：E[P_SCF] 必须等于 SCF 的电子能量。
    let e_check = energy(&reference, &rotated_density(&reference, 0, 0, 0.0));
    println!(
        "E[P_SCF] = {e_check:.10}，SCF 电子能量 = {:.10}，差 {:.3e}",
        reference.electronic_energy,
        e_check - reference.electronic_energy
    );
    assert!(
        (e_check - reference.electronic_energy).abs() < 1.0e-8,
        "测试里的能量泛函与 SCF 电子能量不一致，后面的二阶导不可信"
    );
    let mut worst = 0.0_f64;
    let mut ratio_sum = 0.0;
    let mut count = 0.0;
    for (i, a) in [(0_usize, 0_usize), (1, 3), (4, 1)] {
        let step = 1.0e-3;
        println!(
            "  eps_i = {:.6}, eps_a = {:.6}, 4Δε = {:.6}, 2Δε = {:.6}",
            reference.orbital_energies[i],
            reference.orbital_energies[n_occ + a],
            4.0 * (reference.orbital_energies[n_occ + a] - reference.orbital_energies[i]),
            2.0 * (reference.orbital_energies[n_occ + a] - reference.orbital_energies[i])
        );
        let e0 = energy(&reference, &rotated_density(&reference, i, a, 0.0));
        let plus = energy(&reference, &rotated_density(&reference, i, a, step));
        let minus = energy(&reference, &rotated_density(&reference, i, a, -step));
        let second = (plus - 2.0 * e0 + minus) / (step * step);
        // 算子的对角元：把单位向量 e_p 代进算子。
        let p = pair_index(n_vir, i, a);
        let mut unit = vec![0.0_f64; n_occ * n_vir];
        unit[p] = 1.0;
        let applied = orbital_hessian_apply(&integrals, &reference.orbital_energies, &unit);
        let operator = applied[p];
        println!(
            "pair (i={i}, a={n_occ}+{a})：d²E/dλ² = {second:.10}，算子对角 = {operator:.10}，\
             比值 = {:.6}",
            second / operator
        );
        ratio_sum += second / operator;
        count += 1.0;
        worst = worst.max((second - 2.0 * operator).abs());
    }
    println!(
        "三次测试的平均比值 = {:.6}（下面按 2 断言）",
        ratio_sum / count
    );
    assert!(
        worst < 1.0e-4,
        "d²E/dλ² 与 2·(A+B)_{{pp}} 差 {worst:.3e}；比值说明因子写错了"
    );
}
