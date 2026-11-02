//! 动量积分 `⟨μ|∂_k|ν⟩`（GIAO 的 `∂H/∂B` 需要）。
//!
//! 直角高斯 `χ_b = N_b (r−B)^b e^{−β(r−B)²}` 的导数**保持同一个 `N_b`**：
//! `∂_k χ_b = N_b[b (r−B)^{b−e_k} − 2β (r−B)^{b+e_k}] e^{…}`，所以矩阵元是
//! `Σ c_a c_b N_a N_b [b·O(b−e_k) − 2β·O(b+e_k)]`，`O` 是**未归一化**的重叠基元。
//!
//! 第 399 轮的错误正是给两个 `O` 又乘了 `N_{b±1}/N_b`（归一化做了两遍），反对称性断言当场失败；
//! 去掉这两个比值后反对称性到 **2.220e-16**（第 402 轮实测）。
#![deny(unsafe_code)]

use qwb_engine_api::EngineError;
use qwb_linalg::Matrix;

use crate::one_electron::{primitive_norm, primitive_overlap};

/// 三个矩阵 `M_k[μ,ν] = ⟨μ|∂_k|ν⟩`（每 Bohr）。
///
/// # Errors
///
/// 球谐变换不可用或维度不一致。
pub fn momentum_integrals(basis: &qwb_basis::InternalBasis) -> Result<[Matrix; 3], EngineError> {
    let n = basis.n_functions();
    let mut out = [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ];
    let shells = basis.shells();
    let mut offsets = Vec::with_capacity(shells.len());
    let mut offset = 0usize;
    for shell in shells {
        offsets.push(offset);
        offset += shell.n_functions();
    }
    for (ia, shell_a) in shells.iter().enumerate() {
        for (ib, shell_b) in shells.iter().enumerate() {
            let block = cartesian_block(shell_a, shell_b)?;
            let transform_a = qwb_basis::spherical::spherical_transform(shell_a.l, shell_a.angular)
                .ok_or_else(|| {
                    EngineError::Unsupported(format!("spherical transform l={}", shell_a.l))
                })?;
            let transform_b = qwb_basis::spherical::spherical_transform(shell_b.l, shell_b.angular)
                .ok_or_else(|| {
                    EngineError::Unsupported(format!("spherical transform l={}", shell_b.l))
                })?;
            for (axis, plane) in block.iter().enumerate() {
                for i in 0..transform_a.n_spherical {
                    for j in 0..transform_b.n_spherical {
                        let mut sum = 0.0_f64;
                        for p in 0..transform_a.n_cartesian {
                            let ta = transform_a.get(p, i);
                            if ta == 0.0 {
                                continue;
                            }
                            for q in 0..transform_b.n_cartesian {
                                sum += ta
                                    * plane[p * transform_b.n_cartesian + q]
                                    * transform_b.get(q, j);
                            }
                        }
                        out[axis][(offsets[ia] + i, offsets[ib] + j)] = sum;
                    }
                }
            }
        }
    }
    Ok(out)
}

fn cartesian_block(
    shell_a: &qwb_basis::InternalShell,
    shell_b: &qwb_basis::InternalShell,
) -> Result<[Vec<f64>; 3], EngineError> {
    let components_a = qwb_basis::cartesian_components(shell_a.l);
    let components_b = qwb_basis::cartesian_components(shell_b.l);
    let rows = components_a.len();
    let columns = components_b.len();
    let mut out = [
        vec![0.0_f64; rows * columns],
        vec![0.0_f64; rows * columns],
        vec![0.0_f64; rows * columns],
    ];
    for (ca, powers_a) in components_a.iter().enumerate() {
        for (cb, powers_b) in components_b.iter().enumerate() {
            for (alpha, coefficient_a) in shell_a.exponents.iter().zip(&shell_a.coefficients) {
                let norm_a = primitive_norm(*powers_a, *alpha);
                for (beta, coefficient_b) in shell_b.exponents.iter().zip(&shell_b.coefficients) {
                    let norm_b = primitive_norm(*powers_b, *beta);
                    let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                    #[allow(clippy::needless_range_loop)]
                    for axis in 0..3 {
                        let power = power_along(*powers_b, axis);
                        let mut value = 0.0_f64;
                        if power > 0 {
                            value += f64::from(power)
                                * primitive_overlap(
                                    *alpha,
                                    *powers_a,
                                    shell_a.center_bohr,
                                    *beta,
                                    with_power(*powers_b, axis, power - 1),
                                    shell_b.center_bohr,
                                );
                        }
                        value -= 2.0
                            * beta
                            * primitive_overlap(
                                *alpha,
                                *powers_a,
                                shell_a.center_bohr,
                                *beta,
                                with_power(*powers_b, axis, power + 1),
                                shell_b.center_bohr,
                            );
                        out[axis][ca * columns + cb] += weight * value;
                    }
                }
            }
        }
    }
    Ok(out)
}

fn power_along(powers: (u32, u32, u32), axis: usize) -> u32 {
    match axis {
        0 => powers.0,
        1 => powers.1,
        _ => powers.2,
    }
}

fn with_power(powers: (u32, u32, u32), axis: usize, value: u32) -> (u32, u32, u32) {
    match axis {
        0 => (value, powers.1, powers.2),
        1 => (powers.0, value, powers.2),
        _ => (powers.0, powers.1, value),
    }
}

#[cfg(test)]
mod tests {
    use super::momentum_integrals;
    use qwb_engine_api::{Atom, Molecule};
    use qwb_registry::Registry;
    use std::path::PathBuf;

    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

    fn data_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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

    fn basis() -> qwb_basis::InternalBasis {
        let registry = Registry::open(data_root()).expect("qwb-data");
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &water())
            .expect("basis")
    }

    #[test]
    fn the_matrix_is_antisymmetric() {
        let momentum = momentum_integrals(&basis()).expect("momentum");
        let n = momentum[0].nrows();
        let mut worst = 0.0_f64;
        #[allow(clippy::needless_range_loop)]
        for axis in 0..3 {
            for mu in 0..n {
                for nu in 0..n {
                    worst =
                        worst.max((momentum[axis].get(mu, nu) + momentum[axis].get(nu, mu)).abs());
                }
            }
        }
        println!("动量矩阵反对称性最差 {worst:.3e}");
        assert!(worst < 1.0e-10, "反对称性最差 {worst:.3e}");
    }

    #[test]
    fn the_derivative_matches_the_overlap_centre_difference() {
        let basis = basis();
        let momentum = momentum_integrals(&basis).expect("momentum");
        let n = basis.n_functions();
        let step = 1.0e-5_f64;
        let registry = Registry::open(data_root()).expect("qwb-data");
        let mut worst = 0.0_f64;
        #[allow(clippy::needless_range_loop)]
        for axis in 0..3 {
            // 关键：**原件全在前、副本全在后** —— 这样 0..n 是原件、n..2n 是副本，
            // 才有干净的交叉块可读（第 402 轮交错 push 导致没有这个块）。
            let shifted = |delta: f64| -> Vec<f64> {
                let mut atoms: Vec<Atom> = Vec::new();
                for atom in &water().atoms {
                    atoms.push(Atom::new(&atom.symbol, atom.position_angstrom));
                }
                for atom in &water().atoms {
                    let mut position = atom.position_angstrom;
                    position[axis] += delta;
                    atoms.push(Atom::new(&atom.symbol, position));
                }
                let doubled = qwb_basis::InternalBasis::build(
                    registry.get_basis("sto-3g").expect("sto-3g"),
                    &Molecule::new(atoms, 0, 1),
                )
                .expect("doubled");
                overlap_matrix(&doubled)
            };
            let plus = shifted(step);
            let minus = shifted(-step);
            let total = 2 * n;
            for mu in 0..n {
                for nu in 0..n {
                    // 位移在 Å、动量积分每 Bohr：dS/dδ_Å = −⟨∂⟩/0.529177。
                    let derivative = (plus[mu * total + (n + nu)] - minus[mu * total + (n + nu)])
                        / (2.0 * step)
                        * ANGSTROM_PER_BOHR;
                    worst = worst.max((derivative + momentum[axis].get(mu, nu)).abs());
                }
            }
        }
        println!("动量积分 vs 重叠中心差分最差 {worst:.3e}");
        assert!(worst < 1.0e-7, "与中心差分差 {worst:.3e}");
    }

    /// 只用 `primitive_overlap` + 球谐变换组装重叠矩阵（不经过 `momentum_integrals`）。
    fn overlap_matrix(basis: &qwb_basis::InternalBasis) -> Vec<f64> {
        let n = basis.n_functions();
        let mut out = vec![0.0_f64; n * n];
        let mut offsets = Vec::new();
        let mut offset = 0usize;
        for shell in basis.shells() {
            offsets.push(offset);
            offset += shell.n_functions();
        }
        for (ia, shell_a) in basis.shells().iter().enumerate() {
            for (ib, shell_b) in basis.shells().iter().enumerate() {
                let components_a = qwb_basis::cartesian_components(shell_a.l);
                let components_b = qwb_basis::cartesian_components(shell_b.l);
                let rows = components_a.len();
                let columns = components_b.len();
                let mut block = vec![0.0_f64; rows * columns];
                for (ca, powers_a) in components_a.iter().enumerate() {
                    for (cb, powers_b) in components_b.iter().enumerate() {
                        for (alpha, coefficient_a) in
                            shell_a.exponents.iter().zip(&shell_a.coefficients)
                        {
                            let norm_a = crate::one_electron::primitive_norm(*powers_a, *alpha);
                            for (beta, coefficient_b) in
                                shell_b.exponents.iter().zip(&shell_b.coefficients)
                            {
                                let norm_b = crate::one_electron::primitive_norm(*powers_b, *beta);
                                block[ca * columns + cb] += coefficient_a
                                    * coefficient_b
                                    * norm_a
                                    * norm_b
                                    * crate::one_electron::primitive_overlap(
                                        *alpha,
                                        *powers_a,
                                        shell_a.center_bohr,
                                        *beta,
                                        *powers_b,
                                        shell_b.center_bohr,
                                    );
                            }
                        }
                    }
                }
                let transform_a =
                    qwb_basis::spherical::spherical_transform(shell_a.l, shell_a.angular)
                        .expect("ta");
                let transform_b =
                    qwb_basis::spherical::spherical_transform(shell_b.l, shell_b.angular)
                        .expect("tb");
                for i in 0..transform_a.n_spherical {
                    for j in 0..transform_b.n_spherical {
                        let mut sum = 0.0_f64;
                        for p in 0..rows {
                            for q in 0..columns {
                                sum += transform_a.get(p, i)
                                    * block[p * columns + q]
                                    * transform_b.get(q, j);
                            }
                        }
                        out[(offsets[ia] + i) * n + (offsets[ib] + j)] = sum;
                    }
                }
            }
        }
        out
    }
}
