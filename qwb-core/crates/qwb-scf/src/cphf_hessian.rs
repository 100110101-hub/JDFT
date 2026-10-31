//! RS-05 的第一层：AO 基里的**核位移导数块**（CPHF 解析 Hessian 的右端与显式项都要它）。
//!
//! # 这里产出什么
//!
//! 对每个核位移方向 `x = (atom, axis)`（共 `3N` 个）给三张 `n×n` 矩阵：
//!
//! ```text
//! S^x_{μν} = ∂S_{μν}/∂x
//! h^x_{μν} = ∂(T+V)_{μν}/∂x       ← 只含「AO 中心在动」的那一半
//! G^x_{μν} = 2 ∂E_2e/∂x 的 AO 形式 = Σ_{λσ}P_{λσ}(μν|λσ)^x − ½Σ_{λσ}P_{λσ}(μλ|νσ)^x
//! ```
//!
//! # 为什么先做这一层
//!
//! CPHF 的右端是**MO 基下的骨架 Fock 导数** `F^x_{ai} − ε_i S^x_{ai}`，显式二阶导项是
//! `Σ_{μν}P_{μν}h^{xy}_{μν} + Γ^{xy} − Σ W S^{xy}` —— 两者都只是这些块的线性组合/导数。
//! 先把块做对、钉死，后面的组装才有对照物。
//!
//! # 约定（与解析梯度逐条对齐）
//!
//! 全部在**直角分量基**上（与 `analytic_gradient` 一致）：球谐密度先用展开表摊到直角分量。
//! `G^x` 的系数 ½ 与能量 `½Σ PP(μν|λσ) − ¼Σ PP(μλ|νσ)` 对齐，所以
//! `Σ_{μν}P_{μν}G^x_{μν} = 2·(解析梯度的 2e 项)` —— 这个 2 倍在测试里是显式判据。
//!
//! 一电子那一半**不能用平动不变性**：`V` 依赖固定的核位置，所以 `∂/∂A + ∂/∂B ≠ 0`。
//! 四中心 2e 积分没有external 中心，才可以用 `∂/∂A+∂/∂B+∂/∂C+∂/∂D = 0`。

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_integrals::one_electron::primitive_norm;
use qwb_linalg::Matrix;

use crate::analytic_gradient::{
    CartAo, PairKind, cartesian_density, cartesian_orbitals, nuclei_of, pair_derivative,
    powers_tuple,
};

/// 一个核位移方向的 AO 基导数块（直角分量基，`n_cart × n_cart`）。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DirectionBlock {
    /// 原子下标。
    pub atom: usize,
    /// 笛卡尔轴（0 = x，1 = y，2 = z）。
    pub axis: usize,
    /// `S^x_{μν}`。
    pub overlap: Matrix,
    /// `h^x_{μν} = T^x_{μν} + V^x_{μν}`（AO 中心移动的那一半）。
    pub one_electron: Matrix,
    /// `G^x_{μν}`（见模块文档的系数约定）。
    pub two_electron: Matrix,
}

/// 全部 `3N` 个方向的导数块，顺序是 `3*atom + axis`。
///
/// # Errors
///
/// 基组/密度维度不匹配、球谐展开不一致、元素缺核电荷。
pub fn cartesian_direction_blocks(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
) -> Result<Vec<DirectionBlock>, EngineError> {
    let (cartesian, expansion) = cartesian_orbitals(basis, molecule)?;
    let n = cartesian.len();
    if density.nrows() != basis.n_functions() || density.ncols() != basis.n_functions() {
        return Err(EngineError::Internal(format!(
            "CPHF Hessian: density is {}x{}, expected {}x{}",
            density.nrows(),
            density.ncols(),
            basis.n_functions(),
            basis.n_functions()
        )));
    }
    let density_cart = cartesian_density(density, &expansion, n);
    let nuclei = nuclei_of(molecule)?;
    let natoms = molecule.atoms.len();

    let mut blocks: Vec<DirectionBlock> = (0..3 * natoms)
        .map(|index| DirectionBlock {
            atom: index / 3,
            axis: index % 3,
            overlap: Matrix::zeros(n, n),
            one_electron: Matrix::zeros(n, n),
            two_electron: Matrix::zeros(n, n),
        })
        .collect();

    one_electron_blocks(&cartesian, &nuclei, &mut blocks);
    two_electron_blocks(&cartesian, &density_cart, &mut blocks);
    Ok(blocks)
}

/// 一电子块（`S^x`、`h^x`）：一趟 AO 对遍历，两侧的中心各算各的导数。
fn one_electron_blocks(
    cartesian: &[CartAo],
    nuclei: &[([f64; 3], f64)],
    blocks: &mut [DirectionBlock],
) {
    let n = cartesian.len();
    for mu in 0..n {
        for nu in mu..n {
            let mut overlap_mu = [0.0_f64; 3];
            let mut overlap_nu = [0.0_f64; 3];
            let mut kinetic_mu = [0.0_f64; 3];
            let mut kinetic_nu = [0.0_f64; 3];
            let mut nuclear_mu = [0.0_f64; 3];
            let mut nuclear_nu = [0.0_f64; 3];
            for axis in 0..3 {
                overlap_mu[axis] = pair_derivative(
                    &cartesian[mu],
                    &cartesian[nu],
                    nuclei,
                    axis,
                    PairKind::Overlap,
                );
                kinetic_mu[axis] = pair_derivative(
                    &cartesian[mu],
                    &cartesian[nu],
                    nuclei,
                    axis,
                    PairKind::Kinetic,
                );
                nuclear_mu[axis] = pair_derivative(
                    &cartesian[mu],
                    &cartesian[nu],
                    nuclei,
                    axis,
                    PairKind::Nuclear,
                );
                // 另一侧（`nu` 的中心）的导数必须**直接算**：`V` 依赖固定的核位置，
                // 所以 `∂/∂A + ∂/∂B ≠ 0`，不能拿 −∂/∂A 代替。
                overlap_nu[axis] = pair_derivative(
                    &cartesian[nu],
                    &cartesian[mu],
                    nuclei,
                    axis,
                    PairKind::Overlap,
                );
                kinetic_nu[axis] = pair_derivative(
                    &cartesian[nu],
                    &cartesian[mu],
                    nuclei,
                    axis,
                    PairKind::Kinetic,
                );
                nuclear_nu[axis] = pair_derivative(
                    &cartesian[nu],
                    &cartesian[mu],
                    nuclei,
                    axis,
                    PairKind::Nuclear,
                );
            }
            let center_mu = cartesian[mu].center_index;
            let center_nu = cartesian[nu].center_index;
            // 对称填充：`μ≠ν` 时两个三角都要写同一份值（矩阵对称，且测试/组装都按有序对求和）；
            // `μ==ν` 时只写一次 —— 否则对角元会被写两遍（第一版就吃了这个亏，对角元一律翻倍）。
            let mirror = mu != nu;
            for axis in 0..3 {
                if center_mu == center_nu {
                    // 两个函数在同一个原子上：该原子的**总**导数 = 两侧之和。
                    let direction = 3 * center_mu + axis;
                    let overlap = overlap_mu[axis] + overlap_nu[axis];
                    let one =
                        kinetic_mu[axis] + kinetic_nu[axis] + nuclear_mu[axis] + nuclear_nu[axis];
                    blocks[direction].overlap[(mu, nu)] += overlap;
                    blocks[direction].one_electron[(mu, nu)] += one;
                    if mirror {
                        blocks[direction].overlap[(nu, mu)] += overlap;
                        blocks[direction].one_electron[(nu, mu)] += one;
                    }
                } else {
                    let direction_mu = 3 * center_mu + axis;
                    let one_mu = kinetic_mu[axis] + nuclear_mu[axis];
                    blocks[direction_mu].overlap[(mu, nu)] += overlap_mu[axis];
                    blocks[direction_mu].one_electron[(mu, nu)] += one_mu;
                    let direction_nu = 3 * center_nu + axis;
                    let one_nu = kinetic_nu[axis] + nuclear_nu[axis];
                    blocks[direction_nu].overlap[(mu, nu)] += overlap_nu[axis];
                    blocks[direction_nu].one_electron[(mu, nu)] += one_nu;
                    if mirror {
                        blocks[direction_mu].overlap[(nu, mu)] += overlap_mu[axis];
                        blocks[direction_mu].one_electron[(nu, mu)] += one_mu;
                        blocks[direction_nu].overlap[(nu, mu)] += overlap_nu[axis];
                        blocks[direction_nu].one_electron[(nu, mu)] += one_nu;
                    }
                }
            }
        }
    }
}

/// 2e 块 `G^x`：与 `analytic_gradient::two_electron_all` 同一套规范 quartet + 8 重置换，
/// 但把权重累加到**矩阵元素**而不是原子向量上。
fn two_electron_blocks(cartesian: &[CartAo], density: &Matrix, blocks: &mut [DirectionBlock]) {
    let n = cartesian.len();
    let natoms = blocks.len() / 3;
    // 每个 AO 的「收缩系数 × 原函数归一化」。
    let weighted: Vec<Vec<f64>> = cartesian
        .iter()
        .map(|orbital| {
            let powers = powers_tuple(orbital.powers);
            orbital
                .exponents
                .iter()
                .zip(orbital.coefficients.iter())
                .map(|(alpha, coefficient)| coefficient * primitive_norm(powers, *alpha))
                .collect()
        })
        .collect();
    // 平铺的累加缓冲：`[3*natoms][n*n]`（矩阵元素累加比原子向量散得多，用平铺数组更快）。
    let mut buffer = vec![0.0_f64; 3 * natoms * n * n];

    const PERMS: [[usize; 4]; 8] = [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 0, 1],
        [2, 3, 1, 0],
        [3, 2, 1, 0],
    ];

    for a in 0..n {
        for b in a..n {
            for c in 0..n {
                for d in c..n {
                    if (a, b) > (c, d) {
                        continue;
                    }
                    let mut any = false;
                    for perm in PERMS.iter() {
                        let indices = [a, b, c, d];
                        let (i0, i1, i2, i3) = (
                            indices[perm[0]],
                            indices[perm[1]],
                            indices[perm[2]],
                            indices[perm[3]],
                        );
                        if density[(i0, i1)] != 0.0 && density[(i2, i3)] != 0.0 {
                            any = true;
                            break;
                        }
                        if density[(i0, i2)] != 0.0 && density[(i1, i3)] != 0.0 {
                            any = true;
                            break;
                        }
                    }
                    if !any {
                        continue;
                    }

                    let mut slot_derivatives = [[0.0_f64; 3]; 4];
                    for (ia, &alpha) in cartesian[a].exponents.iter().enumerate() {
                        for (ib, &beta) in cartesian[b].exponents.iter().enumerate() {
                            for (ic, &gamma) in cartesian[c].exponents.iter().enumerate() {
                                for (id, &delta) in cartesian[d].exponents.iter().enumerate() {
                                    let exponents = [alpha, beta, gamma, delta];
                                    let weight = weighted[a][ia]
                                        * weighted[b][ib]
                                        * weighted[c][ic]
                                        * weighted[d][id];
                                    if weight == 0.0 {
                                        continue;
                                    }
                                    let local =
                                        qwb_integrals::hermite2e::primitive_component_derivatives(
                                            cartesian[a].powers,
                                            cartesian[a].center,
                                            exponents[0],
                                            cartesian[b].powers,
                                            cartesian[b].center,
                                            exponents[1],
                                            cartesian[c].powers,
                                            cartesian[c].center,
                                            exponents[2],
                                            cartesian[d].powers,
                                            cartesian[d].center,
                                            exponents[3],
                                        );
                                    for slot in 0..4 {
                                        for axis in 0..3 {
                                            slot_derivatives[slot][axis] +=
                                                weight * local[slot][axis];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let mut seen: [[usize; 4]; 8] = [[usize::MAX; 4]; 8];
                    let mut count = 0;
                    for perm in PERMS.iter() {
                        let base = [a, b, c, d];
                        let member = [base[perm[0]], base[perm[1]], base[perm[2]], base[perm[3]]];
                        if seen[..count].contains(&member) {
                            continue;
                        }
                        seen[count] = member;
                        count += 1;
                        let p_ket = density[(member[2], member[3])];
                        let p_exchange = density[(member[1], member[3])];
                        if p_ket == 0.0 && p_exchange == 0.0 {
                            continue;
                        }
                        for slot in 0..4 {
                            let atom = cartesian[member[slot]].center_index;
                            let canonical = perm[slot];
                            let row = 3 * atom;
                            #[allow(clippy::needless_range_loop)] // slot_derivatives 按 axis 索引
                            for axis in 0..3 {
                                let value = slot_derivatives[canonical][axis];
                                if value == 0.0 {
                                    continue;
                                }
                                let base_index = (row + axis) * n * n;
                                if p_ket != 0.0 {
                                    // 库仑：G_{m0,m1} += P_{m2,m3}·∂(m0m1|m2m3)
                                    buffer[base_index + member[0] * n + member[1]] += p_ket * value;
                                }
                                if p_exchange != 0.0 {
                                    // 交换：G_{m0,m2} += −½·P_{m1,m3}·∂(m0m2|m1m3)
                                    buffer[base_index + member[0] * n + member[2]] +=
                                        -0.5 * p_exchange * value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for index in 0..blocks.len() {
        for mu in 0..n {
            for nu in 0..n {
                blocks[index].two_electron[(mu, nu)] = buffer[index * n * n + mu * n + nu];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! 导数块的验收：块与**已验的解析梯度分量**必须逐项对上。
    //!
    //! 三条恒等式（见模块文档的系数约定）：
    //!
    //! ```text
    //! Σ_{μν} P_{μν} h^x_{μν}  = kinetic + nuclear_ao
    //! Σ_{μν} P_{μν} G^x_{μν}  = 2 · two_electron
    //! Σ_{μν} W_{μν} S^x_{μν}  = −pulay
    //! ```
    //!
    //! 这三条把「块」与「梯度」钉在一起：梯度那一侧已经被数值梯度验过（G-01 系列），
    //! 所以块写错（符号、中心归属、系数）在这里必然露出来。
    use std::path::PathBuf;

    use qwb_basis::InternalBasis;
    use qwb_engine_api::{Atom, Molecule};
    use qwb_integrals::one_electron::one_electron;
    use qwb_integrals::two_electron::two_electron;
    use qwb_registry::Registry;

    use super::cartesian_direction_blocks;
    use crate::analytic_gradient::{cartesian_density, cartesian_orbitals, gradient_components};
    use crate::pulay::pulay_w;
    use crate::rks::{RksOptions, rks_scf};

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

    fn check(basis_name: &str) {
        let registry = Registry::open(data_root()).expect("registry");
        let molecule = water();
        let basis = InternalBasis::build(registry.get_basis(basis_name).expect("basis"), &molecule)
            .expect("build basis");
        let one = one_electron(&basis, &molecule).expect("1e");
        let tensor = two_electron(&basis, &molecule).expect("2e");
        let grid =
            qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
                .expect("grid");
        let options = RksOptions {
            max_iter: 400,
            energy_threshold: 1.0e-12,
            density_threshold: 1.0e-10,
            ..RksOptions::default()
        };
        let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("scf");
        let fock = crate::build_fock(&one.core_hamiltonian, &tensor, &result.density)
            .expect("fock")
            .fock;
        let components = gradient_components(&basis, &molecule, &result.density, &fock)
            .expect("gradient components");
        let (cartesian, expansion) = cartesian_orbitals(&basis, &molecule).expect("cartesian");
        let n = cartesian.len();
        let p_cart = cartesian_density(&result.density, &expansion, n);
        let w = pulay_w(&result.density, &fock).expect("pulay w");
        let w_cart = cartesian_density(&w, &expansion, n);
        let blocks =
            cartesian_direction_blocks(&basis, &molecule, &result.density).expect("blocks");
        assert_eq!(blocks.len(), 3 * molecule.atoms.len());

        let mut worst_one = 0.0_f64;
        let mut worst_two = 0.0_f64;
        let mut worst_pulay = 0.0_f64;
        for block in &blocks {
            let (atom, axis) = (block.atom, block.axis);
            let mut one_sum = 0.0_f64;
            let mut two_sum = 0.0_f64;
            let mut pulay_sum = 0.0_f64;
            for mu in 0..n {
                for nu in 0..n {
                    let p = p_cart[(mu, nu)];
                    one_sum += p * block.one_electron[(mu, nu)];
                    two_sum += p * block.two_electron[(mu, nu)];
                    pulay_sum += w_cart[(mu, nu)] * block.overlap[(mu, nu)];
                }
            }
            let expected_one = components.kinetic[atom][axis] + components.nuclear_ao[atom][axis];
            let expected_two = 2.0 * components.two_electron[atom][axis];
            let expected_pulay = -components.pulay[atom][axis];
            let scale_one = expected_one.abs().max(one_sum.abs()).max(1.0e-3);
            let scale_two = expected_two.abs().max(two_sum.abs()).max(1.0e-3);
            let scale_pulay = expected_pulay.abs().max(pulay_sum.abs()).max(1.0e-3);
            let diff_one = (one_sum - expected_one).abs() / scale_one;
            let diff_two = (two_sum - expected_two).abs() / scale_two;
            let diff_pulay = (pulay_sum - expected_pulay).abs() / scale_pulay;
            assert!(
                diff_one < 1.0e-9,
                "{basis_name} 1e 方向 ({atom},{axis}): 块 {one_sum:.12e} vs 梯度 {expected_one:.12e}（相对差 {diff_one:.2e}）"
            );
            assert!(
                diff_two < 1.0e-9,
                "{basis_name} 2e 方向 ({atom},{axis}): 块 {two_sum:.12e} vs 2×梯度 {expected_two:.12e}（相对差 {diff_two:.2e}）"
            );
            assert!(
                diff_pulay < 1.0e-9,
                "{basis_name} Pulay 方向 ({atom},{axis}): 块 {pulay_sum:.12e} vs −梯度 {expected_pulay:.12e}（相对差 {diff_pulay:.2e}）"
            );
            worst_one = worst_one.max(diff_one);
            worst_two = worst_two.max(diff_two);
            worst_pulay = worst_pulay.max(diff_pulay);
        }
        println!(
            "{basis_name}: 最大相对差 1e {worst_one:.2e} / 2e {worst_two:.2e} / Pulay {worst_pulay:.2e}"
        );
    }

    #[test]
    fn direction_blocks_match_gradient_components_sto3g() {
        check("sto-3g");
    }

    #[test]
    fn direction_blocks_match_gradient_components_def2svp() {
        check("def2-svp");
    }
}
