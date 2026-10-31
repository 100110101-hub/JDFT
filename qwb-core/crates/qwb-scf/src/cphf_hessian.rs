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
    powers_tuple, raise_lower,
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
    /// 核位置移动给 `V` 的那一半（`V^x_nucleus`）：`d/dR_A[−Z_A/|r−R_A|]`。
    ///
    /// **两个一电子块必须相加**才是「位移一个原子」的一电子骨架导数 —— 原子一动，
    /// 它的基函数和它的核一起动。平动不变性把它们连起来：
    /// `V^x_nucleus = −(∂V^{(A)}/∂A + ∂V^{(A)}/∂B)`（单核项，按 AO 中心的降幂求导）。
    pub nuclear_center: Matrix,
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
            nuclear_center: Matrix::zeros(n, n),
            two_electron: Matrix::zeros(n, n),
        })
        .collect();

    one_electron_blocks(&cartesian, &nuclei, &mut blocks);
    nuclear_center_blocks(&cartesian, &nuclei, &mut blocks);
    two_electron_blocks(&cartesian, &density_cart, &mut blocks);
    Ok(blocks)
}

/// 核位置那一半：`V^x_nucleus = −(d_a + d_b)`，其中 `d_a`、`d_b` 是**单核项**对两侧
/// AO 中心的降幂导数（与 `analytic_gradient::nuclear_center_axis` 同一个式子，
/// 只是这里保留矩阵结构）。
fn nuclear_center_blocks(
    cartesian: &[CartAo],
    nuclei: &[([f64; 3], f64)],
    blocks: &mut [DirectionBlock],
) {
    use qwb_integrals::one_electron::primitive_nuclear;

    let n = cartesian.len();
    for (atom, (position, charge)) in nuclei.iter().enumerate() {
        for mu in 0..n {
            for nu in mu..n {
                let mut values = [[0.0_f64; 3]; 2];
                for (alpha, ca) in cartesian[mu]
                    .exponents
                    .iter()
                    .zip(&cartesian[mu].coefficients)
                {
                    let na = primitive_norm(powers_tuple(cartesian[mu].powers), *alpha);
                    for (beta, cb) in cartesian[nu]
                        .exponents
                        .iter()
                        .zip(&cartesian[nu].coefficients)
                    {
                        let nb = primitive_norm(powers_tuple(cartesian[nu].powers), *beta);
                        let weight = ca * cb * na * nb;
                        if weight == 0.0 {
                            continue;
                        }
                        #[allow(clippy::needless_range_loop)]
                        // axis 同时给 raise_lower 和 values 用
                        for axis in 0..3 {
                            let d_a = raise_lower(*alpha, cartesian[mu].powers, axis, |powers| {
                                primitive_nuclear(
                                    *alpha,
                                    (powers[0], powers[1], powers[2]),
                                    cartesian[mu].center,
                                    *beta,
                                    powers_tuple(cartesian[nu].powers),
                                    cartesian[nu].center,
                                    *position,
                                    *charge,
                                )
                            });
                            let d_b = raise_lower(*beta, cartesian[nu].powers, axis, |powers| {
                                primitive_nuclear(
                                    *alpha,
                                    powers_tuple(cartesian[mu].powers),
                                    cartesian[mu].center,
                                    *beta,
                                    (powers[0], powers[1], powers[2]),
                                    cartesian[nu].center,
                                    *position,
                                    *charge,
                                )
                            });
                            values[0][axis] += weight * d_a;
                            values[1][axis] += weight * d_b;
                        }
                    }
                }
                for axis in 0..3 {
                    let value = -(values[0][axis] + values[1][axis]);
                    blocks[3 * atom + axis].nuclear_center[(mu, nu)] += value;
                    if mu != nu {
                        blocks[3 * atom + axis].nuclear_center[(nu, mu)] += value;
                    }
                }
            }
        }
    }
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

/// 一个原函数四元组的 4 个中心两两二阶导数（4 槽 × 4 槽 × 3 轴 × 3 轴）。
type SlotSecond = [[[[f64; 3]; 3]; 4]; 4];

/// 2e 项的**显式**二阶导数：`E^{xy}_2e = ½Σ_{μνλσ}P P(μν|λσ)^{xy} − ¼Σ P P(μλ|νσ)^{xy}`。
///
/// 返回 `3N × 3N`（下标 `3·atom + axis`，与 `DirectionBlock` 同序）。
///
/// # 结构：把梯度公式里的一阶导数换成二阶导数
///
/// 规范 quartet + 8 重置换 + 库仑 ½ / 交换 −¼ 的权重，与
/// `analytic_gradient::two_electron_all` **逐字一致**；区别只有两点：换用
/// [`qwb_integrals::hermite2e::primitive_component_second_derivatives`]，以及把值散到
/// `(方向, 方向)` 而不是原子向量上。于是它有一条独立的判据：
///
/// ```text
/// E^{xy}_2e = ∂/∂y [ ½Σ_{μν}P_{μν}G^x_{μν} ]        （密度冻结）
/// ```
///
/// 右边是「2e 梯度对几何的中心差分」，用的是已经钉住的 `G^x` 块。
///
/// # Errors
///
/// 维度不匹配（密度的行列数必须等于直角分量数）。
pub fn two_electron_hessian_contracted(
    cartesian: &[CartAo],
    density: &Matrix,
) -> Result<Matrix, EngineError> {
    let n = cartesian.len();
    if density.nrows() != n || density.ncols() != n {
        return Err(EngineError::Internal(format!(
            "2e Hessian: density is {}x{}, expected {n}x{n}",
            density.nrows(),
            density.ncols()
        )));
    }
    let natoms = cartesian
        .iter()
        .map(|orbital| orbital.center_index)
        .max()
        .map_or(0, |index| index + 1);
    let dof = 3 * natoms;
    let mut out = Matrix::zeros(dof, dof);
    // 诊断计时（默认静默）：RS-05 的 k 值必须知道时间花在基元上还是花在累加上。
    let trace = std::env::var_os("QWB_HESS_TRACE").is_some();
    let mut primitive_calls = 0usize;
    let mut primitive_seconds = 0.0_f64;
    let overall = std::time::Instant::now();
    let atom_of: Vec<usize> = cartesian
        .iter()
        .map(|orbital| orbital.center_index)
        .collect();
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

                    // 先把**收缩后**的二阶导数累出来（每个四元组只算一次 144 个分量）。
                    let mut slot: SlotSecond = [[[[0.0_f64; 3]; 3]; 4]; 4];
                    for (ia, &alpha) in cartesian[a].exponents.iter().enumerate() {
                        for (ib, &beta) in cartesian[b].exponents.iter().enumerate() {
                            for (ic, &gamma) in cartesian[c].exponents.iter().enumerate() {
                                for (id, &delta) in cartesian[d].exponents.iter().enumerate() {
                                    let weight = weighted[a][ia]
                                        * weighted[b][ib]
                                        * weighted[c][ic]
                                        * weighted[d][id];
                                    if weight == 0.0 {
                                        continue;
                                    }
                                    let clock = std::time::Instant::now();
                                    let local =
                                        qwb_integrals::hermite2e::primitive_component_second_derivatives(
                                            cartesian[a].powers,
                                            cartesian[a].center,
                                            alpha,
                                            cartesian[b].powers,
                                            cartesian[b].center,
                                            beta,
                                            cartesian[c].powers,
                                            cartesian[c].center,
                                            gamma,
                                            cartesian[d].powers,
                                            cartesian[d].center,
                                            delta,
                                        );
                                    primitive_seconds += clock.elapsed().as_secs_f64();
                                    primitive_calls += 1;
                                    for s in 0..4 {
                                        for t in 0..4 {
                                            for i in 0..3 {
                                                for j in 0..3 {
                                                    slot[s][t][i][j] += weight * local[s][t][i][j];
                                                }
                                            }
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
                        let coulomb =
                            0.5 * density[(member[0], member[1])] * density[(member[2], member[3])];
                        let exchange = -0.25
                            * density[(member[0], member[2])]
                            * density[(member[1], member[3])];
                        let weight = coulomb + exchange;
                        if weight == 0.0 {
                            continue;
                        }
                        for s in 0..4 {
                            let first = 3 * atom_of[member[s]];
                            // 成员第 `s` 个位置上的函数是规范 quartet 的第 `perm[s]` 个槽位 ——
                            // 二阶导数的下标必须跟着置换走（第一版直接用 `slot[s][t]`，
                            // 库仑部分因此差了 4 倍、交换部分全错；见 choices 一百一十四）。
                            let canonical_s = perm[s];
                            for t in 0..4 {
                                let second = 3 * atom_of[member[t]];
                                let canonical_t = perm[t];
                                for i in 0..3 {
                                    for j in 0..3 {
                                        out[(first + i, second + j)] +=
                                            weight * slot[canonical_s][canonical_t][i][j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if trace {
        let total = overall.elapsed().as_secs_f64();
        eprintln!(
            "hess 2e trace: {n} cart, {primitive_calls} 次基元调用 {primitive_seconds:.3} s ({:.1}%), 累加 {:.3} s, 合计 {total:.3} s",
            primitive_seconds / total * 100.0,
            total - primitive_seconds
        );
    }
    Ok(out)
}

///
/// CPHF 的右端骨架：`F^x_{pq}`、`S^x_{pq}` 与 `B^x_{ai} = F^x_{ai} − ε_i S^x_{ai}`。
///
/// `right_hand_sides[d]` 里 `p = i·n_vir + a`，与 `qwb_response::cphf` 的展平约定一致。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CphfRhs {
    /// 占据轨道数。
    pub n_occupied: usize,
    /// 虚轨道数。
    pub n_virtual: usize,
    /// 方向数（`3N`）。
    pub n_directions: usize,
    /// `F^x_{pq} = C^T(h^x + V^x_nucleus + G^x)C`（MO 基，`n×n`）。
    pub fock_mo: Vec<Matrix>,
    /// `S^x_{pq}`（MO 基，`n×n`）。
    pub overlap_mo: Vec<Matrix>,
    /// `B^x_{ai}`，`p = i·n_vir + a`。
    pub right_hand_sides: Vec<Vec<f64>>,
    /// 直角分量基下的 MO 系数 `C_cart = T·C`（显式二阶导项要复用它）。
    pub coefficients_cartesian: Matrix,
}

/// 由导数块与 MO 数据组装 CPHF 右端（`3N` 个方向）。
///
/// 一电子骨架导数是**两块之和**：`one_electron`（AO 中心动）+ `nuclear_center`（核动）。
/// 2e 那块 `G^x` 已按密度收缩好。
///
/// # Errors
///
/// 维度不匹配（`coefficients` 的列数必须等于 `basis.n_functions()`、
/// `mo_energies` 长度不够、`n_occupied` 越界）。
pub fn cphf_right_hand_sides(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
    coefficients: &Matrix,
    mo_energies: &[f64],
    n_occupied: usize,
) -> Result<CphfRhs, EngineError> {
    let n_mo = basis.n_functions();
    if coefficients.nrows() != n_mo || coefficients.ncols() != n_mo {
        return Err(EngineError::Internal(format!(
            "CPHF: MO coefficient matrix is {}x{}, expected {n_mo}x{n_mo}",
            coefficients.nrows(),
            coefficients.ncols()
        )));
    }
    if mo_energies.len() < n_mo {
        return Err(EngineError::Internal(format!(
            "CPHF: only {} orbital energies for {n_mo} functions",
            mo_energies.len()
        )));
    }
    if n_occupied == 0 || n_occupied >= n_mo {
        return Err(EngineError::InvalidInput(format!(
            "CPHF needs at least one occupied and one virtual orbital (n_occ = {n_occupied}, n = {n_mo})"
        )));
    }
    let (cartesian, expansion) = cartesian_orbitals(basis, molecule)?;
    let n_cart = cartesian.len();
    let coefficients_cartesian = cartesian_mo_coefficients(coefficients, &expansion, n_cart);
    let blocks = cartesian_direction_blocks(basis, molecule, density)?;
    let n_virtual = n_mo - n_occupied;

    let mut fock_mo = Vec::with_capacity(blocks.len());
    let mut overlap_mo = Vec::with_capacity(blocks.len());
    let mut right_hand_sides = Vec::with_capacity(blocks.len());
    for block in &blocks {
        let mut fock = Matrix::zeros(n_mo, n_mo);
        let mut overlap = Matrix::zeros(n_mo, n_mo);
        for p in 0..n_mo {
            for q in 0..n_mo {
                let mut fock_sum = 0.0_f64;
                let mut overlap_sum = 0.0_f64;
                for mu in 0..n_cart {
                    let c_mu_p = coefficients_cartesian[(mu, p)];
                    if c_mu_p == 0.0 {
                        continue;
                    }
                    for nu in 0..n_cart {
                        let c_nu_q = coefficients_cartesian[(nu, q)];
                        if c_nu_q == 0.0 {
                            continue;
                        }
                        let weight = c_mu_p * c_nu_q;
                        fock_sum += weight
                            * (block.one_electron[(mu, nu)]
                                + block.nuclear_center[(mu, nu)]
                                + block.two_electron[(mu, nu)]);
                        overlap_sum += weight * block.overlap[(mu, nu)];
                    }
                }
                fock[(p, q)] = fock_sum;
                overlap[(p, q)] = overlap_sum;
            }
        }
        let mut vector = vec![0.0_f64; n_occupied * n_virtual];
        for i in 0..n_occupied {
            for a in 0..n_virtual {
                let virtual_index = n_occupied + a;
                vector[i * n_virtual + a] =
                    fock[(virtual_index, i)] - mo_energies[i] * overlap[(virtual_index, i)];
            }
        }
        fock_mo.push(fock);
        overlap_mo.push(overlap);
        right_hand_sides.push(vector);
    }

    Ok(CphfRhs {
        n_occupied,
        n_virtual,
        n_directions: blocks.len(),
        fock_mo,
        overlap_mo,
        right_hand_sides,
        coefficients_cartesian,
    })
}

/// `C_cart = T·C`：把球谐基的 MO 系数摊到直角分量基（`T[μ,p]` 是展开表里的权重）。
fn cartesian_mo_coefficients(
    coefficients: &Matrix,
    expansion: &[Vec<(usize, f64)>],
    n_cart: usize,
) -> Matrix {
    let n_mo = coefficients.ncols();
    let mut out = Matrix::zeros(n_cart, n_mo);
    for (p, entries) in expansion.iter().enumerate() {
        for (mu, weight) in entries {
            for q in 0..n_mo {
                let value = coefficients[(p, q)];
                if value == 0.0 {
                    continue;
                }
                out[(*mu, q)] += weight * value;
            }
        }
    }
    out
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
        let mut worst_center = 0.0_f64;
        for block in &blocks {
            let (atom, axis) = (block.atom, block.axis);
            let mut one_sum = 0.0_f64;
            let mut two_sum = 0.0_f64;
            let mut pulay_sum = 0.0_f64;
            let mut center_sum = 0.0_f64;
            for mu in 0..n {
                for nu in 0..n {
                    let p = p_cart[(mu, nu)];
                    one_sum += p * block.one_electron[(mu, nu)];
                    center_sum += p * block.nuclear_center[(mu, nu)];
                    two_sum += p * block.two_electron[(mu, nu)];
                    pulay_sum += w_cart[(mu, nu)] * block.overlap[(mu, nu)];
                }
            }
            let expected_one = components.kinetic[atom][axis] + components.nuclear_ao[atom][axis];
            let expected_two = 2.0 * components.two_electron[atom][axis];
            let expected_pulay = -components.pulay[atom][axis];
            let expected_center = components.nuclear_center[atom][axis];
            let scale_one = expected_one.abs().max(one_sum.abs()).max(1.0e-3);
            let scale_two = expected_two.abs().max(two_sum.abs()).max(1.0e-3);
            let scale_pulay = expected_pulay.abs().max(pulay_sum.abs()).max(1.0e-3);
            let scale_center = expected_center.abs().max(center_sum.abs()).max(1.0e-3);
            let diff_one = (one_sum - expected_one).abs() / scale_one;
            let diff_two = (two_sum - expected_two).abs() / scale_two;
            let diff_pulay = (pulay_sum - expected_pulay).abs() / scale_pulay;
            let diff_center = (center_sum - expected_center).abs() / scale_center;
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
            assert!(
                diff_center < 1.0e-9,
                "{basis_name} 核位置方向 ({atom},{axis}): 块 {center_sum:.12e} vs 梯度 {expected_center:.12e}（相对差 {diff_center:.2e}）"
            );
            worst_one = worst_one.max(diff_one);
            worst_two = worst_two.max(diff_two);
            worst_pulay = worst_pulay.max(diff_pulay);
            worst_center = worst_center.max(diff_center);
        }
        println!(
            "{basis_name}: 最大相对差 1e {worst_one:.2e} / 2e {worst_two:.2e} / Pulay {worst_pulay:.2e} / 核位置 {worst_center:.2e}"
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

    /// 骨架 Fock 导数 `F^x_{pq}` 与「位移几何上用**固定**密度和**固定** C 重建 Fock」的
    /// 中心差分对照（独立于整套导数块：只用到现有的 `one_electron` / `two_electron` /
    /// `build_fock`）。
    #[test]
    fn skeleton_fock_derivative_matches_finite_difference() {
        const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
        let delta = 1.0e-3_f64;
        let registry = Registry::open(data_root()).expect("registry");
        let molecule = water();
        let basis = InternalBasis::build(registry.get_basis("sto-3g").expect("basis"), &molecule)
            .expect("build basis");
        let grid =
            qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
                .expect("grid");
        let tensor = two_electron(&basis, &molecule).expect("2e");
        let options = RksOptions {
            max_iter: 400,
            energy_threshold: 1.0e-12,
            density_threshold: 1.0e-10,
            ..RksOptions::default()
        };
        let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("scf");
        let n_occupied = (molecule
            .atoms
            .iter()
            .map(|atom| qwb_registry::elements::atomic_number(&atom.symbol).unwrap_or(0))
            .sum::<u32>() as usize
            - molecule.charge as usize)
            / 2;
        let energies: Vec<f64> = (0..basis.n_functions())
            .map(|index| result.mo_energies_hartree.get(index))
            .collect();
        let rhs = crate::cphf_hessian::cphf_right_hand_sides(
            &basis,
            &molecule,
            &result.density,
            &result.coefficients,
            &energies,
            n_occupied,
        )
        .expect("rhs");
        assert_eq!(rhs.n_directions, 3 * molecule.atoms.len());
        assert_eq!(rhs.n_virtual, basis.n_functions() - n_occupied);

        let n = basis.n_functions();
        let transform = |fock: &qwb_linalg::Matrix| {
            let mut out = vec![0.0_f64; n * n];
            for p in 0..n {
                for q in 0..n {
                    let mut total = 0.0;
                    for mu in 0..n {
                        let c_mu_p = result.coefficients[(mu, p)];
                        if c_mu_p == 0.0 {
                            continue;
                        }
                        for nu in 0..n {
                            let c_nu_q = result.coefficients[(nu, q)];
                            if c_nu_q == 0.0 {
                                continue;
                            }
                            total += c_mu_p * c_nu_q * fock[(mu, nu)];
                        }
                    }
                    out[p * n + q] = total;
                }
            }
            out
        };
        let mut worst = 0.0_f64;
        for atom in 0..molecule.atoms.len() {
            for axis in 0..3 {
                let displaced = |amount: f64| {
                    let mut moved = molecule.clone();
                    moved.atoms[atom].position_angstrom[axis] += amount * ANGSTROM_PER_BOHR;
                    moved
                };
                let fock_at = |moved: &Molecule| {
                    let moved_basis =
                        InternalBasis::build(registry.get_basis("sto-3g").expect("basis"), moved)
                            .expect("basis");
                    let moved_one = one_electron(&moved_basis, moved).expect("1e");
                    let moved_tensor = two_electron(&moved_basis, moved).expect("2e");
                    let fock = crate::build_fock(
                        &moved_one.core_hamiltonian,
                        &moved_tensor,
                        &result.density,
                    )
                    .expect("fock");
                    transform(&fock.fock)
                };
                // Richardson：`D(δ)` 与 `D(δ/2)` 外推掉 `O(δ²)` 项。小矩阵元（1e-4 量级）
                // 的截断误差会把相对判据打穿，不消掉就只能靠松判据。
                let step = |amount: f64| {
                    let plus_mo = fock_at(&displaced(amount));
                    let minus_mo = fock_at(&displaced(-amount));
                    (0..n * n)
                        .map(|index| (plus_mo[index] - minus_mo[index]) / (2.0 * amount))
                        .collect::<Vec<f64>>()
                };
                let coarse = step(delta);
                let fine = step(0.5 * delta);
                let direction = 3 * atom + axis;
                for p in 0..n {
                    for q in 0..n {
                        let index = p * n + q;
                        let numeric = (4.0 * fine[index] - coarse[index]) / 3.0;
                        let analytic = rhs.fock_mo[direction][(p, q)];
                        let scale = numeric.abs().max(analytic.abs()).max(1.0e-3);
                        let difference = (numeric - analytic).abs() / scale;
                        assert!(
                            difference < 1.0e-7,
                            "F^{direction} MO ({p},{q}): 解析 {analytic:.10e} vs 差分 {numeric:.10e}（相对差 {difference:.2e}）"
                        );
                        worst = worst.max(difference);
                    }
                }
            }
        }
        println!("骨架 Fock 导数的最大相对差：{worst:.2e}");
    }

    /// 2e 显式二阶导项的独立判据：`E^{xy}_2e = ∂/∂y[½Σ_{μν}P_{μν}G^x_{μν}]`（密度冻结）。
    ///
    /// 右边用已经钉住的 `G^x` 块：在 `R±δ` 上重建几何、用**参考密度**重算 `G^x`、
    /// 与密度缩成标量，再中心差分。左边是新的二阶导数收缩。两者只在「二阶导数基元」
    /// 这一处不同，所以这条测试抓的是 AO 层组装（置换映射、中心归属、权重）。
    #[test]
    fn two_electron_hessian_matches_frozen_density_difference() {
        const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
        let delta = 1.0e-3_f64;
        let registry = Registry::open(data_root()).expect("registry");
        let molecule = water();
        let basis = InternalBasis::build(registry.get_basis("sto-3g").expect("basis"), &molecule)
            .expect("build basis");
        let grid =
            qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
                .expect("grid");
        let tensor = two_electron(&basis, &molecule).expect("2e");
        let options = RksOptions {
            max_iter: 400,
            energy_threshold: 1.0e-12,
            density_threshold: 1.0e-10,
            ..RksOptions::default()
        };
        let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("scf");
        let (cartesian, expansion) = cartesian_orbitals(&basis, &molecule).expect("cartesian");
        let n = cartesian.len();
        let density_cart = cartesian_density(&result.density, &expansion, n);
        let analytic =
            crate::cphf_hessian::two_electron_hessian_contracted(&cartesian, &density_cart)
                .expect("hessian");
        let dof = 3 * molecule.atoms.len();
        assert_eq!(analytic.nrows(), dof);

        // 冻结密度下的 2e 梯度标量：½Σ P G^x。
        let scalar = |moved: &Molecule| -> Vec<f64> {
            let moved_basis =
                InternalBasis::build(registry.get_basis("sto-3g").expect("basis"), moved)
                    .expect("basis");
            let moved_tensor = two_electron(&moved_basis, moved).expect("2e");
            let one = one_electron(&moved_basis, moved).expect("1e");
            let fock = crate::build_fock(&one.core_hamiltonian, &moved_tensor, &result.density)
                .expect("fock");
            let components = gradient_components(&moved_basis, moved, &result.density, &fock.fock)
                .expect("components");
            let (moved_cartesian, moved_expansion) =
                cartesian_orbitals(&moved_basis, moved).expect("cartesian");
            let moved_n = moved_cartesian.len();
            let _ = moved_n;
            let _ = moved_expansion;
            let mut out = vec![0.0_f64; dof];
            for atom in 0..molecule.atoms.len() {
                for axis in 0..3 {
                    out[3 * atom + axis] = components.two_electron[atom][axis];
                }
            }
            out
        };
        let displaced = |atom: usize, axis: usize, amount: f64| {
            let mut moved = molecule.clone();
            moved.atoms[atom].position_angstrom[axis] += amount * ANGSTROM_PER_BOHR;
            moved
        };
        let mut worst = 0.0_f64;
        for atom in 0..molecule.atoms.len() {
            for axis in 0..3 {
                let second_column = 3 * atom + axis;
                // Richardson：把 `O(δ²)` 的截断项外推掉，否则判据只能停在 1e-6 量级。
                let coarse = {
                    let plus = scalar(&displaced(atom, axis, delta));
                    let minus = scalar(&displaced(atom, axis, -delta));
                    (0..dof)
                        .map(|row| (plus[row] - minus[row]) / (2.0 * delta))
                        .collect::<Vec<f64>>()
                };
                let fine = {
                    let plus = scalar(&displaced(atom, axis, 0.5 * delta));
                    let minus = scalar(&displaced(atom, axis, -0.5 * delta));
                    (0..dof)
                        .map(|row| (plus[row] - minus[row]) / delta)
                        .collect::<Vec<f64>>()
                };
                for first_row in 0..dof {
                    // 差分给的是 ∂/∂y[E_2e 梯度_x]，因此它是 H[x][y]。
                    let numeric = (4.0 * fine[first_row] - coarse[first_row]) / 3.0;
                    let value = analytic[(first_row, second_column)];
                    let scale = numeric.abs().max(value.abs()).max(1.0e-3);
                    let difference = (numeric - value).abs() / scale;
                    assert!(
                        difference < 1.0e-7,
                        "E_2e[{first_row}][{second_column}]: 解析 {value:.10e} vs 差分 {numeric:.10e}（相对差 {difference:.2e}）"
                    );
                    worst = worst.max(difference);
                }
            }
        }
        // 对称性：二阶导必须对称（与差分无关的一条独立性质）。
        for i in 0..dof {
            for j in 0..dof {
                // 噪声地板 1e-6：二阶导的量级是 1–20，纯零元的差分噪声在 1e-15 量级。
                let scale = analytic[(i, j)]
                    .abs()
                    .max(analytic[(j, i)].abs())
                    .max(1.0e-6);
                assert!(
                    (analytic[(i, j)] - analytic[(j, i)]).abs() < 1.0e-9 * scale,
                    "2e 二阶导不对称：({i},{j}) = {:.6e} vs {:.6e}",
                    analytic[(i, j)],
                    analytic[(j, i)]
                );
            }
        }
        println!("2e 显式二阶导的最大相对差：{worst:.2e}");
    }

    /// 成本测量：AO 层 2e 二阶导收缩 + 全部 3N 个方向的块（水/def2-SVP）。
    ///
    /// 这是 RS-05 的 20× 能不能达成的**真实**输入数字（`docs/choices.md` 一百一十
    /// 承诺过「到 AO 层再量一次」）。默认 `#[ignore]`，不进主门禁。
    #[test]
    #[ignore = "成本测量，不进主门禁"]
    fn cost_measurement_def2svp() {
        use std::time::Instant;
        let registry = Registry::open(data_root()).expect("registry");
        let molecule = water();
        let basis = InternalBasis::build(registry.get_basis("def2-svp").expect("basis"), &molecule)
            .expect("build basis");
        let grid =
            qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
                .expect("grid");
        let tensor = two_electron(&basis, &molecule).expect("2e");
        let options = RksOptions {
            max_iter: 400,
            energy_threshold: 1.0e-12,
            density_threshold: 1.0e-10,
            ..RksOptions::default()
        };
        let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("scf");
        let (cartesian, expansion) = cartesian_orbitals(&basis, &molecule).expect("cartesian");
        let n = cartesian.len();
        let density = cartesian_density(&result.density, &expansion, n);
        let dof = 3 * molecule.atoms.len();

        // 预热（缓存/分支预测）。
        let _ = crate::cphf_hessian::two_electron_hessian_contracted(&cartesian, &density)
            .expect("warmup");
        let started = Instant::now();
        let hessian = crate::cphf_hessian::two_electron_hessian_contracted(&cartesian, &density)
            .expect("hessian");
        let second = started.elapsed().as_secs_f64();

        let started = Instant::now();
        let blocks =
            crate::cphf_hessian::cartesian_direction_blocks(&basis, &molecule, &result.density)
                .expect("blocks");
        let first = started.elapsed().as_secs_f64();

        let started = Instant::now();
        let _rhs = crate::cphf_hessian::cphf_right_hand_sides(
            &basis,
            &molecule,
            &result.density,
            &result.coefficients,
            &(0..basis.n_functions())
                .map(|index| result.mo_energies_hartree.get(index))
                .collect::<Vec<f64>>(),
            (molecule
                .atoms
                .iter()
                .map(|atom| qwb_registry::elements::atomic_number(&atom.symbol).unwrap_or(0))
                .sum::<u32>() as usize
                - molecule.charge as usize)
                / 2,
        )
        .expect("rhs");
        let rhs_seconds = started.elapsed().as_secs_f64();

        // 数值 Hessian 的基线：6N 个位移、每个位移 1 次 SCF + 1 次解析梯度。
        let fock = crate::build_fock(
            &one_electron(&basis, &molecule)
                .expect("1e")
                .core_hamiltonian,
            &tensor,
            &result.density,
        )
        .expect("fock")
        .fock;
        let _ =
            crate::analytic_gradient::analytic_gradient(&basis, &molecule, &result.density, &fock)
                .expect("warmup gradient");
        let started = std::time::Instant::now();
        let _gradient =
            crate::analytic_gradient::analytic_gradient(&basis, &molecule, &result.density, &fock)
                .expect("gradient");
        let gradient_seconds = started.elapsed().as_secs_f64();
        let baseline = 6.0 * molecule.atoms.len() as f64 * gradient_seconds;
        println!(
            "一次解析梯度 {gradient_seconds:.3} s ⇒ 数值 Hessian 基线 ≈ {baseline:.1} s（6N = {} 个位移）；解析路径（2e 二阶导 + 块 + 右端）≈ {:.1} s ⇒ {:.2}×",
            6 * molecule.atoms.len(),
            second + first + rhs_seconds,
            baseline / (second + first + rhs_seconds)
        );
        let mut max_offdiag = 0.0_f64;
        for i in 0..dof {
            for j in 0..dof {
                if i != j {
                    max_offdiag = max_offdiag.max(hessian[(i, j)].abs());
                }
            }
        }
        println!(
            "水/def2-SVP（n_cart={n}, dof={dof}）：2e 二阶导 {second:.3} s、3N 个块 {first:.3} s（{} 个方向）、右端 {rhs_seconds:.3} s；max|offdiag| {max_offdiag:.3e}；块数 {}",
            blocks.len(),
            blocks.len()
        );
    }
}
