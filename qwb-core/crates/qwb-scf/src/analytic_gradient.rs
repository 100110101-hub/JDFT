//! 解析梯度（G-01/G-02）：闭壳层 HF / KS 的总能量对核坐标的导数，**任意角动量**。
//!
//! # 结构
//!
//! ```text
//! dE/dR_A = Tr(P dH/dR_A)                     Hellmann–Feynman（1e）
//!          + ½ Σ P P [d(μν|λσ) − ½ d(μλ|νσ)]/dR_A   2e
//!          − Tr(W dS/dR_A)                     Pulay（轨道弛豫）
//!          + dE_nuc/dR_A
//! ```
//!
//! `W = ½ P F P`，恒等式 `Tr(F dP) = −Tr(W dS)` 由 `tests/g01_pulay.rs` 独立验证。
//!
//! # 怎么做到「任意角动量」
//!
//! 两件事，都不需要新积分：
//!
//! 1. **抬/降幂恒等式**。`u = r − A`，`∂u/∂A = −1`，于是
//!
//!    ```text
//!    ∂/∂A_x [u_x^a u_y^b u_z^c e^{−αu²}] = −a·u_x^{a−1}(…) + 2α·u_x^{a+1}(…)
//!    ```
//!
//!    所以任何积分对中心的导数都只是「同一套原函数积分换两个幂次」：
//!    `d = 2α·(抬阶) − a_x·(降阶)`。1e、Pulay、2e 全是这个套路，
//!    而 `hermite2e::primitive_component` 本来就吃任意幂次。
//! 2. **球谐基回到直角分量**。球谐 AO 是直角分量的线性组合，所以先把密度张量展开到
//!    直角分量（逐壳层 `ρ̃ = Tᵀ P T`），再在直角基上做第 1 步。梯度是标量，不用变回去。
//!
//! 核吸引里「核本身在动」那一项**不需要新闭式**：`⟨a_A|−Z/r_C|b_B⟩` 只依赖 `A−C`、`B−C`，
//! 于是 `∂/∂C = −(∂/∂A + ∂/∂B)`（C 固定），右边正是第 1 步已经会算的东西。
//! （s 型的那个显式闭式 `nuclear_center_derivative` 仍然保留，给 `g01_nuclear_derivative`
//! 独立钉住——它当初写错过，值得留一条只测它的测试。）
//!
//! # 分项与自检
//!
//! [`gradient_components`] 把结果按物理来源拆成六组，`tests/g01_components.rs` 用它做分项
//! 对照；[`analytic_gradient`] 只是六项之和。总和差一点点时，先定位到**哪一项**，否则只能靠猜。
//!
//! # 范围
//!
//! 任意 `l`（只要基组数据里有）。2e 项按 `n_cart⁴` 直接遍历：STO-3G 级别毫秒级，
//! def2-SVP 明显变慢（要优化得先上分块/预筛）。

use qwb_basis::{InternalBasis, cartesian_components};
use qwb_engine_api::{EngineError, Molecule};
use qwb_integrals::boys::boys_functions;
use qwb_integrals::hermite2e::primitive_component;
use qwb_integrals::one_electron::{
    primitive_kinetic, primitive_norm, primitive_nuclear, primitive_overlap,
};
use qwb_linalg::Matrix;
use qwb_registry::elements::atomic_number;

use crate::pulay::pulay_w;

/// [u32; 3] → 积分库要的 Powers 元组。
fn powers_tuple(powers: [u32; 3]) -> (u32, u32, u32) {
    (powers[0], powers[1], powers[2])
}

const PI: f64 = std::f64::consts::PI;

/// 一个**直角分量** AO：球谐 AO 拆开后的最小单位。
#[derive(Clone)]
pub struct CartAo {
    /// 球心（Bohr）。
    pub center: [f64; 3],
    /// 所属原子下标。
    pub center_index: usize,
    /// 原函数指数。
    pub exponents: Vec<f64>,
    /// 收缩系数（与指数一一对应；`InternalBasis` 已含归一化）。
    pub coefficients: Vec<f64>,
    /// 直角分量幂次。
    pub powers: [u32; 3],
}

/// 抬/降幂恒等式：对中心求导 = `2α·抬阶 − a_x·降阶`。
fn raise_lower<F>(alpha: f64, powers: [u32; 3], axis: usize, integral: F) -> f64
where
    F: Fn([u32; 3]) -> f64,
{
    let mut raised = powers;
    raised[axis] += 1;
    let mut total = 2.0 * alpha * integral(raised);
    if powers[axis] > 0 {
        let mut lowered = powers;
        lowered[axis] -= 1;
        total -= f64::from(powers[axis]) * integral(lowered);
    }
    total
}

/// 一个球谐 AO 的展开：(直角分量下标, 系数) 列表。
type Expansion = Vec<Vec<(usize, f64)>>;

/// 把基组拆成直角分量 AO，并给出「球谐 AO → 直角分量」的展开表。
///
/// # Errors
///
/// 球谐变换缺失，或维度与直角分量数不符。
pub fn cartesian_orbitals(
    basis: &InternalBasis,
    _molecule: &Molecule,
) -> Result<(Vec<CartAo>, Expansion), EngineError> {
    let mut cartesian = Vec::new();
    let mut expansion: Expansion = vec![Vec::new(); basis.n_functions()];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        if components.len() != shell.n_functions() && shell.l > 1 {
            // 球谐壳层：直角分量数 = (l+1)(l+2)/2；两者不等是正常的（l=2 时 6 vs 5）
        }
        let base = cartesian.len();
        for powers in &components {
            cartesian.push(CartAo {
                center: shell.center_bohr,
                center_index: shell.atom_index,
                exponents: shell.exponents.clone(),
                coefficients: shell.coefficients.clone(),
                powers: [powers.0, powers.1, powers.2],
            });
        }
        match qwb_basis::spherical::spherical_transform(shell.l, shell.angular) {
            None => {
                if components.len() != shell.n_functions() {
                    return Err(EngineError::Internal(format!(
                        "l={} 的直角分量数 {} 与壳层函数数 {} 不符",
                        shell.l,
                        components.len(),
                        shell.n_functions()
                    )));
                }
                for component in 0..components.len() {
                    expansion[shell.first_function + component] = vec![(base + component, 1.0)];
                }
            }
            Some(transform) => {
                if transform.n_spherical != shell.n_functions() {
                    return Err(EngineError::Internal(format!(
                        "l={} 的球谐维数 {} 与壳层函数数 {} 不符",
                        shell.l,
                        transform.n_spherical,
                        shell.n_functions()
                    )));
                }
                for spherical in 0..transform.n_spherical {
                    let mut column = Vec::new();
                    for component in 0..components.len() {
                        let weight = transform.get(component, spherical);
                        if weight != 0.0 {
                            column.push((base + component, weight));
                        }
                    }
                    expansion[shell.first_function + spherical] = column;
                }
            }
        }
    }
    Ok((cartesian, expansion))
}

/// 把球谐密度展开到直角分量基：`ρ̃_{cd} = Σ_{μν} P_{μν} t_{μc} t_{νd}`。
fn cartesian_density(
    density: &Matrix,
    expansion: &[Vec<(usize, f64)>], // Expansion 的借用形式
    n_cartesian: usize,
) -> Matrix {
    let mut out = Matrix::zeros(n_cartesian, n_cartesian);
    for mu in 0..density.nrows() {
        for nu in 0..density.ncols() {
            let value = density[(mu, nu)];
            if value == 0.0 {
                continue;
            }
            for (cart_mu, weight_mu) in &expansion[mu] {
                for (cart_nu, weight_nu) in &expansion[nu] {
                    out[(*cart_mu, *cart_nu)] += value * weight_mu * weight_nu;
                }
            }
        }
    }
    out
}

fn nuclei_of(molecule: &Molecule) -> Result<Vec<([f64; 3], f64)>, EngineError> {
    molecule
        .atoms
        .iter()
        .map(|atom| {
            let z = atomic_number(&atom.symbol)
                .map(f64::from)
                .ok_or_else(|| EngineError::Internal(format!("unknown element {}", atom.symbol)))?;
            Ok((atom.position_bohr(), z))
        })
        .collect::<Result<_, EngineError>>()
}

/// 梯度按物理来源拆分的结果（诊断与分项回归用）。
#[derive(Clone, Debug, Default)]
pub struct GradientComponents {
    /// `Tr(P dT/dR)`。
    pub kinetic: Vec<[f64; 3]>,
    /// `Tr(P dV/dR)` 里 AO 移动的那一半。
    pub nuclear_ao: Vec<[f64; 3]>,
    /// `Tr(P dV/dR)` 里「核本身在动」的那一半。
    pub nuclear_center: Vec<[f64; 3]>,
    /// 2e 项。
    pub two_electron: Vec<[f64; 3]>,
    /// `−Tr(W dS)`。
    pub pulay: Vec<[f64; 3]>,
    /// 核排斥导数。
    pub nuclear_repulsion: Vec<[f64; 3]>,
}

impl GradientComponents {
    fn total(&self, atom: usize, axis: usize) -> f64 {
        self.kinetic[atom][axis]
            + self.nuclear_ao[atom][axis]
            + self.nuclear_center[atom][axis]
            + self.two_electron[atom][axis]
            + self.pulay[atom][axis]
            + self.nuclear_repulsion[atom][axis]
    }
}

/// 一对直角 AO 在「`first` 这一侧中心移动」时的导数贡献（未乘密度）。
enum PairKind {
    Kinetic,
    Nuclear,
    Overlap,
}

fn pair_derivative(
    first: &CartAo,
    second: &CartAo,
    nuclei: &[([f64; 3], f64)],
    axis: usize,
    kind: PairKind,
) -> f64 {
    let mut total = 0.0;
    for (alpha, ca) in first.exponents.iter().zip(&first.coefficients) {
        let na = primitive_norm(powers_tuple(first.powers), *alpha);
        for (beta, cb) in second.exponents.iter().zip(&second.coefficients) {
            let nb = primitive_norm(powers_tuple(second.powers), *beta);
            let weight = ca * cb * na * nb;
            let value = match kind {
                PairKind::Kinetic => raise_lower(*alpha, first.powers, axis, |powers| {
                    primitive_kinetic(
                        *alpha,
                        (powers[0], powers[1], powers[2]),
                        first.center,
                        *beta,
                        powers_tuple(second.powers),
                        second.center,
                    )
                }),
                PairKind::Overlap => raise_lower(*alpha, first.powers, axis, |powers| {
                    primitive_overlap(
                        *alpha,
                        (powers[0], powers[1], powers[2]),
                        first.center,
                        *beta,
                        powers_tuple(second.powers),
                        second.center,
                    )
                }),
                PairKind::Nuclear => {
                    let mut sum = 0.0;
                    for (nucleus, charge) in nuclei {
                        sum += raise_lower(*alpha, first.powers, axis, |powers| {
                            primitive_nuclear(
                                *alpha,
                                (powers[0], powers[1], powers[2]),
                                first.center,
                                *beta,
                                powers_tuple(second.powers),
                                second.center,
                                *nucleus,
                                *charge,
                            )
                        });
                    }
                    sum
                }
            };
            total += weight * value;
        }
    }
    total
}

/// 2e：`(ab|cd)` 的 `slot`（0=a,1=b,2=c,3=d）所在中心求导。
///
/// `d` 只有「槽位 0」，这里用 `(i j|k l)` 的对内互换与换对把目标槽位换到 0。
fn quartet_derivative(
    quartet: [usize; 4],
    exponents: &[f64; 4],
    cartesian: &[CartAo],
    slot: usize,
    axis: usize,
) -> f64 {
    let order: [usize; 4] = match slot {
        0 => [0, 1, 2, 3],
        1 => [1, 0, 2, 3],
        2 => [2, 3, 0, 1],
        _ => [3, 2, 0, 1],
    };
    let index = [
        quartet[order[0]],
        quartet[order[1]],
        quartet[order[2]],
        quartet[order[3]],
    ];
    let exponent = [
        exponents[order[0]],
        exponents[order[1]],
        exponents[order[2]],
        exponents[order[3]],
    ];
    raise_lower(exponent[0], cartesian[index[0]].powers, axis, |powers| {
        primitive_component(
            powers,
            cartesian[index[0]].center,
            exponent[0],
            cartesian[index[1]].powers,
            cartesian[index[1]].center,
            exponent[1],
            cartesian[index[2]].powers,
            cartesian[index[2]].center,
            exponent[2],
            cartesian[index[3]].powers,
            cartesian[index[3]].center,
            exponent[3],
        )
    })
}

/// 2e 项：`½ Σ ρ̃ρ̃ [d(ab|cd) − ½ d(ac|bd)]`。
fn two_electron_all(cartesian: &[CartAo], density: &Matrix) -> Vec<[f64; 3]> {
    let n = cartesian.len();
    let natoms = cartesian
        .iter()
        .map(|orbital| orbital.center_index)
        .max()
        .map_or(0, |index| index + 1);
    let mut out = vec![[0.0_f64; 3]; natoms];
    // 原函数归一化只与（轨道、指数）有关：**预先算一次**，不要在 3N 遍里反复算。
    let norms: Vec<Vec<f64>> = cartesian
        .iter()
        .map(|orbital| {
            let powers = powers_tuple(orbital.powers);
            orbital
                .exponents
                .iter()
                .map(|alpha| primitive_norm(powers, *alpha))
                .collect()
        })
        .collect();
    for a in 0..n {
        for b in 0..n {
            let p_ab = density[(a, b)];
            if p_ab == 0.0 {
                continue;
            }
            for c in 0..n {
                for d in 0..n {
                    let p_cd = density[(c, d)];
                    if p_cd == 0.0 {
                        continue;
                    }
                    let coulomb_owners = [a, b, c, d];
                    let exchange_owners = [a, c, b, d];
                    // 每个槽位、每个轴的导数先在本 quartet 内累起来，最后一次性加到
                    // 该槽位所属的原子上（槽位 → 原子的映射每个 quartet 只查一次）。
                    let mut coulomb = [[0.0_f64; 3]; 4];
                    let mut exchange = [[0.0_f64; 3]; 4];
                    for (ia, &alpha) in cartesian[a].exponents.iter().enumerate() {
                        let ca = cartesian[a].coefficients[ia];
                        let na = norms[a][ia];
                        for (ib, &beta) in cartesian[b].exponents.iter().enumerate() {
                            let cb = cartesian[b].coefficients[ib];
                            let nb = norms[b][ib];
                            for (ic, &gamma) in cartesian[c].exponents.iter().enumerate() {
                                let cc = cartesian[c].coefficients[ic];
                                let nc = norms[c][ic];
                                for (id, &delta) in cartesian[d].exponents.iter().enumerate() {
                                    let cd = cartesian[d].coefficients[id];
                                    let nd = norms[d][id];
                                    let weight = ca * cb * cc * cd * na * nb * nc * nd;
                                    let exponents = [alpha, beta, gamma, delta];
                                    // 交换积分的槽位顺序是 a,c,b,d，指数必须跟着换位
                                    let exchange_exponents = [alpha, gamma, beta, delta];
                                    for slot in 0..4 {
                                        for axis in 0..3 {
                                            coulomb[slot][axis] += weight
                                                * quartet_derivative(
                                                    coulomb_owners,
                                                    &exponents,
                                                    cartesian,
                                                    slot,
                                                    axis,
                                                );
                                            exchange[slot][axis] += weight
                                                * quartet_derivative(
                                                    exchange_owners,
                                                    &exchange_exponents,
                                                    cartesian,
                                                    slot,
                                                    axis,
                                                );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let factor = 0.5 * p_ab * p_cd;
                    for slot in 0..4 {
                        let atom = cartesian[coulomb_owners[slot]].center_index;
                        for axis in 0..3 {
                            out[atom][axis] += factor * coulomb[slot][axis];
                        }
                        let atom = cartesian[exchange_owners[slot]].center_index;
                        for axis in 0..3 {
                            out[atom][axis] -= factor * 0.5 * exchange[slot][axis];
                        }
                    }
                }
            }
        }
    }
    out
}

/// 核排斥导数：`−Σ_B Z_A Z_B (R_A − R_B)_axis / |R_A − R_B|³`。
fn nuclear_repulsion_axis(nuclei: &[([f64; 3], f64)], atom: usize, axis: usize) -> f64 {
    let (own_position, own_charge) = nuclei[atom];
    let mut total = 0.0;
    for (other, (other_position, other_charge)) in nuclei.iter().enumerate() {
        if other == atom {
            continue;
        }
        let delta = [
            own_position[0] - other_position[0],
            own_position[1] - other_position[1],
            own_position[2] - other_position[2],
        ];
        let r = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        total -= own_charge * other_charge * delta[axis] / (r * r * r);
    }
    total
}

/// 核吸引里「核本身在动」那一项：`∂/∂C = −(∂/∂A + ∂/∂B)`。
fn nuclear_center_axis(
    cartesian: &[CartAo],
    density: &Matrix,
    nuclei: &[([f64; 3], f64)],
    atom: usize,
    axis: usize,
) -> f64 {
    let n = cartesian.len();
    let (position, charge) = nuclei[atom];
    let mut total = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            let p_mu_nu = density[(mu, nu)];
            if p_mu_nu == 0.0 {
                continue;
            }
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
                    let d_a = raise_lower(*alpha, cartesian[mu].powers, axis, |powers| {
                        primitive_nuclear(
                            *alpha,
                            (powers[0], powers[1], powers[2]),
                            cartesian[mu].center,
                            *beta,
                            powers_tuple(cartesian[nu].powers),
                            cartesian[nu].center,
                            position,
                            charge,
                        )
                    });
                    let d_b = raise_lower(*beta, cartesian[nu].powers, axis, |powers| {
                        primitive_nuclear(
                            *alpha,
                            powers_tuple(cartesian[mu].powers),
                            cartesian[mu].center,
                            *beta,
                            powers_tuple(powers),
                            cartesian[nu].center,
                            position,
                            charge,
                        )
                    });
                    total -= p_mu_nu * weight * (d_a + d_b);
                }
            }
        }
    }
    total
}

/// 把解析梯度按物理来源拆开。`fock` 必须是**收敛后的** Fock 矩阵。
///
/// # Errors
///
/// `density` / `fock` 形状与基组不符。
pub fn gradient_components(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
    fock: &Matrix,
) -> Result<GradientComponents, EngineError> {
    let (cartesian, expansion) = cartesian_orbitals(basis, molecule)?;
    let n = cartesian.len();
    if density.nrows() != basis.n_functions() || fock.nrows() != basis.n_functions() {
        return Err(EngineError::Internal(
            "analytic_gradient: density/fock size mismatch".to_string(),
        ));
    }
    let w = pulay_w(density, fock)?;
    let density_cart = cartesian_density(density, &expansion, n);
    let w_cart = cartesian_density(&w, &expansion, n);
    let nuclei = nuclei_of(molecule)?;
    let natoms = molecule.atoms.len();

    // 2e 导数是整条梯度里最贵的一段：一次遍历 quartet、把四个中心分别累加，
    // 不要按 3N 个坐标各扫一遍 n^4（旧实现就是那样，实测 5–7 倍浪费）。
    let two_electron_all = two_electron_all(&cartesian, &density_cart);

    let mut components = GradientComponents {
        kinetic: vec![[0.0; 3]; natoms],
        nuclear_ao: vec![[0.0; 3]; natoms],
        nuclear_center: vec![[0.0; 3]; natoms],
        two_electron: vec![[0.0; 3]; natoms],
        pulay: vec![[0.0; 3]; natoms],
        nuclear_repulsion: vec![[0.0; 3]; natoms],
    };
    #[allow(clippy::needless_range_loop)]
    for atom in 0..natoms {
        for axis in 0..3 {
            let mut kinetic = 0.0;
            let mut nuclear_ao = 0.0;
            let mut pulay = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    let p_mu_nu = density_cart[(mu, nu)];
                    let w_mu_nu = w_cart[(mu, nu)];
                    if cartesian[mu].center_index == atom {
                        if p_mu_nu != 0.0 {
                            kinetic += p_mu_nu
                                * pair_derivative(
                                    &cartesian[mu],
                                    &cartesian[nu],
                                    &nuclei,
                                    axis,
                                    PairKind::Kinetic,
                                );
                            nuclear_ao += p_mu_nu
                                * pair_derivative(
                                    &cartesian[mu],
                                    &cartesian[nu],
                                    &nuclei,
                                    axis,
                                    PairKind::Nuclear,
                                );
                        }
                        if w_mu_nu != 0.0 {
                            pulay -= w_mu_nu
                                * pair_derivative(
                                    &cartesian[mu],
                                    &cartesian[nu],
                                    &nuclei,
                                    axis,
                                    PairKind::Overlap,
                                );
                        }
                    }
                    // 另一侧（`nu` 的中心是 atom）同样要计入
                    if cartesian[nu].center_index == atom {
                        if p_mu_nu != 0.0 {
                            kinetic += p_mu_nu
                                * pair_derivative(
                                    &cartesian[nu],
                                    &cartesian[mu],
                                    &nuclei,
                                    axis,
                                    PairKind::Kinetic,
                                );
                            nuclear_ao += p_mu_nu
                                * pair_derivative(
                                    &cartesian[nu],
                                    &cartesian[mu],
                                    &nuclei,
                                    axis,
                                    PairKind::Nuclear,
                                );
                        }
                        if w_mu_nu != 0.0 {
                            pulay -= w_mu_nu
                                * pair_derivative(
                                    &cartesian[nu],
                                    &cartesian[mu],
                                    &nuclei,
                                    axis,
                                    PairKind::Overlap,
                                );
                        }
                    }
                }
            }
            components.kinetic[atom][axis] = kinetic;
            components.nuclear_ao[atom][axis] = nuclear_ao;
            components.pulay[atom][axis] = pulay;
            components.nuclear_center[atom][axis] =
                nuclear_center_axis(&cartesian, &density_cart, &nuclei, atom, axis);
            components.two_electron[atom][axis] = two_electron_all[atom][axis];
            components.nuclear_repulsion[atom][axis] = nuclear_repulsion_axis(&nuclei, atom, axis);
        }
    }
    Ok(components)
}

/// 闭壳层 HF/KS 的解析梯度（任意角动量）。
///
/// `fock` 必须是**收敛后的** Fock 矩阵（与 `density` 一致）。
///
/// # Errors
///
/// 形状不匹配，或基组的球谐变换不一致。
pub fn analytic_gradient(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
    fock: &Matrix,
) -> Result<Vec<[f64; 3]>, EngineError> {
    let components = gradient_components(basis, molecule, density, fock)?;
    Ok((0..molecule.atoms.len())
        .map(|atom| {
            let mut grad = [0.0_f64; 3];
            for (axis, slot) in grad.iter_mut().enumerate() {
                *slot = components.total(atom, axis);
            }
            grad
        })
        .collect())
}

/// `s` 型核吸引「核移动」的显式闭式（只给 `tests/g01_nuclear_derivative.rs` 用）。
///
/// 新实现对任意角动量走 `∂/∂C = −(∂/∂A + ∂/∂B)`，不再需要它；保留是因为它当初写错过，
/// 值得留一条只钉它的测试。
pub fn nuclear_center_derivative(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    c: [f64; 3],
    z: f64,
    axis: usize,
) -> f64 {
    let p = alpha + beta;
    let center = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    let ab2 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
    let pc = (0..3).map(|k| (center[k] - c[k]).powi(2)).sum::<f64>();
    let t = p * pc;
    let f1 = boys_functions(1, t)[1];
    let prefactor = -z * 2.0 * PI / p * (-alpha * beta / p * ab2).exp();
    let d_f0 = -f1 * (-2.0 * p * (center[axis] - c[axis]));
    prefactor * d_f0
}

/// `s` 型四中心原函数对槽位 0 的导数（只给 `tests/g01_2e_primitive.rs` 用）。
#[allow(clippy::too_many_arguments)]
pub fn d_ssss(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    gamma: f64,
    c: [f64; 3],
    delta: f64,
    d: [f64; 3],
    axis: usize,
) -> f64 {
    let s = [0u32; 3];
    raise_lower(alpha, s, axis, |powers| {
        primitive_component(powers, a, alpha, s, b, beta, s, c, gamma, s, d, delta)
    })
}

/// 把槽位换到 0（只给 `tests/g01_2e_primitive.rs` 用）。
pub fn d_ssss_slot(prims: &[(f64, [f64; 3]); 4], slot: usize, axis: usize) -> f64 {
    let order: [usize; 4] = match slot {
        0 => [0, 1, 2, 3],
        1 => [1, 0, 2, 3],
        2 => [2, 3, 0, 1],
        _ => [3, 2, 0, 1],
    };
    let a = prims[order[0]];
    let b = prims[order[1]];
    let c = prims[order[2]];
    let d = prims[order[3]];
    d_ssss(a.0, a.1, b.0, b.1, c.0, c.1, d.0, d.1, axis)
}
