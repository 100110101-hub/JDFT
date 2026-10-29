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
use qwb_linalg::{Matrix, Vector};
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
    /// 2e 项（`库仑 + α_hf·交换`，`α_hf = 1` 即 HF；保留给"全 HF"的调用方）。
    pub two_electron: Vec<[f64; 3]>,
    /// 2e 项里的**库仑**部分（`½ Σ P P (μν|λσ)^R`）。
    pub two_electron_coulomb: Vec<[f64; 3]>,
    /// 2e 项里的**交换**部分（`−¼ Σ P P (μλ|νσ)^R`，即 `α_hf = 1` 时的值）。
    ///
    /// 纯泛函（LDA/GGA）没有这一项，杂化泛函要乘 `α_hf`；把它单独拆出来，
    /// 才不会把 HF 交换当成 KS 能量的一部分（这是 `g01_analytic_pbe` 一开始
    /// 差 8.8e-2 的根因）。
    pub two_electron_exchange: Vec<[f64; 3]>,
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

/// 2e 项：`½ Σ P P d(ab|cd) − ¼ Σ P P d(ac|bd)`。
///
/// # 两个乘数级的加速（2026-10-25，T4 的前置）
///
/// 1. **把交换项搬到同一张积分上**：令 `b↔c` 换标号，
///    `Σ P_ab P_cd (ac|bd) = Σ P_ac P_bd (ab|cd)`，
///    于是库仑与交换共用同一组 `(ab|cd)` 的槽位导数（原来要算两遍积分）。
/// 2. **8 重置换对称**：`(ab|cd)` 在「对内互换」与「换对」下不变。于是只遍历**规范** quartet
///    （`a≤b`、`c≤d`、`(a,b)≤(c,d)`），每个 quartet 只算 12 个槽位导数（4 槽 × 3 轴），
///    再把权重按轨道对称性乘回去。原来每个 quartet 算 24 个导数、且全量 `n⁴` 遍历。
///
/// 两条合起来把导数积分次数降到原来的 ≈1/16。判据不变：`g01_analytic`（H2 3.4e-9）、
/// `g01_analytic_water`（6.7e-7）、`g01_components`（逐项）、`g01_analytic_pbe`（8.4e-7）、
/// `t6_b3lyp_gradient`（7.3e-8）全部继续成立 —— 加速不能改变数字。
fn two_electron_all(cartesian: &[CartAo], density: &Matrix) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let n = cartesian.len();
    let natoms = cartesian
        .iter()
        .map(|orbital| orbital.center_index)
        .max()
        .map_or(0, |index| index + 1);
    let mut coulomb_out = vec![[0.0_f64; 3]; natoms];
    let mut exchange_out = vec![[0.0_f64; 3]; natoms];

    // 每个 AO 的「收缩系数 × 原函数归一化」预先算一次（原来在 quartet 内层重复乘）。
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

    // 8 重对称的置换：对内互换（a,b）/（c,d）与换对（ab）↔（cd）。
    // 用轨道下标的元组去重：索引重合时多个置换会给出同一个 quartet，不能重复累加。
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
                    // 换对对称：只保留 (a,b) ≤ (c,d)
                    if (a, b) > (c, d) {
                        continue;
                    }
                    // 该 quartet 的权重（库仑 + 交换）是否全零：先看 8 个成员
                    let mut any = false;
                    for perm in PERMS.iter() {
                        let i = [a, b, c, d];
                        let (i0, i1, i2, i3) = (i[perm[0]], i[perm[1]], i[perm[2]], i[perm[3]]);
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

                    // 规范 quartet 的 12 个槽位导数（每个原函数组合）
                    let mut slot_derivatives = [[[0.0_f64; 3]; 4]; 4]; // [prim][slot][axis]
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
                                    #[allow(clippy::needless_range_loop)]
                                    // slot/axis 同时用作下标
                                    // 平动不变性：`∂/∂A + ∂/∂B + ∂/∂C + ∂/∂D = 0`（四个中心同时平移
                                    // 时积分不变），所以第 4 个槽位可以直接从前面三个得到 —— 省掉
                                    // 1/4 的 `primitive_component` 调用（这是梯度里最贵的一段）。
                                    #[allow(clippy::needless_range_loop)]
                                    // 一次 setup 出全部 12 个导数：`primitive_component_derivatives`
                                    // 用同一份 `PrimitiveContext`（预因子 + Hermite-Coulomb 核缓存）
                                    // 拿四个中心的三轴导数，`∂/∂D` 由平动不变性得到。
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
                                    #[allow(clippy::needless_range_loop)]
                                    for slot in 0..4 {
                                        for axis in 0..3 {
                                            slot_derivatives[0][slot][axis] +=
                                                weight * local[slot][axis];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    #[allow(clippy::needless_range_loop)] // slot/axis 同时用作下标
                    let derivatives = &slot_derivatives[0];

                    // 把权重按 8 个置换乘回去（去重）
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
                        let coulomb_weight =
                            0.5 * density[(member[0], member[1])] * density[(member[2], member[3])];
                        let exchange_weight = -0.25
                            * density[(member[0], member[2])]
                            * density[(member[1], member[3])];
                        if coulomb_weight == 0.0 && exchange_weight == 0.0 {
                            continue;
                        }
                        for slot in 0..4 {
                            let atom = cartesian[member[slot]].center_index;
                            // 成员的第 `slot` 个轨道来自规范 quartet 的第 `perm[slot]` 槽位。
                            // 必须用置换映射，**不能**按"下标相等"去找规范槽位：索引重复时
                            // （例如 a==b）不同槽位的导数是不同的量。
                            let canonical = perm[slot];
                            for axis in 0..3 {
                                let value = derivatives[canonical][axis];
                                coulomb_out[atom][axis] += coulomb_weight * value;
                                exchange_out[atom][axis] += exchange_weight * value;
                            }
                        }
                    }
                }
            }
        }
    }
    (coulomb_out, exchange_out)
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
    // 库仑与交换分开返回：纯泛函没有交换项，杂化泛函要按 α_hf 缩放。
    let (two_electron_coulomb, two_electron_exchange) = two_electron_all(&cartesian, &density_cart);

    let mut components = GradientComponents {
        kinetic: vec![[0.0; 3]; natoms],
        nuclear_ao: vec![[0.0; 3]; natoms],
        nuclear_center: vec![[0.0; 3]; natoms],
        two_electron: vec![[0.0; 3]; natoms],
        two_electron_coulomb: vec![[0.0; 3]; natoms],
        two_electron_exchange: vec![[0.0; 3]; natoms],
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
            components.two_electron_coulomb[atom][axis] = two_electron_coulomb[atom][axis];
            components.two_electron_exchange[atom][axis] = two_electron_exchange[atom][axis];
            components.two_electron[atom][axis] =
                two_electron_coulomb[atom][axis] + two_electron_exchange[atom][axis];
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

/// UHF 的解析梯度（= [`open_shell_analytic_gradient`]，`α_hf = 1`、无 XC）。
///
/// # Errors
///
/// 同 [`open_shell_analytic_gradient`]。
pub fn uhf_analytic_gradient(
    basis: &InternalBasis,
    molecule: &Molecule,
    p_alpha: &Matrix,
    p_beta: &Matrix,
    w_alpha: &Matrix,
    w_beta: &Matrix,
) -> Result<Vec<[f64; 3]>, EngineError> {
    let zeros = vec![[0.0_f64; 3]; molecule.atoms.len()];
    open_shell_analytic_gradient(
        basis, molecule, p_alpha, p_beta, w_alpha, w_beta, 1.0, &zeros,
    )
}

/// **UKS 的解析梯度**（开壳层 DFT；SC-02）。
///
/// 与闭壳层的 [`crate::rks::rks_analytic_gradient_with_xc`] 对应，只是把 `J` 用总密度、
/// `K`/`v_xc` 按自旋拆开：
///
/// ```text
/// F^σ = H + J(P_tot) − ½α_hf K(P^σ) + v_xc^σ    ⇒   W^σ = P^σ F^σ P^σ
/// ```
///
/// `v_xc^σ` 由 [`crate::uks::evaluate_xc_spin_cached`] 给出（GGA/meta-GGA 的 ∇ρ 项已经在
/// 泛函导数里），`xc_gradient` 是固定密度下的 XC 几何导数。
///
/// # Errors
///
/// XC 求值失败、尺寸不匹配、泛函不支持自旋。
#[allow(clippy::too_many_arguments)]
pub fn uks_analytic_gradient(
    basis: &InternalBasis,
    molecule: &Molecule,
    grid: &qwb_grid::Grid,
    hcore: &Matrix,
    tensor: &qwb_integrals::two_electron::TwoElectronIntegrals,
    p_alpha: &Matrix,
    p_beta: &Matrix,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
    hf_exchange: f64,
    xc_gradient: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, EngineError> {
    let n = basis.n_functions();
    let mut p_total = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            p_total[(mu, nu)] = p_alpha[(mu, nu)] + p_beta[(mu, nu)];
        }
    }
    let coulomb = crate::build_fock(hcore, tensor, &p_total)
        .map_err(|err| EngineError::Internal(format!("uks_analytic_gradient: J: {err}")))?
        .coulomb;
    let k_alpha = crate::build_fock(hcore, tensor, p_alpha)
        .map_err(|err| EngineError::Internal(format!("uks_analytic_gradient: K_alpha: {err}")))?
        .exchange;
    let k_beta = crate::build_fock(hcore, tensor, p_beta)
        .map_err(|err| EngineError::Internal(format!("uks_analytic_gradient: K_beta: {err}")))?
        .exchange;
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);
    let cache =
        crate::rks::GridAoCached::build(basis, grid, needs_sigma, needs_laplacian, needs_tau)?;
    let xc = crate::uks::evaluate_xc_spin_cached(basis, grid, &cache, p_alpha, p_beta, terms)?;

    let mut w_alpha = Matrix::zeros(n, n);
    let mut w_beta = Matrix::zeros(n, n);
    let mut fock = Matrix::zeros(n, n);
    for spin_alpha in [true, false] {
        let (p, v_xc, k) = if spin_alpha {
            (p_alpha, &xc.v_alpha, &k_alpha)
        } else {
            (p_beta, &xc.v_beta, &k_beta)
        };
        for mu in 0..n {
            for nu in 0..n {
                fock[(mu, nu)] = hcore[(mu, nu)] + coulomb[(mu, nu)]
                    // 开壳层没有 ½：`uks.rs` 的 Fock 是 H + J + v − α_hf·K_σ
                    - hf_exchange * k[(mu, nu)]
                    + v_xc[(mu, nu)];
            }
        }
        let pf = qwb_linalg::matmul(p, &fock)
            .map_err(|err| EngineError::Internal(format!("uks_analytic_gradient: PF: {err}")))?;
        let pfp = qwb_linalg::matmul(&pf, p)
            .map_err(|err| EngineError::Internal(format!("uks_analytic_gradient: PFP: {err}")))?;
        let target = if spin_alpha {
            &mut w_alpha
        } else {
            &mut w_beta
        };
        for mu in 0..n {
            for nu in 0..n {
                target[(mu, nu)] = pfp[(mu, nu)];
            }
        }
    }
    open_shell_analytic_gradient(
        basis,
        molecule,
        p_alpha,
        p_beta,
        &w_alpha,
        &w_beta,
        hf_exchange,
        xc_gradient,
    )
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

/// 由（球谐）MO 系数与轨道能构造**能量权重密度矩阵** `W = Σ_{i<n_occ} ε_i C_μi C_νi`。
///
/// 收敛的 HF/KS 里它与 `P F P` 等价（`P = Σ_i C_iC_iᵀ`、`F C_i = ε_i C_i`），但这里
/// 不需要 Fock —— 开壳层两条自旋各有自己的 `(C, ε)`，直接按自旋算最省事。
///
/// # Errors
///
/// 尺寸不匹配或 `n_occupied > 轨道数` → [`EngineError::Internal`]。
pub fn energy_weighted_density(
    coefficients: &Matrix,
    mo_energies: &Vector,
    n_occupied: usize,
) -> Result<Matrix, EngineError> {
    let n = coefficients.nrows();
    if coefficients.ncols() != mo_energies.len() || n_occupied > mo_energies.len() {
        return Err(EngineError::Internal(format!(
            "energy_weighted_density: {}x{} coefficients for {} MO energies, {} occupied",
            coefficients.nrows(),
            coefficients.ncols(),
            mo_energies.len(),
            n_occupied
        )));
    }
    let mut w = Matrix::zeros(n, n);
    for i in 0..n_occupied {
        let eps = mo_energies.get(i);
        for mu in 0..n {
            let c_mu = coefficients.get(mu, i);
            if c_mu == 0.0 {
                continue;
            }
            for nu in 0..n {
                w[(mu, nu)] += eps * c_mu * coefficients.get(nu, i);
            }
        }
    }
    Ok(w)
}

/// **UHF 的解析梯度**（开壳层，纯 HF；SC-02）。
///
/// 能量与导数：
///
/// ```text
/// E = Σ_σ [ Tr(P^σ H) + ½ Tr(P^σ G^σ) ] + E_nuc ,   G^σ = J(P^α+P^β) − K(P^σ)
/// dE/dR = Σ_σ Tr(P^σ dH/dR) + ½ Σ_σ Tr(P^σ dG^σ/dR) − Tr(W dS/dR) + dE_nuc/dR
/// ```
///
/// 注意 `½ Σ_σ Tr(P^σ dG^σ)` **不能**写成闭壳层的 `½Tr(P_tot dG(P_tot))`：
/// 库仑项用 `P_tot` 收缩，交换项必须**按自旋分开**收缩。本函数里的做法是
///
/// ```text
/// 2e 项 = 库仑(P_tot) + 2·[交换(P^α) + 交换(P^β)]
/// ```
///
/// 其中 `交换(P)` 取闭壳层 `two_electron_all` 的交换分量（值 = `−¼ ΣPP(μλ|νσ)^R`），
/// 系数 2 来自 `−½ Σ_σ Σ P^σP^σ(μλ|νσ)^R = 2 Σ_σ 交换(P^σ)`。闭壳层极限
/// （`P^α = P^β = P/2`）下它与 [`analytic_gradient`] 逐位一致 —— 测试
/// `tests/sc02_uhf_gradient.rs` 钉住这一点以及 OH/O2 的数值梯度判据。
///
/// # Errors
///
/// 尺寸不匹配，或基组的球谐变换不一致。
///
/// `hf_exchange` 缩放交换项（UHF = 1、纯泛函 = 0、杂化 = α），`xc_gradient` 是
/// **固定密度**下的 XC 几何导数（网格/权重随核移动那一半；UHF 传全零）。
///
/// # Errors
///
/// 尺寸不匹配、`xc_gradient` 长度不对，或基组的球谐变换不一致。
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn open_shell_analytic_gradient(
    basis: &InternalBasis,
    molecule: &Molecule,
    p_alpha: &Matrix,
    p_beta: &Matrix,
    w_alpha: &Matrix,
    w_beta: &Matrix,
    hf_exchange: f64,
    xc_gradient: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, EngineError> {
    let (cartesian, expansion) = cartesian_orbitals(basis, molecule)?;
    let n = cartesian.len();
    let n_functions = basis.n_functions();
    for (name, matrix) in [
        ("p_alpha", p_alpha),
        ("p_beta", p_beta),
        ("w_alpha", w_alpha),
        ("w_beta", w_beta),
    ] {
        if matrix.nrows() != n_functions || matrix.ncols() != n_functions {
            return Err(EngineError::Internal(format!(
                "uhf_analytic_gradient: {name} is {}x{} for {n_functions} AO",
                matrix.nrows(),
                matrix.ncols()
            )));
        }
    }

    if xc_gradient.len() != molecule.atoms.len() {
        return Err(EngineError::Internal(format!(
            "open_shell_analytic_gradient: xc gradient has {} entries, molecule has {} atoms",
            xc_gradient.len(),
            molecule.atoms.len()
        )));
    }
    let mut p_total = Matrix::zeros(n_functions, n_functions);
    let mut w_total = Matrix::zeros(n_functions, n_functions);
    for mu in 0..n_functions {
        for nu in 0..n_functions {
            p_total[(mu, nu)] = p_alpha[(mu, nu)] + p_beta[(mu, nu)];
            w_total[(mu, nu)] = w_alpha[(mu, nu)] + w_beta[(mu, nu)];
        }
    }
    let density_cart = cartesian_density(&p_total, &expansion, n);
    let w_cart = cartesian_density(&w_total, &expansion, n);
    let alpha_cart = cartesian_density(p_alpha, &expansion, n);
    let beta_cart = cartesian_density(p_beta, &expansion, n);

    let nuclei = nuclei_of(molecule)?;
    let natoms = molecule.atoms.len();
    // 2e 导数：库仑用总密度收缩一次，交换按自旋各收缩一次。
    let (coulomb, _) = two_electron_all(&cartesian, &density_cart);
    let (_, exchange_alpha) = two_electron_all(&cartesian, &alpha_cart);
    let (_, exchange_beta) = two_electron_all(&cartesian, &beta_cart);

    let mut gradient = vec![[0.0_f64; 3]; natoms];
    for atom in 0..natoms {
        for axis in 0..3 {
            gradient[atom][axis] = nuclear_repulsion_axis(&nuclei, atom, axis)
                + nuclear_center_axis(&cartesian, &density_cart, &nuclei, atom, axis)
                + coulomb[atom][axis]
                + 2.0 * hf_exchange * (exchange_alpha[atom][axis] + exchange_beta[atom][axis])
                + xc_gradient[atom][axis];
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
            gradient[atom][axis] += kinetic + nuclear_ao + pulay;
        }
    }
    Ok(gradient)
}
