//! I-05：ECP（有效核心势）矩阵元 —— 局域部分 + 非局域投影。
//!
//! # 数据形态（K1 `ecp`）
//!
//! `V_l(r) = Σ_k c_k r^{n_k} e^{−α_k r²}`，通道按 `l` 分组；**最大的 `l` 是局域通道**
//! （EMSL/Stuttgart 约定），其余 `l < L` 是投影通道。于是
//!
//! ```text
//! V_ECP(r) = V_L(r) + Σ_{l<L} [V_l(r) − V_L(r)] P_l
//! P_l      = Σ_m |lm⟩⟨lm|
//! ```
//!
//! 电子-核的点电荷吸引那一半**不在这里**：它用 `z_eff = Z − n_core` 交给
//! `qwb_integrals::one_electron_with_charges`，这里只给"超出点电荷的那部分"。
//!
//! # 求积方式（半数值）
//!
//! 局域部分是被积函数 `φ_μ(r) V_L(|r−A|) φ_ν(r)` 在 ECP 中心 `A` 上的球面积分：
//!
//! ```text
//! ⟨μ|V_L|ν⟩ = Σ_i w_i r_i² V_L(r_i) Σ_j u_j φ_μ(A + r_i Ω_j) φ_ν(A + r_i Ω_j)
//! ```
//!
//! 非局域部分用**角向投影**而不是求 `P_l` 的解析式：在每个径向壳层 `r_i` 上把 AO 值
//! 投影到实球谐 `Y_lm` 上
//!
//! ```text
//! c^μ_{lm}(r_i) = Σ_j u_j Y_lm(Ω_j) φ_μ(A + r_i Ω_j)
//! ⟨μ|[V_l−V_L]P_l|ν⟩ = Σ_i w_i r_i² [V_l(r_i) − V_L(r_i)] Σ_m c^μ_{lm}(r_i) c^ν_{lm}(r_i)
//! ```
//!
//! 这样一中心（μ、ν 都在 ECP 原子上）与异中心（one-center expansion 是无穷级数）两种情况
//! 用**同一套公式**处理，不需要那个无穷展开。
//!
//! # 精度与代价
//!
//! * 一中心有**解析**公式（[`one_center_matrix`]）：`V(r)` 球对称，所以只有
//!   `(l_μ, m_μ) = (l_ν, m_ν)` 的元非零，径向积分是
//!   `∫ r^{2l+2+n} e^{−(α+β+α_v)r²} dr` 的闭式。半数值实现对它必须一致到 1e-10
//!   （`tests/i05_ecp.rs`），这是"求积对不对"的判据。
//! * ECP 的高斯指数可以很大（LANL2DZ 的 Cu 有 α = 512），所以径向网格必须在 `r ≈ 0.03`
//!   附近足够密：默认用 `grid-veryfine` 的定义（140 径向点、M4 映射）。
//! * 代价：`n_points × n_ao²` 一次，几何定了只算一次（不随 SCF 迭代变化）。

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::ecp::{Ecp, RPowerConvention};

/// 一个 ECP 高斯项：`coefficient * r^power * exp(-exponent * r^2)`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcpTerm {
    /// `r` 的幂次（按内部约定 `r^n`）。
    pub power: i32,
    /// 高斯指数（Bohr^-2）。
    pub exponent: f64,
    /// 系数。
    pub coefficient: f64,
}

/// 一个角动量通道。
#[derive(Debug, Clone, PartialEq)]
pub struct EcpChannel {
    /// 角量子数。
    pub l: u32,
    /// 项列表。
    pub terms: Vec<EcpTerm>,
}

/// ECP 算子在某个分子上的形态：中心、局域通道、投影通道、有效核电荷。
#[derive(Debug, Clone, PartialEq)]
pub struct EcpOperator {
    /// ECP 中心所在的原子下标。
    pub atom_index: usize,
    /// 中心坐标（Bohr）。
    pub center_bohr: [f64; 3],
    /// 元素符号（诊断用）。
    pub element: String,
    /// `z_eff = Z − n_core`：点电荷吸引那一半的电荷。
    pub z_eff: f64,
    /// 局域通道 `V_L`。
    pub local: Vec<EcpTerm>,
    /// 投影通道（`l < L`，按 `l` 升序）。
    pub projectors: Vec<EcpChannel>,
}

impl EcpOperator {
    /// 从 K1 的 [`Ecp`] 构造，并在分子里定位它的元素（必须**恰好一个**原子）。
    ///
    /// # Errors
    ///
    /// * 分子的元素与 ECP 的 `element` 不匹配 → [`EngineError::Data`]；
    /// * 出现两个同元素原子（一个 `ecp_name` 只描述一个中心）→ [`EngineError::Unsupported`]；
    /// * `r_power_convention` 是 `r^(n-2)`（本该在数据层就归一成 `r^n`）→ [`EngineError::Data`]。
    pub fn from_ecp(ecp: &Ecp, molecule: &Molecule) -> Result<Self, EngineError> {
        let mut matching = Vec::new();
        for (index, atom) in molecule.atoms.iter().enumerate() {
            if atom.symbol == ecp.element {
                matching.push(index);
            }
        }
        let atom_index = match matching.as_slice() {
            [only] => *only,
            [] => {
                return Err(EngineError::Data(format!(
                    "ecp `{}` is for {} but the molecule has no {} atom",
                    ecp.name, ecp.element, ecp.element
                )));
            }
            many => {
                return Err(EngineError::Unsupported(format!(
                    "ecp `{}`: {} atoms of {} found at {:?}; a single K2 `ecp_name` describes \
                     one centre only (use a per-atom ECP list, which K2 does not have yet)",
                    ecp.name,
                    many.len(),
                    ecp.element,
                    many
                )));
            }
        };
        if ecp.r_power_convention != RPowerConvention::Rn {
            return Err(EngineError::Data(format!(
                "ecp `{}`: r_power_convention is {:?}, expected r^n (the loader must have \
                 normalised it)",
                ecp.name, ecp.r_power_convention
            )));
        }
        let convert = |terms: &[qwb_registry::ecp::EcpTerm]| -> Vec<EcpTerm> {
            terms
                .iter()
                .map(|term| EcpTerm {
                    power: term.power,
                    exponent: term.exponent,
                    coefficient: term.coefficient,
                })
                .collect()
        };
        let mut projectors: Vec<EcpChannel> = ecp
            .nonlocal_parts
            .iter()
            .map(|part| EcpChannel {
                l: part.l,
                terms: convert(&part.terms),
            })
            .collect();
        projectors.sort_by_key(|channel| channel.l);
        Ok(Self {
            atom_index,
            center_bohr: molecule.atoms[atom_index].position_bohr(),
            element: ecp.element.clone(),
            z_eff: ecp.z_eff,
            local: convert(&ecp.local_part.terms),
            projectors,
        })
    }

    /// 局域通道的 `V_L(r)`。
    #[must_use]
    pub fn local_value(&self, r: f64) -> f64 {
        channel_value(&self.local, r)
    }

    /// 某个投影通道的 `V_l(r)`。
    #[must_use]
    pub fn projector_value(&self, l: u32, r: f64) -> f64 {
        self.projectors
            .iter()
            .find(|channel| channel.l == l)
            .map_or(0.0, |channel| channel_value(&channel.terms, r))
    }
}

fn channel_value(terms: &[EcpTerm], r: f64) -> f64 {
    let mut total = 0.0;
    for term in terms {
        total += term.coefficient * radial_power(r, term.power) * (-term.exponent * r * r).exp();
    }
    total
}

/// `r^n`，覆盖 ECP 数据里出现的 `n ∈ [-2, 2]`（`0^0 = 1`、`0^{-k} = ∞` 这类边界在
/// 求积点上不会出现：`r = 0` 的权重是 0）。
fn radial_power(r: f64, power: i32) -> f64 {
    match power {
        0 => 1.0,
        1 => r,
        2 => r * r,
        -1 => 1.0 / r,
        -2 => 1.0 / (r * r),
        other => r.powi(other),
    }
}

/// 半数值矩阵元：ECP 中心上的径向 × 角向求积（生产路径）。
///
/// 角向/径向规则取自 `qwb-data/grid/` 的同一个等级定义（[`qwb_registry::grid::GridLevelDef`]），
/// 所以"ECP 的求积"和"XC 的求积"用同一套参数，不需要在代码里塞点数。
///
/// # Errors
///
/// 球谐投影只实现到 `l ≤ 2`；更高通道 → [`EngineError::Unsupported`]。
/// 求积规则不可用（数据问题）→ [`EngineError::Data`]。
pub fn matrix_elements(
    basis: &InternalBasis,
    operator: &EcpOperator,
    level: &qwb_registry::grid::GridLevelDef,
) -> Result<Matrix, EngineError> {
    let (directions, angular_weights) = qwb_grid::angular_rule(level)
        .map_err(|err| EngineError::Data(format!("ecp angular rule: {err}")))?;
    let (radii, radial_weights) = qwb_grid::radial_rule(level)
        .map_err(|err| EngineError::Data(format!("ecp radial rule: {err}")))?;
    let n = basis.n_functions();
    let mut matrix = Matrix::zeros(n, n);

    let projector_ls: Vec<u32> = operator
        .projectors
        .iter()
        .map(|channel| channel.l)
        .collect();
    let max_l = projector_ls.iter().copied().max().unwrap_or(0);
    let spherical = real_spherical_harmonics(max_l, &directions)?;
    let stride = 2 * max_l as usize + 1;
    let mut projections = vec![0.0_f64; (max_l as usize + 1) * stride * n];

    let mut values = Vec::with_capacity(n);
    let mut local_blocks = Vec::with_capacity(directions.len());
    for (radius, radial_weight) in radii.iter().zip(&radial_weights) {
        let local = operator.local_value(*radius);
        if local == 0.0 && projector_ls.is_empty() {
            continue;
        }
        // 角向上的 AO 值（每点一组）
        local_blocks.clear();
        projections.iter_mut().for_each(|slot| *slot = 0.0);
        for (direction_index, (direction, angular_weight)) in
            directions.iter().zip(&angular_weights).enumerate()
        {
            let point = [
                operator.center_bohr[0] + radius * direction[0],
                operator.center_bohr[1] + radius * direction[1],
                operator.center_bohr[2] + radius * direction[2],
            ];
            values.clear();
            values.extend_from_slice(&crate::rks::ao_value(basis, point)?);
            local_blocks.push((*angular_weight, values.clone()));
            for l in 0..=max_l as usize {
                let offset = l * stride;
                for m in 0..(2 * l + 1) {
                    let harmonic = spherical[offset + m][direction_index];
                    if harmonic == 0.0 {
                        continue;
                    }
                    let weight = angular_weight * harmonic;
                    let base = (offset + m) * n;
                    for (mu, value) in values.iter().enumerate() {
                        projections[base + mu] += weight * value;
                    }
                }
            }
        }

        // 局域部分：所有 (μ,ν) 对
        if local != 0.0 {
            for (angular_weight, ao) in &local_blocks {
                let w = radial_weight * local * angular_weight;
                for mu in 0..n {
                    if ao[mu] == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        matrix[(mu, nu)] += w * ao[mu] * ao[nu];
                    }
                }
            }
        }

        // 非局域投影：Σ_m c^μ_lm c^ν_lm × (V_l − V_L)
        for l in projector_ls.clone() {
            let delta = operator.projector_value(l, *radius) - local;
            if delta == 0.0 {
                continue;
            }
            let offset = l as usize * stride;
            for m in 0..(2 * l as usize + 1) {
                let base = (offset + m) * n;
                let w = radial_weight * delta;
                for mu in 0..n {
                    let left = projections[base + mu];
                    if left == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        matrix[(mu, nu)] += w * left * projections[base + nu];
                    }
                }
            }
        }
    }
    Ok(matrix)
}

/// 实球谐值表：`[l * (2*max_l+1) + m]` → 每个方向上的 `Y_lm`（`m` 从 `−l` 到 `l`）。
///
/// 归一化是 `∫ Y_lm Y_l'm' dΩ = δ`，所以 `P_l = Σ_m |lm⟩⟨lm|` 的投影系数无需额外因子。
///
/// # Errors
///
/// 只实现 `l ≤ 2`（LANL2DZ/def2-ECP 的通道都在这个范围）；更高 → [`EngineError::Unsupported`]。
pub fn real_spherical_harmonics(
    max_l: u32,
    directions: &[[f64; 3]],
) -> Result<Vec<Vec<f64>>, EngineError> {
    if max_l > 2 {
        return Err(EngineError::Unsupported(format!(
            "ECP angular projection is implemented for l <= 2, got l = {max_l}"
        )));
    }
    let stride = 2 * max_l as usize + 1;
    let mut table: Vec<Vec<f64>> =
        vec![Vec::with_capacity(directions.len()); (max_l as usize + 1) * stride];
    for direction in directions {
        let [x, y, z] = *direction;
        let norm = (x * x + y * y + z * z).sqrt();
        let (x, y, z) = (x / norm, y / norm, z / norm);
        table[0].push((4.0 * std::f64::consts::PI).sqrt().recip());
        if max_l >= 1 {
            let offset = stride;
            let c = (3.0 / (4.0 * std::f64::consts::PI)).sqrt();
            table[offset].push(c * y); // m = -1
            table[offset + 1].push(c * z); // m = 0
            table[offset + 2].push(c * x); // m = +1
        }
        if max_l >= 2 {
            let offset = 2 * stride;
            let c1 = (15.0 / (4.0 * std::f64::consts::PI)).sqrt();
            let c0 = (5.0 / (16.0 * std::f64::consts::PI)).sqrt();
            table[offset].push(c1 * x * y); // m = -2
            table[offset + 1].push(c1 * y * z); // m = -1
            table[offset + 2].push(c0 * (3.0 * z * z - 1.0)); // m = 0
            table[offset + 3].push(c1 * x * z); // m = +1
            table[offset + 4].push((15.0 / (16.0 * std::f64::consts::PI)).sqrt() * (x * x - y * y));
            // m = +2
        }
    }
    Ok(table)
}

/// 半数值路径用的 AO 值（`qwb_scf::rks::ao_value` 的转发，避免循环依赖的假象）。
#[allow(dead_code)]
fn qwb_scf_ao_value(basis: &InternalBasis, point: [f64; 3]) -> Result<Vec<f64>, EngineError> {
    crate::rks::ao_value(basis, point)
}

/// 一中心的**解析**矩阵元（只有 μ、ν 都在 ECP 原子上时才非零）。
///
/// 球对称的 `V(r)` 在直角分量空间里给出
///
/// ```text
/// ⟨μ|V|ν⟩ = Σ_k c_k Σ_{p,q} a_p b_q N_μp N_νq · A(powers_μ + powers_ν)
///           · ∫_0^∞ r^{D + n_k + 2} e^{−(α_p + β_q + α_k) r²} dr
/// A(A,B,C) = 4π (A−1)!!(B−1)!!(C−1)!! / (D+1)!!,  D = A+B+C
/// ```
///
/// `∫ r^m e^{−γr²} dr = ½ γ^{−(m+1)/2} Γ((m+1)/2)`。这是**独立的第二条实现**
/// （不复用求积），用来给半数值路径做 1e-10 量级的对照 —— `tests/i05_ecp.rs` 的判据。
///
/// # Errors
///
/// 球谐变换缺失（`l ≥ 3`）→ [`EngineError::Unsupported`]。
pub fn one_center_matrix(
    basis: &InternalBasis,
    operator: &EcpOperator,
) -> Result<Matrix, EngineError> {
    let n_spherical = basis.n_functions();
    let mut matrix = Matrix::zeros(n_spherical, n_spherical);

    for shell_a in basis.shells() {
        if shell_a.atom_index != operator.atom_index {
            continue;
        }
        let transform_a = spherical_transform(shell_a)?;
        let components_a = qwb_basis::cartesian_components(shell_a.l);
        for shell_b in basis.shells() {
            if shell_b.atom_index != operator.atom_index {
                continue;
            }
            let transform_b = spherical_transform(shell_b)?;
            let components_b = qwb_basis::cartesian_components(shell_b.l);

            // 该壳层对上的算符通道：局域始终 + 角动量匹配的投影通道。
            let mut channels: Vec<(f64, &[EcpTerm])> = vec![(1.0, &operator.local)];
            for channel in &operator.projectors {
                if channel.l == shell_a.l && channel.l == shell_b.l {
                    channels.push((1.0, &channel.terms));
                    channels.push((-1.0, &operator.local));
                }
            }

            for (component_a, pa) in components_a.iter().enumerate() {
                for (component_b, pb) in components_b.iter().enumerate() {
                    let sums = [pa.0 + pb.0, pa.1 + pb.1, pa.2 + pb.2];
                    let Some(angular) = angular_integral(sums) else {
                        continue;
                    };
                    let degree = i32::try_from(sums[0] + sums[1] + sums[2]).expect("small");
                    let mut element = 0.0;
                    for (sign, terms) in &channels {
                        for term in terms.iter() {
                            let mut sum = 0.0;
                            for (alpha, coefficient_a) in
                                shell_a.exponents.iter().zip(&shell_a.coefficients)
                            {
                                let norm_a = qwb_integrals::one_electron::primitive_norm(
                                    (pa.0, pa.1, pa.2),
                                    *alpha,
                                );
                                for (beta, coefficient_b) in
                                    shell_b.exponents.iter().zip(&shell_b.coefficients)
                                {
                                    let norm_b = qwb_integrals::one_electron::primitive_norm(
                                        (pb.0, pb.1, pb.2),
                                        *beta,
                                    );
                                    let gamma = alpha + beta + term.exponent;
                                    let radial = radial_integral(degree + term.power + 2, gamma)?;
                                    sum += coefficient_a * coefficient_b * norm_a * norm_b * radial;
                                }
                            }
                            element += sign * term.coefficient * angular * sum;
                        }
                    }
                    for sa in 0..transform_a.n_spherical {
                        let ta = transform_a.get(component_a, sa);
                        if ta == 0.0 {
                            continue;
                        }
                        for sb in 0..transform_b.n_spherical {
                            let tb = transform_b.get(component_b, sb);
                            if tb == 0.0 {
                                continue;
                            }
                            matrix[(shell_a.first_function + sa, shell_b.first_function + sb)] +=
                                ta * tb * element;
                        }
                    }
                }
            }
        }
    }
    Ok(matrix)
}

fn spherical_transform(
    shell: &qwb_basis::InternalShell,
) -> Result<qwb_basis::spherical::Transform, EngineError> {
    qwb_basis::spherical::spherical_transform(shell.l, shell.angular).ok_or_else(|| {
        EngineError::Unsupported(format!(
            "ECP one-centre integrals: spherical transformation for l={} is not implemented",
            shell.l
        ))
    })
}

/// `A(A,B,C)`：单位球面上 `x̂^A ŷ^B ẑ^C` 的积分；任一指数为奇数则为 0。
fn angular_integral(sums: [u32; 3]) -> Option<f64> {
    if sums.iter().any(|value| value % 2 == 1) {
        return None;
    }
    let degree = sums[0] + sums[1] + sums[2];
    let numerator = double_factorial(sums[0].saturating_sub(1))
        * double_factorial(sums[1].saturating_sub(1))
        * double_factorial(sums[2].saturating_sub(1));
    let denominator = double_factorial(degree + 1);
    Some(4.0 * std::f64::consts::PI * numerator / denominator)
}

/// `n!!`，约定 `(−1)!! = 0!! = 1`。
fn double_factorial(n: u32) -> f64 {
    let mut value = 1.0;
    let mut k = n;
    while k > 1 {
        value *= f64::from(k);
        k -= 2;
    }
    value
}

/// `∫_0^∞ r^m e^{−γ r²} dr = ½ γ^{−(m+1)/2} Γ((m+1)/2)`。
fn radial_integral(m: i32, gamma: f64) -> Result<f64, EngineError> {
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(EngineError::Internal(format!(
            "ecp radial integral: gamma = {gamma} must be positive"
        )));
    }
    if m < -1 {
        return Err(EngineError::Internal(format!(
            "ecp radial integral: r^{m} is not integrable at the origin"
        )));
    }
    let half = 0.5 * f64::from(m + 1);
    Ok(0.5 * gamma.powf(-half) * gamma_half(half))
}

/// `Γ(x)`，`x` 是 `1/2` 的整数倍：`Γ(1/2)=√π`、`Γ(1)=1`、`Γ(3/2)=√π/2`，
/// 其余用 `Γ(x) = (x−1)Γ(x−1)` 递推。
fn gamma_half(x: f64) -> f64 {
    let mut value = x;
    let mut product = 1.0;
    // 往上推到 [0.5, 1.5]：Γ(x) = (x−1)(x−2)…Γ(剩)
    while value > 1.5 + 1.0e-12 {
        value -= 1.0;
        product *= value.max(1.0e-300);
    }
    // 往下推：Γ(x) = Γ(x+1)/x
    while value < 0.5 - 1.0e-12 {
        product /= value.max(1.0e-300);
        value += 1.0;
    }
    let base = if (value - 0.5).abs() < 1.0e-12 {
        std::f64::consts::PI.sqrt()
    } else if (value - 1.0).abs() < 1.0e-12 {
        1.0
    } else {
        0.5 * std::f64::consts::PI.sqrt()
    };
    product * base
}
