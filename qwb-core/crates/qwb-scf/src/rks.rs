//! S-02：闭壳层 **RKS SCF**（LDA 泛函）。
//!
//! # 能量表达
//!
//! ```text
//! E = Tr(P H_core) + ½ Tr(P G) + E_xc + E_nuc
//! G = J（纯库仑；交换关联全部在 E_xc/v_xc 里）,  E_xc = Σ_g w_g ρ_g ε(ρ_g)
//! v_xc[i,j] = Σ_g w_g φ_i φ_j (ε + ρ dε/dρ)
//! ```
//!
//! # 办法
//!
//! 1. `X = S^{-1/2}`（对称特征分解，I-00），把广义本征问题变标准形式；
//! 2. 初猜：对角化 `H_core`；
//! 3. 迭代：密度 → 网格上的 ρ → XC 能量与 `v_xc` → `F` → 对角化 → 新 P；
//!    直到 `|ΔE| < threshold`；
//! 4. 每步都检查 `ΔP`，两者都达标才算收敛。
//!
//! # 第一阶段的范围（写清楚，免得误以为支持）
//!
//! - **只支持 LDA**（`requires.sigma == false`）。需要 σ 的 GGA 返回 `Unsupported`
//!   —— 网格上还没有梯度求值。
//! - **只支持 l <= 1 的 AO 求值**（s、p 直角坐标）；l >= 2 需要球谐变换后的 AO 求值，
//!   而积分层已经能算 l=2（R-06 已接），但网格上还缺球谐 AO 求值，故此处显式拒绝。
//! - 无 DIIS（S-03）、无 UKS（S-04）、无 checkpoint（S-05）。

#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;

use qwb_basis::{InternalBasis, cartesian_components};
use qwb_engine_api::{EngineError, Molecule};
use qwb_grid::{Grid, GridPoint};
use qwb_integrals::one_electron::primitive_norm;
use qwb_integrals::two_electron::{TwoElectronIntegrals, two_electron};
use qwb_linalg::{LinalgError, Matrix, Vector, matmul, symmetric_eigen};
use qwb_registry::elements::atomic_number;
use qwb_xc::{DensityBatch, FunctionalTerm};

use crate::diis::Diis;
use crate::xc_device::{
    XcGridAccelerator, XcGridBackendHandle, XcGridDeviceOutcome, XcGridDeviceReport, XcGridPath,
    XcGridRequest,
};
use crate::{FockMatrices, build_fock};

/// 由**当前密度**给出的一电子算子 —— SCF 内的"外部自洽场"通道。
///
/// 目前唯一的用户是 COSMO/C-PCM 的溶剂反应场（SV-01）：表面电荷由溶质的静电势决定、
/// 反应场再回代进 Fock，所以算子必须每步重算。契约只有两件事：
///
/// * [`ExternalField::operator_and_energy`] 返回**加进 `H_core` 的算子** `V` 与一个能量修正
///   `ΔE`；
/// * SCF 报告的电子能是 `Tr(P H_eff) + ½Tr(P G) + E_xc + Tr(P V) + ΔE`（`H_eff = H + V`），
///   其中 `½Tr(P G)` 那一项不变。因此实现方要保证
///
///   ```text
///   E_总 = E_真空 + Tr(P V) + ΔE
///   ```
///
///   与它自己的物理能量定义一致 —— 本仓库的 COSMO 实现里 `½ΣV_K q_K` 是溶剂化自由能，
///   所以 `ΔE = ½ΣV_K q_K − Tr(P V)`（见 `qwb-engine::cosmo_scf`）。
pub trait ExternalField: Send + Sync {
    /// 由密度构建算子与能量修正（单位：Hartree；矩阵尺寸 `n_ao × n_ao`）。
    ///
    /// # Errors
    ///
    /// 尺寸不符、求解失败等 → [`EngineError`]（会终止 SCF，不做静默降级）。
    fn operator_and_energy(&self, density: &Matrix) -> Result<(Matrix, f64), EngineError>;

    /// 供日志与错误消息用的名字。
    fn name(&self) -> &'static str;
}

/// [`ExternalField`] 的句柄（`RksOptions` 需要 `Clone + PartialEq + Debug`）。
///
/// `PartialEq` 比的是句柄本身（`Arc` 指针），不是场的内容 —— 场没有可比较的语义。
#[derive(Clone)]
pub struct ExternalFieldHandle(std::sync::Arc<dyn ExternalField>);

impl ExternalFieldHandle {
    /// 包一个场实现。
    #[must_use]
    pub fn new(field: std::sync::Arc<dyn ExternalField>) -> Self {
        Self(field)
    }

    /// 取出内部实现。
    #[must_use]
    pub fn field(&self) -> &dyn ExternalField {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for ExternalFieldHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ExternalField({})", self.0.name())
    }
}

impl PartialEq for ExternalFieldHandle {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

/// SCF 控制参数。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RksOptions {
    /// 最大迭代步数。
    pub max_iter: u32,
    /// 能量收敛阈值（Hartree）。
    pub energy_threshold: f64,
    /// 密度矩阵收敛阈值。
    pub density_threshold: f64,
    /// HF 交换的混入比例 `α`（hybrid 泛函；纯泛函为 0）。
    ///
    /// 数据侧写在 functional 的 `components` 里：`- id: HfExchange, weight: 0.25`
    /// （`rks_from_data` 会把它读进这个字段，而不是当成能量密度项）。
    /// 进入 Fock 的是 `α·(−½K)`，进入能量的是 `α·(−¼Tr(PK))`。
    pub hf_exchange: f64,
    /// 是否用 Pulay DIIS 加速（默认开）。
    pub use_diis: bool,
    /// DIIS 子空间最多保留多少组向量。
    pub diis_max_vectors: usize,
    /// 初猜里 `H_core` 非对角块的缩放系数（默认 1.0 = 纯 core guess）。
    ///
    /// N2/STO-3G 这类近简并体系上，纯 core guess 会稳定收敛到**激发态**（N2 实测高
    /// 0.684 Ha）。配合 [`rks_scf_multistart`] 用几个不同的缩放因子各跑一次、取最低解。
    pub initial_guess_scale: f64,
    /// 从检查点续算（S-05）。
    pub resume_from: Option<crate::checkpoint::ScfCheckpoint>,
    /// 检查点文件路径（`None` = 不写检查点）。
    pub checkpoint_path: Option<std::path::PathBuf>,
    /// 每多少步写一次检查点（0 = 只在结束时写）。
    pub checkpoint_every: u32,
    /// I-05：每个原子对**电子**的核电荷。`None` = 用元素表里的 `Z`（全电子）；
    /// 有 ECP 的原子传 `z_eff = Z − n_core`。
    pub nuclear_charges: Option<Vec<f64>>,
    /// I-05：ECP 的电子-核有效势矩阵元（`n_ao × n_ao`），会被加进 `H_core`。
    /// `None` = 没有 ECP。用 [`crate::ecp::matrix_elements`] 算。
    pub ecp_potential: Option<Matrix>,
    /// 静态能级移动 `λ`：把虚轨道抬起来，抑制电荷振荡（`0` = 不做）。
    ///
    /// 作用在 Fock 上：`F ← F + λ (S − ½ S P S)`（闭壳层下这就是 `λ S P_virtual S`）。
    /// 近简并 / 强电荷转移体系（LANL2DZ 的 ZnO 是典型）上纯 DIIS 会一直在几个占据
    /// 组态之间跳，能级移动是标准对策之一。
    pub level_shift: f64,
    /// 前多少步用**费米展宽**（分数占据）跑 SCF；`0` = 不做。
    ///
    /// 用途：给"初始猜测很差"的体系一个平滑的入轨路径（退火）。典型是
    /// LANL2DZ 的 ZnO —— 闭壳层受限的 O 原子本身就是伪解（O 的 ³P 有两个单占 p 轨道），
    /// core guess 于是从一个伪组态出发、在几个组态之间跳。展宽让占据连续变化，
    /// 走完 `smearing_steps` 步后切回整数占据。
    pub smearing_steps: u32,
    /// 展宽温度（Hartree）。`0.2`–`0.5` 通常够；只在 `smearing_steps` 内生效。
    pub smearing_temperature: f64,
    /// 密度的线性混合系数（`0` = 直接替换；常用 `0.2`–`0.4`）。
    ///
    /// 近简并 / 电荷转移体系（含 d¹⁰ 的重元素、ECP 体系）上纯 DIIS 会震荡，
    /// 阻尼是标准对策。默认 0（不改变既有行为）。
    pub damping: f64,
    /// 外部自洽场（目前只有 COSMO/C-PCM 的溶剂反应场用）。
    ///
    /// `None`（默认）= 真空 SCF，逐位不变。
    pub external_field: Option<ExternalFieldHandle>,
    /// EX-03：网格 XC 收缩走 CPU 还是设备。默认 [`XcGridPath::Cpu`]。
    ///
    /// 默认值就是产品路径（`tests/xc_grid_bit_pattern.rs` 钉住的那条），所以不显式选
    /// `Gpu` 时行为一个字没变。选 [`XcGridPath::Gpu`] 却没有注入
    /// [`Self::xc_backend`]（或后端报告不可用、显存不够、这台机器没有 GPU）时，
    /// **退回 CPU 并继续**，原因写进 [`RksResult::xc_fallback_reason`] —— 绝不让没有 GPU
    /// 的机器红，也绝不静默给出少一项的势。
    pub xc_path: XcGridPath,
    /// EX-03：网格 XC 的设备收缩后端（`None` = 没注入）。
    ///
    /// `qwb-scf` **不依赖** `qwb-gpu`：实现由上层在 `gpu-kernel` feature 下注入
    /// （见 [`crate::xc_device`] 的模块文档）。默认 `None`。
    pub xc_backend: Option<XcGridBackendHandle>,
}

impl Default for RksOptions {
    fn default() -> Self {
        Self {
            max_iter: 128,
            energy_threshold: 1.0e-8,
            density_threshold: 1.0e-6,
            hf_exchange: 0.0,
            use_diis: true,
            diis_max_vectors: 8,
            initial_guess_scale: 1.0,
            resume_from: None,
            checkpoint_path: None,
            checkpoint_every: 0,
            nuclear_charges: None,
            ecp_potential: None,
            level_shift: 0.0,
            smearing_steps: 0,
            smearing_temperature: 0.0,
            damping: 0.0,
            external_field: None,
            xc_path: XcGridPath::Cpu,
            xc_backend: None,
        }
    }
}

/// SCF 结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RksResult {
    /// 总能量，Hartree。
    pub energy_hartree: f64,
    /// 是否收敛。
    pub scf_converged: bool,
    /// 实际迭代步数。
    pub iterations: u32,
    /// 电子能量（不含核排斥）。
    pub electronic_hartree: f64,
    /// 交换相关能。
    pub exchange_correlation_hartree: f64,
    /// 核排斥能。
    pub nuclear_repulsion_hartree: f64,
    /// 密度矩阵。
    pub density: Matrix,
    /// 轨道能（升序）。
    pub mo_energies_hartree: Vector,
    /// **MO 系数矩阵**（列 = MO，AO 为基；`C` 满足 `Cᵀ S C = 1`）。
    ///
    /// 之前 `RksResult` 只带密度矩阵，于是 `qwb-engine` 拿密度去填 K2 的
    /// `JobResult.mo_coeffs` —— 字段名与内容不符（P-10 的轨道等值面也因此做不了）。
    pub coefficients: Matrix,
    /// 最后一次能量变化。
    pub last_energy_change: f64,
    /// 网格点数（诊断用）。
    pub grid_points: usize,
    /// 终态检查点（可直接存盘续算）。
    pub checkpoint: crate::checkpoint::ScfCheckpoint,
    /// EX-03：网格 XC 实际走的路径（默认 [`XcGridPath::Cpu`]）。
    pub xc_path: XcGridPath,
    /// EX-03：设备路径退回 CPU 的原因（`None` = 没发生退回）。
    pub xc_fallback_reason: Option<String>,
    /// EX-03：设备路径的累计分项记账（CPU 路径为 `None`）。
    ///
    /// 四项时间（上传 / 手写核 / gemm / 下载）是互斥的阶段计时，逐次累加；
    /// `uploads` 是其中真的重传了 `Φ`/`∇Φ` 的次数 —— 网格与基组不变时它应当只有 1。
    pub xc_device_report: Option<XcGridDeviceReport>,
}

pub(crate) fn linalg(err: LinalgError) -> EngineError {
    EngineError::Internal(format!("linear algebra: {err}"))
}

/// 求 `S^{-1/2}`（对称正交化）。
pub(crate) fn inverse_sqrt_symmetric(s: &Matrix) -> Result<Matrix, EngineError> {
    let (values, vectors) = symmetric_eigen(s).map_err(linalg)?;
    let n = s.nrows();
    let largest = (0..n).map(|i| values.get(i)).fold(0.0_f64, f64::max);
    if !(largest.is_finite() && largest > 0.0) {
        return Err(EngineError::Internal(format!(
            "overlap matrix is degenerate (largest eigenvalue {largest})"
        )));
    }
    // 秩截断的规范正交化：近线性相关的基组（6-311G(d,p) 这类共享指数的三重分裂）
    // 会出现 ~1e-16 的本征值。标准做法是**丢弃**它们（对应的方向不进正交化基），
    // 而不是断言矩阵正定——断言只会让这类基组直接不可用。
    let threshold = largest * 1.0e-10;
    let mut dropped = 0usize;
    for i in 0..n {
        let lambda = values.get(i);
        if !lambda.is_finite() || lambda < -threshold {
            return Err(EngineError::Internal(format!(
                "overlap matrix is not positive semidefinite (eigenvalue {lambda})"
            )));
        }
        if lambda < threshold {
            dropped += 1;
        }
    }
    if dropped > 0 {
        // 每个进程只提示一次：SCF 每次迭代都会调这个函数，逐次打印会把真正的输出淹掉。
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "warning: overlap matrix has {dropped} near-zero eigenvalue(s) below {threshold:.2e}; \
using rank-truncated orthogonalization (basis is nearly linearly dependent)"
            );
        });
    }
    // X = U Λ^{-1/2} Uᵀ，被截断的方向取 0
    let mut scaled = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let lambda = values.get(j);
            scaled[(i, j)] = if lambda < threshold {
                0.0
            } else {
                vectors[(i, j)] / lambda.sqrt()
            };
        }
    }
    let vectors_t = vectors.transpose();
    matmul(&scaled, &vectors_t).map_err(linalg)
}

/// 核排斥能 `Σ_{A>B} Z_A Z_B / R_AB`（`charges` 是每个原子对电子的核电荷；
/// 有 ECP 的原子传 `z_eff`）。
pub(crate) fn nuclear_repulsion(charges: &[f64], molecule: &Molecule) -> Result<f64, EngineError> {
    let mut centers: Vec<([f64; 3], f64)> = Vec::with_capacity(molecule.n_atoms());
    for (atom, charge) in molecule.atoms.iter().zip(charges) {
        centers.push((atom.position_bohr(), *charge));
    }
    let mut energy = 0.0;
    for i in 0..centers.len() {
        for j in (i + 1)..centers.len() {
            let distance = (0..3)
                .map(|axis| (centers[i].0[axis] - centers[j].0[axis]).powi(2))
                .sum::<f64>()
                .sqrt();
            energy += centers[i].1 * centers[j].1 / distance;
        }
    }
    Ok(energy)
}

/// 全电子时的核电荷表（`Z` 来自元素表）。
///
/// # Errors
///
/// 元素表里没有的符号 → [`EngineError::Internal`]（IC-05：ECP 的 `z_eff` 由调用方覆盖）。
pub(crate) fn default_charges(molecule: &Molecule) -> Result<Vec<f64>, EngineError> {
    let mut charges = Vec::with_capacity(molecule.n_atoms());
    for atom in &molecule.atoms {
        if atom.ghost {
            // 幽灵原子：只带基函数，核电荷与电子数为 0（counterpoise/BSSE 用）。
            charges.push(0.0);
            continue;
        }
        let z = atomic_number(&atom.symbol)
            .ok_or_else(|| EngineError::Internal(format!("unknown element {}", atom.symbol)))?;
        charges.push(f64::from(z));
    }
    Ok(charges)
}

/// 一个 AO 在给定点处的值（只支持 l <= 1）。pub 是为了让测试能直接查有限性。
#[allow(clippy::needless_range_loop)] // 分量下标与 components 一一对应，比迭代器更清楚
pub fn ao_value(basis: &InternalBasis, point: [f64; 3]) -> Result<Vec<f64>, EngineError> {
    let n = basis.n_functions();
    let mut values = vec![0.0; n];
    // 热路径：网格上每个点都要调用，所以这里**不做任何堆分配**（`MAX_CARTESIAN` 够 l ≤ 3）。
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let mut cartesian = [0.0_f64; MAX_CARTESIAN];
        for (component, powers) in components.iter().enumerate() {
            let mut value = 0.0;
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm(*powers, *alpha);
                let d = [
                    point[0] - shell.center_bohr[0],
                    point[1] - shell.center_bohr[1],
                    point[2] - shell.center_bohr[2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let polynomial = d[0].powi(i32::try_from(powers.0).expect("small"))
                    * d[1].powi(i32::try_from(powers.1).expect("small"))
                    * d[2].powi(i32::try_from(powers.2).expect("small"));
                value += coefficient * norm * polynomial * (-alpha * r2).exp();
            }
            cartesian[component] = value; // 显式下标：与 components 一一对应
        }
        write_transformed(
            &mut values,
            shell.first_function,
            &cartesian[..components.len()],
            shell.l,
            shell.angular,
        )?;
    }
    Ok(values)
}

/// 把直角分量写进 AO 数组：`Cartesian` 直接抄，`Spherical` 乘球谐变换矩阵。
///
/// `l ≤ 1` 时直角与球谐是同一组函数（`T` 是单位阵），所以这个函数对 STO-3G 完全不改行为。
#[allow(clippy::needless_range_loop)] // 矩阵收缩：两个下标各有含义，range 最直观
fn write_transformed(
    values: &mut [f64],
    first: usize,
    cartesian: &[f64],
    l: u32,
    angular: qwb_registry::Angular,
) -> Result<(), EngineError> {
    let count = cartesian.len();
    match qwb_basis::spherical::spherical_transform(l, angular) {
        None => {
            for (component, value) in cartesian.iter().enumerate() {
                values[first + component] = *value;
            }
            Ok(())
        }
        Some(transform) => {
            if transform.n_spherical > count {
                return Err(EngineError::Internal(format!(
                    "球谐变换维度 {} 超过直角分量数 {count}",
                    transform.n_spherical
                )));
            }
            for spherical in 0..transform.n_spherical {
                let mut sum = 0.0;
                for component in 0..count {
                    sum += transform.get(component, spherical) * cartesian[component];
                }
                values[first + spherical] = sum;
            }
            Ok(())
        }
    }
}

/// AO 值 + 梯度（GGA 的前置）。
///
/// 对笛卡尔高斯 `x^a y^b z^c e^{−α|r−A|²}`（`u = r − A`）：
///
/// ```text
/// ∂/∂x = [ a·u_x^{a−1} − 2α·u_x^{a+1} ] · u_y^b · u_z^c · e^{−α|u|²}
/// ```
///
/// 也就是说梯度的每个分量仍是「同一族单项式」，只是 x 方向的幂次减一或加二。
///
/// # Errors
///
/// 与 [`ao_value`] 相同：遇到 `l ≥ 2`。
pub fn ao_value_and_gradient(
    basis: &InternalBasis,
    point: [f64; 3],
) -> Result<(Vec<f64>, Vec<[f64; 3]>), EngineError> {
    let n = basis.n_functions();
    let mut values = vec![0.0; n];
    let mut gradients = vec![[0.0_f64; 3]; n];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let d = [
            point[0] - shell.center_bohr[0],
            point[1] - shell.center_bohr[1],
            point[2] - shell.center_bohr[2],
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let mut cartesian_values = [0.0_f64; MAX_CARTESIAN];
        let mut cartesian_gradients = [[0.0_f64; 3]; MAX_CARTESIAN];
        for (component, powers) in components.iter().enumerate() {
            let powers = [powers.0, powers.1, powers.2];
            let mut value = 0.0;
            let mut gradient = [0.0_f64; 3];
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm((powers[0], powers[1], powers[2]), *alpha);
                let weight = coefficient * norm * (-alpha * r2).exp();
                let monomial = |exponents: [u32; 3]| {
                    d[0].powi(i32::try_from(exponents[0]).expect("small"))
                        * d[1].powi(i32::try_from(exponents[1]).expect("small"))
                        * d[2].powi(i32::try_from(exponents[2]).expect("small"))
                };
                value += weight * monomial(powers);
                for axis in 0..3 {
                    // a·u^(a−1) 项（a ≥ 1 才有）
                    let mut derivative = 0.0;
                    if powers[axis] >= 1 {
                        let mut lowered = powers;
                        lowered[axis] -= 1;
                        derivative += f64::from(powers[axis]) * monomial(lowered);
                    }
                    // −2α·u^(a+1) 项
                    let mut raised = powers;
                    raised[axis] += 1;
                    derivative -= 2.0 * alpha * monomial(raised);
                    gradient[axis] += weight * derivative;
                }
            }
            cartesian_values[component] = value;
            cartesian_gradients[component] = gradient;
        }
        write_transformed(
            &mut values,
            shell.first_function,
            &cartesian_values[..components.len()],
            shell.l,
            shell.angular,
        )?;
        // 梯度是对**函数**逐分量线性组合，所以同一个变换矩阵作用在三个分量上。
        let range = if shell.angular == qwb_registry::Angular::Spherical {
            spherical_count(shell.l)
        } else {
            components.len()
        };
        let mut slice = [0.0_f64; MAX_CARTESIAN];
        for axis in 0..3 {
            for component in 0..components.len() {
                slice[component] = cartesian_gradients[component][axis];
            }
            let mut flat = [0.0_f64; MAX_CARTESIAN];
            write_transformed(
                &mut flat,
                0,
                &slice[..components.len()],
                shell.l,
                shell.angular,
            )?;
            for slot in 0..range {
                gradients[shell.first_function + slot][axis] = flat[slot];
            }
        }
    }
    Ok((values, gradients))
}

/// 球谐函数个数：`l ≤ 1` 与直角相同，`l = 2` 是 5。
const MAX_CARTESIAN: usize = 10;

/// 球谐函数个数：l ≤ 1 与直角相同，l = 2 是 5。
fn spherical_count(l: u32) -> usize {
    (2 * l as usize + 1).min(6)
}

/// 每个 AO 在 `point` 处的**拉普拉斯量** `∇²φ`。
///
/// 对笛卡尔分量 `φ = N·u_x^a u_y^b u_z^c e^{−αu²}`（`u = r − A`）：
///
/// ```text
/// ∂²/∂u_x² [u_x^a e^{−αu²}] = a(a−1)u_x^{a−2} − 2α(2a+1)u_x^a + 4α²u_x^{a+2}
/// ```
///
/// 三个轴各按这个式子取一项（只改本轴的幂次），求和即 `∇²φ`。球谐壳层同样按
/// `spherical_transform` 的列做线性组合——与 `ao_value_and_gradient` 一致。
///
/// # Errors
///
/// 形状或球谐变换维度不一致。
#[allow(clippy::needless_range_loop)]
pub fn ao_laplacian(basis: &InternalBasis, point: [f64; 3]) -> Result<Vec<f64>, EngineError> {
    let mut values = vec![0.0_f64; basis.n_functions()];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let mut cartesian = [0.0_f64; MAX_CARTESIAN];
        let d = [
            point[0] - shell.center_bohr[0],
            point[1] - shell.center_bohr[1],
            point[2] - shell.center_bohr[2],
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for (component, powers) in components.iter().enumerate() {
            let powers = [powers.0, powers.1, powers.2];
            let monomial = |exponents: [u32; 3]| {
                d[0].powi(i32::try_from(exponents[0]).expect("small"))
                    * d[1].powi(i32::try_from(exponents[1]).expect("small"))
                    * d[2].powi(i32::try_from(exponents[2]).expect("small"))
            };
            let mut value = 0.0;
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm((powers[0], powers[1], powers[2]), *alpha);
                let weight = coefficient * norm * (-alpha * r2).exp();
                let mut laplacian = 0.0;
                for axis in 0..3 {
                    // a(a−1)u^{a−2}
                    if powers[axis] >= 2 {
                        let mut lowered = powers;
                        lowered[axis] -= 2;
                        laplacian += f64::from(powers[axis])
                            * f64::from(powers[axis] - 1)
                            * monomial(lowered);
                    }
                    // −2α(2a+1)u^a
                    laplacian -=
                        2.0 * alpha * (2.0 * f64::from(powers[axis]) + 1.0) * monomial(powers);
                    // +4α²u^{a+2}
                    let mut raised = powers;
                    raised[axis] += 2;
                    laplacian += 4.0 * alpha * alpha * monomial(raised);
                }
                value += weight * laplacian;
            }
            cartesian[component] = value;
        }
        write_transformed(
            &mut values,
            shell.first_function,
            &cartesian[..components.len()],
            shell.l,
            shell.angular,
        )?;
    }
    Ok(values)
}

/// 在网格上求密度与 XC 能量/v_xc。
///
/// # Errors
///
/// 网格上的 AO 数据（`φ`、`∇φ`、`∇²φ`），与密度无关，**可以在 SCF 迭代之间复用**。
///
/// SCF 每步都要在整张网格上重算 AO 三元组，这是 XC 求值的主要开销（苯/STO-3G 实测约 5 s/步，
/// 400 步就是半小时）。网格与基组在 SCF 期间不变，所以这里把它算一次、后面只做密度缩并。
pub struct GridAoCached {
    values: Vec<Vec<f64>>,
    gradients: Vec<Vec<[f64; 3]>>,
    laplacians: Vec<Vec<f64>>,
}

impl GridAoCached {
    /// 按需求（σ / ∇²ρ / τ）预取网格上的 AO 数据。
    ///
    /// τ 与 σ 都只要一阶导数，所以 `needs_tau` 与 `needs_sigma` 共用同一份
    /// `∇φ` 缓存（FX-01）。
    ///
    /// # Errors
    /// AO 求值失败（例如基函数角动量超出现有实现）。
    pub fn build(
        basis: &InternalBasis,
        grid: &Grid,
        needs_sigma: bool,
        needs_laplacian: bool,
        needs_tau: bool,
    ) -> Result<Self, EngineError> {
        let mut values = Vec::with_capacity(grid.len());
        let mut gradients = Vec::with_capacity(grid.len());
        let mut laplacians = Vec::with_capacity(grid.len());
        for GridPoint { position_bohr, .. } in &grid.points {
            let (value, gradient) = if needs_sigma || needs_laplacian || needs_tau {
                ao_value_and_gradient(basis, *position_bohr)?
            } else {
                (ao_value(basis, *position_bohr)?, Vec::new())
            };
            let laplacian = if needs_laplacian {
                ao_laplacian(basis, *position_bohr)?
            } else {
                Vec::new()
            };
            values.push(value);
            gradients.push(gradient);
            laplacians.push(laplacian);
        }
        Ok(Self {
            values,
            gradients,
            laplacians,
        })
    }

    /// 第 `index` 个网格点上的 AO 值 `φ_μ`。
    ///
    /// # Panics
    ///
    /// `index` 越界。
    #[must_use]
    pub fn values_at(&self, index: usize) -> &[f64] {
        &self.values[index]
    }

    /// 第 `index` 个网格点上的 AO 梯度 `∇φ_μ`（未预取时为空切片）。
    ///
    /// # Panics
    ///
    /// `index` 越界。
    #[must_use]
    pub fn gradients_at(&self, index: usize) -> &[[f64; 3]] {
        &self.gradients[index]
    }
}

/// AO 求值遇到 l>=2，或密度出现非有限值。
pub fn evaluate_xc(
    basis: &InternalBasis,
    grid: &Grid,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
) -> Result<(f64, Matrix), EngineError> {
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);
    let cache = GridAoCached::build(basis, grid, needs_sigma, needs_laplacian, needs_tau)?;
    evaluate_xc_cached(basis, grid, &cache, density, terms)
}

/// 与 [`evaluate_xc`] 相同，但 AO 数据由调用方预先算好（SCF 里每个迭代步复用）。
///
/// # Errors
/// 密度或密度梯度出现非有限值。
pub fn evaluate_xc_cached(
    basis: &InternalBasis,
    grid: &Grid,
    cache: &GridAoCached,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
) -> Result<(f64, Matrix), EngineError> {
    let n = basis.n_functions();
    // Requirements 是 non_exhaustive，只能逐字段问，不能构造字面量
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);

    let mut rho_values = Vec::with_capacity(grid.len());
    let mut sigma_values = Vec::with_capacity(grid.len());
    let mut laplacian_values = Vec::with_capacity(grid.len());
    let mut tau_values = Vec::with_capacity(grid.len());
    let mut grad_rho: Vec<[f64; 3]> = Vec::with_capacity(grid.len());

    // 密度的**行主序**扁平副本（一次性 O(n²)）。
    //
    // 为什么：`Matrix` 是列主序，而这里的累加顺序是「i 外层、j 内层」。直接写
    // `density[(i, j)]` 时，内层 j 的步长是 n 个 double —— 每个元素命中一条新的
    // cache line，却只用到其中一个 double。摊平之后内层是连续的，**求和顺序一个没动**
    // （仍是 i 外层、j 内层的同一串加法），所以 `xc_grid_bit_pattern` 的位级锚点不变。
    // 这是 EX-03 量出来的瓶颈（格点 XC 的收缩段占每步 85–96%）能拿到的、不破坏
    // 「CPU 位级回归」约束的那部分收益。
    let mut flat_density = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            flat_density[i * n + j] = density[(i, j)];
        }
    }

    for (index, GridPoint { .. }) in grid.points.iter().enumerate() {
        // AO 三元组来自预取缓存（见 GridAoCached）：与密度无关，SCF 各步复用。
        //
        // **借用而不是 clone**：这里曾经每个网格点 clone 三个 `Vec`（φ、∇φ、∇²φ），随后又把
        // 它们 push 进 `ao_values` 一类的 Vec-of-Vec 里 —— 相当于每个点拷两遍、每步付 2·n_points·5n
        // 个双精度的分配与拷贝（苯/def2-SVP、grid-normal 约 70 MB/步）。去掉之后算术一行没动，
        // 所以 `xc_grid_bit_pattern` 那条位级锚点仍然逐位相同。
        let values = &cache.values[index];
        let gradients = &cache.gradients[index];
        let ao_laplacian_here = &cache.laplacians[index];
        // ∇²ρ = Σ_μν P_μν (∇²φ_μ φ_ν + 2 ∇φ_μ·∇φ_ν + φ_μ ∇²φ_ν)
        let laplacian_here = if needs_laplacian {
            let mut total = 0.0;
            for i in 0..n {
                let g_i = gradients[i];
                let v_i = values[i];
                let l_i = ao_laplacian_here[i];
                let base = i * n;
                for j in 0..n {
                    let p = flat_density[base + j];
                    if p == 0.0 {
                        continue;
                    }
                    let g_j = gradients[j];
                    let dot = g_i[0] * g_j[0] + g_i[1] * g_j[1] + g_i[2] * g_j[2];
                    total += p * (l_i * values[j] + 2.0 * dot + v_i * ao_laplacian_here[j]);
                }
            }
            if !total.is_finite() {
                // 诊断：把参与求和的量抽样吐出来，定位是 P、φ、∇φ 还是 ∇²φ 出了问题
                let mut max_p = 0.0_f64;
                let mut max_value = 0.0_f64;
                let mut max_gradient = 0.0_f64;
                let mut max_laplacian = 0.0_f64;
                for i in 0..n {
                    max_value = max_value.max(values[i].abs());
                    max_laplacian = max_laplacian.max(ao_laplacian_here[i].abs());
                    for value in &gradients[i] {
                        max_gradient = max_gradient.max(value.abs());
                    }
                    for j in 0..n {
                        max_p = max_p.max(density[(i, j)].abs());
                    }
                }
                return Err(EngineError::Internal(format!(
                    "density laplacian is not finite ({total}) at a grid point: \
                     max|P| {max_p:.3e}, max|φ| {max_value:.3e}, max|∇φ| {max_gradient:.3e}, \
                     max|∇²φ| {max_laplacian:.3e}"
                )));
            }
            total
        } else {
            0.0
        };
        laplacian_values.push(laplacian_here);
        let mut rho = 0.0;
        for i in 0..n {
            let v_i = values[i];
            let base = i * n;
            for j in 0..n {
                rho += v_i * flat_density[base + j] * values[j];
            }
        }
        // 防御：网格坐标或 AO 求值一旦产生非有限值，这里必须报错。
        // 历史教训：`f64::max(0.0)` 会把 NaN 静默变成 0，于是密度"看起来正常"，
        // 而 v_xc 里保留着 NaN —— 上一版就是这样把 NaN 带进 SCF 的。
        if !rho.is_finite() {
            return Err(EngineError::Internal(format!(
                "density is not finite ({rho}) at a grid point; grid coordinates or AO \
                 values are corrupted"
            )));
        }
        // ∇ρ = Σ_μν P_μν (∇φ_μ φ_ν + φ_μ ∇φ_ν)
        let mut gradient_here = [0.0_f64; 3];
        if needs_sigma {
            for i in 0..n {
                let g_i = gradients[i];
                let v_i = values[i];
                let base = i * n;
                for j in 0..n {
                    let p = flat_density[base + j];
                    if p == 0.0 {
                        continue;
                    }
                    let v_j = values[j];
                    let g_j = gradients[j];
                    for axis in 0..3 {
                        gradient_here[axis] += p * (g_i[axis] * v_j + v_i * g_j[axis]);
                    }
                }
            }
            if !gradient_here.iter().all(|value| value.is_finite()) {
                return Err(EngineError::Internal(
                    "density gradient is not finite at a grid point".to_string(),
                ));
            }
        }
        rho_values.push(rho.max(0.0));
        sigma_values.push(
            gradient_here[0] * gradient_here[0]
                + gradient_here[1] * gradient_here[1]
                + gradient_here[2] * gradient_here[2],
        );
        // τ = ½ Σ_μν P_μν ∇φ_μ·∇φ_ν（闭壳层：P 是自旋求和密度，τ 也是求和的）
        let tau_here = if needs_tau {
            let mut total = 0.0;
            for i in 0..n {
                let g_i = gradients[i];
                let base = i * n;
                for j in 0..n {
                    let p = flat_density[base + j];
                    if p == 0.0 {
                        continue;
                    }
                    let g_j = gradients[j];
                    let dot = g_i[0] * g_j[0] + g_i[1] * g_j[1] + g_i[2] * g_j[2];
                    total += p * dot;
                }
            }
            total *= 0.5;
            if !total.is_finite() {
                return Err(EngineError::Internal(format!(
                    "kinetic energy density is not finite ({total}) at a grid point"
                )));
            }
            total
        } else {
            0.0
        };
        tau_values.push(tau_here);
        grad_rho.push(gradient_here);
    }

    let batch = match (needs_laplacian, needs_sigma, needs_tau) {
        (true, _, _) => {
            DensityBatch::with_all(&rho_values, &sigma_values, &laplacian_values, &tau_values)
        }
        (false, _, true) => DensityBatch::with_tau(&rho_values, &sigma_values, &tau_values),
        (false, true, false) => DensityBatch::new(&rho_values, &sigma_values),
        (false, false, false) => DensityBatch::from_rho(&rho_values),
    };
    // 泛函项的四路逐点导数：**与设备通道共用同一段代码**（`term_derivatives`），
    // 所以两条路径的 `ε`、`dε/dρ`、`dε/dσ`、`dε/d∇²ρ`、`dε/dτ` 只可能因为 `ρ`/`σ`
    // 从哪来而不同，不可能因为「谁写的这段循环」而不同。
    let TermDerivatives {
        epsilon,
        d_epsilon,
        d_sigma,
        d_laplacian,
        d_tau,
    } = term_derivatives(&batch, grid.len(), terms);

    let mut energy = 0.0;
    let mut v_xc = Matrix::zeros(n, n);
    for (index, GridPoint { weight, .. }) in grid.points.iter().enumerate() {
        let rho = rho_values[index];
        energy += weight * rho * epsilon[index];
        // LDA 部分：v = ε + ρ dε/dρ；ρ=0 时不乘（0×inf = NaN）。
        let potential = if rho > 0.0 {
            epsilon[index] + rho * d_epsilon[index]
        } else {
            epsilon[index]
        };
        // GGA 部分：∫ φ_μ(−∇·(2·∂f/∂σ·∇ρ))φ_ν 用分部积分转成
        //   + 2·∂f/∂σ·∇ρ·∇(φ_μ φ_ν)，其中 ∂f/∂σ = ρ·dε/dσ。
        // 这样不需要在网格上做数值散度，矩阵元直接可达。
        let sigma_coefficient = if needs_sigma && rho > 0.0 {
            2.0 * rho * d_sigma[index]
        } else {
            0.0
        };
        // ∇²ρ 通道：∫ φ_μ(∇²(ρ·∂ε/∂∇²ρ))φ_ν（一遍分部积分）
        //   = ∫ (ρ·∂ε/∂∇²ρ)·∇²(φ_μ φ_ν)，而
        //   ∇²(φ_μ φ_ν) = ∇²φ_μ φ_ν + 2 ∇φ_μ·∇φ_ν + φ_μ ∇²φ_ν ——三样都已经在手上。
        let laplacian_coefficient = if needs_laplacian && rho > 0.0 {
            rho * d_laplacian[index]
        } else {
            0.0
        };
        // τ 通道：能量密度按**每电子** ε 约定（E = ∫ρ ε dV），所以
        //   ∂E/∂P_μν ⊃ ∫ ρ·(∂ε/∂τ)·(½∇φ_μ·∇φ_ν) dV
        // —— 系数是 **½ρ·dτ**，漏掉 ρ 会让 τ 项的矩阵元整体偏小（FX-01 踩过：
        // 线性 τ 项的能量对不上 (c/2)Tr[PT]，差 9 倍）。
        // 与 σ/∇²ρ 通道不同，∂τ/∂P_μν 是**直接**的，不需要分部积分。
        let tau_coefficient = if needs_tau && rho > 0.0 {
            0.5 * rho * d_tau[index]
        } else {
            0.0
        };
        // 临时诊断：哪个系数先坏
        if !(potential.is_finite()
            && sigma_coefficient.is_finite()
            && laplacian_coefficient.is_finite()
            && tau_coefficient.is_finite())
        {
            return Err(EngineError::Internal(format!(
                "coefficient not finite at grid point {index}: potential {potential:.3e}, \
                 sigma_coeff {sigma_coefficient:.3e}, laplacian_coeff {laplacian_coefficient:.3e}, \
                 rho {rho:.3e}, d_eps {:.3e}, d_sigma {:.3e}, d_lap {:.3e}",
                d_epsilon[index], d_sigma[index], d_laplacian[index]
            )));
        }
        let values = &cache.values[index];
        let gradients = &cache.gradients[index];
        let laplacians = &cache.laplacians[index];
        let gradient_rho = grad_rho[index];
        for i in 0..n {
            if values[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                let mut element = values[i] * values[j] * potential;
                if sigma_coefficient != 0.0 {
                    // ∇ρ · ∇(φ_μ φ_ν)
                    let dot = (0..3)
                        .map(|axis| {
                            gradient_rho[axis]
                                * (gradients[i][axis] * values[j] + values[i] * gradients[j][axis])
                        })
                        .sum::<f64>();
                    element += sigma_coefficient * dot;
                }
                if laplacian_coefficient != 0.0 {
                    let dot = gradients[i]
                        .iter()
                        .zip(gradients[j].iter())
                        .map(|(left, right)| left * right)
                        .sum::<f64>();
                    element += laplacian_coefficient
                        * (laplacians[i] * values[j] + 2.0 * dot + values[i] * laplacians[j]);
                }
                if tau_coefficient != 0.0 {
                    // 与上面 σ 通道共用同一个 ∇φ_μ·∇φ_ν
                    let dot = gradients[i]
                        .iter()
                        .zip(gradients[j].iter())
                        .map(|(left, right)| left * right)
                        .sum::<f64>();
                    element += tau_coefficient * dot;
                }
                v_xc[(i, j)] += weight * element;
            }
        }
    }
    Ok((energy, v_xc))
}

/// 泛函项对 `ρ`、`σ`、`∇²ρ`、`τ` 的四路逐点导数（`ε` 是能量密度）。
struct TermDerivatives {
    epsilon: Vec<f64>,
    d_epsilon: Vec<f64>,
    d_sigma: Vec<f64>,
    d_laplacian: Vec<f64>,
    d_tau: Vec<f64>,
}

/// CPU 与设备两条通道**共用**的泛函求和：逐项算进局部数组、再按点累加。
///
/// 从 `evaluate_xc_cached` 里抽出来（纯搬移，累加序与分配大小一字未动），目的是让设备
/// 通道的 `ε`/`dε/d*` 与产品路径**走同一段代码** —— 两条路径的差别只剩「`ρ`/`σ` 从哪来」。
/// `tests/xc_grid_bit_pattern.rs` 是这次搬移的守卫：位级锚点必须逐位不变。
fn term_derivatives(
    batch: &DensityBatch,
    points: usize,
    terms: &[Box<dyn FunctionalTerm>],
) -> TermDerivatives {
    let mut epsilon = vec![0.0; points];
    let mut d_epsilon = vec![0.0; points];
    let mut d_sigma = vec![0.0; points];
    let mut d_laplacian = vec![0.0; points];
    let mut d_tau = vec![0.0; points];

    for term in terms {
        let mut local_epsilon = vec![0.0; points];
        let mut local_d_rho = vec![0.0; points];
        let mut local_d_sigma = vec![0.0; points];
        let mut local_d_laplacian = vec![0.0; points];
        let mut local_d_tau = vec![0.0; points];
        term.energy_density(batch, &mut local_epsilon);
        term.potential_derivatives_full(
            batch,
            &mut local_d_rho,
            &mut local_d_sigma,
            &mut local_d_laplacian,
            &mut local_d_tau,
        );
        for index in 0..points {
            epsilon[index] += local_epsilon[index];
            d_epsilon[index] += local_d_rho[index];
            d_sigma[index] += local_d_sigma[index];
            d_laplacian[index] += local_d_laplacian[index];
            d_tau[index] += local_d_tau[index];
        }
    }
    TermDerivatives {
        epsilon,
        d_epsilon,
        d_sigma,
        d_laplacian,
        d_tau,
    }
}

/// 设备收缩要的**行主序扁平 AO 数据**（`n_points × n`，`∇Φ` 按轴三张）。
///
/// 一次构造、SCF 各步复用。为什么单独一个类型而不是就地摊平：摊平是 `O(n_points·n)`
/// 的拷贝（水/def2-SVP、grid-normal 实测 `Φ` 14 MB、`∇Φ` 42 MB），每步重做一次就等于
/// 把设备驻留省下来的 H2D 又还回去一半。`residency()` 是给设备后端的键：
/// **键不同 = 必须重传**，所以它只需要单调不重复，不需要全局唯一。
pub struct XcGridFlatAo {
    residency: u64,
    n: usize,
    n_points: usize,
    values: Vec<f64>,
    gradients: Option<[Vec<f64>; 3]>,
    weights: Vec<f64>,
}

impl XcGridFlatAo {
    /// 把 [`GridAoCached`] 摊平成设备核的布局。
    ///
    /// `need_gradients` 为假时只摊平 `Φ`（LDA 通道），为真时连 `∇Φ` 的三个分量一起
    /// （GGA 通道）。
    ///
    /// # Errors
    ///
    /// AO 缓存与网格/基组的形状对不上（点数或每个点的基函数个数不符）。
    pub fn build(
        basis: &InternalBasis,
        grid: &Grid,
        cache: &GridAoCached,
        need_gradients: bool,
    ) -> Result<Self, EngineError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_RESIDENCY: AtomicU64 = AtomicU64::new(1);

        let n = basis.n_functions();
        let n_points = grid.len();
        if cache.values.len() != n_points {
            return Err(EngineError::Internal(format!(
                "AO cache has {} points, grid has {n_points}",
                cache.values.len()
            )));
        }
        let mut values = Vec::with_capacity(n_points * n);
        let mut gradients = if need_gradients {
            Some([
                Vec::with_capacity(n_points * n),
                Vec::with_capacity(n_points * n),
                Vec::with_capacity(n_points * n),
            ])
        } else {
            None
        };
        for index in 0..n_points {
            let row = &cache.values[index];
            if row.len() != n {
                return Err(EngineError::Internal(format!(
                    "AO cache row {index} has {} functions, basis has {n}",
                    row.len()
                )));
            }
            values.extend_from_slice(row);
            if let Some([gx, gy, gz]) = gradients.as_mut() {
                let point_gradients = &cache.gradients[index];
                if point_gradients.len() != n {
                    return Err(EngineError::Internal(format!(
                        "AO gradient row {index} has {} functions, basis has {n}",
                        point_gradients.len()
                    )));
                }
                for element in point_gradients {
                    gx.push(element[0]);
                    gy.push(element[1]);
                    gz.push(element[2]);
                }
            }
        }
        let weights: Vec<f64> = grid.points.iter().map(|point| point.weight).collect();
        Ok(Self {
            residency: NEXT_RESIDENCY.fetch_add(1, Ordering::Relaxed),
            n,
            n_points,
            values,
            gradients,
            weights,
        })
    }

    /// 驻留键（同一份 `Φ`/`∇Φ` 在 SCF 各步之间不变）。
    #[must_use]
    pub fn residency(&self) -> u64 {
        self.residency
    }

    /// 基函数个数。
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// 网格点数。
    #[must_use]
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// 摊平后占多少字节（`Φ` + `∇Φ` + `w`）—— 记进报告用。
    #[must_use]
    pub fn host_bytes(&self) -> u64 {
        let mut doubles = self.values.len() + self.weights.len();
        if let Some(gradients) = &self.gradients {
            for axis in gradients {
                doubles += axis.len();
            }
        }
        (doubles as u64) * 8
    }

    /// 按这次要走的通道组一个设备请求（借用摊平后的 AO 与调用方给的密度）。
    ///
    /// `with_gradients == false` 时 `ao_gradients` 是 `None`（LDA 通道）；`true` 时把三张
    /// `∇Φ` 一起给出去（GGA 通道）—— 没预取 `∇Φ` 时仍然是 `None`，`evaluate_xc_on_device`
    /// 会把它当形状错误报出来，而不是让设备核去读空切片。
    #[must_use]
    pub fn request<'a>(&'a self, density: &'a [f64], with_gradients: bool) -> XcGridRequest<'a> {
        XcGridRequest {
            residency: self.residency,
            n: self.n,
            n_points: self.n_points,
            ao_values: &self.values,
            ao_gradients: if with_gradients {
                self.gradients
                    .as_ref()
                    .map(|[gx, gy, gz]| [&gx[..], &gy[..], &gz[..]])
            } else {
                None
            },
            density,
            weights: &self.weights,
        }
    }
}

/// 一次设备收缩的结局（设备不可用不是错误 —— 是**退回 CPU** 的信号）。
#[derive(Debug, Clone)]
pub enum XcDeviceEvaluation {
    /// 设备路径完成。
    Done {
        /// `E_xc`，Hartree。
        energy: f64,
        /// `v_xc`（`n × n`）。
        v_xc: Matrix,
    },
    /// 设备不可用（附原因）；调用方退回 [`evaluate_xc_cached`]。
    Unavailable {
        /// 为什么退回。
        reason: String,
    },
}

/// 把这一点的收缩交给设备后端（EX-03）。
///
/// 与 [`evaluate_xc_cached`] 的关系：**`ρ`/`σ` 由设备算、泛函求值仍在 host**（`terms` 是
/// trait object，搬不进 CUDA 源），`E_xc` 的 `O(n_points)` 累加也在 host、且与 CPU 路径
/// 同一个函数。需要 `∇²ρ` 或 `τ` 的泛函直接返回 [`XcDeviceEvaluation::Unavailable`] ——
/// 设备核只实现到 GGA，**不假装**支持那两个通道。
///
/// # Errors
///
/// 密度与摊平后的 AO 形状不一致、设备返回的 `v_xc` 长度不对 → [`EngineError::Internal`]。
/// 设备侧的任何失败**不是**错误：它回来是 `Ok(Unavailable { reason })`，调用方走 CPU。
#[allow(clippy::type_complexity)] // 两个回调签名就是两条通道的契约，起别名只会更难对着核核对
pub fn evaluate_xc_on_device(
    flat: &XcGridFlatAo,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
    backend: &dyn XcGridAccelerator,
    report: &mut XcGridDeviceReport,
) -> Result<XcDeviceEvaluation, EngineError> {
    let n = flat.n;
    if density.nrows() != n || density.ncols() != n {
        return Err(EngineError::Internal(format!(
            "density is {}x{}, device XC contraction needs {n}x{n}",
            density.nrows(),
            density.ncols()
        )));
    }
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);
    if needs_laplacian || needs_tau {
        return Ok(XcDeviceEvaluation::Unavailable {
            reason: "泛函项需要 ∇²ρ/τ 通道；设备核只实现到 GGA（ρ, σ），没有假装支持".to_string(),
        });
    }
    if needs_sigma && flat.gradients.is_none() {
        return Err(EngineError::Internal(
            "device XC contraction needs ∇Φ but the flattened AO cache has no gradients"
                .to_string(),
        ));
    }

    // 行主序密度副本（`O(n²)`，每步一次）。与 CPU 路径同一个 `i` 外层、`j` 内层。
    let mut flat_density = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            flat_density[i * n + j] = density[(i, j)];
        }
    }
    let request = flat.request(&flat_density, needs_sigma);

    let outcome = if needs_sigma {
        let mut evaluate =
            |rho: &[f64], sigma: &[f64]| -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
                let batch = DensityBatch::new(rho, sigma);
                let derivatives = term_derivatives(&batch, rho.len(), terms);
                // 与 `evaluate_xc_cached` 的 GGA 分支同一个 batch 形状、同一段求和。
                let mut epsilon = derivatives.epsilon;
                if epsilon.len() != rho.len() {
                    return Err(format!(
                        "泛函返回 {} 个 ε，应为 n_points = {}",
                        epsilon.len(),
                        rho.len()
                    ));
                }
                epsilon.truncate(rho.len());
                Ok((epsilon, derivatives.d_epsilon, derivatives.d_sigma))
            };
        backend.contract_gga(&request, &mut evaluate)
    } else {
        let mut evaluate = |rho: &[f64]| -> Result<(Vec<f64>, Vec<f64>), String> {
            let batch = DensityBatch::from_rho(rho);
            let derivatives = term_derivatives(&batch, rho.len(), terms);
            let epsilon = derivatives.epsilon;
            if epsilon.len() != rho.len() {
                return Err(format!(
                    "泛函返回 {} 个 ε，应为 n_points = {}",
                    epsilon.len(),
                    rho.len()
                ));
            }
            // v = ε + ρ dε/dρ；ρ=0 时不乘（0×∞ = NaN）—— 与产品路径同一条分支。
            let potential = (0..rho.len())
                .map(|point| {
                    if rho[point] > 0.0 {
                        epsilon[point] + rho[point] * derivatives.d_epsilon[point]
                    } else {
                        epsilon[point]
                    }
                })
                .collect();
            Ok((epsilon, potential))
        };
        backend.contract_lda(&request, &mut evaluate)
    };

    match outcome {
        XcGridDeviceOutcome::Unavailable { reason } => Ok(XcDeviceEvaluation::Unavailable {
            reason: format!("{}：{reason}", backend.name()),
        }),
        XcGridDeviceOutcome::Done {
            energy,
            v_xc,
            report: call_report,
        } => {
            report.accumulate(&call_report);
            if v_xc.len() != n * n {
                return Err(EngineError::Internal(format!(
                    "device XC returned {} v_xc entries for n = {n}",
                    v_xc.len()
                )));
            }
            let mut matrix = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    matrix[(i, j)] = v_xc[i * n + j];
                }
            }
            Ok(XcDeviceEvaluation::Done {
                energy,
                v_xc: matrix,
            })
        }
    }
}

/// RP-02：收敛验收里 DIIS 交换子误差相对密度判据的宽松倍数。
///
/// 交换子误差对密度的响应是线性的（`e ≈ (∂/∂P)(F P S − S P F)·ΔP`，量级 `‖F‖·‖ΔP‖`），
/// 所以「密度已经不动了（`max|ΔP| < density_threshold`）」在交换子身上对应的上限就是
/// `density_threshold · ‖F‖_F`。这个常数只是**上限之上的余量**，不是判据本身。
/// 实测（`QWB_SCF_TRACE=1`，默认 `density_threshold = 1e-6` ⇒ 上限 `|e|/|F| = 1e-4`）：
///
/// * 真收敛：HCN/PBE/def2-SVP `8.7e-10`、H2O/PBE/def2-SVP `7.6e-10`、
///   H2O/SVWN5/STO-3G `1.0e-9` —— 比上限低**五个**数量级；
/// * RP-02 的缺陷态（密度冻在 `H_core` 初猜上、DIIS 子空间全是零误差的陈旧向量）：
///   `3.058e-1` —— 比上限高**三千倍**。
///
/// 见 `tests/rp02_stale_diis_guard.rs`。
const DIIS_ERROR_SLACK: f64 = 100.0;

/// 收敛验收允许的交换子误差上限（`|e|_diis` 的量纲与 `F` 相同）。
fn diis_error_limit(options: &RksOptions, fock_norm: f64) -> f64 {
    DIIS_ERROR_SLACK * options.density_threshold * (fock_norm + 1.0)
}

/// 闭壳层 RKS SCF。
///
/// # Errors
///
/// - 电子数为奇数 → `InvalidInput`（闭壳层要求偶数）；
/// - 泛函需要 σ（GGA）或 AO 需要 l>=2 → `Unsupported`；
/// - 线性代数失败 → `Internal`。
pub fn rks_scf(
    basis: &InternalBasis,
    molecule: &Molecule,
    tensor: &TwoElectronIntegrals,
    grid: &Grid,
    terms: &[Box<dyn FunctionalTerm>],
    options: RksOptions,
) -> Result<RksResult, EngineError> {
    let n = basis.n_functions();
    if tensor.n_functions != n {
        return Err(EngineError::Internal(
            "two-electron tensor size does not match the basis".to_string(),
        ));
    }
    // I-05：有 ECP 时核电荷用 z_eff（点电荷吸引那一半），电子数也按 z_eff 数。
    let charges = match &options.nuclear_charges {
        Some(charges) => {
            if charges.len() != molecule.n_atoms() {
                return Err(EngineError::Internal(format!(
                    "RksOptions::nuclear_charges has {} entries for {} atoms",
                    charges.len(),
                    molecule.n_atoms()
                )));
            }
            charges.clone()
        }
        None => default_charges(molecule)?,
    };
    let mut electrons = 0i64 - i64::from(molecule.charge);
    for charge in &charges {
        // z_eff 是整数（Z − n_core），这里按最近整数累加，避免 f64 截断。
        electrons += charge.round() as i64;
    }
    if electrons <= 0 || electrons % 2 != 0 {
        return Err(EngineError::InvalidInput(format!(
            "closed-shell RKS needs an even, positive electron count, got {electrons}"
        )));
    }
    let n_occ = (electrons / 2) as usize;

    let integrals =
        qwb_integrals::one_electron::one_electron_with_charges(basis, molecule, &charges)?;
    let overlap = &integrals.overlap;
    // H_core = T + V(z_eff) + V_ECP（ECP 的电子-核有效势由调用方算好传进来）。
    let mut hcore = integrals.core_hamiltonian.clone();
    if let Some(potential) = &options.ecp_potential {
        if potential.nrows() != n || potential.ncols() != n {
            return Err(EngineError::Internal(format!(
                "RksOptions::ecp_potential is {}x{}, expected {n}x{n}",
                potential.nrows(),
                potential.ncols()
            )));
        }
        for mu in 0..n {
            for nu in 0..n {
                hcore[(mu, nu)] += potential[(mu, nu)];
            }
        }
    }
    let hcore = &hcore;
    let x = inverse_sqrt_symmetric(overlap)?;
    let e_nuc = nuclear_repulsion(&charges, molecule)?;
    let mut diis = if options.use_diis {
        Some(Diis::new(options.diis_max_vectors))
    } else {
        None
    };

    // 用 Xᵀ H X 对角化得到初猜（`initial_guess_scale` 缩放非对角块，多初猜时用）。
    let guess_hcore = if (options.initial_guess_scale - 1.0).abs() < f64::EPSILON {
        hcore.clone()
    } else {
        let scale = options.initial_guess_scale;
        let mut scaled = hcore.clone();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    scaled[(i, j)] = hcore[(i, j)] * scale;
                }
            }
        }
        scaled
    };
    let mut density = core_guess(&guess_hcore, &x, n_occ)?;
    let mut previous_energy = f64::NAN;
    let mut completed_steps = 0u32;
    if let Some(checkpoint) = &options.resume_from {
        if checkpoint.n_functions != n {
            return Err(EngineError::InvalidInput(format!(
                "resume checkpoint has {} functions but this basis has {n}",
                checkpoint.n_functions
            )));
        }
        density = checkpoint.density_matrix()?;
        if let Some(energy) = checkpoint.energy_hartree {
            previous_energy = energy;
        }
        completed_steps = checkpoint.iterations;
    }
    // S-05：恢复 DIIS 子空间，否则续算的轨迹与不中断的跑法不同（末位会差几个 ulp）。
    if let Some(checkpoint) = &options.resume_from
        && let Some(accelerator) = diis.as_mut()
        && !checkpoint.diis_focks.is_empty()
    {
        *accelerator = Diis::restore(
            options.diis_max_vectors,
            n,
            checkpoint.diis_focks.clone(),
            checkpoint.diis_errors.clone(),
        )
        .map_err(linalg)?;
    }

    // 统一的检查点构造（带上 DIIS 子空间）。
    // 用嵌套 fn 而不是闭包：闭包会长期借用 `diis`，与循环里的 `diis.as_mut()` 冲突。
    fn build_checkpoint(
        density: &Matrix,
        energy: f64,
        steps: u32,
        diis: Option<&Diis>,
    ) -> crate::checkpoint::ScfCheckpoint {
        let base = crate::checkpoint::ScfCheckpoint::new(density, Some(energy), steps);
        match diis {
            Some(accelerator) => {
                let (focks, errors) = accelerator.snapshot();
                base.with_diis(focks, errors)
            }
            None => base,
        }
    }

    let mut iterations = 0;
    let mut converged = false;
    let mut last_change = f64::INFINITY;
    let mut mo_energies = Vector::zeros(n);
    // 最后一次对角化的 MO 系数（列 = MO）；收敛后即 K2 mo_coeffs 的真实内容。
    let mut final_coefficients = Matrix::zeros(n, n);
    let mut e_xc = 0.0;
    let mut electronic = 0.0;

    // AO 三元组与密度无关，整次 SCF 只算一次（苯/STO-3G 的网格求值实测占每步开销的大头）。
    let needs_sigma = !terms.is_empty() && terms.iter().any(|term| term.requires().sigma);
    let xc_cache = if terms.is_empty() {
        None
    } else {
        let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
        let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);
        Some(GridAoCached::build(
            basis,
            grid,
            needs_sigma,
            needs_laplacian,
            needs_tau,
        )?)
    };

    // ── EX-03：设备收缩（默认关） ───────────────────────────────────────────────
    //
    // 三件事在这里定下来，循环里只做「调用 + 记账」：
    //   * `xc_backend`：真的拿到后端才不是 `None`。选了 Gpu 但没有后端（没注入 /
    //     构建没开 `gpu-kernel`）时**在循环之前**就记下原因、什么都不试；
    //   * `xc_flat`：设备核要的行主序 AO，一次摊平（`O(n_points·n)`），各步复用；
    //   * `xc_report`：逐次累加的分项记账，最后放进 `RksResult::xc_device_report`。
    let mut xc_backend: Option<&dyn XcGridAccelerator> = None;
    let mut xc_fallback_reason: Option<String> = None;
    let mut xc_path_used = XcGridPath::Cpu;
    let mut xc_report = XcGridDeviceReport::default();
    if !terms.is_empty() {
        match (options.xc_path, &options.xc_backend) {
            (XcGridPath::Gpu, Some(handle)) => xc_backend = Some(handle.backend()),
            (XcGridPath::Gpu, None) => {
                xc_fallback_reason = Some(
                    "选了 XcGridPath::Gpu 但 RksOptions::xc_backend 是 None（没注入设备后端）"
                        .to_string(),
                );
            }
            (XcGridPath::Cpu, _) => {}
        }
    }
    let xc_flat = match (&xc_cache, xc_backend) {
        (Some(cache), Some(_)) => Some(XcGridFlatAo::build(basis, grid, cache, needs_sigma)?),
        _ => None,
    };

    for step in 0..options.max_iter {
        iterations = step + 1;
        // 外部自洽场（COSMO）：算子依赖密度，所以每步重建。`hcore` 在本循环体内被
        // 遮蔽为 `H_eff = H + V`，于是能量式 `Tr(P H_eff) + ½Tr(P G)` 自动带上 `Tr(P V)`，
        // 实现方只需把 `ΔE` 补上（见 trait 文档）。
        let (field_operator, field_energy) = match &options.external_field {
            Some(handle) => {
                let (operator, energy) = handle.field().operator_and_energy(&density)?;
                (Some(operator), energy)
            }
            None => (None, 0.0),
        };
        let hcore_effective = match &field_operator {
            Some(operator) => {
                if operator.nrows() != n || operator.ncols() != n {
                    return Err(EngineError::Internal(format!(
                        "external field `{}` returned a {}x{} operator, expected {n}x{n}",
                        options
                            .external_field
                            .as_ref()
                            .map_or("unknown", |handle| handle.field().name()),
                        operator.nrows(),
                        operator.ncols()
                    )));
                }
                let mut effective = hcore.clone();
                for i in 0..n {
                    for j in 0..n {
                        effective[(i, j)] += operator[(i, j)];
                    }
                }
                effective
            }
            None => hcore.clone(),
        };
        let hcore = &hcore_effective;
        let fock_data: FockMatrices = build_fock(hcore, tensor, &density).map_err(linalg)?;
        // 纯 HF（terms 为空）**不碰网格**：E_xc = 0、v_xc = 0 是精确结果，
        // 而走一遍网格要按点求 AO 值与密度（H2O/STO-3G 的 grid-coarse 是 29400 点 × 7 个 AO，
        // 每步都算一次会让 HF 的 SCF 慢两个数量级——实测每个 SCF 约 30 秒）。
        let (xc_energy, v_xc) = if terms.is_empty() {
            (0.0, Matrix::zeros(basis.n_functions(), basis.n_functions()))
        } else {
            // EX-03：先试设备后端；它说不可用就**本步**退回 CPU、把原因记下来，
            // 并且**此后不再试**（否则每一步都要付一次失败的探测代价）。
            let mut device = None;
            if let (Some(backend), Some(flat)) = (xc_backend, xc_flat.as_ref()) {
                match evaluate_xc_on_device(flat, &density, terms, backend, &mut xc_report)? {
                    XcDeviceEvaluation::Done {
                        energy,
                        v_xc: device_vxc,
                    } => {
                        xc_path_used = XcGridPath::Gpu;
                        device = Some((energy, device_vxc));
                    }
                    XcDeviceEvaluation::Unavailable { reason } => {
                        xc_fallback_reason = Some(reason);
                        xc_backend = None;
                    }
                }
            }
            match device {
                Some(pair) => pair,
                None => match &xc_cache {
                    Some(cache) => evaluate_xc_cached(basis, grid, cache, &density, terms)?,
                    None => evaluate_xc(basis, grid, &density, terms)?,
                },
            }
        };
        e_xc = xc_energy;

        // 两种约定：
        //  * Kohn–Sham（有 XC 泛函）：F = h + J + v_xc，交换关联全部经 E_xc/v_xc 进入；
        //  * Hartree–Fock（terms 为空）：F = h + J − ½K，即 build_fock 给出的 HF Fock。
        let hf_mode = terms.is_empty();
        let mut fock = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                fock[(i, j)] = if hf_mode {
                    fock_data.fock[(i, j)]
                } else {
                    // KS：F = h + J + v_xc（+ hybrid 的 α·(−½K)）
                    hcore[(i, j)] + fock_data.coulomb[(i, j)] + v_xc[(i, j)]
                        - 0.5 * options.hf_exchange * fock_data.exchange[(i, j)]
                };
            }
        }

        // 静态能级移动：F ← F + λ (S − ½ S P S)，即把虚轨道抬高 λ（闭壳层）。
        // 放在 DIIS 之前：DIIS 外推的是"已经抬过"的 Fock，误差向量也按抬过的算。
        if options.level_shift > 0.0 {
            let sp = matmul(overlap, &density).map_err(linalg)?;
            for i in 0..n {
                for j in 0..n {
                    fock[(i, j)] += options.level_shift * (overlap[(i, j)] - 0.5 * sp[(i, j)]);
                }
            }
        }

        // DIIS：把当步 Fock 与交换子误差推入子空间，用外推的 Fock 去对角化。
        let mut fock_used = fock.clone();
        if let Some(accelerator) = diis.as_mut() {
            let error = Diis::commutator(&fock, &density, overlap).map_err(linalg)?;
            accelerator.push(fock.clone(), error);
            if let Some(extrapolated) = accelerator.extrapolate().map_err(linalg)? {
                fock_used = extrapolated;
            }
        }
        // 交换子误差范数 ‖F P S − S P F‖_F（用子空间里最后推入的那一组，就是当步的）。
        // 没开 DIIS 时它是 0 —— 所以下面的收敛验收只在 `use_diis` 时才拿它当判据。
        let diis_error = diis.as_ref().map_or(0.0, Diis::error_norm);
        // ‖F‖_F：交换子误差的量级基准（见 `DIIS_ERROR_SLACK`）。O(n²)，与 Fock 构建比可忽略。
        let mut fock_norm_squared = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                fock_norm_squared += fock[(i, j)] * fock[(i, j)];
            }
        }
        let fock_norm = fock_norm_squared.sqrt();

        // F' = Xᵀ F X，对角化，C = X C'
        let xt = x.transpose();
        let f_prime = matmul(&matmul(&xt, &fock_used).map_err(linalg)?, &x).map_err(linalg)?;
        let (values, vectors) = symmetric_eigen(&f_prime).map_err(linalg)?;
        mo_energies = values.clone();
        let coefficients = matmul(&x, &vectors).map_err(linalg)?;
        final_coefficients = coefficients.clone();

        // 新密度：P = Σ_i f_i c_i c_i^T（整数占据时 f_i = 2；展宽阶段用费米权重）
        let occupations = if options.smearing_steps > 0
            && step < options.smearing_steps
            && options.smearing_temperature > 0.0
        {
            fermi_occupations(
                values.as_slice(),
                electrons as f64,
                options.smearing_temperature,
            )?
        } else {
            let mut integer = vec![0.0_f64; n];
            for slot in integer.iter_mut().take(n_occ.min(n)) {
                *slot = 2.0;
            }
            integer
        };
        // 诊断用：占据数之和必须等于体系电子数（Σ f_i = n_electrons）。展宽阶段它是一次
        // 真实的检验 —— 费米二分写错方向时它会塌到 0 或 2n，两者都让密度失去意义。
        let occupation_sum: f64 = occupations.iter().sum();
        let mut new_density = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for orbital in 0..n {
                    let weight = occupations[orbital];
                    if weight == 0.0 {
                        continue;
                    }
                    sum += weight * coefficients[(i, orbital)] * coefficients[(j, orbital)];
                }
                new_density[(i, j)] = sum;
            }
        }

        // E = Tr(P H) + ½ Tr(P G) + E_xc + E_nuc
        let mut tr_ph = 0.0;
        let mut tr_pg = 0.0;
        let mut tr_pk = 0.0;
        for i in 0..n {
            for j in 0..n {
                tr_ph += new_density[(i, j)] * hcore[(i, j)];
                // KS 的 2e 项是纯库仑 J；HF 的 2e 项才是 J − ½K。
                tr_pk += new_density[(i, j)] * fock_data.exchange[(i, j)];
                tr_pg += new_density[(i, j)]
                    * if hf_mode {
                        fock_data.fock[(i, j)] - hcore[(i, j)]
                    } else {
                        fock_data.coulomb[(i, j)]
                    };
            }
        }
        // hybrid：E += α·(−¼Tr(PK))；纯 HF 走的是 hf_mode（build_fock 已经含 −½K），不重复计
        let hybrid = if hf_mode {
            0.0
        } else {
            options.hf_exchange * (-0.25 * tr_pk)
        };
        // `field_energy` 是「物理场能 − Tr(PV)」的差（Tr(PV) 已随 H_eff 进 tr_ph）。
        electronic = tr_ph + 0.5 * tr_pg + xc_energy + hybrid + field_energy;
        let energy = electronic + e_nuc;

        let mut max_dp = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                max_dp = max_dp.max((new_density[(i, j)] - density[(i, j)]).abs());
            }
        }
        last_change = if previous_energy.is_nan() {
            f64::INFINITY
        } else {
            (energy - previous_energy).abs()
        };
        // 线性混合（`damping`，0 = 直接替换）：`P ← (1−α)P_old + α P_new`。
        // 近简并/电荷转移体系（ZnO/LANL2DZ 这类含 d10 的体系）上纯 DIIS 会来回震荡，
        // 阻尼是标准对策；DIIS 与阻尼可以同时开（DIIS 外推 Fock，阻尼作用在密度上）。
        if options.damping > 0.0 && options.damping < 1.0 {
            let alpha = options.damping;
            for i in 0..n {
                for j in 0..n {
                    density[(i, j)] = (1.0 - alpha) * density[(i, j)] + alpha * new_density[(i, j)];
                }
            }
        } else {
            density = new_density;
        }
        previous_energy = energy;

        // 诊断（QWB_SCF_TRACE=1）：逐迭代打印能量与密度变化。只在排查收敛时用，
        // 不写进结果（产品面的日志走 `Reporter`）。
        if std::env::var_os("QWB_SCF_TRACE").is_some() {
            eprintln!(
                "scf step {iterations}: E = {energy:+.9}  dE = {last_change:.3e}  max|dP| = {max_dp:.3e}  \
                 sum(f) = {occupation_sum:.6}  |e|_diis = {diis_error:.3e}  \
                 |e|/|F| = {:.3e}",
                diis_error / (fock_norm + 1.0)
            );
        }

        // S-05：按间隔落盘，进程被 kill 也能从最近一步续算。
        if let Some(path) = options.checkpoint_path.as_deref()
            && options.checkpoint_every > 0
            && (step as usize + 1).is_multiple_of(options.checkpoint_every as usize)
        {
            let snapshot = build_checkpoint(
                &density,
                previous_energy,
                completed_steps + iterations,
                diis.as_ref(),
            );
            crate::checkpoint::save_checkpoint(path, &snapshot)?;
        }

        if last_change < options.energy_threshold
            && max_dp < options.density_threshold
            && step >= options.smearing_steps
        {
            // ── RP-02：收敛验收的第三道门 ──────────────────────────────────────────
            //
            // 「能量不再变 + 密度不再动」两条**不足以**说明自洽：密度塌成零矩阵、或者
            // 停在 `H_core` 初猜上时，这两条照样成立，而解一个都没自洽。这正是 RP-02
            // 的形状 —— 展宽二分把占据数打成 0，密度归零，展宽结束后 SCF 从零密度出发
            // 落在初猜的固定点上，DIIS 子空间里全是零误差的陈旧向量，于是 23 步
            // 「收敛」到 -81.7122547541 Ha（PySCF：-93.172 Ha）。
            //
            // `Diis::error_norm` 正是这种状态的反例度量：真解上 `F` 与 `P` 对易
            // （`e → 0`），停在没走的那一步上的 `e` 不小。判据按**调用方自己的密度阈值**
            // 取上限（交换子对密度是线性响应），所以它不会跟一个更松的阈值打架 ——
            // 你声明的密度收敛到什么程度，交换子就得降到相应的量级（见 `DIIS_ERROR_SLACK`）。
            //
            // 能级移动不影响这个度量：`λ(S − ½SPS)` 与 `P` 的交换子是**恒零**
            // （`(S − ½SPS)PS = SPS − ½SPSPS = SP(S − ½SPS)`），所以抬过能级的 Fock
            // 判出来的 `e` 与物理 Fock 相同。没开 DIIS（阻尼 Roothaan 那一级）时
            // 拿不到交换子，这一门跳过 —— 那条路由 `fermi_occupations` 的电子数
            // 不变量兜底（它保证占据数不会塌成 0）。
            let self_consistent = diis_error <= diis_error_limit(&options, fock_norm);
            if !options.use_diis || self_consistent {
                converged = true;
            } else if std::env::var_os("QWB_SCF_TRACE").is_some() {
                eprintln!(
                    "scf step {iterations}: REFUSED as converged: |e|_diis = {diis_error:.3e} > \
                     上限 {:.3e}（密度没停在自洽解上，不返回能量）",
                    diis_error_limit(&options, fock_norm)
                );
            }
            // 无论接受还是拒绝都结束这一级退路：拒绝时 `converged = false`，调用方拿到
            // `ScfNotConverged`，引擎按既有顺序换下一级退路（RP-02 实测就是靠这一步
            // 落到阻尼 Roothaan，给出 -93.1720303941 Ha）。
            break;
        }
    }

    if !converged {
        // S-05：不收敛也落盘，否则「kill 后续算」拿不到状态。
        if let Some(path) = options.checkpoint_path.as_deref() {
            let snapshot = build_checkpoint(
                &density,
                previous_energy,
                completed_steps + iterations,
                diis.as_ref(),
            );
            crate::checkpoint::save_checkpoint(path, &snapshot)?;
        }
        return Err(EngineError::ScfNotConverged {
            iters: iterations,
            max: options.max_iter,
        });
    }

    let checkpoint = build_checkpoint(
        &density,
        previous_energy,
        completed_steps + iterations,
        diis.as_ref(),
    );

    Ok(RksResult {
        energy_hartree: previous_energy + e_nuc - e_nuc, // 保持写法对称：electronic + e_nuc
        scf_converged: converged,
        // 续算时报**总**步数（含检查点之前完成的），不是本次新增的步数。
        iterations: completed_steps + iterations,
        electronic_hartree: electronic,
        exchange_correlation_hartree: e_xc,
        nuclear_repulsion_hartree: e_nuc,
        density,
        mo_energies_hartree: mo_energies,
        coefficients: final_coefficients,
        last_energy_change: last_change,
        grid_points: grid.len(),
        checkpoint,
        xc_path: xc_path_used,
        xc_fallback_reason,
        xc_device_report: if xc_report.calls == 0 {
            None
        } else {
            Some(xc_report)
        },
    })
}

/// 费米展宽下的轨道占据（每个轨道最多 2 个电子）：
/// `f_i = 2 / (1 + exp((ε_i − μ)/T))`，μ 由 `Σ f_i = n_electrons` 二分求出。
///
/// 用途是 SCF 的**退火**：从分数占据平滑地过渡到整数占据，避开"初始猜测很差 →
/// 在几个组态之间跳"的死循环（LANL2DZ 的 ZnO 是活生生的例子：闭壳层受限的 O 原子
/// 本身就是伪解，core guess 从一个伪组态出发）。展宽只在 `smearing_steps` 内生效，
/// 之后切回整数占据，所以**收敛解与能量定义不变**。
///
/// # Errors
///
/// 电子数超出 `[0, 2n]`，或温度非正 → [`EngineError::InvalidInput`]。
pub fn fermi_occupations(
    energies: &[f64],
    n_electrons: f64,
    temperature: f64,
) -> Result<Vec<f64>, EngineError> {
    let n = energies.len();
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(EngineError::InvalidInput(format!(
            "fermi_occupations: temperature {temperature} must be positive"
        )));
    }
    if n_electrons <= 0.0 || n_electrons > 2.0 * n as f64 {
        return Err(EngineError::InvalidInput(format!(
            "fermi_occupations: {n_electrons} electrons for {n} orbitals"
        )));
    }
    let fill = |mu: f64| -> f64 {
        energies
            .iter()
            .map(|energy| 2.0 / (1.0 + ((energy - mu) / temperature).exp()))
            .sum()
    };
    // μ 落在 [min−40T, max+40T] 里一定能把电子数夹住（exp(±40) 已是机器精度外）。
    let mut low = energies.iter().copied().fold(f64::INFINITY, f64::min) - 40.0 * temperature;
    let mut high = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 40.0 * temperature;
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        // `fill(μ)` 对 μ **单调递增**：μ 越高，占据越多。所以
        //   * `fill(middle) > n_electrons` ⇒ 根在 middle 的**左边** ⇒ 上界收到 middle；
        //   * 否则根在右边 ⇒ 下界抬到 middle。
        // 这两个分支原先写反了（`low = middle` 放在上面），二分于是**发散**到
        // `high = max + 40T`：每一级的 `fill` 都比目标大，代码却把它当成「还不够」。
        // 后果是占据数全部塌到 **0**，密度变成零矩阵，能量退化成纯核排斥
        // （HCN/PBE/def2-SVP 实测 step 1–20 全是 `sum(f) = 0`、`E = +23.883886438`），
        // 展宽结束后 SCF 从**零密度**重新出发，落在 `H_core` 初猜的解上却因为
        // 「密度不再动」而报收敛（RP-02：报 -81.71 Ha，PySCF 是 -93.17 Ha）。
        if fill(middle) > n_electrons {
            high = middle;
        } else {
            low = middle;
        }
        if (high - low).abs() < 1.0e-14 {
            break;
        }
    }
    let mu = 0.5 * (low + high);
    let occupations: Vec<f64> = energies
        .iter()
        .map(|energy| 2.0 / (1.0 + ((energy - mu) / temperature).exp()))
        .collect();
    // 二分必须真的解出了 `Σ f_i = n_electrons`。这一条是 RP-02 的**根因级**护栏：
    // 方向写反时返回的是「全部 0」（或全部 2）的占据，把这个数直接交给 SCF 会让密度
    // 失去意义，而下游只看到「密度不动了」。宁可在这里报错让引擎换下一级退路，
    // 也不许把一串假的占据数交给 Fock 构建。
    let total: f64 = occupations.iter().sum();
    if (total - n_electrons).abs() > 1.0e-8 * n_electrons.max(1.0) {
        return Err(EngineError::Internal(format!(
            "fermi_occupations: bisection returned occupations summing to {total}, \
             not {n_electrons} electrons (T = {temperature}, {} orbitals)",
            energies.len()
        )));
    }
    Ok(occupations)
}

/// 初猜密度：对角化 `H_core` 并占据前 `n_occ` 个轨道（`P = 2 sum c c^T`）。
///
/// # Errors
///
/// 线性代数失败。
pub fn core_guess(hcore: &Matrix, x: &Matrix, n_occ: usize) -> Result<Matrix, EngineError> {
    let n = hcore.nrows();
    let xt = x.transpose();
    let h_prime = matmul(&matmul(&xt, hcore).map_err(linalg)?, x).map_err(linalg)?;
    let (_values, vectors) = symmetric_eigen(&h_prime).map_err(linalg)?;
    let coefficients = matmul(x, &vectors).map_err(linalg)?;
    let mut density = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for orbital in 0..n_occ.min(n) {
                sum += coefficients[(i, orbital)] * coefficients[(j, orbital)];
            }
            density[(i, j)] = 2.0 * sum;
        }
    }
    Ok(density)
}

/// 多初猜 SCF：用若干个 `initial_guess_scale` 各跑一次，返回**能量最低**的收敛解。
///
/// 动机（实测）：N2/STO-3G 的纯 core guess 会收敛到激发态——总能量比 CCCBDB 的
/// HF 值高 **0.684 Ha**，占据轨道能级也明显不对（1π_u 不成对简并、HOMO 被抬到 -0.18 Ha）。
/// 近简并体系（三键、同核双原子）上这是 SCF 初猜的经典问题，多初猜是标准对策。
///
/// # Errors
///
/// 所有初猜都不收敛，或线性代数失败。
pub fn rks_scf_multistart(
    basis: &InternalBasis,
    molecule: &Molecule,
    tensor: &qwb_integrals::two_electron::TwoElectronIntegrals,
    grid: &qwb_grid::Grid,
    terms: &[Box<dyn FunctionalTerm>],
    options: RksOptions,
) -> Result<RksResult, EngineError> {
    let mut best: Option<RksResult> = None;
    let mut last_error: Option<EngineError> = None;
    for scale in [1.0, 0.5, 1.5, 0.25, 2.0] {
        let mut attempt = options.clone();
        attempt.initial_guess_scale = scale;
        match rks_scf(basis, molecule, tensor, grid, terms, attempt) {
            Ok(result) => {
                let better = match &best {
                    None => true,
                    Some(current) => result.energy_hartree < current.energy_hartree - 1.0e-12,
                };
                if better {
                    best = Some(result);
                }
            }
            Err(err) => last_error = Some(err),
        }
    }
    match (best, last_error) {
        (Some(result), _) => Ok(result),
        (None, Some(err)) => Err(err),
        (None, None) => Err(EngineError::Internal(
            "multistart SCF: no attempt produced a result".to_string(),
        )),
    }
}

/// 便捷入口：从注册表里的基组/泛函/网格名做一次 RKS（T1 路径用）。
///
/// # Errors
///
/// 名字不存在、数据错误、SCF 输入非法。
pub fn rks_from_data(
    registry: &qwb_registry::Registry,
    molecule: &Molecule,
    basis_name: &str,
    functional_name: &str,
    grid_name: &str,
    options: RksOptions,
) -> Result<RksResult, EngineError> {
    let basis_k1 = registry.get_basis(basis_name)?;
    let functional = registry.get_functional(functional_name)?;
    let grid_level = registry.get_grid(grid_name)?;
    let basis = InternalBasis::build(basis_k1, molecule)?;
    let tensor = two_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(grid_level, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;

    let parameters: BTreeMap<String, f64> = functional.parameters.clone();
    let mut options = options;
    let mut terms: Vec<Box<dyn FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        // `HfExchange` 不是一个能量密度项，而是 hybrid 的混入比例：读进 options，不建 term。
        if component.id == "HfExchange" {
            options.hf_exchange += component.weight;
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters)
            .map_err(|err| EngineError::Data(format!("functional `{functional_name}`: {err}")))?;
        // 权重 ≠ 1（复合泛函）由 qwb-xc 的 ScaledTerm 处理：能量与两个导数用同一权重。
        let term = qwb_xc::scale_term(term, component.weight);
        // GGA（σ 依赖）现在由 evaluate_xc 支持（D-10）。
        terms.push(term);
    }

    rks_scf(&basis, molecule, &tensor, &grid, &terms, options)
}

// ---------------------------------------------------------------------------
// Kohn–Sham 解析梯度的 XC 显式项（G-01 / T6）
// ---------------------------------------------------------------------------

/// 每个 AO 在 `point` 处的**二阶导数张量** `∂²φ_μ/∂r_i∂r_j`（对称 3×3）。
///
/// 为什么需要它：GGA 的 XC 梯度里有 σ 通道，`σ = |∇ρ|²`，于是
/// `∂σ/∂R_A = 2∇ρ·∂(∇ρ)/∂R_A`，而 `∂(∇φ_μ)/∂R_A = −∂²φ_μ/∂r_i∂r_j`
/// —— 就是这里的量。只靠一阶 AO 导数写不出这一项（HF 没有它，所以这个缺口
/// 直到把 Fock 换成 KS 才暴露）。
///
/// 对笛卡尔分量 `φ = N·u_x^a u_y^b u_z^c e^{−αu²}`（`u = r − A`）：
///
/// ```text
/// ∂²/∂u_i∂u_j (g·e^{−αu²}) = e^{−αu²}[ ∂_i∂_j g − 2α(u_j ∂_i g + u_i ∂_j g) − 2α δ_ij g + 4α² u_i u_j g ]
/// ```
///
/// 令 `j = i` 就退回 `ao_laplacian` 里那条闭式（`a(a−1)u^{a−2} − 2α(2a+1)u^a + 4α²u^{a+2}`），
/// 两条公式的一致性由 `gga_ao_gradient` 的有限差分测试覆盖。
///
/// # Errors
///
/// 球谐变换维度不一致（`l >= 2` 的球谐壳层）。
pub fn ao_hessian(
    basis: &InternalBasis,
    point: [f64; 3],
) -> Result<Vec<[[f64; 3]; 3]>, EngineError> {
    let mut hessian = vec![[[0.0_f64; 3]; 3]; basis.n_functions()];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let d = [
            point[0] - shell.center_bohr[0],
            point[1] - shell.center_bohr[1],
            point[2] - shell.center_bohr[2],
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let mut cartesian = [[[0.0_f64; 3]; 3]; MAX_CARTESIAN];
        for (component, powers) in components.iter().enumerate() {
            let p = [powers.0, powers.1, powers.2];
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm((p[0], p[1], p[2]), *alpha);
                let weight = coefficient * norm * (-alpha * r2).exp();
                let monomial = |exponents: [u32; 3]| {
                    d[0].powi(i32::try_from(exponents[0]).expect("small"))
                        * d[1].powi(i32::try_from(exponents[1]).expect("small"))
                        * d[2].powi(i32::try_from(exponents[2]).expect("small"))
                };
                for i in 0..3 {
                    for j in 0..3 {
                        let mut term = 0.0;
                        // ∂_i∂_j g（单变量二阶 / 混合二阶）
                        if i == j {
                            if p[i] >= 2 {
                                let mut lowered = p;
                                lowered[i] -= 2;
                                term += f64::from(p[i] * (p[i] - 1)) * monomial(lowered);
                            }
                        } else if p[i] >= 1 && p[j] >= 1 {
                            let mut lowered = p;
                            lowered[i] -= 1;
                            lowered[j] -= 1;
                            term += f64::from(p[i] * p[j]) * monomial(lowered);
                        }
                        // −2α(u_j ∂_i g + u_i ∂_j g)
                        if p[i] >= 1 {
                            let mut shifted = p;
                            shifted[i] -= 1;
                            shifted[j] += 1;
                            term -= 2.0 * alpha * f64::from(p[i]) * monomial(shifted);
                        }
                        if p[j] >= 1 {
                            let mut shifted = p;
                            shifted[j] -= 1;
                            shifted[i] += 1;
                            term -= 2.0 * alpha * f64::from(p[j]) * monomial(shifted);
                        }
                        // −2α δ_ij g
                        if i == j {
                            term -= 2.0 * alpha * monomial(p);
                        }
                        // +4α² u_i u_j g
                        let mut raised = p;
                        raised[i] += 1;
                        raised[j] += 1;
                        term += 4.0 * alpha * alpha * monomial(raised);
                        cartesian[component][i][j] += weight * term;
                    }
                }
            }
        }
        // 球谐变换作用在每个 (i,j) 分量上（与 `ao_value_and_gradient` 同一套写法）。
        let range = if shell.angular == qwb_registry::Angular::Spherical {
            spherical_count(shell.l)
        } else {
            components.len()
        };
        for i in 0..3 {
            for j in 0..3 {
                let mut slice = [0.0_f64; MAX_CARTESIAN];
                for component in 0..components.len() {
                    slice[component] = cartesian[component][i][j];
                }
                let mut flat = [0.0_f64; MAX_CARTESIAN];
                write_transformed(
                    &mut flat,
                    0,
                    &slice[..components.len()],
                    shell.l,
                    shell.angular,
                )?;
                for slot in 0..range {
                    hessian[shell.first_function + slot][i][j] = flat[slot];
                }
            }
        }
    }
    Ok(hessian)
}

/// 每个 AO 的**拉普拉斯量的中心导数** `∂(∇²φ_μ)/∂r_a`（`a = 0,1,2`）。
///
/// 为什么需要它：LYP（B3LYP 的组件之一）的能量密度依赖 `∇²ρ`，于是 XC 梯度里有
/// `(ρ·∂ε/∂∇²ρ)·∂(∇²ρ)/∂R` 一项，而 `∂(∇²ρ)` 里含 `∂(∇²φ_μ)/∂R = −∂_a(∇²φ_μ)`。
///
/// 对笛卡尔分量 `∇²φ = Σ_prims w·L(u)`（`L` 是 `ao_laplacian` 里那串多项式），
/// 把 `∂_a` 作用上去：`∂_a(w·L) = w·(∂_a L − 2α u_a L)`。
///
/// # Errors
///
/// 球谐变换维度不一致。
pub fn ao_laplacian_derivative(
    basis: &InternalBasis,
    point: [f64; 3],
) -> Result<Vec<[f64; 3]>, EngineError> {
    let mut derivatives = vec![[0.0_f64; 3]; basis.n_functions()];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let d = [
            point[0] - shell.center_bohr[0],
            point[1] - shell.center_bohr[1],
            point[2] - shell.center_bohr[2],
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let mut cartesian = [[0.0_f64; 3]; MAX_CARTESIAN];
        for (component, powers) in components.iter().enumerate() {
            let p = [powers.0, powers.1, powers.2];
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm((p[0], p[1], p[2]), *alpha);
                let weight = coefficient * norm * (-alpha * r2).exp();
                let monomial = |exponents: [u32; 3]| {
                    d[0].powi(i32::try_from(exponents[0]).expect("small"))
                        * d[1].powi(i32::try_from(exponents[1]).expect("small"))
                        * d[2].powi(i32::try_from(exponents[2]).expect("small"))
                };
                // L(u) 本身（与 ao_laplacian 同式）
                let mut lap_polynomial = 0.0;
                for k in 0..3 {
                    if p[k] >= 2 {
                        let mut lowered = p;
                        lowered[k] -= 2;
                        lap_polynomial += f64::from(p[k] * (p[k] - 1)) * monomial(lowered);
                    }
                    lap_polynomial -= 2.0 * alpha * (2.0 * f64::from(p[k]) + 1.0) * monomial(p);
                    let mut raised = p;
                    raised[k] += 2;
                    lap_polynomial += 4.0 * alpha * alpha * monomial(raised);
                }
                for a in 0..3 {
                    // ∂_a L(u)
                    let mut derivative = 0.0;
                    for k in 0..3 {
                        if p[k] >= 2 {
                            let mut lowered = p;
                            lowered[k] -= 2;
                            let mut target = lowered;
                            if target[a] >= 1 {
                                target[a] -= 1;
                                derivative += f64::from(p[k] * (p[k] - 1))
                                    * f64::from(lowered[a])
                                    * monomial(target);
                            }
                        }
                        if p[a] >= 1 {
                            let mut target = p;
                            target[a] -= 1;
                            derivative -= 2.0
                                * alpha
                                * (2.0 * f64::from(p[k]) + 1.0)
                                * f64::from(p[a])
                                * monomial(target);
                        }
                        let mut raised = p;
                        raised[k] += 2;
                        if raised[a] >= 1 {
                            let mut target = raised;
                            target[a] -= 1;
                            derivative +=
                                4.0 * alpha * alpha * f64::from(raised[a]) * monomial(target);
                        }
                    }
                    cartesian[component][a] +=
                        weight * (derivative - 2.0 * alpha * d[a] * lap_polynomial);
                }
            }
        }
        let range = if shell.angular == qwb_registry::Angular::Spherical {
            spherical_count(shell.l)
        } else {
            components.len()
        };
        for a in 0..3 {
            let mut slice = [0.0_f64; MAX_CARTESIAN];
            for component in 0..components.len() {
                slice[component] = cartesian[component][a];
            }
            let mut flat = [0.0_f64; MAX_CARTESIAN];
            write_transformed(
                &mut flat,
                0,
                &slice[..components.len()],
                shell.l,
                shell.angular,
            )?;
            for slot in 0..range {
                derivatives[shell.first_function + slot][a] = flat[slot];
            }
        }
    }
    Ok(derivatives)
}

/// XC 能量的**显式基组运动项** `∂E_xc/∂R_A`（Kohn–Sham 解析梯度缺的那一块）。
///
/// Kohn–Sham 的能量是 `E = Tr(PH) + E_2e + E_xc(P) + E_nuc`，对核坐标求导时
/// `Tr(dP·F_KS) = −Tr(W dS)` 那一项由 `W = ½P F_KS P` 的 Pulay 项负责（`F_KS` 里
/// 已经含 `v_xc`），但 `E_xc` 对**基函数位置**的显式依赖必须单独加：
///
/// ```text
/// ∂E_xc/∂R_A,x = Σ_g w_g [ (ε + ρ ε_ρ)·∂ρ/∂R_A,x + (ρ ε_σ)·∂σ/∂R_A,x ]
/// ∂ρ/∂R_A,x    = Σ_{μ∈A} (∂φ_μ/∂R_A,x)·Ỹ_μ
/// ∂∇ρ/∂R_A,x   = Σ_{μ∈A} [ (∂∇φ_μ/∂R_A,x)·Ỹ_μ + (∂φ_μ/∂R_A,x)·Z̃_μ ]
/// ∂σ/∂R_A,x    = 2 ∇ρ·∂∇ρ/∂R_A,x
/// ```
///
/// 其中 `Ỹ_μ = Σ_ν (P_μν + P_νμ) φ_ν`、`Z̃_μ = Σ_ν (P_μν + P_νμ) ∇φ_ν`
/// （用 `P + Pᵀ` 是为了不假设 P 严格对称），`∂φ_μ/∂R_A,x = −∂φ_μ/∂r_x`（平移不变）。
///
/// **网格本身随原子移动那一半没算**（Becke 权重的导数 + 网格点位置的导数）。
/// 早先这里写的是"通行做法，通常是 1e-6 量级"—— **实测不是**：
/// H2O/STO-3G/PBE 上解析（只有这一半）与数值梯度的最大分量差是 **3.30e-4 Ha/Bohr**，
/// 而 H2/STO-3G/PBE 只有 8.4e-7（被对称性抹掉了，所以早期测试看不出问题）。
/// 生产路径（`qwb_engine` 的 `opt`/Hessian）因此改用
/// [`rks_analytic_gradient_with_xc`]，把这一半交给
/// `qwb_engine::gradient::xc_geometric_derivative`（固定密度下对 `E_xc(R)` 做中心差分）。
///
/// # Errors
///
/// 需要 `∇²ρ` 的 meta-GGA 不在解析梯度的支持范围内（显式拒绝，不静默忽略）；
/// 密度或 AO 数据出现非有限值。
pub fn xc_gradient_terms(
    basis: &InternalBasis,
    molecule: &Molecule,
    grid: &Grid,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
) -> Result<Vec<[f64; 3]>, EngineError> {
    let n = basis.n_functions();
    let natoms = molecule.atoms.len();
    let mut gradient = vec![[0.0_f64; 3]; natoms];
    if density.nrows() != n || density.ncols() != n {
        return Err(EngineError::Internal(
            "xc_gradient_terms: density size mismatch".to_string(),
        ));
    }
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    // LYP（B3LYP 的组件之一）需要 ∇²ρ，所以这一路也要算：`∂(∇²ρ)/∂R` 用
    // `ao_laplacian_derivative`（AO 拉普拉斯量再对中心求一次导）。
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    if terms.is_empty() {
        return Ok(gradient);
    }

    // AO → 原子（球谐/直角都按壳层的 function 区间取）
    let mut owner = vec![usize::MAX; n];
    for shell in basis.shells() {
        for slot in 0..shell.n_functions() {
            owner[shell.first_function + slot] = shell.atom_index;
        }
    }

    for GridPoint {
        position_bohr,
        weight,
        ..
    } in &grid.points
    {
        let point = *position_bohr;
        // ∇φ 一直要：LDA 的显式项也靠它（∂φ/∂R = −∂φ/∂r），只有 Hessian 能省。
        let (values, gradients) = ao_value_and_gradient(basis, point)?;
        let hessian = if needs_sigma {
            ao_hessian(basis, point)?
        } else {
            Vec::new()
        };
        let ao_laplacians = if needs_laplacian {
            ao_laplacian(basis, point)?
        } else {
            Vec::new()
        };
        let ao_laplacian_derivatives = if needs_laplacian {
            ao_laplacian_derivative(basis, point)?
        } else {
            Vec::new()
        };
        // ∇²ρ = Σ P (∇²φ_μ φ_ν + 2 ∇φ_μ·∇φ_ν + φ_μ ∇²φ_ν)
        let mut density_laplacian = 0.0;
        if needs_laplacian {
            for mu in 0..n {
                for nu in 0..n {
                    let p = density[(mu, nu)];
                    if p == 0.0 {
                        continue;
                    }
                    let dot = gradients[mu][0] * gradients[nu][0]
                        + gradients[mu][1] * gradients[nu][1]
                        + gradients[mu][2] * gradients[nu][2];
                    density_laplacian += p
                        * (ao_laplacians[mu] * values[nu]
                            + 2.0 * dot
                            + values[mu] * ao_laplacians[nu]);
                }
            }
        }

        // ρ 与 ∇ρ（与 SCF 的 evaluate_xc 同一条缩并）
        let mut rho = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                rho += values[mu] * density[(mu, nu)] * values[nu];
            }
        }
        if !rho.is_finite() {
            return Err(EngineError::Internal(
                "xc_gradient_terms: density is not finite at a grid point".to_string(),
            ));
        }
        let mut grad_rho = [0.0_f64; 3];
        if needs_sigma {
            for mu in 0..n {
                for nu in 0..n {
                    let p = density[(mu, nu)];
                    if p == 0.0 {
                        continue;
                    }
                    for axis in 0..3 {
                        grad_rho[axis] += p
                            * (gradients[mu][axis] * values[nu] + values[mu] * gradients[nu][axis]);
                    }
                }
            }
        }
        let sigma =
            grad_rho[0] * grad_rho[0] + grad_rho[1] * grad_rho[1] + grad_rho[2] * grad_rho[2];
        let rho = rho.max(0.0);

        // 泛函在这一点的一阶导数
        let rho_slice = [rho];
        let sigma_slice = [sigma];
        let laplacian_slice = [density_laplacian];
        let batch = if needs_laplacian {
            DensityBatch::with_laplacian(&rho_slice, &sigma_slice, &laplacian_slice)
        } else if needs_sigma {
            DensityBatch::new(&rho_slice, &sigma_slice)
        } else {
            DensityBatch::from_rho(&rho_slice)
        };
        let mut epsilon = [0.0_f64];
        let mut d_rho = [0.0_f64];
        let mut d_sigma = [0.0_f64];
        let mut d_laplacian = [0.0_f64];
        for term in terms {
            let mut local_epsilon = [0.0_f64];
            let mut local_rho = [0.0_f64];
            let mut local_sigma = [0.0_f64];
            let mut local_laplacian = [0.0_f64];
            term.energy_density(&batch, &mut local_epsilon);
            term.potential_derivatives_laplacian(
                &batch,
                &mut local_rho,
                &mut local_sigma,
                &mut local_laplacian,
            );
            epsilon[0] += local_epsilon[0];
            d_rho[0] += local_rho[0];
            d_sigma[0] += local_sigma[0];
            d_laplacian[0] += local_laplacian[0];
        }
        let v_laplacian = if needs_laplacian && rho > 0.0 {
            rho * d_laplacian[0]
        } else {
            0.0
        };
        let v_rho = if rho > 0.0 {
            epsilon[0] + rho * d_rho[0]
        } else {
            epsilon[0]
        };
        let v_sigma = if needs_sigma && rho > 0.0 {
            rho * d_sigma[0]
        } else {
            0.0
        };

        // Ỹ_μ = Σ_ν (P_μν + P_νμ) φ_ν 、 Z̃_μ = Σ_ν (P_μν + P_νμ) ∇φ_ν
        let mut y_tilde = vec![0.0_f64; n];
        let mut z_tilde = vec![[0.0_f64; 3]; n];
        for mu in 0..n {
            let mut y = 0.0;
            let mut z = [0.0_f64; 3];
            for nu in 0..n {
                let p = density[(mu, nu)] + density[(nu, mu)];
                if p == 0.0 {
                    continue;
                }
                y += p * values[nu];
                if needs_sigma {
                    for axis in 0..3 {
                        z[axis] += p * gradients[nu][axis];
                    }
                }
            }
            y_tilde[mu] = y;
            z_tilde[mu] = z;
        }
        // L̃_μ = Σ_ν (P_μν + P_νμ) ∇²φ_ν（只在需要 ∇²ρ 通道时用）
        let mut lap_tilde = vec![0.0_f64; n];
        if needs_laplacian {
            for mu in 0..n {
                let mut total = 0.0;
                for nu in 0..n {
                    let p = density[(mu, nu)] + density[(nu, mu)];
                    if p != 0.0 {
                        total += p * ao_laplacians[nu];
                    }
                }
                lap_tilde[mu] = total;
            }
        }

        for mu in 0..n {
            let atom = owner[mu];
            if atom == usize::MAX || y_tilde[mu] == 0.0 {
                continue;
            }
            for axis in 0..3 {
                let d_phi = -gradients[mu][axis];
                let mut sigma_axis = 0.0;
                if needs_sigma {
                    let mut grad_rho_axis = [0.0_f64; 3];
                    for j in 0..3 {
                        grad_rho_axis[j] =
                            -hessian[mu][axis][j] * y_tilde[mu] + d_phi * z_tilde[mu][j];
                    }
                    sigma_axis = 2.0
                        * (grad_rho[0] * grad_rho_axis[0]
                            + grad_rho[1] * grad_rho_axis[1]
                            + grad_rho[2] * grad_rho_axis[2]);
                }
                let mut laplacian_axis = 0.0;
                if needs_laplacian {
                    // ∂_A,a(∇²ρ) = Σ_{μ∈A} [ (∂_a∇²φ_μ)·Ỹ_μ + 2(∂_a∇φ_μ)·Z̃_μ + (∂_aφ_μ)·L̃_μ ]
                    laplacian_axis =
                        -ao_laplacian_derivatives[mu][axis] * y_tilde[mu] + d_phi * lap_tilde[mu];
                    for j in 0..3 {
                        laplacian_axis += 2.0 * (-hessian[mu][axis][j]) * z_tilde[mu][j];
                    }
                }
                gradient[atom][axis] += weight
                    * (v_rho * d_phi * y_tilde[mu]
                        + v_sigma * sigma_axis
                        + v_laplacian * laplacian_axis);
            }
        }
    }

    for slot in &gradient {
        if !slot.iter().all(|value| value.is_finite()) {
            return Err(EngineError::Internal(
                "xc_gradient_terms: gradient is not finite".to_string(),
            ));
        }
    }
    Ok(gradient)
}

/// Kohn–Sham 解析梯度：`骨架项 + ½α_hf 交换 + Pulay(W = ½P F_KS P) + XC 显式项`。
///
/// # 为什么必须由这里自己搭 Fock
///
/// Pulay 项用 `W = ½ P F_KS P`，而 `F_KS = H + J − ½α_hf·K + v_xc`：
///
/// * 纯泛函（LDA/GGA）**没有** HF 交换（`α_hf = 0`），`F` 里不能出现 `−½K`；
/// * 杂化泛函（B3LYP 0.20 / PBE0 0.25）要按 `α_hf` 缩放；
/// * 2e 骨架项同理：库仑恒有，交换乘 `α_hf`。
///
/// 把 `fock` 交给调用方去拼，就等于把"别把 HF 交换算进 KS 能量"这件事交给运气
/// —— 实测差 8.8e-2 Ha/Bohr，正是 `g01_analytic_pbe` 一开始失败的原因。
///
/// # Errors
///
/// 形状不匹配、AO 角动量超出实现、或 XC 是 meta-GGA（需要 `∇²ρ`，未支持）。
#[allow(clippy::too_many_arguments)]
pub fn rks_analytic_gradient(
    basis: &InternalBasis,
    molecule: &Molecule,
    grid: &Grid,
    hcore: &Matrix,
    tensor: &TwoElectronIntegrals,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
    hf_exchange: f64,
) -> Result<Vec<[f64; 3]>, EngineError> {
    let xc = xc_gradient_terms(basis, molecule, grid, density, terms)?;
    rks_analytic_gradient_with_xc(
        basis,
        molecule,
        grid,
        hcore,
        tensor,
        density,
        terms,
        hf_exchange,
        &xc,
    )
}

/// 与 [`rks_analytic_gradient`] 相同，但 **XC 的几何导数由调用方给**。
///
/// # 为什么需要这个入口
///
/// [`xc_gradient_terms`] 只算了"密度随原子移动"那一半（ρ 通道 + σ 通道 + `∇²ρ` 通道），
/// **没有**算"网格本身随原子移动"那一半（Becke 权重的导数、网格点位置的导数）。
/// 这一项在 H2 上被对称性抹掉（所以早期测试都是 1e-7），但在 H2O/STO-3G/PBE 上实测
/// **3.30e-4 Ha/Bohr**，直接把 def2-SVP 的优化几何带偏（O–H 0.9745 vs 0.9682 Å）。
///
/// 补它的两条路：① 解析地算 Becke 权重导数（还没做）；② 在**固定密度**下对
/// `E_xc(R)` 做中心差分（`qwb_engine::gradient::xc_geometric_derivative` 做的就是这个，
/// 每次梯度多 `6N` 次 XC 求值，比一次 SCF 还便宜）。两条路给的是同一个量，
/// 这个入口就是让 ② 的结果能和 ① 一样接进总梯度。
///
/// # Errors
///
/// 形状不匹配、AO 角动量实现缺失、XC 需要 `∇²ρ` 而基组不支持。
#[allow(clippy::too_many_arguments)]
pub fn rks_analytic_gradient_with_xc(
    basis: &InternalBasis,
    molecule: &Molecule,
    grid: &Grid,
    hcore: &Matrix,
    tensor: &TwoElectronIntegrals,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
    hf_exchange: f64,
    xc_gradient: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, EngineError> {
    if xc_gradient.len() != molecule.atoms.len() {
        return Err(EngineError::Internal(format!(
            "rks_analytic_gradient_with_xc: xc gradient has {} entries, molecule has {} atoms",
            xc_gradient.len(),
            molecule.atoms.len()
        )));
    }
    let fock_hf = build_fock(hcore, tensor, density).map_err(|err| {
        EngineError::Internal(format!("rks_analytic_gradient: build_fock: {err}"))
    })?;
    let (_, v_xc) = evaluate_xc(basis, grid, density, terms)?;
    // F_KS = H + J − ½α_hf K + v_xc
    let n = basis.n_functions();
    let mut fock = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            fock[(mu, nu)] = hcore[(mu, nu)] + fock_hf.coulomb[(mu, nu)]
                - 0.5 * hf_exchange * fock_hf.exchange[(mu, nu)]
                + v_xc[(mu, nu)];
        }
    }
    let components =
        crate::analytic_gradient::gradient_components(basis, molecule, density, &fock)?;

    let mut gradient = vec![[0.0_f64; 3]; molecule.atoms.len()];
    for (atom, slot) in gradient.iter_mut().enumerate() {
        for (axis, value) in slot.iter_mut().enumerate() {
            *value = components.kinetic[atom][axis]
                + components.nuclear_ao[atom][axis]
                + components.nuclear_center[atom][axis]
                + components.nuclear_repulsion[atom][axis]
                + components.two_electron_coulomb[atom][axis]
                + hf_exchange * components.two_electron_exchange[atom][axis]
                + components.pulay[atom][axis]
                + xc_gradient[atom][axis];
        }
    }
    Ok(gradient)
}
