//! SV-01 的第三半：把 COSMO 的表面电荷**接回 SCF**（双向耦合），得到 `ΔG_solv`。
//!
//! # 为什么需要这一层
//!
//! `qwb-props::cosmo` 只回答"给定表面电势 `V`，表面电荷 `q` 是多少"。真实计算里
//! `V` 由溶质的 SCF 密度给出，而 `q` 又通过反应场算子 `V_solv` 改变密度 —— 两边必须
//! 同时自洽。本模块把这个耦合做成 [`qwb_scf::rks::ExternalField`]：SCF 每步用当前
//! 密度重建算子，实现里只做线性代数。
//!
//! # 算子与能量（与 SCF 的约定要对齐）
//!
//! 记 `Ω_K,μν = ∫ φ_μ φ_ν / |r − r_K| d³r`（`qwb-integrals` 的核吸引积分，把 tessera
//! 当成一个 +1 点电荷）。于是
//!
//! ```text
//! V_K   = Σ_a Z_a/|R_a − r_K| − Σ_μν P_μν Ω_K,μν        （表面电势）
//! q^solv = f(ε) A σ,   Σ_K' q_K'/d_KK' = −V_K            （COSMO 线性方程）
//! V_solv,μν = Σ_K q_K Ω_K,μν                             （反应场算子）
//! ΔG_el = ½ Σ_K V_K q_K                                  （静电溶剂化自由能）
//! ```
//!
//! SCF 的电子能是 `Tr(P H_eff) + ½Tr(P G) + E_xc`，`H_eff = H + V_solv` 已经带上
//! `Tr(P V_solv)`。而 `ΔG_el` 里同样的量以 `½` 出现，所以本模块返回的能量修正是
//!
//! ```text
//! ΔE = ΔG_el − Tr(P V_solv)
//! ```
//!
//! 代回 SCF 的总能量式即得 `E_总 = E_真空 + ΔG_el` —— 这条恒等式在测试里被直接钉住。
//!
//! # 一个必须说清的近似
//!
//! 只做**静电**项。真实溶剂化自由能还有空腔生成、色散、排斥三项（SMD 的 SV-03），
//! 所以本模块的 `ΔG` 与实验值的差里有一部分来自缺失的非极性项 —— 账本里如实写，
//! 不拿"参数调得好"冒充物理解释。

use std::sync::Arc;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_integrals::one_electron::one_electron_with_charges;
use qwb_linalg::Matrix;
use qwb_props::cavity::Tessera;
use qwb_props::cosmo::{CosmoOptions, cosmo_surface_charges};
use qwb_registry::elements::atomic_number;
use qwb_scf::rks::{ExternalField, ExternalFieldHandle};

/// 把 COSMO 反应场接到 SCF 的场实现。
///
/// 构造时预计算全部与密度无关的量（`Ω_K` 与核电势），之后每步只剩 `O(M·n² + M³/3)`。
#[derive(Debug)]
pub struct CosmoScfField {
    /// 空腔表面碎片（位置 + 面积），供 COSMO 方程用。
    tesserae: Vec<Tessera>,
    /// `Ω_K = ∫φφ/|r−r_K|`，每个碎片一个 `n×n` 矩阵。
    omega: Vec<Matrix>,
    /// 核电荷在碎片中心处的电势（Hartree/e）。
    nuclear: Vec<f64>,
    /// 介电常数。
    epsilon: f64,
    /// 求解选项（残差阈值、1.07 自作用常数）。
    options: CosmoOptions,
    /// AO 数。
    n_functions: usize,
    /// 最近一次求解的静电溶剂化能（Hartree）—— 日志与测试用，不参与 SCF 收敛判据。
    last_solvation_energy: std::sync::Mutex<f64>,
    /// 最近一次的表面电荷总数（e）—— 诊断用。
    last_total_charge: std::sync::Mutex<f64>,
}

impl CosmoScfField {
    /// 构建场：预计算 `Ω_K` 与核电势。
    ///
    /// `tesserae` 通常来自 [`qwb_props::cavity::cavity`] 或 `analytic_area_volume`。
    ///
    /// # Errors
    ///
    /// 空腔为空、基组与分子不匹配、积分失败、`ε < 1` → [`EngineError`]。
    pub fn build(
        basis: &InternalBasis,
        molecule: &Molecule,
        tesserae: &[Tessera],
        epsilon: f64,
    ) -> Result<Self, EngineError> {
        if tesserae.is_empty() {
            return Err(EngineError::InvalidInput(
                "cosmo: the cavity has no tesserae (SV-01)".into(),
            ));
        }
        // 介电常数先校验一次，免得跑到 SCF 第 1 步才发现
        qwb_props::cosmo::dielectric_scale(epsilon)?;
        let bohr = qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
        let np = molecule.n_atoms();
        let mut charges = Vec::with_capacity(np);
        for atom in &molecule.atoms {
            let z = atomic_number(&atom.symbol).ok_or_else(|| {
                EngineError::Data(format!("cosmo: unknown element `{}`", atom.symbol))
            })?;
            charges.push(f64::from(z));
        }
        let mut omega = Vec::with_capacity(tesserae.len());
        let mut nuclear = Vec::with_capacity(tesserae.len());
        for tessera in tesserae {
            // 把一个 +1 点电荷放在碎片中心：核吸引积分正好是 ∫φφ/|r−r_K|
            let probe = Molecule::new(vec![Atom::new("H", tessera.center_angstrom)], 0, 1);
            let integral = one_electron_with_charges(basis, &probe, &[1.0])?;
            // `qwb-integrals` 的核吸引矩阵取的是**吸引**符号（`V = −Z∫φφ/r`），
            // 而这里要的是正的 `Ω_K = +∫φφ/|r−r_K|`，所以整体取反。
            let mut omega_k = integral.nuclear;
            for mu in 0..omega_k.nrows() {
                for nu in 0..omega_k.ncols() {
                    omega_k[(mu, nu)] = -omega_k[(mu, nu)];
                }
            }
            omega.push(omega_k);
            let mut potential = 0.0;
            let mut valid = true;
            for (index, atom) in molecule.atoms.iter().enumerate() {
                let dx = tessera.center_angstrom[0] - atom.position_angstrom[0];
                let dy = tessera.center_angstrom[1] - atom.position_angstrom[1];
                let dz = tessera.center_angstrom[2] - atom.position_angstrom[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt() * bohr;
                if distance <= 0.0 {
                    valid = false;
                    break;
                }
                potential += charges[index] / distance;
            }
            if !valid {
                return Err(EngineError::InvalidInput(
                    "cosmo: a tessera center coincides with a nucleus (SV-01)".into(),
                ));
            }
            nuclear.push(potential);
        }
        Ok(Self {
            tesserae: tesserae.to_vec(),
            omega,
            nuclear,
            epsilon,
            options: CosmoOptions::default(),
            n_functions: basis.n_functions(),
            last_solvation_energy: std::sync::Mutex::new(0.0),
            last_total_charge: std::sync::Mutex::new(0.0),
        })
    }

    /// 包成 SCF 用的句柄（`Arc` 共享，句柄拿不走的诊断值仍可读）。
    #[must_use]
    pub fn handle(self: &Arc<Self>) -> ExternalFieldHandle {
        ExternalFieldHandle::new(Arc::clone(self) as Arc<dyn ExternalField>)
    }

    /// 最近一次迭代的静电溶剂化能（Hartree）。
    #[must_use]
    pub fn last_solvation_energy(&self) -> f64 {
        *self.last_solvation_energy.lock().expect("cosmo field lock")
    }

    /// 最近一次迭代的表面电荷总和（e，应接近 −体系总电荷的电子部分）。
    #[must_use]
    pub fn last_total_charge(&self) -> f64 {
        *self.last_total_charge.lock().expect("cosmo field lock")
    }

    /// 表面电势 `V_K`（Hartree/e）。
    fn surface_potential(&self, density: &Matrix) -> Vec<f64> {
        let mut potentials = Vec::with_capacity(self.tesserae.len());
        for (index, omega) in self.omega.iter().enumerate() {
            let mut electronic = 0.0;
            for mu in 0..self.n_functions {
                for nu in 0..self.n_functions {
                    electronic += density[(mu, nu)] * omega[(mu, nu)];
                }
            }
            potentials.push(self.nuclear[index] - electronic);
        }
        potentials
    }

    /// 由密度算 (反应场算子, SCF 能量修正, ΔG_el)。
    fn field(&self, density: &Matrix) -> Result<(Matrix, f64, f64), EngineError> {
        if density.nrows() != self.n_functions || density.ncols() != self.n_functions {
            return Err(EngineError::Internal(format!(
                "cosmo: density is {}x{} for {} AO",
                density.nrows(),
                density.ncols(),
                self.n_functions
            )));
        }
        let potentials = self.surface_potential(density);
        let result =
            cosmo_surface_charges(&self.tesserae, &potentials, self.epsilon, self.options)?;
        if !result.converged {
            return Err(EngineError::Internal(format!(
                "cosmo: surface charge solve did not converge (residual {:.3e}, SV-01)",
                result.residual
            )));
        }
        // **符号**：`ΔG = ½ΣV_K q_K = −½ qᵀSq`（`Sq = −V`），对密度求导给出
        // `∂ΔG/∂P_μν = −V_solv,μν` —— 不是 `+V_solv`。判据是变分一致性：
        // 数值 `dΔG/dλ` 必须等于 `−Tr(V_solv·dP)`（见 tests/sv01_solvation_variational.rs）。
        let mut operator = Matrix::zeros(self.n_functions, self.n_functions);
        for (index, charge) in result.charges.iter().enumerate() {
            let omega = &self.omega[index];
            for mu in 0..self.n_functions {
                for nu in 0..self.n_functions {
                    operator[(mu, nu)] -= charge * omega[(mu, nu)];
                }
            }
        }
        if std::env::var_os("QWB_COSMO_TRACE").is_some() {
            let mut min_p = f64::INFINITY;
            let mut max_p = f64::NEG_INFINITY;
            let mut sum_p = 0.0;
            for (index, value) in potentials.iter().enumerate() {
                min_p = min_p.min(*value);
                max_p = max_p.max(*value);
                sum_p += value;
                if index < 3 {
                    let mut electronic = 0.0;
                    for mu in 0..self.n_functions {
                        for nu in 0..self.n_functions {
                            electronic += density[(mu, nu)] * self.omega[index][(mu, nu)];
                        }
                    }
                    eprintln!(
                        "cosmo V[{index}]: nuclear={:+.6} electronic={:+.6} total={:+.6} area={:.6}",
                        self.nuclear[index],
                        -electronic,
                        value,
                        self.tesserae[index].area_angstrom2
                    );
                }
            }
            let min_q = result.charges.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_q = result
                .charges
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "cosmo step: V in [{min_p:+.4},{max_p:+.4}] mean={:+.4}  q in [{min_q:+.4},{max_q:+.4}]  M={}",
                sum_p / potentials.len() as f64,
                potentials.len()
            );
        }
        // ΔG_el = ½ΣV_K q_K
        let mut solvation = 0.0;
        let mut total = 0.0;
        for (index, charge) in result.charges.iter().enumerate() {
            solvation += 0.5 * potentials[index] * charge;
            total += charge;
        }
        *self.last_solvation_energy.lock().expect("cosmo field lock") = solvation;
        *self.last_total_charge.lock().expect("cosmo field lock") = total;
        // Tr(P V_solv)
        let mut trace = 0.0;
        for mu in 0..self.n_functions {
            for nu in 0..self.n_functions {
                trace += density[(mu, nu)] * operator[(mu, nu)];
            }
        }
        Ok((operator, solvation - trace, solvation))
    }
}

impl ExternalField for CosmoScfField {
    fn operator_and_energy(&self, density: &Matrix) -> Result<(Matrix, f64), EngineError> {
        let (operator, shift, _) = self.field(density)?;
        Ok((operator, shift))
    }

    fn name(&self) -> &'static str {
        "cosmo"
    }
}
