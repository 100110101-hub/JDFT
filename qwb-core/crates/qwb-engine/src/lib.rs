//! 引擎顶层：实现 K2 的 [`EngineBackend`]，把注册表数据接到 SCF 上。
//!
//! 这一层是 T1 的最后一环：`JobInput` → 加载基组/泛函/网格 → RKS → `JobResult`
//! （含 `env` 与 `data_snapshot`，G5 要求）。
//!
//! # 已实装的作业类型（写清楚，免得误以为支持）
//!
//! - **`JobType::Sp`**：单点能（T1）。
//! - **`JobType::Opt`**：几何优化（T3 的 `opt::optimize` + T8 的接线）。梯度优先用解析
//!   梯度（HF 与 KS 都支持），算不出来才退数值梯度。
//! - **`JobType::Freq`**：**未接线**，显式拒绝。`hessian::numerical_hessian` 是有的
//!   （F-01/F-02 的测试直接调它），但它还没接到 `EngineBackend` 上；T8 的判据只要 `opt`。
//!
//! 其余维度：闭壳层 LDA/GGA/hybrid + vacuum；ECP 未接入（I-05），
//! `options.ecp_name = Some(_)` 显式拒绝；溶剂属 §8.5 的接口-only，非 vacuum 也拒绝。
//!
//! # 为什么 `opt` 的结果要经过一个"旁路"（[`OptSink`]）
//!
//! K2 的 `JobResult` 没有"优化后的几何"字段，而 K2 是冻结契约。优化后的几何属于
//! K3 目录里 `artifacts/**`（自由形态）的东西，所以引擎不往 `JobResult` 里塞，
//! 而是通过 [`OptSink`] 把 [`OptReport`] 交给调用方（CLI 拿它写
//! `artifacts/optimized.xyz`）。这也让"优化轨迹"能被写下来而不改契约。

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod cosmo_scf;
pub mod env;
pub mod gradient;
pub mod hessian;
pub mod opt;

use std::sync::{Arc, Mutex};

use qwb_engine_api::{
    CorrelationResult, EngineBackend, EngineError, JobInput, JobResult, JobType, Molecule, Mp2Kind,
    Reporter,
};

/// 优化过程的旁路产物（见模块文档：K2 的 `JobResult` 放不下几何）。
#[derive(Debug, Clone)]
pub struct OptReport {
    /// 优化后的几何。
    pub molecule: Molecule,
    /// 优化后的能量（Hartree）。
    pub energy_hartree: f64,
    /// 优化后的梯度（Ha/Bohr），形状 `[natoms][3]`。
    pub gradient: Vec<[f64; 3]>,
    /// 每一步的 `(能量, 最大力)`，第 0 项是初值。
    pub trajectory: Vec<(f64, f64)>,
    /// 实际走了几步。
    pub iterations: u32,
    /// 是否满足力判据。
    pub converged: bool,
    /// 是否全程用的解析梯度。
    pub used_analytic_gradient: bool,
    /// 退到数值梯度的原因（解析成功时为 `None`）。
    pub analytic_fallback: Option<String>,
}

/// 优化产物的交接点：引擎写、调用方（CLI）读。
///
/// 用 `Arc<Mutex<..>>` 而不是 `&mut self`：`EngineBackend::run` 是 `&self`（K2 冻结），
/// 而 `Engine::with_backend` 会把后端装箱搬走——调用方拿不回 `&mut`。
/// 这个槽只放一次结果，读完即空（`take`），不会跨作业串味。
#[derive(Debug, Clone, Default)]
pub struct OptSink {
    slot: Arc<Mutex<Option<OptReport>>>,
}

impl OptSink {
    /// 空槽。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 取走优化结果（没有 `opt` 作业跑过时为 `None`）。
    #[must_use]
    pub fn take(&self) -> Option<OptReport> {
        self.slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    fn store(&self, report: OptReport) {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *slot = Some(report);
    }
}

/// 闭壳层 RKS 后端。
pub struct RksBackend {
    registry: qwb_registry::Registry,
    opt_sink: OptSink,
}

impl RksBackend {
    /// 用已加载的数据注册表构造后端。
    #[must_use]
    pub fn new(registry: qwb_registry::Registry) -> Self {
        Self {
            registry,
            opt_sink: OptSink::new(),
        }
    }

    /// 挂一个优化结果的交接点（`opt` 作业用；不挂也没问题，结果就没人取）。
    #[must_use]
    pub fn with_opt_sink(mut self, sink: OptSink) -> Self {
        self.opt_sink = sink;
        self
    }

    /// 数据注册表（结果快照从它来）。
    #[must_use]
    pub fn registry(&self) -> &qwb_registry::Registry {
        &self.registry
    }
}

fn grid_level_from_api(level: qwb_engine_api::GridLevel) -> qwb_registry::grid::GridLevelName {
    match level {
        qwb_engine_api::GridLevel::Coarse => qwb_registry::grid::GridLevelName::Coarse,
        qwb_engine_api::GridLevel::Fine => qwb_registry::grid::GridLevelName::Fine,
        qwb_engine_api::GridLevel::VeryFine => qwb_registry::grid::GridLevelName::VeryFine,
        _ => qwb_registry::grid::GridLevelName::Normal,
    }
}

/// K2 的 `method_name` → 引擎内部的泛函名。
///
/// K2 规定 HF 用 `"hf"`，而 `qwb-data/functional/` 里**没有** `hf.yaml`
/// （HF 不是泛函数据，它是"没有 XC 项 + `α_hf = 1`"）。所以这里做唯一的翻译：
/// `"hf"`（大小写不敏感）→ 空名字，其余原样交给注册表。
fn functional_name_from_method(method: &str) -> &str {
    if method.eq_ignore_ascii_case("hf") {
        ""
    } else {
        method
    }
}

/// 泛函名 → `(网格上的能量密度项, α_hf)`；空名字 = HF。
///
/// 与 `qwb_scf::rks::rks_from_data` 同一套规则：`HfExchange` 不是能量密度项，
/// 而是 hybrid 的混入比例。对外公开是因为"自己搭 Fock/自己算梯度"的调用方
/// （测试、T6 的 B3LYP 组合、`opt` 的解析梯度路径）必须用同一套规则，
/// 各写一遍就会有一处把 `α_hf` 漏掉。
///
/// # Errors
///
/// 泛函名不存在，或某个分量的密度项/参数不被 `qwb-xc` 支持。
pub fn terms_and_hf_exchange(
    registry: &qwb_registry::Registry,
    functional_name: &str,
) -> Result<(Vec<Box<dyn qwb_xc::FunctionalTerm>>, f64), EngineError> {
    if functional_name.is_empty() {
        return Ok((Vec::new(), 1.0));
    }
    let functional = registry.get_functional(functional_name)?;
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    let mut hf_exchange = 0.0;
    for component in &functional.components {
        if component.id == "HfExchange" {
            hf_exchange += component.weight;
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &functional.parameters)
            .map_err(|err| EngineError::Data(format!("functional `{}`: {err}", functional.name)))?;
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    Ok((terms, hf_exchange))
}

/// FX-01 的一致性守卫：**数据声明的 `requires.kinetic_density` 必须与实现一致**。
///
/// 两个方向都要拒：
/// * 数据说需要 τ、但没有任何组件真的读 τ → 用户会拿到一个 GGA 水平的答案，
///   却以为算了 meta-GGA（§10.3 禁止的静默降级）；
/// * 组件读 τ、数据却没声明 → 数据对自己的需求说谎，`qwb data verify` 的语义就废了。
///
/// # Errors
///
/// 不一致时返回 [`EngineError::Unsupported`]，消息里点明是哪一侧。
pub fn check_meta_gga_declaration(
    functional_name: &str,
    data_requires_tau: bool,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
) -> Result<(), EngineError> {
    let implementation_reads_tau = qwb_xc::terms_require_tau(terms);
    if data_requires_tau && !implementation_reads_tau {
        return Err(EngineError::Unsupported(format!(
            "functional `{functional_name}` declares requires.kinetic_density = true, but none \
             of its components reads the kinetic energy density τ; refusing to compute a \
             GGA-level answer silently (FX-01/DA-01)"
        )));
    }
    if implementation_reads_tau && !data_requires_tau {
        return Err(EngineError::Unsupported(format!(
            "functional `{functional_name}` has a component that reads τ, but its data does \
             not declare requires.kinetic_density = true; fix the data (FX-01/DA-01)"
        )));
    }
    Ok(())
}

fn density_to_api(matrix: &qwb_linalg::Matrix) -> nalgebra::DMatrix<f64> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    let mut data = Vec::with_capacity(rows * columns);
    for i in 0..rows {
        for j in 0..columns {
            data.push(matrix[(i, j)]);
        }
    }
    nalgebra::DMatrix::from_row_slice(rows, columns, &data)
}

/// 解析后的"作业规格"：泛函名（空 = HF）、网格注册名、α_hf、密度项。
struct JobSpec {
    functional_name: String,
    grid_name: String,
    terms: Vec<Box<dyn qwb_xc::FunctionalTerm>>,
    hf_exchange: f64,
}

impl RksBackend {
    fn spec(&self, input: &JobInput) -> Result<JobSpec, EngineError> {
        let functional_name = functional_name_from_method(&input.method_name).to_string();
        let grid_name = self
            .registry
            .grid_name_by_level(grid_level_from_api(input.grid_level))?
            .to_string();
        let (terms, hf_exchange) = terms_and_hf_exchange(&self.registry, &functional_name)?;
        Ok(JobSpec {
            functional_name,
            grid_name,
            terms,
            hf_exchange,
        })
    }

    /// ECP 的两样东西：每个原子对电子的核电荷（ECP 原子用 `z_eff`）与 ECP 矩阵元。
    ///
    /// 求积用同一个网格等级（`grid_name`）——ECP 的求积与 XC 的求积用同一套参数，
    /// 不在代码里塞点数（I-05 的精度见 `qwb-scf/tests/i05_ecp.rs`：解析 vs 半数值 3e-14，
    /// 网格加密 6e-14）。
    fn ecp_terms(
        &self,
        input: &JobInput,
        basis: &qwb_basis::InternalBasis,
        grid_name: &str,
        ecp_name: &str,
    ) -> Result<(Vec<f64>, qwb_linalg::Matrix), EngineError> {
        let ecp = self.registry.get_ecp(ecp_name)?;
        let operator = qwb_scf::ecp::EcpOperator::from_ecp(ecp, &input.molecule)?;
        let mut charges = Vec::with_capacity(input.molecule.n_atoms());
        for atom in &input.molecule.atoms {
            let z = qwb_registry::elements::atomic_number(&atom.symbol).ok_or_else(|| {
                EngineError::Internal(format!("unknown element `{}`", atom.symbol))
            })?;
            charges.push(if atom.symbol == operator.element {
                operator.z_eff
            } else {
                f64::from(z)
            });
        }
        let level = self.registry.get_grid(grid_name)?;
        let potential = qwb_scf::ecp::matrix_elements(basis, &operator, level)?;
        Ok((charges, potential))
    }

    /// 单点能（**UKS**，阶段二 SC-01b / M1b）：自旋极化 DFT。
    ///
    /// 与 UHF 路径的差别：Fock 的交换项换成自旋分辨的 XC 势（`qwb_scf::uks`），
    /// 结果里除 K9 的 `spin` 块之外还要报 `E_xc`（KS 能量的分解里它是独立的一项）。
    ///
    /// # Errors
    ///
    /// 泛函没有自旋实现（B88/LYP）、非 `sp` 作业类型、SCF 不收敛。
    fn run_uks_single_point(
        &self,
        input: &JobInput,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        if input.jobtype != JobType::Sp {
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `uks` currently supports `sp` only (open-shell gradients \
                 are phase-2 SC-02); jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }
        let basis_k1 = self.registry.get_basis(&input.basis_name)?;
        let spec = self.spec(input)?;
        if spec.terms.is_empty() {
            return Err(EngineError::Unsupported(
                "options.scf_type = `uks` needs a density functional; for Hartree-Fock use \
                 scf_type = `uhf` (SC-01)"
                    .to_string(),
            ));
        }
        let data_requires_tau = self
            .registry
            .get_functional(&spec.functional_name)?
            .requires
            .kinetic_density;
        check_meta_gga_declaration(&spec.functional_name, data_requires_tau, &spec.terms)?;

        reporter.progress("load", 0.0);
        reporter.log(&format!(
            "scf_type = uks (unrestricted Kohn-Sham), method = {}, basis = {} ({} elements), \
             grid = {}, {} atoms, multiplicity {}",
            spec.functional_name,
            basis_k1.name,
            basis_k1.elements.len(),
            spec.grid_name,
            input.molecule.n_atoms(),
            input.molecule.multiplicity
        ));

        reporter.progress("integrals", 25.0);
        let basis = qwb_basis::InternalBasis::build(basis_k1, &input.molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, &input.molecule)?;
        let grid =
            qwb_grid::molecular_grid(self.registry.get_grid(&spec.grid_name)?, &input.molecule)
                .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        reporter.log(&format!("grid points = {}", grid.len()));

        let mut options = qwb_scf::uks::UksOptions::default();
        options.max_iter = input.options.scf_max_iter;
        options.energy_threshold = input.options.scf_energy_threshold_hartree;
        options.density_threshold = input.options.scf_density_threshold;
        options.diis_max_vectors = input.options.diis_history as usize;
        options.hf_exchange = spec.hf_exchange;
        if let Some(ecp_name) = &input.options.ecp_name {
            let (charges, potential) = self.ecp_terms(input, &basis, &spec.grid_name, ecp_name)?;
            options.nuclear_charges = Some(charges);
            options.ecp_potential = Some(potential);
        }

        reporter.progress("scf", 40.0);
        let result = qwb_scf::uks::uks_scf(
            &basis,
            &input.molecule,
            &tensor,
            &grid,
            &spec.terms,
            options,
        )?;
        reporter.log(&format!(
            "uks {} in {} iterations: E = {:.10} Ha, E_xc = {:.10} Ha, <S^2> = {:.6}, \
             contamination = {:+.3e}",
            if result.scf_converged {
                "converged"
            } else {
                "NOT converged"
            },
            result.iterations,
            result.energy_hartree,
            result.xc_energy_hartree,
            result.s_squared,
            result.spin_contamination
        ));
        reporter.log(&format!(
            "uks spin: n_alpha = {}, n_beta = {}, beta HOMO = {:.6} Ha, beta LUMO = {:.6} Ha",
            result.n_alpha,
            result.n_beta,
            result.mo_energies_beta.get(result.n_beta.saturating_sub(1)),
            result.mo_energies_beta.get(result.n_beta),
        ));
        if !result.scf_converged {
            return Err(EngineError::ScfNotConverged {
                iters: result.iterations,
                max: input.options.scf_max_iter,
            });
        }
        reporter.progress("done", 100.0);

        let env = env::build(&self.registry);
        let dipole = qwb_props::dipole_debye(&basis, &input.molecule, &result.density)?;
        let spin = qwb_engine_api::SpinResult::new(
            result.n_alpha as u32,
            result.n_beta as u32,
            input.molecule.multiplicity,
            result.s_squared,
        );
        Ok(JobResult::new(
            result.energy_hartree,
            result.scf_converged,
            self.registry.snapshot(),
            env,
        )
        .with_dipole(nalgebra::DVector::from_column_slice(&dipole))
        .with_orbitals(
            nalgebra::DVector::from_column_slice(result.mo_energies_alpha.as_slice()),
            density_to_api(&result.coefficients_alpha),
        )
        .with_spin(spin)
        .with_log(format!(
            "scf_type = uks, method = {}, basis = {}, grid = {}, iterations = {}, \
             E_elec = {:.10}, E_xc = {:.10}, <S^2> = {:.6} (contamination {:+.3e}), \
             n_alpha = {}, n_beta = {}, dipole = [{:.4}, {:.4}, {:.4}] D. \
             mo_energies_hartree / mo_coeffs are the ALPHA set.",
            spec.functional_name,
            basis_k1.name,
            spec.grid_name,
            result.iterations,
            result.electronic_hartree,
            result.xc_energy_hartree,
            result.s_squared,
            result.spin_contamination,
            result.n_alpha,
            result.n_beta,
            dipole[0],
            dipole[1],
            dipole[2]
        )))
    }

    /// 单点能（**UHF**，阶段二 SC-01）：开壳层路径。
    ///
    /// 与闭壳层路径的三个差别：① 走 [`qwb_scf::uhf::uhf_scf`]（双套密度、⟨S²⟩）；
    /// ② 结果里带 K9 的 `spin` 块（自旋污染必须报出来）；③ 轨道填**α** 的那一套
    ///（K2 的 `mo_energies_hartree` / `mo_coeffs` 是单套字段，β 的轨道能写进日志）。
    fn run_uhf_single_point(
        &self,
        input: &JobInput,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        let basis_k1 = self.registry.get_basis(&input.basis_name)?;
        let spec = self.spec(input)?;
        if !spec.terms.is_empty() {
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `uhf` is Hartree-Fock only, but method `{}` is a \
                 density functional; unrestricted KS (UKS) is phase-2 SC-01b",
                input.method_name
            )));
        }
        if input.jobtype != JobType::Sp {
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `uhf` currently supports `sp` only (gradients for open                  shells are SC-02); jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }

        reporter.progress("load", 0.0);
        reporter.log(&format!(
            "scf_type = uhf (Hartree-Fock), basis = {} ({} elements), {} atoms, multiplicity {}",
            basis_k1.name,
            basis_k1.elements.len(),
            input.molecule.n_atoms(),
            input.molecule.multiplicity
        ));

        reporter.progress("integrals", 25.0);
        let basis = qwb_basis::InternalBasis::build(basis_k1, &input.molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, &input.molecule)?;

        let mut options = qwb_scf::uhf::UhfOptions::default();
        options.max_iter = input.options.scf_max_iter;
        options.energy_threshold = input.options.scf_energy_threshold_hartree;
        options.density_threshold = input.options.scf_density_threshold;
        options.diis_max_vectors = input.options.diis_history as usize;
        if let Some(ecp_name) = &input.options.ecp_name {
            let (charges, potential) = self.ecp_terms(input, &basis, &spec.grid_name, ecp_name)?;
            options.nuclear_charges = Some(charges);
            options.ecp_potential = Some(potential);
        }

        reporter.progress("scf", 40.0);
        let result = qwb_scf::uhf::uhf_scf(&basis, &input.molecule, &tensor, options)?;
        reporter.log(&format!(
            "uhf {} in {} iterations: E = {:.10} Ha, <S^2> = {:.6}, contamination = {:+.3e}",
            if result.scf_converged {
                "converged"
            } else {
                "NOT converged"
            },
            result.iterations,
            result.energy_hartree,
            result.s_squared,
            result.spin_contamination
        ));
        reporter.log(&format!(
            "uhf spin: n_alpha = {}, n_beta = {}, beta HOMO = {:.6} Ha, beta LUMO = {:.6} Ha",
            result.n_alpha,
            result.n_beta,
            result.mo_energies_beta.get(result.n_beta.saturating_sub(1)),
            result.mo_energies_beta.get(result.n_beta),
        ));
        if !result.scf_converged {
            return Err(EngineError::ScfNotConverged {
                iters: result.iterations,
                max: input.options.scf_max_iter,
            });
        }
        reporter.progress("done", 100.0);

        let env = env::build(&self.registry);
        let dipole = qwb_props::dipole_debye(&basis, &input.molecule, &result.density)?;
        let spin = qwb_engine_api::SpinResult::new(
            result.n_alpha as u32,
            result.n_beta as u32,
            input.molecule.multiplicity,
            result.s_squared,
        );
        Ok(JobResult::new(
            result.energy_hartree,
            result.scf_converged,
            self.registry.snapshot(),
            env,
        )
        .with_dipole(nalgebra::DVector::from_column_slice(&dipole))
        .with_orbitals(
            nalgebra::DVector::from_column_slice(result.mo_energies_alpha.as_slice()),
            density_to_api(&result.coefficients_alpha),
        )
        .with_spin(spin)
        .with_log(format!(
            "scf_type = uhf, basis = {}, iterations = {}, E_elec = {:.10}, \
             <S^2> = {:.6} (contamination {:+.3e}), n_alpha = {}, n_beta = {}, \
             dipole = [{:.4}, {:.4}, {:.4}] D. mo_energies_hartree / mo_coeffs are the ALPHA set.",
            basis_k1.name,
            result.iterations,
            result.electronic_hartree,
            result.s_squared,
            result.spin_contamination,
            result.n_alpha,
            result.n_beta,
            dipole[0],
            dipole[1],
            dipole[2]
        )))
    }

    /// 单点能（`Sp` 的全部逻辑，也是 `opt` 收尾时复用的那一段）。
    fn run_single_point(
        &self,
        input: &JobInput,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        // 阶段二 M1/M1b：UHF/UKS 走独立路径（自旋通道 + ⟨S²⟩），
        // 与闭壳层 RKS 的数值链不同。
        match input.options.scf_type {
            Some(qwb_engine_api::ScfType::Uhf) => {
                return self.run_uhf_single_point(input, reporter);
            }
            Some(qwb_engine_api::ScfType::Uks) => {
                return self.run_uks_single_point(input, reporter);
            }
            _ => {}
        }
        reporter.progress("load", 0.0);
        let basis_k1 = self.registry.get_basis(&input.basis_name)?;
        let spec = self.spec(input)?;
        // FX-01：数据声明的 τ 需求必须与实现一致（meta-GGA 的前置门禁）。
        if !spec.functional_name.is_empty() {
            let data_requires_tau = self
                .registry
                .get_functional(&spec.functional_name)?
                .requires
                .kinetic_density;
            check_meta_gga_declaration(&spec.functional_name, data_requires_tau, &spec.terms)?;
        }
        reporter.log(&format!(
            "basis = {} ({} elements), method = {}, grid = {}, jobtype = {}, {} atoms",
            basis_k1.name,
            basis_k1.elements.len(),
            if spec.functional_name.is_empty() {
                "hf".to_string()
            } else {
                spec.functional_name.clone()
            },
            spec.grid_name,
            input.jobtype.as_str(),
            input.molecule.n_atoms()
        ));

        reporter.progress("integrals", 25.0);
        let basis = qwb_basis::InternalBasis::build(basis_k1, &input.molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, &input.molecule)?;
        let grid =
            qwb_grid::molecular_grid(self.registry.get_grid(&spec.grid_name)?, &input.molecule)
                .map_err(|err| EngineError::Data(format!("grid: {err}")))?;

        reporter.progress("scf", 40.0);
        reporter.log(&format!("grid points = {}", grid.len()));

        // `RksOptions` 是 non_exhaustive：先 Default 再按字段覆盖。
        let mut options = qwb_scf::rks::RksOptions::default();
        options.max_iter = input.options.scf_max_iter;
        options.energy_threshold = input.options.scf_energy_threshold_hartree;
        options.density_threshold = input.options.scf_density_threshold;
        options.hf_exchange = spec.hf_exchange;
        // K2 的 `TaskOptions.diis_history` 真的接到 DIIS 子空间长度上（之前漏掉了）。
        options.diis_max_vectors = input.options.diis_history as usize;

        // I-05：ECP。核吸引用 z_eff，超出的部分作为矩阵元加进 H_core。
        if let Some(ecp_name) = &input.options.ecp_name {
            let (charges, potential) = self.ecp_terms(input, &basis, &spec.grid_name, ecp_name)?;
            reporter.log(&format!(
                "ecp = {ecp_name} on {} (n_core = {}, z_eff = {:.0}); \
                 ECP matrix elements added to H_core",
                self.registry.get_ecp(ecp_name)?.element,
                self.registry.get_ecp(ecp_name)?.n_core_electrons,
                self.registry.get_ecp(ecp_name)?.z_eff
            ));
            options.nuclear_charges = Some(charges);
            options.ecp_potential = Some(potential);
        }

        // SCF 的三级退路（近简并体系：N2、含 d/f 的重元素、ECP 体系）。全部**确定**：
        // 初猜序列/温度/步数都固定，没有随机化 ⇒ 同一输入给同一输出。
        //
        // 顺序是试出来的，不是猜的（2026-10-26 的实测）：
        //  * **多初猜在前**：苯/STO-3G/PBE 的单初猜 300 步不收敛，但 `initial_guess_scale`
        //    = 0.5 / 1.5 分别 40 / 15 步收敛到 -229.1170417 Ha（CCCBDB -229.117005，差
        //    0.04 mHa）。每次尝试给 `min(scf_max_iter, 60)` 步的预算，别把一次退路变成
        //    五分钟的等待。
        //  * **退火在后**：ZnO/LANL2DZ 只有它管用（多初猜 5 个都失败，退火 23 步收敛）。
        //    但退火**可能停在错误的解上**：同一套参数下苯/STO-3G/PBE 退火会收敛到
        //    -209.9115 Ha（比 CCCBDB 高 19 Ha 的伪态）。所以退火之后再做一次**有界**
        //    多初猜交叉验证，两者都收敛时取**能量更低**的那个 —— 这一条正是苯的教训。
        //  * **阻尼 Roothaan 兜底**：前面都不收敛时最后试一次（关 DIIS、密度阻尼 0.3）。
        //
        // `scf_max_iter` 是 K2 的输入契约：退路只允许**少于**它的步数，绝不允许偷偷加步
        //（`test_scf_not_converged_carries_iters` 钉的就是这一点）。
        const RETRY_MAX_ITER: u32 = 60;
        let retry_budget = options.max_iter.min(RETRY_MAX_ITER);
        let attempt_multistart = |label: &str| -> Option<qwb_scf::rks::RksResult> {
            let mut retry = options.clone();
            retry.max_iter = retry_budget;
            match qwb_scf::rks::rks_scf_multistart(
                &basis,
                &input.molecule,
                &tensor,
                &grid,
                &spec.terms,
                retry,
            ) {
                Ok(result) => {
                    reporter.log(&format!(
                        "scf: {label} multistart converged in {} iterations: E = {:.10} Ha",
                        result.iterations, result.energy_hartree
                    ));
                    Some(result)
                }
                Err(err) => {
                    reporter.log(&format!("scf: {label} multistart failed ({err})"));
                    None
                }
            }
        };

        let result = match qwb_scf::rks::rks_scf(
            &basis,
            &input.molecule,
            &tensor,
            &grid,
            &spec.terms,
            options.clone(),
        ) {
            Ok(result) => result,
            Err(first_error) => {
                reporter.log(&format!(
                    "scf: single guess failed ({first_error}), retrying with multistart"
                ));
                match attempt_multistart("retry") {
                    Some(result) => result,
                    None => {
                        reporter.log(
                            "scf: retrying with annealed smearing (Fermi broadening, then \
                             integer occupations)",
                        );
                        let mut annealed = options.clone();
                        annealed.smearing_steps = 20;
                        annealed.smearing_temperature = 0.3;
                        match qwb_scf::rks::rks_scf(
                            &basis,
                            &input.molecule,
                            &tensor,
                            &grid,
                            &spec.terms,
                            annealed,
                        ) {
                            Ok(annealed_result) => {
                                reporter.log(&format!(
                                    "scf: annealing converged in {} iterations: E = {:.10} Ha",
                                    annealed_result.iterations, annealed_result.energy_hartree
                                ));
                                // 退火可能停在伪态上（苯/STO-3G/PBE 实测高 19 Ha）：
                                // 再跑一次有界多初猜，取更低的那个。
                                match attempt_multistart("annealing cross-check") {
                                    Some(better)
                                        if better.energy_hartree
                                            < annealed_result.energy_hartree - 1.0e-9 =>
                                    {
                                        reporter.log(
                                            "scf: multistart found a lower state than annealing; \
                                             using it (annealing may stop in a spurious branch)",
                                        );
                                        better
                                    }
                                    _ => annealed_result,
                                }
                            }
                            Err(second_error) => {
                                reporter.log(&format!(
                                    "scf: annealing failed ({second_error}), retrying with damped \
                                     Roothaan iterations (no DIIS, damping 0.3)"
                                ));
                                let mut plain = options.clone();
                                plain.use_diis = false;
                                plain.damping = 0.3;
                                qwb_scf::rks::rks_scf(
                                    &basis,
                                    &input.molecule,
                                    &tensor,
                                    &grid,
                                    &spec.terms,
                                    plain,
                                )?
                            }
                        }
                    }
                }
            }
        };
        reporter.log(&format!(
            "scf {} in {} iterations: E = {:.10} Ha, E_xc = {:.10} Ha",
            if result.scf_converged {
                "converged"
            } else {
                "NOT converged"
            },
            result.iterations,
            result.energy_hartree,
            result.exchange_correlation_hartree
        ));
        if !result.scf_converged {
            return Err(EngineError::ScfNotConverged {
                iters: result.iterations,
                max: input.options.scf_max_iter,
            });
        }
        // ── M3 / SC-04：canonical MP2（闭壳层参考、单点）─────────────
        //
        // 参考态就是上面这一遍 SCF（**同一个**基组与**同一份** 2e 张量），所以 MP2
        // 的分母与 `(ia|jb)` 都是自洽的；相关能进 K2 的 `correlation` 块 ——
        // SC-04 的第三条验收就是「`correlation` 块进 `result.json`」。
        let correlation = match &input.options.mp2 {
            Some(mp2) if mp2.kind == Mp2Kind::Mp2 => {
                reporter.progress("mp2", 90.0);
                let n_occupied = closed_shell_electrons(&input.molecule)? / 2;
                let orbital_energies: Vec<f64> = (0..result.mo_energies_hartree.len())
                    .map(|i| result.mo_energies_hartree.get(i))
                    .collect();
                let outcome = qwb_scf::mp2::mp2_correlation(
                    &tensor,
                    &result.coefficients,
                    &orbital_energies,
                    n_occupied,
                )?;
                reporter.log(&format!(
                    "mp2: E_corr = {:.10} Ha, n_occ = {}, n_vir = {}, \
                     largest pair contribution = {:.6} Ha",
                    outcome.e2_hartree,
                    outcome.n_occupied,
                    outcome.n_virtual,
                    outcome.dominant_pair_energy
                ));
                Some(CorrelationResult::new(
                    Mp2Kind::Mp2,
                    outcome.e2_hartree,
                    result.energy_hartree + outcome.e2_hartree,
                ))
            }
            // ── M3 / SC-05：RI-MP2 ────────────────────────────────
            Some(mp2) if mp2.kind == Mp2Kind::RiMp2 => {
                reporter.progress("mp2", 85.0);
                let aux_name = mp2.aux_basis_name.as_deref().ok_or_else(|| {
                    EngineError::InvalidInput(
                        "options.mp2.aux_basis_name is required for ri-mp2".into(),
                    )
                })?;
                let n_occupied = closed_shell_electrons(&input.molecule)? / 2;
                let orbital_energies: Vec<f64> = (0..result.mo_energies_hartree.len())
                    .map(|i| result.mo_energies_hartree.get(i))
                    .collect();
                let aux_basis =
                    qwb_integrals::ri::load_aux_basis(&self.registry, aux_name, &input.molecule)?;
                let three = qwb_integrals::ri::three_center(&basis, &aux_basis, &input.molecule)?;
                let v = qwb_integrals::ri::two_center_coulomb(&aux_basis, &input.molecule)?;
                let outcome = qwb_scf::rimp2::ri_mp2_correlation(
                    &three,
                    &v,
                    &result.coefficients,
                    &orbital_energies,
                    n_occupied,
                )?;
                // K9/K10：近似必须留痕 —— 用同一份 4 中心张量再算一遍 canonical，
                // 把差值写进 `ri_error_hartree`。这一步是 `kind: ri-mp2` 的一次作业
                // 比纯 RI 慢的原因（见 docs/choices.md 五十八）。
                reporter.progress("mp2", 95.0);
                let canonical = qwb_scf::mp2::mp2_correlation(
                    &tensor,
                    &result.coefficients,
                    &orbital_energies,
                    n_occupied,
                )?;
                let ri_error = (outcome.e2_hartree - canonical.e2_hartree).abs();
                reporter.log(&format!(
                    "ri-mp2: aux = {aux_name} ({} 个辅助函数，V 的秩 {}), \
                     E_corr = {:.10} Ha vs canonical {:.10} Ha, \
                     ri_error = {ri_error:.3e} Ha",
                    outcome.n_auxiliary, outcome.v_rank, outcome.e2_hartree, canonical.e2_hartree
                ));
                Some(
                    CorrelationResult::new(
                        Mp2Kind::RiMp2,
                        outcome.e2_hartree,
                        result.energy_hartree + outcome.e2_hartree,
                    )
                    .with_aux_basis(aux_name)
                    .with_ri_error(ri_error),
                )
            }
            _ => None,
        };
        reporter.progress("done", 100.0);

        let env = env::build(&self.registry);
        // S-06：偶极矩（Debye）——中性分子对原点不变，`qwb-props` 的测试里有这条自检。
        let dipole = qwb_props::dipole_debye(&basis, &input.molecule, &result.density)?;
        // ── M4 / RS-02..03：TDA（hf 参考下即 CIS）──────────────
        let excited_states = match &input.options.excited_states {
            Some(request) => {
                reporter.progress("tda", 92.0);
                let n_occupied = closed_shell_electrons(&input.molecule)? / 2;
                let orbital_energies: Vec<f64> = (0..result.mo_energies_hartree.len())
                    .map(|i| result.mo_energies_hartree.get(i))
                    .collect();
                let dipole_ao = qwb_integrals::dipole::dipole_integrals(&basis, &input.molecule)?;
                let cphf_input = qwb_response::mo_integrals::build_cphf_integrals(
                    &tensor,
                    &result.coefficients,
                    &dipole_ao,
                    n_occupied,
                )?;
                let tda = qwb_response::tda::tda_singlet(
                    &cphf_input,
                    &orbital_energies,
                    request.n_states as usize,
                )?;
                let transitions: Vec<qwb_engine_api::Transition> = tda
                    .excitations
                    .iter()
                    .enumerate()
                    .map(|(index, state)| {
                        qwb_engine_api::Transition::new(
                            index + 1,
                            state.energy_hartree,
                            1_239.841_984 / (state.energy_hartree * 27.211_386_245_988),
                            state.oscillator_strength,
                        )
                    })
                    .collect();
                let n_electrons = f64::from(u32::try_from(n_occupied * 2).unwrap_or(u32::MAX));
                reporter.log(&format!(
                    "tda: {} state(s) from a {}x{} problem; first = {:.6} Ha ({:.3} eV), \
                     f = {:.5}; sum of f = {:.6} (N_elec = {n_electrons})",
                    transitions.len(),
                    tda.dimension,
                    tda.dimension,
                    tda.excitations.first().map_or(0.0, |s| s.energy_hartree),
                    tda.excitations
                        .first()
                        .map_or(0.0, |s| s.energy_hartree * 27.211_386_245_988),
                    tda.excitations
                        .first()
                        .map_or(0.0, |s| s.oscillator_strength),
                    tda.oscillator_sum
                ));
                Some(qwb_engine_api::ExcitedStates::new(
                    request.method,
                    transitions.len(),
                    transitions,
                    qwb_engine_api::SumRules::new(tda.oscillator_sum, n_electrons),
                ))
            }
            None => None,
        };
        let mut job = JobResult::new(
            result.energy_hartree,
            result.scf_converged,
            self.registry.snapshot(),
            env,
        )
        .with_dipole(nalgebra::DVector::from_column_slice(&dipole))
        .with_orbitals(
            nalgebra::DVector::from_column_slice(result.mo_energies_hartree.as_slice()),
            // K2 的 `mo_coeffs` 是**系数矩阵**（列 = MO）。
            density_to_api(&result.coefficients),
        )
        .with_log(format!(
            "method = {}, basis = {}, grid = {}, iterations = {}, E_elec = {:.10}, E_xc = {:.10}, dipole = [{:.4}, {:.4}, {:.4}] D",
            if spec.functional_name.is_empty() {
                "hf".to_string()
            } else {
                spec.functional_name.clone()
            },
            basis_k1.name,
            spec.grid_name,
            result.iterations,
            result.electronic_hartree,
            result.exchange_correlation_hartree,
            dipole[0],
            dipole[1],
            dipole[2]
        ));
        if let Some(excited_states) = excited_states {
            job = job.with_excited_states(excited_states);
        }
        if let Some(correlation) = correlation {
            job = job
                .with_log(format!(
                    "mp2 = {}, E_corr = {:.10} Ha, E_total = {:.10} Ha",
                    correlation.method.as_str(),
                    correlation.e2_hartree,
                    correlation.e_total_hartree
                ))
                .with_correlation(correlation);
        }
        Ok(job)
    }

    /// 几何优化（`Opt`）：BFGS + 解析梯度，收尾再跑一次单点拿轨道/偶极。
    fn run_optimisation(
        &self,
        input: &JobInput,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        if let Some(ecp_name) = &input.options.ecp_name {
            // I-05 只做了 ECP 的**矩阵元**（能量）；ECP 对核坐标的导数还没实现，
            // 所以带 ECP 的几何优化会给出没有 ECP 导数项的梯度 —— 那是错的，显式拒绝。
            return Err(EngineError::Unsupported(format!(
                "geometry optimisation with an ECP (`{ecp_name}`) is not implemented yet: \
                 the ECP derivative w.r.t. nuclear coordinates is missing (I-05 covers the \
                 matrix elements only). Single-point jobs with ECP are supported"
            )));
        }
        let spec = self.spec(input)?;
        let mut options = opt::OptOptions::new(
            &input.basis_name,
            spec.functional_name.clone(),
            spec.grid_name.clone(),
        );
        options.max_iterations = input.options.max_opt_steps;
        options.force_threshold = input.options.opt_gradient_threshold_hartree_per_bohr;
        // 优化里每一步 SCF 也遵守输入的 SCF 阈值/步数上限（K2 的 `TaskOptions`）。
        options.scf_max_iter = input.options.scf_max_iter;
        options.scf_energy_threshold = input.options.scf_energy_threshold_hartree;
        options.scf_density_threshold = input.options.scf_density_threshold;

        reporter.progress("load", 0.0);
        reporter.log(&format!(
            "jobtype = opt, method = {}, basis = {}, grid = {}, max steps = {}, \
             force threshold = {:.1e} Ha/Bohr",
            if spec.functional_name.is_empty() {
                "hf".to_string()
            } else {
                spec.functional_name.clone()
            },
            input.basis_name,
            spec.grid_name,
            options.max_iterations,
            options.force_threshold
        ));

        let max_steps = f64::from(options.max_iterations.max(1));
        let mut on_step = |step: &opt::OptStep| -> bool {
            reporter.log(&format!(
                "opt step {:>3}: E = {:.10} Ha, max|grad| = {:.3e} Ha/Bohr",
                step.iteration, step.energy_hartree, step.max_force
            ));
            let pct = 40.0 + 60.0 * (f64::from(step.iteration) / max_steps).min(1.0);
            reporter.progress("opt", pct);
            !reporter.is_cancelled()
        };

        let outcome =
            opt::optimize_with_progress(&self.registry, &input.molecule, &options, &mut on_step)?;

        // 旁路产物先落定：即使下面判定"没收敛"，CLI 也要能把轨迹和最后一帧几何写出来。
        self.opt_sink.store(OptReport {
            molecule: outcome.molecule.clone(),
            energy_hartree: outcome.energy_hartree,
            gradient: outcome.gradient.clone(),
            trajectory: outcome.trajectory.clone(),
            iterations: outcome.iterations,
            converged: outcome.converged,
            used_analytic_gradient: outcome.used_analytic_gradient,
            analytic_fallback: outcome.analytic_fallback.clone(),
        });

        if let Some(reason) = &outcome.analytic_fallback {
            reporter.log(&format!(
                "warning: analytic gradient unavailable ({reason}); fell back to numerical \
                 gradients, this run is much slower"
            ));
        }

        if !outcome.converged {
            // K2 没有"优化未收敛"的变体，且 K1–K6 已冻结：用 ScfNotConverged 的**退出码
            // 语义**（K5 的 2 = 收敛失败）如实报告。`iters/max` 是优化步数。
            // 断言与记账见 docs/choices.md 的四十节。
            return Err(EngineError::ScfNotConverged {
                iters: outcome.iterations,
                max: options.max_iterations,
            });
        }

        // 收尾单点：`JobResult` 要的是**优化后几何**上的轨道/偶极，不是初值的。
        // 不改 `jobtype`：日志里要能看出这是**优化作业**里的一次 SCF。
        let mut final_input = input.clone();
        final_input.molecule = outcome.molecule.clone();
        let sp = self.run_single_point(&final_input, reporter)?;
        // K2 的 `gradient_hartree_per_bohr` 是 3N×1（按原子、再按 x/y/z 摊平）。
        let mut flat = Vec::with_capacity(outcome.gradient.len() * 3);
        for row in &outcome.gradient {
            flat.extend_from_slice(row);
        }
        let gradient = nalgebra::DMatrix::from_column_slice(flat.len(), 1, &flat);
        reporter.log(&format!(
            "optimisation converged in {} steps: E = {:.10} Ha, max|grad| = {:.3e} Ha/Bohr \
             (gradient: {})",
            outcome.iterations,
            outcome.energy_hartree,
            outcome
                .gradient
                .iter()
                .flat_map(|row| row.iter())
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
            if outcome.used_analytic_gradient {
                "analytic"
            } else {
                "numerical"
            }
        ));
        Ok(sp.with_gradient(gradient))
    }
}

impl EngineBackend for RksBackend {
    fn run(&self, input: &JobInput, reporter: &dyn Reporter) -> Result<JobResult, EngineError> {
        // 阶段二（project-phase2.md §4.1）：契约 crate 已经能**表达**这些选项，
        // 但引擎还没实现。这里必须显式拒绝 —— 静默忽略会让用户拿一个"看起来成功"
        // 的闭壳层结果去当他想要的 UKS/MP2/溶剂化结果（那正是 §10.3 禁止的事）。
        reject_unimplemented_options(input)?;
        if input.solvent != "vacuum" {
            return Err(EngineError::Unsupported(format!(
                "solvent `{}` is not implemented (first phase: vacuum only, 8.5)",
                input.solvent
            )));
        }
        // UHF 只有单点：`opt` 会在 `run_optimisation` 里白跑一遍闭壳层优化才发现不兼容，
        // 所以把拒绝提前到分派处（梯度是 SC-02 的事）。
        if matches!(
            input.options.scf_type,
            Some(qwb_engine_api::ScfType::Uhf | qwb_engine_api::ScfType::Uks)
        ) && input.jobtype != JobType::Sp
        {
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `{}` currently supports `sp` only (open-shell gradients \
                 are phase-2 SC-02); jobtype = `{}`",
                input
                    .options
                    .scf_type
                    .map_or("closed-shell", qwb_engine_api::ScfType::as_str),
                input.jobtype.as_str()
            )));
        }
        match input.jobtype {
            JobType::Sp => self.run_single_point(input, reporter),
            JobType::Opt => self.run_optimisation(input, reporter),
            other => Err(EngineError::Unsupported(format!(
                "jobtype `{}` is not wired into the engine backend yet (frequencies are \
                 available through qwb_engine::hessian, see F-01/F-02)",
                other.as_str()
            ))),
        }
    }

    fn request_cancel(&self) {
        // 单点能/优化都是同步执行：取消由 `Reporter::is_cancelled` 在步与步之间观察到。
    }
}

/// 闭壳层电子数（`Σ Z − charge`）。
///
/// MP2 的占据轨道数就是它的一半；奇数电子在闭壳层参考里没有定义，所以显式报错
/// 而不是悄悄取整。
fn closed_shell_electrons(molecule: &Molecule) -> Result<usize, EngineError> {
    let mut electrons = -i64::from(molecule.charge);
    for atom in &molecule.atoms {
        let z = qwb_registry::elements::atomic_number(&atom.symbol)
            .ok_or_else(|| EngineError::Data(format!("MP2: unknown element `{}`", atom.symbol)))?;
        electrons += i64::from(z);
    }
    if electrons <= 0 || electrons % 2 != 0 {
        return Err(EngineError::InvalidInput(format!(
            "MP2 needs an even, positive electron count, got {electrons}"
        )));
    }
    usize::try_from(electrons).map_err(|_| {
        EngineError::Internal(format!(
            "MP2: electron count {electrons} does not fit usize"
        ))
    })
}

/// 阶段二选项的"契约已就位、引擎未实现"守卫。
///
/// 判据不是"字段非空就拒"：`solvent_model: vacuum` 这种与阶段一行为等价的写法要放过，
/// 否则旧输入会被无故拒绝。每条错误都指向负责它的 WP，便于把"还没做"和"不支持"分开。
fn reject_unimplemented_options(input: &JobInput) -> Result<(), EngineError> {
    let options = &input.options;
    if let Some(scf_type) = options.scf_type {
        // 阶段二已接：闭壳层 KS/RHF（阶段一行为）、**UHF**（SC-01）、**UKS**（SC-01b）。
        // 只剩 ROHF 没接线（SC-01c 记为 blocked，见 docs/status-v2.yaml）。
        let supported = matches!(
            scf_type,
            qwb_engine_api::ScfType::Rks
                | qwb_engine_api::ScfType::Rhf
                | qwb_engine_api::ScfType::Uhf
                | qwb_engine_api::ScfType::Uks
        );
        if !supported {
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `{}` is not wired yet \
                 (ROHF is phase-2 SC-01c, currently blocked: the Guest–Saunders effective \
                 Fock converges to the wrong stationary point, see docs/status-v2.yaml)",
                scf_type.as_str()
            )));
        }
    }
    if let Some(mp2) = &options.mp2 {
        // M3：**canonical MP2（SC-04）与 RI-MP2（SC-05）都已接**（闭壳层参考、单点）。
        // 这里只剩"这个枚举将来可能加新变体"的兜底：不认识的就拒，不能静默降级。
        match mp2.kind {
            Mp2Kind::Mp2 | Mp2Kind::RiMp2 => {}
            _ => {
                return Err(EngineError::Unsupported(format!(
                    "options.mp2.kind = `{}` is not implemented (M3 ships canonical MP2 and RI-MP2)",
                    mp2.kind.as_str()
                )));
            }
        }
        mp2.validate()?;
        if let Some(scf_type) = options.scf_type
            && scf_type.is_unrestricted()
        {
            return Err(EngineError::Unsupported(format!(
                "options.mp2 needs a closed-shell reference; scf_type = `{}` is open-shell \
                 (open-shell MP2 is not in phase-2 M3)",
                scf_type.as_str()
            )));
        }
        if input.jobtype != JobType::Sp {
            return Err(EngineError::Unsupported(format!(
                "options.mp2 currently supports `sp` only (the MP2 gradient is not implemented); \
                 jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }
        if options.ecp_name.is_some() {
            // ECP 会把核电子拿掉，占据空间要按 `z_eff` 数；这条路径没做（M3 不需要）。
            return Err(EngineError::Unsupported(
                "options.mp2 with an ECP is not implemented (the occupied space would have to \
                 be counted from z_eff; M3 validates MP2 on all-electron references)"
                    .into(),
            ));
        }
    }
    if let Some(excited) = &options.excited_states {
        // M4 / RS-02..03：cis 与 tda 都接（hf 参考下同一套方程），tddft 还没做。
        match excited.method {
            qwb_engine_api::ExcitedStateMethod::Cis | qwb_engine_api::ExcitedStateMethod::Tda => {}
            _ => {
                return Err(EngineError::Unsupported(format!(
                    "options.excited_states.method = `{}` is not wired yet (full TDDFT with the \
                     B block and the XC kernel is ST-01; cis/tda are available)",
                    excited.method.as_str()
                )));
            }
        }
        if let Some(scf_type) = options.scf_type
            && scf_type.is_unrestricted()
        {
            return Err(EngineError::Unsupported(format!(
                "options.excited_states needs a closed-shell reference; scf_type = `{}` is \
                 open-shell (open-shell CIS/TDA is not in M4)",
                scf_type.as_str()
            )));
        }
        if input.jobtype != JobType::Sp {
            return Err(EngineError::Unsupported(format!(
                "options.excited_states currently supports `sp` only; jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }
        if !input.method_name.eq_ignore_ascii_case("hf") {
            // 现在实现的是纯 CIS / TDA：A 矩阵只有 2(ia|jb) − (ij|ab)，没有 XC kernel。
            // 拿 pbe 之类的参考跑它，会把 CIS 的数标成 tda 交出去 —— 静默降级，
            // 所以显式拒绝，指向 ST-01 / FX-03。
            return Err(EngineError::Unsupported(format!(
                "options.excited_states needs an `hf` reference for now (method_name = `{}`): \
                 the XC kernel needed for a DFT reference is ST-01 / FX-03",
                input.method_name
            )));
        }
    }
    if options.numerics.is_some() {
        // 精度等级/近似路径现在没有任何代码在读它：放过就等于"静默忽略"。
        // 想用近似路径，得等 IN-03（PS）或 IN-04（RI-J/JK）落地并过 K10 三条门禁。
        return Err(EngineError::Unsupported(
            "options.numerics is not wired yet (approximation paths are phase-2 IN-03/IN-04, \
             gated by K10 equivalence)"
                .into(),
        ));
    }
    if let Some(model) = options.solvent_model
        && model != qwb_engine_api::SolventModel::Vacuum
    {
        return Err(EngineError::Unsupported(format!(
            "options.solvent_model = `{}` is not wired yet (implicit solvation is phase-2 M5 / SV-01..03)",
            model.as_str()
        )));
    }
    Ok(())
}
