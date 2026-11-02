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
pub mod scan;
pub mod ts;

use std::collections::BTreeMap;
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
        // ST-05：开了 `options.epr` 才报（K2 的 validate 已经保证它是开壳层 + sp 作业）。
        let epr = if input.options.epr {
            reporter.progress("epr", 96.0);
            let block = epr_block(
                &basis,
                &input.molecule,
                &result.density_alpha,
                &result.density_beta,
                "uks",
            )?;
            reporter.log(&format!(
                "epr: {} nuclei, method = uks, first a_iso = {:.4} MHz ({:.4} G)",
                block.nuclei.len(),
                block.nuclei.first().map_or(0.0, |n| n.a_iso_mhz),
                block.nuclei.first().map_or(0.0, |n| n.a_iso_gauss),
            ));
            Some(block)
        } else {
            None
        };
        let mut job = JobResult::new(
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
        .with_spin(spin);
        if let Some(block) = epr {
            job = job.with_epr(block);
        }
        Ok(job.with_log(format!(
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
    /// **ROHF** 单点（SC-01c，第 383 轮接线）：纯自旋的开壳层 HF。
    ///
    /// 与 UHF 的差别只在 SCF 内核（Roothaan 等效 Fock）；结果侧多一件事、少一件事：
    /// * `⟨S²⟩` 是解析的 `S(S+1)`（CSF），不是行列式的期望值；
    /// * 只有一套空间轨道，`mo_energies` 给的是**等效 Fock** 的本征值（不是 Koopmans 轨道能）。
    ///
    /// 自旋密度 = `P_α − P_β = D_o`，所以 `options.epr` 在 ROHF 上同样可用。
    fn run_rohf_single_point(
        &self,
        input: &JobInput,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        let basis_k1 = self.registry.get_basis(&input.basis_name)?;
        let spec = self.spec(input)?;
        if !spec.terms.is_empty() {
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `rohf` is Hartree-Fock only, but method `{}` is a \
                 density functional; use `uks`/`rks`",
                input.method_name
            )));
        }
        if !matches!(input.jobtype, JobType::Sp | JobType::Opt) {
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `rohf` supports `sp` and `opt` only (SC-01c); jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }
        if input.molecule.multiplicity <= 1 {
            return Err(EngineError::InvalidInput(
                "options.scf_type = `rohf` needs an open shell (multiplicity > 1)".to_string(),
            ));
        }

        reporter.progress("load", 0.0);
        reporter.log(&format!(
            "scf_type = rohf (restricted open-shell Hartree-Fock), basis = {} \
             ({} elements), {} atoms, multiplicity {}",
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
        let result = qwb_scf::rohf::rohf_scf(&basis, &input.molecule, &tensor, &options)?;
        reporter.log(&format!(
            "rohf {} in {} iterations: E = {:.10} Ha, <S^2> = {:.6} (exact S(S+1))",
            if result.scf_converged {
                "converged"
            } else {
                "NOT converged"
            },
            result.iterations,
            result.energy_hartree,
            result.s_squared
        ));
        if !result.scf_converged {
            return Err(EngineError::ScfNotConverged {
                iters: result.iterations,
                max: input.options.scf_max_iter,
            });
        }
        reporter.progress("done", 100.0);

        let env = env::build(&self.registry);
        let dipole = qwb_props::dipole_debye(&basis, &input.molecule, &result.density_total)?;
        let spin = qwb_engine_api::SpinResult::new(
            result.n_alpha as u32,
            result.n_beta as u32,
            input.molecule.multiplicity,
            result.s_squared,
        );
        // ROHF 的自旋密度就是开壳层密度（`P_α − P_β = D_o`）。
        let mut alpha_density = result.density_total.clone();
        for i in 0..alpha_density.nrows() {
            for j in 0..alpha_density.ncols() {
                alpha_density[(i, j)] -= result.density_open.get(i, j);
            }
        }
        let epr = if input.options.epr {
            reporter.progress("epr", 96.0);
            let block = epr_block(
                &basis,
                &input.molecule,
                &alpha_density,
                &result.density_closed.clone(),
                "rohf",
            )?;
            reporter.log(&format!(
                "epr: {} nuclei, method = rohf, first a_iso = {:.4} MHz ({:.4} G)",
                block.nuclei.len(),
                block.nuclei.first().map_or(0.0, |n| n.a_iso_mhz),
                block.nuclei.first().map_or(0.0, |n| n.a_iso_gauss),
            ));
            Some(block)
        } else {
            None
        };
        let mut job = JobResult::new(
            result.energy_hartree,
            result.scf_converged,
            self.registry.snapshot(),
            env,
        )
        .with_dipole(nalgebra::DVector::from_column_slice(&dipole))
        .with_orbitals(
            nalgebra::DVector::from_column_slice(result.mo_energies.as_slice()),
            density_to_api(&result.coefficients),
        )
        .with_spin(spin);
        if let Some(block) = epr {
            job = job.with_epr(block);
        }
        Ok(job.with_log(format!(
            "scf_type = rohf, basis = {}, iterations = {}, E_elec = {:.10}, <S^2> = {:.6} \
             (exact), n_alpha = {}, n_beta = {}, dipole = [{:.4}, {:.4}, {:.4}] D. \
             mo_energies_hartree are the EFFECTIVE (Roothaan) Fock eigenvalues, not Koopmans.",
            basis_k1.name,
            result.iterations,
            result.electronic_hartree,
            result.s_squared,
            result.n_alpha,
            result.n_beta,
            dipole[0],
            dipole[1],
            dipole[2]
        )))
    }

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
        // `opt` 的**收尾单点**也走这里（`run_optimisation` 有意不改 jobtype），所以放行；
        // freq/ts/irc/scan 由分派处拒绝（见 `reject_unimplemented_options` 上方）。
        if !matches!(input.jobtype, JobType::Sp | JobType::Opt) {
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `uhf` supports `sp` and `opt` only (the rest of the \
                 open-shell track is later WPs); jobtype = `{}`",
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
        // ST-05：UHF 路径同样支持 `options.epr`（自旋密度 = P_α − P_β）。
        let epr = if input.options.epr {
            reporter.progress("epr", 96.0);
            let block = epr_block(
                &basis,
                &input.molecule,
                &result.density_alpha,
                &result.density_beta,
                "uhf",
            )?;
            reporter.log(&format!(
                "epr: {} nuclei, method = uhf, first a_iso = {:.4} MHz ({:.4} G)",
                block.nuclei.len(),
                block.nuclei.first().map_or(0.0, |n| n.a_iso_mhz),
                block.nuclei.first().map_or(0.0, |n| n.a_iso_gauss),
            ));
            Some(block)
        } else {
            None
        };
        let mut job = JobResult::new(
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
        .with_spin(spin);
        if let Some(block) = epr {
            job = job.with_epr(block);
        }
        Ok(job.with_log(format!(
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
            Some(qwb_engine_api::ScfType::Rohf) => {
                return self.run_rohf_single_point(input, reporter);
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

        // SV-01 / KV-02：隐式溶剂（COSMO / C-PCM）。SV-03 起同一套表面方程也服务 SMD ——
        // 差别只有两处：半径换成 SMD 的参数化半径（eq. 16），能量里加一项 `G_CDS`（eq. 4.77）。
        //
        // 空腔的碎片数是**稠密**表面方程的规模（`O(M³)` 一次分解、每步 SCF 重解一次），
        // 所以这里既给默认值也设上限：大分子自动降采样并**记日志**，不静默改变精度。
        // 250 点/原子是 `docs/choices.md` 六十二里的实测选择（160→250 差 0.6 kcal/mol）。
        const COSMO_POINTS_PER_ATOM: usize = 250;
        const COSMO_MAX_TESSERAE: usize = 1500;
        // SMD 的 SASA 用同一个 Fibonacci 球面采样。实测：32768 点时水相 `G_CDS` 对上游
        // `libsolvent` 最差 8.4e-4、8192 点 1.63e-3（`smd_sigma` 的收敛表，O(1/√N) 采样噪声）。
        // 4096 点是"精度还在 2e-3 量级、成本可忽略"的折中，且**写进日志**，不静默取值。
        const SMD_POINTS_PER_ATOM: usize = 4096;
        let mut cosmo: Option<(std::sync::Arc<cosmo_scf::CosmoScfField>, f64, f64, usize)> = None;
        // SV-03：`G_CDS` 的三个数（总数 / 每原子那一半 / 分子那一半），单位 Hartree。
        let mut smd_cds: Option<(f64, f64, f64)> = None;
        let mut solvent_model_used: Option<qwb_engine_api::SolventModel> = None;
        let solvent_name = input.solvent.clone();
        if solvent_name != "vacuum" {
            let water = solvent_name.eq_ignore_ascii_case("water");
            let (epsilon, scale, radii, model) = {
                let solvent = self.registry.get_solvent(&solvent_name)?;
                // 数据里的 `cavity_model` 是**默认值**；输入给了 `options.solvent_model` 就听输入
                // （§8.2：模型是显式选择，不猜）。
                let data_model = match solvent.cavity_model {
                    qwb_registry::CavityModel::Pcm => qwb_engine_api::SolventModel::Cosmo,
                    qwb_registry::CavityModel::Smd => qwb_engine_api::SolventModel::Smd,
                    qwb_registry::CavityModel::None => {
                        return Err(EngineError::Data(format!(
                            "solvent `{solvent_name}` declares cavity_model = none"
                        )));
                    }
                    _ => {
                        return Err(EngineError::Unsupported(format!(
                            "solvent `{solvent_name}` declares an unimplemented cavity_model"
                        )));
                    }
                };
                let model = input.options.solvent_model.unwrap_or(data_model);
                if model == qwb_engine_api::SolventModel::Cosmo
                    && data_model == qwb_engine_api::SolventModel::Smd
                {
                    // 两套半径家族不能混：数据说 SMD 空腔，就不许用 COSMO 的键去跑它。
                    return Err(EngineError::InvalidInput(format!(
                        "solvent `{solvent_name}` declares cavity_model = smd; ask for \
                         `solvent_model: smd` instead of letting COSMO use another radius set"
                    )));
                }
                let use_smd = model == qwb_engine_api::SolventModel::Smd;
                // SV-03：`solvent_model: smd` 时半径一律用 SMD 的参数化半径（eq. 16），
                // 与数据里声明的 `radii_set`（Bondi/UFF）无关 —— 那套是 COSMO 的键。
                let radii_set = if use_smd {
                    Some(qwb_registry::RadiiSet::Smd)
                } else {
                    solvent.radii_set
                };
                let smd_data = if radii_set == Some(qwb_registry::RadiiSet::Smd) {
                    Some(self.registry.get_smd_atomic(SMD_ATOMIC_NAME)?)
                } else {
                    None
                };
                let (radii, scale) = cosmo_radii(
                    &input.molecule,
                    radii_set,
                    &solvent.atomic_radii,
                    solvent.scale,
                    solvent.smd.as_ref().and_then(|block| block.abraham_alpha),
                    smd_data,
                )?;
                if use_smd {
                    let data = smd_data.ok_or_else(|| {
                        EngineError::Data(
                            "solvent_model = smd needs the `smd-atomic` data table".to_string(),
                        )
                    })?;
                    // 描述符缺一个就不是 SMD（非水相）；水相走单独的 σ 表、不需要它们。
                    let descriptors = smd_descriptors_or_none(solvent.smd.as_ref());
                    let surface = qwb_props::smd_surface::smd_surface(
                        data,
                        &input.molecule,
                        SMD_POINTS_PER_ATOM,
                        0.0,
                    )?;
                    let sts = qwb_props::smd_sigma::smd_sts(
                        data,
                        &input.molecule,
                        water,
                        descriptors.as_ref(),
                    )?;
                    let cssigm =
                        qwb_props::smd::cds_constant_term(data, water, descriptors.as_ref());
                    // eq. 4.77 的两半分开算：`Σ σ_j A_j` 与 `σ^[M] Σ A_j`。
                    let zeros = vec![0.0_f64; surface.areas_angstrom2.len()];
                    let atomic_kcal = qwb_props::smd_sigma::cds_energy_kcal_from_sts(
                        &sts,
                        0.0,
                        &surface.areas_angstrom2,
                    )
                    .map_err(EngineError::Data)?;
                    let molecular_kcal = qwb_props::smd_sigma::cds_energy_kcal_from_sts(
                        &zeros,
                        cssigm,
                        &surface.areas_angstrom2,
                    )
                    .map_err(EngineError::Data)?;
                    let to_hartree = 1.0 / HARTREE_PER_KCAL_MOL;
                    let atomic = atomic_kcal * to_hartree;
                    let molecular = molecular_kcal * to_hartree;
                    reporter.log(&format!(
                        "solvation: smd cds (eq. 4.77), solvent = {solvent_name}{},                      {SMD_POINTS_PER_ATOM} points/atom, A_total = {:.3} A^2,                      G_CDS = {:.6} kcal/mol (atomic {:.6} + molecular {:.6})",
                        if water {
                            " (aqueous sigma table; molecular term is 0)"
                        } else {
                            ""
                        },
                        surface.total_area_angstrom2,
                        atomic_kcal + molecular_kcal,
                        atomic_kcal,
                        molecular_kcal
                    ));
                    smd_cds = Some((atomic + molecular, atomic, molecular));
                }
                (solvent.epsilon, scale, radii, model)
            };
            if !epsilon.is_finite() || epsilon < 1.0 {
                return Err(EngineError::Data(format!(
                    "solvent `{solvent_name}` has epsilon = {epsilon}"
                )));
            }
            if let Some(model) = input.options.solvent_model
                && model == qwb_engine_api::SolventModel::Vacuum
            {
                return Err(EngineError::InvalidInput(format!(
                    "solvent `{solvent_name}` was requested together with solvent_model = vacuum"
                )));
            }
            let atoms = input.molecule.n_atoms().max(1);
            let mut points = COSMO_POINTS_PER_ATOM;
            while points > 60 && points.saturating_mul(atoms) > COSMO_MAX_TESSERAE {
                points /= 2;
            }
            let cavity = qwb_props::cavity::cavity(&input.molecule, &radii, scale, points)?;
            let field = std::sync::Arc::new(cosmo_scf::CosmoScfField::build(
                &basis,
                &input.molecule,
                &cavity.tesserae,
                epsilon,
            )?);
            reporter.log(&format!(
                "solvation: {model}, solvent = {solvent_name} (eps = {epsilon}), \
                 {} tesserae at {points} points/atom, surface A = {:.3} A^2, V = {:.3} A^3",
                cavity.tesserae.len(),
                cavity.surface_area_angstrom2,
                cavity.volume_angstrom3,
                model = model.as_str()
            ));
            options.external_field = Some(field.handle());
            solvent_model_used = Some(model);
            cosmo = Some((
                field,
                cavity.surface_area_angstrom2,
                cavity.volume_angstrom3,
                cavity.tesserae.len(),
            ));
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
                // ST-01：`method: tddft` 走完整 RPA（含 B 块），`cis`/`tda` 走 TDA。
                // 两条路都返回同一形状的涨落，报告里只靠 `method` 区分 —— 不允许
                // 把 TDA 的数标成 tddft 交出去（validate 已经把非 hf 参考挡在门外）。
                let (excitations, oscillator_sum, dimension, label) = match request.method {
                    qwb_engine_api::ExcitedStateMethod::Tddft => {
                        let out = qwb_response::tddft_singlet(
                            &cphf_input,
                            &orbital_energies,
                            request.n_states as usize,
                        )?;
                        (out.excitations, out.oscillator_sum, out.dimension, "tddft")
                    }
                    _ => {
                        let out = qwb_response::tda_singlet(
                            &cphf_input,
                            &orbital_energies,
                            request.n_states as usize,
                        )?;
                        (out.excitations, out.oscillator_sum, out.dimension, "tda")
                    }
                };
                let transitions: Vec<qwb_engine_api::Transition> = excitations
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
                    "{label}: {} state(s) from a {dimension}x{dimension} problem; \
                     first = {:.6} Ha ({:.3} eV), f = {:.5}; sum of f = {oscillator_sum:.6} \
                     (N_elec = {n_electrons})",
                    transitions.len(),
                    excitations.first().map_or(0.0, |s| s.energy_hartree),
                    excitations
                        .first()
                        .map_or(0.0, |s| s.energy_hartree * 27.211_386_245_988),
                    excitations.first().map_or(0.0, |s| s.oscillator_strength),
                ));
                Some(qwb_engine_api::ExcitedStates::new(
                    request.method,
                    transitions.len(),
                    transitions,
                    qwb_engine_api::SumRules::new(oscillator_sum, n_electrons),
                ))
            }
            None => None,
        };
        // RS-08：无迹四极矩进 K7 的 `multipoles` 块。原点取**核电荷加权的质心**
        // （`qwb-registry` 没有质量表，所以这不是"质心"而是"核电荷中心"——名字与
        // `origin_angstrom` 一起落盘，读的人能看清用的是哪个原点）。
        let origin_angstrom = {
            let mut weight = 0.0;
            let mut moment = [0.0_f64; 3];
            for atom in &input.molecule.atoms {
                let z = f64::from(qwb_registry::elements::atomic_number(&atom.symbol).unwrap_or(0));
                weight += z;
                for (axis, value) in atom.position_angstrom.iter().enumerate() {
                    moment[axis] += z * value;
                }
            }
            if weight > 0.0 {
                [moment[0] / weight, moment[1] / weight, moment[2] / weight]
            } else {
                [0.0; 3]
            }
        };
        let quadrupole = qwb_props::quadrupole_traceless_debye_angstrom(
            &basis,
            &input.molecule,
            &result.density,
            origin_angstrom,
        )?;
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
        // RS-08：四极矩（K7）。`origin_angstrom` 一起落盘 —— 多极矩对原点敏感。
        job = job.with_multipoles(
            qwb_engine_api::Multipoles::new(origin_angstrom).with_quadrupole(quadrupole),
        );
        // PR-04：把激发态展宽成 UV/Vis 谱。展宽参数来自数据（`numerics.spectrum_defaults`），
        // 不在这里写常量；数据缺这一块时只记一条日志，不让整个作业失败（旧数据仍可用）。
        let mut uv_vis_spectrum = None;
        if let Some(states) = &excited_states {
            let transitions: Vec<(f64, f64)> = states
                .transitions
                .iter()
                .map(|t| (t.energy_hartree * 27.211_386_245_988, t.oscillator_strength))
                .collect();
            match qwb_props::spectrum::SpectrumOptions::from_registry(&self.registry) {
                Ok(options) => {
                    match qwb_props::spectrum::broadened_spectrum(&transitions, &options) {
                        Ok(spectrum) => {
                            let points = spectrum
                                .points
                                .iter()
                                .map(|(energy, intensity)| [*energy, *intensity])
                                .collect::<Vec<[f64; 2]>>();
                            reporter.log(&format!(
                                "spectrum: {} point(s), HWHM = {:.3} eV, {} local maximum(a)",
                                points.len(),
                                spectrum.gaussian_hwhm_ev,
                                qwb_engine_api::UvVisSpectrum::new(
                                    spectrum.gaussian_hwhm_ev,
                                    spectrum.grid_step_ev,
                                    points.clone(),
                                )
                                .peak_energies_ev()
                                .len()
                            ));
                            uv_vis_spectrum = Some(qwb_engine_api::UvVisSpectrum::new(
                                spectrum.gaussian_hwhm_ev,
                                spectrum.grid_step_ev,
                                points,
                            ));
                        }
                        Err(err) => reporter.log(&format!("spectrum: skipped ({err})")),
                    }
                }
                Err(err) => reporter.log(&format!("spectrum: skipped ({err})")),
            }
        }
        if let Some(excited_states) = excited_states {
            job = job.with_excited_states(excited_states);
        }
        if let Some(spectrum) = uv_vis_spectrum {
            job = job.with_uv_vis_spectrum(spectrum);
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
        if let Some((field, area, volume, n_tesserae)) = &cosmo {
            // K8：`total_hartree` 按**对账恒等式**取溶剂化自由能
            // （`total = 静电 + 非极性三项`，见 `SolvationResult::decomposition_gap_hartree`），
            // 体系绝对能量仍在 `JobResult.energy_hartree`。COSMO 没有非极性项，
            // 所以 total 恰好等于 electrostatic。
            //
            // 静电项取 **`E_溶 + E_gas` 的差**（SMx 的 G_EP 约定）：它含溶质的极化形变能，
            // 而 `½ΣV_K q_K` 只是溶剂-溶质相互作用的一半，两者差 5–10e-4 Ha。
            // 判据（与实验比 1.5 kcal/mol）必须用前者，所以这里真的再跑一次气相 SCF。
            let mut gas_options = options.clone();
            gas_options.external_field = None;
            let gas = qwb_scf::rks::rks_scf(
                &basis,
                &input.molecule,
                &tensor,
                &grid,
                &spec.terms,
                gas_options,
            )?;
            let distortion =
                field.last_solvation_energy() - (result.energy_hartree - gas.energy_hartree);
            reporter.log(&format!(
                "solvation: G_EP = E_sol - E_gas = {:.10} Ha, ½ΣVq = {:.10} Ha, \
                 polarisation/distortion term = {:.3e} Ha",
                result.energy_hartree - gas.energy_hartree,
                field.last_solvation_energy(),
                distortion
            ));
            let electrostatic = result.energy_hartree - gas.energy_hartree;
            let model = solvent_model_used.unwrap_or(qwb_engine_api::SolventModel::Cosmo);
            let block = qwb_engine_api::SolvationResult::new(
                model,
                &solvent_name,
                electrostatic,
                electrostatic,
                qwb_engine_api::solvation::CavityGeometry::new(*area, *volume, *n_tesserae),
                // 表面电荷每步 SCF 重解一次，所以「求解次数」= SCF 迭代数
                result.iterations as usize,
            )
            .with_converged(result.scf_converged);
            // SV-03：SMD 的非极性项是**一项** `G_CDS`（eq. 4.77），不是三项。`total_hartree`
            // 按对账恒等式取 `G_EP + G_CDS`（该块的 `total` 是**溶剂化自由能**，不是体系
            // 绝对能量；绝对能量仍在 `JobResult.energy_hartree`）。
            let (block, total) = match smd_cds {
                Some((cds, atomic, molecular)) => {
                    let total = electrostatic + cds;
                    (block.with_cds(cds, atomic, molecular), total)
                }
                None => (block, electrostatic),
            };
            let mut block = block;
            block.total_hartree = total;
            block.validate()?;
            job = job
                .with_log(format!(
                    "solvation = {}/{}: dG_el = {:.10} Ha ({:.3} kcal/mol), G_CDS = {:.6} kcal/mol, \
                     dG_solv = {:.3} kcal/mol, A = {:.3} A^2, V = {:.3} A^3, {} tesserae",
                    model.as_str(),
                    solvent_name,
                    electrostatic,
                    electrostatic * HARTREE_PER_KCAL_MOL,
                    total * HARTREE_PER_KCAL_MOL - electrostatic * HARTREE_PER_KCAL_MOL,
                    total * HARTREE_PER_KCAL_MOL,
                    area,
                    volume,
                    n_tesserae
                ))
                .with_solvation(block);
        }
        Ok(job)
    }

    /// 过渡态搜索（`Ts`，phase-2 `RP-01`）：Hessian 引导的局部本征向量跟随。
    ///
    /// 判据是**两条一起**：`max|g| < options.ts.gradient_threshold_hartree_per_bohr`（缺省
    /// 4.5e-4）**且** 收敛后的 Hessian 恰好一个虚频。第二条不能省 —— 梯度小只说明停在驻点，
    /// 极小点也满足它，所以引擎在收敛后重算一次 Hessian 并核对虚频个数，不为 1 就报错。
    fn run_ts(&self, input: &JobInput, reporter: &dyn Reporter) -> Result<JobResult, EngineError> {
        if let Some(ecp_name) = &input.options.ecp_name {
            return Err(EngineError::Unsupported(format!(
                "transition-state search with an ECP (`{ecp_name}`) is not implemented: the ECP \
                 derivative w.r.t. nuclear coordinates is missing (I-05 covers energies only)"
            )));
        }
        if input
            .options
            .scf_type
            .is_some_and(|scf_type| scf_type.is_open_shell())
        {
            return Err(EngineError::Unsupported(
                "transition-state search is implemented for closed-shell HF/KS only \
                 (open-shell gradients have no acceptance tests)"
                    .to_string(),
            ));
        }
        let options_ts = input.options.ts.clone().unwrap_or_default();
        let spec = self.spec(input)?;
        let mut options = opt::OptOptions::new(
            &input.basis_name,
            spec.functional_name.clone(),
            spec.grid_name.clone(),
        );
        options.max_iterations = options_ts.max_steps;
        options.force_threshold = options_ts.gradient_threshold_hartree_per_bohr;
        options.scf_max_iter = input.options.scf_max_iter;
        options.scf_energy_threshold = input.options.scf_energy_threshold_hartree;
        options.scf_density_threshold = input.options.scf_density_threshold;
        options.scf_type = input.options.scf_type;
        // 每一步最多走 0.2 Bohr：本征向量跟随在鞍点附近很好用，但远离鞍点时大步会发散。
        options.max_step_bohr = 0.2;

        reporter.progress("ts", 0.0);
        reporter.log(&format!(
            "jobtype = ts, method = {}, basis = {}, grid = {} (eigenvector following, \
             max steps = {}, threshold = {:.1e} Ha/Bohr)",
            if spec.functional_name.is_empty() {
                "hf".to_string()
            } else {
                spec.functional_name.clone()
            },
            input.basis_name,
            spec.grid_name,
            options_ts.max_steps,
            options_ts.gradient_threshold_hartree_per_bohr
        ));
        let outcome = ts::search(
            &self.registry,
            &input.molecule,
            &options,
            options_ts.max_steps,
            options_ts.gradient_threshold_hartree_per_bohr,
            reporter,
        )?;
        let mut block = ts::to_api(&outcome);
        block.iterations = outcome.iterations;
        if outcome.n_imaginary != 1 {
            return Err(EngineError::Internal(format!(
                "ts: search reported {} real imaginary modes, expected exactly 1",
                outcome.n_imaginary
            )));
        }
        block.validate()?;
        reporter.progress("done", 100.0);
        let summary = format!(
            "ts = eig-following: E = {:.10} Ha, max|g| = {:.3e} Ha/Bohr, imaginary = {:.2} cm^-1, \
             {} steps",
            block.energy_hartree,
            block.max_gradient_hartree_per_bohr,
            block.imaginary_wavenumber_cm,
            block.iterations
        );
        reporter.log(&summary);
        // 收尾的单点在**鞍点几何**上跑（不是输入几何）：`energy_hartree` 与块里的能量
        // 因此指向同一个点，轨道/偶极也是那个点的。
        let mut final_input = input.clone();
        final_input.molecule = outcome.molecule.clone();
        final_input.jobtype = JobType::Sp;
        let mut job = self.run_single_point(&final_input, reporter)?;
        job = job.with_log(summary).with_ts_result(block);
        Ok(job)
    }

    /// PES 扫描（`Scan`，phase-2 `RP-03`）：把某个内坐标从 `start` 扫到 `stop`。
    ///
    /// `relax: false`（缺省）是**刚性**扫描：每个点只把坐标挪过去、其余原子不动，然后算一次
    /// 单点；`relax: true` 是**弛豫**扫描：每个点从上一个点的几何出发做一次自由优化，报的是
    /// 优化后的能量与步数。两者的差别写进结果的 `relaxed` 字段 —— 混着读会得出错的结论。
    ///
    /// 每个点都会重新校验"挪完的内坐标确实等于目标值"（角度 1e-8、距离 1e-10），失败就报错，
    /// 不会拿一个"差不多"的几何继续算下去。
    fn run_scan(
        &self,
        input: &JobInput,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        if let Some(ecp_name) = &input.options.ecp_name {
            return Err(EngineError::Unsupported(format!(
                "PES scans with an ECP (`{ecp_name}`) are not implemented: the ECP derivative \
                 w.r.t. nuclear coordinates is missing (relaxed scans need it)"
            )));
        }
        if input
            .options
            .scf_type
            .is_some_and(|scf_type| scf_type.is_open_shell())
        {
            return Err(EngineError::Unsupported(
                "PES scans are implemented for closed-shell HF/KS only (open-shell second \
                 derivatives and gradients have no acceptance tests yet)"
                    .to_string(),
            ));
        }
        let scan = input.options.scan.as_ref().ok_or_else(|| {
            EngineError::InvalidInput(
                "jobtype: scan needs options.scan (kind / atoms / start / stop / points)"
                    .to_string(),
            )
        })?;
        // 弛豫扫描：每个点都是**约束优化**（`opt::Constraint`：梯度投影 + 每步重设内坐标）。
        // 目录里曾经拒绝过它 —— 那时"每个点自由优化"会让所有点塌回同一个极小点，曲线是平的
        // 却看着像扫描；现在约束那一半补上了，所以打开。
        let values = scan.values(input.molecule.n_atoms())?;
        reporter.progress("scan", 0.0);
        reporter.log(&format!(
            "jobtype = scan, kind = {}, atoms = {:?}, {} -> {} {}, {} points, relaxed = {}",
            scan.kind.as_str(),
            scan.atoms,
            scan.start,
            scan.stop,
            scan.kind.unit(),
            values.len(),
            scan.relax
        ));

        let total = values.len();
        let mut points = Vec::with_capacity(total);
        // 扫描没有"一个"单点结果：`JobResult.energy_hartree` 取**能量最低的那个点**（那是对
        // 这条 PES 最有用的一个数），轨道/偶极等其余字段也来自那一点；块里逐点都有数。
        let mut best: Option<qwb_engine_api::JobResult> = None;
        for (index, value) in values.iter().enumerate() {
            let geometry = scan::set_coordinate(&input.molecule, scan.kind, &scan.atoms, *value)?;
            let mut point_input = input.clone();
            point_input.molecule = geometry;
            let (job, iterations, optimised) = if scan.relax {
                let job = self.run_optimisation(&point_input, reporter)?;
                let report = self.opt_sink.take();
                let iterations = report.as_ref().map_or(0, |report| report.iterations);
                let converged = report.as_ref().is_some_and(|report| report.converged);
                if !converged {
                    return Err(EngineError::Unsupported(format!(
                        "relaxed scan point {} ({value}) did not converge; the scan would mix \
                         optimised and non-optimised points",
                        index + 1
                    )));
                }
                // 优化之后内坐标必须仍在目标上（每步都重设过，这里只是最后再核一次）。
                // `OptReport` 刚被 `take()` 走了，所以量的是优化器返回的几何 —— 它在
                // `run_optimisation` 里已经写进 `artifacts/optimized.xyz`，这里重跑一次量的是
                // 同一份数据（`run_optimisation` 收尾的单点用的就是优化后的几何）。
                let achieved = report
                    .as_ref()
                    .map(|report| scan::measure(&report.molecule, scan.kind, &scan.atoms))
                    .transpose()?
                    .unwrap_or(*value);
                if (achieved - *value).abs() > 1.0e-6 {
                    return Err(EngineError::Internal(format!(
                        "relaxed scan point {} drifted: wanted {value}, measured {achieved}",
                        index + 1
                    )));
                }
                (job, iterations, true)
            } else {
                (self.run_single_point(&point_input, reporter)?, 0_u32, false)
            };
            let energy = job.energy_hartree;
            reporter.progress("scan", 100.0 * (index + 1) as f64 / total as f64);
            reporter.log(&format!(
                "scan point {}/{}: {} = {value:.6} {}, E = {energy:.10} Ha{}",
                index + 1,
                total,
                scan.kind.as_str(),
                scan.kind.unit(),
                if optimised {
                    format!(" (relaxed in {iterations} steps)")
                } else {
                    String::new()
                }
            ));
            points.push(qwb_engine_api::PesPoint::new(
                *value, energy, optimised, iterations,
            ));
            if best
                .as_ref()
                .is_none_or(|current| energy < current.energy_hartree)
            {
                best = Some(job);
            }
        }

        let block = qwb_engine_api::PesScan::new(
            scan.kind.as_str(),
            scan.kind.unit(),
            scan.atoms.clone(),
            scan.relax,
            points,
        );
        block.validate()?;
        let minimum_at = block.minimum().map(|point| point.value);
        let summary = format!(
            "scan: {} points ({}), max adjacent |dE| = {:.6e} Ha, minimum at {:?}",
            block.points.len(),
            if block.relaxed { "relaxed" } else { "rigid" },
            block.max_adjacent_delta_hartree,
            minimum_at
        );
        reporter.log(&summary);
        reporter.progress("done", 100.0);
        let job =
            best.ok_or_else(|| EngineError::Internal("scan produced no points".to_string()))?;
        Ok(job.with_log(summary).with_pes_scan(block))
    }

    /// 频率作业（`Freq`）：数值 Hessian → 简正模 → 谐振频率 + 红外强度 + 理想气体热化学。
    ///
    /// 这一步只做**数值** Hessian（`hessian::numerical_hessian`）：解析 Hessian 是 RS-05
    /// （blocked），所以 `vibrational_frequencies.method` 写死 `numerical-hessian`，
    /// 读数的人一眼就知道频率的来源。
    ///
    /// # 两个显式拒绝（都是"会给出看起来正常但错的数"的情形）
    ///
    /// * **虚频**：`hessian` 里"取最大的 `3N−6` 个模当振动"这条规则只在**极小点**成立 ——
    ///   鞍点上虚频（负波数）排在平移/转动模之下，会被当成非振动模丢掉。所以出现虚频
    ///   一律拒绝并点名 RP-01（要正确的分类得先把振动子空间投影出来），不猜。
    /// * **开壳层 / ECP**：频率的数值路径复用了 `opt` 的解析梯度，而开壳层与 ECP 的
    ///   二阶导没有任何验收；放过就是拿闭壳层的 Hessian 当真。
    fn run_frequencies(
        &self,
        input: &JobInput,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        if let Some(ecp_name) = &input.options.ecp_name {
            return Err(EngineError::Unsupported(format!(
                "frequencies with an ECP (`{ecp_name}`) are not implemented: the ECP derivative \
                 w.r.t. nuclear coordinates is missing (I-05 covers energies only)"
            )));
        }
        if input
            .options
            .scf_type
            .is_some_and(|scf_type| scf_type.is_open_shell())
        {
            return Err(EngineError::Unsupported(
                "frequencies are implemented for closed-shell HF/KS only: the numerical Hessian \
                 reuses the analytic gradient, and open-shell (uhf/rohf/uks) second derivatives \
                 have no acceptance tests yet"
                    .to_string(),
            ));
        }
        let spec = self.spec(input)?;
        let mut options = opt::OptOptions::new(
            &input.basis_name,
            spec.functional_name.clone(),
            spec.grid_name.clone(),
        );
        options.max_iterations = input.options.max_opt_steps;
        options.force_threshold = input.options.opt_gradient_threshold_hartree_per_bohr;
        options.scf_max_iter = input.options.scf_max_iter;
        options.scf_energy_threshold = input.options.scf_energy_threshold_hartree;
        options.scf_density_threshold = input.options.scf_density_threshold;
        options.scf_type = input.options.scf_type;

        reporter.progress("single point", 5.0);
        let result = self.run_single_point(input, reporter)?;

        reporter.progress("hessian", 20.0);
        reporter.log(&format!(
            "jobtype = freq, method = {}, basis = {}, grid = {} (numerical Hessian, {})",
            if spec.functional_name.is_empty() {
                "hf".to_string()
            } else {
                spec.functional_name.clone()
            },
            input.basis_name,
            spec.grid_name,
            hessian::HESSIAN_METHOD
        ));
        let hessian = hessian::numerical_hessian(&self.registry, &input.molecule, &options)?;
        // 只数**振动模**里的虚频：3N 个本征值里有 3 个平动 + 3 个（线性 5 个）转动是零模，
        // 数值上它们的本征值会在 0 两侧随机落 —— 把它们的负号也算成"虚频"，极小点也会被拒
        // （实测：H2O/STO-3G 的光滑极小点上有 4 个 −48.7 cm⁻¹ 的零模噪声）。
        // 真虚频的判据带阈值：3N 个本征值里有 3 个平动 + 3 个转动是零模，数值噪声让它们在
        // 0 两侧随机落 —— 实测 H2O/STO-3G 的优化几何上最负的零模噪声是 −48.7 cm⁻¹，而拉长
        // O–H 的鞍点给出的是 −893 cm⁻¹ 量级。所以「所有模里 < −100 cm⁻¹」才算真虚频；
        // 这个数不是拍的，是那两条实测之间的中点，并且写进日志说明。
        let imaginary: Vec<f64> = hessian
            .modes
            .iter()
            .filter(|mode| mode.wavenumber < IMAGINARY_MODE_THRESHOLD_CM)
            .map(|mode| mode.wavenumber)
            .collect();
        if !imaginary.is_empty() {
            return Err(EngineError::Unsupported(format!(
                "frequency analysis found {} imaginary mode(s) (most negative = {:.1} cm^-1, \
                 threshold {:.0} cm^-1): the 3N-6 vibrational set is selected by wavenumber, \
                 which is only correct at a minimum - a saddle point needs the vibrational \
                 subspace projected out (RP-01)",
                imaginary.len(),
                imaginary.first().copied().unwrap_or_default(),
                IMAGINARY_MODE_THRESHOLD_CM
            )));
        }

        reporter.progress("ir", 70.0);
        let dipoles = hessian::dipole_derivatives(
            &self.registry,
            &input.molecule,
            &input.basis_name,
            &spec.functional_name,
            &spec.grid_name,
        )?;
        let intensities = hessian::ir_intensities(&hessian, &dipoles);
        let start = hessian
            .modes
            .len()
            .saturating_sub(hessian.vibrational_modes);
        let modes: Vec<qwb_engine_api::VibrationalMode> = hessian
            .modes
            .iter()
            .zip(&intensities)
            .skip(start)
            .map(|(mode, intensity)| {
                qwb_engine_api::VibrationalMode::new(mode.wavenumber).with_ir(*intensity)
            })
            .collect();
        let frequencies = qwb_engine_api::VibrationalFrequencies::new(
            hessian::HESSIAN_METHOD,
            modes,
            hessian.zero_point_cm,
            hessian.vibrational_modes,
            hessian.degrees_of_freedom,
        );
        frequencies.validate()?;

        reporter.progress("thermochemistry", 90.0);
        let given_symmetry = input.options.symmetry_number;
        let symmetry = if given_symmetry > 1 {
            Some(f64::from(given_symmetry))
        } else {
            None
        };
        let thermo = hessian::thermochemistry(
            &input.molecule,
            &hessian,
            result.energy_hartree,
            THERMO_TEMPERATURE_K,
            THERMO_PRESSURE_PA,
            symmetry,
        )?;
        let thermochemistry = qwb_engine_api::Thermochemistry::new(
            thermo.temperature_k,
            thermo.pressure_pa,
            thermo.zero_point_hartree,
            thermo.internal_energy_hartree,
            thermo.enthalpy_hartree,
            thermo.entropy_j_per_mol_k,
            thermo.gibbs_hartree,
            thermo.rotational_constants_cm.clone(),
            symmetry.map_or(1, |value| value as u32),
        );
        thermochemistry.validate(result.energy_hartree)?;

        reporter.log(&format!(
            "freq = numerical-hessian: {} modes, ZPE = {:.2} cm^-1 ({:.6} Ha), \
             H_corr = {:.6} Ha, S = {:.2} J/(mol K), G = {:.6} Ha at {:.2} K / {:.0} Pa \
             (rotational symmetry number = {})",
            frequencies.vibrational_modes,
            frequencies.zero_point_cm,
            thermo.zero_point_hartree,
            thermo.enthalpy_hartree,
            thermo.entropy_j_per_mol_k,
            thermo.gibbs_hartree,
            thermo.temperature_k,
            thermo.pressure_pa,
            thermochemistry.symmetry_number
        ));
        reporter.progress("done", 100.0);
        Ok(result
            .with_vibrational_frequencies(frequencies)
            .with_thermochemistry(thermochemistry))
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
        // SC-02：开壳层（UHF）的解析梯度。UKS 还没接（见 `opt::analytic_energy_and_gradient`）。
        options.scf_type = input.options.scf_type;
        // RP-03：弛豫扫描的每个点都是**约束优化**（固定内坐标）。目标值取当前几何上的内坐标值
        // —— 调用方（`run_scan`）已经把坐标设成扫描点，所以这里量到的就是那一点的目标。
        if input.jobtype == JobType::Scan
            && let Some(scan) = &input.options.scan
            && scan.relax
        {
            let target = scan::measure(&input.molecule, scan.kind, &scan.atoms)?;
            options.constraint = Some(opt::Constraint::new(scan.kind, scan.atoms.clone(), target));
        }

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
        // **幽灵原子（counterpoise/BSSE）**：只支持 `sp`，且不支持色散校正 / ECP / 溶剂。
        //
        // 为什么这么窄：ghost 的语义是"只提供基函数、没有核与电子"。原子核相关的量
        // （核排斥、偶极的核项、布居的 Z、EPR 的核 g 因子）、以及按元素配对的色散校正
        // 都必须逐个审过才能放行 —— 没审过就放过，用户会拿到一个数值上"看起来正常"但
        // 把 ghost 当成真实原子算的结果。所以这里按"能量路径"白名单，其余显式拒绝。
        if input.molecule.has_ghosts() {
            if input.jobtype != JobType::Sp {
                return Err(EngineError::Unsupported(format!(
                    "ghost atoms (counterpoise/BSSE) are implemented for `sp` only; jobtype = `{}`",
                    input.jobtype.as_str()
                )));
            }
            if input.options.dispersion.is_some()
                || input.options.ecp_name.is_some()
                || input.solvent != "vacuum"
            {
                return Err(EngineError::Unsupported(
                    "ghost atoms are not audited against dispersion / ECP / solvent yet; \
                     run the counterpoise energies without them"
                        .to_string(),
                ));
            }
        }
        if input.solvent != "vacuum" {
            // SV-01：单点的隐式溶剂（COSMO / C-PCM）已实装，但**还没接到**梯度的
            // 自洽求解里（SV-02）。`opt`/`freq` 走的是另一条路径，如果在这里放行，
            // 用户会拿到一个"溶剂化优化"的假结果 —— 所以显式拒绝，不静默降级。
            if input.jobtype != JobType::Sp {
                return Err(EngineError::Unsupported(format!(
                    "solvent `{}` currently supports `sp` only (solvated gradients/optimisation                      are SV-02); jobtype = `{}`",
                    input.solvent,
                    input.jobtype.as_str()
                )));
            }
        }
        // SC-02 只接了一半：UHF 的解析梯度与 `opt` 通了，UKS 的 XC 自旋拆分还没做。
        // 别让 `uks` + `opt` 走进闭壳层的优化器拿到"看起来成功"的结果。
        if matches!(input.options.scf_type, Some(qwb_engine_api::ScfType::Uhf))
            && !matches!(input.jobtype, JobType::Sp | JobType::Opt)
        {
            // freq/ts/irc/scan 里没有开壳层实现：放过去就是拿闭壳层的 Hessian/路径当真。
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `uhf` supports `sp` and `opt` only (SC-02); jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }
        if matches!(input.options.scf_type, Some(qwb_engine_api::ScfType::Uks))
            && input.jobtype != JobType::Sp
        {
            // UKS 的梯度**装配**已写好并过了闭壳层极限（1.5e-9 Ha/Bohr），但开壳层 UKS SCF
            // （OH/SVWN5/STO-3G）400 步不收敛（DIIS/阻尼/能级移动/初猜缩放都不行），
            // 所以还不接进生产路径 —— 见 docs/choices.md 六十七。
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `uks` currently supports `sp` only: open-shell UKS SCF \
                 convergence is the remaining half of SC-02; jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }
        // FX-04：色散是**可加项**。它只接在单点上 —— 优化器的能量/梯度来自
        // `run_optimisation` 内部的闭包，那里没有色散；在这个函数上事后加会得到一个
        // 「优化时没考虑色散、报出来的能量却含色散」的自相矛盾结果。所以 `opt` 显式
        // 拒绝而不是静默给半个答案（下一步要把它推进优化循环本身）。
        if input.options.dispersion.is_some() && input.jobtype != JobType::Sp {
            return Err(EngineError::Unsupported(format!(
                "options.dispersion currently supports `sp` only: the optimisation loop \
                 evaluates its own energy/gradient, and adding the term after the fact would \
                 report a total that the optimiser never minimised; jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }
        match input.jobtype {
            JobType::Sp => {
                let job = self.run_single_point(input, reporter)?;
                self.apply_dispersion(input, job, reporter)
            }
            JobType::Opt => self.run_optimisation(input, reporter),
            JobType::Freq => self.run_frequencies(input, reporter),
            JobType::Scan => self.run_scan(input, reporter),
            JobType::Ts => self.run_ts(input, reporter),
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

impl RksBackend {
    /// 给单点结果加上 D3(BJ) 色散（`options.dispersion`，FX-04）。
    ///
    /// 三个不变量：
    ///
    /// 1. **总能量 = SCF 能量 + 色散能**，两部分都落盘（`energies.scf_hartree` 与
    ///    `dispersion_energy_hartree`），所以「哪些项加进去了」是可核对的；
    /// 2. 梯度（若有）同步加上色散的解析梯度 —— 不加上去就会出现「能量含色散、
    ///    梯度不含」的半套结果；
    /// 3. 没给这个选项时行为**逐位不变**（阶段一回归不受影响）。
    fn apply_dispersion(
        &self,
        input: &JobInput,
        mut job: JobResult,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        let Some(options) = &input.options.dispersion else {
            return Ok(job);
        };
        let functional = options
            .functional
            .as_deref()
            .unwrap_or(&input.method_name)
            .to_string();
        let symbols: Vec<&str> = input
            .molecule
            .atoms
            .iter()
            .map(|atom| atom.symbol.as_str())
            .collect();
        let positions: Vec<[f64; 3]> = input
            .molecule
            .atoms
            .iter()
            .map(|atom| atom.position_bohr())
            .collect();
        // D3 与 D4 是**两套数据类型**（形状不同，见 `qwb-registry` 的 `d4` 模块），所以按数据键名
        // 分派；D4 还需要体系总电荷（EEQ 电荷是电荷相关的），D3 不需要。
        let (dispersion_energy, dispersion_gradient) = if options.data == "d4" {
            let data = self.registry.get_d4_dispersion("d4")?;
            let total_charge = f64::from(input.molecule.charge);
            let result = qwb_scf::d4::dispersion_gradient(
                data,
                &functional,
                &symbols,
                &positions,
                total_charge,
            )
            .map_err(|err| EngineError::Data(format!("dispersion `{}`: {err}", options.data)))?;
            (result.energy_hartree, result.gradient)
        } else {
            let data = self.registry.get_dispersion(&options.data)?;
            let result =
                qwb_scf::dispersion::dispersion_energy(data, &functional, &symbols, &positions)
                    .map_err(|err| {
                        EngineError::Data(format!("dispersion `{}`: {err}", options.data))
                    })?;
            (result.energy_hartree, result.gradient)
        };
        let scf_hartree = job.energy_hartree;
        job.energy_hartree = scf_hartree + dispersion_energy;
        // 梯度是 3N×1 的矩阵（列主序，按原子优先）。
        if let Some(gradient) = job.gradient_hartree_per_bohr.as_mut()
            && gradient.nrows() == positions.len() * 3
        {
            for atom in 0..positions.len() {
                for axis in 0..3 {
                    gradient[(atom * 3 + axis, 0)] += dispersion_gradient[atom][axis];
                }
            }
        }
        reporter.log(&format!(
            "dispersion = {} (functional = {functional}), E_disp = {:.10} Ha = {:.4} kcal/mol, \
             total = {:.10} Ha",
            options.data,
            dispersion_energy,
            dispersion_energy * 627.509_474_063_1,
            job.energy_hartree
        ));
        let log = format!(
            "{}\nE_scf = {scf_hartree:.12} Ha, E_dispersion = {:.12} Ha, 
",
            job.log, dispersion_energy
        );
        job = job.with_dispersion_energy(dispersion_energy).with_log(log);
        Ok(job)
    }
}

/// 闭壳层电子数（`Σ Z − charge`）。
///
/// MP2 的占据轨道数就是它的一半；奇数电子在闭壳层参考里没有定义，所以显式报错
/// 而不是悄悄取整。
/// ST-05：由 `P_α − P_β` 算 EPR 块（每个核的 Fermi 接触耦合）。
///
/// AO 在核位置的取值由 `qwb-scf` 提供（`qwb-props` 不能反向依赖它），自旋密度在这里减出来。
fn epr_block(
    basis: &qwb_basis::InternalBasis,
    molecule: &qwb_engine_api::Molecule,
    alpha: &qwb_linalg::Matrix,
    beta: &qwb_linalg::Matrix,
    method: &str,
) -> Result<qwb_engine_api::EprResult, EngineError> {
    let n = basis.n_functions();
    let mut spin = qwb_linalg::Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            spin.set(mu, nu, alpha[(mu, nu)] - beta[(mu, nu)]);
        }
    }
    let mut values = Vec::with_capacity(molecule.atoms.len());
    for atom in &molecule.atoms {
        values.push(qwb_scf::rks::ao_value(basis, atom.position_bohr())?);
    }
    let couplings = qwb_props::epr::isotropic_hyperfine(molecule, &values, &spin)?;
    let nuclei = couplings
        .into_iter()
        .map(|coupling| {
            qwb_engine_api::HyperfineCoupling::new(
                coupling.nucleus_index,
                coupling.symbol,
                coupling.spin_density_au,
                coupling.a_iso_mhz,
                coupling.a_iso_gauss,
            )
        })
        .collect();
    Ok(qwb_engine_api::EprResult::new(method, nuclei))
}

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
    if options.scf_type.is_none() && input.molecule.multiplicity != 1 {
        // 缺省路径是闭壳层，它代表不了 `multiplicity > 1` 的自旋态。旧行为会算出一个
        // **能量高 0.616 Ha 的闭壳层态**并返回成功（O2 三重态实测 −147.0158990576 vs
        // ROHF −147.6321670245），那是最坏的一类静默降级：结果看着正常、退出码 0。
        // K2 的 `validate` 也挡一次，但直接走 Engine API 的调用方不经过它。
        return Err(EngineError::InvalidInput(format!(
            "molecule.multiplicity = {} needs an explicit options.scf_type: the default \
             path is closed-shell (rhf/rks) and cannot represent a spin state. \
             Use `rohf` or `uhf` for Hartree-Fock, `uks` for a density functional",
            input.molecule.multiplicity
        )));
    }
    if options.epr {
        // ST-05：EPR 是**自旋密度**的性质。闭壳层的自旋密度恒为零，报出来全是 0 —— 那等于
        // 静默失败，所以显式拒绝（K2 的 `validate` 也查一遍，但直接走 Engine API 的调用方
        // 不经过 `validate`，所以这里必须再挡一次）。
        if input.molecule.multiplicity <= 1 {
            return Err(EngineError::InvalidInput(format!(
                "options.epr needs an open-shell system (multiplicity > 1), got multiplicity = {}",
                input.molecule.multiplicity
            )));
        }
        if input.jobtype != JobType::Sp {
            return Err(EngineError::InvalidInput(format!(
                "options.epr is only available for `sp` jobs, got jobtype = `{}`",
                input.jobtype.as_str()
            )));
        }
    }
    if let Some(scf_type) = options.scf_type {
        // 阶段二已接：闭壳层 KS/RHF（阶段一行为）、**UHF**（SC-01）、**UKS**（SC-01b）、
        // **ROHF**（SC-01c，第 383 轮接线；5/5 体系与 PySCF 一致到 3.6e-8）。ROHF 只在
        // Hartree-Fock 方法上可用（DFT 走 UKS/RKS）。
        let supported = matches!(
            scf_type,
            qwb_engine_api::ScfType::Rks
                | qwb_engine_api::ScfType::Rhf
                | qwb_engine_api::ScfType::Uhf
                | qwb_engine_api::ScfType::Uks
                | qwb_engine_api::ScfType::Rohf
        );
        if !supported {
            // 目前 5 个变体都已接线，这条只剩"将来枚举加新变体"的兜底：不认识的就拒，
            // 不能静默按闭壳层算。
            return Err(EngineError::Unsupported(format!(
                "options.scf_type = `{}` is not wired yet (rks/rhf/uhf/uks/rohf are all \
                 wired; a new variant needs its own WP)",
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
        // M4 / RS-02..03 + ST-01：cis / tda / tddft 三条都接（hf 参考下 cis == tda == TDA，
        // tddft 是含 B 块的完整 RPA）。TDDFT 的 XC kernel（DFT 参考）仍属 FX-03。
        match excited.method {
            qwb_engine_api::ExcitedStateMethod::Cis
            | qwb_engine_api::ExcitedStateMethod::Tda
            | qwb_engine_api::ExcitedStateMethod::Tddft => {}
            _ => {
                return Err(EngineError::Unsupported(format!(
                    "options.excited_states.method = `{}` is not wired yet",
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
                 a DFT reference needs the XC kernel f_xc in the response matrix, which is FX-03 \
                 (ST-01 delivered the B block / RPA coupling itself)",
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
    if let Some(model) = options.solvent_model {
        // SV-01 落地后 cosmo 是**已实装**的（需要 `input.solvent` 给出溶剂名）；
        // SV-03（第 387 轮）起 smd 也接了（CDS 非极性项 + 参数化半径），PBF 仍等 SV-04。
        match model {
            qwb_engine_api::SolventModel::Vacuum
            | qwb_engine_api::SolventModel::Cosmo
            | qwb_engine_api::SolventModel::Smd => {}
            qwb_engine_api::SolventModel::Pbf => {
                return Err(EngineError::Unsupported(
                    "options.solvent_model = `pbf` is not implemented yet (boundary/finite                      element Poisson-Boltzmann is SV-04)"
                        .into(),
                ));
            }
            _ => {
                return Err(EngineError::Unsupported(format!(
                    "options.solvent_model = `{}` is not implemented",
                    model.as_str()
                )));
            }
        }
    }
    Ok(())
}

/// `kind: smd-atomic` 数据在仓库里的名字。
const SMD_ATOMIC_NAME: &str = "smd-atomic";

/// kcal/mol → Hartree（CODATA 2018；与色散/溶剂化日志里用的是同一个数）。
const HARTREE_PER_KCAL_MOL: f64 = 627.509_474_063_6;

/// 频率作业的热化学标准态：298.15 K、101 325 Pa（1 atm）。
///
/// 写成常量而不是 K2 字段：这是**本项目报告的约定**（与 `spectrum_defaults` 不同 ——
/// 那一个是展示约定，需要版本化）。将来要支持变温，就在这里加 K2 字段、并把它从常量
/// 改成读输入；`thermochemistry.temperature_k` / `pressure_pa` 已经落在结果块里，
/// 读的人能核对。
const THERMO_TEMPERATURE_K: f64 = 298.15;
const THERMO_PRESSURE_PA: f64 = 101_325.0;

/// "真虚频"的波数阈值（cm⁻¹）：零模（平动/转动）的数值噪声实测到 −48.7，而鞍点的虚频是
/// −893 量级，所以 −100 把两者分开。低于它的模说明几何不是极小点。
const IMAGINARY_MODE_THRESHOLD_CM: f64 = -100.0;

/// 从 K1 的 `smd:` 块取 SMD 需要的六个溶剂描述符（`SmdDescriptors` 的顺序是
/// SOLA/SOLB/SOLC/SOLG/SOLH/SOLN —— 与上游 `mnsol.F` 一致）。
///
/// 六个里缺任何一个都返回 `None`：**水相**不需要它们（走单独的 σ 表），非水相缺了就由
/// `qwb_props::smd_sigma` 显式拒绝 —— 这里不替数据补齐，也不许拿 0 顶替。
///
/// 注意 `phi_sq`/`psi_sq` 这两个 **K1 字段名**是历史遗留：MNSOL 表给的是 φ 与 ψ 本身
/// （芳香性/卤素性分数），平方在 eq. 4.80 里做（`smd_constant_term` 里就是这个乘法）。
fn smd_descriptors_or_none(
    parameters: Option<&qwb_registry::solvent::SmdParameters>,
) -> Option<qwb_props::smd::SmdDescriptors> {
    let parameters = parameters?;
    Some(qwb_props::smd::SmdDescriptors::new(
        parameters.abraham_alpha?,
        parameters.abraham_beta?,
        parameters.phi_sq?,
        parameters.gamma_cal_per_mol_angstrom2?,
        parameters.psi_sq?,
        parameters.refractive_index_20c?,
    ))
}

/// 依据溶剂的 `radii_set` 选 COSMO/C-PCM 的空腔半径，返回 `(每元素半径, 缩放因子)`。
///
/// * `Some(RadiiSet::Smd)`：用 SMD 的参数化半径（eq. 16，见 [`qwb_props::smd_radii`]），
///   缩放**固定 1.0** —— eq. 16 给出的就是最终半径，再乘 Bondi 那套的 `scale` 会重复放大；
///   O 半径由溶剂的 Abraham α 决定，所以缺 `abraham_alpha` 或 `smd-atomic` 数据时**显式报错**。
/// * 其它（`bondi` / `uff` / 未声明）：返回文件里的 `atomic_radii` 与文件里的 `scale`，
///   行为与第 375 轮之前**逐位相同**。
///
/// # Errors
///
/// `radii_set = smd` 但缺 `smd-atomic` 数据或溶剂的 `abraham_alpha` → [`EngineError::Data`]。
pub fn cosmo_radii(
    molecule: &Molecule,
    radii_set: Option<qwb_registry::RadiiSet>,
    atomic_radii: &BTreeMap<String, f64>,
    scale: f64,
    alpha: Option<f64>,
    smd: Option<&qwb_registry::smd_atomic::SmdAtomic>,
) -> Result<(BTreeMap<String, f64>, f64), EngineError> {
    if radii_set != Some(qwb_registry::RadiiSet::Smd) {
        return Ok((atomic_radii.clone(), scale));
    }
    let data = smd.ok_or_else(|| {
        EngineError::Data(
            "solvent declares radii_set = smd but the `smd-atomic` data is not loaded".to_string(),
        )
    })?;
    let alpha = alpha.ok_or_else(|| {
        EngineError::Data(
            "solvent declares radii_set = smd but has no `smd.abraham_alpha` (the oxygen radius \
             depends on it)"
                .to_string(),
        )
    })?;
    let per_atom = qwb_props::smd_radii::smd_parameterised_radii(data, molecule, alpha)?;
    let mut radii = BTreeMap::new();
    for (atom, radius) in molecule.atoms.iter().zip(per_atom) {
        radii.insert(atom.symbol.clone(), radius);
    }
    Ok((radii, 1.0))
}
