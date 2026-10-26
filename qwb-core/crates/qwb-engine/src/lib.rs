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

pub mod env;
pub mod gradient;
pub mod hessian;
pub mod opt;

use std::sync::{Arc, Mutex};

use qwb_engine_api::{
    EngineBackend, EngineError, JobInput, JobResult, JobType, Molecule, Reporter,
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

    /// 单点能（`Sp` 的全部逻辑，也是 `opt` 收尾时复用的那一段）。
    fn run_single_point(
        &self,
        input: &JobInput,
        reporter: &dyn Reporter,
    ) -> Result<JobResult, EngineError> {
        reporter.progress("load", 0.0);
        let basis_k1 = self.registry.get_basis(&input.basis_name)?;
        let spec = self.spec(input)?;
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
                // 单初猜不收敛时的三级退路（近简并体系：N2、含 d/f 的重元素、ECP 体系）。
                // 三条都**确定**：初猜序列/温度/步数全固定，没有随机化 ⇒ 同样的输入给同样的输出。
                //
                // 顺序是有讲究的：**先试退火**（费米展宽跑若干步再切回整数占据）。
                // 它最便宜（ZnO/LANL2DZ 只要 13 步就收敛），而且正好治"初始猜测落在
                // 伪组态里"这个最常见的死法；多初猜要跑 5 次完整 SCF，放后面。
                reporter.log(&format!(
                    "scf: single guess failed ({first_error}), retrying with annealed smearing"
                ));
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
                    Ok(result) => result,
                    Err(second_error) => {
                        reporter.log(&format!(
                            "scf: annealing failed ({second_error}), retrying with multistart"
                        ));
                        match qwb_scf::rks::rks_scf_multistart(
                            &basis,
                            &input.molecule,
                            &tensor,
                            &grid,
                            &spec.terms,
                            options.clone(),
                        ) {
                            Ok(result) => result,
                            Err(third_error) => {
                                reporter.log(&format!(
                                    "scf: multistart failed ({third_error}), retrying with damped \
                                     Roothaan iterations (no DIIS, damping 0.3)"
                                ));
                                let mut plain = options.clone();
                                plain.use_diis = false;
                                plain.damping = 0.3;
                                // **不改用户给的 `scf_max_iter`**：K2 的 `TaskOptions` 是输入契约，
                                // 引擎不能"为了收敛"偷偷跑更多步
                                // （`test_scf_not_converged_carries_iters` 就是钉这一点的）。
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
        reporter.progress("done", 100.0);

        let env = env::build(&self.registry);
        // S-06：偶极矩（Debye）——中性分子对原点不变，`qwb-props` 的测试里有这条自检。
        let dipole = qwb_props::dipole_debye(&basis, &input.molecule, &result.density)?;
        Ok(JobResult::new(
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
        )))
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
        if input.solvent != "vacuum" {
            return Err(EngineError::Unsupported(format!(
                "solvent `{}` is not implemented (first phase: vacuum only, 8.5)",
                input.solvent
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
