//! 引擎顶层：实现 K2 的 [`EngineBackend`]，把注册表数据接到 SCF 上。
//!
//! 这一层是 T1 的最后一环：`JobInput` → 加载基组/泛函/网格 → RKS → `JobResult`
//! （含 `env` 与 `data_snapshot`，G5 要求）。
//!
//! # 第一阶段能力（写清楚，免得误以为支持）
//!
//! - **只有 `JobType::Sp`**（单点能）；`Opt`/`Freq` 属 G/F 轨。
//! - **只有闭壳层 LDA + vacuum**（GGA 需要 σ，溶剂属 §8.5 的接口-only）。
//! - ECP 未接入（I-05），`options.ecp_name = Some(_)` 显式拒绝。
//! - 无并行（rayon 属 I-04/X-03 的并行项），`threads` 记 1。

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod env;
pub mod gradient;

use qwb_engine_api::{EngineBackend, EngineError, JobInput, JobResult, JobType, Reporter};

/// 闭壳层 RKS 后端。
pub struct RksBackend {
    registry: qwb_registry::Registry,
}

impl RksBackend {
    /// 用已加载的数据注册表构造后端。
    #[must_use]
    pub fn new(registry: qwb_registry::Registry) -> Self {
        Self { registry }
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

impl EngineBackend for RksBackend {
    fn run(&self, input: &JobInput, reporter: &dyn Reporter) -> Result<JobResult, EngineError> {
        if !matches!(input.jobtype, JobType::Sp) {
            return Err(EngineError::Unsupported(format!(
                "only single-point jobs are implemented (jobtype = {}); optimisation and \
                 frequencies belong to the G/F tracks",
                input.jobtype.as_str()
            )));
        }
        if input.options.ecp_name.is_some() {
            return Err(EngineError::Unsupported(
                "ECP is not implemented yet (I-05)".to_string(),
            ));
        }
        if input.solvent != "vacuum" {
            return Err(EngineError::Unsupported(format!(
                "solvent `{}` is not implemented (first phase: vacuum only, 8.5)",
                input.solvent
            )));
        }

        reporter.progress("load", 0.0);
        let basis_k1 = self.registry.get_basis(&input.basis_name)?;
        let functional = self.registry.get_functional(&input.method_name)?;
        let grid_level = self
            .registry
            .get_grid_by_level(grid_level_from_api(input.grid_level))?;
        reporter.log(&format!(
            "basis = {} ({} elements), method = {}, grid = {}, {} atoms",
            basis_k1.name,
            basis_k1.elements.len(),
            functional.name,
            grid_level.name,
            input.molecule.n_atoms()
        ));

        reporter.progress("integrals", 25.0);
        let basis = qwb_basis::InternalBasis::build(basis_k1, &input.molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, &input.molecule)?;
        let grid = qwb_grid::molecular_grid(grid_level, &input.molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;

        let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
        for component in &functional.components {
            if (component.weight - 1.0).abs() > 1e-12 {
                return Err(EngineError::Unsupported(format!(
                    "weighted functional components are not implemented yet (needed for \
                     B3LYP, T6): `{}` has weight {}",
                    component.id, component.weight
                )));
            }
            terms.push(
                qwb_xc::build_term(&component.id, &functional.parameters).map_err(|err| {
                    EngineError::Data(format!("functional `{}`: {err}", functional.name))
                })?,
            );
        }

        reporter.progress("scf", 40.0);
        reporter.log(&format!("grid points = {}", grid.len()));

        // `RksOptions` 是 non_exhaustive：先 Default 再按字段覆盖。
        let mut options = qwb_scf::rks::RksOptions::default();
        options.max_iter = input.options.scf_max_iter;
        options.energy_threshold = input.options.scf_energy_threshold_hartree;
        options.density_threshold = input.options.scf_density_threshold;

        let result =
            qwb_scf::rks::rks_scf(&basis, &input.molecule, &tensor, &grid, &terms, options)?;
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
            density_to_api(&result.density),
        )
        .with_log(format!(
            "method = {}, basis = {}, grid = {}, iterations = {}, E_elec = {:.10}, E_xc = {:.10}, dipole = [{:.4}, {:.4}, {:.4}] D",
            functional.name,
            basis_k1.name,
            grid_level.name,
            result.iterations,
            result.electronic_hartree,
            result.exchange_correlation_hartree,
            dipole[0],
            dipole[1],
            dipole[2]
        )))
    }

    fn request_cancel(&self) {
        // 单点能没有可取消的异步阶段；取消在 `Engine::run` 启动前由
        // `Reporter::is_cancelled` 检查。
    }
}
