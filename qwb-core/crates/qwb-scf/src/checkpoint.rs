//! SCF 检查点（S-05）：把迭代状态存盘，被 kill 之后可以从那里继续。
//!
//! 只存**恢复迭代所必需**的东西：密度矩阵、当前能量、已完成的迭代数。
//! 不存 DIIS 子空间——恢复后 DIIS 从空开始，轨迹会略有不同，但自洽解是同一个
//! （验收要求「结果 identical」，用收敛能量的差来判，不要求逐位相同的轨迹）。
//!
//! 格式是 JSON（serde），密度按**列主序**展平，便于和 `faer`/`nalgebra` 互转。

use std::path::Path;

use qwb_engine_api::EngineError;
use qwb_linalg::Matrix;
use serde::{Deserialize, Serialize};

/// SCF 的可续算状态。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScfCheckpoint {
    /// 密度矩阵（列主序展平，长度 `n*n`）。
    pub density: Vec<f64>,
    /// 当前总能量（Hartree）；`null` 表示「还没有能量」（首轮用）。
    #[serde(default)]
    pub energy_hartree: Option<f64>,
    /// 已完成的迭代步数。
    pub iterations: u32,
    /// 密度矩阵维数（校验用）。
    pub n_functions: usize,
    /// DIIS 子空间的 Fock 矩阵（列主序展平）；恢复后轨迹才能与不中断的跑法逐位一致。
    #[serde(default)]
    pub diis_focks: Vec<Vec<f64>>,
    /// DIIS 子空间的误差向量（列主序展平）。
    #[serde(default)]
    pub diis_errors: Vec<Vec<f64>>,
}

impl ScfCheckpoint {
    /// 从密度矩阵与能量构造。
    #[must_use]
    pub fn new(density: &Matrix, energy_hartree: Option<f64>, iterations: u32) -> Self {
        Self {
            density: density.as_column_major().to_vec(),
            energy_hartree,
            iterations,
            n_functions: density.nrows(),
            diis_focks: Vec::new(),
            diis_errors: Vec::new(),
        }
    }

    /// 附带 DIIS 子空间（`ScfCheckpoint::new` 之后链式调用）。
    #[must_use]
    pub fn with_diis(mut self, focks: Vec<Vec<f64>>, errors: Vec<Vec<f64>>) -> Self {
        self.diis_focks = focks;
        self.diis_errors = errors;
        self
    }

    /// 还原密度矩阵。
    ///
    /// # Errors
    ///
    /// 展平长度与 `n_functions` 不一致（文件被改坏）。
    pub fn density_matrix(&self) -> Result<Matrix, EngineError> {
        if self.density.len() != self.n_functions * self.n_functions {
            return Err(EngineError::Data(format!(
                "scf checkpoint: density has {} values but n_functions = {} implies {}",
                self.density.len(),
                self.n_functions,
                self.n_functions * self.n_functions
            )));
        }
        Matrix::from_column_major(self.n_functions, self.n_functions, self.density.clone())
            .map_err(|err| EngineError::Data(format!("scf checkpoint: {err}")))
    }
}

/// 写检查点（先写临时文件再改名，避免中途被 kill 留下半个 JSON）。
///
/// # Errors
///
/// 序列化或写盘失败。
pub fn save_checkpoint(path: &Path, checkpoint: &ScfCheckpoint) -> Result<(), EngineError> {
    let json = serde_json::to_vec(checkpoint)
        .map_err(|err| EngineError::Internal(format!("scf checkpoint serialise: {err}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            EngineError::Internal(format!("scf checkpoint mkdir {}: {err}", parent.display()))
        })?;
    }
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, &json).map_err(|err| {
        EngineError::Internal(format!(
            "scf checkpoint write {}: {err}",
            temporary.display()
        ))
    })?;
    std::fs::rename(&temporary, path).map_err(|err| {
        EngineError::Internal(format!("scf checkpoint rename {}: {err}", path.display()))
    })?;
    Ok(())
}

/// 读检查点。
///
/// # Errors
///
/// 文件缺失、不是合法 JSON、或字段不自洽。
pub fn load_checkpoint(path: &Path) -> Result<ScfCheckpoint, EngineError> {
    let bytes = std::fs::read(path).map_err(|err| {
        EngineError::Internal(format!("scf checkpoint read {}: {err}", path.display()))
    })?;
    let checkpoint: ScfCheckpoint = serde_json::from_slice(&bytes).map_err(|err| {
        EngineError::Data(format!(
            "scf checkpoint {} is not a valid checkpoint: {err}",
            path.display()
        ))
    })?;
    // 立刻验一次密度，避免把坏文件带进 SCF。
    checkpoint.density_matrix()?;
    Ok(checkpoint)
}
