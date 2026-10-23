//! 网格管理与缓存（X-04）。
//!
//! 设计要点：
//!
//! * 网格用 `Arc<Grid>` 持有，调用方可以廉价地共享；
//! * **切换等级时旧网格必须被释放**——这是验收判据，用 `Arc::strong_count` 断言；
//! * 新网格先构建、成功了再替换，所以构建失败不会把当前网格弄丢（也不会有半替换状态）。

use std::sync::Arc;

use qwb_engine_api::Molecule;
use qwb_registry::grid::GridLevelDef;

use crate::{Grid, GridError, molecular_grid};

/// 一个「当前网格」缓存：持有某个等级下某分子的网格。
#[derive(Debug, Clone)]
pub struct GridCache {
    label: String,
    grid: Arc<Grid>,
}

impl GridCache {
    /// 构建并缓存一个网格。
    ///
    /// # Errors
    ///
    /// 网格构建失败（见 [`molecular_grid`]）。
    pub fn build(
        label: impl Into<String>,
        level: &GridLevelDef,
        molecule: &Molecule,
    ) -> Result<Self, GridError> {
        let grid = molecular_grid(level, molecule)?;
        Ok(Self {
            label: label.into(),
            grid: Arc::new(grid),
        })
    }

    /// 当前网格的等级标签。
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// 共享当前网格（`Arc` 克隆，不复制点）。
    #[must_use]
    pub fn grid(&self) -> Arc<Grid> {
        Arc::clone(&self.grid)
    }

    /// 当前是否缓存着 `label` 这个等级的网格。
    #[must_use]
    pub fn is(&self, label: &str) -> bool {
        self.label == label
    }

    /// 切到另一个等级：**先构建成功再替换**，旧的 `Arc` 在替换时丢弃。
    ///
    /// 若新等级构建失败，当前缓存保持不变（返回错误）。
    ///
    /// # Errors
    ///
    /// 新网格构建失败。
    pub fn switch(
        &mut self,
        label: impl Into<String>,
        level: &GridLevelDef,
        molecule: &Molecule,
    ) -> Result<(), GridError> {
        let fresh = molecular_grid(level, molecule)?;
        self.label = label.into();
        self.grid = Arc::new(fresh);
        Ok(())
    }

    /// 当前网格被多少处共享（1 = 只有缓存自己）。
    #[must_use]
    pub fn shared_by(&self) -> usize {
        Arc::strong_count(&self.grid)
    }
}
