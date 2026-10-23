//! 确定性并行的原语（G7 / R-13）。
//!
//! 规则：**并行归约必须与线程数无关**。rayon 的无序 `reduce`/`fold` 依赖线程调度，
//! 浮点加法不满足结合律，结果会随线程数漂移；因此这里只提供「按索引映射」：
//!
//! ```text
//! DeterministicReduce::map_ordered(items, f) -> Vec<R>
//! ```
//!
//! 它保证**输出顺序与输入索引严格一致**（`par_iter().map().collect()` 的语义），
//! 于是随后的任何累加都可以在单线程里按索引顺序做，结果与线程数无关。
//! 需要求和时用 [`DeterministicReduce::sum_ordered`]，它内部也是按索引顺序累加。
//!
//! 本模块是整个 crate 里唯一允许出现 `par_iter` 的地方（`check_rayon_determinism.sh`
//! 的规则 2 会检查这一点），避免并行模式散落各处、各自发明累加顺序。

use rayon::prelude::*;

/// 确定性并行的门面类型（名字即契约，见模块文档）。
pub struct DeterministicReduce;

impl DeterministicReduce {
    /// 按索引映射：输出 `Vec<R>` 的第 `i` 项严格对应 `items[i]`。
    #[must_use]
    pub fn map_ordered<T, R, F>(items: &[T], map: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync + Send,
    {
        items.par_iter().map(map).collect()
    }

    /// 按索引顺序求和：先并行算出每一项，再**串行**按索引累加。
    ///
    /// 与 `par_iter().map(..).sum()` 不同，后者的累加顺序由调度决定。
    #[must_use]
    pub fn sum_ordered<T, F>(items: &[T], map: F) -> f64
    where
        T: Sync,
        F: Fn(&T) -> f64 + Sync + Send,
    {
        Self::map_ordered(items, map).into_iter().sum()
    }
}
