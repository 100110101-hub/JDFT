//! 观测面（K2）：进度、日志、取消。引擎**不得**直接写 stdout。

/// 引擎执行期间的回调面。
///
/// 引擎只通过它对外说话；CLI/GUI/Python 各自实现（K6 要求 Python 侧批量推送，
/// 禁止逐行回调跨 GIL）。
pub trait Reporter: Send + Sync {
    /// 报告阶段进度，`pct` 取值 `0.0..=100.0`。
    fn progress(&self, stage: &str, pct: f64);

    /// 追加一行日志。
    fn log(&self, line: &str);

    /// 是否已被外部请求取消。引擎需在迭代边界轮询。
    fn is_cancelled(&self) -> bool;
}

/// 什么都不做的 [`Reporter`]（测试与批处理静默模式用）。
#[derive(Debug, Clone, Copy, Default)]
pub struct NullReporter;

impl Reporter for NullReporter {
    fn progress(&self, _stage: &str, _pct: f64) {}

    fn log(&self, _line: &str) {}

    fn is_cancelled(&self) -> bool {
        false
    }
}
