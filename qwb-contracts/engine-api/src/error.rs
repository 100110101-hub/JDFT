//! 引擎错误（K2；变体只允许新增，禁止删除）。

use thiserror::Error;

/// 引擎执行错误。
///
/// 变体集合参与 K5 退出码映射（见 [`EngineError::exit_code`]），
/// 因此**只允许新增变体**，禁止删除或改变既有变体语义。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EngineError {
    /// 输入语义非法：YAML/JSON 可解析，但取值越界（多重度 0、阈值 <= 0、键名为空……）。
    ///
    /// 对应 K5 退出码 `1`。语法层面的解析失败由调用方（CLI/Python）自己报，不进引擎。
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// 数据缺失、schema 不符、sha256 不匹配、citation/license 缺失。
    #[error("data load failed: {0}")]
    Data(String),

    /// SCF 未在 `max` 步内收敛。
    #[error("scf not converged in {iters} iterations (max {max})")]
    ScfNotConverged {
        /// 实际迭代步数。
        iters: u32,
        /// 允许的最大步数。
        max: u32,
    },

    /// 该构建或该数据组合未支持的 feature（如 meta-GGA、PCM、ECP 未实现）。
    #[error("unsupported feature: {0}")]
    Unsupported(String),

    /// IO 失败。
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// 内部不变量被破坏（bug，不是用户输入问题）。
    #[error("internal error: {0}")]
    Internal(String),
}

impl EngineError {
    /// 映射到 K5 冻结的 CLI 退出码。
    ///
    /// | 变体 | 退出码 | K5 含义 |
    /// |---|---|---|
    /// | [`EngineError::InvalidInput`] | `1` | 输入错误 |
    /// | [`EngineError::ScfNotConverged`] | `2` | 收敛失败 |
    /// | [`EngineError::Data`] | `3` | 数据校验失败 |
    /// | [`EngineError::Unsupported`] | `4` | 内部错误（不支持的特性） |
    /// | [`EngineError::Io`] | `4` | 内部错误 |
    /// | [`EngineError::Internal`] | `4` | 内部错误 |
    ///
    /// `0` 表示无错误，不由本方法返回。
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::InvalidInput(_) => 1,
            Self::ScfNotConverged { .. } => 2,
            Self::Data(_) => 3,
            Self::Unsupported(_) | Self::Io(_) | Self::Internal(_) => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EngineError;

    #[test]
    fn k5_exit_code_mapping_is_frozen() {
        assert_eq!(EngineError::InvalidInput("x".into()).exit_code(), 1);
        assert_eq!(
            EngineError::ScfNotConverged { iters: 9, max: 8 }.exit_code(),
            2
        );
        assert_eq!(EngineError::Data("x".into()).exit_code(), 3);
        assert_eq!(EngineError::Unsupported("x".into()).exit_code(), 4);
        assert_eq!(EngineError::Io(std::io::Error::other("x")).exit_code(), 4);
        assert_eq!(EngineError::Internal("x".into()).exit_code(), 4);
    }
}
