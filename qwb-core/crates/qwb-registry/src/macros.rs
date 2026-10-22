//! 内部宏：为数据类型实现 [`crate::common::DataCommon`]。
//!
//! # 为什么共同字段是“复制粘贴”的
//!
//! K1 的共同字段（`name`/`version`/`citation`/`license`/`created`/`sha256`/`notes`）
//! 在每个数据类型里**直接内联声明**，原因有两条：
//!
//! 1. `#[serde(flatten)]` + `deny_unknown_fields` 不兼容（serde 明确不支持），
//!    而本项目要求“字段拼错必须报错”；
//! 2. Rust 的 `macro_rules!` **不能出现在 struct 的字段列表位置**（只在 item /
//!    type / expr / pattern 位置可用），所以没法用宏展开字段。
//!
//! 复制粘贴的风险由**测试**兜底，而不是靠约定：`schema.rs` 的
//! `common_fields_are_identical_across_kinds` 断言 5 个数据类别的共同字段在生成的
//! JSON Schema 里逐字节一致 —— 任何一处漏字段或改类型都会让 C-01 验收失败。

/// 为具体类型实现 [`crate::common::DataCommon`]。
macro_rules! impl_data_common {
    ($t:ty) => {
        impl crate::common::DataCommon for $t {
            fn name(&self) -> &str {
                &self.name
            }
            fn version(&self) -> &str {
                &self.version
            }
            fn citations(&self) -> &[crate::common::Citation] {
                &self.citation
            }
            fn license(&self) -> &crate::common::License {
                &self.license
            }
            fn created(&self) -> &str {
                &self.created
            }
            fn declared_sha256(&self) -> Option<&str> {
                self.sha256.as_deref()
            }
        }
    };
}
