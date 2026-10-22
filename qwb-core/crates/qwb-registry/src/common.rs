//! K1 共同字段：引用（citation）与许可（license），以及所有数据类型共享的校验。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// 一条引用来源（§4 K1：每项 `title` + `doi`/`url`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Citation {
    /// 文献/数据标题。
    pub title: String,
    /// DOI（与 `url` 至少给一个）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doi: Option<String>,
    /// URL（与 `doi` 至少给一个）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl Citation {
    /// 构造一条带 DOI 的引用。
    #[must_use]
    pub fn with_doi(title: impl Into<String>, doi: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            doi: Some(doi.into()),
            url: None,
        }
    }

    /// 构造一条带 URL 的引用。
    #[must_use]
    pub fn with_url(title: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            doi: None,
            url: Some(url.into()),
        }
    }

    /// 是否满足“有标题且有 doi 或 url”。
    #[must_use]
    pub fn is_complete(&self) -> bool {
        let has_title = !self.title.trim().is_empty();
        let has_locator = self.doi.as_deref().is_some_and(|d| !d.trim().is_empty())
            || self.url.as_deref().is_some_and(|u| !u.trim().is_empty());
        has_title && has_locator
    }
}

/// 许可声明（§4 K1 / §8.6：`commercial_ok` / `redistribute_ok` 必须显式）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct License {
    /// SPDX 标识（如 `Apache-2.0`、`CC-BY-4.0`）。
    pub spdx: String,
    /// 是否允许商用。
    pub commercial_ok: bool,
    /// 是否允许再分发。
    pub redistribute_ok: bool,
}

impl License {
    /// 构造许可声明。
    #[must_use]
    pub fn new(spdx: impl Into<String>, commercial_ok: bool, redistribute_ok: bool) -> Self {
        Self {
            spdx: spdx.into(),
            commercial_ok,
            redistribute_ok,
        }
    }

    /// SPDX 标识非空。
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.spdx.trim().is_empty()
    }
}

/// 所有 K1 数据类型共有的元信息访问面（字段由 `k1_common_fields!` 内联）。
pub trait DataCommon {
    /// 键名。
    fn name(&self) -> &str;
    /// 版本。
    fn version(&self) -> &str;
    /// 引用列表。
    fn citations(&self) -> &[Citation];
    /// 许可。
    fn license(&self) -> &License;
    /// 创建时间。
    fn created(&self) -> &str;
    /// 文件内声明的 sha256（工具生成，可能尚未生成）。
    fn declared_sha256(&self) -> Option<&str>;

    /// G2 数据门禁的通用部分：citation 非空且每项完整、license 完整、版本/时间非空。
    ///
    /// 返回人类可读的问题列表；空 `Vec` 表示通过。
    fn check_common(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.name().trim().is_empty() {
            issues.push("name must not be empty".to_string());
        }
        if self.version().trim().is_empty() {
            issues.push("version must not be empty".to_string());
        }
        if self.created().trim().is_empty() {
            issues.push("created must not be empty".to_string());
        }
        if self.citations().is_empty() {
            issues.push("citation must contain at least one entry (G2)".to_string());
        }
        for (i, cite) in self.citations().iter().enumerate() {
            if !cite.is_complete() {
                issues.push(format!(
                    "citation[{i}] needs a non-empty title and at least one of doi/url (G2)"
                ));
            }
        }
        if !self.license().is_complete() {
            issues.push("license.spdx must not be empty (G2)".to_string());
        }
        if self.declared_sha256().is_none() {
            issues.push(
                "sha256 is missing: run `qwb data hash` / `gen-schemas` to stamp it (G2)"
                    .to_string(),
            );
        }
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::{Citation, License};

    #[test]
    fn citation_needs_title_and_locator() {
        assert!(Citation::with_doi("T", "10.1000/x").is_complete());
        assert!(Citation::with_url("T", "https://x").is_complete());
        assert!(!Citation::with_doi("", "10.1000/x").is_complete());
        assert!(!Citation::with_doi("T", "   ").is_complete());
        let bare = Citation {
            title: "T".into(),
            doi: None,
            url: None,
        };
        assert!(!bare.is_complete());
    }

    #[test]
    fn license_needs_spdx() {
        assert!(License::new("Apache-2.0", true, true).is_complete());
        assert!(!License::new(" ", true, true).is_complete());
    }
}
