//! 基组数据（§4 K1 / §8.1）。
//!
//! # K1 formalization（见 `docs/choices.md`）
//!
//! K1 写的是“`elements[]` + `shells[]`”。但 `qwb-data` 的基组文件**可以覆盖多元素**
//! （D-03：STO-3G 覆盖 H/C/N/O/F），平坦的 `shells[]` 无法区分元素。因此这里
//! `shells` 的元素类型是 [`ShellBlock`]（元素 + 该元素的壳层），字段名保持 `shells`。
//!
//! # 归一化（§8.1“归一化只有一处”）
//!
//! `normalization` 字段声明该文件里的收缩系数是否已归一化。加载器在**加载时**一次性
//! 处理，运行时不转换；`Engine` 显式读取 `angular`，禁止隐式默认（R-06）。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};
use crate::elements;

/// 球谐 vs 笛卡尔（§8.1 / R-06：唯一入口就是这个字段）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Angular {
    /// 球谐（`(2l+1)` 个函数）。
    Spherical,
    /// 笛卡尔（`(l+1)(l+2)/2` 个函数）。
    Cartesian,
}

/// 收缩系数的归一化约定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Normalization {
    /// 每个收缩函数已按 L2 归一（文献与 BSE 导出的默认）。
    L2,
    /// 未归一化，加载器负责归一。
    None,
}

/// 基组角色（§4 K1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum BasisRole {
    /// 轨道基组。
    #[serde(rename = "orbital")]
    Orbital,
    /// RI-J 辅助基组。
    #[serde(rename = "ri-j")]
    RiJ,
    /// RI-JK 辅助基组。
    #[serde(rename = "ri-jk")]
    RiJk,
    /// ECP 拟合基组。
    #[serde(rename = "ecp-fit")]
    EcpFit,
}

/// 收缩类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ContractionType {
    /// 分段收缩：每个收缩函数只用自己的一段指数。
    Segmented,
    /// 通用收缩：多个收缩函数共享同一套指数（def2-TZVP 等）。
    General,
}

/// 一个壳层。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Shell {
    /// 角量子数 `l`（0=s, 1=p, ...）。第一阶段 `l <= 4`。
    pub l: u32,
    /// 指数（Bohr^-2），从大到小。
    pub exponents: Vec<f64>,
    /// 收缩系数：**每个内层列表是一个收缩函数**，长度必须等于 `exponents.len()`。
    ///
    /// 分段收缩 = 每个收缩函数只有一个非零项；通用收缩 = 多个收缩函数共享指数。
    /// K1 只写了 `coefficients[]`；要同时表达 def2-TZVP 的通用收缩（D-05）必须是二维。
    pub coefficients: Vec<Vec<f64>>,
    /// 收缩类型。
    pub contraction_type: ContractionType,
}

impl Shell {
    /// 收缩函数个数（= 该壳层贡献的基函数个数 × (2l+1) 之前的值）。
    #[must_use]
    pub fn n_contractions(&self) -> usize {
        self.coefficients.len()
    }

    /// 该壳层对基函数数的贡献（按球谐计）。
    #[must_use]
    pub fn n_basis_functions(&self, angular: Angular) -> usize {
        let per = match angular {
            Angular::Spherical => 2 * self.l as usize + 1,
            Angular::Cartesian => {
                let l = self.l as usize;
                (l + 1) * (l + 2) / 2
            }
        };
        per * self.n_contractions()
    }
}

/// 一个元素的全部壳层（`BasisSet::shells` 的元素类型）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ShellBlock {
    /// 元素符号（必须在 `BasisSet::elements` 里）。
    pub element: String,
    /// 该元素的壳层，按 `l` 升序。
    pub shells: Vec<Shell>,
}

/// 基组（`qwb-data/basis/*.yaml` 的形态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct BasisSet {
    /// 数据键名（数据目录下的标识）。
    pub name: String,
    /// 数据版本（独立 semver；§8.6 要求变更即 bump）。
    pub version: String,
    /// 引用来源，至少 1 条；每条需 `title` + `doi` 或 `url`。
    pub citation: Vec<Citation>,
    /// 许可声明。
    pub license: License,
    /// 创建/修订时间，ISO 8601 UTC。
    pub created: String,
    /// 内容摘要（工具生成，见 [`crate::sha256`]）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// 备注/来源说明（可选）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// 覆盖的元素符号，如 `["H", "C", "N", "O", "F"]`。
    pub elements: Vec<String>,
    /// 球谐/笛卡尔（唯一入口，R-06）。
    pub angular: Angular,
    /// 归一化约定。
    pub normalization: Normalization,
    /// 角色。
    pub role: BasisRole,
    /// 每元素一个块，必须与 `elements` 一一对应。
    pub shells: Vec<ShellBlock>,
}

impl_data_common!(BasisSet);

impl BasisSet {
    /// 该基组的基函数总数（按 `angular` 计）。
    #[must_use]
    pub fn n_basis_functions(&self) -> usize {
        self.shells
            .iter()
            .flat_map(|b| b.shells.iter())
            .map(|s| s.n_basis_functions(self.angular))
            .sum()
    }

    /// 某元素的壳层块。
    #[must_use]
    pub fn shells_for(&self, element: &str) -> Option<&ShellBlock> {
        self.shells.iter().find(|b| b.element == element)
    }

    /// 某元素贡献的基函数数（按 `angular` 计）；元素不在本基组里返回 `None`。
    ///
    /// I-01 的验收（"H2/STO-3G AO 数 = 2；C/STO-3G AO 数 = 5"）就靠它。
    #[must_use]
    pub fn n_basis_functions_for(&self, element: &str) -> Option<usize> {
        self.shells_for(element).map(|block| {
            block
                .shells
                .iter()
                .map(|shell| shell.n_basis_functions(self.angular))
                .sum()
        })
    }

    /// 完整校验（`qwb data verify` 用）：G2 门禁 + 结构检查。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = self.check_common();
        issues.extend(self.check_structure());
        issues
    }

    /// 结构性检查：**不含** G2 的 citation/license/sha256 门禁。
    ///
    /// 这是 `Registry::open` 使用的检查（§8.6：load 只要求文件语法/结构可解析，
    /// 空注册表、缺 citation 的文件都不该让加载器不可用）。
    #[must_use]
    pub fn check_structure(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.elements.is_empty() {
            issues.push("elements must not be empty".to_string());
        }
        let mut seen = std::collections::BTreeSet::new();
        for element in &self.elements {
            if elements::atomic_number(element).is_none() {
                issues.push(format!(
                    "elements contains unknown symbol `{element}` (first phase covers H..Kr)"
                ));
            }
            if !seen.insert(element.as_str()) {
                issues.push(format!("elements contains duplicate `{element}`"));
            }
        }

        if self.shells.len() != self.elements.len() {
            issues.push(format!(
                "shells has {} blocks but elements has {} entries: exactly one block per element is required",
                self.shells.len(),
                self.elements.len()
            ));
        }
        for block in &self.shells {
            if !self.elements.iter().any(|e| e == &block.element) {
                issues.push(format!(
                    "shells block for `{}` is not listed in elements",
                    block.element
                ));
            }
            if block.shells.is_empty() {
                issues.push(format!("shells block for `{}` is empty", block.element));
            }
            let mut last_l = 0;
            for (i, shell) in block.shells.iter().enumerate() {
                let at = format!("shells[{}].shells[{i}]", block.element);
                if shell.l > 4 {
                    issues.push(format!("{at}.l = {} exceeds first-phase limit 4", shell.l));
                }
                if shell.l < last_l {
                    issues.push(format!("{at} is not sorted by l ascending"));
                }
                last_l = shell.l;
                if shell.exponents.is_empty() {
                    issues.push(format!("{at}.exponents must not be empty"));
                }
                if shell.exponents.iter().any(|e| !e.is_finite() || *e <= 0.0) {
                    issues.push(format!("{at}.exponents must all be positive finite"));
                }
                if shell
                    .exponents
                    .windows(2)
                    .any(|w| w[0].is_finite() && w[1].is_finite() && w[0] < w[1])
                {
                    issues.push(format!(
                        "{at}.exponents must be ordered large -> small (§8.1)"
                    ));
                }
                if shell.coefficients.is_empty() {
                    issues.push(format!(
                        "{at}.coefficients must contain at least one contraction"
                    ));
                }
                for (j, column) in shell.coefficients.iter().enumerate() {
                    if column.len() != shell.exponents.len() {
                        issues.push(format!(
                            "{at}.coefficients[{j}] has {} values but exponents has {}",
                            column.len(),
                            shell.exponents.len()
                        ));
                    }
                    if column.iter().any(|c| !c.is_finite()) {
                        issues.push(format!("{at}.coefficients[{j}] contains non-finite values"));
                    }
                    if column.iter().all(|c| *c == 0.0) {
                        issues.push(format!("{at}.coefficients[{j}] is all zeros"));
                    }
                }
                if strip_zeros(shell).is_empty() {
                    issues.push(format!("{at} has no non-zero contraction"));
                }
                match shell.contraction_type {
                    ContractionType::Segmented => {
                        let non_zero_per_contraction = shell
                            .coefficients
                            .iter()
                            .map(|c| c.iter().filter(|v| **v != 0.0).count())
                            .collect::<Vec<_>>();
                        if non_zero_per_contraction.iter().any(|n| *n != 1)
                            && shell.n_contractions() > 1
                        {
                            issues.push(format!(
                                "{at} declares segmented contraction but a contraction has more than one non-zero coefficient"
                            ));
                        }
                    }
                    ContractionType::General => {}
                }
            }
        }

        // 每个元素都必须有块（上面已按数量比较，这里给出更具体的定位）。
        for element in &self.elements {
            if self.shells_for(element).is_none() {
                issues.push(format!("no shells block for element `{element}`"));
            }
        }
        issues
    }
}

fn strip_zeros(shell: &Shell) -> Vec<f64> {
    shell
        .coefficients
        .iter()
        .flatten()
        .filter(|v| **v != 0.0)
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Angular, BasisRole, BasisSet, ContractionType, Normalization, Shell, ShellBlock};
    use crate::common::{Citation, License};

    pub(crate) fn h_sto3g() -> BasisSet {
        BasisSet {
            name: "sto-3g".into(),
            version: "1.0.0".into(),
            citation: vec![Citation::with_doi("Hehre 1969", "10.1063/1.1672392")],
            license: License::new("Apache-2.0", true, true),
            created: "2025-01-10T08:00:00Z".into(),
            sha256: Some("a".repeat(64)),
            notes: None,
            elements: vec!["H".into()],
            angular: Angular::Spherical,
            normalization: Normalization::L2,
            role: BasisRole::Orbital,
            shells: vec![ShellBlock {
                element: "H".into(),
                shells: vec![Shell {
                    l: 0,
                    exponents: vec![3.42525091, 0.62391373, 0.16885540],
                    coefficients: vec![vec![0.15432897, 0.53532814, 0.44463454]],
                    contraction_type: ContractionType::Segmented,
                }],
            }],
        }
    }

    #[test]
    fn hydrogen_sto3g_has_one_basis_function() {
        let b = h_sto3g();
        assert_eq!(b.n_basis_functions(), 1);
        assert!(b.check().is_empty(), "{:?}", b.check());
    }

    #[test]
    fn check_catches_structural_problems() {
        let mut b = h_sto3g();
        b.elements.push("C".into()); // 没有 C 的块
        assert!(b.check().iter().any(|m| m.contains("no shells block")));

        let mut b = h_sto3g();
        b.elements = vec!["Xx".into()];
        assert!(b.check().iter().any(|m| m.contains("unknown symbol")));

        let mut b = h_sto3g();
        b.shells[0].shells[0].coefficients = vec![vec![1.0, 2.0]];
        assert!(
            b.check()
                .iter()
                .any(|m| m.contains("values but exponents has"))
        );

        let mut b = h_sto3g();
        b.shells[0].shells[0].exponents = vec![0.1, 3.4];
        assert!(b.check().iter().any(|m| m.contains("large -> small")));

        let mut b = h_sto3g();
        b.citation.clear();
        assert!(b.check().iter().any(|m| m.contains("citation")));

        let mut b = h_sto3g();
        b.sha256 = None;
        assert!(b.check().iter().any(|m| m.contains("sha256 is missing")));
    }

    #[test]
    fn general_contraction_shares_exponents() {
        let shell = Shell {
            l: 0,
            exponents: vec![5.0, 1.0, 0.3],
            coefficients: vec![vec![0.5, 0.4, 0.0], vec![0.0, 0.6, 0.5]],
            contraction_type: ContractionType::General,
        };
        assert_eq!(shell.n_contractions(), 2);
        assert_eq!(shell.n_basis_functions(Angular::Spherical), 2);
        let mut b = h_sto3g();
        b.shells[0].shells[0] = shell;
        assert!(b.check().is_empty(), "{:?}", b.check());
    }

    #[test]
    fn cartesian_and_spherical_differ_for_d_shells() {
        let shell = Shell {
            l: 2,
            exponents: vec![1.0],
            coefficients: vec![vec![1.0]],
            contraction_type: ContractionType::Segmented,
        };
        assert_eq!(shell.n_basis_functions(Angular::Spherical), 5);
        assert_eq!(shell.n_basis_functions(Angular::Cartesian), 6);
    }
}
