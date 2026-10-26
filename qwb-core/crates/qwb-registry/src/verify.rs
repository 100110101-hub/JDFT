//! 数据文件校验：C-01 验收要求的 `qwb data verify` 的后端。
//!
//! 校验分四层，任一层失败都报**具体字段**，不报“文件坏了”：
//!
//! 1. `yaml.parse` —— YAML 语法；
//! 2. `schema.deserialize` —— 结构（`serde` + `deny_unknown_fields`，字段拼错即报）；
//! 3. `sha256.*` —— 内容摘要与文件内声明是否一致（G2）；
//! 4. `semantic` / `g2.*` —— 语义与门禁（citation、license、量纲、自洽性）。

use std::path::Path;

use serde::de::DeserializeOwned;

use crate::basis::BasisSet;
use crate::common::DataCommon;
use crate::ecp::Ecp;
use crate::functional::Functional;
use crate::grid::GridLevelDef;
use crate::molecule::MoleculeFile;
use crate::numerics::NumericsDef;
use crate::registry::RegistryManifest;
use crate::sha256;
use crate::solvent::Solvent;

/// 数据类别（对应 `qwb-data/` 子目录与 K1 schema 文件名）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum DataKind {
    /// `basis/`，`basis.schema.*`。
    Basis,
    /// `ecp/`，`ecp.schema.*`。
    Ecp,
    /// `functional/`，`functional.schema.*`。
    Functional,
    /// `grid/`，`grid.schema.*`。
    Grid,
    /// `solvent/`，`solvent.schema.*`。
    Solvent,
    /// `solvent/`，`solvent.schema.*`（已在上方）。
    /// `molecule/`，`molecule.schema.*`。
    Molecule,
    /// `numerics/`，`numerics.schema.*`（阶段二 K10）。
    Numerics,
    /// `registry.yaml`，`registry.schema.*`。
    Registry,
}

impl DataKind {
    /// 全部类别（schema 生成顺序即此顺序）。
    pub const ALL: [DataKind; 8] = [
        DataKind::Basis,
        DataKind::Ecp,
        DataKind::Functional,
        DataKind::Grid,
        DataKind::Solvent,
        DataKind::Numerics,
        DataKind::Molecule,
        DataKind::Registry,
    ];

    /// 数据目录名 / schema 文件名主干。
    #[must_use]
    pub fn dir_name(self) -> &'static str {
        match self {
            DataKind::Basis => "basis",
            DataKind::Ecp => "ecp",
            DataKind::Functional => "functional",
            DataKind::Grid => "grid",
            DataKind::Solvent => "solvent",
            DataKind::Numerics => "numerics",
            DataKind::Molecule => "molecule",
            DataKind::Registry => "registry",
        }
    }

    /// 该类别是否属于 G2 许可/citation 门禁的“数据集”。
    #[must_use]
    pub fn requires_license(self) -> bool {
        !matches!(self, DataKind::Molecule | DataKind::Registry)
    }

    /// 由目录名反查类别。
    #[must_use]
    pub fn from_dir_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.dir_name() == name)
    }

    /// 由 schema 文件名主干反查类别。
    #[must_use]
    pub fn from_schema_stem(stem: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.dir_name() == stem)
    }
}

/// 问题严重程度。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Severity {
    /// 拒绝该文件（`qwb data verify` 退出码 3）。
    Error,
    /// 不拒绝，但要在报告里提示。
    Warning,
}

impl Severity {
    /// 文本标签。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

/// 一条校验问题。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Issue {
    /// 严重程度。
    pub severity: Severity,
    /// 机器可读代码（`yaml.parse` / `schema.deserialize` / `sha256.mismatch` / `g2.citation` / `semantic` …）。
    pub code: &'static str,
    /// 人类可读说明。
    pub message: String,
}

/// 一个文件的校验报告。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Report {
    /// 文件类别。
    pub kind: DataKind,
    /// 文件路径（报告用的标签）。
    pub path: String,
    /// 文件内声明的 sha256。
    pub declared_sha256: Option<String>,
    /// 本次计算出的规范化 sha256。
    pub computed_sha256: Option<String>,
    /// 问题列表（按发现顺序）。
    pub issues: Vec<Issue>,
}

impl Report {
    /// 构造空报告。
    #[must_use]
    pub fn new(kind: DataKind, path: impl Into<String>) -> Self {
        Self {
            kind,
            path: path.into(),
            declared_sha256: None,
            computed_sha256: None,
            issues: Vec::new(),
        }
    }

    /// 记录一个错误。
    pub fn error(&mut self, code: &'static str, message: impl Into<String>) {
        self.issues.push(Issue {
            severity: Severity::Error,
            code,
            message: message.into(),
        });
    }

    /// 记录一个警告。
    pub fn warn(&mut self, code: &'static str, message: impl Into<String>) {
        self.issues.push(Issue {
            severity: Severity::Warning,
            code,
            message: message.into(),
        });
    }

    /// 是否通过（无 `Error`）。
    #[must_use]
    pub fn is_ok(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// 错误条数。
    #[must_use]
    pub fn n_errors(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .count()
    }

    /// 警告条数。
    #[must_use]
    pub fn n_warnings(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .count()
    }

    /// 只有错误的迭代器。
    pub fn errors(&self) -> impl Iterator<Item = &Issue> {
        self.issues.iter().filter(|i| i.severity == Severity::Error)
    }
}

/// 校验失败（不是数据不合规，而是根本没法校验）。
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// 读文件失败。
    #[error("cannot read {path}: {source}")]
    Io {
        /// 路径。
        path: String,
        /// 底层错误。
        source: std::io::Error,
    },
    /// 无法从路径推断类别。
    #[error(
        "cannot infer data kind for {0}: put the file under basis/ ecp/ functional/ grid/ solvent/ molecule/ or name it registry.yaml"
    )]
    UnknownKind(String),
}

/// 校验一段字节（不碰文件系统）。
///
/// 这是 `data verify` 的核心；GUI/CI 可以就地校验内存里的数据。
#[must_use]
pub fn verify_bytes(kind: DataKind, path: &str, bytes: &[u8]) -> Report {
    let mut report = Report::new(kind, path);

    let value = match sha256::parse(bytes) {
        Ok(value) => value,
        Err(err) => {
            report.error("yaml.parse", err.to_string());
            return report;
        }
    };
    match sha256::content_sha256(&value) {
        Ok(digest) => report.computed_sha256 = Some(digest),
        Err(err) => report.error("sha256.compute", err.to_string()),
    }

    match kind {
        DataKind::Basis => verify_data::<BasisSet>(&mut report, bytes, BasisSet::check),
        DataKind::Ecp => verify_data::<Ecp>(&mut report, bytes, Ecp::check),
        DataKind::Functional => verify_data::<Functional>(&mut report, bytes, Functional::check),
        DataKind::Grid => verify_data::<GridLevelDef>(&mut report, bytes, GridLevelDef::check),
        DataKind::Solvent => verify_data::<Solvent>(&mut report, bytes, Solvent::check),
        DataKind::Numerics => verify_data::<NumericsDef>(&mut report, bytes, NumericsDef::check),
        DataKind::Molecule => verify_molecule(&mut report, bytes),
        DataKind::Registry => verify_manifest(&mut report, bytes),
    }

    report
}

/// 读文件并校验（类别由路径推断）。
///
/// # Errors
///
/// 读失败或推断不出类别。
pub fn verify_file(path: &Path) -> Result<Report, VerifyError> {
    let bytes = std::fs::read(path).map_err(|source| VerifyError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let kind =
        infer_kind(path).ok_or_else(|| VerifyError::UnknownKind(path.display().to_string()))?;
    Ok(verify_bytes(kind, &path.display().to_string(), &bytes))
}

/// 由路径推断数据类别：先看父目录名，再看文件名 `registry.yaml`。
#[must_use]
pub fn infer_kind(path: &Path) -> Option<DataKind> {
    if let Some(parent) = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|s| s.to_str())
        && let Some(kind) = DataKind::from_dir_name(parent)
    {
        return Some(kind);
    }
    let stem = path.file_stem()?.to_str()?;
    if stem == "registry" {
        return Some(DataKind::Registry);
    }
    DataKind::from_dir_name(stem)
}

/// 校验整个注册表：清单本身 + 清单里每个条目文件 + 两处 sha256 的交叉一致性。
///
/// # Errors
///
/// manifest 读不了。
pub fn verify_registry(manifest_path: &Path) -> Result<Vec<Report>, VerifyError> {
    let bytes = std::fs::read(manifest_path).map_err(|source| VerifyError::Io {
        path: manifest_path.display().to_string(),
        source,
    })?;
    let mut manifest_report = verify_bytes(
        DataKind::Registry,
        &manifest_path.display().to_string(),
        &bytes,
    );

    let manifest = match serde_yaml::from_slice::<RegistryManifest>(&bytes) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(vec![manifest_report]),
    };

    let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let third_party = crate::thirdparty::third_party_paths(base).unwrap_or_default();
    let mut reports = Vec::new();
    for (domain, entry) in manifest.all_entries() {
        let path = base.join(&entry.path);
        match verify_file(&path) {
            Ok(report) => {
                if report.kind.dir_name() != domain {
                    manifest_report.error(
                        "registry.kind",
                        format!(
                            "{domain}/{} points at {} which is a {} file",
                            entry.name,
                            entry.path,
                            report.kind.dir_name()
                        ),
                    );
                }
                if let Some(declared) = &report.declared_sha256
                    && declared != &entry.sha256
                {
                    manifest_report.error(
                        "registry.sha256",
                        format!(
                            "{domain}/{}: manifest says {} but {} declares {}",
                            entry.name, entry.sha256, entry.path, declared
                        ),
                    );
                }
                reports.push(report);
            }
            Err(err) => {
                // 第三方数据缺失是**正常状态**（仓库不提交它们，见 `crate::thirdparty`），
                // 所以那类缺失要把"怎么装"写进错误里，否则读的人会以为数据坏了；
                // 自研数据缺失（不在 sources.json 里）是数据根被裁剪，不该被这句提示淡化。
                let is_third_party =
                    third_party.contains(&crate::thirdparty::normalize(&entry.path));
                let hint = match &err {
                    VerifyError::Io { source, .. }
                        if source.kind() == std::io::ErrorKind::NotFound && is_third_party =>
                    {
                        "（第三方数据，本地未取回：跑 `qwb data install`）"
                    }
                    VerifyError::Io { source, .. }
                        if source.kind() == std::io::ErrorKind::NotFound =>
                    {
                        "（该文件不是第三方数据：数据根被裁剪过，请从仓库恢复）"
                    }
                    _ => "",
                };
                // 文件读不出：能推断类别就记成**该文件的**失败报告（这样 `N file(s) checked`
                // 的计数与真实条目数一致），推断不出才退回清单层。
                match infer_kind(&path) {
                    Some(kind) => {
                        let mut report = Report::new(kind, path.display().to_string());
                        report.error("registry.path", format!("{err}{hint}"));
                        reports.push(report);
                    }
                    None => manifest_report.error(
                        "registry.path",
                        format!("{domain}/{}: {err}{hint}", entry.name),
                    ),
                }
            }
        }
    }
    let mut all = vec![manifest_report];
    all.extend(reports);
    Ok(all)
}

fn verify_data<T>(report: &mut Report, bytes: &[u8], check: fn(&T) -> Vec<String>)
where
    T: DeserializeOwned + DataCommon,
{
    match serde_yaml::from_slice::<T>(bytes) {
        Ok(value) => {
            report.declared_sha256 = value.declared_sha256().map(ToString::to_string);
            compare_sha256(report);
            for message in check(&value) {
                report.error(classify(&message), message);
            }
        }
        Err(err) => report.error("schema.deserialize", format!("schema violation: {err}")),
    }
}

fn verify_molecule(report: &mut Report, bytes: &[u8]) {
    match serde_yaml::from_slice::<MoleculeFile>(bytes) {
        Ok(value) => {
            for message in value.check() {
                report.error(classify(&message), message);
            }
        }
        Err(err) => report.error("schema.deserialize", format!("schema violation: {err}")),
    }
}

fn verify_manifest(report: &mut Report, bytes: &[u8]) {
    match serde_yaml::from_slice::<RegistryManifest>(bytes) {
        Ok(value) => {
            for message in value.check() {
                report.error(classify(&message), message);
            }
        }
        Err(err) => report.error("schema.deserialize", format!("schema violation: {err}")),
    }
}

fn compare_sha256(report: &mut Report) {
    match (&report.declared_sha256, &report.computed_sha256) {
        (Some(declared), Some(computed)) if declared != computed => {
            report.error(
                "sha256.mismatch",
                format!("declared sha256 {declared} but content hashes to {computed}"),
            );
        }
        _ => {}
    }
}

/// 把语义检查的文本映射到机器可读代码（保持报告可过滤）。
fn classify(message: &str) -> &'static str {
    if message.contains("citation") {
        "g2.citation"
    } else if message.contains("license") {
        "g2.license"
    } else if message.contains("sha256") {
        "g2.sha256"
    } else {
        "semantic"
    }
}

#[cfg(test)]
mod tests {
    use super::{DataKind, Severity, infer_kind, verify_bytes, verify_registry};
    use crate::sha256::stamp_bytes;

    const GOOD_BASIS: &str = "name: sto-3g
version: 1.0.0
citation:
  - title: Hehre 1969
    doi: 10.1063/1.1672392
license:
  spdx: Apache-2.0
  commercial_ok: true
  redistribute_ok: true
created: 2025-01-10T08:00:00Z
elements:
  - H
angular: spherical
normalization: l2
role: orbital
shells:
  - element: H
    shells:
      - l: 0
        exponents: [3.42525091, 0.62391373, 0.16885540]
        coefficients:
          - [0.15432897, 0.53532814, 0.44463454]
        contraction_type: segmented
";

    fn stamped(source: &str) -> Vec<u8> {
        let (_, bytes) = stamp_bytes(source.as_bytes()).expect("stamp");
        bytes
    }

    #[test]
    fn good_basis_file_passes() {
        let report = verify_bytes(DataKind::Basis, "basis/sto-3g.yaml", &stamped(GOOD_BASIS));
        assert!(report.is_ok(), "{:#?}", report.issues);
        assert_eq!(report.n_errors(), 0);
        assert_eq!(report.declared_sha256, report.computed_sha256);
    }

    #[test]
    fn missing_citation_is_rejected() {
        // C-01 验收：`qwb data verify` 必须拒载缺 citation 的文件。
        // 两条路径都在：字段整个缺失 → schema 层就拒；字段存在但为空 → G2 语义层拒。
        let bad = GOOD_BASIS.replace(
            "citation:\n  - title: Hehre 1969\n    doi: 10.1063/1.1672392\n",
            "",
        );
        let report = verify_bytes(DataKind::Basis, "basis/bad.yaml", &stamped(&bad));
        assert!(!report.is_ok());
        assert!(
            report.errors().any(|i| i.code == "schema.deserialize"),
            "citation 是 required 字段，缺失必须在 schema 层被拒"
        );

        let empty = GOOD_BASIS.replace(
            "citation:\n  - title: Hehre 1969\n    doi: 10.1063/1.1672392\n",
            "citation: []\n",
        );
        let report = verify_bytes(DataKind::Basis, "basis/bad.yaml", &stamped(&empty));
        assert!(!report.is_ok());
        assert!(
            report.errors().any(|i| i.code == "g2.citation"),
            "空 citation 必须在 G2 语义层被拒: {:#?}",
            report.issues
        );
    }

    #[test]
    fn citation_without_locator_is_rejected() {
        let bad = GOOD_BASIS.replace("    doi: 10.1063/1.1672392\n", "");
        let report = verify_bytes(DataKind::Basis, "basis/bad.yaml", &stamped(&bad));
        assert!(report.errors().any(|i| i.code == "g2.citation"));
    }

    #[test]
    fn typo_in_field_name_is_rejected() {
        let bad = GOOD_BASIS.replace("angular: spherical", "angelar: spherical");
        let report = verify_bytes(DataKind::Basis, "basis/bad.yaml", &stamped(&bad));
        assert!(report.errors().any(|i| i.code == "schema.deserialize"));
    }

    #[test]
    fn missing_sha256_is_rejected_and_mismatch_is_rejected() {
        let report = verify_bytes(DataKind::Basis, "basis/x.yaml", GOOD_BASIS.as_bytes());
        assert!(report.errors().any(|i| i.code == "g2.sha256"));

        let mut bytes = stamped(GOOD_BASIS);
        let text = String::from_utf8(bytes.clone()).unwrap();
        let tampered = text.replace("version: 1.0.0", "version: 2.0.0");
        bytes = tampered.into_bytes();
        let report = verify_bytes(DataKind::Basis, "basis/x.yaml", &bytes);
        assert!(report.errors().any(|i| i.code == "sha256.mismatch"));
    }

    #[test]
    fn broken_yaml_reports_parse_error() {
        let report = verify_bytes(DataKind::Basis, "basis/x.yaml", b"name: [unclosed");
        assert!(report.errors().any(|i| i.code == "yaml.parse"));
        assert_eq!(report.n_errors(), 1);
    }

    #[test]
    fn warnings_do_not_fail_the_file() {
        let mut report = verify_bytes(DataKind::Basis, "basis/x.yaml", &stamped(GOOD_BASIS));
        assert!(report.is_ok());
        report.warn("test.warning", "example warning");
        assert!(report.is_ok());
        assert_eq!(report.n_warnings(), 1);
        assert_eq!(report.issues[0].severity, Severity::Warning);
    }

    #[test]
    fn kind_inference_uses_parent_directory() {
        let p = std::path::Path::new("/x/qwb-data/basis/sto-3g.yaml");
        assert_eq!(infer_kind(p), Some(DataKind::Basis));
        let p = std::path::Path::new("/x/qwb-data/registry.yaml");
        assert_eq!(infer_kind(p), Some(DataKind::Registry));
        let p = std::path::Path::new("/x/qwb-data/examples/solvent/water.yaml");
        assert_eq!(infer_kind(p), Some(DataKind::Solvent));
        let p = std::path::Path::new("/x/elsewhere/data.yaml");
        assert_eq!(infer_kind(p), None);
    }

    #[test]
    fn registry_verification_cross_checks_sha256() {
        let dir = std::env::temp_dir().join(format!("qwb-registry-{}", std::process::id()));
        let basis_dir = dir.join("basis");
        std::fs::create_dir_all(&basis_dir).unwrap();
        let basis_bytes = stamped(GOOD_BASIS);
        std::fs::write(basis_dir.join("sto-3g.yaml"), &basis_bytes).unwrap();
        let declared = {
            let report = verify_bytes(DataKind::Basis, "basis/sto-3g.yaml", &basis_bytes);
            report.declared_sha256.unwrap()
        };

        let manifest = format!(
            "schema_version: {v}\nname: qwb-data\nversion: 0.1.0\ncreated: 2025-01-10T08:00:00Z\nfeatures: [jobtype:sp]\nbases:\n  - name: sto-3g\n    path: basis/sto-3g.yaml\n    sha256: {declared}\n",
            v = crate::K1_VERSION
        );
        std::fs::write(dir.join("registry.yaml"), &manifest).unwrap();
        let reports = verify_registry(&dir.join("registry.yaml")).unwrap();
        assert!(reports.iter().all(super::Report::is_ok), "{:#?}", reports);

        // 清单里的 sha256 被改 → 交叉校验必须报错。
        let tampered = manifest.replace(&declared, &"f".repeat(64));
        std::fs::write(dir.join("registry.yaml"), &tampered).unwrap();
        let reports = verify_registry(&dir.join("registry.yaml")).unwrap();
        assert!(reports[0].errors().any(|i| i.code == "registry.sha256"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
