//! 结果类型（K2）与矩阵的 JSON 表示（K4 二选一定案：**list-of-list**）。
//!
//! 矩阵不依赖 nalgebra 的 serde 内部布局：本模块显式定义
//! `DMatrix<f64>` ↔ `Vec<Vec<f64>>`、`DVector<f64>` ↔ `Vec<f64>` 的映射，
//! 使得 `result.json` 的字节形态由本项目决定，而不是由依赖的私有格式决定（K4）。

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// 矩阵/向量与 JSON 数组之间的显式转换（K4）。
mod json_array {
    use nalgebra::{DMatrix, DVector};

    pub(super) fn matrix_to_rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
        (0..m.nrows())
            .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
            .collect()
    }

    pub(super) fn matrix_from_rows(rows: Vec<Vec<f64>>) -> Result<DMatrix<f64>, String> {
        let ncols = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != ncols) {
            return Err("ragged matrix: row lengths differ".to_string());
        }
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        Ok(DMatrix::from_row_slice(rows.len(), ncols, &flat))
    }

    pub(super) fn vector_to_list(v: &DVector<f64>) -> Vec<f64> {
        v.iter().copied().collect()
    }

    pub(super) fn vector_from_list(x: Vec<f64>) -> DVector<f64> {
        DVector::from_vec(x)
    }
}

/// `Option<DMatrix<f64>>` 的 list-of-list 序列化（`None` → `null`）。
pub(crate) mod opt_matrix_json {
    use nalgebra::DMatrix;
    use serde::{Deserialize, Deserializer, Serializer};

    use super::json_array;

    pub(crate) fn serialize<S: Serializer>(
        value: &Option<DMatrix<f64>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            None => s.serialize_none(),
            Some(m) => s.serialize_some(&json_array::matrix_to_rows(m)),
        }
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<DMatrix<f64>>, D::Error> {
        let rows = Option::<Vec<Vec<f64>>>::deserialize(d)?;
        rows.map(json_array::matrix_from_rows)
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

/// `Option<DVector<f64>>` 的 list 序列化（`None` → `null`）。
pub(crate) mod opt_vector_json {
    use nalgebra::DVector;
    use serde::{Deserialize, Deserializer, Serializer};

    use super::json_array;

    pub(crate) fn serialize<S: Serializer>(
        value: &Option<DVector<f64>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            None => s.serialize_none(),
            Some(v) => s.serialize_some(&json_array::vector_to_list(v)),
        }
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<DVector<f64>>, D::Error> {
        let list = Option::<Vec<f64>>::deserialize(d)?;
        Ok(list.map(json_array::vector_from_list))
    }
}

/// 是否 64 位小写十六进制（sha256 的文本形态）。
#[must_use]
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// 数据快照：`相对路径 -> sha256`（K2 / K3 `data_snapshot.json`）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
#[non_exhaustive]
pub struct DataSnapshot {
    entries: BTreeMap<String, String>,
}

impl DataSnapshot {
    /// 空快照。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 记录一条 `path -> sha256`。
    pub fn insert(&mut self, path: impl Into<String>, sha256: impl Into<String>) {
        self.entries.insert(path.into(), sha256.into());
    }

    /// 读取某路径的 sha256。
    #[must_use]
    pub fn get(&self, path: &str) -> Option<&str> {
        self.entries.get(path).map(String::as_str)
    }

    /// 条目数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 迭代 `(path, sha256)`，按路径字典序（确定性）。
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// 快照的规范化摘要输入：`path=sha256` 按字典序逐行拼接。
    #[must_use]
    pub fn digest_input(&self) -> String {
        self.iter()
            .map(|(p, h)| format!("{p}={h}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 校验每个 sha256 都是 64 位小写十六进制。
    ///
    /// # Errors
    ///
    /// 任一条目格式非法时返回 [`EngineError::Data`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        for (path, sha) in self.iter() {
            if !is_sha256_hex(sha) {
                return Err(EngineError::Data(format!(
                    "data_snapshot[{path}] is not a 64-char lowercase sha256 hex: {sha:?}"
                )));
            }
        }
        Ok(())
    }
}

/// 复现所需的运行环境（K3 `env.json`；§2.1 要求含 toolchain、Cargo.lock sha256、crate 版本清单）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct EnvInfo {
    /// 引擎版本（`CARGO_PKG_VERSION`）。
    pub engine_version: String,
    /// `rustc --version` 原文。
    pub rustc_version: String,
    /// `Cargo.lock` 的 sha256。
    pub cargo_lock_sha256: String,
    /// 本次计算用到的数据文件快照。
    pub data_snapshot: DataSnapshot,
    /// 实际线程数（`rayon` 实际生效值，不是请求值）。
    pub threads: u32,
    /// OS 描述（如 `windows 11 x86_64`）。
    pub os: String,
    /// ISO 8601 UTC 时间戳。
    pub created_at_utc: String,
    /// `rust-toolchain.toml` 的 channel。
    pub toolchain_channel: Option<String>,
    /// 依赖 crate 版本清单（crate 名 → 版本）。
    pub crate_versions: BTreeMap<String, String>,
}

impl EnvInfo {
    /// 用必需字段构造；`toolchain_channel` / `crate_versions` 默认空。
    #[must_use]
    pub fn new(
        engine_version: impl Into<String>,
        rustc_version: impl Into<String>,
        cargo_lock_sha256: impl Into<String>,
        data_snapshot: DataSnapshot,
        threads: u32,
        os: impl Into<String>,
        created_at_utc: impl Into<String>,
    ) -> Self {
        Self {
            engine_version: engine_version.into(),
            rustc_version: rustc_version.into(),
            cargo_lock_sha256: cargo_lock_sha256.into(),
            data_snapshot,
            threads,
            os: os.into(),
            created_at_utc: created_at_utc.into(),
            toolchain_channel: None,
            crate_versions: BTreeMap::new(),
        }
    }

    /// 补上 toolchain channel。
    #[must_use]
    pub fn with_toolchain_channel(mut self, channel: impl Into<String>) -> Self {
        self.toolchain_channel = Some(channel.into());
        self
    }

    /// 补上 crate 版本清单。
    #[must_use]
    pub fn with_crate_versions(mut self, versions: BTreeMap<String, String>) -> Self {
        self.crate_versions = versions;
        self
    }

    /// G5 复现门禁：`env.json` 必须完整，否则结果视为不可复现。
    ///
    /// # Errors
    ///
    /// 任一必需字段缺失/格式非法时返回 [`EngineError::Data`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.engine_version.trim().is_empty() {
            return Err(EngineError::Data("env.engine_version is empty".into()));
        }
        if self.rustc_version.trim().is_empty() {
            return Err(EngineError::Data("env.rustc_version is empty".into()));
        }
        if !is_sha256_hex(&self.cargo_lock_sha256) {
            return Err(EngineError::Data(
                "env.cargo_lock_sha256 is not a sha256 hex".into(),
            ));
        }
        if self.os.trim().is_empty() {
            return Err(EngineError::Data("env.os is empty".into()));
        }
        if self.threads == 0 {
            return Err(EngineError::Data("env.threads must be >= 1".into()));
        }
        if self.created_at_utc.trim().is_empty() {
            return Err(EngineError::Data("env.created_at_utc is empty".into()));
        }
        self.data_snapshot.validate()
    }
}

/// 一次作业的结果（K2 冻结；同时是 `result.json` 的形态，见 K4）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct JobResult {
    /// 总能量，单位 Hartree。
    pub energy_hartree: f64,
    /// 解析梯度，单位 Hartree/Bohr，形状 `3N×1`；`Sp` 作业通常为 `None`。
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_matrix_json"
    )]
    pub gradient_hartree_per_bohr: Option<DMatrix<f64>>,
    /// Hessian，单位 Hartree/Bohr²，形状 `3N×3N`。
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_matrix_json"
    )]
    pub hessian_hartree_per_bohr2: Option<DMatrix<f64>>,
    /// 偶极矩，单位 Debye，`[x, y, z]`。
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_vector_json"
    )]
    pub dipole_debye: Option<DVector<f64>>,
    /// 轨道能，单位 Hartree。
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_vector_json"
    )]
    pub mo_energies_hartree: Option<DVector<f64>>,
    /// 轨道系数（列 = MO）。
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_matrix_json"
    )]
    pub mo_coeffs: Option<DMatrix<f64>>,
    /// SCF 是否收敛。
    pub scf_converged: bool,
    /// 本次计算用到的数据快照。
    pub data_snapshot: DataSnapshot,
    /// 引擎日志（不得写 stdout，全部进这里或 `Reporter`）。
    pub log: String,
    /// 复现环境。
    pub env: EnvInfo,
}

impl JobResult {
    /// 用必需字段构造（`non_exhaustive` 结构体不提供下游 struct literal）。
    #[must_use]
    pub fn new(
        energy_hartree: f64,
        scf_converged: bool,
        data_snapshot: DataSnapshot,
        env: EnvInfo,
    ) -> Self {
        Self {
            energy_hartree,
            gradient_hartree_per_bohr: None,
            hessian_hartree_per_bohr2: None,
            dipole_debye: None,
            mo_energies_hartree: None,
            mo_coeffs: None,
            scf_converged,
            data_snapshot,
            log: String::new(),
            env,
        }
    }

    /// 挂上梯度。
    #[must_use]
    pub fn with_gradient(mut self, gradient_hartree_per_bohr: DMatrix<f64>) -> Self {
        self.gradient_hartree_per_bohr = Some(gradient_hartree_per_bohr);
        self
    }

    /// 挂上 Hessian。
    #[must_use]
    pub fn with_hessian(mut self, hessian_hartree_per_bohr2: DMatrix<f64>) -> Self {
        self.hessian_hartree_per_bohr2 = Some(hessian_hartree_per_bohr2);
        self
    }

    /// 挂上偶极矩。
    #[must_use]
    pub fn with_dipole(mut self, dipole_debye: DVector<f64>) -> Self {
        self.dipole_debye = Some(dipole_debye);
        self
    }

    /// 挂上轨道（能级 + 系数）。
    #[must_use]
    pub fn with_orbitals(mut self, energies_hartree: DVector<f64>, coeffs: DMatrix<f64>) -> Self {
        self.mo_energies_hartree = Some(energies_hartree);
        self.mo_coeffs = Some(coeffs);
        self
    }

    /// 设置日志。
    #[must_use]
    pub fn with_log(mut self, log: impl Into<String>) -> Self {
        self.log = log.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use super::{DataSnapshot, EnvInfo, JobResult, is_sha256_hex};

    fn snapshot() -> DataSnapshot {
        let mut s = DataSnapshot::new();
        s.insert("basis/sto-3g.yaml", "a".repeat(64));
        s.insert("functional/svwn5.yaml", "b".repeat(64));
        s
    }

    fn env() -> EnvInfo {
        EnvInfo::new(
            "0.1.0",
            "rustc 1.97.1",
            "c".repeat(64),
            snapshot(),
            4,
            "windows 11 x86_64",
            "2025-01-10T08:00:00Z",
        )
    }

    #[test]
    fn sha256_hex_checker() {
        assert!(is_sha256_hex(&"a".repeat(64)));
        assert!(!is_sha256_hex(&"A".repeat(64)));
        assert!(!is_sha256_hex("abc"));
    }

    #[test]
    fn env_info_g5_validation() {
        assert!(env().validate().is_ok());

        let mut bad = env();
        bad.rustc_version = String::new();
        assert!(bad.validate().is_err());

        let mut bad = env();
        bad.cargo_lock_sha256 = "nothex".into();
        assert!(bad.validate().is_err());

        let mut bad = env();
        bad.threads = 0;
        assert!(bad.validate().is_err());

        let mut bad = env();
        bad.data_snapshot.insert("x", "zz");
        assert!(bad.validate().is_err());
    }

    #[test]
    fn matrices_serialize_as_list_of_list() {
        let grad = DMatrix::from_row_slice(3, 1, &[0.1, 0.2, 0.3]);
        let result = JobResult::new(1.5, true, snapshot(), env()).with_gradient(grad);
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(
            json.contains("\"gradient_hartree_per_bohr\":[[0.1],[0.2],[0.3]]"),
            "K4 requires list-of-list, got {json}"
        );
        assert!(!json.contains("hessian_hartree_per_bohr2"));
    }

    #[test]
    fn result_roundtrips_through_json() {
        let hess = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let dipole = DVector::from_vec(vec![0.0, 0.0, 0.7]);
        let result = JobResult::new(1.5, true, snapshot(), env())
            .with_hessian(hess)
            .with_dipole(dipole)
            .with_orbitals(
                DVector::from_vec(vec![-0.5, 0.2]),
                DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            );
        let json = serde_json::to_string(&result).expect("serialize");
        let back: JobResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(result, back);
    }

    #[test]
    fn ragged_matrix_is_rejected() {
        let json = r#"{"energy_hartree":1.0,"gradient_hartree_per_bohr":[[1.0,2.0],[3.0]],
            "scf_converged":true,"data_snapshot":{},"log":"",
            "env":{"engine_version":"0.1.0","rustc_version":"rustc 1.97.1",
            "cargo_lock_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "data_snapshot":{},"threads":1,"os":"w","created_at_utc":"t",
            "toolchain_channel":null,"crate_versions":{}}}"#;
        assert!(serde_json::from_str::<JobResult>(json).is_err());
    }

    #[test]
    fn data_snapshot_digest_is_order_independent() {
        let mut a = DataSnapshot::new();
        a.insert("z.yaml", "1");
        a.insert("a.yaml", "2");
        let mut b = DataSnapshot::new();
        b.insert("a.yaml", "2");
        b.insert("z.yaml", "1");
        assert_eq!(a.digest_input(), b.digest_input());
        assert_eq!(a.len(), 2);
        assert_eq!(a.get("a.yaml"), Some("2"));
        assert!(a.validate().is_err());
    }
}
