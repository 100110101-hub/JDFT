//! 内容摘要（`sha256`）规则 —— K1 的“工具生成”落地。
//!
//! # 规范化规则（本项目自定义，必须在 K1 内一致）
//!
//! 文件里的 `sha256` 字段是**自我引用**，所以摘要不能直接对文件字节取 hash。
//! 本项目的规则是：
//!
//! 1. 解析 YAML 为值树；
//! 2. 删除顶层 `sha256` 键；
//! 3. 转成 JSON 值（映射按键字典序，`serde_json` 默认 `BTreeMap`）；
//! 4. 对紧凑 JSON 字节取 SHA-256，小写十六进制。
//!
//! 好处：**与 YAML 排版、注释、键顺序无关**。改一个数字必然改摘要，
//! 重排键或加注释不会。
//!
//! 代价：YAML 注释不参与摘要；`stamp` 写回时注释也会丢失（数据文件是工具产物）。

use sha2::{Digest, Sha256};

/// `CanonError`：规范化失败的原因。
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CanonError {
    /// YAML 语法错误。
    #[error("yaml parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// 顶层不是映射（数据文件必须是 `key: value` 结构）。
    #[error("top-level YAML value must be a mapping")]
    NotMapping,
    /// 值无法表示为 JSON（例如 YAML 的非字符串键或特殊浮点）。
    #[error("value cannot be represented as JSON: {0}")]
    NotJson(String),
}

/// 字节串的 sha256（小写十六进制）。
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(digest)
}

/// 是否 64 位小写十六进制。
#[must_use]
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// 解析 YAML 字节为值树。
///
/// # Errors
///
/// YAML 语法错误。
pub fn parse(bytes: &[u8]) -> Result<serde_yaml::Value, CanonError> {
    Ok(serde_yaml::from_slice::<serde_yaml::Value>(bytes)?)
}

/// 去掉 `sha256` 并转成 JSON 值。
///
/// # Errors
///
/// 顶层不是映射，或值无法转成 JSON。
pub fn canonical_value(value: &serde_yaml::Value) -> Result<serde_json::Value, CanonError> {
    let mut mapping = match value {
        serde_yaml::Value::Mapping(m) => m.clone(),
        _ => return Err(CanonError::NotMapping),
    };
    mapping.remove(serde_yaml::Value::String("sha256".to_string()));
    let json = serde_json::to_value(serde_yaml::Value::Mapping(mapping))
        .map_err(|e| CanonError::NotJson(e.to_string()))?;
    Ok(json)
}

/// 规范化内容摘要（见模块文档）。
///
/// # Errors
///
/// 同 [`canonical_value`]。
pub fn content_sha256(value: &serde_yaml::Value) -> Result<String, CanonError> {
    let json = canonical_value(value)?;
    let bytes = serde_json::to_vec(&json).map_err(|e| CanonError::NotJson(e.to_string()))?;
    Ok(sha256_hex(&bytes))
}

/// 校验声明的 sha256 是否与内容一致。
///
/// # Errors
///
/// 规范化失败。
pub fn check_sha256(value: &serde_yaml::Value, declared: &str) -> Result<bool, CanonError> {
    Ok(content_sha256(value)? == declared)
}

/// 给值树打上 `sha256`（放在映射第一个键位）并返回该摘要。
///
/// # Errors
///
/// 顶层不是映射。
pub fn stamp(value: &mut serde_yaml::Value) -> Result<String, CanonError> {
    let computed = content_sha256(value)?;
    let mapping = match value {
        serde_yaml::Value::Mapping(m) => m,
        _ => return Err(CanonError::NotMapping),
    };
    mapping.remove(serde_yaml::Value::String("sha256".to_string()));
    let mut rebuilt = serde_yaml::Mapping::new();
    rebuilt.insert(
        serde_yaml::Value::String("sha256".to_string()),
        serde_yaml::Value::String(computed.clone()),
    );
    for (k, v) in mapping.iter() {
        rebuilt.insert(k.clone(), v.clone());
    }
    *mapping = rebuilt;
    Ok(computed)
}

/// 给 YAML 字节打上 `sha256`，返回新的 YAML 字节。
///
/// # Errors
///
/// 解析失败、顶层不是映射、或序列化失败。
pub fn stamp_bytes(bytes: &[u8]) -> Result<(String, Vec<u8>), CanonError> {
    let mut value = parse(bytes)?;
    let digest = stamp(&mut value)?;
    let out = serde_yaml::to_string(&value)?;
    Ok((digest, out.into_bytes()))
}

#[cfg(test)]
mod tests {
    use super::{
        CanonError, canonical_value, content_sha256, is_sha256_hex, parse, sha256_hex, stamp,
        stamp_bytes,
    };

    const BODY: &str = "name: x\nversion: 1.0.0\ncreated: 2025-01-10T08:00:00Z\n";

    #[test]
    fn sha256_of_known_string() {
        // sha256("abc") 的标准值。
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(is_sha256_hex(&sha256_hex(b"")));
        assert!(!is_sha256_hex("ABC"));
    }

    #[test]
    fn digest_ignores_sha256_field_and_key_order() {
        let a = parse(BODY.as_bytes()).unwrap();
        let b = parse(format!("{BODY}sha256: {}\n", "0".repeat(64)).as_bytes()).unwrap();
        let c =
            parse("created: 2025-01-10T08:00:00Z\nversion: 1.0.0\nname: x\n".as_bytes()).unwrap();
        let da = content_sha256(&a).unwrap();
        assert_eq!(da, content_sha256(&b).unwrap());
        assert_eq!(da, content_sha256(&c).unwrap());

        // 改一个数字必须换摘要。
        let mut changed = a.clone();
        if let serde_yaml::Value::Mapping(m) = &mut changed {
            m.insert(
                serde_yaml::Value::String("version".into()),
                serde_yaml::Value::String("1.0.1".into()),
            );
        }
        assert_ne!(da, content_sha256(&changed).unwrap());
    }

    #[test]
    fn stamp_is_idempotent() {
        let mut value = parse(BODY.as_bytes()).unwrap();
        let d1 = stamp(&mut value).unwrap();
        let d2 = stamp(&mut value).unwrap();
        assert_eq!(d1, d2);
        assert_eq!(value["sha256"], serde_yaml::Value::String(d1.clone()));
        assert_eq!(content_sha256(&value).unwrap(), d1, "打戳后重算必须一致");

        let yaml = serde_yaml::to_string(&value).unwrap();
        let (d3, restamped) = stamp_bytes(yaml.as_bytes()).unwrap();
        assert_eq!(d1, d3);
        let back = parse(&restamped).unwrap();
        assert_eq!(content_sha256(&back).unwrap(), d3);
    }

    #[test]
    fn non_mapping_is_rejected() {
        let items = parse(b"- 1\n- 2\n").unwrap();
        assert!(matches!(
            canonical_value(&items),
            Err(CanonError::NotMapping)
        ));
        let mut items = items;
        assert!(stamp(&mut items).is_err());
    }
}
