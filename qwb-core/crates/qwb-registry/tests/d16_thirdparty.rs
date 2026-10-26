//! D-16 验收：第三方数据按需取回时，**加载器必须能区分"还没下载"与"数据坏了"**。
//!
//! 判据（对应 `qwb-data/sources.json` 与 project.md §14 #20 的分发约束）：
//!
//! ① `sources.json` 列出的文件缺失 → `Registry::open` **照常成功**，缺的那条记进
//!    `missing_data()`，按名字取它时错误里带取数命令（退出码 3 的那条路）；
//! ② 不在 `sources.json` 里的文件缺失 → **硬错误**（数据根被裁剪过，不能当"还没下载"）；
//! ③ 没有 `sources.json`（老数据根）→ 什么都不放过，行为与以前一致；
//! ④ `sources.json` 坏了 → 报错（取数配方是仓库产物，读不出来是真问题）；
//! ⑤ `data verify` 把缺失记成**该文件的**失败报告，并在消息里给出取数命令。

use std::path::{Path, PathBuf};

use qwb_engine_api::EngineError;
use qwb_registry::Registry;
use qwb_registry::thirdparty::{check_pins, missing_data};
use qwb_registry::verify;

/// 一个合法的 H/STO-3G 形状的基组文件（带完整 G2 字段）。
const H_BASIS: &str = r#"name: h-sto3g
version: 1.0.0
citation:
  - title: fixture
    doi: 10.1000/fixture
license:
  spdx: Apache-2.0
  commercial_ok: true
  redistribute_ok: true
created: 2026-10-22T00:00:00Z
elements: [H]
angular: spherical
normalization: l2
role: orbital
shells:
  - element: H
    shells:
      - l: 0
        exponents: [3.42525091, 0.62391373, 0.16885540]
        coefficients: [[0.15432897, 0.53532814, 0.44463454]]
        contraction_type: segmented
"#;

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("qwb-d16-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create temp root");
    root
}

fn write(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent");
    }
    std::fs::write(&path, content).expect("write fixture");
}

/// 生成 registry.yaml。`entries` = (域名, 名字, 相对路径)。
fn manifest(entries: &[(&str, &str, &str)]) -> String {
    let mut text = String::from(
        "schema_version: 1.0.0\nname: fixture-data\nversion: 0.1.0\ncreated: 2026-10-22T00:00:00Z\n",
    );
    let sections = [
        ("basis", "bases"),
        ("ecp", "ecps"),
        ("functional", "functionals"),
        ("grid", "grids"),
    ];
    for (domain, section) in sections {
        let matching: Vec<_> = entries.iter().filter(|(d, _, _)| *d == domain).collect();
        if matching.is_empty() {
            continue;
        }
        text.push_str(&format!("{section}:\n"));
        for (_, name, path) in matching {
            text.push_str(&format!(
                "  - name: {name}\n    path: {path}\n    sha256: {}\n",
                "a".repeat(64)
            ));
        }
    }
    text
}

/// 生成 sources.json（只列路径与钉，够加载器用）。
fn sources(paths: &[&str]) -> String {
    let entries: Vec<String> = paths
        .iter()
        .map(|path| format!(r#"{{"path": "{path}", "sha256": "{}"}}"#, "a".repeat(64)))
        .collect();
    format!(
        r#"{{"schema_version": 1, "entries": [{}]}}"#,
        entries.join(", ")
    )
}

fn data_error(err: EngineError) -> String {
    match err {
        EngineError::Data(message) => message,
        other => panic!("expected EngineError::Data, got {other:?}"),
    }
}

// ───────── ① 缺失的第三方文件被跳过，取它时给取数命令 ─────────

#[test]
fn absent_third_party_files_are_skipped_and_hinted() {
    let root = temp_root("skip");
    write(&root, "basis/h-sto3g.yaml", H_BASIS);
    write(
        &root,
        "registry.yaml",
        &manifest(&[
            ("basis", "h-sto3g", "basis/h-sto3g.yaml"),
            ("basis", "lanl2dz", "basis/lanl2dz.yaml"),
            ("ecp", "lanl2dz-cu", "ecp/lanl2dz-cu.yaml"),
        ]),
    );
    write(
        &root,
        "sources.json",
        &sources(&["basis/lanl2dz.yaml", "ecp/lanl2dz-cu.yaml"]),
    );

    let registry = Registry::open(&root).expect("缺第三方数据不该让整个注册表打不开");
    // 在的那条照常可用
    assert!(registry.get_basis("h-sto3g").is_ok());
    assert_eq!(registry.basis_names().count(), 1, "缺失的那条没有进映射");

    let missing = registry.missing_data();
    assert_eq!(missing.len(), 2, "{missing:?}");
    assert_eq!(missing[0].domain, "basis");
    assert_eq!(missing[0].name, "lanl2dz");
    assert_eq!(missing[1].domain, "ecp");
    assert_eq!(missing[1].name, "lanl2dz-cu");

    // 按名字取缺失条目：错误里必须有"怎么办"，而不是只说 unknown basis
    let message = data_error(registry.get_basis("lanl2dz").unwrap_err());
    assert!(message.contains("lanl2dz"), "{message}");
    assert!(message.contains("第三方数据"), "{message}");
    assert!(message.contains("qwb data install"), "{message}");
    assert!(
        message.contains("basis/lanl2dz.yaml"),
        "消息里要指出是哪个文件：{message}"
    );

    let message = data_error(registry.get_ecp("lanl2dz-cu").unwrap_err());
    assert!(message.contains("qwb data install"), "{message}");

    std::fs::remove_dir_all(&root).ok();
}

// ───────── ② 非第三方文件缺失仍然是硬错误 ─────────

#[test]
fn absent_non_third_party_file_is_a_hard_error() {
    let root = temp_root("hard");
    write(&root, "basis/h-sto3g.yaml", H_BASIS);
    write(
        &root,
        "registry.yaml",
        &manifest(&[
            ("basis", "h-sto3g", "basis/h-sto3g.yaml"),
            ("grid", "grid-normal", "grid/grid-normal.yaml"),
        ]),
    );
    // grid 不在清单里：它是随仓库提交的自研数据，缺了就是数据根被裁剪
    write(&root, "sources.json", &sources(&[]));

    let message = data_error(Registry::open(&root).unwrap_err());
    assert!(message.contains("cannot read"), "{message}");
    assert!(message.contains("grid-normal"), "{message}");

    std::fs::remove_dir_all(&root).ok();
}

// ───────── ③ 没有 sources.json：什么都不放过（老数据根行为不变）─────────

#[test]
fn without_sources_json_nothing_is_skipped() {
    let root = temp_root("nosources");
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "lanl2dz", "basis/lanl2dz.yaml")]),
    );

    let message = data_error(Registry::open(&root).unwrap_err());
    assert!(message.contains("cannot read"), "{message}");
    assert!(
        !message.contains("第三方数据"),
        "没有清单就不该谈第三方：{message}"
    );

    std::fs::remove_dir_all(&root).ok();
}

// ───────── ④ 清单坏了要报错 ─────────

#[test]
fn malformed_sources_json_is_an_error() {
    let root = temp_root("badsources");
    write(&root, "basis/h-sto3g.yaml", H_BASIS);
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "h-sto3g", "basis/h-sto3g.yaml")]),
    );
    write(&root, "sources.json", "{ not json");

    let message = data_error(Registry::open(&root).unwrap_err());
    assert!(message.contains("sources.json"), "{message}");

    std::fs::remove_dir_all(&root).ok();
}

// ───────── ⑤ data verify 把缺失记成该文件的失败 ─────────

#[test]
fn verify_reports_missing_third_party_per_file() {
    let root = temp_root("verify");
    write(&root, "basis/h-sto3g.yaml", H_BASIS);
    write(
        &root,
        "registry.yaml",
        &manifest(&[
            ("basis", "h-sto3g", "basis/h-sto3g.yaml"),
            ("basis", "lanl2dz", "basis/lanl2dz.yaml"),
        ]),
    );
    write(&root, "sources.json", &sources(&["basis/lanl2dz.yaml"]));

    let reports = verify::verify_registry(&root.join("registry.yaml")).expect("verify runs");
    // 1 个清单报告 + 2 个文件报告：缺失的那条**单独**记一笔，而不是把清单标红
    assert_eq!(reports.len(), 3, "{reports:#?}");
    assert!(
        reports[0].is_ok(),
        "清单本身没问题（缺文件是那条文件的事）：{:#?}",
        reports[0].issues
    );
    let missing_report = reports
        .iter()
        .find(|report| report.path.contains("lanl2dz"))
        .expect("缺失文件应当有自己的报告");
    assert!(!missing_report.is_ok());
    let messages = missing_report
        .issues
        .iter()
        .map(|issue| issue.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(messages.contains("qwb data install"), "{messages}");

    std::fs::remove_dir_all(&root).ok();
}

// ───────── 清单与 registry.yaml 的钉必须一致（仓库自检）─────────

#[test]
fn missing_data_helper_matches_registry_view() {
    let root = temp_root("helper");
    write(&root, "basis/h-sto3g.yaml", H_BASIS);
    write(
        &root,
        "registry.yaml",
        &manifest(&[
            ("basis", "h-sto3g", "basis/h-sto3g.yaml"),
            ("basis", "lanl2dz", "basis/lanl2dz.yaml"),
        ]),
    );
    write(&root, "sources.json", &sources(&["basis/lanl2dz.yaml"]));

    // 两种口径（加载器看到的 missing_data 与 CLI 预检用的 missing_data）必须一致
    let from_helper = missing_data(&root).expect("helper works");
    let registry = Registry::open(&root).expect("open");
    assert_eq!(from_helper.len(), 1);
    assert_eq!(from_helper[0].path, registry.missing_data()[0].path);

    // 钉一致时 check_pins 无问题
    let manifest_text = std::fs::read_to_string(root.join("registry.yaml")).expect("read");
    let parsed: qwb_registry::RegistryManifest =
        serde_yaml::from_str(&manifest_text).expect("parse registry.yaml");
    assert!(check_pins(&root, &parsed).expect("check pins").is_empty());

    std::fs::remove_dir_all(&root).ok();
}
