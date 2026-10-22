//! D-01 验收（project.md §6.2）：
//!
//! ① **空注册表必须能 `open`**（`Ok`，0 条目）；
//! ② 3 个畸形文件各返回**字段级** `Data` 错误；
//! ③ `load_pair` 元素不匹配 → `Data`。
//!
//! 另加验 §8.6 的 load ≠ verify：缺 G2 字段（`citation: []`、无 `sha256`）的文件
//! **能加载**但**过不了 verify**。

use std::path::{Path, PathBuf};

use qwb_engine_api::EngineError;
use qwb_registry::Registry;
use qwb_registry::grid::GridLevelName;
use qwb_registry::verify::{self, DataKind};

// ───────────────────────────── 夹具 ─────────────────────────────

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

/// 与 `H_BASIS` 相同但**没有 G2 字段**（空 citation、无 sha256）——用于 load≠verify。
const H_BASIS_NO_G2: &str = r#"name: h-sto3g
version: 1.0.0
citation: []
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

/// Cu 价基（用于 `load_pair` 的正确配对）。
const CU_BASIS: &str = r#"name: cu-val
version: 1.0.0
citation:
  - title: fixture
    doi: 10.1000/fixture
license:
  spdx: Apache-2.0
  commercial_ok: true
  redistribute_ok: true
created: 2026-10-22T00:00:00Z
elements: [Cu]
angular: spherical
normalization: l2
role: orbital
shells:
  - element: Cu
    shells:
      - l: 0
        exponents: [3.0, 1.0]
        coefficients: [[0.5, 0.5]]
        contraction_type: segmented
"#;

/// Cu ECP（`z_eff = 29 - 28 = 1` 自洽）。
const CU_ECP: &str = r#"name: cu-ecp
version: 1.0.0
citation:
  - title: fixture
    doi: 10.1000/fixture
license:
  spdx: Apache-2.0
  commercial_ok: true
  redistribute_ok: true
created: 2026-10-22T00:00:00Z
element: Cu
n_core_electrons: 28
z_eff: 1.0
r_power_convention: "r^(n-2)"
local_part:
  id: local
  terms:
    - power: 2
      exponent: 1.0
      coefficient: -1.0
nonlocal_parts:
  - l: 2
    id: d
    terms:
      - power: 2
        exponent: 1.5
        coefficient: 0.5
"#;

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("qwb-d01-{tag}-{}", std::process::id()));
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
fn manifest(entries: &[(&str, &str, &str)], features: &[&str]) -> String {
    let mut text = String::from(
        "schema_version: 1.0.0\nname: fixture-data\nversion: 0.1.0\ncreated: 2026-10-22T00:00:00Z\n",
    );
    if !features.is_empty() {
        text.push_str("features:\n");
        for feature in features {
            text.push_str(&format!("  - \"{feature}\"\n"));
        }
    }
    for (domain, name, path) in entries {
        let section = match *domain {
            "basis" => "bases",
            "ecp" => "ecps",
            "functional" => "functionals",
            "grid" => "grids",
            "solvent" => "solvents",
            other => panic!("unknown domain {other}"),
        };
        if !text.contains(&format!("\n{section}:\n")) {
            text.push_str(&format!("{section}:\n"));
        }
        text.push_str(&format!(
            "  - name: {name}\n    path: {path}\n    sha256: {}\n",
            "a".repeat(64)
        ));
    }
    text
}

fn data_error(err: EngineError) -> String {
    match err {
        EngineError::Data(message) => message,
        other => panic!("expected EngineError::Data, got {other:?}"),
    }
}

// ─────────────────── ① 空注册表必须能 open ───────────────────

#[test]
fn empty_registry_opens() {
    let root = temp_root("empty");
    write(&root, "registry.yaml", &manifest(&[], &[]));

    let registry = Registry::open(&root).expect("空注册表必须 Ok（§8.6：冷启动要可用）");
    assert_eq!(registry.basis_names().count(), 0);
    assert_eq!(registry.features().count(), 0);
    assert!(registry.snapshot().is_empty());
    assert_eq!(registry.root(), root.as_path());

    // 但 verify 必须拒绝它（G2）：这才是 load ≠ verify。
    let reports = verify::verify_registry(&root.join("registry.yaml")).expect("verify runs");
    assert!(
        !reports[0].is_ok(),
        "空注册表必须过不了 G2（data verify 退出码 3）"
    );
    assert!(
        reports[0]
            .errors()
            .any(|issue| issue.message.contains("no data entries")),
        "{:#?}",
        reports[0].issues
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn load_accepts_missing_g2_fields_but_verify_rejects_them() {
    let root = temp_root("nog2");
    write(&root, "basis/h.yaml", H_BASIS_NO_G2);
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "h-sto3g", "basis/h.yaml")], &[]),
    );

    // load：结构合法就够 → Ok
    let registry = Registry::open(&root).expect("缺 G2 字段不应阻止加载");
    assert_eq!(
        registry.get_basis("h-sto3g").unwrap().n_basis_functions(),
        1
    );

    // verify：G2 门禁 → 拒
    let report = verify::verify_bytes(DataKind::Basis, "basis/h.yaml", H_BASIS_NO_G2.as_bytes());
    assert!(!report.is_ok());
    let codes: Vec<_> = report.errors().map(|issue| issue.code).collect();
    assert!(codes.contains(&"g2.citation"), "codes = {codes:?}");
    assert!(codes.contains(&"g2.sha256"), "codes = {codes:?}");

    std::fs::remove_dir_all(&root).ok();
}

// ──────────────── ② 3 个畸形文件 → 字段级 Data ────────────────

#[test]
fn malformed_yaml_is_a_data_error() {
    let root = temp_root("bad-yaml");
    write(&root, "basis/broken.yaml", "name: [unclosed\n");
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "broken", "basis/broken.yaml")], &[]),
    );

    let message = data_error(Registry::open(&root).expect_err("must fail"));
    assert!(message.contains("basis/broken"), "message = {message}");
    assert!(message.contains("basis/broken.yaml"), "message = {message}");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn unknown_field_is_a_field_level_data_error() {
    let root = temp_root("bad-field");
    let broken = H_BASIS.replace("angular: spherical", "angelar: spherical");
    write(&root, "basis/badfield.yaml", &broken);
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "h-sto3g", "basis/badfield.yaml")], &[]),
    );

    let message = data_error(Registry::open(&root).expect_err("must fail"));
    assert!(
        message.contains("unknown field") && message.contains("angelar"),
        "字段级错误应点名拼错的字段，实际 = {message}"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn semantic_violation_is_a_field_level_data_error() {
    let root = temp_root("bad-semantic");
    // 指数升序（违反 §8.1 "从大到小"）。
    let broken = H_BASIS.replace(
        "exponents: [3.42525091, 0.62391373, 0.16885540]",
        "exponents: [0.16885540, 0.62391373, 3.42525091]",
    );
    write(&root, "basis/badsem.yaml", &broken);
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "h-sto3g", "basis/badsem.yaml")], &[]),
    );

    let message = data_error(Registry::open(&root).expect_err("must fail"));
    assert!(
        message.contains("exponents") && message.contains("large -> small"),
        "字段级错误应点名 shells 的 exponents 顺序，实际 = {message}"
    );
    std::fs::remove_dir_all(&root).ok();
}

// ──────────────── ③ load_pair 元素不匹配 → Data ────────────────

#[test]
fn load_pair_rejects_element_mismatch() {
    let root = temp_root("pair-mismatch");
    write(&root, "basis/h.yaml", H_BASIS);
    write(&root, "ecp/cu.yaml", CU_ECP);
    write(
        &root,
        "registry.yaml",
        &manifest(
            &[
                ("basis", "h-sto3g", "basis/h.yaml"),
                ("ecp", "cu-ecp", "ecp/cu.yaml"),
            ],
            &[],
        ),
    );

    let registry = Registry::open(&root).expect("both files are individually valid");
    let message = data_error(
        registry
            .load_pair("h-sto3g", "cu-ecp")
            .expect_err("H 基组 + Cu ECP 必须被拒"),
    );
    assert!(message.contains("Cu"), "message = {message}");
    assert!(message.contains("h-sto3g"), "message = {message}");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn load_pair_accepts_matching_pair() {
    let root = temp_root("pair-ok");
    write(&root, "basis/cu.yaml", CU_BASIS);
    write(&root, "ecp/cu.yaml", CU_ECP);
    write(
        &root,
        "registry.yaml",
        &manifest(
            &[
                ("basis", "cu-val", "basis/cu.yaml"),
                ("ecp", "cu-ecp", "ecp/cu.yaml"),
            ],
            &[],
        ),
    );

    let registry = Registry::open(&root).expect("open");
    let (basis, ecp) = registry.load_pair("cu-val", "cu-ecp").expect("pairing ok");
    assert_eq!(basis.elements, vec!["Cu".to_string()]);
    assert_eq!(ecp.n_core_electrons, 28);
    assert!((ecp.z_eff - 1.0).abs() < 1e-12);

    std::fs::remove_dir_all(&root).ok();
}

// ─────────────────────── 其它结构性拒绝 ───────────────────────

#[test]
fn bad_root_and_missing_manifest_are_data_errors() {
    let err = Registry::open(std::env::temp_dir().join("qwb-d01-does-not-exist"))
        .expect_err("missing root");
    assert!(data_error(err).contains("not a directory"));

    let root = temp_root("no-manifest");
    let err = Registry::open(&root).expect_err("missing registry.yaml");
    assert!(data_error(err).contains("registry.yaml"));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn missing_entry_file_is_a_data_error() {
    let root = temp_root("no-file");
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "h-sto3g", "basis/nope.yaml")], &[]),
    );
    let message = data_error(Registry::open(&root).expect_err("must fail"));
    assert!(message.contains("cannot read"), "message = {message}");
    assert!(message.contains("basis/nope.yaml"), "message = {message}");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn manifest_name_must_match_the_file_name() {
    let root = temp_root("name-mismatch");
    write(&root, "basis/h.yaml", H_BASIS);
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "some-other-name", "basis/h.yaml")], &[]),
    );
    let message = data_error(Registry::open(&root).expect_err("must fail"));
    assert!(
        message.contains("declares name `h-sto3g`"),
        "message = {message}"
    );
    assert!(message.contains("some-other-name"), "message = {message}");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn duplicate_entry_names_are_rejected() {
    let root = temp_root("duplicate");
    write(&root, "basis/h.yaml", H_BASIS);
    // 同一个名字在 manifest 里被索引两次。
    let text = format!(
        "{}  - name: h-sto3g\n    path: basis/h.yaml\n    sha256: {}\n",
        manifest(&[("basis", "h-sto3g", "basis/h.yaml")], &[]),
        "b".repeat(64)
    );
    write(&root, "registry.yaml", &text);
    let message = data_error(Registry::open(&root).expect_err("must fail"));
    assert!(message.contains("duplicate"), "message = {message}");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn unknown_name_lists_the_available_ones() {
    let root = temp_root("unknown");
    write(&root, "basis/h.yaml", H_BASIS);
    write(
        &root,
        "registry.yaml",
        &manifest(&[("basis", "h-sto3g", "basis/h.yaml")], &[]),
    );
    let registry = Registry::open(&root).expect("open");
    let message = data_error(registry.get_basis("def2-svp").expect_err("unknown"));
    assert!(message.contains("h-sto3g"), "message = {message}");
    std::fs::remove_dir_all(&root).ok();
}

// ─────────────────────── 真实数据仓冒烟 ───────────────────────

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

#[test]
fn real_qwb_data_registry_loads() {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data must load");

    // 不写死条目数（数据会随 D 轨增长）：用清单自身的条目数做交叉核对。
    assert_eq!(
        registry.snapshot().len(),
        registry.manifest().all_entries().count(),
        "snapshot 必须覆盖清单里的每一条"
    );
    assert!(registry.snapshot().len() >= 5, "至少还有 5 个样例条目");

    // 六类样例都必须仍可加载（D 轨只会新增，不会删样例）。
    assert_eq!(
        registry
            .get_basis("example-sto-3g-H")
            .expect("basis")
            .n_basis_functions(),
        1
    );
    assert!(registry.get_ecp("example-ecp-cu").is_ok());
    assert!(registry.get_functional("example-svwn5").is_ok());
    assert!(registry.get_grid("example-grid-normal").is_ok());
    assert!(registry.get_solvent("example-water").is_ok());

    let grid = registry
        .get_grid_by_level(GridLevelName::Normal)
        .expect("normal grid");
    assert_eq!(grid.lebedev_order, 302);

    // feature 声明要能直接喂给 K2 的引擎句柄。
    let handle = registry.engine_handle();
    let declared: Vec<&str> = handle.declared_features().collect();
    assert!(declared.contains(&"jobtype:sp"), "declared = {declared:?}");
    for feature in &declared {
        assert!(
            qwb_engine_api::SUPPORTED_FEATURES.contains(feature),
            "feature `{feature}` 不在 K2 词汇表里"
        );
    }
}

#[test]
fn registry_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Registry>();
}
