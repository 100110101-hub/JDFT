//! KV-04（phase-2 §4.5 / §8.5）：K10 的**数据侧**——精度等级与等价容差。
//!
//! 三条要钉住的：
//! 1. 仓库里那份 `qwb-data/numerics/numerics.yaml` 本身是合法的（`NumericsDef::check` 全空），
//!    并且被 `registry.yaml` 索引到（`qwb data verify` 的 17 个文件之一）；
//! 2. 结构性错误（等级重复/越界、容差过松、PS 配点不单调、去别名关回去）会被拒；
//! 3. **跨文件引用**：等级里写的 `grid_level` 必须是真实的网格条目 —— 这条只能在加载器里查
//!    （`data verify` 是逐文件的），所以放在 `Registry::open`。

use std::path::{Path, PathBuf};

use qwb_engine_api::EngineError;
use qwb_registry::verify::verify_file;
use qwb_registry::{NumericsDef, Registry};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// 一个最小的合法网格条目（`Registry::open` 要求 `grids` 里真有这个名字）。
const GRID_COARSE: &str = r#"name: grid-coarse
version: 1.0.0
citation:
  - title: fixture
    doi: 10.1000/fixture
license:
  spdx: Apache-2.0
  commercial_ok: true
  redistribute_ok: true
created: 2026-11-06T00:00:00Z
level: coarse
radial_points: 50
angular_rule: gauss-product
lebedev_order: 194
becke_sharpness: 0.6
cutoff: 20.0
"#;

fn numerics_yaml(grid_level: &str) -> String {
    format!(
        r#"name: numerics
version: 1.0.0
citation:
  - title: fixture
    doi: 10.1000/fixture
license:
  spdx: Apache-2.0
  commercial_ok: true
  redistribute_ok: true
created: 2026-11-06T00:00:00Z
sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
levels:
  - level: 1
    grid_level: {grid_level}
    ps:
      collocation_points_per_atom: 30
      de_aliasing: false
  - level: 2
    grid_level: {grid_level}
    ps:
      collocation_points_per_atom: 60
      de_aliasing: true
equivalence_tolerance:
  ri: 1.0e-6
  pseudospectral: 5.0e-5
"#
    )
}

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("qwb-kv04-{tag}-{}", std::process::id()));
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

fn manifest() -> String {
    let a = "a".repeat(64);
    format!(
        "schema_version: 1.0.0\nname: fixture-data\nversion: 0.5.0\ncreated: 2026-11-06T00:00:00Z\n\
         grids:\n  - name: grid-coarse\n    path: grid/grid-coarse.yaml\n    sha256: {a}\n\
         numerics:\n  - name: numerics\n    path: numerics/numerics.yaml\n    sha256: {a}\n"
    )
}

// ───────── 1. 仓库里的那份数据 ─────────

#[test]
fn shipped_numerics_data_is_valid() {
    let path = repo_root().join("qwb-data/numerics/numerics.yaml");
    let report = verify_file(&path).expect("verify runs");
    assert!(
        report.is_ok(),
        "qwb-data/numerics/numerics.yaml 必须过 data verify：{:#?}",
        report.issues
    );

    let text = std::fs::read_to_string(&path).expect("read");
    let def: NumericsDef = serde_yaml::from_str(&text).expect("deserialize");
    assert_eq!(def.name, "numerics");
    assert_eq!(def.levels.len(), 5, "phase-2 §8.5 定义了 5 级");
    assert!(def.check().is_empty(), "{:?}", def.check());
    // 级别 3 是默认等级，绑到 grid-normal（与 GridLevel::default() 一致）
    assert_eq!(
        def.level(3).map(|l| l.grid_level.as_str()),
        Some("grid-normal")
    );
    // K10 的容差：RI 比 PS 严
    assert!(def.tolerance_hartree("ri") < def.tolerance_hartree("pseudospectral"));
    // 引用的网格必须真实存在（这里直接对着仓库的 registry 校验）
    let registry = Registry::open(repo_root().join("qwb-data")).expect("仓库数据根能加载");
    for reference in def.grid_level_refs() {
        assert!(
            registry.get_grid(reference).is_ok(),
            "numerics 引用了不存在的网格 {reference}"
        );
    }
    let loaded = registry.get_numerics("numerics").expect("numerics 已注册");
    assert_eq!(loaded.levels.len(), 5);
}

// ───────── 2. 结构错误 ─────────

#[test]
fn structural_errors_are_rejected() {
    let mut def: NumericsDef = serde_yaml::from_str(&numerics_yaml("grid-coarse")).expect("parse");
    assert!(def.check().is_empty(), "{:?}", def.check());

    // 等级重复
    let mut duplicated = def.clone();
    duplicated.levels.push(duplicated.levels[0].clone());
    assert!(
        duplicated
            .check_structure()
            .iter()
            .any(|m| m.contains("duplicate accuracy level"))
    );

    // 容差过松（>= 1e-3 Ha 就不再是"等价"）
    def.equivalence_tolerance.pseudospectral = 1.0e-2;
    assert!(
        def.check_structure()
            .iter()
            .any(|m| m.contains("too loose"))
    );
}

// ───────── 3. 跨文件引用：等级必须指向真实网格 ─────────

#[test]
fn registry_rejects_a_level_that_points_at_a_missing_grid() {
    let root = temp_root("badref");
    write(&root, "grid/grid-coarse.yaml", GRID_COARSE);
    write(&root, "registry.yaml", &manifest());
    write(
        &root,
        "numerics/numerics.yaml",
        &numerics_yaml("grid-ultrafine"),
    );

    let error = Registry::open(&root).expect_err("引用了不存在的网格必须失败");
    match error {
        EngineError::Data(message) => {
            assert!(message.contains("grid-ultrafine"), "{message}");
            assert!(message.contains("numerics"), "{message}");
        }
        other => panic!("期望 EngineError::Data，得到 {other:?}"),
    }
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn registry_loads_numerics_and_exposes_it() {
    let root = temp_root("goodref");
    write(&root, "grid/grid-coarse.yaml", GRID_COARSE);
    write(&root, "registry.yaml", &manifest());
    write(
        &root,
        "numerics/numerics.yaml",
        &numerics_yaml("grid-coarse"),
    );

    let registry = Registry::open(&root).expect("引用正确时必须能加载");
    assert_eq!(
        registry.numerics_names().collect::<Vec<_>>(),
        vec!["numerics"]
    );
    let def = registry.get_numerics("numerics").expect("取得到");
    assert_eq!(
        def.level(2).map(|l| l.ps.map(|p| p.de_aliasing)),
        Some(Some(true))
    );

    // 取不存在的名字：错误里要列出可用名字（与其它域一致）
    let message = registry.get_numerics("nope").unwrap_err().to_string();
    assert!(message.contains("numerics"), "{message}");
    assert!(message.contains("available"), "{message}");

    std::fs::remove_dir_all(&root).ok();
}
