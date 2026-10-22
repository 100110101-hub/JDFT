//! D-13 验收（project.md §6.2）：网格等级 `coarse/normal/fine/veryfine`。
//!
//! §6.2 冻结的初始参数（`radial_points`, `lebedev_order`, `becke_sharpness`）：
//! `(50,194,0.6)`、`(75,302,0.6)`、`(99,590,0.6)`、`(140,974,0.8)`，`cutoff = 20 Bohr`。
//!
//! "相邻等级能量差 < 0.1 mHa" 是 T2 的断言（需要 SCF + XC 网格求值），本测试只覆盖
//! **数据与查找路径**；能量门禁在 T2 之前不会被标 done。

use std::path::{Path, PathBuf};

use qwb_registry::Registry;
use qwb_registry::grid::{GridLevelDef, GridLevelName, LEBEDEV_ORDERS};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn registry() -> Registry {
    Registry::open(repo_root().join("qwb-data")).expect("qwb-data must load")
}

/// (等级, radial, lebedev, sharpness) —— §6.2 冻结值。
const FROZEN: [(GridLevelName, u32, u32, f64); 4] = [
    (GridLevelName::Coarse, 50, 194, 0.6),
    (GridLevelName::Normal, 75, 302, 0.6),
    (GridLevelName::Fine, 99, 590, 0.6),
    (GridLevelName::VeryFine, 140, 974, 0.8),
];

#[test]
fn four_levels_exist_with_the_frozen_parameters() {
    let registry = registry();
    for (level, radial, lebedev, sharpness) in FROZEN {
        let grid = registry
            .get_grid_by_level(level)
            .unwrap_or_else(|err| panic!("缺 {level:?} 等级: {err}"));

        assert_eq!(grid.level, level);
        assert_eq!(grid.radial_points, radial, "{level:?} radial_points");
        assert_eq!(grid.lebedev_order, lebedev, "{level:?} lebedev_order");
        assert_eq!(grid.becke_sharpness, sharpness, "{level:?} becke_sharpness");
        assert_eq!(grid.cutoff, 20.0, "{level:?} cutoff");
        assert!(
            LEBEDEV_ORDERS.contains(&grid.lebedev_order),
            "{level:?}: {} 不是合法 Lebedev 阶数",
            grid.lebedev_order
        );
        assert!(grid.check().is_empty(), "{level:?}: {:?}", grid.check());
    }
}

#[test]
fn levels_are_monotonically_refined() {
    let registry = registry();
    let mut previous: Option<GridLevelDef> = None;
    for (level, ..) in FROZEN {
        let current = registry.get_grid_by_level(level).expect("level");
        if let Some(previous) = &previous {
            assert!(
                current.radial_points > previous.radial_points,
                "{level:?} 的径向点数必须比上一级多"
            );
            assert!(
                current.lebedev_order > previous.lebedev_order,
                "{level:?} 的角向阶数必须比上一级多"
            );
        }
        previous = Some(current.clone());
    }
}

#[test]
fn each_level_is_declared_exactly_once() {
    let registry = registry();
    // 四个等级都要有定义，且 `grid_names()` 恰好覆盖它们。
    let names: Vec<&str> = registry.grid_names().collect();
    assert_eq!(names.len(), FROZEN.len(), "names = {names:?}");
    for (level, ..) in FROZEN {
        let count = names
            .iter()
            .filter(|name| {
                registry
                    .get_grid(name)
                    .map(|grid| grid.level == level)
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(count, 1, "{level:?} 必须恰好有一个定义");
    }
}

#[test]
fn ambiguous_or_missing_levels_are_data_errors() {
    // 无网格定义 → Data（并列出可用等级）。
    let root = std::env::temp_dir().join(format!("qwb-d13-empty-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("registry.yaml"),
        "schema_version: 1.0.0\nname: empty\nversion: 0.1.0\ncreated: 2026-10-22T00:00:00Z\n",
    )
    .unwrap();
    let empty = Registry::open(&root).expect("空注册表可加载");
    let err = empty
        .get_grid_by_level(GridLevelName::Normal)
        .expect_err("没有网格定义");
    let message = format!("{err}");
    assert!(message.contains("no grid definition"), "{message}");

    // 两个条目声明同一等级 → Data（歧义，不许“先到先得”）。
    std::fs::create_dir_all(root.join("grid")).unwrap();
    let grid_body = |name: &str| {
        format!(
            "name: {name}\nversion: 1.0.0\ncitation:\n  - title: t\n    doi: 10.1/x\nlicense:\n  spdx: Apache-2.0\n  commercial_ok: true\n  redistribute_ok: true\ncreated: 2026-10-22T00:00:00Z\nlevel: normal\nradial_points: 75\nlebedev_order: 302\nbecke_sharpness: 0.6\ncutoff: 20.0\n"
        )
    };
    std::fs::write(root.join("grid/a.yaml"), grid_body("a")).unwrap();
    std::fs::write(root.join("grid/b.yaml"), grid_body("b")).unwrap();
    std::fs::write(
        root.join("registry.yaml"),
        format!(
            "schema_version: 1.0.0\nname: dup\nversion: 0.1.0\ncreated: 2026-10-22T00:00:00Z\ngrids:\n  - name: a\n    path: grid/a.yaml\n    sha256: {z}\n  - name: b\n    path: grid/b.yaml\n    sha256: {z}\n",
            z = "a".repeat(64)
        ),
    )
    .unwrap();

    let duplicated = Registry::open(&root).expect("两个文件本身合法");
    let err = duplicated
        .get_grid_by_level(GridLevelName::Normal)
        .expect_err("等级重复必须报错");
    let message = format!("{err}");
    assert!(message.contains("ambiguous grid level"), "{message}");
    assert!(message.contains('a') && message.contains('b'), "{message}");

    std::fs::remove_dir_all(&root).ok();
}
