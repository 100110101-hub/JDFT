//! DA-04 验收：8 个溶剂都能加载，ε / n 与声明的来源表一致，`radii_set` 显式、license 齐全。
//!
//! 判据（§6, DA-04）：「ε 与 n 与 NIST/原始文献一致（水 78.3553 @ 298.15 K）；`radii_set`
//! 显式；缺 license 即失败（G2）」。本仓库八个溶剂用的是一条来源（SMD 2009 Table 1，
//! 水 78.355）；NIST 手册值在第三位小数上与它有出入，两个都算「原始文献」。这里把
//! **实际声明的值**钉住，并断言 `radii_set`/`license`/`cavity_model` 都存在 —— 任何一次
//! 改动都会被看见。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_registry::solvent::{CavityModel, RadiiSet};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// SMD 2009 Table 1（298.15 K）的 ε 与 n。
fn expected() -> BTreeMap<&'static str, (f64, f64)> {
    BTreeMap::from([
        ("water", (78.355, 1.3328)),
        ("methanol", (32.613, 1.3288)),
        ("ethanol", (24.852, 1.3611)),
        ("acetonitrile", (35.688, 1.3442)),
        ("dmso", (46.826, 1.4783)),
        ("chloroform", (4.7113, 1.4459)),
        ("toluene", (2.3741, 1.4961)),
        ("thf", (7.4257, 1.405)),
    ])
}

#[test]
fn da04_eight_solvents_are_declared_and_consistent() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let names: Vec<&str> = registry.solvent_names().collect();
    println!("solvents: {names:?}");
    let expected = expected();
    assert_eq!(
        names.len(),
        expected.len(),
        "DA-04 requires the eight solvents, found {names:?}"
    );
    for (name, (epsilon, refractive)) in &expected {
        let solvent = registry
            .get_solvent(name)
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        println!(
            "{name}: eps = {}, n = {:?}, radii_set = {:?}, cavity = {:?}",
            solvent.epsilon, solvent.refractive_index, solvent.radii_set, solvent.cavity_model
        );
        assert!(
            (solvent.epsilon - epsilon).abs() < 1.0e-3,
            "{name}: epsilon = {} but the declared source table says {epsilon}",
            solvent.epsilon
        );
        let declared = solvent
            .refractive_index
            .unwrap_or_else(|| panic!("{name}: refractive index missing"));
        assert!(
            (declared - refractive).abs() < 1.0e-4,
            "{name}: n = {declared} but the source table says {refractive}"
        );
        assert_eq!(
            solvent.radii_set,
            Some(RadiiSet::Bondi),
            "{name}: radii_set must be explicit"
        );
        assert_eq!(solvent.cavity_model, CavityModel::Pcm);
        assert!(!solvent.citation.is_empty(), "{name}: citation missing");
        assert!(
            !solvent.license.spdx.is_empty(),
            "{name}: license missing (G2)"
        );
        assert!(
            !solvent.atomic_radii.is_empty(),
            "{name}: atomic radii missing"
        );
        assert!(
            (solvent.temperature_k - 298.15).abs() < 1.0e-9,
            "{name}: temperature must be 298.15 K"
        );
    }
}

#[test]
fn da04_a_solvent_without_a_license_fails_to_load() {
    // G2：缺 license 必须失败。做法是在临时副本目录里删掉 license 段。
    let source = data_root();
    let temporary = std::env::temp_dir().join(format!("qwb-da04-{}", std::process::id()));
    if temporary.exists() {
        std::fs::remove_dir_all(&temporary).expect("clean");
    }
    let copy = |from: &std::path::Path, to: &std::path::Path| {
        std::fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
        std::fs::copy(from, to).expect("copy");
    };
    // 复制整棵数据树（小文件，够快）
    for entry in walk(&source) {
        let relative = entry.strip_prefix(&source).expect("relative");
        copy(&entry, &temporary.join(relative));
    }
    let target = temporary.join("solvent/chloroform.yaml");
    let text = std::fs::read_to_string(&target).expect("read");
    let stripped: String = text
        .lines()
        .filter(|line| {
            !line.starts_with("license:")
                && !line.starts_with("  spdx:")
                && !line.starts_with("  commercial_ok:")
                && !line.starts_with("  redistribute_ok:")
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&target, stripped).expect("write");
    let result = Registry::open(&temporary);
    println!("without license: {result:?}");
    assert!(
        result.is_err(),
        "a solvent without a license must fail to load (G2)"
    );
    let _ = std::fs::remove_dir_all(&temporary);
}

fn walk(root: &std::path::Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(&directory).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files
}
