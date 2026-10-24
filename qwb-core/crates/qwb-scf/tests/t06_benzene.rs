//! **慢档**（`#[ignore]`）：苯的 B3LYP / PBE 与 CCCBDB 的比对（T6 / D-12 判据的第三条体系）。
//!
//! 为什么单独一个文件：苯有 42 个基函数、grid-normal 上单次 SCF 要十几分钟，
//! 放进默认 gate 会把 `core: test` 拖到 20 分钟以上。按 project.md V-04 的 CI 分层，
//! 它属于"每日/每周"档：`cargo test --release -p qwb-scf --test t06_benzene -- --ignored --nocapture`。
//!
//! 判据：B3LYP 与 CCCBDB（Release 22，`energy3x.asp` 对应页）差 < 2 mHa。
#![allow(clippy::field_reassign_with_default)]
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};
use qwb_xc::FunctionalTerm;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

/// 从参考 CSV 里取苯在某个模型化学下的几何（CCCBDB 自己优化的结构）。
fn benzene(method: &str) -> Molecule {
    let text = std::fs::read_to_string(reference_dir().join("cccbdb_geometries.csv")).expect("csv");
    let mut atoms = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f[0] == "C6H6" && f[1] == method && f[2] == "STO-3G" {
            atoms.push(Atom::new(
                f[4],
                [
                    f[5].parse().expect("x"),
                    f[6].parse().expect("y"),
                    f[7].parse().expect("z"),
                ],
            ));
        }
    }
    assert_eq!(atoms.len(), 12, "苯的几何行数不对");
    Molecule::new(atoms, 0, 1)
}

fn reference_energy(method: &str) -> f64 {
    let text = std::fs::read_to_string(reference_dir().join("cccbdb_energies.csv")).expect("csv");
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f[0] == "C6H6" && f[1] == method && f[2] == "STO-3G" {
            return f[3].parse().expect("energy");
        }
    }
    panic!("没有苯 / {method} 的参考值");
}

fn xc_parameters() -> BTreeMap<String, f64> {
    let mut p = BTreeMap::new();
    p.insert("b88_beta".to_string(), 0.0042);
    p.insert("lyp_a".to_string(), 0.049_18);
    p.insert("lyp_b".to_string(), 0.132);
    p.insert("lyp_c".to_string(), 0.253_3);
    p.insert("lyp_d".to_string(), 0.349);
    p.insert("vwn5_p_A".to_string(), 0.0310907);
    p.insert("vwn5_p_x0".to_string(), -0.409286);
    p.insert("vwn5_p_b".to_string(), 13.072);
    p.insert("vwn5_p_c".to_string(), 42.7198);
    p.insert("slater_alpha".to_string(), 2.0 / 3.0);
    p
}

/// B3LYP：0.08 Slater + 0.72 B88 + 0.81 LYP + 0.19 VWN5 + 0.20 HF。
fn b3lyp_terms() -> Vec<Box<dyn FunctionalTerm>> {
    let p = xc_parameters();
    vec![
        qwb_xc::scale_term(qwb_xc::build_term("LdaSlaterX", &p).expect("s"), 0.08),
        qwb_xc::scale_term(qwb_xc::build_term("GgaB88X", &p).expect("b"), 0.72),
        qwb_xc::scale_term(qwb_xc::build_term("GgaLypC", &p).expect("l"), 0.81),
        qwb_xc::scale_term(qwb_xc::build_term("LdaSvwn5", &p).expect("v"), 0.19),
    ]
}

fn solve(registry: &Registry, method: &str, grid_name: &str) -> (f64, u32, bool) {
    let molecule = benzene(method);
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("b"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name).expect("g"), &molecule)
        .expect("grid");
    let (terms, hf): (Vec<Box<dyn FunctionalTerm>>, f64) = match method {
        "B3LYP" => (b3lyp_terms(), 0.20),
        name => {
            let f = registry
                .get_functional(match name {
                    "PBEPBE" => "pbe",
                    other => panic!("未映射 {other}"),
                })
                .expect("f");
            (
                f.components
                    .iter()
                    .map(|c| {
                        qwb_xc::scale_term(
                            qwb_xc::build_term(&c.id, &f.parameters).expect("t"),
                            c.weight,
                        )
                    })
                    .collect(),
                0.0,
            )
        }
    };
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-9;
    options.hf_exchange = hf;
    let r = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");
    (r.energy_hartree, r.iterations, r.scf_converged)
}

#[test]
#[ignore = "慢档（约 20 分钟）：苯 42 个基函数；用 --release 跑"]
fn benzene_matches_cccbdb() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    for (method, grid_name) in [
        ("B3LYP", "grid-coarse"),
        ("B3LYP", "grid-normal"),
        ("PBEPBE", "grid-coarse"),
    ] {
        let (energy, iterations, converged) = solve(&registry, method, grid_name);
        let reference = reference_energy(method);
        let delta = energy - reference;
        println!(
            "C6H6 {method:7} {grid_name:12} ours {energy:.9}  CCCBDB {reference:.9}  \
             Δ = {delta:+.2e} Ha ({:+.2} mHa) iters={iterations} conv={converged}",
            delta * 1e3
        );
        assert!(converged, "{method}/{grid_name} 必须收敛");
        if grid_name == "grid-normal" {
            assert!(
                delta.abs() < 2.0e-3,
                "苯的 {method} 与 CCCBDB 差 {delta:+.2e} Ha，超出 2 mHa"
            );
        }
    }
}
