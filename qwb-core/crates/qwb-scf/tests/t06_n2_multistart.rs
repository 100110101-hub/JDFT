//! N2 的初猜陷阱与多初猜修复（2026-11-03 定位）。
//!
//! **现象**：N2/STO-3G 的 HF 总能量比 CCCBDB 高 **0.684 Ha**，而 H2/H2O/CO/CH4/NH3
//! 五个体系都在 1e-6~1e-4 内。
//!
//! **根因**：纯 core guess（对角化 `H_core`）在 N2 这种近简并体系上会收敛到**激发态解**。
//! 证据：该解占据轨道能级是 `[…, -0.5889, -0.5081, -0.1829]`——1π_u 不成对简并、HOMO 被
//! 抬到 -0.18 Ha；把初猜的 `H_core` 非对角块缩到 0.5 之后，能级变成
//! `[…, -0.54850, -0.54850, -0.53021]`（π_u 精确简并、HOMO = 3σ_g = -0.530），
//! 总能量落到 **-107.5006542**，与 CCCBDB 的 -107.500654 差 2.4e-7 Ha。
//!
//! **修复**：`RksOptions.initial_guess_scale` + `rks_scf_multistart`（依次试
//! 1.0 / 0.5 / 1.5 / 0.25 / 2.0，取能量最低的收敛解）。这个测试就是回归：
//! 五个模型化学都必须与 CCCBDB 差 < 2 mHa。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf_multistart};
use qwb_xc::FunctionalTerm;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

fn n2(method: &str) -> (Molecule, f64) {
    let geo = std::fs::read_to_string(reference_dir().join("cccbdb_geometries.csv")).expect("geo");
    let mut atoms = Vec::new();
    for line in geo.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f[0] == "N2" && f[1] == method && f[2] == "STO-3G" {
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
    assert_eq!(atoms.len(), 2, "N2 应有 2 个原子");
    let ene = std::fs::read_to_string(reference_dir().join("cccbdb_energies.csv")).expect("ene");
    for line in ene.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f[0] == "N2" && f[1] == method && f[2] == "STO-3G" {
            return (Molecule::new(atoms, 0, 1), f[3].parse().expect("energy"));
        }
    }
    panic!("没有 N2 / {method} 的参考值");
}

fn terms(registry: &Registry, method: &str) -> (Vec<Box<dyn FunctionalTerm>>, f64) {
    let mut p = std::collections::BTreeMap::new();
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
    match method {
        "HF" => (Vec::new(), 0.0),
        "B3LYP" => (
            vec![
                qwb_xc::scale_term(qwb_xc::build_term("LdaSlaterX", &p).unwrap(), 0.08),
                qwb_xc::scale_term(qwb_xc::build_term("GgaB88X", &p).unwrap(), 0.72),
                qwb_xc::scale_term(qwb_xc::build_term("GgaLypC", &p).unwrap(), 0.81),
                qwb_xc::scale_term(qwb_xc::build_term("LdaSvwn5", &p).unwrap(), 0.19),
            ],
            0.20,
        ),
        "BLYP" => (
            vec![
                qwb_xc::build_term("GgaB88X", &p).unwrap(),
                qwb_xc::build_term("GgaLypC", &p).unwrap(),
            ],
            0.0,
        ),
        name => {
            let f = registry
                .get_functional(match name {
                    "LSDA" => "svwn5",
                    "PBEPBE" => "pbe",
                    other => panic!("未映射 {other}"),
                })
                .unwrap();
            (
                f.components
                    .iter()
                    .map(|c| {
                        qwb_xc::scale_term(
                            qwb_xc::build_term(&c.id, &f.parameters).unwrap(),
                            c.weight,
                        )
                    })
                    .collect(),
                0.0,
            )
        }
    }
}

#[test]
fn n2_matches_cccbdb_with_multistart() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    for method in ["HF", "LSDA", "PBEPBE", "BLYP", "B3LYP"] {
        let (molecule, reference) = n2(method);
        let basis =
            qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule)
                .expect("basis");
        let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
        let grid = qwb_grid::molecular_grid(registry.get_grid("grid-normal").unwrap(), &molecule)
            .expect("grid");
        let (terms, hf) = terms(&registry, method);
        let mut options = RksOptions::default();
        options.max_iter = 400;
        options.energy_threshold = 1.0e-9;
        options.hf_exchange = hf;
        let result = rks_scf_multistart(&basis, &molecule, &tensor, &grid, &terms, options)
            .expect("multistart scf");
        let delta = result.energy_hartree - reference;
        println!(
            "N2 {method:7}: ours {:.9}  CCCBDB {reference:.9}  Δ = {delta:+.2e} Ha ({:+.3} mHa)",
            result.energy_hartree,
            delta * 1e3
        );
        assert!(
            delta.abs() < 2.0e-3,
            "N2/{method} 与 CCCBDB 差 {delta:+.2e} Ha（多初猜应该能收敛到基态）"
        );
    }
}
