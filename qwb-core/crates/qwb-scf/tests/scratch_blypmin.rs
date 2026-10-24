//! 临时诊断（用完即删）：BLYP 的 SCF 是不是真的把 E_BLYP 压到了更低。
//! 方法：BLYP 密度 vs LSDA 密度，各自算 E_BLYP（同一套 one/two-electron 与 E_nuc）。
#![allow(clippy::field_reassign_with_default)]
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, evaluate_xc, rks_scf};
use qwb_xc::FunctionalTerm;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.000, 0.000, 0.136]),
            Atom::new("H", [0.000, 0.771, -0.543]),
            Atom::new("H", [0.000, -0.771, -0.543]),
        ],
        0,
        1,
    )
}

fn blyp_terms() -> Vec<Box<dyn FunctionalTerm>> {
    let mut p = BTreeMap::new();
    p.insert("b88_beta".to_string(), 0.0042);
    p.insert("lyp_a".to_string(), 0.049_18);
    p.insert("lyp_b".to_string(), 0.132);
    p.insert("lyp_c".to_string(), 0.253_3);
    p.insert("lyp_d".to_string(), 0.349);
    vec![
        qwb_xc::build_term("GgaB88X", &p).expect("b88"),
        qwb_xc::build_term("GgaLypC", &p).expect("lyp"),
    ]
}

#[test]
fn blyp_minimises() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("b"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-normal").expect("g"), &molecule)
        .expect("g");
    let svwn5 = registry.get_functional("svwn5").expect("svwn5");
    let lsda: Vec<Box<dyn FunctionalTerm>> = svwn5
        .components
        .iter()
        .map(|c| {
            qwb_xc::scale_term(
                qwb_xc::build_term(&c.id, &svwn5.parameters).expect("t"),
                c.weight,
            )
        })
        .collect();
    let mut options = RksOptions::default();
    options.max_iter = 800;
    options.energy_threshold = 1.0e-12;
    let mut densities = Vec::new();
    for (label, terms) in [("LSDA", lsda), ("BLYP", blyp_terms())] {
        let r = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options.clone()).expect("scf");
        println!(
            "{label} SCF: E = {:.9} iters = {} conv = {}",
            r.energy_hartree, r.iterations, r.scf_converged
        );
        densities.push(r.density.clone());
    }
    let blyp = blyp_terms();
    let enuc = 9.193_913_3_f64;
    for (label, p) in [("P_lsda", &densities[0]), ("P_blyp", &densities[1])] {
        let n = basis.n_functions();
        let mut e = 0.0;
        for i in 0..n {
            for j in 0..n {
                e += p[(i, j)] * one.core_hamiltonian[(i, j)];
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        e += 0.5 * p[(i, j)] * p[(k, l)] * tensor.get(i, j, k, l);
                    }
                }
            }
        }
        let (exc, _) = evaluate_xc(&basis, &grid, p, &blyp).expect("xc");
        println!("E_BLYP[{label}] = {:.9}  (E_xc {exc:.9})", e + exc + enuc);
    }
    let _ = Matrix::zeros(1, 1);
}
