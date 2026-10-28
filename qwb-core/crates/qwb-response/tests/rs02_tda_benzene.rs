//! RS-02: benzene TDA (CIS) excitation energies against PySCF.
//!
//! Criteria (project-phase2.md section 6.6): the 1B2u / 1B1u states of benzene within
//! 0.3 eV. The reference is produced by scripts/pyscf_tda_benzene_reference.py
//! (PySCF 2.14 tdscf.rhf.TDA, same geometry and basis).
//!
//! Note on size: benzene/STO-3G has 36 basis functions, 21 occupied and 15 virtual
//! orbitals, so the TDA problem is 315-dimensional - cheap. (Benzene/def2-SVP would be
//! 1953-dimensional and is not run in the default tier.)
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::dipole::dipole_integrals;
use qwb_registry::Registry;
use qwb_response::mo_integrals::build_cphf_integrals;
use qwb_response::tda::tda_singlet;
use qwb_scf::rks::{RksOptions, rks_scf};

const HARTREE_TO_EV: f64 = 27.211_386_245_988;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("tda_benzene.csv")
}

fn benzene() -> Molecule {
    let ring = [
        [0.0, 1.3941],
        [1.2074, 0.69705],
        [1.2074, -0.69705],
        [0.0, -1.3941],
        [-1.2074, -0.69705],
        [-1.2074, 0.69705],
    ];
    let hydrogen = [
        [0.0, 2.4815],
        [2.149, 1.24075],
        [2.149, -1.24075],
        [0.0, -2.4815],
        [-2.149, -1.24075],
        [-2.149, 1.24075],
    ];
    let mut atoms: Vec<Atom> = ring
        .iter()
        .map(|[x, y]| Atom::new("C", [*x, *y, 0.0]))
        .collect();
    atoms.extend(hydrogen.iter().map(|[x, y]| Atom::new("H", [*x, *y, 0.0])));
    Molecule::new(atoms, 0, 1)
}

struct Reference {
    energy_hartree: f64,
    oscillator_strength: f64,
}

fn load_reference() -> Vec<Reference> {
    let text = std::fs::read_to_string(reference_path()).expect("tda_benzene.csv");
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 7,
            "row {lineno} has {} fields",
            fields.len()
        );
        out.push(Reference {
            energy_hartree: fields[4].parse().expect("energy"),
            oscillator_strength: fields[6].parse().expect("oscillator"),
        });
    }
    out.sort_by(|a, b| a.energy_hartree.partial_cmp(&b.energy_hartree).unwrap());
    assert!(!out.is_empty(), "empty benzene TDA reference");
    out
}

#[test]
fn benzene_tda_matches_pyscf() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = benzene();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-11;
    options.density_threshold = 1.0e-9;
    let scf = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("HF");
    let n_occ = 21;
    let energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();
    let dipole_ao = dipole_integrals(&basis, &molecule).expect("dipole");
    let integrals =
        build_cphf_integrals(&tensor, &scf.coefficients, &dipole_ao, n_occ).expect("CPHF");
    let result = tda_singlet(&integrals, &energies, 6).expect("TDA");
    let reference = load_reference();
    println!(
        "C6H6/STO-3G: n_orb = {}, dim = {}, oscillator sum = {:.6}",
        scf.mo_energies_hartree.len(),
        result.dimension,
        result.oscillator_sum
    );
    assert_eq!(result.excitations.len(), 6, "expected 6 states");
    let mut worst = 0.0_f64;
    for (index, (state, expected)) in result.excitations.iter().zip(reference.iter()).enumerate() {
        let ours = state.energy_hartree * HARTREE_TO_EV;
        let theirs = expected.energy_hartree * HARTREE_TO_EV;
        let delta = ours - theirs;
        worst = worst.max(delta.abs());
        println!(
            "state {index}: ours {ours} eV, pyscf {theirs} eV, delta {delta} eV, f ours {}, pyscf {}",
            state.oscillator_strength, expected.oscillator_strength
        );
        // Criterion 0.3 eV for the 1B2u/1B1u pair; apply it to all six.
        assert!(
            delta.abs() < 0.3,
            "state {index}: ours {ours} eV vs pyscf {theirs} eV, delta {delta} eV, criterion 0.3"
        );
        let tolerance = expected.oscillator_strength.abs().max(0.01) * 0.05;
        let f_delta = state.oscillator_strength - expected.oscillator_strength;
        assert!(
            f_delta.abs() < tolerance,
            "state {index}: f ours {} vs pyscf {}, delta {f_delta}",
            state.oscillator_strength,
            expected.oscillator_strength
        );
    }
    println!("worst energy deviation = {worst} eV");
}
