//! VF-01 (excited-state benchmark set): acetone as the fourth molecule.
//!
//! Criterion: the benchmark reference carries a source, and our values agree with it
//! within the section 6.6 tolerances. Because VF-01's wording ("difference from the
//! library") is compared here against a *method-matched* reference (PySCF TDA on the
//! same geometry and basis), the relevant tolerance is the implementation one, not the
//! experiment-level 0.2-0.5 eV (see docs/choices.md 60).
//!
//! Reference: scripts/pyscf_tda_acetone_reference.py (PySCF 2.14 tdscf.rhf.TDA).
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
        .join("tda_acetone.csv")
}

fn acetone() -> Molecule {
    let carbons = [[0.0, 0.0, 0.0], [1.348, 0.0, -0.45], [-1.348, 0.0, -0.45]];
    let hydrogens = [
        [1.7, 0.89, 0.07],
        [1.7, -0.89, 0.07],
        [2.0, 0.0, -1.3],
        [-1.7, 0.89, 0.07],
        [-1.7, -0.89, 0.07],
        [-2.0, 0.0, -1.3],
    ];
    let mut atoms: Vec<Atom> = carbons
        .iter()
        .map(|p| Atom::new("C", [p[0], p[1], p[2]]))
        .collect();
    atoms.push(Atom::new("O", [0.0, 0.0, 1.222]));
    atoms.extend(hydrogens.iter().map(|p| Atom::new("H", [p[0], p[1], p[2]])));
    Molecule::new(atoms, 0, 1)
}

struct Reference {
    energy_hartree: f64,
    oscillator_strength: f64,
}

fn load_reference() -> Vec<Reference> {
    let text = std::fs::read_to_string(reference_path()).expect("tda_acetone.csv");
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
        assert!(
            !fields[7].trim().is_empty(),
            "row {lineno} has no source note (VF-01 requires a source)"
        );
        out.push(Reference {
            energy_hartree: fields[4].parse().expect("energy"),
            oscillator_strength: fields[6].parse().expect("oscillator"),
        });
    }
    out.sort_by(|a, b| a.energy_hartree.partial_cmp(&b.energy_hartree).unwrap());
    assert!(!out.is_empty(), "empty acetone reference");
    out
}

#[test]
fn acetone_tda_matches_the_benchmark_reference() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = acetone();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 500;
    options.energy_threshold = 1.0e-11;
    options.density_threshold = 1.0e-9;
    let scf = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("HF");
    let n_occ = 16; // 32 electrons
    let energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();
    let dipole_ao = dipole_integrals(&basis, &molecule).expect("dipole");
    let integrals =
        build_cphf_integrals(&tensor, &scf.coefficients, &dipole_ao, n_occ).expect("CPHF");
    let result = tda_singlet(&integrals, &energies, 4).expect("TDA");
    let reference = load_reference();
    println!(
        "C3H6O/STO-3G: n_orb = {}, dim = {}, oscillator sum = {:.6}",
        scf.mo_energies_hartree.len(),
        result.dimension,
        result.oscillator_sum
    );
    assert_eq!(result.excitations.len(), 4, "expected 4 states");
    let mut worst = 0.0_f64;
    for (index, (state, expected)) in result.excitations.iter().zip(reference.iter()).enumerate() {
        let ours = state.energy_hartree * HARTREE_TO_EV;
        let theirs = expected.energy_hartree * HARTREE_TO_EV;
        let delta = ours - theirs;
        worst = worst.max(delta.abs());
        println!(
            "state {index}: ours {ours} eV, reference {theirs} eV, delta {delta} eV, f ours {}, reference {}",
            state.oscillator_strength, expected.oscillator_strength
        );
        // Implementation tolerance against a method-matched reference (VF-01, restated).
        assert!(
            delta.abs() < 1.0e-3,
            "state {index}: ours {ours} eV vs reference {theirs} eV, delta {delta} eV, criterion 1e-3"
        );
        let tolerance = expected.oscillator_strength.abs().max(0.01) * 0.05;
        let f_delta = state.oscillator_strength - expected.oscillator_strength;
        assert!(
            f_delta.abs() < tolerance,
            "state {index}: f ours {} vs reference {}, delta {f_delta}",
            state.oscillator_strength,
            expected.oscillator_strength
        );
    }
    println!("worst energy deviation = {worst} eV");
}
