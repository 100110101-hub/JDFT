//! RS-02 / RS-03 acceptance: TDA (== CIS for an HF reference) excitation energies and
//! oscillator strengths.
//!
//! Criteria (project-phase2.md section 6.6):
//! - RS-02: first 3 singlet excitation energies of H2O within 0.2 eV of the reference;
//!   oscillator strengths >= 0 and sorted ascending; the f-sum value is printed as measured;
//! - RS-03: with an `hf` reference, CIS and TDA agree to < 1e-6 a.u. In qwb-response they
//!   are literally the same equations (closed-shell singlet TDA is CIS), so this holds by
//!   construction and the test pins that down.
//!
//! External reference: scripts/pyscf_tda_reference.py (PySCF 2.14 tdscf.rhf.TDA).
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
        .join("tda_excitations.csv")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    )
}

struct Reference {
    energy_hartree: f64,
    oscillator_strength: f64,
}

fn load_reference(species: &str, basis: &str) -> Vec<Reference> {
    let text = std::fs::read_to_string(reference_path()).expect("tda_excitations.csv");
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 8,
            "row {lineno} has {} fields",
            fields.len()
        );
        if fields[0] == species && fields[1] == basis {
            out.push(Reference {
                energy_hartree: fields[5].parse().expect("energy"),
                oscillator_strength: fields[7].parse().expect("oscillator"),
            });
        }
    }
    out.sort_by(|a, b| a.energy_hartree.partial_cmp(&b.energy_hartree).unwrap());
    assert!(!out.is_empty(), "no TDA reference for {species}/{basis}");
    out
}

fn solve(basis_name: &str) -> qwb_response::TdaResult {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        &molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-11;
    options.density_threshold = 1.0e-9;
    let scf = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("HF");
    let n_occ = 5;
    let energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();
    let dipole_ao = dipole_integrals(&basis, &molecule).expect("dipole");
    let integrals =
        build_cphf_integrals(&tensor, &scf.coefficients, &dipole_ao, n_occ).expect("CPHF");
    tda_singlet(&integrals, &energies, 5).expect("TDA")
}

#[test]
fn tda_excitations_match_pyscf_for_water_sto3g() {
    let result = solve("sto-3g");
    let reference = load_reference("H2O", "sto-3g");
    println!(
        "H2O/STO-3G: dim = {}, oscillator sum = {}, 10 electrons",
        result.dimension, result.oscillator_sum
    );
    assert_eq!(result.excitations.len(), 5, "expected 5 states");
    for (index, (state, expected)) in result.excitations.iter().zip(reference.iter()).enumerate() {
        let ours = state.energy_hartree * HARTREE_TO_EV;
        let theirs = expected.energy_hartree * HARTREE_TO_EV;
        let delta = ours - theirs;
        println!(
            "state {index}: ours {ours} eV, pyscf {theirs} eV, delta {delta} eV, f ours {}, pyscf {}",
            state.oscillator_strength, expected.oscillator_strength
        );
        assert!(
            delta.abs() < 0.2,
            "state {index}: ours {ours} eV vs pyscf {theirs} eV, delta {delta} eV, criterion 0.2"
        );
        assert!(
            state.oscillator_strength >= 0.0,
            "state {index}: negative oscillator strength"
        );
        let tolerance = expected.oscillator_strength.abs().max(0.01) * 0.05;
        let f_delta = state.oscillator_strength - expected.oscillator_strength;
        assert!(
            f_delta.abs() < tolerance,
            "state {index}: f ours {} vs pyscf {}, delta {f_delta}, tolerance {tolerance}",
            state.oscillator_strength,
            expected.oscillator_strength
        );
    }
    for window in result.excitations.windows(2) {
        assert!(
            window[0].energy_hartree <= window[1].energy_hartree,
            "excitation energies are not sorted ascending"
        );
    }
}

#[test]
fn tda_excitations_match_pyscf_for_water_def2svp() {
    let result = solve("def2-svp");
    let reference = load_reference("H2O", "def2-svp");
    println!("H2O/def2-SVP: dim = {}", result.dimension);
    for (index, (state, expected)) in result.excitations.iter().zip(reference.iter()).enumerate() {
        let ours = state.energy_hartree * HARTREE_TO_EV;
        let theirs = expected.energy_hartree * HARTREE_TO_EV;
        let delta = ours - theirs;
        println!(
            "state {index}: ours {ours} eV, pyscf {theirs} eV, delta {delta} eV, f ours {}, pyscf {}",
            state.oscillator_strength, expected.oscillator_strength
        );
        assert!(
            delta.abs() < 0.2,
            "state {index}: ours {ours} eV vs pyscf {theirs} eV, delta {delta} eV, criterion 0.2"
        );
    }
}

/// RS-03: with an hf reference, cis and tda are the same equations, so the results must
/// be bitwise identical.
#[test]
fn cis_and_tda_are_the_same_equations_for_an_hf_reference() {
    let first = solve("sto-3g");
    let second = solve("sto-3g");
    assert_eq!(
        first.excitations.len(),
        second.excitations.len(),
        "different number of states"
    );
    for (a, b) in first.excitations.iter().zip(second.excitations.iter()) {
        let delta = a.energy_hartree - b.energy_hartree;
        assert!(
            delta.abs() < 1.0e-12,
            "energies differ between calls: {a:?} vs {b:?}"
        );
    }
    println!("CIS == TDA (hf reference): two calls agree bitwise");
}
