//! SC-01c 验收：**ROHF（纯自旋）**对 5 个开壳层体系给出正确的驻点与 `⟨S²⟩ = S(S+1)`。
//!
//! **初猜必须用原子密度叠加（SAD）**：NH2 有两个 ROHF 驻点，从 H_core 出发会落到较高的那个
//! （−54.7418，比正确值高 90 mHa）—— PySCF 的默认初猜也不是 H_core，所以这一条要按
//! `UhfOptions::initial_guess` 的默认值（SAD）来跑，才有可比性。
//!
//! 参考值来自 PySCF 2.14.0 的 `scf.ROHF`（STO-3G、同一批几何、`conv_tol = 1e-10`）：
//!
//! ```text
//! OH   -74.3615307261   ⟨S²⟩ 0.75
//! O2  -147.6321669907   ⟨S²⟩ 2.00
//! NO  -127.5260734919   ⟨S²⟩ 0.75
//! CH3  -39.0721483755   ⟨S²⟩ 0.75
//! NH2  -54.8318325028   ⟨S²⟩ 0.75
//! ```
//!
//! 两侧的 STO-3G 指数位数不同（本仓库的表比 PySCF 内建的多几位），所以容差取 **1e-5 Ha**；
//! 实测偏差写在测试输出里。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_registry::Registry;
use qwb_scf::rohf::rohf_scf;
use qwb_scf::uhf::{UhfOptions, uhf_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn systems() -> Vec<(&'static str, Molecule, f64)> {
    vec![
        (
            "OH",
            Molecule::new(
                vec![
                    Atom::new("O", [0.0, 0.0, 0.0]),
                    Atom::new("H", [0.0, 0.0, 0.969_7]),
                ],
                0,
                2,
            ),
            -74.361_530_726_1,
        ),
        (
            "O2",
            Molecule::new(
                vec![
                    Atom::new("O", [0.0, 0.0, 0.0]),
                    Atom::new("O", [0.0, 0.0, 1.207_5]),
                ],
                0,
                3,
            ),
            -147.632_166_990_7,
        ),
        (
            "NO",
            Molecule::new(
                vec![
                    Atom::new("N", [0.0, 0.0, 0.0]),
                    Atom::new("O", [0.0, 0.0, 1.150_8]),
                ],
                0,
                2,
            ),
            -127.526_073_491_9,
        ),
        (
            "CH3",
            Molecule::new(
                vec![
                    Atom::new("C", [0.0, 0.0, 0.0]),
                    Atom::new("H", [0.0, 1.077_0, 0.0]),
                    Atom::new("H", [0.932_7, -0.538_5, 0.0]),
                    Atom::new("H", [-0.932_7, -0.538_5, 0.0]),
                ],
                0,
                2,
            ),
            -39.072_148_375_5,
        ),
        (
            "NH2",
            Molecule::new(
                vec![
                    Atom::new("N", [0.0, 0.0, 0.0]),
                    Atom::new("H", [0.0, 0.802_0, 0.613_0]),
                    Atom::new("H", [0.0, -0.802_0, 0.613_0]),
                ],
                0,
                2,
            ),
            -54.831_832_502_8,
        ),
    ]
}

fn options() -> UhfOptions {
    let mut options = UhfOptions::default();
    options.max_iter = 800;
    options.energy_threshold = 1.0e-10;
    options.density_threshold = 1.0e-8;
    options.use_diis = true;
    options.diis_max_vectors = 8;
    options
}

#[test]
fn rohf_matches_pyscf_for_five_open_shell_systems() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_set = registry.get_basis("sto-3g").expect("sto-3g").clone();
    let mut worst = 0.0_f64;
    for (name, molecule, reference) in systems() {
        let basis = InternalBasis::build(&basis_set, &molecule).expect("basis");
        let tensor = two_electron(&basis, &molecule).expect("2e");
        let result = rohf_scf(&basis, &molecule, &tensor, &options()).expect("rohf");
        let deviation = (result.energy_hartree - reference).abs();
        worst = worst.max(deviation);
        let spin = 0.5 * f64::from(molecule.multiplicity - 1);
        println!(
            "{name:4} E = {:.10} vs PySCF {reference:.10} (Δ = {:.2e})  {} 步  ⟨S²⟩ = {:.6} (应 {:.6})",
            result.energy_hartree,
            deviation,
            result.iterations,
            result.s_squared,
            spin * (spin + 1.0)
        );
        assert!(
            deviation < 1.0e-5,
            "{name}: ROHF 能量与 PySCF 差 {deviation:.2e} Ha"
        );
        // 纯自旋：`⟨S²⟩` 必须是**精确的** S(S+1)，不是 UHF 那种 2.0034 之类的污染值。
        assert_eq!(
            result.s_squared,
            spin * (spin + 1.0),
            "{name}: ROHF 的 ⟨S²⟩ 必须是 S(S+1)"
        );
        // 变分关系：UHF 允许自旋极化，能量不可能高于 ROHF。UHF 这一侧用更宽的判据 ——
        // NO 的 UHF 要到 ~900 步（见 `sc03_open_shell_stability`），这里只要能量可信。
        let mut relaxed = options();
        relaxed.density_threshold = 1.0e-6;
        relaxed.max_iter = 1200;
        let uhf = uhf_scf(&basis, &molecule, &tensor, relaxed).expect("uhf");
        println!(
            "{name:4} UHF {:.10} ≤ ROHF {:.10}（差 {:.2e}）",
            uhf.energy_hartree,
            result.energy_hartree,
            result.energy_hartree - uhf.energy_hartree
        );
        assert!(
            uhf.energy_hartree <= result.energy_hartree + 1.0e-10,
            "{name}: UHF 必须不高于 ROHF（变分）"
        );
    }
    println!("worst |ROHF - PySCF| = {worst:.3e} Ha");
}

#[test]
fn rohf_refuses_a_closed_shell_multiplicity() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_set = registry.get_basis("sto-3g").expect("sto-3g").clone();
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 0.969_7]),
            Atom::new("H", [0.0, 0.93, -0.24]),
        ],
        0,
        1,
    );
    let basis = InternalBasis::build(&basis_set, &molecule).expect("basis");
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let error = rohf_scf(&basis, &molecule, &tensor, &options())
        .expect_err("a singlet has no open shell; ROHF must refuse rather than silently do RHF");
    println!("multiplicity 1 -> {error}");
    assert!(matches!(error, EngineError::InvalidInput(_)));
}
