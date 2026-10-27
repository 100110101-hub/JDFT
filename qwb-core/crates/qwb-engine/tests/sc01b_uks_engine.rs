//! SC-01b / M1b 的引擎侧验收：`qwb run` 真的能算开壳层 DFT（UKS）。
//!
//! crate 级测试（`qwb-scf/tests/sc01b_uks.rs`）证明数值对；这里证明**产品面接得上**：
//!
//! 1. `scf_type: uks` + `method_name: pbe` 走 UKS 路径，能量与 PySCF 参考集一致（< 1e-6）；
//! 2. 结果里带 K9 的 `spin` 块（`⟨S²⟩` 与污染必须落盘）；
//! 3. 与 UKS 不兼容的组合**显式拒绝**：没有自旋实现的泛函（B88/LYP）、`opt`（SC-02）。
//!
//! 参考值来自 `qwb-scf/tests/reference/uks_open_shell.csv`（PySCF + libxc，§14 #12）。

use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, ScfType, TaskOptions,
};
use qwb_registry::Registry;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(repo_root().join("qwb-data")).expect("qwb-data"))
}

fn o2_input() -> JobInput {
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.208]),
        ],
        0,
        3,
    );
    let mut options = TaskOptions::default();
    options.scf_type = Some(ScfType::Uks);
    options.scf_max_iter = 300;
    options.scf_energy_threshold_hartree = 1.0e-8;
    options.scf_density_threshold = 1.0e-6;
    JobInput::new(
        molecule,
        "sto-3g",
        "pbe",
        GridLevel::Fine,
        "vacuum",
        JobType::Sp,
        options,
    )
}

/// 参考集里的一行（`qwb-scf/tests/reference/uks_open_shell.csv`）。
fn reference(species: &str, functional: &str) -> (f64, f64, f64) {
    let path = repo_root().join("qwb-core/crates/qwb-scf/tests/reference/uks_open_shell.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("读不到 {}: {error}", path.display()));
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() >= 9 && fields[0] == species && fields[1] == functional {
            return (
                fields[4].parse().expect("energy"),
                fields[5].parse().expect("s_squared"),
                fields[8].parse().expect("tolerance"),
            );
        }
    }
    panic!("参考集里没有 {species}/{functional}");
}

#[test]
fn uks_single_point_lands_energy_and_spin_in_the_result() {
    let (reference_energy, reference_s2, tolerance) = reference("O2", "pbe");
    let result = backend()
        .run(&o2_input(), &qwb_engine_api::reporter::NullReporter)
        .expect("O2/UKS-PBE 应当跑通");
    assert!(
        (result.energy_hartree - reference_energy).abs() < tolerance,
        "引擎给的 O2/UKS 能量 = {:.12}，参照 {:.12}（容差 {tolerance:.1e}）",
        result.energy_hartree,
        reference_energy
    );
    let spin = result.spin.expect("K9 的 spin 块必须落盘（禁止只写日志）");
    assert_eq!(spin.n_alpha, 9);
    assert_eq!(spin.n_beta, 7);
    assert_eq!(spin.multiplicity, 3);
    assert!(
        (spin.s_squared - reference_s2).abs() < 1.0e-4,
        "<S^2> = {:.9}，参照 {:.9}",
        spin.s_squared,
        reference_s2
    );
    // UKS 的 ⟨S²⟩ 近纯（2.000392），污染量比 UHF 小两个数量级，但**仍然要报**。
    assert!(
        spin.spin_contamination.is_finite() && spin.spin_contamination > 0.0,
        "污染必须被报出来（实测 {:+.3e}）",
        spin.spin_contamination
    );
    assert!(
        spin.spin_contamination < 1.0e-3,
        "UKS 的 O2 污染应当远小于 UHF 的 3.4e-3，实测 {:+.3e}",
        spin.spin_contamination
    );
    let energies = result.mo_energies_hartree.expect("alpha orbital energies");
    assert!(
        (energies[8] < 0.0) && (energies[9] > 0.0),
        "alpha HOMO/LUMO 符号"
    );
}

#[test]
fn uks_rejects_functionals_without_spin_support() {
    // b88 的交换项没有自旋实现：必须显式拒绝，不许静默用闭壳层公式。
    let mut input = o2_input();
    input.method_name = "b88".into();
    let error = backend()
        .run(&input, &qwb_engine_api::reporter::NullReporter)
        .expect_err("uks + b88 必须被拒");
    match error {
        EngineError::Unsupported(message) => {
            assert!(
                message.contains("spin") || message.contains("GgaB88X"),
                "{message}"
            );
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn uks_rejects_hartree_fock() {
    let mut input = o2_input();
    input.method_name = "hf".into();
    let error = backend()
        .run(&input, &qwb_engine_api::reporter::NullReporter)
        .expect_err("uks + hf 必须被拒");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("uhf"), "{message}");
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn uks_is_single_point_only_for_now() {
    let mut input = o2_input();
    input.jobtype = JobType::Opt;
    let error = backend()
        .run(&input, &qwb_engine_api::reporter::NullReporter)
        .expect_err("uks + opt 必须被拒");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("SC-02"), "{message}");
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}
