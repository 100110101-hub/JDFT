//! SC-01 的引擎侧验收（phase-2 §6.5 / §7 M1）：`qwb run` 真的能算开壳层。
//!
//! crate 级测试（`qwb-scf/tests/sc01_uhf.rs`）证明数值对；这里证明**产品面接得上**：
//!
//! 1. `scf_type: uhf` + `method_name: hf` 走 UHF 路径，能量与参考集一致；
//! 2. 结果里带 K9 的 `spin` 块（`⟨S²⟩` 与污染都必须落盘，不许只写在日志里）；
//! 3. 与 UHF 不兼容的组合**显式拒绝**：DFT 泛函（要 UKS，SC-01b）、`opt`（梯度是 SC-02）；
//! 4. 闭壳层退化：H2 的 `uhf` 与默认路径（RHF/RKS-HF）能量一致。
//!
//! 参考值来自 `qwb-scf/tests/reference/uhf_open_shell.csv`（PySCF 黑盒，§14 #12）。

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
    options.scf_type = Some(ScfType::Uhf);
    options.scf_max_iter = 200;
    // 1e-8/1e-6 已足够对上参照的 1e-6 容差；再紧只会让测试变慢（SCF 收敛是二阶的）。
    options.scf_energy_threshold_hartree = 1.0e-8;
    options.scf_density_threshold = 1.0e-6;
    JobInput::new(
        molecule,
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        options,
    )
}

/// 参考集里的一列（与 `qwb-scf/tests/sc01_uhf.rs` 同一份 CSV）。
fn reference(species: &str, method: &str, guess: &str) -> (f64, f64) {
    let path = repo_root().join("qwb-core/crates/qwb-scf/tests/reference/uhf_open_shell.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("读不到 {}: {error}", path.display()));
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("species,") || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() >= 8 && fields[0] == species && fields[1] == method && fields[3] == guess {
            return (
                fields[5].parse().expect("energy"),
                fields[6].parse().expect("s2"),
            );
        }
    }
    panic!("参考集里没有 {species}/{method}/{guess}");
}

#[test]
fn uhf_single_point_lands_energy_and_spin_in_the_result() {
    let (reference_energy, reference_s2) = reference("O2", "UHF", "atom");
    let result = backend()
        .run(&o2_input(), &qwb_engine_api::reporter::NullReporter)
        .expect("O2/UHF 应当跑通");

    assert!(
        (result.energy_hartree - reference_energy).abs() < 1.0e-6,
        "引擎给的 O2/UHF 能量 = {:.12}，参照 {:.12}",
        result.energy_hartree,
        reference_energy
    );
    let spin = result.spin.expect("K9 的 spin 块必须落盘（禁止只写日志）");
    assert_eq!(spin.n_alpha, 9);
    assert_eq!(spin.n_beta, 7);
    assert_eq!(spin.multiplicity, 3);
    assert!(
        (spin.s_squared - reference_s2).abs() < 1.0e-6,
        "<S^2> = {:.12}，参照 {:.12}",
        spin.s_squared,
        reference_s2
    );
    assert!(
        spin.spin_contamination > 1.0e-4,
        "污染必须被报出来（实测 {:+.3e}）",
        spin.spin_contamination
    );
    // 轨道字段是 α 的那一套：α 有 9 个占据，HOMO/LUMO 应当一负一正。
    let energies = result.mo_energies_hartree.expect("alpha orbital energies");
    assert!(
        (energies[8] < 0.0) && (energies[9] > 0.0),
        "alpha HOMO/LUMO 符号"
    );
}

#[test]
fn uhf_is_hartree_fock_only() {
    // `scf_type: uhf` 配 DFT 泛函：必须显式拒绝并指向 UKS 的 WP，而不是偷偷算个 RKS。
    let mut input = o2_input();
    input.method_name = "pbe".into();
    let error = backend()
        .run(&input, &qwb_engine_api::reporter::NullReporter)
        .expect_err("uhf + pbe 必须被拒");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("SC-01b"), "{message}");
            assert!(
                message.contains("UKS") || message.contains("unrestricted"),
                "{message}"
            );
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn uhf_supports_sp_and_opt_but_not_the_rest() {
    // SC-02 之后 `uhf` + `opt` 是**实装**的（解析梯度，见 tests/sc02_uhf_optimisation.rs），
    // 这里改成钉住"其余作业类型仍然被拒"，否则会拿闭壳层的 Hessian/路径当真。
    let mut input = o2_input();
    input.jobtype = JobType::Freq;
    let error = backend()
        .run(&input, &qwb_engine_api::reporter::NullReporter)
        .expect_err("uhf + freq 必须被拒");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("SC-02"), "{message}");
            assert!(message.contains("sp"), "{message}");
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn closed_shell_uhf_matches_the_closed_shell_path() {
    // 闭壳层体系：`scf_type: uhf` 与不指定 scf_type（RKS/RHF 路径）必须同值。
    let molecule = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    );
    let build = |scf_type: Option<ScfType>| {
        let mut options = TaskOptions::default();
        options.scf_type = scf_type;
        options.scf_energy_threshold_hartree = 1.0e-8;
        options.scf_density_threshold = 1.0e-6;
        JobInput::new(
            molecule.clone(),
            "sto-3g",
            "hf",
            GridLevel::Coarse,
            "vacuum",
            JobType::Sp,
            options,
        )
    };
    let closed = backend()
        .run(&build(None), &qwb_engine_api::reporter::NullReporter)
        .expect("闭壳层路径");
    let open = backend()
        .run(
            &build(Some(ScfType::Uhf)),
            &qwb_engine_api::reporter::NullReporter,
        )
        .expect("UHF 路径");
    assert!(
        (closed.energy_hartree - open.energy_hartree).abs() < 1.0e-9,
        "闭壳层退化：{:.12} vs {:.12}",
        closed.energy_hartree,
        open.energy_hartree
    );
    let spin = open.spin.expect("UHF 一定带 spin 块");
    assert!(spin.s_squared.abs() < 1.0e-10);
    // 闭壳层输入没有 spin 块（additive：只在 UHF 出现）。
    assert!(closed.spin.is_none());
}
