//! SC-01（phase-2 §6.5 / M1）：**UHF 与自旋诊断**。
//!
//! 判据来自 `tests/reference/uhf_open_shell.csv`（**外部来源**：PySCF 2.14.0 黑盒，
//! §14 #12 的边界），几何与那份 CSV 逐字一致。被测的五件事：
//!
//! 1. **闭壳层退化**：H2/STO-3G 的 UHF 必须复现 RHF（能量 < 1e-6 Ha），`⟨S²⟩ = 0`，
//!    两个自旋通道的密度逐元素相同；
//! 2. **开壳层能量**：O2 三重态、OH 双峰的能量与外部参照差 < 1e-6 Ha；
//! 3. **自旋污染必须报出来**：O2 的 `⟨S²⟩ = 2.0034`（不是 2.0）、OH 的 `0.7533`
//!    （不是 0.75）—— 与 PySCF 逐位相同。这条同时是"§7 M1 原来那句 `⟨S²⟩ = 2.0`
//!    不可能成立"的证据：纯自旋态要 ROHF（`SC-01b`），UHF 的本性就是污染；
//! 4. **解的选择**：O2 的 UHF 有两个 SCF 解，`hcore` 初猜落到 +0.255 Ha 的较高解、
//!    原子密度初猜落到变分极小。CSV 里两个都记着，本文件两个都验 ——
//!    前者证明"同一初猜下与独立实现等价"，后者是真正的目标值；
//! 5. **电子数自洽**：`Tr[P_σ S] = n_σ`，且 `n_α − n_β = 2S`。
//!
//! 另加两条不依赖外部值的检查：两次运行逐位相同（确定性）、多重度与电子数矛盾时报
//! `InvalidInput`（不做"尽力猜"）。

use std::path::PathBuf;

use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::uhf::{UhfInitialGuess, UhfOptions, UhfResult, uhf_scf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn reference_csv() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/uhf_open_shell.csv")
}

/// 从参考 CSV 里取一行 `(energy, s_squared, tolerance)`。
///
/// 列：`species,method,basis,guess,multiplicity,energy_hartree,s_squared,tolerance_hartree,...`
fn reference(species: &str, method: &str, guess: &str) -> (f64, f64, f64) {
    let text = std::fs::read_to_string(reference_csv()).expect("读 reference/uhf_open_shell.csv");
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("species,") || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 8 {
            continue;
        }
        if fields[0] == species && fields[1] == method && fields[3] == guess {
            return (
                fields[5].parse().expect("energy_hartree"),
                fields[6].parse().expect("s_squared"),
                fields[7].parse().expect("tolerance_hartree"),
            );
        }
    }
    panic!("参考集里没有 {species}/{method}/{guess}");
}

fn molecule(atoms: &[(&str, [f64; 3])], multiplicity: u32) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|(symbol, position)| Atom::new(*symbol, *position))
            .collect(),
        0,
        multiplicity,
    )
}

/// 紧收敛（`⟨S²⟩` 对密度比能量敏感：松阈值会留 ~3e-5 的偏差）。
fn tight() -> UhfOptions {
    let mut options = UhfOptions::default();
    options.energy_threshold = 1.0e-11;
    options.density_threshold = 1.0e-10;
    options.max_iter = 400;
    options
}

fn run(basis: &qwb_basis::InternalBasis, molecule: &Molecule, options: UhfOptions) -> UhfResult {
    let tensor = qwb_integrals::two_electron::two_electron(basis, molecule).expect("2e");
    uhf_scf(basis, molecule, &tensor, options).expect("UHF 应当收敛")
}

fn basis_for(molecule: &Molecule) -> qwb_basis::InternalBasis {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");
    qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), molecule)
        .expect("basis")
}

fn h2() -> Molecule {
    molecule(&[("H", [0.0, 0.0, -0.37]), ("H", [0.0, 0.0, 0.37])], 1)
}

fn o2() -> Molecule {
    molecule(&[("O", [0.0, 0.0, 0.0]), ("O", [0.0, 0.0, 1.208])], 3)
}

fn oh() -> Molecule {
    molecule(&[("O", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.970])], 2)
}

fn trace_of_product(density: &Matrix, overlap: &Matrix) -> f64 {
    let ps = qwb_linalg::matmul(density, overlap).expect("matmul");
    let mut trace = 0.0;
    for i in 0..ps.nrows() {
        trace += ps.get(i, i);
    }
    trace
}

// ───────── 1. 闭壳层退化：UHF 必须复现 RHF ─────────

#[test]
fn uhf_reproduces_rhf_for_a_closed_shell_molecule() {
    let (reference_energy, _, tolerance) = reference("H2", "UHF", "atom");
    let molecule = h2();
    let result = run(&basis_for(&molecule), &molecule, tight());

    assert!(
        (result.energy_hartree - reference_energy).abs() < tolerance,
        "H2/UHF/STO-3G = {:.12} Ha，参照 {:.12}（差 {:.2e}）",
        result.energy_hartree,
        reference_energy,
        (result.energy_hartree - reference_energy).abs()
    );
    assert_eq!(result.n_alpha, result.n_beta, "闭壳层 n_α == n_β");
    let mut worst = 0.0_f64;
    for i in 0..result.density_alpha.nrows() {
        for j in 0..result.density_alpha.ncols() {
            worst =
                worst.max((result.density_alpha.get(i, j) - result.density_beta.get(i, j)).abs());
        }
    }
    assert!(
        worst < 1e-10,
        "闭壳层的 α/β 密度必须相同（实测最大差 {worst:.2e}）"
    );
    assert!(
        result.s_squared.abs() < 1e-10,
        "闭壳层 ⟨S²⟩ 必须是 0，实测 {:.3e}",
        result.s_squared
    );
}

// ───────── 2/3. 开壳层：能量与自旋污染都要对上外部参照 ─────────

#[test]
fn o2_triplet_reaches_the_variational_minimum_with_the_atomic_guess() {
    let (reference_energy, reference_s2, tolerance) = reference("O2", "UHF", "atom");
    let molecule = o2();
    let result = run(&basis_for(&molecule), &molecule, tight());

    assert!(
        (result.energy_hartree - reference_energy).abs() < tolerance,
        "O2/UHF/STO-3G = {:.12} Ha，参照 {:.12}（差 {:.2e}）",
        result.energy_hartree,
        reference_energy,
        (result.energy_hartree - reference_energy).abs()
    );
    assert!(
        (result.s_squared - reference_s2).abs() < tolerance,
        "O2 的 ⟨S²⟩ = {:.12}，参照 {:.12}（差 {:.2e}）",
        result.s_squared,
        reference_s2,
        (result.s_squared - reference_s2).abs()
    );
    // M1 计划缺陷的**证据**：UHF 的 O2 不是纯自旋态（纯自旋要 ROHF）。
    assert!(
        result.s_squared > 2.0 + 1e-4,
        "UHF 的 O2 ⟨S²⟩ 应当略大于 2.0（自旋污染），实测 {:.6}",
        result.s_squared
    );
    // O2 共 16 个电子，三重态 → n_α = 9、n_β = 7（PySCF 同值）
    assert_eq!(result.n_alpha, 9);
    assert_eq!(result.n_beta, 7);
}

#[test]
fn o2_uhf_has_a_second_higher_solution_reachable_from_the_core_guess() {
    // 同一个体系、同一个代码、只换初猜：这是"解的选择"问题，不是 bug。
    // CSV 记着 PySCF 用 `init_guess='hcore'` 的结果，我们必须在同一初猜下**对上它**。
    let (reference_energy, reference_s2, tolerance) = reference("O2", "UHF", "hcore");
    let molecule = o2();
    let mut options = tight();
    options.initial_guess = UhfInitialGuess::Core;
    let result = run(&basis_for(&molecule), &molecule, options);

    assert!(
        (result.energy_hartree - reference_energy).abs() < tolerance,
        "O2（core guess）= {:.12} Ha，参照 {:.12}（差 {:.2e}）",
        result.energy_hartree,
        reference_energy,
        (result.energy_hartree - reference_energy).abs()
    );
    assert!(
        (result.s_squared - reference_s2).abs() < 1e-6,
        "O2（core guess）的 ⟨S²⟩ = {:.12}，参照 {:.12}",
        result.s_squared,
        reference_s2
    );

    // 两个解的差距是**真差距**（0.25 Ha 量级），不是数值噪声 —— 断言它存在，
    // 免得以后有人把"换初猜"当成不影响结果的实现细节。
    let (best, _, _) = reference("O2", "UHF", "atom");
    assert!(
        result.energy_hartree > best + 0.1,
        "core guess 的解应当明显高于变分极小（{:.6} vs {:.6}）",
        result.energy_hartree,
        best
    );
}

#[test]
fn oh_doublet_matches_the_external_reference_including_contamination() {
    let (reference_energy, reference_s2, tolerance) = reference("OH", "UHF", "atom");
    let molecule = oh();
    let result = run(&basis_for(&molecule), &molecule, tight());

    assert!(
        (result.energy_hartree - reference_energy).abs() < tolerance,
        "OH/UHF/STO-3G = {:.12} Ha，参照 {:.12}（差 {:.2e}）",
        result.energy_hartree,
        reference_energy,
        (result.energy_hartree - reference_energy).abs()
    );
    assert!(
        (result.s_squared - reference_s2).abs() < tolerance,
        "OH 的 ⟨S²⟩ = {:.12}，参照 {:.12}（差 {:.2e}）",
        result.s_squared,
        reference_s2,
        (result.s_squared - reference_s2).abs()
    );
    // 0.75 + 污染：纯自旋要等 ROHF（SC-01b）。
    assert!(
        result.spin_contamination > 1e-4,
        "OH 的 UHF 有自旋污染（实测 {:.3e}）",
        result.spin_contamination
    );
    assert_eq!(result.n_alpha, 5);
    assert_eq!(result.n_beta, 4);
}

// ───────── 4. 电子数自洽 ─────────

#[test]
fn electron_counts_are_consistent_with_the_densities() {
    for (label, molecule, expected_alpha, expected_beta) in [
        ("H2", h2(), 1usize, 1usize),
        ("O2", o2(), 9, 7),
        ("OH", oh(), 5, 4),
    ] {
        let basis = basis_for(&molecule);
        let result = run(&basis, &molecule, tight());

        // 全电子体系：核电荷 = 元素表里的 Z。
        let charges: Vec<f64> = molecule
            .atoms
            .iter()
            .map(|atom| {
                f64::from(
                    qwb_registry::elements::atomic_number(&atom.symbol)
                        .unwrap_or_else(|| panic!("未知元素 {}", atom.symbol)),
                )
            })
            .collect();
        let overlap =
            qwb_integrals::one_electron::one_electron_with_charges(&basis, &molecule, &charges)
                .expect("1e")
                .overlap;

        let n_alpha = trace_of_product(&result.density_alpha, &overlap);
        let n_beta = trace_of_product(&result.density_beta, &overlap);
        assert!(
            (n_alpha - expected_alpha as f64).abs() < 1e-8,
            "{label}: Tr[P_α S] = {n_alpha:.9}，期望 {expected_alpha}"
        );
        assert!(
            (n_beta - expected_beta as f64).abs() < 1e-8,
            "{label}: Tr[P_β S] = {n_beta:.9}，期望 {expected_beta}"
        );
        let two_s = (molecule.multiplicity - 1) as usize;
        assert_eq!(result.n_alpha - result.n_beta, two_s, "{label}: 2S");
    }
}

// ───────── 确定性 ─────────

#[test]
fn two_runs_are_bitwise_identical() {
    let molecule = oh();
    let basis = basis_for(&molecule);
    let first = run(&basis, &molecule, tight());
    let second = run(&basis, &molecule, tight());
    assert_eq!(
        first.energy_hartree.to_bits(),
        second.energy_hartree.to_bits()
    );
    assert_eq!(first.s_squared.to_bits(), second.s_squared.to_bits());
    assert_eq!(first.iterations, second.iterations);
}

// ───────── 输入校验：不自洽就报错，不猜 ─────────

#[test]
fn multiplicity_must_be_consistent_with_the_electron_count() {
    // 奇数电子配单重态（H 原子）→ n_α − n_β = 0 解不出整数
    let radical = molecule(&[("H", [0.0, 0.0, 0.0])], 1);
    let basis = basis_for(&radical);
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &radical).expect("2e");
    let error =
        uhf_scf(&basis, &radical, &tensor, UhfOptions::default()).expect_err("多重度矛盾必须报错");
    match error {
        EngineError::InvalidInput(message) => {
            assert!(message.contains("multiplicity"), "{message}");
            assert!(message.contains("electrons"), "{message}");
        }
        other => panic!("期望 InvalidInput，得到 {other:?}"),
    }
}
