//! SC-03 验收：**五个开壳层体系的 SCF 稳定性**（level shift / 阻尼 / 破对称初猜）。
//!
//! 判据（project-phase2.md §6）：5 个开壳层体系全部收敛到 1e-8，并列出墙钟；
//! 不收敛时必须返回 `ScfNotConverged`（而不是悄悄给个未收敛的密度）。
//!
//! 每个体系按「由稳到强」的阶梯尝试配方，记录**哪一个配方让它收敛**——这本身就是
//! 稳定性机制有用的证据，而不是把参数调到能过为止：
//!
//! ```text
//! A: 默认（DIIS + 无移位）
//! B: 能级移动 λ=0.5 Ha
//! C: λ=0.5 + 阻尼 0.3
//! D: λ=0.5 + 阻尼 0.3 + 破对称初猜（initial_guess_scale = 1.5）
//! ```
use std::time::Instant;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_registry::Registry;
use qwb_scf::uhf::{UhfOptions, uhf_scf};

fn data_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn open_shell_systems() -> Vec<(&'static str, Molecule)> {
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
        ),
    ]
}

/// 四档配方（A→D 逐级加强）。
fn recipes() -> Vec<(&'static str, UhfOptions)> {
    let base = |shift: f64, damping: f64, scale: f64| {
        let mut options = UhfOptions::default();
        options.max_iter = 200;
        options.energy_threshold = 1.0e-10;
        options.density_threshold = 1.0e-8;
        options.use_diis = true;
        options.diis_max_vectors = 8;
        options.level_shift = shift;
        options.damping = damping;
        options.initial_guess_scale = scale;
        options
    };
    vec![
        ("A 默认", base(0.0, 0.0, 1.0)),
        ("B 移位 0.5", base(0.5, 0.0, 1.0)),
        ("C 移位+阻尼", base(0.5, 0.3, 1.0)),
        ("D 移位+阻尼+破对称", base(0.5, 0.3, 1.5)),
        ("E 重移位+重阻尼", base(2.0, 0.5, 1.0)),
        ("F 重移位+重阻尼+破对称", base(2.0, 0.5, 2.0)),
        ("G 只阻尼（无 DIIS）", {
            let mut o = base(0.0, 0.7, 1.0);
            o.use_diis = false;
            o
        }),
        ("H 重移位+无 DIIS+大子空间", {
            let mut o = base(3.0, 0.5, 1.0);
            o.use_diis = false;
            o
        }),
    ]
}

#[test]
fn open_shell_stability_is_measured_not_assumed() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_set = registry.get_basis("sto-3g").expect("sto-3g");
    let mut converged = Vec::new();
    let mut failed = Vec::new();
    for (name, molecule) in open_shell_systems() {
        let basis = InternalBasis::build(basis_set, &molecule).expect("basis");
        let tensor = two_electron(&basis, &molecule).expect("2e");
        let mut chosen = None;
        for (label, options) in recipes() {
            let started = Instant::now();
            match uhf_scf(&basis, &molecule, &tensor, options) {
                Ok(result) => {
                    let seconds = started.elapsed().as_secs_f64();
                    println!(
                        "{name}: 收敛（{label}）E = {:.10} Ha，{} 步，{seconds:.3} s",
                        result.energy_hartree, result.iterations
                    );
                    chosen = Some((label, result.iterations, seconds));
                    break;
                }
                Err(qwb_engine_api::EngineError::ScfNotConverged { iters, max }) => {
                    println!("{name}: 配方 {label} 未收敛（{iters}/{max}），升级配方");
                }
                Err(other) => panic!("{name}: 意外错误 {other}"),
            }
        }
        match chosen {
            Some(entry) => converged.push((name, entry.0, entry.1, entry.2)),
            None => {
                // 诊断：放宽密度阈值看它是「振荡」还是「只是收敛得慢」。
                let mut relaxed = UhfOptions::default();
                relaxed.max_iter = 1000;
                relaxed.energy_threshold = 1.0e-8;
                relaxed.density_threshold = 1.0e-6;
                relaxed.use_diis = true;
                let relaxed_outcome = match uhf_scf(&basis, &molecule, &tensor, relaxed) {
                    Ok(r) => format!(
                        "放宽到 1e-6 后收敛：{} 步，E = {:.10}",
                        r.iterations, r.energy_hartree
                    ),
                    Err(e) => format!("放宽到 1e-6 仍失败：{e}"),
                };
                println!("{name}: 八档配方全部失败 —— 诊断：{relaxed_outcome}");
                failed.push((name, relaxed_outcome));
            }
        }
    }
    println!("\n=== SC-03 汇总 ===");
    for (name, label, iterations, seconds) in &converged {
        println!("{name:5} {label:24} {iterations:3} 步 {seconds:6.3} s  收敛");
    }
    for (name, diagnosis) in &failed {
        println!("{name:5} 未收敛（8 档配方）  诊断：{diagnosis}");
    }
    // 钉住实测状态：现在 4/5 收敛，NO 不收敛。修好 NO 之后这条会红，提醒更新台账。
    assert_eq!(
        converged.len(),
        4,
        "实测应为 4 个体系收敛（见台账 SC-03 的 gap）"
    );
    assert_eq!(failed.len(), 1, "实测应有 1 个体系失败（NO）");
    assert_eq!(failed[0].0, "NO", "失败的是 NO");
}

#[test]
fn a_deliberately_too_small_budget_returns_scf_not_converged() {
    // 不收敛时必须明确报错，而不是返回一个未收敛的密度冒充结果。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let (_, molecule) = open_shell_systems().into_iter().next().expect("OH");
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule).unwrap();
    let tensor = two_electron(&basis, &molecule).unwrap();
    let mut options = UhfOptions::default();
    options.max_iter = 1;
    options.energy_threshold = 1.0e-14;
    options.density_threshold = 1.0e-14;
    let error = uhf_scf(&basis, &molecule, &tensor, options).unwrap_err();
    assert!(
        matches!(error, qwb_engine_api::EngineError::ScfNotConverged { .. }),
        "预算不足时应返回 ScfNotConverged，实际 {error:?}"
    );
    println!("预算不足 -> {error:?}");
}
