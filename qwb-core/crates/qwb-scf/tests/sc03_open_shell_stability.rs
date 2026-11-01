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
        options.level_shift_decay = 0.0;
        options.damping = damping;
        options.initial_guess_scale = scale;
        options
    };
    // 退火配方：λ_k = λ_0·decay^k（SC-02 救活 OH 的机制，见 choices 七十六）。
    let annealed = |shift: f64, decay: f64, scale: f64| {
        let mut options = base(shift, 0.0, scale);
        options.level_shift_decay = decay;
        options.max_iter = 400;
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
        ("I 退火 0.5/0.90", annealed(0.5, 0.90, 1.0)),
        ("J 退火 2.0/0.95", annealed(2.0, 0.95, 1.0)),
        ("K 退火 5.0/0.97", annealed(5.0, 0.97, 1.0)),
        ("L 退火 2.0/0.90+破对称", annealed(2.0, 0.90, 1.5)),
        // M（第 380 轮）：NO 的病不是振荡而是**很长的下坡** —— 前 200 步 E 还在
        // −127.52999 附近缓慢下降，908 步才到真正的 UHF 驻点 −127.5303995618。
        // 十二档配方全部卡在 200/400 步的预算上，所以缺的是**预算**不是配方。
        // 独立的 PySCF UHF（damp 0.7 收敛到 192 步，Newton 复核）给 −127.5303995296、
        // ⟨S²⟩ = 0.9650 —— 两者差 3.2e-8 Ha（本仓库 STO-3G 指数比 PySCF 内建表多几位）。
        ("M 大预算（1200 步）", {
            let mut o = base(0.0, 0.0, 1.0);
            o.max_iter = 1200;
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
                    if name == "NO" {
                        // PySCF 的 UHF（damp 0.7，192 步；`newton()` 复核）＝ −127.5303995296。
                        assert!(
                            (result.energy_hartree - (-127.530_399_529_6)).abs() < 1.0e-6,
                            "NO 的 UHF 能量 {:.10} 与 PySCF 的 −127.5303995296 差得太多",
                            result.energy_hartree
                        );
                    }
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
                println!(
                    "{name}: 全部 {} 档配方失败 —— 诊断：{relaxed_outcome}",
                    recipes().len()
                );
                failed.push((name, relaxed_outcome));
            }
        }
    }
    println!("\n=== SC-03 汇总 ===");
    for (name, label, iterations, seconds) in &converged {
        println!("{name:5} {label:24} {iterations:3} 步 {seconds:6.3} s  收敛");
    }
    for (name, diagnosis) in &failed {
        println!(
            "{name:5} 未收敛（{} 档配方）  诊断：{diagnosis}",
            recipes().len()
        );
    }
    // 钉住实测状态：**5/5 收敛**（第 380 轮：NO 的卡点是预算不是配方 —— 十二档配方全部卡在
    // 200/400 步，而它从 −127.52999 一路缓慢下坡，908 步才到 UHF 驻点）。
    assert_eq!(
        converged.len(),
        5,
        "实测应为 5 个体系收敛（SC-03 的判据：5/5 到 1e-8）"
    );
    assert!(failed.is_empty(), "不应再有失败的体系：{failed:?}");

    // NO 的**独立核对**：PySCF 2.14.0 的 UHF（damp 0.7 收敛到 192 步，`newton()` 复核）
    // 给 −127.5303995296 Ha、⟨S²⟩ = 0.9650。本仓库的 STO-3G 指数比 PySCF 内建表多几位，
    // 实测差 **3.2e-8 Ha**。这条断言钉住"908 步收敛到的是**同一个** UHF 驻点"，
    // 而不是一个更低的伪态 —— 顺带说明：早先"放宽到 1e-6、103 步"的那个解是
    // **−127.5288656**，比真驻点高 1.5 mHa，是个伪态。
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
