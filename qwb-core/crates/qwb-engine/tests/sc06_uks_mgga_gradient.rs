//! SC-06 判据 ① 的后半条：**开壳层 UKS × meta-GGA 的解析梯度**。
//!
//! 前半条（能量对 PySCF）在 `qwb-scf/tests/sc06_uks_mgga.rs`：O2/STO-3G 的 M06-L 差 3.3e-5 Ha、M06-L 差 4.5e-5 Ha。
//! 这一条管「给出梯度」：`opt::analytic_energy_and_gradient`（UKS 分支 —— `v_xc` 按自旋进
//! Pulay、XC 的几何半走固定自旋密度的中心差分，**含 τ**）必须与整条 SCF 的数值梯度一致。
//!
//! 为什么这条能判 τ：m06-l 的 ε 依赖 `(ρ_σ, σ_σσ, τ_σ)`，而这份解析梯度的两条腿分别是
//! ① `∂E/∂P^σ`（走 `v_xc`，τ 项在 `½(∂e/∂τ_σ)∇φ∇φ` 里）② 固定密度下 `∂E_xc/∂R`
//! （网格/AO 随核移动，含 `∇φ` → τ）。τ 的任何一处漏项都会让两者对不上 1e-4 Ha/Bohr。
//!
//! 判据用 **1e-4 Ha/Bohr**（不是 UHF 那套 1e-6）：解析梯度的 XC 几何半本身是中心差分
//! （步长 1e-3 Bohr），它的截断误差就在 1e-5 量级；这一条判的是「有没有漏项」，不是
//! 「差分精度」。实测值打印在测试里。

use std::path::PathBuf;

use qwb_engine::gradient::{GradientOptions, numerical_gradient};
use qwb_engine::opt::{OptOptions, analytic_energy_and_gradient};
use qwb_engine_api::{Atom, Molecule, ScfType};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// O2 三重态，几何与 `qwb-scf/tests/reference/uks_mgga.csv` 完全一致（1.208 Å）。
///
/// **为什么是 M06-L 而不是 M06-L**：M06-L 的 UKS 在 O2 与 OH 上都收敛不到 Brillouin 点
/// （修掉 `brillouin_residual` 里漏掉的 `−α_hf·K` 之后仍有 `‖FPS−SPF‖ = 1.28e-2`），
/// 引擎因此**正确地**拒绝它的梯度；那是 UKS hybrid 的 SCF 收敛质量问题（SC-02 的剩余一半），
/// 不是梯度装配的缺口 —— 两者共用同一条 `uks_analytic_gradient` 路径。
fn dioxygen() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.208]),
        ],
        0,
        3,
    )
}

#[test]
fn open_shell_uks_m06l_analytic_gradient_matches_numerics() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = dioxygen();

    let mut options = OptOptions::new("sto-3g", "m06-l", "grid-normal");
    options.scf_type = Some(ScfType::Uks);
    options.scf_max_iter = 300;
    let (energy, analytic) = analytic_energy_and_gradient(&registry, &molecule, &options)
        .expect("uks + m06-l 的解析梯度应当可用（SC-06 的 τ 接线）");
    println!("O2/M06-L/STO-3G: E = {energy:.10} Ha");
    for (atom, row) in analytic.iter().enumerate() {
        println!(
            "  analytic[{atom}] = [{:+.6e}, {:+.6e}, {:+.6e}]",
            row[0], row[1], row[2]
        );
    }

    let mut numeric_options = GradientOptions::new("sto-3g", "m06-l", "grid-normal");
    numeric_options.delta_bohr = 1.0e-3;
    numeric_options.scf_type = Some(ScfType::Uks);
    let numeric =
        numerical_gradient(&registry, &molecule, &numeric_options).expect("整条 SCF 的数值梯度");

    let mut worst = 0.0_f64;
    for (atom, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        for axis in 0..3 {
            let delta = (a[axis] - n[axis]).abs();
            worst = worst.max(delta);
            println!(
                "  atom {atom} axis {axis}: analytic {:+.6e} numeric {:+.6e} Δ {delta:.2e}",
                a[axis], n[axis]
            );
        }
    }
    println!("worst |analytic − numeric| = {worst:.3e} Ha/Bohr");
    // 两个 O 在 z 轴上：梯度的非零分量应当只有两个 O 的 z 方向（对称性抹掉其余 10 个）。
    let nonzero: Vec<(usize, usize, f64)> = analytic
        .iter()
        .enumerate()
        .flat_map(|(atom, row)| {
            (0..3)
                .filter(|axis| row[*axis].abs() > 1.0e-9)
                .map(move |axis| (atom, axis, row[axis]))
        })
        .collect();
    println!("analytic 的非零分量: {nonzero:?}");
    assert_eq!(nonzero.len(), 2, "O2 沿 z 轴：解析梯度应当只有两个 z 分量");
    assert!(
        worst < 1.0e-4,
        "解析梯度与数值梯度差 {worst:.3e} Ha/Bohr（判据 1e-4）—— τ 通道有漏项"
    );
}
