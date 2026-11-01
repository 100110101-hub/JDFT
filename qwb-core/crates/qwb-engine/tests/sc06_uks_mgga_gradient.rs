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
/// **数值对照为什么用 M06-L**：它没有精确交换，梯度对 SCF 收敛质量最不敏感，能在
/// `grid-normal` 上把「解析 vs 数值」压到 2.4e-7；M06-2X 同一条路径的**可用性**由下面的
/// `open_shell_m06_2x_gradient_is_available` 单独守（含 1e-6 的驻点判据与对称性）。
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

/// SC-06 判据的**字面**那一半：**O2 三重态 + M06-2X**（α_hf = 0.54），要求引擎**给出梯度**。
///
/// 用 O2 而不是 OH：它就是 `sc06_open_shell_dispersion.rs` 里「M06-2X + D3 跑通」那个体系，
/// 于是判据的两半落在**同一个体系**上（D3 通路 + 梯度）。
///
/// 这里不重复「解析 vs 数值」的对照 —— 那一条在 M06-L 上做（两者共用同一条
/// `uks_analytic_gradient` 路径，τ 与 hybrid 的项都在里面）。本测试守两件事：
///
/// 1. 引擎**不再**以「密度不是驻点」为由拒绝 M06-2X 的梯度。驻点判据（`‖FPS−SPF‖ < 1e-6`）
///    由 `analytic_energy_and_gradient` 内部检查 —— 它返回 `Ok` 就说明过了；
/// 2. 梯度的**对称性**：O2 沿 z 轴，所以只有两个 z 分量非零，其余四个（x/y）必须 < 1e-9。
///
/// 这条曾经红过：`brillouin_residual` 建 Fock 时漏了 `−α_hf·K`，任何 hybrid 的残差永远
/// 不为零；补上 K 之后 OH/M06-2X 的密度阈值还需要收到 1e-10 才把残差从 5.9e-6 压到 1.8e-10
/// （见 `uks_scf_with_convergence_aids` 的阈值阶梯与 `docs/choices.md` 一百四十九）。
#[test]
fn open_shell_m06_2x_gradient_is_available() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = dioxygen();
    let mut options = OptOptions::new("sto-3g", "m06-2x", "grid-normal");
    options.scf_type = Some(ScfType::Uks);
    options.scf_max_iter = 300;
    let (energy, gradient) = analytic_energy_and_gradient(&registry, &molecule, &options)
        .expect("uks + m06-2x 的开壳层解析梯度必须可用（SC-06 判据）");
    println!("O2/M06-2X/STO-3G: E = {energy:.10} Ha");
    for (atom, row) in gradient.iter().enumerate() {
        println!(
            "  gradient[{atom}] = [{:+.6e}, {:+.6e}, {:+.6e}]",
            row[0], row[1], row[2]
        );
    }
    assert!(energy.is_finite());
    let nonzero: Vec<(usize, usize, f64)> = gradient
        .iter()
        .enumerate()
        .flat_map(|(atom, row)| {
            (0..3)
                .filter(|axis| row[*axis].abs() > 1.0e-9)
                .map(move |axis| (atom, axis, row[axis]))
        })
        .collect();
    println!("gradient 的非零分量: {nonzero:?}");
    assert_eq!(
        nonzero.len(),
        2,
        "O2 沿 z 轴：解析梯度应当只有两个 z 分量（实测 {nonzero:?}）"
    );
    assert!(
        nonzero.iter().all(|(_, axis, _)| *axis == 2),
        "两个非零分量都必须落在 z 轴: {nonzero:?}"
    );
    // 再对一遍**数值**梯度：M06-2X 与 M06-L 共用同一条装配路径，但 hybrid 的 `W` 里多了
    // `−α_hf·K`，所以必须单独量一次（不能靠「同一条路径」推断）。
    //
    // 只对 **H 的 z 坐标**差分（2 次 SCF）而不是整条 `numerical_gradient`（12 次）：x/y 的
    // 解析分量已经是 <1e-9 的零（上面按对称性断言过），它们的数值版本只是网格舍入噪声；
    // z 上两个分量又必须等大反向（平动不变），所以量一个就够 —— 网小但够密，而且这一步
    // 从 147 s 降到 ~25 s（门禁的测试段要留在 8 min 以内）。
    let delta_bohr = 1.0e-3;
    let plus = shifted_energy(&registry, delta_bohr);
    let minus = shifted_energy(&registry, -delta_bohr);
    let numeric_hz = (plus - minus) / (2.0 * delta_bohr);
    let analytic_hz = gradient[1][2];
    let worst = (analytic_hz - numeric_hz).abs();
    println!("  O2 z: analytic {analytic_hz:+.9e} numeric {numeric_hz:+.9e} Δ {worst:.2e}");
    println!("O2/M06-2X: worst |analytic − numeric| (second O, z) = {worst:.3e} Ha/Bohr");
    assert!(
        worst < 1.0e-4,
        "M06-2X 的解析梯度与数值梯度差 {worst:.3e} Ha/Bohr（判据 1e-4）"
    );
}

/// 把第二个 O 沿 z 平移 `z_shift` Bohr 后，同一套 M06-2X/STO-3G/`grid-normal` 的 UKS 总能量。
///
/// 这是差分用的「单点能量」，走**和引擎相同**的 `uks_scf`（默认阈值 + DIIS），不经过驻点
/// 守卫 —— 差分只需要能量。
fn shifted_energy(registry: &Registry, z_shift: f64) -> f64 {
    // `Molecule` 的坐标是 **Å**，而差分步长给的是 **Bohr** —— 漏掉这一步会差 1.8897 倍
    // （实测：不换算时差分给 0.1518、解析 0.0803，差 7.1e-2，正好是这个因子）。
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let shift_angstrom = z_shift * ANGSTROM_PER_BOHR;
    let moved = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.208 + shift_angstrom]),
        ],
        0,
        3,
    );
    let basis = qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &moved)
        .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &moved).expect("2e");
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-normal").unwrap(), &moved).expect("grid");
    let (terms, hf_exchange) =
        qwb_engine::terms_and_hf_exchange(registry, "m06-2x").expect("terms");
    let mut options = qwb_scf::uks::UksOptions::default();
    options.hf_exchange = hf_exchange;
    options.max_iter = 400;
    qwb_scf::uks::uks_scf(&basis, &moved, &tensor, &grid, &terms, options)
        .expect("shifted SCF")
        .energy_hartree
}
