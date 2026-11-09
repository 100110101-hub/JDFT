//! SV-03 第四半验收：CDS 的**解析梯度**。
//!
//! 判据五条，都不依赖上游（`get_cds_legacy` 的梯度用的是它自己的切换宽度，逐位不可比）：
//!
//! 1. **Richardson 中心差分**：解析梯度对"只算能量"那条路径（`smd_surface` + `smd_sts`）
//!    的数值导数 —— 这条抓的是面积导数、`STS` 链式（含 N 的三链）与单位换算；
//! 2. **平移不变性**：`Σ_A ∂E/∂R_A = 0`（能量只依赖相对坐标），这条抓符号与求和；
//! 3. `w <= 0` 必须报错 —— 二值遮蔽下的面积不可微，不许悄悄给 0；
//! 4. **协变坐标架也要能求导**（补上坐标架雅可比）：传 `Covariant` 必须成功，
//!    并且与**协变面自己的**能量差分一致（< 1e-5 Ha/Bohr）。协变面的完整判据表在
//!    `tests/sv02_covariant_gradient.rs`；
//! 5. **全局路径逐位不变**：坐标架项只进协变分支，所以 `Global` 的梯度位模式必须一动不动
//!    （`the_global_path_is_pinned_bit_for_bit`）。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_props::cavity::CavityFrame;
use qwb_props::smd_gradient::{smd_cds_energy_gradient, smd_cds_energy_gradient_with_frame};
use qwb_props::smd_sigma::{cds_energy_kcal_from_sts, smd_sts};
use qwb_props::smd_surface::smd_surface_with_frame;
use qwb_registry::Registry;
use qwb_registry::smd_atomic::SmdAtomic;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn smd() -> SmdAtomic {
    let registry = Registry::open(data_root()).expect("qwb-data");
    registry
        .get_smd_atomic("smd-atomic")
        .expect("smd-atomic")
        .clone()
}

/// 只算平滑面（`w > 0`）的能量 —— 有限差分的对照路径，**不经过**梯度代码。
///
/// 第 435 轮起 `smd_surface()` 的默认坐标架是协变，而 `smd_cds_energy_gradient` 这个入口
/// 保留的是**全局**坐标架（逐位不变）。跨路径比较必须在同一个坐标架里，否则数分的不是同一
/// 张面 —— 所以这里显式点名 `CavityFrame::Global`。协变面的对应版本见下面的
/// [`smooth_energy_covariant`] 与 `tests/sv02_covariant_gradient.rs`。
fn smooth_energy(data: &SmdAtomic, molecule: &Molecule, points: usize, width: f64) -> f64 {
    let surface = smd_surface_with_frame(data, molecule, points, width, CavityFrame::Global)
        .expect("surface");
    let sts = smd_sts(data, molecule, true, None).expect("sts");
    cds_energy_kcal_from_sts(&sts, 0.0, &surface.areas_angstrom2).expect("energy")
}

/// 同上，但用**默认（协变）**坐标架的面 —— 只为量出"解析梯度不是这张面的导数"差多少。
fn smooth_energy_covariant(
    data: &SmdAtomic,
    molecule: &Molecule,
    points: usize,
    width: f64,
) -> f64 {
    let surface = smd_surface_with_frame(data, molecule, points, width, CavityFrame::Covariant)
        .expect("surface");
    let sts = smd_sts(data, molecule, true, None).expect("sts");
    cds_energy_kcal_from_sts(&sts, 0.0, &surface.areas_angstrom2).expect("energy")
}

fn perturbed(molecule: &Molecule, atom: usize, axis: usize, delta: f64) -> Molecule {
    let mut atoms: Vec<Atom> = molecule.atoms.clone();
    atoms[atom].position_angstrom[axis] += delta;
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

fn systems() -> Vec<(&'static str, Molecule)> {
    vec![
        (
            "water",
            Molecule::new(
                vec![
                    Atom::new("O", [0.0, 0.0, 0.117_790]),
                    Atom::new("H", [0.0, 0.755_450, -0.471_160]),
                    Atom::new("H", [0.0, -0.755_450, -0.471_160]),
                ],
                0,
                1,
            ),
        ),
        (
            "methane",
            Molecule::new(
                vec![
                    Atom::new("C", [0.0, 0.0, 0.0]),
                    Atom::new("H", [0.6276, 0.6276, 0.6276]),
                    Atom::new("H", [-0.6276, -0.6276, 0.6276]),
                    Atom::new("H", [-0.6276, 0.6276, -0.6276]),
                    Atom::new("H", [0.6276, -0.6276, -0.6276]),
                ],
                0,
                1,
            ),
        ),
        (
            "methanol",
            Molecule::new(
                vec![
                    Atom::new("C", [0.0, 0.0, 0.0]),
                    Atom::new("O", [1.4, 0.0, 0.0]),
                    Atom::new("H", [-0.3636, 0.9382, 0.0]),
                    Atom::new("H", [-0.3636, -0.4691, 0.8126]),
                    Atom::new("H", [-0.3636, -0.4691, -0.8126]),
                    Atom::new("H", [1.76, 0.83, 0.0]),
                ],
                0,
                1,
            ),
        ),
        (
            "acetonitrile",
            Molecule::new(
                vec![
                    Atom::new("C", [0.0, 0.0, 0.0]),
                    Atom::new("C", [0.0, 0.0, 1.46]),
                    Atom::new("N", [0.0, 0.0, 2.62]),
                    Atom::new("H", [1.03, 0.0, -0.36]),
                    Atom::new("H", [-0.515, 0.89, -0.36]),
                    Atom::new("H", [-0.515, -0.89, -0.36]),
                ],
                0,
                1,
            ),
        ),
    ]
}

#[test]
fn the_analytic_gradient_matches_richardson_differences() {
    let data = smd();
    // 梯度必须用平滑面；`w = 0` 的二值面不可微（见模块文档）。
    let points = 2048;
    let step = 1.0e-4;
    for width in [0.10_f64, 0.30] {
        for (name, molecule) in systems() {
            let result =
                // 第 439 轮：默认入口已切到 Covariant（与能量侧一致），这条历史 Richardson 对照的
    // 数字与意图都属于 Global 路径，所以显式传 Global —— 共变路径的 FD 对照在 sv02 里。
    smd_cds_energy_gradient_with_frame(
        &data, &molecule, points, width, true, None, CavityFrame::Global,
    )
    .expect("grad");
            let mut worst = 0.0_f64;
            let mut largest = 0.0_f64;
            for atom in 0..molecule.n_atoms() {
                for axis in 0..3 {
                    let plus = smooth_energy(
                        &data,
                        &perturbed(&molecule, atom, axis, step),
                        points,
                        width,
                    );
                    let minus = smooth_energy(
                        &data,
                        &perturbed(&molecule, atom, axis, -step),
                        points,
                        width,
                    );
                    let numeric = (plus - minus) / (2.0 * step);
                    let analytic = result.gradient_kcal_per_angstrom[atom][axis];
                    worst = worst.max((numeric - analytic).abs());
                    largest = largest.max(analytic.abs());
                }
            }
            println!(
                "{name:14} w = {width:.2}: max |grad| = {largest:.6e}, \
                 max |analytic - numeric| = {worst:.3e} kcal/(mol A)"
            );
            assert!(
                largest > 1.0e-4,
                "{name}: the gradient must not be trivially zero"
            );
            assert!(
                worst < 1.0e-5 * largest.max(1.0),
                "{name}: analytic gradient disagrees with finite differences by {worst:.3e}"
            );
        }
    }
}

#[test]
fn the_gradient_is_translation_invariant() {
    let data = smd();
    let points = 2048;
    for (name, molecule) in systems() {
        let result = smd_cds_energy_gradient_with_frame(
            &data,
            &molecule,
            points,
            0.15,
            true,
            None,
            CavityFrame::Global,
        )
        .expect("grad");
        let mut total = [0.0_f64; 3];
        for vector in &result.gradient_kcal_per_angstrom {
            for axis in 0..3 {
                total[axis] += vector[axis];
            }
        }
        let norm = (total[0] * total[0] + total[1] * total[1] + total[2] * total[2]).sqrt();
        println!("{name:14} |sum of gradients| = {norm:.3e} kcal/(mol A)");
        assert!(
            norm < 1.0e-10,
            "{name}: the total gradient must vanish under translation, got {norm:.3e}"
        );
    }
}

#[test]
fn the_covariant_frame_is_now_differentiated_instead_of_refused() {
    // 第 435 轮起这条测试钉的是"协变坐标架必须显式报 `Unsupported`"：那时采样点
    // `p_k = c_i + R_i·F_i·l_k` 里 `F_i` 对核坐标的导数（坐标架雅可比）没实装，
    // 解析(Global) 与协变面**自己的**数值导数差 **1.488e-2 kcal/(mol·Å)**（本测试的历史实测，
    // 见下面 `worst_frames` 那一行）。
    //
    // 这一轮把那一项补上了（`cavity::atom_frame_jacobian` + `smd_gradient` 的通量收缩），
    // 于是断言反过来，钉三件事：
    // ① 传协变必须**成功** —— 缺口是补上的，不是绕开的（不许再回来）；
    // ② 协变的解析梯度与协变面自己的能量中心差分一致（< 1e-5 Ha/Bohr）；
    // ③ 两种坐标架的梯度确实不是同一个东西（历史差距仍 > 1e-3），所以"拿 Global 的导数当
    //    默认面的导数"一直是错的 —— 那正是要补这一项的理由。
    let data = smd();
    let (name, molecule) = systems().remove(0);
    let points = 2048;
    let width = 0.10;
    let step = 1.0e-5;
    /// kcal/(mol·Å) → Ha/Bohr。
    const KCAL_PER_ANGSTROM_TO_HARTREE_PER_BOHR: f64 = 0.529_177_210_903 / 627.509_474_063_6;

    let covariant = smd_cds_energy_gradient_with_frame(
        &data,
        &molecule,
        points,
        width,
        true,
        None,
        CavityFrame::Covariant,
    )
    .expect("the covariant surface must now have an analytic gradient");
    // 全局坐标架仍然照常可用（显式传 Global；默认入口现在是 Covariant）。
    let global = smd_cds_energy_gradient_with_frame(
        &data,
        &molecule,
        points,
        width,
        true,
        None,
        CavityFrame::Global,
    )
    .expect("global gradient");

    let mut worst_fd = 0.0_f64;
    let mut worst_frames = 0.0_f64;
    for atom in 0..molecule.n_atoms() {
        for axis in 0..3 {
            let plus = smooth_energy_covariant(
                &data,
                &perturbed(&molecule, atom, axis, step),
                points,
                width,
            );
            let minus = smooth_energy_covariant(
                &data,
                &perturbed(&molecule, atom, axis, -step),
                points,
                width,
            );
            let numeric = (plus - minus) / (2.0 * step);
            let analytic = covariant.gradient_kcal_per_angstrom[atom][axis];
            worst_fd = worst_fd.max((numeric - analytic).abs());
            worst_frames =
                worst_frames.max((analytic - global.gradient_kcal_per_angstrom[atom][axis]).abs());
        }
    }
    println!(
        "{name}: |analytic(Covariant) - numeric(Covariant)| = {worst_fd:.3e} kcal/(mol A) = \
         {:.3e} Ha/Bohr (criterion 1e-5)",
        worst_fd * KCAL_PER_ANGSTROM_TO_HARTREE_PER_BOHR
    );
    println!(
        "{name}: |analytic(Covariant) - analytic(Global)| = {worst_frames:.3e} kcal/(mol A) \
         (the historical 1.488e-2 that used to be the missing frame-Jacobian term)"
    );
    assert!(
        worst_fd * KCAL_PER_ANGSTROM_TO_HARTREE_PER_BOHR < 1.0e-5,
        "the covariant analytic gradient disagrees with finite differences of the covariant \
         energy by {:.3e} Ha/Bohr",
        worst_fd * KCAL_PER_ANGSTROM_TO_HARTREE_PER_BOHR
    );
    assert!(
        worst_frames > 1.0e-3,
        "if the two frames produced the same gradient the frame Jacobian would be unnecessary; \
         measured {worst_frames:.3e} kcal/(mol A)"
    );
}

#[test]
fn the_global_path_is_pinned_bit_for_bit() {
    // 坐标架项只加在 `CavityFrame::Covariant` 分支（`Global` 的 `F_i` 恒为单位架、导数为 0），
    // 所以全局路径必须**逐位**不变。这里把位模式钉住：若协变项不小心也流进全局分支
    // （多余的乘加、求和顺序变化），这几行会红。
    //
    // 这些位是**改动前后各跑一次**得到的同一组数（改动前用 `git show HEAD:…` 的旧实现跑）：
    // `0xbfa88561e79f84ba 0x3f875bdf80c0e0c5 0x3fc756c3c8209eb6` 等逐位相同。
    let data = smd();
    let (_, molecule) = systems().remove(0);
    let global = smd_cds_energy_gradient_with_frame(
        &data,
        &molecule,
        256,
        0.10,
        true,
        None,
        CavityFrame::Global,
    )
    .expect("global gradient");
    assert_eq!(global.energy_kcal.to_bits(), 0x3ff8_6ea2_263d_26c1);
    let expected: [[u64; 3]; 3] = [
        [
            0xbfa8_8561_e79f_84ba,
            0x3f87_5bdf_80c0_e0c5,
            0x3fc7_56c3_c820_9eb6,
        ],
        [
            0xbfa5_e901_67a2_e771,
            0x3fd8_c2ee_ff07_6028,
            0xbfba_ff11_dcee_fc9f,
        ],
        [
            0x3fb7_3731_a7a1_3614,
            0xbfd9_7dcd_fb0d_672d,
            0xbfb3_ae75_b352_40ce,
        ],
    ];
    for (atom, atom_expected) in expected.iter().enumerate() {
        for (axis, value) in atom_expected.iter().enumerate() {
            let actual = global.gradient_kcal_per_angstrom[atom][axis].to_bits();
            println!("water Global atom {atom} axis {axis}: {actual:#018x}");
            assert_eq!(
                actual, *value,
                "the global-frame gradient moved at atom {atom} axis {axis}"
            );
        }
    }
}

#[test]
fn the_binary_limit_is_refused_for_gradients() {
    let data = smd();
    let (_, molecule) = systems().remove(0);
    for width in [0.0_f64, -0.1] {
        let error = smd_cds_energy_gradient(&data, &molecule, 1024, width, true, None)
            .expect_err("a non-differentiable surface must be refused");
        println!("w = {width}: {error}");
    }
}

#[test]
fn the_gradient_depends_on_the_switch_width_as_documented() {
    // 梯度是**模型**导数：切换宽度是模型选择，所以这里把三种宽度的梯度量级记下来，
    // 而不是假装它们应该相同（能量对宽度的依赖在 `sv03_smd_surface.rs` 里另有记录）。
    let data = smd();
    let (_, molecule) = systems().remove(0);
    for width in [0.05_f64, 0.10, 0.30] {
        let result =
            smd_cds_energy_gradient(&data, &molecule, 2048, width, true, None).expect("grad");
        let largest = result
            .gradient_kcal_per_angstrom
            .iter()
            .map(|vector| {
                (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
            })
            .fold(0.0_f64, f64::max);
        println!(
            "w = {width:.2}: E = {:.6e} kcal/mol, max |dE/dR| = {largest:.6e}",
            result.energy_kcal
        );
        assert!(largest > 0.0);
    }
}
