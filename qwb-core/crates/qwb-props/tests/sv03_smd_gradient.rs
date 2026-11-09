//! SV-03 第四半验收：CDS 的**解析梯度**。
//!
//! 判据三条，都不依赖上游（`get_cds_legacy` 的梯度用的是它自己的切换宽度，逐位不可比）：
//!
//! 1. **Richardson 中心差分**：解析梯度对"只算能量"那条路径（`smd_surface` + `smd_sts`）
//!    的数值导数 —— 这条抓的是面积导数、`STS` 链式（含 N 的三链）与单位换算；
//! 2. **平移不变性**：`Σ_A ∂E/∂R_A = 0`（能量只依赖相对坐标），这条抓符号与求和；
//! 3. `w <= 0` 必须报错 —— 二值遮蔽下的面积不可微，不许悄悄给 0。
use std::path::PathBuf;

use qwb_engine_api::{Atom, EngineError, Molecule};
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
/// 第 435 轮起 `smd_surface()` 的默认坐标架是协变，而这里的解析梯度只对**全局**坐标架
/// 成立（坐标架雅可比未实装，见 `qwb_props::smd_gradient` 的模块文档）。所以对照路径必须
/// 显式点名 `CavityFrame::Global`：跨路径比较必须在同一个坐标架里，否则数分的不是同一张面。
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
                smd_cds_energy_gradient(&data, &molecule, points, width, true, None).expect("grad");
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
        let result =
            smd_cds_energy_gradient(&data, &molecule, points, 0.15, true, None).expect("grad");
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
fn the_covariant_frame_is_refused_by_the_analytic_gradient() {
    // 解析梯度只对 `CavityFrame::Global` 的面求导：协变坐标架依赖核坐标，采样点
    // `p_k = c_i + R_i·F_i·l_k` 对核坐标的导数多一项 `R_i·(∂F_i/∂c_m)·l_k`（坐标架雅可比），
    // 那一项**没实装**。这条测试钉两件事：
    //
    // ① 传协变坐标架必须**显式报错**（`Unsupported`），不许静默把另一张面的导数当本面返回；
    // ② 把"若静默返回"的差距量出来：对默认（协变）面的能量做 Richardson 差分会与解析梯度
    //    差千分之几 kcal/(mol·Å) 量级 —— 那是真差异，不是舍入，所以"报错"是唯一诚实的选项。
    let data = smd();
    let (name, molecule) = systems().remove(0);
    let points = 2048;
    let width = 0.10;
    let step = 1.0e-4;

    let refused = smd_cds_energy_gradient_with_frame(
        &data,
        &molecule,
        points,
        width,
        true,
        None,
        CavityFrame::Covariant,
    )
    .expect_err("the covariant frame has no frame Jacobian, so the gradient must be refused");
    println!("{name} [Covariant] -> {refused}");
    assert!(
        matches!(refused, EngineError::Unsupported(_)),
        "expected Unsupported, got {refused:?}"
    );
    // 全局坐标架仍然照常可用（默认入口走的就是它）。
    let global = smd_cds_energy_gradient(&data, &molecule, points, width, true, None)
        .expect("global gradient");

    let mut worst = 0.0_f64;
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
            let analytic = global.gradient_kcal_per_angstrom[atom][axis];
            worst = worst.max((numeric - analytic).abs());
        }
    }
    println!(
        "{name}: max |analytic(Global) - numeric(Covariant)| = {worst:.3e} kcal/(mol A) \
         (that is the missing frame-Jacobian term; the test above pins the analytic gradient to \
         the same frame instead of hiding this behind a loose tolerance)"
    );
    assert!(
        worst > 1.0e-3,
        "if the covariant frame moved the gradient by less than 1e-3 kcal/(mol A) the refusal \
         would be over-cautious; measured {worst:.3e} - re-derive the frame Jacobian requirement"
    );
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
