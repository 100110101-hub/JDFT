//! SV-02 的前置：**协变坐标架（默认）的 CDS 解析梯度**。
//!
//! 第 435 轮把默认坐标架翻成转动协变的 [`CavityFrame::Covariant`] 之后，`crate::smd_gradient`
//! 的解析梯度只对 [`CavityFrame::Global`] 成立（采样点 `p_k = c_i + R_i·F_i·l_k` 里 `F_i`
//! 依赖构型，那一项坐标架雅可比没实装），传 `Covariant` 会显式报 `Unsupported`；把
//! 解析(Global) 与协变面的数值导数相比，实测差 **1.488e-2 kcal/(mol·Å)**（水、`w = 0.10`、
//! 2048 点，见 `sv03_smd_gradient.rs`）。那个差就是缺项。
//!
//! 这一轮把缺项补上：`∂F_i/∂c_m` 用**解析**雅可比（[`atom_frame_jacobian`]，每个原子最多
//! 三个非零源），`∂E/∂F_i` 的收缩也在解析式里做完（通量矩阵，见
//! [`qwb_props::smd_gradient`] 的 `surface_areas_and_gradients`）。本文件逐条量它：
//!
//! 1. **坐标架雅可比本身**：`atom_frame_jacobian` 对 `atom_frame` 的中心差分，逐元素
//!    （含"雅可比为零的源原子"必须真的是零）；
//! 2. **总梯度**：解析 vs **独立**的能量中心差分（`smd_surface_with_frame` + `smd_sts`，
//!    不经过梯度代码），水 + 甲醇 × `w = 0.10/0.15` × 2048/4096 点 × 两个步长；
//! 3. **平移不变性**：`Σ_A ∂E/∂R_A = 0`；
//! 4. **分支边界**（`atom_frame` 的离散选择）：把"临界构型上导数不存在"这件事量出来 ——
//!    不是假设、也不放宽判据。
//!
//! # 为什么判据表用的是"错开对称性"的几何
//!
//! `atom_frame` 的 `e1`/`e2` 来自**离散**选择（最近邻、第一根不共线方向，并列时按原子序号
//! 取小者）。在"两个候选距离严格相等"的构型上（对称的水里 `O` 的两个 `H`、手搭甲醇里
//! `C` 的两个 `H`），坐标架在 `±h` 两侧落在不同分支：**能量的单侧极限不同**，中心差分
//! 量的就不是同一个面。第 4 条测试把这件事量到 `4.5e-3 kcal/mol` 量级（甲醇），所以
//! 判据表（第 2 条）在**错开对称性**的几何上量 —— 那里 `atom_frame` 光滑，中心差分有意义。
//! 这不是放宽判据：恰恰相反，先把"能在哪量"钉住，再在那些点上要求 `< 1e-5 Ha/Bohr`。
use std::path::PathBuf;

use qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
use qwb_engine_api::{Atom, Molecule};
use qwb_props::cavity::{CavityFrame, atom_frame, atom_frame_jacobian};
use qwb_props::smd_gradient::smd_cds_energy_gradient_with_frame;
use qwb_props::smd_sigma::{cds_energy_kcal_from_sts, smd_sts};
use qwb_props::smd_surface::smd_surface_with_frame;
use qwb_registry::Registry;
use qwb_registry::smd_atomic::SmdAtomic;

/// 1 Ha = 627.5094740636 kcal/mol（CODATA 2018，与 `qwb-engine` 用的同一个常数）。
const HARTREE_PER_KCAL: f64 = 627.509_474_063_6;

/// kcal/(mol·Å) → Ha/Bohr（SV-02 判据的单位）。
fn to_hartree_per_bohr(kcal_per_angstrom: f64) -> f64 {
    kcal_per_angstrom / HARTREE_PER_KCAL / BOHR_PER_ANGSTROM
}

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

/// 协变面（默认坐标架）的 CDS 能量 —— 有限差分的对照路径，**不经过**梯度代码。
fn covariant_energy(data: &SmdAtomic, molecule: &Molecule, points: usize, width: f64) -> f64 {
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

/// 把原子按一个固定的伪随机方向错开 `delta` Å，**打破一切距离并列**。
///
/// 目的只有一个：让 `atom_frame` 的离散分支在 `±h` 两侧落在**同一支**（否则中心差分量的不是
/// 同一个面，见文件头）。偏移是确定性的（`sin/cos` 的固定倍数），测试可复现。
fn displaced(molecule: Molecule, delta: f64) -> Molecule {
    let atoms = molecule
        .atoms
        .iter()
        .enumerate()
        .map(|(index, atom)| {
            let mut shifted = atom.clone();
            let k = (index + 1) as f64;
            shifted.position_angstrom[0] += delta * (1.7 * k).sin();
            shifted.position_angstrom[1] += delta * (2.3 * k).cos();
            shifted.position_angstrom[2] += delta * (3.1 * k).sin();
            shifted
        })
        .collect();
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    )
}

fn methanol() -> Molecule {
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
    )
}

fn methane() -> Molecule {
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
    )
}

fn acetonitrile() -> Molecule {
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
    )
}

/// 判据表用的几何：错开 0.02 Å（见 [`displaced`] 与文件头）。
fn generic_systems() -> Vec<(&'static str, Molecule)> {
    vec![
        ("water", displaced(water(), 0.02)),
        ("methanol", displaced(methanol(), 0.02)),
    ]
}

/// **分支穿越检测**：`F(c ± h)` 对线性预测 `F(c) ± h·(∂F/∂c)` 的偏差。
///
/// 光滑情形这项是 `O(h²)`（实测 `h = 1e-4` 时 1.7e-8）；若某个扰动跨过了坐标架的分支
/// （最近邻或 `e2` 伙伴换人），偏差是 `O(1)`（`e1` 可以整根换向）。所以这是"这一步差分
/// 落在同一支里"的可判定守卫 —— 判据表先过它，再比解析梯度与差分。
fn branch_guard(molecule: &Molecule, step: f64) -> f64 {
    let mut worst: f64 = 0.0;
    for index in 0..molecule.n_atoms() {
        let reference = atom_frame(molecule, index);
        let terms = atom_frame_jacobian(molecule, index);
        for source in 0..molecule.n_atoms() {
            for axis in 0..3 {
                for sign in [-1.0_f64, 1.0] {
                    let image = atom_frame(&perturbed(molecule, source, axis, sign * step), index);
                    for row in 0..3 {
                        for column in 0..3 {
                            let jacobian = terms
                                .iter()
                                .find(|(atom, _)| *atom == source)
                                .map_or(0.0, |(_, derivative)| derivative[axis][row][column]);
                            let predicted = reference[row][column] + sign * step * jacobian;
                            worst = worst.max((image[row][column] - predicted).abs());
                        }
                    }
                }
            }
        }
    }
    worst
}

/// 坐标架雅可比对 `atom_frame` 中心差分的最大逐元素偏差。
fn frame_jacobian_error(molecule: &Molecule, step: f64) -> (f64, f64) {
    let mut worst: f64 = 0.0;
    let mut largest: f64 = 0.0;
    for index in 0..molecule.n_atoms() {
        let terms = atom_frame_jacobian(molecule, index);
        for source in 0..molecule.n_atoms() {
            for axis in 0..3 {
                let plus = atom_frame(&perturbed(molecule, source, axis, step), index);
                let minus = atom_frame(&perturbed(molecule, source, axis, -step), index);
                for row in 0..3 {
                    for column in 0..3 {
                        let numeric = (plus[row][column] - minus[row][column]) / (2.0 * step);
                        let analytic = terms
                            .iter()
                            .find(|(atom, _)| *atom == source)
                            .map_or(0.0, |(_, derivative)| derivative[axis][row][column]);
                        worst = worst.max((numeric - analytic).abs());
                        largest = largest.max(numeric.abs());
                    }
                }
            }
        }
    }
    (worst, largest)
}

#[test]
fn the_frame_jacobian_matches_finite_differences_of_atom_frame() {
    // 判据 1：坐标架雅可比本身。逐元素比 `atom_frame` 的中心差分（`h = 1e-6`），**包括**
    // 那些"雅可比应为零"的源原子 —— 漏掉一个非零源与凭空多出零项都会被抓到。
    let step = 1.0e-6;
    for (name, molecule) in [
        ("water", displaced(water(), 0.02)),
        ("methanol", displaced(methanol(), 0.02)),
        ("methane", displaced(methane(), 0.02)),
    ] {
        let (worst, largest) = frame_jacobian_error(&molecule, step);
        let sources = (0..molecule.n_atoms())
            .map(|index| atom_frame_jacobian(&molecule, index).len())
            .max()
            .unwrap_or(0);
        println!(
            "{name:9}: {sources} nonzero sources/atom, max |dF| = {largest:.6e}, \
             max |analytic - FD| = {worst:.3e} per (atom, coordinate, frame element)"
        );
        assert!(
            largest > 1.0e-3,
            "{name}: the frame Jacobian is trivially zero"
        );
        assert!(
            worst < 1.0e-8,
            "{name}: the analytic frame Jacobian disagrees with finite differences of \
             atom_frame by {worst:.3e}"
        );
    }
    // 乙腈（`C–C≡N`，直线骨架）是**近共线**的坏条件例子：甲基碳的 `e2` 来自与 `e1` 夹角只有
    // 0.0100 rad 的方向，于是坐标架的二阶/三阶导数被 `1/sin², 1/sin³` 放大（`max |dF| = 1.06e2`），
    // 中心差分的**截断**在 `h = 1e-6` 就落到 1e-6 量级。这里不设绝对阈值，而是要求残差随 `h`
    // 按 `h²` 缩小 —— 这是"残差来自差分截断、不是解析项错了"的判别特征（解析漏项不会随 h 缩）。
    let molecule = displaced(acetonitrile(), 0.02);
    let (coarse, largest) = frame_jacobian_error(&molecule, 1.0e-6);
    let (fine, _) = frame_jacobian_error(&molecule, 1.0e-7);
    println!(
        "acetonitrile (near-collinear e2, max |dF| = {largest:.3e}): |analytic - FD| = \
         {coarse:.3e} at h = 1e-6, {fine:.3e} at h = 1e-7 (h^2 truncation, ratio {:.0})",
        coarse / fine
    );
    assert!(
        coarse > fine * 10.0,
        "acetonitrile: the frame-Jacobian residual does not shrink like h^2 \
         ({coarse:.3e} at h = 1e-6 vs {fine:.3e} at h = 1e-7) - that is no longer finite-difference \
         truncation"
    );
}

#[test]
fn the_covariant_gradient_matches_finite_differences() {
    // 判据 2：SV-02 的 `< 1e-5 Ha/Bohr`，对**独立**的能量中心差分，两种步长。
    // `h = 1e-4` 与历史测试（全局坐标架那条）同口径；`h = 1e-5` 更紧 —— 若解析梯度少一项，
    // 小步长不会让它变小（实测：只剩差分的 `O(h²)` 截断，见下面的收敛行）。
    let data = smd();
    let mut worst_overall: f64 = 0.0;
    for (name, molecule) in generic_systems() {
        for width in [0.10_f64, 0.15] {
            for points in [2048_usize, 4096] {
                let result = smd_cds_energy_gradient_with_frame(
                    &data,
                    &molecule,
                    points,
                    width,
                    true,
                    None,
                    CavityFrame::Covariant,
                )
                .expect("covariant gradient");
                for step in [1.0e-4_f64, 1.0e-5] {
                    // 先确认这一步没有跨分支（否则比的不是同一个面）。
                    let guard = branch_guard(&molecule, step);
                    assert!(
                        guard < 1.0e-6,
                        "{name} w={width} N={points} h={step}: the finite-difference step crosses a \
                         frame branch (deviation {guard:.3e} from the linear prediction)"
                    );
                    let mut per_atom = vec![0.0_f64; molecule.n_atoms()];
                    let mut largest = 0.0_f64;
                    for (atom, worst_here) in per_atom.iter_mut().enumerate() {
                        for axis in 0..3 {
                            let plus = covariant_energy(
                                &data,
                                &perturbed(&molecule, atom, axis, step),
                                points,
                                width,
                            );
                            let minus = covariant_energy(
                                &data,
                                &perturbed(&molecule, atom, axis, -step),
                                points,
                                width,
                            );
                            let numeric = (plus - minus) / (2.0 * step);
                            let analytic = result.gradient_kcal_per_angstrom[atom][axis];
                            *worst_here = worst_here.max((numeric - analytic).abs());
                            largest = largest.max(analytic.abs());
                        }
                    }
                    let worst = per_atom.iter().cloned().fold(0.0_f64, f64::max);
                    worst_overall = worst_overall.max(worst);
                    println!(
                        "{name:9} w={width:.2} N={points:5} h={step:.0e}: max|g| = {largest:.6e} \
                         kcal/(mol A); per-atom |analytic - FD| = {}",
                        per_atom
                            .iter()
                            .map(|value| format!("{value:.2e}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                    println!(
                        "{name:9} w={width:.2} N={points:5} h={step:.0e}: worst = {worst:.3e} \
                         kcal/(mol A) = {:.3e} Ha/Bohr (relative {:.2e})",
                        to_hartree_per_bohr(worst),
                        worst / largest
                    );
                    assert!(
                        to_hartree_per_bohr(worst) < 1.0e-5,
                        "{name} w={width} N={points} h={step}: the covariant analytic gradient \
                         disagrees with finite differences by {:.3e} Ha/Bohr (criterion 1e-5)",
                        to_hartree_per_bohr(worst)
                    );
                }
            }
        }
    }
    println!(
        "worst over every system/width/point count/step: {worst_overall:.3e} kcal/(mol A) = \
         {:.3e} Ha/Bohr",
        to_hartree_per_bohr(worst_overall)
    );
}

#[test]
fn the_covariant_gradient_is_translation_invariant() {
    // 判据 3：能量只依赖相对坐标，坐标架也是（`atom_frame` 只用相对向量）。
    let data = smd();
    for (name, molecule) in generic_systems() {
        let result = smd_cds_energy_gradient_with_frame(
            &data,
            &molecule,
            2048,
            0.15,
            true,
            None,
            CavityFrame::Covariant,
        )
        .expect("covariant gradient");
        let mut total = [0.0_f64; 3];
        for vector in &result.gradient_kcal_per_angstrom {
            for axis in 0..3 {
                total[axis] += vector[axis];
            }
        }
        let norm = (total[0] * total[0] + total[1] * total[1] + total[2] * total[2]).sqrt();
        println!("{name:9} |sum of gradients| = {norm:.3e} kcal/(mol A)");
        assert!(
            norm < 1.0e-10,
            "{name}: the covariant gradient is not translation invariant, got {norm:.3e}"
        );
    }
}

#[test]
fn the_symmetric_geometries_sit_on_a_frame_branch_boundary() {
    // 判据 4（记账，不是判据的放宽）：`atom_frame` 的离散选择让**对称构型**正好落在分支
    // 边界上。这里量的是"能量在 ±h 两侧差多少"，并且看它**随 h 缩小而缩小**（连续，只剩
    // 解析不可微）还是**不缩**（有跳跃）。
    //
    // 结论（实测见输出）：
    // * 水（`O` 的两个 `H` 严格等距）：偏差 ∝ h —— 能量连续，两个分支的采样差在 1e-6 量级，
    //   所以水的解析梯度对协变面的差分仍然成立（1.488e-2 那条差距是缺项，不是这里）；
    // * 甲醇（`C` 的两个 `H` 严格等距）：偏差**不随 h 缩小**（≈4.5e-3 kcal/mol）—— 坐标架
    //   换支让能量真的跳一下，任何导数（解析或差分）在这一点都不存在。
    let data = smd();
    let points = 2048;
    let step_small = 1.0e-4;
    let mut jumps = Vec::new();
    for (name, molecule) in [("water", water()), ("methanol", methanol())] {
        let base = covariant_energy(&data, &molecule, points, 0.10);
        let mut worst_small: f64 = 0.0;
        let mut worst_medium: f64 = 0.0;
        for atom in 0..molecule.n_atoms() {
            for axis in 0..3 {
                for step in [step_small, 10.0 * step_small] {
                    let plus = covariant_energy(
                        &data,
                        &perturbed(&molecule, atom, axis, step),
                        points,
                        0.10,
                    );
                    let minus = covariant_energy(
                        &data,
                        &perturbed(&molecule, atom, axis, -step),
                        points,
                        0.10,
                    );
                    let jump = (plus - base).abs().max((minus - base).abs());
                    if step == step_small {
                        worst_small = worst_small.max(jump);
                    } else {
                        worst_medium = worst_medium.max(jump);
                    }
                }
            }
        }
        println!(
            "{name:9}: E = {base:.12} kcal/mol; worst |E(c±h) - E(c)| = {worst_small:.3e} (h=1e-4) \
             vs {worst_medium:.3e} (h=1e-3) -> ratio {:.2}",
            worst_medium / worst_small
        );
        jumps.push((name, worst_small, worst_medium));
    }
    let water = jumps[0];
    let methanol = jumps[1];
    // 水：把 h 放大 10 倍，偏差也放大 ~10 倍（一阶变化，没有跳跃）。
    assert!(
        water.2 > 5.0 * water.1,
        "water: the deviation from the symmetric geometry now looks discontinuous \
         ({:.3e} at h=1e-4 vs {:.3e} at h=1e-3) - check the frame branch behaviour",
        water.1,
        water.2
    );
    // 甲醇：把 h 缩小 10 倍，偏差**没有**跟着缩小（跳跃主导），这是"导数不存在"的实测证据。
    assert!(
        methanol.1 > 1.0e-3 && methanol.2 / methanol.1 < 5.0,
        "methanol: the branch jump is gone ({:.3e} at h=1e-4 vs {:.3e} at h=1e-3) - either the \
         tie in atom_frame changed or the gradient can now be validated there too; re-derive \
         instead of loosening the criterion",
        methanol.1,
        methanol.2
    );
}
