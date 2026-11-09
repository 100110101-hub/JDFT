//! SV-01 的几何缺陷回归：空腔必须是**转动协变**的。
//!
//! 上一轮审计量出来的事实（本轮不重新推导，只把它钉住）：`cavity()` 的 Fibonacci 点钉在
//! **全局**坐标轴上，于是"转动同一个分子"会改掉遮蔽边界上的碎片 —— 水 + Bondi 半径、
//! 250 点/原子时面积/体积漂移 4.76e-3 / 3.56e-3（判据 1e-6），而下游 SV-02 的解析梯度
//! 判据（< 1e-5 Ha/Bohr vs 有限差分）在一个"转一下就跳 4.8e-3"的表面上不可能达到。
//!
//! 第 434 轮加的 [`CavityFrame::Covariant`] 把每个原子的点放进由**邻居几何**构造的局部坐标架，
//! 坐标架随分子一起转，于是面积/体积对刚性转动精确不变；第 435 轮把它**翻成默认**（`cavity()`
//! 与 `smd_surface()` 都用它），因为旧默认的转动漂移（水 + Bondi 半径、250 点/原子 4.76e-3 /
//! 3.56e-3，判据 1e-6）是 SV-02 解析梯度上不去、且同一分子转一下就移动 `ΔG_solv` 1.276
//! kcal/mol 的直接原因。这里逐条量：
//!
//! * 协变模式的转动不变性（面积、体积、tessera 数 M）—— 现在的**默认**；
//! * `CavityFrame::default()` 确实是协变、`cavity()` 与它逐位相同；
//! * 被换掉的全局模式的漂移（把缺陷本身留成回归，它仍在，只是不再是默认）；
//! * 两种模式的面积/体积差（"翻默认"的爆炸半径）；
//! * 坐标架本身随转动协变（逐原子逐轴）；
//! * 退化环境（孤立原子 0 邻居、双原子/直线 1 个方向、甲烷这种球形陀螺）的确定性。
use std::collections::BTreeMap;

use qwb_engine_api::{Atom, Molecule};
use qwb_props::cavity::{CavityFrame, atom_frame, cavity, cavity_with_frame, rotate_molecule};

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn bondi_radii() -> BTreeMap<String, f64> {
    let mut radii = BTreeMap::new();
    for (symbol, radius) in [
        ("H", 1.20),
        ("C", 1.70),
        ("N", 1.55),
        ("O", 1.52),
        ("F", 1.47),
        ("S", 1.80),
        ("Cl", 1.75),
    ] {
        radii.insert(symbol.to_string(), radius);
    }
    radii
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

/// 甲醇：C–O 1.42 Å、O–H 0.99 Å、C–H 1.10 Å、C–O–H 108.5° 的手搭结构
/// （与 `sv01_cosmo_scf.rs` 是同一份几何，便于跨测试对账）。
fn methanol() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.420]),
            Atom::new("H", [0.0, 0.885, 1.760]),
            Atom::new("H", [0.0, 1.025, -0.363]),
            Atom::new("H", [0.888, -0.512, -0.363]),
            Atom::new("H", [-0.888, -0.512, -0.363]),
        ],
        0,
        1,
    )
}

/// 甲烷（球形陀螺）：惯量主轴三重简并，所以坐标架**不能**靠主轴 + 破缺规则。
fn methane() -> Molecule {
    let scale = 1.089_f64 / 3.0_f64.sqrt();
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [scale, scale, scale]),
            Atom::new("H", [scale, -scale, -scale]),
            Atom::new("H", [-scale, scale, -scale]),
            Atom::new("H", [-scale, -scale, scale]),
        ],
        0,
        1,
    )
}

/// 审计所用的三种刚性转动（与 `sv01_cavity.rs` 同一组轴/角，便于与已有记录对齐）。
const ROTATIONS: [([f64; 3], f64); 3] = [
    ([0.0, 0.0, 1.0], 0.7),
    ([1.0, 0.0, 0.0], 1.3),
    ([0.3, -0.5, 0.81], 2.9),
];

fn relative_difference(left: f64, right: f64) -> f64 {
    (left - right).abs() / right.abs()
}

/// 逐转动打印（分子, 模式）的面积/体积相对偏差，返回面积/体积两个相对偏差数组。
fn rotation_deviations(molecule: &Molecule, frame: CavityFrame, points: usize) -> [Vec<f64>; 2] {
    let radii = bondi_radii();
    let reference =
        cavity_with_frame(molecule, &radii, 1.0, points, frame).expect("reference cavity");
    let mut area = Vec::new();
    let mut volume = Vec::new();
    for (axis, angle) in ROTATIONS {
        let rotated = rotate_molecule(molecule, axis, angle);
        let other =
            cavity_with_frame(&rotated, &radii, 1.0, points, frame).expect("rotated cavity");
        let area_relative = relative_difference(
            other.surface_area_angstrom2,
            reference.surface_area_angstrom2,
        );
        let volume_relative =
            relative_difference(other.volume_angstrom3, reference.volume_angstrom3);
        println!(
            "  {frame:?}: axis {axis:?} angle {angle}: area rel {area_relative:.3e}, volume rel {volume_relative:.3e}"
        );
        area.push(area_relative);
        volume.push(volume_relative);
    }
    [area, volume]
}

#[test]
fn covariant_area_and_volume_are_rotation_invariant() {
    // SV-01 的判据：面积/体积对刚性转动不变（< 1e-6）。协变模式按构造应当只剩舍入，
    // 这里把实测值打印出来 —— 断言用的是判据本身（1e-6），不是"观测到的更松的值"。
    for (name, molecule) in [("water", water()), ("methanol", methanol())] {
        println!("{name}, 250 points/atom, CavityFrame::Covariant:");
        let [area, volume] = rotation_deviations(&molecule, CavityFrame::Covariant, 250);
        for (index, value) in area.iter().enumerate() {
            assert!(
                *value < 1.0e-6,
                "{name}: covariant area drift {value:.3e} at rotation {index} exceeds the 1e-6 criterion"
            );
        }
        for (index, value) in volume.iter().enumerate() {
            assert!(
                *value < 1.0e-6,
                "{name}: covariant volume drift {value:.3e} at rotation {index} exceeds the 1e-6 criterion"
            );
        }
    }
}

#[test]
fn the_replaced_global_frame_still_drifts_which_is_why_the_default_flipped() {
    // 被换掉的旧默认（`CavityFrame::Global`）**没有**改动，所以漂移仍在。这条的作用有两条：
    // ① 它是"当初为什么要翻默认"的可执行记录（水 4.76e-3 / 3.56e-3，判据 1e-6）；
    // ② 若有人把 Global 的实现也改成协变（那会让历史钉住的数值失去意义），这条会红 ——
    //    那时必须同步改这里与账本，而不是放宽它。
    for (name, molecule) in [("water", water()), ("methanol", methanol())] {
        println!("{name}, 250 points/atom, CavityFrame::Global:");
        let [area, volume] = rotation_deviations(&molecule, CavityFrame::Global, 250);
        let worst_area = area.iter().cloned().fold(0.0_f64, f64::max);
        let worst_volume = volume.iter().cloned().fold(0.0_f64, f64::max);
        println!(
            "  {name}: worst area drift {worst_area:.3e}, worst volume drift {worst_volume:.3e}"
        );
        assert!(
            worst_area > 1.0e-6 || worst_volume > 1.0e-6,
            "{name}: the global frame is now rotation invariant too (worst area {worst_area:.3e}, \
             worst volume {worst_volume:.3e}) - update this test, tests/sv01_cavity.rs and the \
             ledger instead of relaxing it"
        );
    }
}

#[test]
fn the_covariant_cavity_keeps_the_same_tessera_count_under_rotation() {
    // 旧（全局）模式的 M 在转动下从 410 跳到 413 / 从 668 跳到 668+（遮蔽边界碎片翻转）。
    // 协变模式里 M 必须一个不差。全局那一列必须**显式**点名：`cavity()` 已经是协变。
    for (name, molecule) in [("water", water()), ("methanol", methanol())] {
        let radii = bondi_radii();
        let reference = cavity_with_frame(&molecule, &radii, 1.0, 250, CavityFrame::Covariant)
            .expect("reference");
        let global_reference =
            cavity_with_frame(&molecule, &radii, 1.0, 250, CavityFrame::Global).expect("global");
        let mut global_counts = Vec::new();
        for (axis, angle) in ROTATIONS {
            let rotated = rotate_molecule(&molecule, axis, angle);
            let covariant = cavity_with_frame(&rotated, &radii, 1.0, 250, CavityFrame::Covariant)
                .expect("covariant");
            let global =
                cavity_with_frame(&rotated, &radii, 1.0, 250, CavityFrame::Global).expect("global");
            println!(
                "  {name}: rotation {axis:?} {angle} -> M covariant {} (reference {}), global {} (reference {})",
                covariant.tesserae.len(),
                reference.tesserae.len(),
                global.tesserae.len(),
                global_reference.tesserae.len()
            );
            assert_eq!(
                covariant.tesserae.len(),
                reference.tesserae.len(),
                "{name}: the covariant tessera count moved under rotation"
            );
            global_counts.push(global.tesserae.len());
        }
        println!(
            "  {name}: M covariant = {}, M global over the rotations = {global_counts:?} (reference {})",
            reference.tesserae.len(),
            global_reference.tesserae.len()
        );
    }
}

#[test]
fn the_frame_axes_rotate_with_the_molecule() {
    // 协变性的**来源**：`frame(Q·r) = Q·frame(r)`。逐原子逐轴量，不经过面积。
    for (name, molecule) in [
        ("water", water()),
        ("methanol", methanol()),
        ("methane", methane()),
    ] {
        let reference: Vec<[[f64; 3]; 3]> = (0..molecule.n_atoms())
            .map(|index| atom_frame(&molecule, index))
            .collect();
        for (axis, angle) in ROTATIONS {
            let rotated = rotate_molecule(&molecule, axis, angle);
            // 同一个旋转作用在坐标架上：与 `rotate_molecule` 用的 Rodrigues 公式一致。
            let turn = |vector: [f64; 3]| -> [f64; 3] {
                let length = norm(axis);
                let (x, y, z) = (axis[0] / length, axis[1] / length, axis[2] / length);
                let (c, s) = (angle.cos(), angle.sin());
                let along = x * vector[0] + y * vector[1] + z * vector[2];
                [
                    vector[0] * c + (y * vector[2] - z * vector[1]) * s + x * along * (1.0 - c),
                    vector[1] * c + (z * vector[0] - x * vector[2]) * s + y * along * (1.0 - c),
                    vector[2] * c + (x * vector[1] - y * vector[0]) * s + z * along * (1.0 - c),
                ]
            };
            let mut worst: f64 = 0.0;
            for (index, reference_frame) in reference.iter().enumerate() {
                let image = atom_frame(&rotated, index);
                for (row, reference_axis) in reference_frame.iter().enumerate() {
                    let expected = turn(*reference_axis);
                    for component in 0..3 {
                        worst = worst.max((image[row][component] - expected[component]).abs());
                    }
                }
            }
            println!(
                "  {name}: rotation {axis:?} {angle} -> worst frame component difference {worst:.3e}"
            );
            assert!(
                worst < 1.0e-12,
                "{name}: frame is not covariant under rotation {axis:?} {angle} (worst {worst:.3e})"
            );
        }
    }
}

#[test]
fn the_degenerate_frame_cases_are_measured_not_assumed() {
    // 两种**数学上无法做到转动协变**的几何（见 `atom_frame` 的退化段落），把实测写清楚：
    // 孤立原子（0 邻居）与直线体系（只有 1 个方向）。这里不做"假装它们也不变"的断言，
    // 只把量级钉住，免得以后有人以为协变模式对所有体系都精确不变。
    let radii = bondi_radii();
    for (name, molecule) in [
        (
            "lone O",
            Molecule::new(vec![Atom::new("O", [1.3, -0.4, 2.1])], 0, 1),
        ),
        (
            "H2 (collinear)",
            Molecule::new(
                vec![
                    Atom::new("H", [0.0, 0.0, 0.0]),
                    Atom::new("H", [0.7, 0.5, 0.0]),
                ],
                0,
                1,
            ),
        ),
    ] {
        let reference =
            cavity_with_frame(&molecule, &radii, 1.0, 250, CavityFrame::Covariant).expect("ref");
        let mut worst_area: f64 = 0.0;
        let mut worst_volume: f64 = 0.0;
        for (axis, angle) in ROTATIONS {
            let rotated = rotate_molecule(&molecule, axis, angle);
            let other = cavity_with_frame(&rotated, &radii, 1.0, 250, CavityFrame::Covariant)
                .expect("rotated");
            let area = relative_difference(
                other.surface_area_angstrom2,
                reference.surface_area_angstrom2,
            );
            let volume = relative_difference(other.volume_angstrom3, reference.volume_angstrom3);
            println!(
                "  {name}: axis {axis:?} {angle} -> area rel {area:.3e}, volume rel {volume:.3e}"
            );
            worst_area = worst_area.max(area);
            worst_volume = worst_volume.max(volume);
        }
        println!(
            "  {name}: worst area rel {worst_area:.3e}, worst volume rel {worst_volume:.3e} \
             (no covariant frame exists for this geometry)"
        );
        assert!(
            worst_area < 5.0e-2 && worst_volume < 5.0e-2,
            "{name}: the documented fallback frame degraded beyond sampling level \
             (area {worst_area:.3e}, volume {worst_volume:.3e})"
        );
    }
}

#[test]
fn degenerate_environments_get_deterministic_frames() {
    let radii = bondi_radii();
    // 0 个邻居：孤立原子 → 全局单位架（没有取向信息可用）。
    let lone = Molecule::new(vec![Atom::new("O", [0.0, 0.0, 0.0])], 0, 1);
    assert_eq!(
        atom_frame(&lone, 0),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );
    // 孤立原子的面积**精确**是 4πR²（所有碎片都暴露），两种模式逐位相同。
    let global = cavity_with_frame(&lone, &radii, 1.0, 250, CavityFrame::Global).expect("global");
    let covariant =
        cavity_with_frame(&lone, &radii, 1.0, 250, CavityFrame::Covariant).expect("covariant");
    assert_eq!(
        cavity(&lone, &radii, 1.0, 250)
            .expect("default")
            .surface_area_angstrom2,
        covariant.surface_area_angstrom2,
        "the default frame is covariant even for a lone atom"
    );
    println!(
        "lone O: area global {:.12} covariant {:.12}, volume global {:.12} covariant {:.12}",
        global.surface_area_angstrom2,
        covariant.surface_area_angstrom2,
        global.volume_angstrom3,
        covariant.volume_angstrom3
    );
    assert_eq!(
        global.surface_area_angstrom2, covariant.surface_area_angstrom2,
        "a lone atom exposes every tessera in both frames"
    );
    assert_eq!(global.tesserae.len(), covariant.tesserae.len());

    // 1 个方向：双原子（所有其它原子都与 e1 共线）→ 显式兜底的垂直向量，必须真的垂直。
    let diatomic = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 1.2]),
        ],
        0,
        1,
    );
    for index in 0..2 {
        let frame = atom_frame(&diatomic, index);
        let axis = frame[0];
        // e1 指向对方：±z。
        assert!((axis[2].abs() - 1.0).abs() < 1.0e-15);
        assert!(dot(frame[1], axis).abs() < 1.0e-15);
        assert!(dot(frame[2], axis).abs() < 1.0e-15);
        assert!((norm(frame[1]) - 1.0).abs() < 1.0e-15);
        // 确定性：同一个输入两次必须逐位相同。
        assert_eq!(frame, atom_frame(&diatomic, index));
        println!("  diatomic atom {index}: frame {frame:?}");
    }

    // 甲烷（球形陀螺）：四个 H 各有 1 个邻居 C，其余 H 与 H→C 不共线，所以坐标架走
    // 「第一根不共线方向」而不是惯量主轴 —— 主轴在这里三重简并，靠它需要额外破缺规则。
    let methane = methane();
    let mut hydrogen_axes = Vec::new();
    for index in 0..methane.n_atoms() {
        let frame = atom_frame(&methane, index);
        for row in 0..3 {
            let length = norm(frame[row]);
            assert!((length - 1.0).abs() < 1.0e-14, "row {row} length {length}");
            for other in (row + 1)..3 {
                assert!(
                    dot(frame[row], frame[other]).abs() < 1.0e-14,
                    "frame rows {row} and {other} are not orthogonal"
                );
            }
        }
        println!("  methane atom {index}: frame {frame:?}");
        if index > 0 {
            hydrogen_axes.push(frame[0]);
        }
    }
    // 四个 H 的 e1 指向 C 的**反**方向（四面体四个顶点），不能全都一样。
    assert!(
        hydrogen_axes.windows(2).any(|pair| pair[0] != pair[1]),
        "methane hydrogen frames collapsed onto one axis: {hydrogen_axes:?}"
    );

    // 两种模式在甲烷上都必须给出可用的空腔（体积为正、法向为单位向量）。
    let cavity_global =
        cavity_with_frame(&methane, &radii, 1.0, 250, CavityFrame::Global).expect("global methane");
    let cavity_covariant = cavity_with_frame(&methane, &radii, 1.0, 250, CavityFrame::Covariant)
        .expect("covariant methane");
    println!(
        "methane: area global {:.6} covariant {:.6}, volume global {:.6} covariant {:.6}, M {} vs {}",
        cavity_global.surface_area_angstrom2,
        cavity_covariant.surface_area_angstrom2,
        cavity_global.volume_angstrom3,
        cavity_covariant.volume_angstrom3,
        cavity_global.tesserae.len(),
        cavity_covariant.tesserae.len()
    );
    cavity_covariant
        .validate()
        .expect("covariant methane validates");
}

#[test]
fn mode_to_mode_area_and_volume_shift_is_sampling_sized() {
    // 第 435 轮把协变模式翻成默认后，面积/体积相对旧默认（全局）动了多少（爆炸半径）。
    // 两种模式的差别**只**是点集相对空腔的取向，所以差值必须留在采样误差量级（不是结构性
    // 的偏移）。注意这里**必须**显式给 Global：`cavity()` 已经是协变，写成 `cavity()` 会
    // 变成"协变 vs 协变"的自比（差值恒为 0，断言变成空话）。
    for (name, molecule) in [("water", water()), ("methanol", methanol())] {
        let radii = bondi_radii();
        let global =
            cavity_with_frame(&molecule, &radii, 1.0, 250, CavityFrame::Global).expect("global");
        let covariant = cavity_with_frame(&molecule, &radii, 1.0, 250, CavityFrame::Covariant)
            .expect("covariant");
        let area_absolute = covariant.surface_area_angstrom2 - global.surface_area_angstrom2;
        let volume_absolute = covariant.volume_angstrom3 - global.volume_angstrom3;
        let area_relative = relative_difference(
            covariant.surface_area_angstrom2,
            global.surface_area_angstrom2,
        );
        let volume_relative =
            relative_difference(covariant.volume_angstrom3, global.volume_angstrom3);
        println!(
            "{name}: area global {:.6} covariant {:.6} (abs {area_absolute:+.6} A^2, rel {area_relative:.3e})",
            global.surface_area_angstrom2, covariant.surface_area_angstrom2
        );
        println!(
            "{name}: volume global {:.6} covariant {:.6} (abs {volume_absolute:+.6} A^3, rel {volume_relative:.3e})",
            global.volume_angstrom3, covariant.volume_angstrom3
        );
        println!(
            "{name}: M global {} covariant {} (delta {:+})",
            global.tesserae.len(),
            covariant.tesserae.len(),
            covariant.tesserae.len() as i64 - global.tesserae.len() as i64
        );
        assert!(
            area_relative < 5.0e-2,
            "{name}: the frame change moved the area by {area_relative:.3e} - that is no longer \
             a sampling-level difference"
        );
        assert!(
            volume_relative < 5.0e-2,
            "{name}: the frame change moved the volume by {volume_relative:.3e}"
        );
    }
}

#[test]
fn the_covariant_tesserae_still_sit_on_their_atoms_spheres() {
    // 坐标架只该换"点的取向"，几何自洽性必须还在：每个暴露碎片的中心离它所属原子的球心
    // 恰好 `R_i`，法向就是那条单位径向。法向若不是单位向量，碎裂点会离开球面。
    let radii = bondi_radii();
    let molecule = methanol();
    let cavity_ =
        cavity_with_frame(&molecule, &radii, 1.0, 250, CavityFrame::Covariant).expect("cavity");
    let mut worst_radius: f64 = 0.0;
    let mut worst_normal: f64 = 0.0;
    for tessera in &cavity_.tesserae {
        let center = molecule.atoms[tessera.atom_index].position_angstrom;
        let radius = radii[&molecule.atoms[tessera.atom_index].symbol];
        let offset = [
            tessera.center_angstrom[0] - center[0],
            tessera.center_angstrom[1] - center[1],
            tessera.center_angstrom[2] - center[2],
        ];
        worst_radius = worst_radius.max((norm(offset) - radius).abs() / radius);
        worst_normal = worst_normal.max((norm(tessera.normal) - 1.0).abs());
    }
    println!(
        "methanol covariant: {} tesserae, worst | |p−c|/R − 1 | = {worst_radius:.3e}, \
         worst | |n| − 1 | = {worst_normal:.3e}",
        cavity_.tesserae.len()
    );
    assert!(
        worst_radius < 1.0e-12,
        "tesserae left their spheres: {worst_radius:.3e}"
    );
    assert!(
        worst_normal < 1.0e-12,
        "non-unit normal: {worst_normal:.3e}"
    );
}

#[test]
fn the_default_frame_is_covariant_and_the_global_bits_are_pinned() {
    // 第 435 轮翻默认：`CavityFrame::default()` 必须是 `Covariant`，`cavity()` 必须与
    // `cavity_with_frame(..., Covariant)` **逐位**相同；同时把被换掉的全局模式的 f64 位模式
    // 继续钉住（历史数值不许无声变动），并把新默认的位模式也钉上（不许无声漂移）。
    assert_eq!(
        CavityFrame::default(),
        CavityFrame::Covariant,
        "the default frame must stay covariant - tests/sv01_cavity.rs and the ledger assume it"
    );
    let radii = bondi_radii();
    let water_default = cavity(&water(), &radii, 1.0, 250).expect("water");
    let water_covariant = cavity_with_frame(&water(), &radii, 1.0, 250, CavityFrame::Covariant)
        .expect("water covariant");
    let methanol_default = cavity(&methanol(), &radii, 1.0, 250).expect("methanol");
    let methanol_covariant =
        cavity_with_frame(&methanol(), &radii, 1.0, 250, CavityFrame::Covariant)
            .expect("methanol covariant");
    for (name, default, covariant) in [
        ("water", &water_default, &water_covariant),
        ("methanol", &methanol_default, &methanol_covariant),
    ] {
        assert_eq!(
            default.surface_area_angstrom2.to_bits(),
            covariant.surface_area_angstrom2.to_bits(),
            "{name}: cavity() must be bit-identical to the explicit covariant frame"
        );
        assert_eq!(
            default.volume_angstrom3.to_bits(),
            covariant.volume_angstrom3.to_bits(),
            "{name}: cavity() must be bit-identical to the explicit covariant frame"
        );
        assert_eq!(default.tesserae.len(), covariant.tesserae.len());
    }
    // 被换掉的旧默认（`CavityFrame::Global`）：数值逐位不变，钉住它。
    let water_global =
        cavity_with_frame(&water(), &radii, 1.0, 250, CavityFrame::Global).expect("water global");
    let methanol_global = cavity_with_frame(&methanol(), &radii, 1.0, 250, CavityFrame::Global)
        .expect("methanol global");
    println!(
        "water Global: M {} A {:?} V {:?}",
        water_global.tesserae.len(),
        water_global.surface_area_angstrom2,
        water_global.volume_angstrom3
    );
    println!(
        "methanol Global: M {} A {:?} V {:?}",
        methanol_global.tesserae.len(),
        methanol_global.surface_area_angstrom2,
        methanol_global.volume_angstrom3
    );
    assert_eq!(water_global.tesserae.len(), 410);
    assert_eq!(
        water_global.surface_area_angstrom2.to_bits(),
        0x4042_3aa4_e89c_569f
    );
    assert_eq!(
        water_global.volume_angstrom3.to_bits(),
        0x4033_635a_6476_59bf
    );
    assert_eq!(methanol_global.tesserae.len(), 668);
    assert_eq!(
        methanol_global.surface_area_angstrom2.to_bits(),
        0x404d_d531_65ef_fef0
    );
    assert_eq!(
        methanol_global.volume_angstrom3.to_bits(),
        0x4042_5994_36eb_bc6e
    );
    // 新默认（协变）：M 少 2 / 8 个碎片（遮蔽边界上少翻转），面积小 0.52 % / 1.04 %。
    println!(
        "water default (covariant): M {} A {:?} ({:#018x}) V {:?} ({:#018x})",
        water_default.tesserae.len(),
        water_default.surface_area_angstrom2,
        water_default.surface_area_angstrom2.to_bits(),
        water_default.volume_angstrom3,
        water_default.volume_angstrom3.to_bits()
    );
    println!(
        "methanol default (covariant): M {} A {:?} ({:#018x}) V {:?} ({:#018x})",
        methanol_default.tesserae.len(),
        methanol_default.surface_area_angstrom2,
        methanol_default.surface_area_angstrom2.to_bits(),
        methanol_default.volume_angstrom3,
        methanol_default.volume_angstrom3.to_bits()
    );
    assert_eq!(water_default.tesserae.len(), 408);
    assert_eq!(methanol_default.tesserae.len(), 660);
    // 新默认的位模式也钉住（`V` 同上）。若协变实现的求和顺序/坐标架改动，这两行会红。
    assert_eq!(
        water_default.surface_area_angstrom2.to_bits(),
        0x4042_2283_a097_7b67
    );
    assert_eq!(
        water_default.volume_angstrom3.to_bits(),
        0x4033_4e43_0a1e_736c
    );
    assert_eq!(
        methanol_default.surface_area_angstrom2.to_bits(),
        0x404d_8579_2dbd_00c2
    );
    assert_eq!(
        methanol_default.volume_angstrom3.to_bits(),
        0x4042_42ea_1901_2c8c
    );
}

#[test]
fn both_frames_converge_to_the_same_dense_surface() {
    // 点在哪个坐标架里生成不该改变"表面的面积"这个几何量：把点数加大，
    // 两种模式必须收敛到同一个数（差别只剩 O(1/√N) 采样）。这里显式给两种坐标架
    // —— `cavity()` 是协变，那个"全局"的那一半必须自己点名。
    let radii = bondi_radii();
    let molecule = water();
    let global =
        cavity_with_frame(&molecule, &radii, 1.0, 20_000, CavityFrame::Global).expect("global");
    let covariant = cavity_with_frame(&molecule, &radii, 1.0, 20_000, CavityFrame::Covariant)
        .expect("covariant");
    let area_relative = relative_difference(
        covariant.surface_area_angstrom2,
        global.surface_area_angstrom2,
    );
    let volume_relative = relative_difference(covariant.volume_angstrom3, global.volume_angstrom3);
    println!(
        "water 20000 points: area global {:.8} covariant {:.8} (rel {area_relative:.3e}), \
         volume global {:.8} covariant {:.8} (rel {volume_relative:.3e}), M {} vs {}",
        global.surface_area_angstrom2,
        covariant.surface_area_angstrom2,
        global.volume_angstrom3,
        covariant.volume_angstrom3,
        global.tesserae.len(),
        covariant.tesserae.len()
    );
    assert!(
        area_relative < 1.0e-3,
        "the two frames disagree on the dense surface area: {area_relative:.3e}"
    );
    assert!(
        volume_relative < 1.0e-3,
        "the two frames disagree on the dense surface volume: {volume_relative:.3e}"
    );
}
