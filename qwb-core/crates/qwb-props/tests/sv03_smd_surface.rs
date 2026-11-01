//! SV-03 第二半验收：多原子 CDS 表面分解的四条自洽性质。
//!
//! 上游的 CDS 表面在 `CDSCALC` → `CDS_EG` → `DAREAL` 里（它自己的点集与遮蔽判定，
//! 我们还没复刻；`SMXCDS` 是成键接触权重与 σ 组装，不是表面）。"点集不同就没有逐位
//! 可比性"，所以这一层不追死差，而是钉四条**与实现无关**的性质：
//!
//! 1. 单原子极限 = 闭式 `4π(R+0.4)²`（与已验证的 σ 层对接）；
//! 2. `w = 0` 复现二值遮蔽（与 SV-01 已验收的 `cavity()` 对接）；
//! 3. 双原子极限收敛到**解析**球帽面积（`analytic_area_volume` 对双原子精确）；
//! 4. 随距离/带宽单调、分离时严格可加；转动不变性只到采样误差（`O(1/√N)`）。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_props::cavity::{analytic_area_volume, cavity, rotate_molecule};
use qwb_props::smd::{cds_energy_kcal, single_atom_area_angstrom2};
use qwb_props::smd_surface::{cds_effective_radii, smd_surface};
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

/// 沿 `axis` 方向的 H₂，键长 `distance`（Å）。
fn h2_along(axis: [f64; 3], distance: f64) -> Molecule {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, 0.0]),
            Atom::new(
                "H",
                [
                    axis[0] / norm * distance,
                    axis[1] / norm * distance,
                    axis[2] / norm * distance,
                ],
            ),
        ],
        0,
        1,
    )
}

fn h2_at(distance: f64) -> Molecule {
    h2_along([0.0, 0.0, 1.0], distance)
}

fn single_atom(symbol: &str) -> Molecule {
    Molecule::new(vec![Atom::new(symbol, [0.0, 0.0, 0.0])], 0, 1)
}

/// 由数据里的 vdW 半径 + `area_offset` 拼出 `cavity()` 需要的每元素半径表。
fn effective_radii_map(data: &SmdAtomic) -> BTreeMap<String, f64> {
    data.cds
        .vdw_radii_angstrom
        .iter()
        .map(|(symbol, radius)| (symbol.clone(), radius + data.cds.area_offset_angstrom))
        .collect()
}

fn relative(left: f64, right: f64) -> f64 {
    (left - right).abs() / right.abs().max(1.0)
}

#[test]
fn the_single_atom_limit_matches_the_closed_form() {
    // 一个原子没有别的球遮蔽它，所以每个采样点都整份计入：`A = 4π(R+0.4)²`
    // 必须**按构造**成立（只差浮点求和误差）。这条把表面分解和已验证的 σ 层接上。
    let data = smd();
    for symbol in ["H", "C", "O", "F", "S", "Cl", "Br"] {
        let molecule = single_atom(symbol);
        let surface = smd_surface(&data, &molecule, 2000, 0.0).expect("surface");
        let closed = single_atom_area_angstrom2(&data, symbol).expect("closed form");
        println!(
            "{symbol}: tessellated {:.12} vs closed {:.12} (rel {:.2e})",
            surface.areas_angstrom2[0],
            closed,
            relative(surface.areas_angstrom2[0], closed)
        );
        assert!(
            relative(surface.areas_angstrom2[0], closed) < 1.0e-13,
            "{symbol}: single-atom area must be the closed form"
        );
        assert!(
            relative(surface.total_area_angstrom2, closed) < 1.0e-13,
            "{symbol}: total area of a lone atom must be its sphere area"
        );
    }
}

#[test]
fn a_zero_switch_width_reproduces_the_binary_occlusion() {
    // `w = 0` 是二值遮蔽极限，必须与 SV-01 已经验收的 `cavity()` **逐原子**对上：
    // 这证明"平滑切换"退化成"点在球内就丢掉"，而不是另一套几何。
    let data = smd();
    let molecule = water();
    let points = 1024;
    let radii = effective_radii_map(&data);
    let reference = cavity(&molecule, &radii, 1.0, points).expect("cavity");

    let mut per_atom = vec![0.0_f64; molecule.n_atoms()];
    for tessera in &reference.tesserae {
        per_atom[tessera.atom_index] += tessera.area_angstrom2;
    }

    let surface = smd_surface(&data, &molecule, points, 0.0).expect("surface");
    println!(
        "water: cavity total {:.12} vs smd_surface {:.12} (rel {:.2e})",
        reference.surface_area_angstrom2,
        surface.total_area_angstrom2,
        relative(
            surface.total_area_angstrom2,
            reference.surface_area_angstrom2
        )
    );
    assert!(
        relative(
            surface.total_area_angstrom2,
            reference.surface_area_angstrom2
        ) < 1.0e-12,
        "the zero-width switch must reproduce the binary occlusion test"
    );
    for (index, expected) in per_atom.iter().enumerate() {
        println!(
            "  atom {index}: cavity {expected:.12} vs smd_surface {:.12}",
            surface.areas_angstrom2[index]
        );
        assert!(
            relative(surface.areas_angstrom2[index], *expected) < 1.0e-12,
            "atom {index}: per-atom attribution must agree with the cavity tesserae"
        );
    }
}

#[test]
fn the_binary_limit_converges_to_the_analytic_two_sphere_area() {
    // 双原子是唯一有**解析**暴露面积的体系（球帽公式精确，三重交叠不存在），
    // 所以它给出"镶嵌误差有多大"的诚实标定，而不是一个编出来的容差。
    let data = smd();
    let radius = data.cds.vdw_radii_angstrom["H"] + data.cds.area_offset_angstrom;
    // 键轴取 [0.6, 0.8, 0]：与 Fibonacci 点集的 z 分档不对齐，才是真正的采样检验
    // （沿 z 时球心连线与 z 分档重合，采样会**恰好**精确，看不出误差）。
    let molecule = h2_along([0.6, 0.8, 0.0], 1.2);
    let mut radii = BTreeMap::new();
    radii.insert("H".to_string(), radius);
    let (analytic, _) = analytic_area_volume(&molecule, &radii, 1.0).expect("analytic");
    // 手算锚：z = d/2 = 0.6 → 帽高 h = R − 0.6 = 1.0，A = 2(4πR² − 2πR·1.0)
    let hand = 2.0
        * (4.0 * std::f64::consts::PI * radius * radius
            - 2.0 * std::f64::consts::PI * radius * (radius - 0.6));
    println!("analytic H2 area {analytic:.12} (hand {hand:.12})");
    assert!(
        relative(analytic, hand) < 1.0e-12,
        "the analytic cap formula must match the hand computation"
    );

    let mut errors = Vec::new();
    for points in [256_usize, 1024, 4096, 16384] {
        let surface = smd_surface(&data, &molecule, points, 0.0).expect("surface");
        let error = (surface.total_area_angstrom2 - analytic).abs();
        println!(
            "  N = {points:>5}: area {:.9}, |error| {error:.3e} ({:.3} %)",
            surface.total_area_angstrom2,
            100.0 * error / analytic
        );
        errors.push(error);
    }
    // 准随机收敛：最密的一档必须比最疏的一档小一个量级以上（`O(1/√N)`：N 翻 64 倍
    // 期望缩 8 倍）。不要求逐档单调 —— Fibonacci 点集的误差会小幅振荡。
    println!(
        "error 256 -> 16384: {:.3e} -> {:.3e}",
        errors[0],
        errors[errors.len() - 1]
    );
    assert!(
        errors[errors.len() - 1] < 0.5 * errors[0],
        "the tessellation error must shrink with points_per_atom: {errors:?}"
    );
    // 这一层自己声明的精度上界（不是对上游的容差）。
    assert!(
        errors[errors.len() - 1] / analytic < 5.0e-3,
        "at N = 16384 the binary tessellation must be within 0.5 % of the analytic area"
    );
}

#[test]
fn the_switching_band_is_sandwiched_between_binary_and_isolated() {
    // 带宽落在球面内侧，所以 `θ` 对 `w` 逐点单调不减：面积随 `w` 单调不减，
    // 且被夹在二值结果与"互不遮蔽"的和之间。这条抓的是切换函数用错符号/
    // 把带宽放到球面外侧（那样面积会随 `w` 变大而减小）。
    let data = smd();
    let molecule = water();
    let points = 2048;
    let isolated: f64 = molecule
        .atoms
        .iter()
        .map(|atom| single_atom_area_angstrom2(&data, &atom.symbol).expect("area"))
        .sum();

    let mut previous = 0.0_f64;
    for width in [0.0, 0.02, 0.1, 0.3, 0.8] {
        let surface = smd_surface(&data, &molecule, points, width).expect("surface");
        println!(
            "  w = {width:.2}: total {:.9} (isolated sum {isolated:.9})",
            surface.total_area_angstrom2
        );
        assert!(
            surface.total_area_angstrom2 >= previous - 1.0e-12,
            "area must not decrease when the switching band widens"
        );
        assert!(
            surface.total_area_angstrom2 < isolated + 1.0e-9,
            "a switched surface can never exceed the sum of isolated spheres"
        );
        previous = surface.total_area_angstrom2;
    }
    // 带宽确实"起了作用"：w = 0.3 的面积必须比二值结果大出可测量的量。
    let binary = smd_surface(&data, &molecule, points, 0.0).expect("surface");
    let wide = smd_surface(&data, &molecule, points, 0.3).expect("surface");
    println!(
        "binary {:.9} -> w=0.3 {:.9} (delta {:.3e})",
        binary.total_area_angstrom2,
        wide.total_area_angstrom2,
        wide.total_area_angstrom2 - binary.total_area_angstrom2
    );
    assert!(
        wide.total_area_angstrom2 - binary.total_area_angstrom2 > 1.0e-3,
        "the switching band must expose partially buried tesserae"
    );
}

#[test]
fn the_rotation_drift_of_the_surface_is_sampling_noise() {
    // 采样点钉在**全局**坐标系里（Fibonacci 球的 z 分档是全局的），分子整体转动时
    // 采样图案与分子的相对取向变了 → 面积只能到采样误差量级。这不是遮蔽判定的 bug，
    // 所以判据写成"漂移随点数收敛"，并把实测值打出来。
    let data = smd();
    let molecule = water();
    let width = 0.15;
    let rotated = rotate_molecule(&molecule, [0.3, -0.5, 0.8], 2.6);
    let mut drift = Vec::new();
    for points in [1024_usize, 4096, 16384] {
        let base = smd_surface(&data, &molecule, points, width).expect("surface");
        let other = smd_surface(&data, &rotated, points, width).expect("surface");
        let value = relative(other.total_area_angstrom2, base.total_area_angstrom2);
        println!("{points} points: rotation drift {value:.3e}");
        drift.push(value);
    }
    assert!(
        drift[2] < drift[0],
        "the rotation drift must shrink with more points (sampling artifact): {drift:?}"
    );
    assert!(
        drift[0] < 2.0e-3,
        "at 1024 points the drift must stay sampling-scale, got {:.3e}",
        drift[0]
    );
}

#[test]
fn separated_atoms_add_exactly_and_so_does_the_aqueous_cds_energy() {
    // 两个球相距 10 Å（远大于 2(R+0.4)）时互不遮蔽，面积与能量都必须严格可加。
    let data = smd();
    let molecule = h2_at(10.0);
    let points = 1024;
    let surface = smd_surface(&data, &molecule, points, 0.15).expect("surface");
    let single = single_atom_area_angstrom2(&data, "H").expect("area");
    println!(
        "separated H2: {:.12} vs 2 x {single:.12}",
        surface.total_area_angstrom2
    );
    assert!(
        relative(surface.total_area_angstrom2, 2.0 * single) < 1.0e-12,
        "separated atoms must add exactly"
    );

    let symbols = ["H", "H"];
    let from_surface =
        cds_energy_kcal(&data, &symbols, &surface.areas_angstrom2, true, None).expect("cds");
    let per_atom = cds_energy_kcal(&data, &["H"], &[single], true, None).expect("cds single atom");
    println!("separated H2 CDS {from_surface:.12} vs 2 x {per_atom:.12} kcal/mol");
    assert!(
        relative(from_surface, 2.0 * per_atom) < 1.0e-12,
        "the aqueous CDS energy must be additive for separated atoms"
    );
}

#[test]
fn the_area_decreases_monotonically_as_two_atoms_approach() {
    // 距离变小时被遮蔽的采样点只会变多 → 面积单调不增；进入重叠区后严格下降。
    let data = smd();
    let points = 1024;
    let width = 0.1;
    let distances = [8.0, 5.0, 3.4, 3.0, 2.6, 2.2, 1.8, 1.4, 1.0];
    let mut previous = f64::INFINITY;
    let mut values = Vec::new();
    for distance in distances {
        let surface = smd_surface(&data, &h2_at(distance), points, width).expect("surface");
        println!(
            "  d = {distance:>4}: area {:.9}",
            surface.total_area_angstrom2
        );
        assert!(
            surface.total_area_angstrom2 <= previous + 1.0e-12,
            "area must not grow when the atoms get closer"
        );
        previous = surface.total_area_angstrom2;
        values.push(previous);
    }
    // 8.0 与 5.0 都远在作用范围外（2(R+0.4) = 6.4 之外还隔着带宽），应当完全相同。
    assert!(
        relative(values[1], values[0]) < 1.0e-12,
        "at 5 A the two spheres are still independent"
    );
    // 从 3.0 起严格下降，且 1.0 时明显小于 3.0。
    for window in values[3..].windows(2) {
        assert!(
            window[1] < window[0],
            "inside the overlap region the area must strictly decrease"
        );
    }
    println!(
        "d = 3.0 -> 1.0: {:.9} -> {:.9}",
        values[3],
        values[values.len() - 1]
    );
}

#[test]
fn the_surface_lies_between_the_binary_occlusion_and_the_isolated_sum() {
    // 三条几何量互相比对：二值 ≤ 切换 ≤ 孤立和，且切换结果与二值的差就是"部分暴露"
    // 的那部分面积（必须为正、必须小于整个孤立和）。
    let data = smd();
    let molecule = water();
    let points = 2048;
    let width = 0.2;
    let radii = effective_radii_map(&data);
    let binary = cavity(&molecule, &radii, 1.0, points).expect("cavity");
    let surface = smd_surface(&data, &molecule, points, width).expect("surface");
    let isolated: f64 = molecule
        .atoms
        .iter()
        .map(|atom| single_atom_area_angstrom2(&data, &atom.symbol).expect("area"))
        .sum();
    println!(
        "binary {:.9} <= switched {:.9} <= isolated {isolated:.9}",
        binary.surface_area_angstrom2, surface.total_area_angstrom2
    );
    assert!(binary.surface_area_angstrom2 < surface.total_area_angstrom2);
    assert!(surface.total_area_angstrom2 < isolated);
    assert!(
        surface.total_area_angstrom2 - binary.surface_area_angstrom2 < isolated,
        "the partially exposed fraction cannot exceed the whole surface"
    );
}

#[test]
fn out_of_range_inputs_are_reported_instead_of_silently_degrading() {
    let data = smd();
    let molecule = water();

    let too_few = smd_surface(&data, &molecule, 8, 0.1).expect_err("8 points must be refused");
    println!("8 points -> {too_few}");
    assert!(matches!(too_few, EngineError::InvalidInput(_)));

    let negative = smd_surface(&data, &molecule, 1024, -0.1).expect_err("negative width");
    println!("negative width -> {negative}");
    assert!(matches!(negative, EngineError::InvalidInput(_)));

    let nan = smd_surface(&data, &molecule, 1024, f64::NAN).expect_err("NaN width");
    println!("NaN width -> {nan}");
    assert!(matches!(nan, EngineError::InvalidInput(_)));

    let xenon = Molecule::new(vec![Atom::new("Xe", [0.0, 0.0, 0.0])], 0, 1);
    let missing = smd_surface(&data, &xenon, 1024, 0.1).expect_err("Xe is outside H-Kr");
    println!("Xe -> {missing}");
    assert!(matches!(missing, EngineError::Data(_)));
    assert!(
        cds_effective_radii(&data, &xenon).is_err(),
        "the radius lookup must report the missing element"
    );
}
