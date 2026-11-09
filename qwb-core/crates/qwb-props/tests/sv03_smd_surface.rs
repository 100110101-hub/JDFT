//! SV-03 第二半验收：多原子 CDS 表面分解的四条自洽性质。
//!
//! 上游的 CDS 表面在 `CDSCALC` → `CDS_EG` → `DAREAL` 里（它自己的点集与遮蔽判定，
//! 我们还没复刻；`SMXCDS` 是成键接触权重与 σ 组装，不是表面）。"点集不同就没有逐位
//! 可比性"，所以这一层不追死差，而是钉四条**与实现无关**的性质：
//!
//! 1. 单原子极限 = 闭式 `4π(R+0.4)²`（与已验证的 σ 层对接）；
//! 2. `w = 0` 复现二值遮蔽（与 SV-01 已验收的 `cavity()` 对接）；
//! 3. 双原子极限收敛到**解析**球帽面积（`analytic_area_volume` 对双原子精确）；
//! 4. 随距离/带宽单调、分离时严格可加；转动不变性见下。
//!
//! 第 435 轮把 `smd_surface()` 的默认坐标架翻成与 `cavity()` 相同的**协变**（每原子局部
//! 坐标架），所以第 4 条里的"转动不变性只到采样误差"变成了**精确不变**；旧（全局）模式
//! 的 `O(1/√N)` 漂移保留成显式点名的回归，不再是默认。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_props::cavity::{
    CavityFrame, analytic_area_volume, cavity, cavity_with_frame, rotate_molecule,
};
use qwb_props::smd::{cds_energy_kcal, single_atom_area_angstrom2};
use qwb_props::smd_surface::{cds_effective_radii, smd_surface, smd_surface_with_frame};
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
    //
    // 第 435 轮两条路径的**默认**都是协变坐标架，所以这一条是"同坐标架下"的横比（这正是
    // 要求：跨路径比较必须同架）。下面再显式跑一遍 Global 架，保证两条实现**各自**在同一个
    // 坐标架里也逐原子一致 —— 否则"只翻了一半"这类错误会躲在这一条后面。
    let data = smd();
    let molecule = water();
    let points = 1024;
    let radii = effective_radii_map(&data);

    for frame in [CavityFrame::Covariant, CavityFrame::Global] {
        let reference = cavity_with_frame(&molecule, &radii, 1.0, points, frame).expect("cavity");
        let mut per_atom = vec![0.0_f64; molecule.n_atoms()];
        for tessera in &reference.tesserae {
            per_atom[tessera.atom_index] += tessera.area_angstrom2;
        }
        let surface =
            smd_surface_with_frame(&data, &molecule, points, 0.0, frame).expect("surface");
        println!(
            "water [{frame:?}]: cavity total {:.12} vs smd_surface {:.12} (rel {:.2e})",
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
            "{frame:?}: the zero-width switch must reproduce the binary occlusion test"
        );
        for (index, expected) in per_atom.iter().enumerate() {
            println!(
                "  [{frame:?}] atom {index}: cavity {expected:.12} vs smd_surface {:.12}",
                surface.areas_angstrom2[index]
            );
            assert!(
                relative(surface.areas_angstrom2[index], *expected) < 1.0e-12,
                "{frame:?} atom {index}: per-atom attribution must agree with the cavity tesserae"
            );
        }
    }
    // 默认必须是协变，而且 `smd_surface()` 与显式的协变调用逐位相同（"翻默认"这一步
    // 没有顺手改动实现）。
    assert_eq!(CavityFrame::default(), CavityFrame::Covariant);
    let default = smd_surface(&data, &molecule, points, 0.0).expect("default");
    let explicit = smd_surface_with_frame(&data, &molecule, points, 0.0, CavityFrame::Covariant)
        .expect("covariant");
    assert_eq!(
        default.total_area_angstrom2.to_bits(),
        explicit.total_area_angstrom2.to_bits(),
        "smd_surface() must be bit-identical to the explicit covariant frame"
    );
    for (index, area) in default.areas_angstrom2.iter().enumerate() {
        assert_eq!(
            area.to_bits(),
            explicit.areas_angstrom2[index].to_bits(),
            "atom {index}: smd_surface() must be bit-identical to the explicit covariant frame"
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
    let mut global_errors = Vec::new();
    for points in [256_usize, 1024, 4096, 16384] {
        let surface = smd_surface(&data, &molecule, points, 0.0).expect("surface");
        let error = (surface.total_area_angstrom2 - analytic).abs();
        println!(
            "  N = {points:>5}: covariant area {:.9}, |error| {error:.3e} ({:.3} %)",
            surface.total_area_angstrom2,
            100.0 * error / analytic
        );
        errors.push(error);

        let global = smd_surface_with_frame(&data, &molecule, points, 0.0, CavityFrame::Global)
            .expect("global surface");
        let global_error = (global.total_area_angstrom2 - analytic).abs();
        println!(
            "  N = {points:>5}: Global    area {:.9}, |error| {global_error:.3e} ({:.3} %)",
            global.total_area_angstrom2,
            100.0 * global_error / analytic
        );
        global_errors.push(global_error);
    }
    // 准随机收敛：最密的一档必须比最疏的一档小一个量级以上（`O(1/√N)`：N 翻 64 倍
    // 期望缩 8 倍）。不要求逐档单调 —— Fibonacci 点集的误差会小幅振荡。
    println!(
        "Global error 256 -> 16384: {:.3e} -> {:.3e}",
        global_errors[0],
        global_errors[global_errors.len() - 1]
    );
    // 全局坐标架：判据**逐字保留**（第 435 轮翻默认前后都通过）。
    assert!(
        global_errors[global_errors.len() - 1] < 0.5 * global_errors[0],
        "the tessellation error must shrink with points_per_atom (Global frame): {global_errors:?}"
    );
    // 协变坐标架（默认）：N = 256 这一档**恰好**命中解析面积（实测 1.5e-13），因为键轴
    // [0.6, 0.8, 0] 正好是坐标架的 e1，而 Fibonacci 点集的 z 分档关于 0 对称 —— 那是这个
    // 点数的对称性巧合，不是收敛性，所以"最密 < 0.5 × 最疏"在这个坐标架下没有意义
    // （0 不可能更小）。收敛性仍然钉住，参照取第一个"边界已被分辨"的档（N = 1024），
    // 绝对上界（0.5 %）不变。
    println!(
        "covariant error 1024 -> 16384: {:.3e} -> {:.3e} (N = 256 hits the analytic area: {:.3e})",
        errors[1],
        errors[errors.len() - 1],
        errors[0]
    );
    assert!(
        errors.len() > 1 && errors[errors.len() - 1] < 0.5 * errors[1],
        "the tessellation error must shrink with points_per_atom (covariant frame): {errors:?}"
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
fn the_rotation_drift_of_the_surface_is_exactly_zero_in_the_default_frame() {
    // 第 435 轮起 `smd_surface()` 的默认坐标架是每原子局部坐标架（与 `cavity()` 同一个）。
    // 于是"转过一个刚性旋转"不再改点集与分子的相对取向，面积对转动**精确**不变（实测只剩
    // 加法舍入）。旧行为（点钉在全局坐标系里）的采样漂移仍然保留，但必须显式点名
    // `CavityFrame::Global` 才拿得到 —— 那条曲线是"为什么这层以前只到采样误差"的证据。
    let data = smd();
    let molecule = water();
    let width = 0.15;
    let rotated = rotate_molecule(&molecule, [0.3, -0.5, 0.8], 2.6);
    let mut drift = Vec::new();
    let mut global_drift = Vec::new();
    for points in [1024_usize, 4096, 16384] {
        let base = smd_surface(&data, &molecule, points, width).expect("surface");
        let other = smd_surface(&data, &rotated, points, width).expect("surface");
        let value = relative(other.total_area_angstrom2, base.total_area_angstrom2);
        println!("{points} points: covariant (default) rotation drift {value:.3e}");
        drift.push(value);

        let base_global =
            smd_surface_with_frame(&data, &molecule, points, width, CavityFrame::Global)
                .expect("global surface");
        let other_global =
            smd_surface_with_frame(&data, &rotated, points, width, CavityFrame::Global)
                .expect("global surface");
        let global = relative(
            other_global.total_area_angstrom2,
            base_global.total_area_angstrom2,
        );
        println!("{points} points: Global rotation drift {global:.3e}");
        global_drift.push(global);
    }
    for (points, value) in [1024_usize, 4096, 16384].iter().zip(&drift) {
        assert!(
            *value < 1.0e-12,
            "the default (covariant) frame must be exactly rotation invariant at {points} points, \
             got {value:.3e}"
        );
    }
    assert!(
        global_drift[2] < global_drift[0],
        "in the Global frame the rotation drift must shrink with more points (sampling \
         artifact): {global_drift:?}"
    );
    assert!(
        global_drift[0] < 2.0e-3,
        "at 1024 points the Global drift must stay sampling-scale, got {:.3e}",
        global_drift[0]
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

/// 一个来自 `scripts/smd_cds_legacy_reference.py` 的体系（PySCF `get_cds_legacy`）。
struct OracleSystem {
    name: String,
    solvent: String,
    atoms: Vec<(String, [f64; 3])>,
    energy_kcal: f64,
}

/// 读 `tests/reference/smd_cds_legacy.csv`（长表：一行一个原子）。
///
/// 梯度四列先不解析：CDS 的**梯度**还没实现，那几个数留给下一步的验收用。
fn read_oracle_systems() -> Vec<OracleSystem> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/smd_cds_legacy.csv");
    let text = std::fs::read_to_string(&path).expect("smd_cds_legacy.csv");
    let mut systems: Vec<OracleSystem> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 13, "line {}: expected 13 columns", index + 1);
        let name = fields[0].to_string();
        let solvent = fields[1].to_string();
        let symbol = fields[3].to_string();
        let position = [
            fields[4].parse::<f64>().expect("x"),
            fields[5].parse::<f64>().expect("y"),
            fields[6].parse::<f64>().expect("z"),
        ];
        let energy_kcal = fields[7].parse::<f64>().expect("e_cds");
        match systems.last_mut() {
            Some(last) if last.name == name && last.solvent == solvent => {
                last.atoms.push((symbol, position));
            }
            _ => systems.push(OracleSystem {
                name,
                solvent,
                atoms: vec![(symbol, position)],
                energy_kcal,
            }),
        }
    }
    assert!(
        systems.len() >= 4,
        "the oracle file must cover several systems"
    );
    systems
}

#[test]
fn the_multi_atom_cds_energy_matches_the_oracle_when_the_sigma_is_zero_order_only() {
    // 这是这一层**唯一**对上游的死差判据，而且只对"表面张力只有零阶项"的体系成立：
    // 水的 H–O、HF 的 H–F、HCl 的 H–Cl、H₂S 的 H–S 都不触发 `HSIGMA` 与成对修正
    // （`HSIGMA(9)/HSIGMA(17)/HSIGMA(16)` 全为 0，成对项只对 C/N/Si/P 邻居生效），
    // 于是 `E_CDS = Σ σ_i·A_i` 只差**表面分解**。含 C–H/C–O/C–N 的分子要等修正进数据。
    let data = smd();
    // 点数只影响**采样误差**，不影响模型：这一条比的是 `E_CDS = Σσ_i A_i` 的转写。
    // 第 435 轮翻默认（全局 → 协变坐标架）把采样噪声重新洗了一遍：8192 点/原子时旧的
    // 实测最差是 1.86e-4（H₂S，water 1.25e-4），翻完同一档变成 H₂S **1.13e-3** —— 超过
    // 1e-3 的判据（判据没动）。这是 `O(1/√N)` 的噪声，不是模型差，所以处置是**加密采样**
    // 而不是放宽判据：32768 点/原子实测最差 **5.77e-4**（水 7.37e-5、HF 4.49e-5、
    // HCl 5.90e-5、H₂S 5.77e-4），仍然对 1e-3 有 1.7 倍余量。
    let points = 32768;
    // 只取"零阶即全部"的四个体系：含 C/N 的体系在 `sv03_smd_sts.rs` 里连 σ 组装一起验。
    let zero_order = [
        "water",
        "hydrogen-fluoride",
        "hydrogen-chloride",
        "hydrogen-sulfide",
    ];
    let mut worst = 0.0_f64;
    for oracle in read_oracle_systems() {
        // 参考文件里有非水相的同一批分子（`solvent` 列不同）；这一半只跑水相。
        if oracle.solvent != "water" || !zero_order.contains(&oracle.name.as_str()) {
            continue;
        }
        let atoms = oracle
            .atoms
            .iter()
            .map(|(symbol, position)| Atom::new(symbol, *position))
            .collect();
        let molecule = Molecule::new(atoms, 0, 1);
        let symbols: Vec<&str> = molecule
            .atoms
            .iter()
            .map(|atom| atom.symbol.as_str())
            .collect();
        let surface = smd_surface(&data, &molecule, points, 0.0).expect("surface");
        let ours =
            cds_energy_kcal(&data, &symbols, &surface.areas_angstrom2, true, None).expect("cds");
        let difference = (ours - oracle.energy_kcal).abs() / oracle.energy_kcal.abs();
        println!(
            "{:20} ours {ours:.6} vs oracle {:.6} kcal/mol (rel {difference:.2e})",
            oracle.name, oracle.energy_kcal
        );
        worst = worst.max(difference);
    }
    println!("worst zero-order relative deviation: {worst:.3e}");
    assert!(
        worst < 1.0e-3,
        "the multi-atom CDS energy must reproduce the oracle to 0.1 %, got {worst:.3e}"
    );
}
