//! X-03 验收（project.md §6.4）：角向 `Σw = 4π`、径向准确、Becke 分区、UEG 极限。

use std::path::{Path, PathBuf};

use qwb_engine_api::{Atom, Molecule};
use qwb_grid::{angular_rule, atomic_grid, molecular_grid, radial_rule};
use qwb_registry::Registry;
use qwb_registry::grid::{AngularRule, GridLevelName};

const PI: f64 = std::f64::consts::PI;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn level(level: GridLevelName) -> qwb_registry::grid::GridLevelDef {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");
    registry.get_grid_by_level(level).expect("level").clone()
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    )
}

#[test]
fn angular_weights_sum_to_four_pi() {
    for name in [
        GridLevelName::Coarse,
        GridLevelName::Normal,
        GridLevelName::Fine,
        GridLevelName::VeryFine,
    ] {
        let definition = level(name);
        assert_eq!(definition.angular_rule, AngularRule::GaussProduct);
        let (directions, weights) = angular_rule(&definition).expect("angular rule");
        assert_eq!(directions.len(), weights.len());
        let total: f64 = weights.iter().sum();
        assert!(
            (total - 4.0 * PI).abs() < 1e-10,
            "{name:?}: Σw = {total} vs 4π = {}",
            4.0 * PI
        );
        // 方向必须是单位向量。
        for direction in &directions {
            let norm = (direction[0] * direction[0]
                + direction[1] * direction[1]
                + direction[2] * direction[2])
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "方向不是单位向量: {norm}");
        }
    }
}

#[test]
fn lebedev_is_refused_instead_of_substituted() {
    // 数据若声明 lebedev，必须显式报错（仓库里没有点集表），不能悄悄换成乘积规则。
    let mut definition = level(GridLevelName::Normal);
    definition.angular_rule = AngularRule::Lebedev;
    let error = angular_rule(&definition).expect_err("lebedev 表缺失时必须报错");
    let message = error.to_string();
    assert!(message.contains("lebedev"), "{message}");
    assert!(
        message.contains("gauss-product"),
        "错误信息应给出替代方案: {message}"
    );
}

#[test]
fn radial_rule_integrates_a_gaussian_accurately() {
    // ∫₀^∞ r² e^{-r²} dr = √π/4（径向权重不含 4π）
    let definition = level(GridLevelName::VeryFine);
    let (radii, weights) = radial_rule(&definition).expect("radial");
    assert_eq!(radii.len(), definition.radial_points as usize);
    let total: f64 = radii
        .iter()
        .zip(&weights)
        .map(|(r, w)| w * (-r * r).exp())
        .sum();
    let expected = PI.sqrt() / 4.0;
    let relative = (total - expected).abs() / expected;
    assert!(
        relative < 1e-6,
        "径向求积 ∫r²e^{{-r²}} = {total} vs {expected}（相对差 {relative:e}）"
    );
}

#[test]
fn atomic_grid_has_radial_times_angular_points() {
    let definition = level(GridLevelName::Normal);
    let grid = atomic_grid(&definition, [0.0, 0.0, 0.0]).expect("atomic grid");
    let (radii, radial_weights) = radial_rule(&definition).expect("radial");
    let (directions, angular_weights) = angular_rule(&definition).expect("angular");

    assert_eq!(
        grid.len(),
        radii.len() * directions.len(),
        "点数 = 径向 × 角向"
    );
    // 总权重 = Σ_radial w · Σ_angular w = Σ_radial w · 4π
    let expected = radial_weights.iter().sum::<f64>() * angular_weights.iter().sum::<f64>();
    assert!(
        (grid.weight_sum() - expected).abs() < 1e-10 * expected.abs(),
        "Σw = {} vs Σ_radial·Σ_angular = {expected}",
        grid.weight_sum()
    );
}
#[test]
fn becke_partition_sums_to_one_at_every_point() {
    // 分子的总权重 = Σ_atoms (原子网格权重 × 分区权重)。分区权重每点归一，
    // 所以对**单原子分子**，分子网格权重必须与原子网格权重逐位相同。
    let definition = level(GridLevelName::Coarse);
    let single = Molecule::new(vec![Atom::new("O", [0.0, 0.0, 0.0])], 0, 1);
    let atomic = atomic_grid(&definition, [0.0, 0.0, 0.0]).expect("atomic");
    let molecular = molecular_grid(&definition, &single).expect("molecular");
    assert_eq!(molecular.len(), atomic.len());
    for (a, b) in molecular.points.iter().zip(&atomic.points) {
        assert!(
            (a.weight - b.weight).abs() < 1e-12,
            "单原子时分区权重必须为 1：{} vs {}",
            a.weight,
            b.weight
        );
    }
}

#[test]
fn uniform_electron_gas_limit_passes() {
    // 均匀电子气：ρ 恒定 ⇒ 网格加权平均的 ε(ρ) 必须等于解析 ε(ρ)（体积约掉）。
    use qwb_xc::{FunctionalTerm, LdaSlaterX};
    let definition = level(GridLevelName::Normal);
    let molecule = water();
    let grid = molecular_grid(&definition, &molecule).expect("grid");
    let term = LdaSlaterX::new(2.0 / 3.0);

    let rho = vec![0.5_f64; grid.len()];
    let mut energy_density = vec![0.0; grid.len()];
    term.energy_density(&qwb_xc::DensityBatch::from_rho(&rho), &mut energy_density);

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (point, epsilon) in grid.points.iter().zip(&energy_density) {
        numerator += point.weight * rho[0] * epsilon;
        denominator += point.weight * rho[0];
    }
    let average = numerator / denominator;
    let analytic = -0.75 * (3.0 / PI).cbrt() * rho[0].cbrt();
    assert!(
        (average - analytic).abs() < 1e-10,
        "UEG 极限：加权平均 {average} vs 解析 {analytic}"
    );
}

/// **回归测试**：径向映射 `r = a t²/(1−t²)` 在 t→1 时发散，曾产生 inf 坐标，
/// 而 `inf × 0`（方向分量为 0）会变成 NaN，污染密度与积分。
/// 现在 t 被夹到 1−1e-3，所有坐标与权重必须有限。
#[test]
fn all_grid_coordinates_and_weights_are_finite() {
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    );
    for name in [
        GridLevelName::Coarse,
        GridLevelName::Normal,
        GridLevelName::Fine,
        GridLevelName::VeryFine,
    ] {
        let definition = level(name);
        let grid = molecular_grid(&definition, &molecule).expect("grid");
        for point in &grid.points {
            assert!(
                point.position_bohr.iter().all(|v| v.is_finite()),
                "{name:?}: 非有限坐标 {:?}",
                point.position_bohr
            );
            assert!(point.weight.is_finite(), "{name:?}: 非有限权重");
        }
        // 最大半径必须有限（t 夹紧的直接后果）。
        let max_radius = grid
            .points
            .iter()
            .map(|p| {
                (p.position_bohr[0].powi(2)
                    + p.position_bohr[1].powi(2)
                    + p.position_bohr[2].powi(2))
                .sqrt()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_radius.is_finite() && max_radius > 1.0,
            "{name:?}: max_r = {max_radius}"
        );
    }
}
