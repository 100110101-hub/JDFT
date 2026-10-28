//! SV-01（第二半）验收：COSMO 表面电荷的外部判据 = **解析 Born 能**。
//!
//! 中心点电荷 + 球形空腔有解析解 `ΔG = −½(1 − 1/ε) q²/R`。这条判据来自静电学本身，
//! 不是拟合量，所以离散化误差是**实测**的：本文档记录的实测值是 2–4%（`M` = 125…2000、
//! `R` = 1.5/2.0 Å 范围），残差本身在 1e-16 量级（线性方程解对了，误差全来自
//! 1.07 圆盘自作用近似）。真机分子的 SOL 误差远小于裸点电荷（后者能量全部来自近场），
//! 但本文档**不**据此放宽别处判据 —— 见 `docs/choices.md`。
use std::collections::BTreeMap;
use std::f64::consts::PI;

use qwb_engine_api::{Atom, Molecule};
use qwb_props::cavity::{Tessera, cavity};
use qwb_props::cosmo::{
    CosmoOptions, born_solvation_energy, cosmo_surface_charges, dielectric_scale,
};

fn bohr() -> f64 {
    qwb_engine_api::molecule::BOHR_PER_ANGSTROM
}

/// 斐波那契球面上的 `count` 个均匀面积 tessera（面积取解析值 `4πR²/n`）。
fn sphere_tesserae(radius_angstrom: f64, count: usize) -> Vec<Tessera> {
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let total_area = 4.0 * PI * radius_angstrom * radius_angstrom;
    (0..count)
        .map(|k| {
            let z = 1.0 - (2.0 * k as f64 + 1.0) / count as f64;
            let radial = (1.0 - z * z).max(0.0).sqrt();
            let phi = 2.0 * PI * (k as f64 / golden).rem_euclid(1.0);
            let unit = [radial * phi.cos(), radial * phi.sin(), z];
            Tessera::new(
                [
                    unit[0] * radius_angstrom,
                    unit[1] * radius_angstrom,
                    unit[2] * radius_angstrom,
                ],
                unit,
                total_area / count as f64,
                0,
            )
        })
        .collect()
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

#[test]
fn sv01_cosmo_agrees_with_the_analytic_born_energy() {
    let radius = 2.0_f64;
    let epsilon = 78.355_f64;
    let born = born_solvation_energy(radius, 1.0, epsilon).expect("born");
    let potential = 1.0 / (radius * bohr());

    for count in [125usize, 250, 500] {
        let tesserae = sphere_tesserae(radius, count);
        let potentials = vec![potential; count];
        let result =
            cosmo_surface_charges(&tesserae, &potentials, epsilon, CosmoOptions::default())
                .expect("cosmo");
        assert!(
            result.converged,
            "count={count} residual={}",
            result.residual
        );
        assert!(
            result.residual < 1.0e-10,
            "count={count} residual={}",
            result.residual
        );

        // 高斯定律：导体球壳的总诱导电荷 = −q（电荷形式下直接是 Σq_K）
        let induced: f64 = result
            .conductor_charges
            .iter()
            .zip(&tesserae)
            .map(|(sigma, tessera)| sigma * tessera.area_angstrom2 * bohr() * bohr())
            .sum();
        let induced_error = (induced + 1.0).abs();

        let error = ((result.solvation_energy_hartree - born) / born).abs();
        println!(
            "M={count:4}  dG={:.10}  born={:.10}  rel_err={:.3e}  induced={:.10}  ind_err={:.3e}",
            result.solvation_energy_hartree, born, error, induced, induced_error
        );
        // 离散化误差的实测上界（1.07 圆盘自作用近似的近场误差，见本文档头注）
        assert!(error < 5.0e-2, "M={count} Born error {error:.3e}");
        assert!(
            induced_error < 5.0e-2,
            "M={count} induced charge error {induced_error:.3e}"
        );
    }
}

#[test]
fn sv01_cosmo_conductor_limit_and_dielectric_scaling() {
    let radius = 1.5_f64;
    let potential = 1.0 / (radius * bohr());
    let tesserae = sphere_tesserae(radius, 400);
    let potentials = vec![potential; tesserae.len()];

    // ε → ∞：C-PCM 退化为导体，ΔG → −½q²/R
    let conductor = cosmo_surface_charges(&tesserae, &potentials, 1.0e12, CosmoOptions::default())
        .expect("conductor");
    let exact = -0.5 / (radius * bohr());
    let error = ((conductor.solvation_energy_hartree - exact) / exact).abs();
    println!(
        "conductor: dG={:.10} exact={:.10} rel_err={:.3e}",
        conductor.solvation_energy_hartree, exact, error
    );
    assert!(error < 5.0e-2, "conductor limit error {error:.3e}");

    // ε = 1（真空）：电荷与能量都必须严格为 0
    let vacuum = cosmo_surface_charges(&tesserae, &potentials, 1.0, CosmoOptions::default())
        .expect("vacuum");
    assert!(vacuum.charges.iter().all(|value| value.abs() < 1.0e-30));
    assert!(vacuum.solvation_energy_hartree.abs() < 1.0e-30);
    assert!((dielectric_scale(78.355).expect("scale") - 77.355 / 78.355).abs() < 1.0e-15);
    assert!(dielectric_scale(0.5).is_err());
    assert!(dielectric_scale(f64::NAN).is_err());
}

#[test]
fn sv01_cosmo_energy_is_linear_in_the_dielectric_factor() {
    // 真机空腔（水，Bondi 半径）：解对 V 是线性的，所以 ΔG(ε) = f(ε)·K
    let molecule = water();
    let cavity = cavity(&molecule, &bondi_radii(), 1.0, 200).expect("cavity");
    assert!(cavity.tesserae.len() > 100);
    // 测试场只取**核电荷**的库仓势（没有电子屏蔽，所以结果的
    // ε量级不是物理的溶剂化能）—— 本测试只验证 ΔG 对 f(ε) 的线性。
    let mut positions = Vec::new();
    for atom in &molecule.atoms {
        positions.push(atom.position_angstrom);
    }
    let potentials: Vec<f64> = cavity
        .tesserae
        .iter()
        .map(|tessera| {
            let mut value = 0.0;
            for position in &positions {
                let dx = tessera.center_angstrom[0] - position[0];
                let dy = tessera.center_angstrom[1] - position[1];
                let dz = tessera.center_angstrom[2] - position[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt() * bohr();
                value += 1.0 / distance;
            }
            value
        })
        .collect();

    let mut reference: Option<f64> = None;
    for epsilon in [2.0_f64, 10.0, 78.355] {
        let result = cosmo_surface_charges(
            &cavity.tesserae,
            &potentials,
            epsilon,
            CosmoOptions::default(),
        )
        .expect("cosmo");
        assert!(result.converged, "epsilon={epsilon}");
        let scale = dielectric_scale(epsilon).expect("scale");
        let ratio = result.solvation_energy_hartree / scale;
        println!(
            "epsilon={epsilon:8} f={scale:.8} dG={:.10} dG/f={:.10}",
            result.solvation_energy_hartree, ratio
        );
        match reference {
            None => reference = Some(ratio),
            Some(expected) => assert!(
                ((ratio - expected) / expected).abs() < 1.0e-10,
                "epsilon={epsilon}: dG/f={ratio} vs {expected}"
            ),
        }
        // 溶剂化能必须是负的（稳定化）—— 核对上的库仑势为正，q 为负
        assert!(result.solvation_energy_hartree < 0.0);
    }
}

#[test]
fn sv01_cosmo_rejects_bad_input() {
    let tesserae = sphere_tesserae(1.0, 50);
    let potentials = vec![0.1; 50];
    assert!(
        cosmo_surface_charges(
            &tesserae,
            &potentials[..10],
            78.355,
            CosmoOptions::default()
        )
        .is_err()
    );
    assert!(cosmo_surface_charges(&[], &[], 78.355, CosmoOptions::default()).is_err());
    assert!(born_solvation_energy(0.0, 1.0, 78.355).is_err());
    assert!(born_solvation_energy(2.0, 1.0, 0.9).is_err());
    // 面积为 0 的碎片
    let degenerate = vec![Tessera::new([0.0; 3], [0.0, 0.0, 1.0], 0.0, 0)];
    assert!(cosmo_surface_charges(&degenerate, &[1.0], 78.355, CosmoOptions::default()).is_err());
}
