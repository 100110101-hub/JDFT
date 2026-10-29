//! SV-01 的**变分一致性**判据：SCF 里的溶剂化算子必须真的等于 `∂ΔG_el/∂P`。
//!
//! 这条测试是补的一个"签名坑"：第一版把反应场算子写成 `+V_solv`（看起来"电场在溶质上"），
//! 实际由 `ΔG = ½Vᵀq = −½ qᵀS q`（`S q = −V`）求导得到
//!
//! ```text
//! ∂ΔG/∂P_μν = −V_solv,μν                    （符号是负的，且**没有** ½）
//! ```
//!
//! 判据不靠推导、靠数值：沿 `dP = P_solv − P_vac` 方向数值求 `dΔG/dλ`，
//! 它必须等于 `Tr(operator · dP)`；而 `+V_solv` 会给出相反的符号（本测试顺带钉住这一点，
//! 保证这条判据是**有分辨力**的）。
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use qwb_basis::InternalBasis;
use qwb_engine::cosmo_scf::CosmoScfField;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_props::cavity::cavity;
use qwb_registry::Registry;
use qwb_scf::rks::{ExternalField, RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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

fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let n = a.nrows();
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += a.get(i, j) * b.get(i, j);
        }
    }
    sum
}

fn scaled_add(base: &Matrix, direction: &Matrix, factor: f64) -> Matrix {
    let n = base.nrows();
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out.set(i, j, base.get(i, j) + factor * direction.get(i, j));
        }
    }
    out
}

#[test]
fn sv01_solvation_operator_is_the_density_derivative() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule).unwrap();
    let tensor = two_electron(&basis, &molecule).unwrap();
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), &molecule).unwrap();
    let mut radii = BTreeMap::new();
    for (symbol, radius) in [("H", 1.20), ("C", 1.70), ("O", 1.52)] {
        radii.insert(symbol.to_string(), radius);
    }
    let cavity = cavity(&molecule, &radii, 1.0, 120).unwrap();
    let field =
        Arc::new(CosmoScfField::build(&basis, &molecule, &cavity.tesserae, 78.355).unwrap());

    let mut base = RksOptions::default();
    base.max_iter = 64;
    base.energy_threshold = 1.0e-10;
    base.density_threshold = 1.0e-8;
    let vacuum = rks_scf(&basis, &molecule, &tensor, &grid, &[], base.clone()).unwrap();
    let mut with_field = base;
    with_field.external_field = Some(field.handle());
    let solvated = rks_scf(&basis, &molecule, &tensor, &grid, &[], with_field).unwrap();

    // ΔG_el(P) = shift(P) + Tr(P · operator(P))
    let delta_g = |density: &Matrix| -> f64 {
        let (operator, shift) = field.operator_and_energy(density).unwrap();
        shift + trace_product(density, &operator)
    };

    let direction = scaled_add(&solvated.density, &vacuum.density, -1.0);
    let h = 1.0e-3;
    let slope_numeric = (delta_g(&scaled_add(&solvated.density, &direction, h))
        - delta_g(&scaled_add(&solvated.density, &direction, -h)))
        / (2.0 * h);
    let (operator, _shift) = field.operator_and_energy(&solvated.density).unwrap();
    let slope_operator = trace_product(&operator, &direction);

    println!(
        "E_vac = {:.10}  E_solv = {:.10}  dG = {:.10} Ha",
        vacuum.energy_hartree,
        solvated.energy_hartree,
        solvated.energy_hartree - vacuum.energy_hartree
    );
    println!("dΔG/dλ (numeric) = {slope_numeric:+.10}");
    println!("Tr(operator·dP)  = {slope_operator:+.10}");
    println!("difference       = {:+.3e}", slope_operator - slope_numeric);

    // 判据：算子就是 ΔG 对密度的导数（有限差分的截断误差 ~1e-10 量级）
    assert!(
        (slope_operator - slope_numeric).abs() < 1.0e-7,
        "the SCF operator is not d(ΔG)/dP: {slope_operator} vs {slope_numeric}"
    );
    // 有分辨力：反号算子必须**不**满足上面那条
    assert!(
        (-slope_operator - slope_numeric).abs() > 1.0e-5,
        "the sign of the operator is not discriminated by this test"
    );
    // 变分：溶剂化的解必须比真空更低（否则说明算子把密度推向了更高的能量）
    assert!(
        solvated.energy_hartree < vacuum.energy_hartree,
        "the solvated SCF must be below the vacuum SCF: {} vs {}",
        solvated.energy_hartree,
        vacuum.energy_hartree
    );
}
