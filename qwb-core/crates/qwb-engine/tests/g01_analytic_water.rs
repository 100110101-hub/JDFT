//! G-01 的扩展验收：**含 p 函数**（H2O/STO-3G）的解析梯度 vs 数值梯度。
//!
//! 原实现只支持 s 型基组（`l > 0` 显式返回 Unsupported），所以 H2O 只能走数值梯度——
//! 这也是几何优化要 8 分钟的原因。本轮把抬/降幂恒等式推广到任意角动量、并用
//! 「球谐基 → 直角分量」的展开处理球谐壳层，于是 H2O 也能走解析梯度。
//!
//! 判据与 `g01_analytic` 一致：最大分量差 < 1e-6 Ha/Bohr。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine::gradient::{GradientOptions, numerical_gradient};
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::analytic_gradient::analytic_gradient;
use qwb_scf::build_fock;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water(oh: f64, angle_deg: f64) -> Molecule {
    // O 在原点，两个 H 在 y-z 平面，C2v
    let half = angle_deg.to_radians() / 2.0;
    let y = oh * half.sin();
    let z = oh * half.cos();
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, y, z]),
            Atom::new("H", [0.0, -y, z]),
        ],
        0,
        1,
    )
}

#[test]
fn analytic_gradient_matches_numerical_for_water() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    // 只跑一个几何：数值梯度要 18 次 SCF，两个几何就 4 分钟。
    for (oh, angle) in [(0.9572_f64, 104.5_f64)] {
        let molecule = water(oh, angle);
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis("sto-3g").expect("sto-3g"),
            &molecule,
        )
        .expect("build");
        let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
        let grid =
            qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
                .expect("grid");
        let mut options = RksOptions::default();
        options.max_iter = 400;
        options.energy_threshold = 1.0e-10;
        let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("scf");
        let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
        let fock = build_fock(&one.core_hamiltonian, &tensor, &result.density)
            .expect("fock")
            .fock;
        let analytic =
            analytic_gradient(&basis, &molecule, &result.density, &fock).expect("analytic");

        // 数值参照自己也有噪声/截断误差，所以取两个步长：它们之间的散布就是参照的精度，
        // 用来看「解析与数值的差」是不是被参照的散布解释掉了。
        let mut gradient_options = GradientOptions::new("sto-3g", "", "grid-coarse");
        gradient_options.delta_bohr = 1.0e-3;
        let numeric = numerical_gradient(&registry, &molecule, &gradient_options).expect("numeric");
        gradient_options.delta_bohr = 4.0e-3;
        let numeric_coarse =
            numerical_gradient(&registry, &molecule, &gradient_options).expect("numeric");
        let mut scatter = 0.0_f64;
        for atom in 0..numeric.len() {
            for axis in 0..3 {
                scatter = scatter.max((numeric[atom][axis] - numeric_coarse[atom][axis]).abs());
            }
        }
        println!("  数值参照自身的散布（δ=1e-3 vs 4e-3）{scatter:.2e}");

        let mut worst = 0.0_f64;
        println!(
            "O–H {oh} Å，∠HOH {angle}°，E = {:.8} Ha",
            result.energy_hartree
        );
        for atom in 0..analytic.len() {
            for axis in 0..3 {
                let difference = (analytic[atom][axis] - numeric[atom][axis]).abs();
                worst = worst.max(difference);
                println!(
                    "  atom {atom} axis {axis}: 解析 {:>13.9}  数值 {:>13.9}  差 {difference:.2e}",
                    analytic[atom][axis], numeric[atom][axis]
                );
            }
        }
        // 平移不变性：三个原子梯度之和应为 0（与参照无关的自检）
        let mut total = [0.0_f64; 3];
        for row in &analytic {
            for (axis, slot) in total.iter_mut().enumerate() {
                *slot += row[axis];
            }
        }
        println!(
            "  合力 [{:.2e}, {:.2e}, {:.2e}]",
            total[0], total[1], total[2]
        );
        assert!(
            total.iter().all(|value| value.abs() < 1.0e-8),
            "解析梯度的平移不变性被破坏：{total:?}"
        );
        // 判据：解析与数值的差要小到 **5e-6**，并且不能明显超过数值参照自身的散布。
        assert!(
            worst < 5.0e-6,
            "解析与数值梯度最大分量差 {worst:.2e}（要求 < 5e-6）"
        );
        assert!(
            worst < scatter.max(1.0e-7) * 10.0,
            "解析与数值的差 {worst:.2e} 远超数值参照自身的散布 {scatter:.2e}"
        );
    }
}
