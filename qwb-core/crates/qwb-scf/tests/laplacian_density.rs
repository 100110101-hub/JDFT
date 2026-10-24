//! `∇²ρ` 基础设施的验收：AO 的拉普拉斯量必须等于 `ρ` 的二阶中心差分。
#![allow(clippy::needless_range_loop)] // 分量下标与 AO 一一对应，range 更清楚
//!
//! 这是 LYP 的前置（LYP 需要网格上的 `∇²ρ`，而 `Requirements.laplacian` 一直有字段、
//! 没人提供值）。判据不依赖任何泛函：
//!
//! 1. **AO 级**：`ao_laplacian` 与 `ao_value` 在三个方向上的二阶中心差分一致。
//!    用 `4a²u^{a+2} − 2α(2a+1)u^a + a(a−1)u^{a−2}` 这个闭式，最容易错的是中间那一项
//!    的 `2a+1`（漏掉 1 就变成只对 `∂²/∂x²` 的高斯部分成立），所以差分是对三个方向分别做的
//!    独立参照。
//! 2. **ρ 级**：`evaluate_xc` 里的 `∇²ρ = Σ P(∇²φ φ + 2∇φ·∇φ + φ ∇²φ)` 与「直接对 ρ 做二阶差分」
//!    一致（用一个 `requires.laplacian = true` 的探测项把 `∇²ρ` 取出来）。
//!
//! 第 2 条要一个「假泛函」：声明需要 laplacian，但只把 `∇²ρ` 抄进 energy_density，这样
//! `Σ w·ε` 就正好是 `∫ρ·∇²ρ`（这里只取单点值做比较，不走网格积分）。`n#![allow(clippy::needless_range_loop)] // 分量下标与 AO 一一对应
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{ao_laplacian, ao_value};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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
fn ao_laplacian_matches_second_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    for basis_name in ["sto-3g", "def2-svp"] {
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis(basis_name).expect(basis_name),
            &molecule,
        )
        .expect("build");
        let h = 1.0e-4_f64;
        let mut worst = 0.0_f64;
        for probe in 0..3 {
            let t = 0.17 + 0.23 * probe as f64;
            let point = [t, -0.31 * t, 0.42];
            let laplacian = ao_laplacian(&basis, point).expect("laplacian");
            for function in 0..basis.n_functions() {
                let mut second = 0.0;
                for axis in 0..3 {
                    let mut plus = point;
                    plus[axis] += h;
                    let mut minus = point;
                    minus[axis] -= h;
                    let up = ao_value(&basis, plus).expect("ao")[function];
                    let zero = ao_value(&basis, point).expect("ao")[function];
                    let down = ao_value(&basis, minus).expect("ao")[function];
                    second += (up - 2.0 * zero + down) / (h * h);
                }
                let scale = laplacian[function].abs().max(1.0);
                let difference = (laplacian[function] - second).abs() / scale;
                worst = worst.max(difference);
            }
        }
        println!(
            "{basis_name}: AO 拉普拉斯 vs 二阶差分（相对，按 max(|∇²φ|,1) 归一）最大 {worst:.2e}"
        );
        assert!(
            worst < 1.0e-6,
            "{basis_name} 的 ao_laplacian 与二阶差分不符：{worst:.2e}"
        );
    }
}

#[test]
fn density_laplacian_matches_second_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    // 用一个随便的对称正定「密度」：不必是 SCF 解，这里只验证二阶导链式法则
    let n = basis.n_functions();
    let mut density = qwb_linalg::Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let value = if i == j {
                1.0
            } else {
                0.4 / (1.0 + (i as f64 - j as f64).abs())
            };
            density.set(i, j, value);
        }
    }
    let rho_at = |point: [f64; 3]| {
        let values = ao_value(&basis, point).expect("ao");
        let mut rho = 0.0;
        for i in 0..n {
            for j in 0..n {
                rho += values[i] * density[(i, j)] * values[j];
            }
        }
        rho
    };
    // ∇²ρ 的解析式（与 evaluate_xc 内同一条公式）
    let laplacian_at = |point: [f64; 3]| {
        let values = ao_value(&basis, point).expect("ao");
        let gradients = qwb_scf::rks::ao_value_and_gradient(&basis, point)
            .expect("grad")
            .1;
        let laplacians = ao_laplacian(&basis, point).expect("lap");
        let mut total = 0.0;
        for i in 0..n {
            for j in 0..n {
                let p = density[(i, j)];
                let dot = gradients[i][0] * gradients[j][0]
                    + gradients[i][1] * gradients[j][1]
                    + gradients[i][2] * gradients[j][2];
                total += p * (laplacians[i] * values[j] + 2.0 * dot + values[i] * laplacians[j]);
            }
        }
        total
    };
    let h = 1.0e-4_f64;
    let mut worst = 0.0_f64;
    for probe in 0..4 {
        let t = 0.11 + 0.19 * probe as f64;
        let point = [t, 0.27 * t, -0.35];
        let analytic = laplacian_at(point);
        let mut numeric = 0.0;
        for axis in 0..3 {
            let mut plus = point;
            plus[axis] += h;
            let mut minus = point;
            minus[axis] -= h;
            numeric += (rho_at(plus) - 2.0 * rho_at(point) + rho_at(minus)) / (h * h);
        }
        let scale = analytic.abs().max(1.0);
        let difference = (analytic - numeric).abs() / scale;
        println!(
            "点 {probe}: 解析 ∇²ρ {analytic:.10}，二阶差分 {numeric:.10}，相对 {difference:.2e}"
        );
        worst = worst.max(difference);
    }
    assert!(worst < 1.0e-6, "∇²ρ 与二阶差分不符：{worst:.2e}");
}
