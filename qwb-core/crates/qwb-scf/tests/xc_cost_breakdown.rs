//! EX-03 的 XC 成本分解：把「网格 XC」拆成**三段**分别计时，给 GPU 决策提供实测数字。
//!
//! 三段（与 `qwb-scf` 的实际代码路径一一对应）：
//!
//! 1. `GridAoCached::build`：逐点求 `φ`、`∇φ`（`laplacian` 按需求）。**与密度无关**，
//!    SCF 各步复用，所以它只付一次；
//! 2. `evaluate_xc_cached`：逐点做 `Σ_μν P_μν φ_μ φ_ν` 这类二次缩并 + 泛函求值 —— 每步都付；
//! 3. 只算泛函（`energy_density` + `potential_derivatives`）：把缩并的结果当输入，
//!    看泛函本身占多少。
//!
//! 这三段的比例决定 GPU 值不值得上（见 `docs/choices.md` 九十/九十一/九十二的 J 分析）：
//! 缩并是 `O(n_points·n²)` 的稠密乘法（形状像 gemm），泛函是逐点的超越函数（形状像
//! elementwise kernel），AO 求值是分支密集的小循环（GPU 上最差）。
use std::path::PathBuf;
use std::time::Instant;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{GridAoCached, evaluate_xc_cached};
use qwb_xc::{DensityBatch, FunctionalTerm};

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

fn terms_from(registry: &Registry, name: &str) -> Vec<Box<dyn FunctionalTerm>> {
    let functional = registry.get_functional(name).expect(name);
    let mut parameters = std::collections::BTreeMap::new();
    for (key, value) in functional.parameters.iter() {
        parameters.insert(key.clone(), *value);
    }
    let mut terms: Vec<Box<dyn FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        // 精确交换不进 XC 泛函求值（它在 Fock 矩阵里），所以跳过。
        if component.weight.abs() <= 1e-12 || component.id == "HfExchange" {
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    terms
}

#[test]
fn xc_cost_breakdown() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    for (basis_name, grid_name) in [("sto-3g", "grid-normal"), ("def2-svp", "grid-normal")] {
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis(basis_name).expect(basis_name),
            &molecule,
        )
        .expect("basis");
        let grid =
            qwb_grid::molecular_grid(registry.get_grid(grid_name).expect(grid_name), &molecule)
                .expect("grid");
        let n = basis.n_functions();
        let mut density = qwb_linalg::Matrix::zeros(n, n);
        for i in 0..n {
            density.set(i, i, 1.0);
        }
        for functional_name in ["svwn5", "pbe", "m06-2x"] {
            let terms = terms_from(&registry, functional_name);
            let needs_sigma = terms.iter().any(|term| term.requires().sigma);
            let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);

            let started = Instant::now();
            let cache = GridAoCached::build(&basis, &grid, needs_sigma, false, needs_tau)
                .expect("AO cache");
            let ao_build = started.elapsed();

            // 缩并 + 泛函（SCF 每步都付的部分），重复几次取最快。
            let mut xc_time = f64::INFINITY;
            for _ in 0..3 {
                let started = Instant::now();
                let _ = evaluate_xc_cached(&basis, &grid, &cache, &density, &terms)
                    .expect("evaluate_xc_cached");
                xc_time = xc_time.min(started.elapsed().as_secs_f64());
            }

            // 只算泛函：准备一批平凡的 (ρ, σ, τ) 输入。
            let points = grid.len();
            let rho: Vec<f64> = (0..points).map(|i| 0.1 + 0.001 * (i % 17) as f64).collect();
            let sigma: Vec<f64> = (0..points)
                .map(|i| 0.01 + 0.0001 * (i % 13) as f64)
                .collect();
            let tau: Vec<f64> = (0..points)
                .map(|i| 0.05 + 0.0005 * (i % 11) as f64)
                .collect();
            let batch = if needs_tau {
                DensityBatch::with_tau(&rho, &sigma, &tau)
            } else if needs_sigma {
                DensityBatch::new(&rho, &sigma)
            } else {
                DensityBatch::from_rho(&rho)
            };
            let mut epsilon = vec![0.0; points];
            let mut d_rho = vec![0.0; points];
            let mut d_sigma = vec![0.0; points];
            let mut functional_time = f64::INFINITY;
            for _ in 0..3 {
                let started = Instant::now();
                for term in &terms {
                    term.energy_density(&batch, &mut epsilon);
                    term.potential_derivatives(&batch, &mut d_rho, &mut d_sigma);
                }
                functional_time = functional_time.min(started.elapsed().as_secs_f64());
            }
            println!(
                "{basis_name}/{grid_name}/{functional_name}: n = {n}, points = {points} | \
                 AO cache {:.3} s (once) | contract+functional {:.3} s (per iteration) | \
                 functional only {:.3} s ({:.0}% of the per-iteration part)",
                ao_build.as_secs_f64(),
                xc_time,
                functional_time,
                100.0 * functional_time / xc_time.max(1e-12),
            );
            assert!(xc_time > 0.0 && functional_time > 0.0);
        }
    }
}
