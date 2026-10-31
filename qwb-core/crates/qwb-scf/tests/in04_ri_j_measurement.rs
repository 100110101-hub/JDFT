//! IN-04 的**测量关**：RI-J 与 4 索引 J 在同值、耗时上的对照。
//!
//! `#[ignore]`：这是个测量工具，不是门禁用例（门禁不因它变慢）。手动跑：
//!
//! ```text
//! cargo test -p qwb-scf --test in04_ri_j_measurement -- --ignored --nocapture
//! ```
//!
//! K10 三条件在这条路径上的对应：①同值（相对差有界）；②更快（实测）；③可复现（确定性，
//! 后续接进 Fock 时用 `DeterministicReduce`）。IN-04 另外要求「大体系墙钟优势 ≥ 3×」。
use std::time::Instant;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::ri::{inverse_sqrt, three_center, two_center_coulomb};
use qwb_integrals::two_electron::two_electron;
use qwb_registry::Registry;

fn data_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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

/// 苯（def2-SVP 下规模明显大于水，用来衡量 O(n^4) 与 O(n^2·n_aux) 的分化）。
fn benzene() -> Molecule {
    let c = 1.397_f64;
    let h = 2.481_f64;
    let mut atoms = Vec::new();
    for k in 0..6 {
        let angle = std::f64::consts::PI * 2.0 * k as f64 / 6.0;
        atoms.push(Atom::new("C", [c * angle.cos(), c * angle.sin(), 0.0]));
    }
    for k in 0..6 {
        let angle = std::f64::consts::PI * 2.0 * k as f64 / 6.0;
        atoms.push(Atom::new("H", [h * angle.cos(), h * angle.sin(), 0.0]));
    }
    Molecule::new(atoms, 0, 1)
}

fn pseudo_density(n: usize) -> qwb_linalg::Matrix {
    let mut seed = 1_234_u64;
    let mut density = qwb_linalg::Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            density.set(i, j, ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5);
        }
    }
    density
}

#[test]
#[ignore = "测量工具：RI-J 与 4 索引 J 的同值与耗时对照"]
fn ri_j_versus_exact_j() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    for (label, molecule) in [("water", water()), ("benzene", benzene())] {
        let basis = InternalBasis::build(registry.get_basis("def2-svp").unwrap(), &molecule).unwrap();
        let aux = InternalBasis::build(
            registry.get_basis("def2-svp-ri").unwrap(),
            &molecule,
        )
        .unwrap();
        let n = basis.n_functions();
        let n_aux = aux.n_functions();
        let density = pseudo_density(n);

        // 4 索引 J（现路径）
        let started = Instant::now();
        let tensor = two_electron(&basis, &molecule).unwrap();
        let mut j_exact = qwb_linalg::Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0_f64;
                for k in 0..n {
                    for l in 0..n {
                        sum += tensor.get(i, j, k, l) * density[(k, l)];
                    }
                }
                j_exact.set(i, j, sum);
            }
        }
        let exact_seconds = started.elapsed().as_secs_f64();

        // RI-J
        let started = Instant::now();
        let b = three_center(&basis, &aux, &molecule).unwrap();
        let metric = two_center_coulomb(&aux, &molecule).unwrap();
        let m_inv_sqrt = inverse_sqrt(&metric, 1.0e-10).unwrap();
        // 标准 RI：B' = M^(-1/2)·B（在辅助指标上），然后 J = B'·(B'ᵀ·P)。
        // 第一版只乘了一次 M^(-1/2)，相当于少了一次度量逆 —— 差值 12–20 倍就是这么来的。
        let mut b_fit = vec![0.0_f64; n * n * n_aux];
        for i in 0..n {
            for j in 0..n {
                for p in 0..n_aux {
                    let mut sum = 0.0_f64;
                    for q in 0..n_aux {
                        sum += m_inv_sqrt[(p, q)] * b.get(i, j, q);
                    }
                    b_fit[(i * n + j) * n_aux + p] = sum;
                }
            }
        }
        // t[P] = Σ_(λσ) B'[(λσ),P] P_λσ
        let mut t = vec![0.0_f64; n_aux];
        for p in 0..n_aux {
            let mut sum = 0.0_f64;
            for lambda in 0..n {
                for sigma in 0..n {
                    sum += b_fit[(lambda * n + sigma) * n_aux + p] * density[(lambda, sigma)];
                }
            }
            t[p] = sum;
        }
        let mut j_ri = qwb_linalg::Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0_f64;
                for p in 0..n_aux {
                    sum += b_fit[(i * n + j) * n_aux + p] * t[p];
                }
                j_ri.set(i, j, sum);
            }
        }
        let ri_seconds = started.elapsed().as_secs_f64();

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                scale = scale.max(j_exact[(i, j)].abs());
                worst = worst.max((j_exact[(i, j)] - j_ri[(i, j)]).abs());
            }
        }
        println!(
            "{label}: n={n}, n_aux={n_aux}  4-index J {exact_seconds:.3} s, RI-J {ri_seconds:.3} s, \
加速 {:.2}x, 最大绝对差 {worst:.3e}（相对 {:.3e}）",
            exact_seconds / ri_seconds,
            worst / scale.max(1e-30)
        );
    }
}
