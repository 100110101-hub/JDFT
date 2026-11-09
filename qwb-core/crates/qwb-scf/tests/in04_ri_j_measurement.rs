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
use qwb_integrals::ri::{
    inverse_sqrt, three_center, three_center_with_threads, two_center_coulomb,
    two_center_coulomb_with_threads, whiten_three_center_with_threads,
};
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
        let basis =
            InternalBasis::build(registry.get_basis("def2-svp").unwrap(), &molecule).unwrap();
        let aux = InternalBasis::build(
            registry.get_basis("def2-universal-jkfit").unwrap(),
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
        // 到这里为止都是 RI 的**共享开销**（三中心 + 度量 + 逆平方根 + B' 的改造）；
        // 它计入任何 RI-J 变体，不能只算两个 gemm —— 否则加速比是假的。
        let setup_seconds = started.elapsed().as_secs_f64();
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
        let naive_assembly_seconds = started.elapsed().as_secs_f64();

        // 第三种：用库的 matmul（BLAS 式）做同样的两次收缩，看是否值得为它重构。
        let started = Instant::now();
        // t = B'ᵀ · P  —— B' 是 (n² × n_aux)，转成 (n_aux × n²) 再乘 (n² × 1)
        let mut b_fit_t = qwb_linalg::Matrix::zeros(n_aux, n * n);
        for row in 0..n * n {
            for p in 0..n_aux {
                b_fit_t.set(p, row, b_fit[row * n_aux + p]);
            }
        }
        let mut flat_density = qwb_linalg::Matrix::zeros(n * n, 1);
        for row in 0..n * n {
            flat_density.set(row, 0, density[(row / n, row % n)]);
        }
        let t_vec = qwb_linalg::matmul(&b_fit_t, &flat_density).unwrap();
        let mut b_fit_m = qwb_linalg::Matrix::zeros(n * n, n_aux);
        for row in 0..n * n {
            for p in 0..n_aux {
                b_fit_m.set(row, p, b_fit[row * n_aux + p]);
            }
        }
        let j_vec = qwb_linalg::matmul(&b_fit_m, &t_vec).unwrap();
        let mut j_matmul = qwb_linalg::Matrix::zeros(n, n);
        for row in 0..n * n {
            j_matmul.set(row / n, row % n, j_vec[(row, 0)]);
        }
        let matmul_seconds = started.elapsed().as_secs_f64();
        let mut worst_matmul = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                worst_matmul = worst_matmul.max((j_ri[(i, j)] - j_matmul[(i, j)]).abs());
            }
        }

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                scale = scale.max(j_exact[(i, j)].abs());
                worst = worst.max((j_exact[(i, j)] - j_ri[(i, j)]).abs());
            }
        }
        println!(
            "{label}: n={n}, n_aux={n_aux}\n  4-index J {:.3} s\n  RI setup (3c+metric+B') {:.3} s\n  \
assembly naive {:.3} s -> full {:.3} s ({:.2}x)\n  assembly matmul {:.3} s -> full {:.3} s ({:.2}x)\n  \
相对差 {:.2e} | matmul-vs-naive {:.2e}",
            exact_seconds,
            setup_seconds,
            naive_assembly_seconds - setup_seconds,
            naive_assembly_seconds,
            exact_seconds / naive_assembly_seconds,
            matmul_seconds,
            setup_seconds + matmul_seconds,
            exact_seconds / (setup_seconds + matmul_seconds),
            worst / scale.max(1e-30),
            worst_matmul
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// IN-04 第二轮：**分阶段**墙钟 + 逐位确定性（线程数无关）。
//
// 上一轮的结论是「装配只占 0.1%，瓶颈是 setup」，但 setup 内部（三中心 / 度量 / `B'` 变换）
// 三段没有拆开量过。这一版把三段分别计时，并且用一个 `threads` 参数把「单线程 vs 默认线程数」
// 的两条路径放在**同一个二进制、同一个进程**里比对：`b_fit` 与 `J` 的元素必须 `to_bits()` 相等。
// ─────────────────────────────────────────────────────────────────────────────────────────

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn min_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

/// RI setup 的**分阶段**耗时（秒）。
struct Setup {
    b_fit: Vec<f64>,
    total: f64,
    three_center: f64,
    metric: f64,
    inverse_sqrt: f64,
    b_prime: f64,
}

/// RI-J 的共享 setup：3 中心 `(ij|P)` → 2 中心 `(P|Q)` → `M^{-1/2}` → `B' = M^{-1/2}·B`
/// （`B'` 行主序 `(μν) × P`，与 `qwb-gpu/tests/ex03_ri_j_pipeline_measurement.rs` 同一形状）。
///
/// `threads = 0` 用产品默认（rayon 默认线程数），`1` 是单线程 —— 两条路径的结果必须逐位相同。
fn ri_setup(
    basis: &InternalBasis,
    aux: &InternalBasis,
    molecule: &Molecule,
    threads: usize,
) -> Setup {
    let started = Instant::now();
    let b = three_center_with_threads(basis, aux, molecule, threads).expect("three_center");
    let three_center_seconds = started.elapsed().as_secs_f64();

    let metric_started = Instant::now();
    let metric =
        two_center_coulomb_with_threads(aux, molecule, threads).expect("two_center_coulomb");
    let metric_seconds = metric_started.elapsed().as_secs_f64();

    let inverse_started = Instant::now();
    let m_inv_sqrt = inverse_sqrt(&metric, 1.0e-10).expect("inverse_sqrt");
    let inverse_seconds = inverse_started.elapsed().as_secs_f64();

    let b_prime_started = Instant::now();
    let b_fit = whiten_three_center_with_threads(&b, &m_inv_sqrt, threads).expect("whiten");
    Setup {
        b_fit,
        total: started.elapsed().as_secs_f64(),
        three_center: three_center_seconds,
        metric: metric_seconds,
        inverse_sqrt: inverse_seconds,
        b_prime: b_prime_started.elapsed().as_secs_f64(),
    }
}

/// RI-J 的收缩：`t_P = Σ_{λσ} B'_{(λσ)P} P_{λσ}`、`J_{μν} = Σ_P B'_{(μν)P} t_P`。
///
/// 朴素循环（不依赖 BLAS）：上面已量过装配只占全管线约 0.1%，这里只想把它的**绝对值**跟着
/// setup 一起报出来，免得「setup 变快以后收缩变成新瓶颈」这种说法没有数字。
fn ri_contraction(
    n: usize,
    n_aux: usize,
    b_fit: &[f64],
    density: &qwb_linalg::Matrix,
) -> qwb_linalg::Matrix {
    let mut t = vec![0.0_f64; n_aux];
    for (p, slot) in t.iter_mut().enumerate() {
        let mut sum = 0.0_f64;
        for lambda in 0..n {
            for sigma in 0..n {
                sum += b_fit[(lambda * n + sigma) * n_aux + p] * density[(lambda, sigma)];
            }
        }
        *slot = sum;
    }
    let mut j = qwb_linalg::Matrix::zeros(n, n);
    for i in 0..n {
        for jj in 0..n {
            let mut sum = 0.0_f64;
            for p in 0..n_aux {
                sum += b_fit[(i * n + jj) * n_aux + p] * t[p];
            }
            j.set(i, jj, sum);
        }
    }
    j
}

/// 两个 `f64` 切片的最大绝对差与最大的逐位差（后者用 `to_bits()` 比较，出现就是不确定性）。
fn worst_difference(left: &[f64], right: &[f64]) -> (f64, usize) {
    let mut worst = 0.0_f64;
    let mut bits_differ = 0usize;
    for (a, b) in left.iter().zip(right) {
        worst = worst.max((a - b).abs());
        if a.to_bits() != b.to_bits() {
            bits_differ += 1;
        }
    }
    (worst, bits_differ)
}

/// 4 索引 J（张量 + 朴素 `n⁴` 收缩）。`threads = 0` 是产品默认（rayon 默认线程数）。
fn exact_j(
    basis: &InternalBasis,
    molecule: &Molecule,
    n: usize,
    density: &qwb_linalg::Matrix,
    threads: usize,
) -> (Vec<f64>, f64, f64) {
    let started = Instant::now();
    let tensor = qwb_integrals::two_electron::two_electron_with_threads(basis, molecule, threads)
        .expect("two_electron");
    let tensor_seconds = started.elapsed().as_secs_f64();
    let mut j = vec![0.0_f64; n * n];
    for i in 0..n {
        for jj in 0..n {
            let mut sum = 0.0_f64;
            for k in 0..n {
                for l in 0..n {
                    sum += tensor.get(i, jj, k, l) * density[(k, l)];
                }
            }
            j[i * n + jj] = sum;
        }
    }
    (j, started.elapsed().as_secs_f64(), tensor_seconds)
}

fn measure_stages(label: &str, registry: &Registry, molecule: &Molecule, reps: usize) {
    let basis =
        InternalBasis::build(registry.get_basis("def2-svp").unwrap(), molecule).expect("basis");
    let aux = InternalBasis::build(
        registry.get_basis("def2-universal-jkfit").unwrap(),
        molecule,
    )
    .expect("aux basis");
    let n = basis.n_functions();
    let n_aux = aux.n_functions();
    let density = pseudo_density(n);

    let mut exact_runs = Vec::new();
    let mut exact_tensor_runs = Vec::new();
    // 一个二进制、一个进程里同时量两条线程路径：机器漂移（热/其它负载）对两边的权重相同，
    // 所以「串行 setup vs 并行 setup」的比值不是跨进程比出来的。
    let mut setup_runs = Vec::new();
    let mut three_center_runs = Vec::new();
    let mut metric_runs = Vec::new();
    let mut inverse_runs = Vec::new();
    let mut b_prime_runs = Vec::new();
    let mut serial_setup_runs = Vec::new();
    let mut serial_three_center_runs = Vec::new();
    let mut serial_metric_runs = Vec::new();
    let mut serial_inverse_runs = Vec::new();
    let mut serial_b_prime_runs = Vec::new();
    let mut contraction_runs = Vec::new();
    let mut j_exact: Vec<f64> = Vec::new();
    let mut j_ri: Vec<f64> = vec![0.0_f64; n * n];
    let mut b_fit_parallel: Vec<f64> = Vec::new();
    let mut b_fit_serial: Vec<f64> = Vec::new();

    for run in 0..=reps {
        let (j, seconds, tensor_seconds) = exact_j(&basis, molecule, n, &density, 0);
        let setup = ri_setup(&basis, &aux, molecule, 0);
        let started = Instant::now();
        let j_coulomb = ri_contraction(n, n_aux, &setup.b_fit, &density);
        let contraction_seconds = started.elapsed().as_secs_f64();
        // 同一条 setup 的**单线程**路径：这就是并行化之前的成本（逐位结果相同）。
        let serial = ri_setup(&basis, &aux, molecule, 1);
        if run == 0 {
            j_exact = j;
            for i in 0..n {
                for jj in 0..n {
                    j_ri[i * n + jj] = j_coulomb[(i, jj)];
                }
            }
            b_fit_parallel = setup.b_fit.clone();
            b_fit_serial = serial.b_fit;
        } else {
            exact_runs.push(seconds);
            exact_tensor_runs.push(tensor_seconds);
            setup_runs.push(setup.total);
            three_center_runs.push(setup.three_center);
            metric_runs.push(setup.metric);
            inverse_runs.push(setup.inverse_sqrt);
            b_prime_runs.push(setup.b_prime);
            serial_setup_runs.push(serial.total);
            serial_three_center_runs.push(serial.three_center);
            serial_metric_runs.push(serial.metric);
            serial_inverse_runs.push(serial.inverse_sqrt);
            serial_b_prime_runs.push(serial.b_prime);
            contraction_runs.push(contraction_seconds);
        }
    }

    // 逐位确定性：两条线程路径的 `B'` 必须逐位相同（不是"容差内"）。
    let (worst_b, bits_b) = worst_difference(&b_fit_parallel, &b_fit_serial);

    let exact_min = min_of(&exact_runs);
    let exact_med = median(&mut exact_runs.clone());
    let tensor_min = min_of(&exact_tensor_runs);
    let setup_min = min_of(&setup_runs);
    let setup_med = median(&mut setup_runs.clone());
    let three_center_min = min_of(&three_center_runs);
    let metric_min = min_of(&metric_runs);
    let inverse_min = min_of(&inverse_runs);
    let b_prime_min = min_of(&b_prime_runs);
    let serial_setup_min = min_of(&serial_setup_runs);
    let serial_setup_med = median(&mut serial_setup_runs.clone());
    let serial_three_center_min = min_of(&serial_three_center_runs);
    let serial_metric_min = min_of(&serial_metric_runs);
    let serial_inverse_min = min_of(&serial_inverse_runs);
    let serial_b_prime_min = min_of(&serial_b_prime_runs);
    let contraction_min = min_of(&contraction_runs);
    let contraction_med = median(&mut contraction_runs.clone());
    let full_min = setup_min + contraction_min;
    let full_med = setup_med + contraction_med;
    let serial_full_min = serial_setup_min + contraction_min;
    let serial_full_med = serial_setup_med + contraction_med;

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (a, b) in j_exact.iter().zip(&j_ri) {
        worst = worst.max((a - b).abs());
        scale = scale.max(a.abs());
    }

    println!("\n==== {label}: n={n}, n_aux={n_aux}, reps={reps}（另热身 1 次，已丢弃）====");
    println!(
        "4 索引 J        min {exact_min:.3} s / median {exact_med:.3} s（其中张量建立 min {tensor_min:.3} s；rayon 默认线程）"
    );
    println!(
        "RI setup 并行   min {setup_min:.3} s / median {setup_med:.3} s = 三中心 {three_center_min:.3} + 度量 {metric_min:.3} + 逆平方根 {inverse_min:.3} + B' {b_prime_min:.3}"
    );
    println!(
        "RI setup 串行   min {serial_setup_min:.3} s / median {serial_setup_med:.3} s = 三中心 {serial_three_center_min:.3} + 度量 {serial_metric_min:.3} + 逆平方根 {serial_inverse_min:.3} + B' {serial_b_prime_min:.3}"
    );
    println!(
        "每段提速        三中心 {:.2}x / 度量 {:.2}x / 逆平方根 {:.2}x / B' {:.2}x / setup {:.2}x（min）",
        serial_three_center_min / three_center_min,
        serial_metric_min / metric_min,
        serial_inverse_min / inverse_min,
        serial_b_prime_min / b_prime_min,
        serial_setup_min / setup_min
    );
    println!(
        "并行 setup 占比 三中心 {:.2}% / 度量 {:.2}% / 逆平方根 {:.2}% / B' {:.2}%",
        100.0 * three_center_min / setup_min,
        100.0 * metric_min / setup_min,
        100.0 * inverse_min / setup_min,
        100.0 * b_prime_min / setup_min
    );
    println!("RI 收缩         min {contraction_min:.3} s / median {contraction_med:.3} s");
    println!(
        "RI 全管线(并行 setup) min {full_min:.3} s / median {full_med:.3} s -> 加速 {:.2}x（min）/ {:.2}x（median）；收缩占 {:.2}%",
        exact_min / full_min,
        exact_med / full_med,
        100.0 * contraction_min / full_min
    );
    println!(
        "RI 全管线(串行 setup) min {serial_full_min:.3} s / median {serial_full_med:.3} s -> 加速 {:.2}x（min）/ {:.2}x（median）（并行化之前的同一进程对照）",
        exact_min / serial_full_min,
        exact_med / serial_full_med
    );
    println!(
        "同值：max|J_RI − J_exact| = {worst:.3e} Ha（相对 {:.2e}，max|J_exact| = {scale:.3e} Ha）",
        worst / scale.max(1e-30)
    );
    println!(
        "确定性：并行 vs 串行 setup，B' 逐位不同的元素 {bits_b} 个（max|Δ| = {worst_b:.3e}，共 {} 个元素）",
        b_fit_parallel.len()
    );
}

#[test]
#[ignore = "slow: IN-04 RI-J 分阶段测量（水 + 苯，热身 1 + 重复 2–3，串行/并行两条路径同进程对照，本机约 7 min）"]
fn ri_j_stage_breakdown_and_determinism() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    measure_stages("water", &registry, &water(), 3);
    measure_stages("benzene", &registry, &benzene(), 2);
}
