//! EX-03 验收 **A1 / A2**：格点 XC 的收缩段，设备路径 vs CPU 基准路径。
//!
//! * A1：`E_xc` 的 |GPU − CPU| ≤ **1e-10 Ha**，并报出 `v_xc` 的逐元素最大差与它的尺度；
//!   用**项目自己的 SCF 密度**（`rks_scf` + svwn5），不是合成密度。
//! * A2：同一段收缩的 CPU vs GPU 墙钟，报点数、基组、warm-up、重复次数。
//!   **加速比很小或为负也是结果**，这里不对它做断言（`xc_cost_breakdown` 已经量过
//!   这段的形状：`O(n_points·n²)`、访存受限）。
//!
//! # GGA 通道（A4 / A5）
//!
//! 同两条验收，对象换成 **GGA**（`∇ρ`、`σ = |∇ρ|²`、`σ` 通道的 `v_xc`）：
//!
//! * A4：`pbe` 与 `b88` 两个泛函、两个基组，报 `E_xc` / `v_xc` / 以及 **`ρ`、`∇ρ`、`σ`
//!   各有多少个点与 CPU 逐位不同** —— 逐位相同才有资格说"这不是碰巧落在容差里"。
//! * A5：收缩段的墙钟（与 A2 同一套协议），外加**带真泛函**的"整个 XC 步"墙钟，
//!   这样"GPU 覆盖了 CPU XC 步的多大一块"是个数出来的数，不是估的。
//!
//! 没有 GPU / 没开 `gpu-kernel` 时不是"跳过"，而是**跑回退分支**：结果必须与 CPU 路径
//! 逐位相同。门禁在没有 GPU 的机器上照样绿。
#![allow(clippy::field_reassign_with_default)] // options 逐字段覆盖最清楚

use std::path::PathBuf;
use std::time::Instant;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{GridAoCached, RksOptions, evaluate_xc_cached, rks_scf};
use qwb_xc::{DensityBatch, FunctionalTerm};

use qwb_gpu::xc::{XcGridGgaInputs, XcGridInputs, XcGridOptions, XcPath, xc_grid_gga, xc_grid_lda};

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
    let parameters: std::collections::BTreeMap<String, f64> = functional.parameters.clone();
    let mut terms: Vec<Box<dyn FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        // 精确交换不进 XC 泛函求值（它在 Fock 矩阵里）。
        if component.weight.abs() <= 1e-12 || component.id == "HfExchange" {
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    terms
}

/// 与 `qwb_scf::rks::evaluate_xc_cached` 的 **LDA 分支逐句同序**的泛函求值：
/// 逐项算进各自的局部数组、再按点累加，最后 `potential = ε + ρ·dε/dρ`（ρ=0 时取 ε）。
///
/// 同序是有必要的：`energy_density` 是**覆盖式**写入（`*slot = …`），直接在共享累加器上
/// 调它会丢掉前面几项。
// 回调签名就是核的契约（ρ → (ε, v)）；拆成类型别名只会让它更难对着 CPU 路径逐句核对。
#[allow(clippy::type_complexity)]
fn lda_closure<'a>(
    terms: &'a [Box<dyn FunctionalTerm>],
) -> impl FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String> + 'a {
    move |rho: &[f64]| {
        let points = rho.len();
        for term in terms {
            let requires = term.requires();
            if requires.sigma || requires.laplacian || requires.kinetic_density {
                return Err(format!(
                    "泛函项 {} 需要 σ/∇²ρ/τ 通道；设备核只实现 LDA",
                    term.id()
                ));
            }
        }
        let batch = DensityBatch::from_rho(rho);
        let mut epsilon = vec![0.0; points];
        let mut d_rho = vec![0.0; points];
        for term in terms {
            let mut local_epsilon = vec![0.0; points];
            let mut local_d_rho = vec![0.0; points];
            let mut local_d_sigma = vec![0.0; points];
            let mut local_d_laplacian = vec![0.0; points];
            let mut local_d_tau = vec![0.0; points];
            term.energy_density(&batch, &mut local_epsilon);
            term.potential_derivatives_full(
                &batch,
                &mut local_d_rho,
                &mut local_d_sigma,
                &mut local_d_laplacian,
                &mut local_d_tau,
            );
            for index in 0..points {
                epsilon[index] += local_epsilon[index];
                d_rho[index] += local_d_rho[index];
            }
        }
        let potential = (0..points)
            .map(|index| {
                if rho[index] > 0.0 {
                    epsilon[index] + rho[index] * d_rho[index]
                } else {
                    epsilon[index]
                }
            })
            .collect();
        Ok((epsilon, potential))
    }
}

struct Setup {
    basis: qwb_basis::InternalBasis,
    grid: qwb_grid::Grid,
    cache: GridAoCached,
    density: qwb_linalg::Matrix,
    terms: Vec<Box<dyn FunctionalTerm>>,
}

fn solve(basis_name: &str, grid_name: &str, functional: &str) -> Setup {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        &molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name).expect(grid_name), &molecule)
        .expect("grid");
    let terms = terms_from(&registry, functional);
    let mut options = RksOptions::default();
    options.max_iter = 200;
    options.energy_threshold = 1.0e-9;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");
    assert!(result.scf_converged, "{basis_name} 的 SCF 没收敛");
    // AO 三元组与密度无关，SCF 内部也缓存着；这里另建一份给收缩核用。
    let cache = GridAoCached::build(&basis, &grid, false, false, false).expect("ao cache");
    Setup {
        basis,
        grid,
        cache,
        density: result.density,
        terms,
    }
}

/// GGA 通道的夹具：**GGA 自己的 SCF 密度**（`pbe`/`b88` 的 SCF 都收敛）＋ 带 `∇Φ` 的缓存。
///
/// 为什么另外建一个：`∇Φ` 只在需要 σ 时才预取（`GridAoCached::build` 的第三个参数），
/// 而产品 CPU 路径在同一份缓存上算 GGA —— 两侧必须看**同一批 AO 数字**，比较才有意义。
struct GgaSetup {
    basis: qwb_basis::InternalBasis,
    grid: qwb_grid::Grid,
    cache: GridAoCached,
    density: qwb_linalg::Matrix,
    terms: Vec<Box<dyn FunctionalTerm>>,
    /// 产品 CPU 路径（`evaluate_xc_cached`）的 `E_xc` 与 `v_xc` —— GGA 的**唯一**基准。
    energy_cpu: f64,
    vxc_cpu: qwb_linalg::Matrix,
}

fn solve_gga(basis_name: &str, grid_name: &str, functional: &str) -> GgaSetup {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        &molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name).expect(grid_name), &molecule)
        .expect("grid");
    let terms = terms_from(&registry, functional);
    let mut options = RksOptions::default();
    options.max_iter = 200;
    options.energy_threshold = 1.0e-9;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");
    assert!(
        result.scf_converged,
        "{basis_name}/{functional} 的 GGA SCF 没收敛"
    );
    let cache = GridAoCached::build(&basis, &grid, true, false, false).expect("ao cache");
    let (energy_cpu, vxc_cpu) =
        evaluate_xc_cached(&basis, &grid, &cache, &result.density, &terms).expect("cpu xc");
    GgaSetup {
        basis,
        grid,
        cache,
        density: result.density,
        terms,
        energy_cpu,
        vxc_cpu,
    }
}

/// 把 `∇Φ` 按轴摊平成三张 `n_points × n` 行主序。
fn flat_ao_gradients(setup: &GgaSetup) -> [Vec<f64>; 3] {
    let n = setup.basis.n_functions();
    let n_points = setup.grid.len();
    let mut flat = [
        Vec::with_capacity(n_points * n),
        Vec::with_capacity(n_points * n),
        Vec::with_capacity(n_points * n),
    ];
    for index in 0..n_points {
        let row = setup.cache.gradients_at(index);
        assert_eq!(row.len(), n);
        assert_eq!(row[0].len(), 3);
        for element in row {
            for axis in 0..3 {
                flat[axis].push(element[axis]);
            }
        }
    }
    flat
}

fn gga_inputs<'a>(
    n: usize,
    n_points: usize,
    ao: &'a [f64],
    gradients: [&'a [f64]; 3],
    density: &'a [f64],
    weights: &'a [f64],
) -> XcGridGgaInputs<'a> {
    XcGridGgaInputs {
        n,
        n_points,
        ao_values: ao,
        ao_gradients: gradients,
        density,
        weights,
    }
}

/// GGA 的泛函求值：与 `qwb_scf::rks::evaluate_xc_cached` 的 GGA 分支**同序**
/// （逐项算进各自的局部数组、再按点累加），返回 `(ε, dε/dρ, dε/dσ)`。
///
/// 需要 `∇²ρ` / `τ` 的项直接报错，不静默算出一个少了整项的势。
// 回调签名就是核的契约（(ρ, σ) → (ε, dε/dρ, dε/dσ)）；拆成类型别名只会更难对着 CPU 逐句核对。
#[allow(clippy::type_complexity)]
fn gga_closure<'a>(
    terms: &'a [Box<dyn FunctionalTerm>],
) -> impl FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> + 'a {
    move |rho: &[f64], sigma: &[f64]| {
        let points = rho.len();
        for term in terms {
            let requires = term.requires();
            if requires.laplacian || requires.kinetic_density {
                return Err(format!(
                    "泛函项 {} 需要 ∇²ρ/τ 通道；设备核只实现到 GGA（ρ, σ）",
                    term.id()
                ));
            }
        }
        let batch = DensityBatch::new(rho, sigma);
        let mut epsilon = vec![0.0; points];
        let mut d_rho = vec![0.0; points];
        let mut d_sigma = vec![0.0; points];
        for term in terms {
            let mut local_epsilon = vec![0.0; points];
            let mut local_d_rho = vec![0.0; points];
            let mut local_d_sigma = vec![0.0; points];
            let mut local_d_laplacian = vec![0.0; points];
            let mut local_d_tau = vec![0.0; points];
            term.energy_density(&batch, &mut local_epsilon);
            term.potential_derivatives_full(
                &batch,
                &mut local_d_rho,
                &mut local_d_sigma,
                &mut local_d_laplacian,
                &mut local_d_tau,
            );
            for index in 0..points {
                epsilon[index] += local_epsilon[index];
                d_rho[index] += local_d_rho[index];
                d_sigma[index] += local_d_sigma[index];
            }
        }
        Ok((epsilon, d_rho, d_sigma))
    }
}

/// 把 AO 缓存摊平成 `n_points × n` 行主序，连带求积权重。
fn flat_ao(setup: &Setup) -> (Vec<f64>, Vec<f64>) {
    flat_ao_of(&setup.basis, &setup.grid, &setup.cache)
}

fn flat_ao_of(
    basis: &qwb_basis::InternalBasis,
    grid: &qwb_grid::Grid,
    cache: &GridAoCached,
) -> (Vec<f64>, Vec<f64>) {
    let n = basis.n_functions();
    let n_points = grid.len();
    let mut values = Vec::with_capacity(n_points * n);
    for index in 0..n_points {
        let row = cache.values_at(index);
        assert_eq!(row.len(), n);
        values.extend_from_slice(row);
    }
    let weights = grid.points.iter().map(|point| point.weight).collect();
    (values, weights)
}

/// 行主序摊平一个矩阵（`v_xc` 的基准就是 `Matrix`，比较前要摊平）。
fn flat_matrix(matrix: &qwb_linalg::Matrix) -> Vec<f64> {
    flat_density(matrix)
}

fn flat_density(density: &qwb_linalg::Matrix) -> Vec<f64> {
    let n = density.nrows();
    let mut flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            flat.push(density[(i, j)]);
        }
    }
    flat
}

fn inputs<'a>(
    n: usize,
    n_points: usize,
    ao: &'a [f64],
    density: &'a [f64],
    weights: &'a [f64],
) -> XcGridInputs<'a> {
    XcGridInputs {
        n,
        n_points,
        ao_values: ao,
        density,
        weights,
    }
}

#[test]
fn a1_xc_energy_and_potential_match_the_cpu_path() {
    println!("NVRTC 状态：{:?}", qwb_gpu::xc::nvrtc_status());
    for (basis_name, grid_name) in [("sto-3g", "grid-normal"), ("def2-svp", "grid-normal")] {
        let setup = solve(basis_name, grid_name, "svwn5");
        let n = setup.basis.n_functions();
        let n_points = setup.grid.len();
        let (energy_cpu, vxc_cpu) = evaluate_xc_cached(
            &setup.basis,
            &setup.grid,
            &setup.cache,
            &setup.density,
            &setup.terms,
        )
        .expect("cpu xc");
        let (ao, weights) = flat_ao(&setup);
        let flat_p = flat_density(&setup.density);
        let grid_inputs = inputs(n, n_points, &ao, &flat_p, &weights);
        let output = xc_grid_lda(
            &grid_inputs,
            XcGridOptions::default(),
            lda_closure(&setup.terms),
        )
        .expect("gpu xc");

        let mut worst_vxc = 0.0_f64;
        let mut scale_vxc = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let cpu = vxc_cpu[(i, j)];
                scale_vxc = scale_vxc.max(cpu.abs());
                worst_vxc = worst_vxc.max((cpu - output.v_xc[i * n + j]).abs());
            }
        }
        let energy_delta = (output.energy - energy_cpu).abs();
        println!(
            "{basis_name}/{grid_name}/svwn5: n = {n}, points = {n_points}, path = {:?} | \
             E_xc CPU = {energy_cpu:.15e}, GPU = {:.15e}, |ΔE| = {energy_delta:.3e} Ha | \
             max|Δv_xc| = {worst_vxc:.3e}（scale {scale_vxc:.3e}，相对 {:.3e}）{}",
            output.path,
            output.energy,
            worst_vxc / scale_vxc.max(f64::MIN_POSITIVE),
            match &output.fallback_reason {
                Some(why) => format!("，回退原因：{why}"),
                None => String::new(),
            }
        );

        match output.path {
            XcPath::Gpu => {
                assert!(
                    energy_delta <= 1.0e-10,
                    "A1：E_xc 的 GPU≡CPU 差 {energy_delta:.3e} Ha 超过 1e-10 Ha"
                );
                assert!(
                    worst_vxc / scale_vxc.max(f64::MIN_POSITIVE) <= 1.0e-10,
                    "A1：v_xc 的相对差 {:.3e} 超过 1e-10",
                    worst_vxc / scale_vxc
                );
            }
            XcPath::Cpu => {
                // 回退分支：必须与 CPU 基准逐位相同（不假装 GPU 跑过）。
                assert!(output.fallback_reason.is_some(), "走 CPU 时必须给出原因");
                assert_eq!(
                    output.energy.to_bits(),
                    energy_cpu.to_bits(),
                    "回退后的 E_xc 与 CPU 基准不是逐位相同"
                );
                let mut differing = 0usize;
                for i in 0..n {
                    for j in 0..n {
                        if output.v_xc[i * n + j].to_bits() != vxc_cpu[(i, j)].to_bits() {
                            differing += 1;
                        }
                    }
                }
                assert_eq!(
                    differing, 0,
                    "回退后的 v_xc 有 {differing} 个元素与基准不同"
                );
            }
        }
    }
}

#[test]
fn a2_the_contraction_speedup_is_measured() {
    const WARMUP: usize = 2;
    const REPS: usize = 5;
    for (basis_name, grid_name) in [("sto-3g", "grid-normal"), ("def2-svp", "grid-normal")] {
        let setup = solve(basis_name, grid_name, "svwn5");
        let n = setup.basis.n_functions();
        let n_points = setup.grid.len();
        let (ao, weights) = flat_ao(&setup);
        let flat_p = flat_density(&setup.density);
        let grid_inputs = inputs(n, n_points, &ao, &flat_p, &weights);
        // 平凡泛函：两侧都把泛函代价压掉，剩下的就是**收缩段**本身。
        let trivial = |rho: &[f64]| Ok((rho.to_vec(), rho.to_vec()));

        for _ in 0..WARMUP {
            let _ =
                xc_grid_lda(&grid_inputs, XcGridOptions::default(), trivial).expect("warmup gpu");
            let _ = xc_grid_lda(
                &grid_inputs,
                XcGridOptions {
                    force_cpu: true,
                    ..XcGridOptions::default()
                },
                trivial,
            )
            .expect("warmup cpu");
        }

        let mut cpu_seconds = f64::INFINITY;
        let mut gpu_seconds = f64::INFINITY;
        let mut cpu_output = None;
        let mut gpu_output = None;
        for _ in 0..REPS {
            let started = Instant::now();
            let output = xc_grid_lda(
                &grid_inputs,
                XcGridOptions {
                    force_cpu: true,
                    ..XcGridOptions::default()
                },
                trivial,
            )
            .expect("cpu");
            cpu_seconds = cpu_seconds.min(started.elapsed().as_secs_f64());
            cpu_output = Some(output);

            let started = Instant::now();
            let output = xc_grid_lda(&grid_inputs, XcGridOptions::default(), trivial).expect("gpu");
            gpu_seconds = gpu_seconds.min(started.elapsed().as_secs_f64());
            gpu_output = Some(output);
        }
        let cpu_output = cpu_output.expect("cpu run");
        let gpu_output = gpu_output.expect("gpu run");

        let scale = cpu_output
            .v_xc
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let worst = cpu_output
            .v_xc
            .iter()
            .zip(&gpu_output.v_xc)
            .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()));
        println!(
            "{basis_name}/{grid_name}: n = {n}, points = {n_points}, warm-up = {WARMUP}, reps = {REPS} | \
             CPU {:.1} ms, GPU {:.1} ms, 加速 {:.2}x | path = {:?} | max|Δv_xc| = {worst:.3e}（相对 {:.1e}）{}",
            cpu_seconds * 1e3,
            gpu_seconds * 1e3,
            cpu_seconds / gpu_seconds,
            gpu_output.path,
            worst / scale,
            match &gpu_output.fallback_reason {
                Some(why) => format!("，回退原因：{why}"),
                None => String::new(),
            }
        );
        assert!(
            worst / scale <= 1.0e-10,
            "计时用的两次收缩结果不一致：相对 {:.3e}",
            worst / scale
        );
        // A2 不对加速比做断言：这一段是访存受限的 `O(n_points·n²)`，n 小时
        // 传输与 launch 开销占主导，加速比小甚至为负都是合法发现。
        assert!(cpu_seconds > 0.0 && gpu_seconds > 0.0);
    }
}

/// 两个 GGA 泛函都要过：`pbe`（交换 + 相关都依赖 σ）与 `b88`（只有交换依赖 σ）。
///
/// 为什么是这两个：`qwb-data` 里 `pbe` 是完整的 GGA 泛函；`b88` 是纯 GGA 交换，
/// `dε/dσ` 的公式形状（含 `asinh` 与 `σ→0` 的极限分支）与 PBE 完全不同，能抓住
/// "只对 PBE 调出来的实现"。BLYP 在数据里**不存在**（只有 `GgaB88X` + `GgaLypC` 两个组件），
/// 而 LYP 需要 `∇²ρ` —— 那是下一条通道，不属于本轮。
const GGA_FUNCTIONALS: [&str; 2] = ["pbe", "b88"];

/// `ρ`、`∇ρ`、`σ` 三者与 CPU 逐位不同的点数。
fn bitwise_shifted(gpu: &[f64], cpu: &[f64]) -> usize {
    gpu.iter()
        .zip(cpu)
        .filter(|(l, r)| l.to_bits() != r.to_bits())
        .count()
}

#[test]
fn a4_gga_energy_and_potential_match_the_cpu_path() {
    println!("NVRTC 状态：{:?}", qwb_gpu::xc::nvrtc_status());
    for functional in GGA_FUNCTIONALS {
        for (basis_name, grid_name) in [("sto-3g", "grid-normal"), ("def2-svp", "grid-normal")] {
            let setup = solve_gga(basis_name, grid_name, functional);
            let n = setup.basis.n_functions();
            let n_points = setup.grid.len();
            let (ao, weights) = flat_ao_of(&setup.basis, &setup.grid, &setup.cache);
            let gradients = flat_ao_gradients(&setup);
            let flat_p = flat_density(&setup.density);
            let grid_inputs = gga_inputs(
                n,
                n_points,
                &ao,
                [&gradients[0], &gradients[1], &gradients[2]],
                &flat_p,
                &weights,
            );
            let output = xc_grid_gga(
                &grid_inputs,
                XcGridOptions::default(),
                gga_closure(&setup.terms),
            )
            .expect("gpu gga");
            // CPU 基准的 `ρ`/`∇ρ`/`σ` 从回退路径拿（它自己的逐位等价由
            // `tests/ex03_xc_fallback.rs` 钉住）。
            let cpu_path = xc_grid_gga(
                &grid_inputs,
                XcGridOptions {
                    force_cpu: true,
                    ..XcGridOptions::default()
                },
                gga_closure(&setup.terms),
            )
            .expect("cpu gga");

            let mut worst_vxc = 0.0_f64;
            let mut scale_vxc = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let cpu = setup.vxc_cpu[(i, j)];
                    scale_vxc = scale_vxc.max(cpu.abs());
                    worst_vxc = worst_vxc.max((cpu - output.v_xc[i * n + j]).abs());
                }
            }
            let energy_delta = (output.energy - setup.energy_cpu).abs();
            let rho_shifted = bitwise_shifted(&output.rho, &cpu_path.rho);
            let sigma_shifted = bitwise_shifted(&output.sigma, &cpu_path.sigma);
            let grad_shifted = output
                .gradient_rho
                .iter()
                .zip(&cpu_path.gradient_rho)
                .filter(|(l, r)| (0..3).any(|axis| l[axis].to_bits() != r[axis].to_bits()))
                .count();
            println!(
                "{basis_name}/{grid_name}/{functional}: n = {n}, points = {n_points}, \
                 path = {:?} | E_xc CPU = {:.15e}, GPU = {:.15e}, |ΔE| = {energy_delta:.3e} Ha \
                 ({}), max|Δv_xc| = {worst_vxc:.3e}（scale {scale_vxc:.3e}，相对 {:.3e}）| \
                 逐位不同的点数：ρ {rho_shifted}/{n_points}, ∇ρ {grad_shifted}/{n_points}, \
                 σ {sigma_shifted}/{n_points} {}",
                output.path,
                setup.energy_cpu,
                output.energy,
                if output.energy.to_bits() == setup.energy_cpu.to_bits() {
                    "逐位相同"
                } else {
                    "不是逐位相同"
                },
                worst_vxc / scale_vxc.max(f64::MIN_POSITIVE),
                match &output.fallback_reason {
                    Some(why) => format!("，回退原因：{why}"),
                    None => String::new(),
                }
            );

            match output.path {
                XcPath::Gpu => {
                    assert!(
                        energy_delta <= 1.0e-10,
                        "A4：{functional} 的 E_xc 的 GPU≡CPU 差 {energy_delta:.3e} Ha 超过 1e-10 Ha"
                    );
                    assert!(
                        worst_vxc / scale_vxc.max(f64::MIN_POSITIVE) <= 1.0e-10,
                        "A4：{functional} 的 v_xc 相对差 {:.3e} 超过 1e-10",
                        worst_vxc / scale_vxc
                    );
                    // 逐位相同不是"锦上添花"：`∇ρ` 走的是与 CPU 同序的串行累加，
                    // 一旦有人把它换成 gemm 的求和序，这条会先红（而 |ΔE| 可能还是 0）。
                    assert_eq!(rho_shifted, 0, "A4：{functional} 的 ρ 不是逐位相同");
                    assert_eq!(
                        grad_shifted, 0,
                        "A4：{functional} 的 ∇ρ 不是逐位相同（{grad_shifted} 个点不同）"
                    );
                    assert_eq!(
                        sigma_shifted, 0,
                        "A4：{functional} 的 σ 不是逐位相同（{sigma_shifted} 个点不同）"
                    );
                }
                XcPath::Cpu => {
                    assert!(output.fallback_reason.is_some(), "走 CPU 时必须给出原因");
                    assert_eq!(
                        output.energy.to_bits(),
                        setup.energy_cpu.to_bits(),
                        "回退后的 E_xc 与 CPU 基准不是逐位相同"
                    );
                    assert_eq!(
                        bitwise_shifted(&output.v_xc, &flat_matrix(&setup.vxc_cpu)),
                        0,
                        "回退后的 v_xc 与 CPU 基准不是逐位相同"
                    );
                }
            }
        }
    }
}

/// A5：GGA 收缩段的墙钟（与 A2 **同一套**协议：warm-up 2、重复 5、取最小），
/// 外加一栏"带真泛函的整个 XC 步"—— 这样"GPU 覆盖了多大一块"是可数的。
#[test]
fn a5_the_gga_contraction_speedup_is_measured() {
    const WARMUP: usize = 2;
    const REPS: usize = 5;
    const REAL_REPS: usize = 3;
    // GGA 的 `v_xc` 装配有四个 gemm，设备侧**共用一个** cuBLAS handle（见 `jit.rs`）。
    // 这里把"一个 handle 的建/销成本"量出来，好让那个决定有个数、而不是一句感觉。
    if qwb_gpu::available().is_ok() {
        let started = Instant::now();
        for _ in 0..10 {
            qwb_gpu::available().expect("cublasCreate + cublasDestroy");
        }
        println!(
            "cuBLAS handle（cublasCreate + cublasDestroy）×10: {:.2} ms → {:.2} ms/次",
            started.elapsed().as_secs_f64() * 1e3,
            started.elapsed().as_secs_f64() * 1e2
        );
    }
    for (basis_name, grid_name) in [("sto-3g", "grid-normal"), ("def2-svp", "grid-normal")] {
        let setup = solve_gga(basis_name, grid_name, "pbe");
        let n = setup.basis.n_functions();
        let n_points = setup.grid.len();
        let (ao, weights) = flat_ao_of(&setup.basis, &setup.grid, &setup.cache);
        let gradients = flat_ao_gradients(&setup);
        let flat_p = flat_density(&setup.density);
        let grid_inputs = gga_inputs(
            n,
            n_points,
            &ao,
            [&gradients[0], &gradients[1], &gradients[2]],
            &flat_p,
            &weights,
        );
        // 平凡泛函：两侧都把泛函代价压掉，剩下的就是**收缩段**本身。
        let trivial = |rho: &[f64], sigma: &[f64]| Ok((rho.to_vec(), rho.to_vec(), sigma.to_vec()));

        for _ in 0..WARMUP {
            let _ =
                xc_grid_gga(&grid_inputs, XcGridOptions::default(), trivial).expect("warmup gpu");
            let _ = xc_grid_gga(
                &grid_inputs,
                XcGridOptions {
                    force_cpu: true,
                    ..XcGridOptions::default()
                },
                trivial,
            )
            .expect("warmup cpu");
        }

        let mut cpu_seconds = f64::INFINITY;
        let mut gpu_seconds = f64::INFINITY;
        let mut cpu_output = None;
        let mut gpu_output = None;
        for _ in 0..REPS {
            let started = Instant::now();
            let output = xc_grid_gga(
                &grid_inputs,
                XcGridOptions {
                    force_cpu: true,
                    ..XcGridOptions::default()
                },
                trivial,
            )
            .expect("cpu");
            cpu_seconds = cpu_seconds.min(started.elapsed().as_secs_f64());
            cpu_output = Some(output);

            let started = Instant::now();
            let output = xc_grid_gga(&grid_inputs, XcGridOptions::default(), trivial).expect("gpu");
            gpu_seconds = gpu_seconds.min(started.elapsed().as_secs_f64());
            gpu_output = Some(output);
        }
        let cpu_output = cpu_output.expect("cpu run");
        let gpu_output = gpu_output.expect("gpu run");

        let scale = cpu_output
            .v_xc
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let worst = cpu_output
            .v_xc
            .iter()
            .zip(&gpu_output.v_xc)
            .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()));
        println!(
            "{basis_name}/{grid_name}/pbe（只算收缩段，平凡泛函）: n = {n}, points = {n_points}, \
             warm-up = {WARMUP}, reps = {REPS}, 取最小 | CPU {:.1} ms, GPU {:.1} ms, \
             加速 {:.2}x | path = {:?} | max|Δv_xc| = {worst:.3e}（相对 {:.1e}）{}",
            cpu_seconds * 1e3,
            gpu_seconds * 1e3,
            cpu_seconds / gpu_seconds,
            gpu_output.path,
            worst / scale,
            match &gpu_output.fallback_reason {
                Some(why) => format!("，回退原因：{why}"),
                None => String::new(),
            }
        );
        assert!(
            worst / scale <= 1.0e-10,
            "计时用的两次收缩结果不一致：相对 {:.3e}",
            worst / scale
        );
        assert!(cpu_seconds > 0.0 && gpu_seconds > 0.0);

        // 同一段收缩、同一份密度，但泛函用**真的** PBE 项：这一栏才是"整个 XC 步"，
        // 也是"GPU 只覆盖其中一块"的直接证据（泛函求值留在 host 上跑）。
        let mut cpu_real = f64::INFINITY;
        let mut gpu_real = f64::INFINITY;
        let mut gpu_real_output = None;
        let _ = xc_grid_gga(
            &grid_inputs,
            XcGridOptions::default(),
            gga_closure(&setup.terms),
        )
        .expect("warmup gpu real");
        for _ in 0..REAL_REPS {
            let started = Instant::now();
            let _ = xc_grid_gga(
                &grid_inputs,
                XcGridOptions {
                    force_cpu: true,
                    ..XcGridOptions::default()
                },
                gga_closure(&setup.terms),
            )
            .expect("cpu real");
            cpu_real = cpu_real.min(started.elapsed().as_secs_f64());

            let started = Instant::now();
            let output = xc_grid_gga(
                &grid_inputs,
                XcGridOptions::default(),
                gga_closure(&setup.terms),
            )
            .expect("gpu real");
            gpu_real = gpu_real.min(started.elapsed().as_secs_f64());
            gpu_real_output = Some(output);
        }
        let gpu_real_output = gpu_real_output.expect("gpu real run");
        println!(
            "{basis_name}/{grid_name}/pbe（整个 XC 步，含 host 上的泛函求值）: reps = {REAL_REPS}, \
             取最小 | CPU {:.1} ms, GPU {:.1} ms, 加速 {:.2}x | path = {:?} | E_xc GPU = {:.15e}\
             （基准 {:.15e}，|ΔE| = {:.3e} Ha）",
            cpu_real * 1e3,
            gpu_real * 1e3,
            cpu_real / gpu_real,
            gpu_real_output.path,
            gpu_real_output.energy,
            setup.energy_cpu,
            (gpu_real_output.energy - setup.energy_cpu).abs()
        );
        assert!(cpu_real > 0.0 && gpu_real > 0.0);
    }
}
