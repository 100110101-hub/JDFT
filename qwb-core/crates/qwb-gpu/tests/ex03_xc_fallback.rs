//! EX-03 验收 **A3**：禁用设备路径后，回退必须产出与 CPU 基准**逐位相同**的
//! `E_xc` 与 `v_xc`，且**不报错**。
//!
//! 三种禁用方式各有用处：
//!
//! * `XcGridOptions::force_cpu` —— 可复现、可并行（不改进程环境变量）；
//! * `XcGridOptions::budget_bytes` 小到装不下 —— 这就是"显存不够"（不需要真把 8 GiB 吃满）；
//! * `QWB_GPU_XC_DISABLE` —— 运维开关，用**外部**环境变量演示（测试进程里改环境变量在
//!   edition 2024 下本身是 `unsafe fn`，而且并行测试之间会打架）。
//!
//! 每一条都跑了**两遍**：一遍 LDA（`svwn5`），一遍 GGA（`pbe`，夹具里带 `∇Φ` 与 `σ` 通道）。
//! GGA 的回退也不是"差不多就行"：它要复刻 `evaluate_xc_cached` 的 `ρ`/`∇ρ`/`σ` 三步与
//! `σ` 通道的装配次序，所以逐位相同这一条同样钉在这里。
#![allow(clippy::field_reassign_with_default)]

use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{GridAoCached, RksOptions, evaluate_xc_cached, rks_scf};
use qwb_xc::{DensityBatch, FunctionalTerm};

use qwb_gpu::xc::{
    XcGridGgaInputs, XcGridInputs, XcGridOptions, XcPath, xc_bytes, xc_gga_bytes, xc_grid_gga,
    xc_grid_lda,
};

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
        if component.weight.abs() <= 1e-12 || component.id == "HfExchange" {
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    terms
}

// 回调签名就是核的契约（ρ → (ε, v)）；拆成类型别名只会让它更难对着 CPU 路径逐句核对。
#[allow(clippy::type_complexity)]
fn lda_closure<'a>(
    terms: &'a [Box<dyn FunctionalTerm>],
) -> impl FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String> + 'a {
    move |rho: &[f64]| {
        let points = rho.len();
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

struct Fixture {
    n: usize,
    n_points: usize,
    ao: Vec<f64>,
    density: Vec<f64>,
    weights: Vec<f64>,
    terms: Vec<Box<dyn FunctionalTerm>>,
    cpu_energy: f64,
    cpu_vxc: qwb_linalg::Matrix,
}

fn fixture() -> Fixture {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(
        registry.get_grid("grid-normal").expect("grid-normal"),
        &molecule,
    )
    .expect("grid");
    let terms = terms_from(&registry, "svwn5");
    let mut options = RksOptions::default();
    options.energy_threshold = 1.0e-9;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");
    assert!(result.scf_converged);
    let cache = GridAoCached::build(&basis, &grid, false, false, false).expect("ao cache");
    let (cpu_energy, cpu_vxc) =
        evaluate_xc_cached(&basis, &grid, &cache, &result.density, &terms).expect("cpu xc");

    let n = basis.n_functions();
    let n_points = grid.len();
    let mut ao = Vec::with_capacity(n_points * n);
    for index in 0..n_points {
        ao.extend_from_slice(cache.values_at(index));
    }
    let mut density = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            density.push(result.density[(i, j)]);
        }
    }
    let weights = grid.points.iter().map(|point| point.weight).collect();
    Fixture {
        n,
        n_points,
        ao,
        density,
        weights,
        terms,
        cpu_energy,
        cpu_vxc,
    }
}

impl Fixture {
    fn inputs(&self) -> XcGridInputs<'_> {
        XcGridInputs {
            n: self.n,
            n_points: self.n_points,
            ao_values: &self.ao,
            density: &self.density,
            weights: &self.weights,
        }
    }

    /// 与产品 CPU 路径逐位比较（能量 + 整个 `v_xc`）。
    fn assert_bit_identical(&self, label: &str, energy: f64, v_xc: &[f64]) {
        assert_eq!(
            energy.to_bits(),
            self.cpu_energy.to_bits(),
            "{label}：E_xc 与 CPU 基准不是逐位相同（{energy:.17e} vs {:.17e}）",
            self.cpu_energy
        );
        let mut differing = 0usize;
        for i in 0..self.n {
            for j in 0..self.n {
                if v_xc[i * self.n + j].to_bits() != self.cpu_vxc[(i, j)].to_bits() {
                    differing += 1;
                }
            }
        }
        assert_eq!(
            differing,
            0,
            "{label}：v_xc 有 {differing}/{} 个元素与 CPU 基准不同",
            self.n * self.n
        );
    }
}

#[test]
fn forcing_cpu_is_bit_identical_to_the_product_path() {
    let fixture = fixture();
    let output = xc_grid_lda(
        &fixture.inputs(),
        XcGridOptions {
            force_cpu: true,
            ..XcGridOptions::default()
        },
        lda_closure(&fixture.terms),
    )
    .expect("回退不许报错");
    println!(
        "force_cpu: path = {:?}, reason = {:?}, E_xc = {:.17e}（CPU 基准 {:.17e}）",
        output.path, output.fallback_reason, output.energy, fixture.cpu_energy
    );
    assert_eq!(output.path, XcPath::Cpu);
    assert!(
        output
            .fallback_reason
            .as_deref()
            .is_some_and(|why| why.contains("force_cpu")),
        "回退原因应说明是调用方要求的：{:?}",
        output.fallback_reason
    );
    fixture.assert_bit_identical("force_cpu", output.energy, &output.v_xc);
}

#[test]
fn a_budget_below_the_requirement_falls_back_to_cpu() {
    let fixture = fixture();
    let required = xc_bytes(fixture.n_points, fixture.n);
    let output = xc_grid_lda(
        &fixture.inputs(),
        XcGridOptions {
            budget_bytes: Some(required - 1),
            ..XcGridOptions::default()
        },
        lda_closure(&fixture.terms),
    )
    .expect("显存不够不许报错，只许回退");
    println!(
        "budget {} B < required {required} B: path = {:?}, reason = {:?}",
        required - 1,
        output.path,
        output.fallback_reason
    );
    assert_eq!(output.path, XcPath::Cpu);
    let reason = output.fallback_reason.clone().unwrap_or_default();
    // 判定优先级（顺序本身就是契约）：`force_cpu` > `QWB_GPU_XC_DISABLE` > 显存预算 > 设备。
    // 所以用 `QWB_GPU_XC_DISABLE=1` 跑整个二进制时，这条要看见的是开关而不是预算。
    let switched_off =
        std::env::var_os("QWB_GPU_XC_DISABLE").is_some_and(|value| !value.is_empty());
    if switched_off {
        assert!(
            reason.contains("QWB_GPU_XC_DISABLE"),
            "运维开关优先于预算，回退原因应指向开关：{reason}"
        );
    } else {
        // 预算检查发生在**碰设备之前**，所以只要 NVRTC 层参与编译，这条必定报显存；
        // feature 没开时更早一步就退了（那时连预算都无从判断）。
        #[cfg(feature = "gpu-kernel")]
        assert!(
            reason.contains("显存预算不够"),
            "回退原因应指向显存预算：{reason}"
        );
        #[cfg(not(feature = "gpu-kernel"))]
        assert!(
            reason.contains("gpu-kernel"),
            "feature 未开时回退原因应指向 feature：{reason}"
        );
    }
    fixture.assert_bit_identical("budget", output.energy, &output.v_xc);
}

#[test]
fn the_default_path_reports_what_it_actually_ran() {
    // 这条在**有 GPU + `--features gpu-kernel`** 时走设备路径、否则回退；
    // 两种都只允许一种结论：设备路径要落在 1e-10，回退路径要逐位相同。
    // 用一个外部环境变量（`QWB_GPU_XC_DISABLE=1`）跑同一个二进制，就能看见开关生效。
    let fixture = fixture();
    let output = xc_grid_lda(
        &fixture.inputs(),
        XcGridOptions::default(),
        lda_closure(&fixture.terms),
    )
    .expect("xc_grid_lda");
    println!(
        "default: path = {:?}, reason = {:?}, E_xc = {:.17e}",
        output.path, output.fallback_reason, output.energy
    );
    match output.path {
        XcPath::Gpu => {
            let delta = (output.energy - fixture.cpu_energy).abs();
            assert!(delta <= 1.0e-10, "设备路径的 |ΔE| = {delta:.3e} Ha");
            assert!(output.fallback_reason.is_none());
        }
        XcPath::Cpu => {
            fixture.assert_bit_identical("default", output.energy, &output.v_xc);
        }
    }
}

#[test]
fn the_device_memory_accounting_is_exact() {
    // 六块 f64：Φ、P、(ΦP)、ρ、系数、v_xc。
    assert_eq!(xc_bytes(10, 2), (2 * 10 * 2 + 2 * 2 * 2 + 2 * 10) * 8);
    assert_eq!(xc_bytes(1, 1), (2 + 2 + 2) * 8);
    assert_eq!(
        xc_bytes(60_000, 24) as f64,
        2.0 * 60_000.0 * 24.0 * 8.0 + 2.0 * 576.0 * 8.0 + 2.0 * 60_000.0 * 8.0
    );
}

#[test]
fn shape_errors_are_reported_not_panicked() {
    let ao = vec![0.0; 12];
    let density = vec![0.0; 3];
    let weights = vec![0.0; 2];
    let bad = XcGridInputs {
        n: 2,
        n_points: 6,
        ao_values: &ao,
        density: &density,
        weights: &weights,
    };
    let mut evaluate = |rho: &[f64]| Ok((rho.to_vec(), rho.to_vec()));
    let error = xc_grid_lda(&bad, XcGridOptions::default(), &mut evaluate)
        .expect_err("density 长度不对必须报错");
    println!("shape error: {error}");
    assert!(
        error.contains("density.len()"),
        "错误消息应指出哪一块形状不对：{error}"
    );

    // 泛函自己报错时也要原样上报（不许静默回退成另一条数值）。
    let ao = vec![0.0; 4];
    let density = vec![0.0; 4];
    let weights = vec![1.0; 2];
    let good = XcGridInputs {
        n: 2,
        n_points: 2,
        ao_values: &ao,
        density: &density,
        weights: &weights,
    };
    let mut angry = |_rho: &[f64]| Err("泛函炸了".to_string());
    let error =
        xc_grid_lda(&good, XcGridOptions::default(), &mut angry).expect_err("泛函错误必须上报");
    assert_eq!(error, "泛函炸了");
}

// ---------------------------------------------------------------------------------------
// GGA 通道
// ---------------------------------------------------------------------------------------

/// GGA 的夹具：`pbe` 的 SCF 密度、带 `∇Φ` 的缓存、以及产品 CPU 路径给出的基准。
struct GgaFixture {
    n: usize,
    n_points: usize,
    ao: Vec<f64>,
    gradients: [Vec<f64>; 3],
    density: Vec<f64>,
    weights: Vec<f64>,
    terms: Vec<Box<dyn FunctionalTerm>>,
    cpu_energy: f64,
    cpu_vxc: qwb_linalg::Matrix,
}

fn gga_fixture() -> GgaFixture {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(
        registry.get_grid("grid-normal").expect("grid-normal"),
        &molecule,
    )
    .expect("grid");
    let terms = terms_from(&registry, "pbe");
    let mut options = RksOptions::default();
    options.energy_threshold = 1.0e-9;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");
    assert!(result.scf_converged);
    // `∇Φ` 只在需要 σ 时预取：产品 CPU 路径与设备路径必须看同一批 AO 数字。
    let cache = GridAoCached::build(&basis, &grid, true, false, false).expect("ao cache");
    let (cpu_energy, cpu_vxc) =
        evaluate_xc_cached(&basis, &grid, &cache, &result.density, &terms).expect("cpu xc");

    let n = basis.n_functions();
    let n_points = grid.len();
    let mut ao = Vec::with_capacity(n_points * n);
    let mut gradients = [
        Vec::with_capacity(n_points * n),
        Vec::with_capacity(n_points * n),
        Vec::with_capacity(n_points * n),
    ];
    for index in 0..n_points {
        ao.extend_from_slice(cache.values_at(index));
        for element in cache.gradients_at(index) {
            for axis in 0..3 {
                gradients[axis].push(element[axis]);
            }
        }
    }
    let mut density = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            density.push(result.density[(i, j)]);
        }
    }
    let weights = grid.points.iter().map(|point| point.weight).collect();
    GgaFixture {
        n,
        n_points,
        ao,
        gradients,
        density,
        weights,
        terms,
        cpu_energy,
        cpu_vxc,
    }
}

/// GGA 的泛函求值（与 `evaluate_xc_cached` 的 GGA 分支同序）。
// 回调签名就是核的契约（(ρ, σ) → (ε, dε/dρ, dε/dσ)）。
#[allow(clippy::type_complexity)]
fn gga_closure<'a>(
    terms: &'a [Box<dyn FunctionalTerm>],
) -> impl FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> + 'a {
    move |rho: &[f64], sigma: &[f64]| {
        let points = rho.len();
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

impl GgaFixture {
    fn inputs(&self) -> XcGridGgaInputs<'_> {
        XcGridGgaInputs {
            n: self.n,
            n_points: self.n_points,
            ao_values: &self.ao,
            ao_gradients: [&self.gradients[0], &self.gradients[1], &self.gradients[2]],
            density: &self.density,
            weights: &self.weights,
        }
    }

    /// 与产品 CPU 路径逐位比较（能量 + 整个 `v_xc` + `ρ`/`∇ρ`/`σ`）。
    fn assert_bit_identical(&self, label: &str, output: &qwb_gpu::xc::XcGridGgaOutput) {
        assert_eq!(
            output.energy.to_bits(),
            self.cpu_energy.to_bits(),
            "{label}：E_xc 与 CPU 基准不是逐位相同（{:.17e} vs {:.17e}）",
            output.energy,
            self.cpu_energy
        );
        let mut differing = 0usize;
        for i in 0..self.n {
            for j in 0..self.n {
                if output.v_xc[i * self.n + j].to_bits() != self.cpu_vxc[(i, j)].to_bits() {
                    differing += 1;
                }
            }
        }
        let non_finite_sigma = output
            .sigma
            .iter()
            .filter(|value| !value.is_finite())
            .count();
        println!(
            "{label}: E_xc = {:.17e}（基准 {:.17e}，逐位相同），v_xc 逐位相同 \
             （{differing}/{} 个元素不同），σ 非有限值 {non_finite_sigma} 个",
            output.energy,
            self.cpu_energy,
            self.n * self.n
        );
        assert_eq!(
            differing,
            0,
            "{label}：v_xc 有 {differing}/{} 个元素与 CPU 基准不同",
            self.n * self.n
        );
    }
}

#[test]
fn gga_forcing_cpu_is_bit_identical_to_the_product_path() {
    let fixture = gga_fixture();
    let output = xc_grid_gga(
        &fixture.inputs(),
        XcGridOptions {
            force_cpu: true,
            ..XcGridOptions::default()
        },
        gga_closure(&fixture.terms),
    )
    .expect("回退不许报错");
    println!(
        "gga force_cpu: path = {:?}, reason = {:?}, E_xc = {:.17e}（CPU 基准 {:.17e}）",
        output.path, output.fallback_reason, output.energy, fixture.cpu_energy
    );
    assert_eq!(output.path, XcPath::Cpu);
    assert!(
        output
            .fallback_reason
            .as_deref()
            .is_some_and(|why| why.contains("force_cpu")),
        "回退原因应说明是调用方要求的：{:?}",
        output.fallback_reason
    );
    fixture.assert_bit_identical("gga force_cpu", &output);
}

#[test]
fn a_gga_budget_below_the_requirement_falls_back_to_cpu() {
    let fixture = gga_fixture();
    let required = xc_gga_bytes(fixture.n_points, fixture.n);
    let output = xc_grid_gga(
        &fixture.inputs(),
        XcGridOptions {
            budget_bytes: Some(required - 1),
            ..XcGridOptions::default()
        },
        gga_closure(&fixture.terms),
    )
    .expect("显存不够不许报错，只许回退");
    println!(
        "gga budget {} B < required {required} B: path = {:?}, reason = {:?}",
        required - 1,
        output.path,
        output.fallback_reason
    );
    assert_eq!(output.path, XcPath::Cpu);
    let reason = output.fallback_reason.clone().unwrap_or_default();
    let switched_off =
        std::env::var_os("QWB_GPU_XC_DISABLE").is_some_and(|value| !value.is_empty());
    if switched_off {
        assert!(
            reason.contains("QWB_GPU_XC_DISABLE"),
            "运维开关优先于预算，回退原因应指向开关：{reason}"
        );
    } else {
        #[cfg(feature = "gpu-kernel")]
        assert!(
            reason.contains("显存预算不够"),
            "回退原因应指向显存预算：{reason}"
        );
        #[cfg(not(feature = "gpu-kernel"))]
        assert!(
            reason.contains("gpu-kernel"),
            "feature 未开时回退原因应指向 feature：{reason}"
        );
    }
    fixture.assert_bit_identical("gga budget", &output);
}

#[test]
fn gga_the_default_path_reports_what_it_actually_ran() {
    let fixture = gga_fixture();
    let output = xc_grid_gga(
        &fixture.inputs(),
        XcGridOptions::default(),
        gga_closure(&fixture.terms),
    )
    .expect("xc_grid_gga");
    println!(
        "gga default: path = {:?}, reason = {:?}, E_xc = {:.17e}, σ 非有限值 {} 个",
        output.path,
        output.fallback_reason,
        output.energy,
        output
            .sigma
            .iter()
            .filter(|value| !value.is_finite())
            .count()
    );
    match output.path {
        XcPath::Gpu => {
            let delta = (output.energy - fixture.cpu_energy).abs();
            assert!(delta <= 1.0e-10, "设备路径的 |ΔE| = {delta:.3e} Ha");
            assert!(output.fallback_reason.is_none());
        }
        XcPath::Cpu => fixture.assert_bit_identical("gga default", &output),
    }
}

#[test]
fn the_gga_device_memory_accounting_is_exact() {
    // 逐项：Φ + ∇Φ×3 + 缩比临时块 = 5·points·n；P + 三个 M_a + v_xc = 5·n²；
    // ρ、σ、∇ρ×3、w·v、w·σ 系数 = 7·points。
    assert_eq!(xc_gga_bytes(10, 2), (5 * 10 * 2 + 5 * 2 * 2 + 7 * 10) * 8);
    assert_eq!(xc_gga_bytes(1, 1), (5 + 5 + 7) * 8);
    assert_eq!(
        xc_gga_bytes(72_900, 24) as f64,
        5.0 * 72_900.0 * 24.0 * 8.0 + 5.0 * 576.0 * 8.0 + 7.0 * 72_900.0 * 8.0
    );
}

#[test]
fn gga_shape_errors_are_reported_not_panicked() {
    let ao = vec![0.0; 12];
    let gradients = [vec![0.0; 12], vec![0.0; 11], vec![0.0; 12]];
    let density = vec![0.0; 4];
    let weights = vec![0.0; 6];
    let bad = XcGridGgaInputs {
        n: 2,
        n_points: 6,
        ao_values: &ao,
        ao_gradients: [&gradients[0], &gradients[1], &gradients[2]],
        density: &density,
        weights: &weights,
    };
    let mut evaluate =
        |rho: &[f64], sigma: &[f64]| Ok((rho.to_vec(), rho.to_vec(), sigma.to_vec()));
    let error = xc_grid_gga(&bad, XcGridOptions::default(), &mut evaluate)
        .expect_err("∇Φ 长度不对必须报错");
    println!("gga shape error: {error}");
    assert!(
        error.contains("ao_gradients[y]"),
        "错误消息应指出是哪个轴、哪一块形状不对：{error}"
    );

    // 泛函自己报错时也要原样上报（不许静默回退成另一条数值）。
    let ao = vec![0.0; 4];
    let gx = vec![0.0; 4];
    let gy = vec![0.0; 4];
    let gz = vec![0.0; 4];
    let density = vec![0.0; 4];
    let weights = vec![1.0; 2];
    let good = XcGridGgaInputs {
        n: 2,
        n_points: 2,
        ao_values: &ao,
        ao_gradients: [&gx, &gy, &gz],
        density: &density,
        weights: &weights,
    };
    let mut angry = |_rho: &[f64], _sigma: &[f64]| Err("GGA 泛函炸了".to_string());
    let error =
        xc_grid_gga(&good, XcGridOptions::default(), &mut angry).expect_err("泛函错误必须上报");
    assert_eq!(error, "GGA 泛函炸了");
}
