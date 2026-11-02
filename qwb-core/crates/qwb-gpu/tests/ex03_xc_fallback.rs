//! EX-03 验收 **A3**：禁用设备路径后，回退必须产出与 CPU 基准**逐位相同**的
//! `E_xc` 与 `v_xc`，且**不报错**。
//!
//! 三种禁用方式各有用处：
//!
//! * `XcGridOptions::force_cpu` —— 可复现、可并行（不改进程环境变量）；
//! * `XcGridOptions::budget_bytes` 小到装不下 —— 这就是"显存不够"（不需要真把 8 GiB 吃满）；
//! * `QWB_GPU_XC_DISABLE` —— 运维开关，用**外部**环境变量演示（测试进程里改环境变量在
//!   edition 2024 下本身是 `unsafe fn`，而且并行测试之间会打架）。
#![allow(clippy::field_reassign_with_default)]

use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{GridAoCached, RksOptions, evaluate_xc_cached, rks_scf};
use qwb_xc::{DensityBatch, FunctionalTerm};

use qwb_gpu::xc::{XcGridInputs, XcGridOptions, XcPath, xc_bytes, xc_grid_lda};

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
