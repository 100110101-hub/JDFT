//! FX-01 的引擎侧验收：**"数据声明需要 τ"与"实现真的读 τ"必须一致**。
//!
//! 这条门禁的意义是防**静默降级**：如果数据里写了 `requires.kinetic_density: true`
//! 但没有任何组件在读 τ，用户会拿到一个 GGA 水平的答案，却以为算了 meta-GGA
//! （phase-2 §10.3 明令禁止）。反向同理 —— 组件读 τ 而数据没声明，
//! 说明数据对自己的需求说谎，`qwb data verify` 的语义就废了。
//!
//! 真正的 meta-GGA（M06-2X / M06-L）是 DA-01/DA-02 与 FX-02 的事；FX-01 只把
//! 管线与门禁立起来，所以这里用**测试里定义的 τ 项**来构造两种不一致。

use qwb_engine::check_meta_gga_declaration;
use qwb_engine_api::EngineError;
use qwb_xc::{DensityBatch, FunctionalTerm, Requirements, gga::PbeExchange};

/// 需要 τ 的测试项（只为触发门禁）。
struct TauTerm;

impl FunctionalTerm for TauTerm {
    fn id(&self) -> &'static str {
        "TestTauTerm"
    }

    fn requires(&self) -> Requirements {
        Requirements::meta_gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = 0.1 * batch.tau_at(index);
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        d_rho.fill(0.0);
        d_sigma.fill(0.0);
        let _ = batch;
    }

    fn potential_derivatives_full(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
        d_tau: &mut [f64],
    ) {
        self.potential_derivatives(batch, d_rho, d_sigma);
        d_laplacian.fill(0.0);
        d_tau.fill(0.1);
    }
}

fn gga_terms() -> Vec<Box<dyn FunctionalTerm>> {
    vec![Box::new(PbeExchange::new(0.804, 0.219_514_972_764_517_04))]
}

fn tau_terms() -> Vec<Box<dyn FunctionalTerm>> {
    vec![Box::new(TauTerm)]
}

#[test]
fn consistent_declarations_are_accepted() {
    // 普通 GGA：数据不声明 τ、实现也不读 → 放过
    check_meta_gga_declaration("pbe", false, &gga_terms()).expect("GGA 应当放过");
    // meta-GGA：数据声明 τ、实现也读 → 放过
    check_meta_gga_declaration("m06-2x", true, &tau_terms()).expect("meta-GGA 应当放过");
    // 纯 HF（没有泛函项、也没有数据）
    check_meta_gga_declaration("hf", false, &[]).expect("HF 应当放过");
}

#[test]
fn data_declares_tau_but_nothing_reads_it_is_rejected() {
    let error = check_meta_gga_declaration("fake-meta-gga", true, &gga_terms())
        .expect_err("数据说要 τ 但没有项读它，必须拒");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("kinetic_density"), "{message}");
            assert!(
                message.contains("GGA-level") || message.contains("silently"),
                "{message}"
            );
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn implementation_reads_tau_but_data_does_not_declare_it_is_rejected() {
    let error = check_meta_gga_declaration("pbe", false, &tau_terms())
        .expect_err("组件读 τ 而数据没声明，必须拒");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("kinetic_density"), "{message}");
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn shipped_functionals_still_run() {
    // 仓库里现有泛函（pbe / svwn5）都不声明 τ，正向路径必须照旧工作。
    use std::path::PathBuf;

    use qwb_engine::RksBackend;
    use qwb_engine_api::{
        Atom, EngineBackend, GridLevel, JobInput, JobType, Molecule, TaskOptions,
    };
    use qwb_registry::Registry;

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let backend = RksBackend::new(Registry::open(root.join("qwb-data")).expect("qwb-data"));
    let molecule = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    );
    let mut options = TaskOptions::default();
    options.scf_max_iter = 200;
    for method in ["pbe", "svwn5"] {
        let input = JobInput::new(
            molecule.clone(),
            "sto-3g",
            method,
            GridLevel::Normal,
            "vacuum",
            JobType::Sp,
            options.clone(),
        );
        let result = backend
            .run(&input, &qwb_engine_api::reporter::NullReporter)
            .unwrap_or_else(|error| panic!("{method} 应当跑通：{error}"));
        assert!(
            result.energy_hartree.is_finite(),
            "{method} 能量必须是有限数"
        );
    }
}
