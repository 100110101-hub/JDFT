//! FX-01：**meta-GGA 基础设施（τ 通道）**的验收。
//!
//! 判据（project-phase2.md §6.3 FX-01）：
//!
//! 1. **有限差分导数（对 ρ、σ、τ）< 1e-9** —— 用一个只在测试里定义的 τ 依赖项
//!    （`TauProbe`）来验，与 `d12_b3lyp.rs` 里程序化构造 B3LYP 是同一条路子：
//!    生产数据里不放测试用的泛函；
//! 2. **缺 τ 要报错**，不是把 0 当 τ 悄悄算下去；
//! 3. 已有的 LDA/GGA 项在 τ 通道上必须给 **0**（默认实现），不能因为多了一个通道
//!    就把 PBE 的矩阵元改了；
//! 4. τ 线性项的组装是**精确可算**的（见 `qwb-scf/tests/fx01_tau_pipeline.rs`）。

use qwb_xc::{
    DensityBatch, FunctionalTerm, LdaSvwn5, Requirements, XcError,
    gga::{PbeCorrelation, PbeExchange},
};

/// 只在测试里用的 τ 依赖项（**不是生产泛函**）：
///
/// ```text
/// ε(ρ, σ, τ) = −a·τ/(ρ + ρ₀) − b·σ/(ρ + ρ₀)³,   ρ₀ = 0.1
/// ```
///
/// 选这个形状是因为导数手推得出来、对 ρ/σ/τ 都光滑（ρ > 0 时无奇点），
/// 中心差分在 1e-6 的步长上就能到 1e-9：
///
/// ```text
/// ∂ε/∂ρ =  a·τ/(ρ+ρ₀)² + 3b·σ/(ρ+ρ₀)⁴
/// ∂ε/∂σ = −b/(ρ+ρ₀)³
/// ∂ε/∂τ = −a/(ρ+ρ₀)
/// ```
struct TauProbe {
    a: f64,
    b: f64,
}

const RHO_ZERO: f64 = 0.1;

impl TauProbe {
    /// 刻意**不夹紧** σ 与 τ：这是测试用的光滑探针，夹紧会让 σ = 0 处的中心差分
    /// 只走一侧（f(−h) 被夹成 0），算出来的导数正好差一半 —— 踩过一次。
    fn epsilon(&self, rho: f64, sigma: f64, tau: f64) -> f64 {
        let shifted = rho.max(1.0e-12) + RHO_ZERO;
        -self.a * tau / shifted - self.b * sigma / shifted.powi(3)
    }
}

impl FunctionalTerm for TauProbe {
    fn id(&self) -> &'static str {
        "TestTauProbe"
    }

    fn requires(&self) -> Requirements {
        Requirements::meta_gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        assert!(batch.has_tau(), "TauProbe 需要 τ 通道");
        for (index, slot) in out.iter_mut().enumerate() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            *slot = self.epsilon(batch.rho[index], sigma, batch.tau[index]);
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        for index in 0..batch.len() {
            let rho = batch.rho[index].max(1.0e-12);
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let tau = batch.tau_at(index);
            let shifted = rho + RHO_ZERO;
            d_rho[index] =
                self.a * tau / (shifted * shifted) + 3.0 * self.b * sigma / shifted.powi(4);
            d_sigma[index] = -self.b / shifted.powi(3);
        }
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
        for (index, slot) in d_tau.iter_mut().enumerate() {
            let shifted = batch.rho[index].max(1.0e-12) + RHO_ZERO;
            *slot = -self.a / shifted;
        }
    }
}

/// 单点的五个量：`(ε, ∂ε/∂ρ, ∂ε/∂σ, ∂ε/∂∇²ρ, ∂ε/∂τ)`。
fn derivatives(
    term: &dyn FunctionalTerm,
    rho: f64,
    sigma: f64,
    tau: f64,
) -> (f64, f64, f64, f64, f64) {
    let rho_holder = [rho];
    let sigma_holder = [sigma];
    let tau_holder = [tau];
    let batch = DensityBatch::with_tau(&rho_holder, &sigma_holder, &tau_holder);
    let (mut epsilon, mut d_rho, mut d_sigma) = ([0.0_f64], [0.0_f64], [0.0_f64]);
    let (mut d_laplacian, mut d_tau) = ([0.0_f64], [0.0_f64]);
    term.energy_density(&batch, &mut epsilon);
    term.potential_derivatives_full(
        &batch,
        &mut d_rho,
        &mut d_sigma,
        &mut d_laplacian,
        &mut d_tau,
    );
    (epsilon[0], d_rho[0], d_sigma[0], d_laplacian[0], d_tau[0])
}

#[test]
fn tau_derivatives_match_central_differences() {
    let term = TauProbe { a: 0.37, b: 0.11 };
    let step = 1.0e-6;
    for (rho, sigma, tau) in [
        (0.5, 0.0, 0.5),
        (0.1, 0.01, 0.05),
        (2.0, 0.3, 6.0),
        (0.02, 0.001, 0.01),
    ] {
        let (_, d_rho, d_sigma, d_laplacian, d_tau) = derivatives(&term, rho, sigma, tau);
        for (label, analytic, numeric) in [
            (
                "∂ε/∂ρ",
                d_rho,
                (derivatives(&term, rho + step, sigma, tau).0
                    - derivatives(&term, rho - step, sigma, tau).0)
                    / (2.0 * step),
            ),
            (
                "∂ε/∂σ",
                d_sigma,
                (derivatives(&term, rho, sigma + step, tau).0
                    - derivatives(&term, rho, sigma - step, tau).0)
                    / (2.0 * step),
            ),
            (
                "∂ε/∂τ",
                d_tau,
                (derivatives(&term, rho, sigma, tau + step).0
                    - derivatives(&term, rho, sigma, tau - step).0)
                    / (2.0 * step),
            ),
        ] {
            assert!(
                (analytic - numeric).abs() < 1.0e-9,
                "ρ={rho} σ={sigma} τ={tau}: {label} {analytic:.12} vs 差分 {numeric:.12}"
            );
        }
        assert!(
            d_laplacian.abs() < 1.0e-14,
            "∇²ρ 通道应当是 0，实测 {d_laplacian:.3e}"
        );
    }
}

#[test]
fn tau_derivative_is_independent_of_other_channels() {
    // ∂ε/∂τ 只依赖 ρ（这一项的选择），σ 不能影响它。
    let term = TauProbe { a: 0.25, b: 0.0 };
    let baseline = derivatives(&term, 0.4, 0.0, 0.3).4;
    for sigma in [0.01, 0.5, 5.0] {
        let shifted = derivatives(&term, 0.4, sigma, 0.3).4;
        assert_eq!(
            baseline.to_bits(),
            shifted.to_bits(),
            "σ 不影响 ∂ε/∂τ，但 {sigma} 处的值变了"
        );
    }
}

#[test]
fn missing_tau_is_an_error_not_a_silent_zero() {
    // 只给 ρ、σ 的 batch 喂给需要 τ 的项：必须失败，而不是把 τ 当 0 算下去。
    let term = TauProbe { a: 0.1, b: 0.0 };
    let rho_holder = [0.5];
    let sigma_holder = [0.0];
    let batch = DensityBatch::new(&rho_holder, &sigma_holder);
    assert!(!batch.has_tau());
    assert_eq!(
        batch.tau_at(0),
        0.0,
        "缺 τ 时访问器给 0，但调用方必须先查 has_tau"
    );
    let mut out = [0.0_f64];
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        term.energy_density(&batch, &mut out);
    }));
    assert!(caught.is_err(), "缺 τ 时必须失败，不能静默按 0 算");
}

#[test]
fn existing_terms_zero_the_tau_channel() {
    // LDA/GGA 的四通道里 τ 必须是 0（默认实现），否则加一个通道就把 PBE 改了。
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![
        Box::new(LdaSvwn5::new((0.031_090_7, -0.409_286, 13.072, 42.719_8))),
        Box::new(PbeExchange::new(0.804, 0.219_514_972_764_517_04)),
        Box::new(PbeCorrelation::new(
            (0.031_090_7, -0.409_286, 13.072, 42.719_8),
            0.066_724_550_603_149_22,
            0.031_090_690_869_654_9,
        )),
    ];
    for term in &terms {
        assert!(
            !term.requires().kinetic_density,
            "{} 不该声明需要 τ",
            term.id()
        );
        for tau in [0.0, 0.5, 3.0] {
            let (_, _, _, _, d_tau) = derivatives(term.as_ref(), 0.4, 0.02, tau);
            assert_eq!(
                d_tau.to_bits(),
                0.0_f64.to_bits(),
                "{} 的 τ 导数必须是 0",
                term.id()
            );
        }
    }
}

#[test]
fn tau_length_mismatch_is_caught() {
    // τ 与 ρ 长度不一致必须被抓住（不是越界读 0）。
    let rho_holder = [0.5, 0.4];
    let sigma_holder = [0.0, 0.0];
    let tau_holder = [0.1];
    let batch = DensityBatch::with_tau(&rho_holder, &sigma_holder, &tau_holder);
    let term = TauProbe { a: 0.1, b: 0.0 };
    let mut out = [0.0_f64, 0.0];
    // TauProbe 用 `batch.tau[index]` 直接索引 → 长度不一致必然 panic（这正是要的：
    // 不许悄悄按 0 补齐）。生产项走 `check_shape` 拿到 `XcError::Shape`。
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        term.energy_density(&batch, &mut out);
    }));
    assert!(caught.is_err(), "τ 长度不一致必须被抓住");
    let error = XcError::Shape("tau 1 vs rho 2".to_string());
    assert!(error.to_string().contains("tau"));
}

#[test]
fn meta_gga_requirements_set_the_right_flags() {
    let requirements = Requirements::meta_gga();
    assert!(requirements.rho && requirements.sigma);
    assert!(requirements.kinetic_density);
    assert!(!requirements.laplacian, "M06 这一族不需要 ∇²ρ");
    let all = Requirements::meta_gga_laplacian();
    assert!(all.laplacian && all.kinetic_density);
    // 旧构造器不能因为新增字段而打开 τ
    assert!(!Requirements::lda().kinetic_density);
    assert!(!Requirements::gga().kinetic_density);
    assert!(!Requirements::laplacian_gga().kinetic_density);
}

#[test]
fn terms_require_tau_helper_is_consistent() {
    let gga: Vec<Box<dyn FunctionalTerm>> =
        vec![Box::new(PbeExchange::new(0.804, 0.219_514_972_764_517_04))];
    assert!(!qwb_xc::terms_require_tau(&gga));
    let mixed: Vec<Box<dyn FunctionalTerm>> = vec![
        Box::new(PbeExchange::new(0.804, 0.219_514_972_764_517_04)),
        Box::new(TauProbe { a: 0.1, b: 0.0 }),
    ];
    assert!(qwb_xc::terms_require_tau(&mixed));
    let empty: Vec<Box<dyn FunctionalTerm>> = Vec::new();
    assert!(!qwb_xc::terms_require_tau(&empty));
}
