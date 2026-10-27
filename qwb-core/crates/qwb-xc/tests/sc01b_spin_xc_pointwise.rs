//! SC-01b（UKS 的 XC 侧）：**逐点对照 libxc** 的自旋分辨 XC 判据。
//!
//! 判据（phase-2 §4.4 / §9.2 G9，M1b 的第三条断言）：
//!
//! 1. **交换项的自旋标度**（`LdaSlaterX` / `GgaPbeX`）：与 libxc 的 `LDA_X` / `GGA_X_PBE`
//!    逐点一致 —— `e = ρ_α ε(2ρ_α, 4σ_αα) + ρ_β ε(2ρ_β, 4σ_ββ)`、`v_σ = v(2ρ_σ)`；
//! 2. **相关项的自旋插值**（`LdaSvwn5` / `GgaPbeC`）：与 libxc 的
//!    `LDA_C_VWN_RPA` / `GGA_C_PBE` 逐点一致 —— VWN 是两端插值、PBE 是 PW92 的 ζ⁴ 插值，
//!    两者的式子**不一样**，所以各有独立断言；
//! 3. **σ 通道**：`∂e/∂σ_αα`、`∂e/∂σ_ββ`、`∂e/∂σ_αβ` 三个都要对（漏 `∂/∂σ_αβ`
//!    在 ζ = 0 时看不出来，在 O2 上会偏 1e-2 Ha 量级）。
//!
//! 参考数据由 `scripts/pyscf_spin_xc_reference.py` 用 PySCF 调 libxc 生成
//! （`tests/reference/spin_xc_pointwise.csv`），黑盒、不可手改 —— 与 X-01/X-02 的
//! "自算值不能冒充参考值" 是同一条纪律。
//!
//! # 三处已知的容差来源（都不是公式问题）
//!
//! 1. `svwn5.yaml` 的参数只有 7 位有效数字、`pbe.yaml` 的 `pbe_mu = 0.21951` 只有 6 位，
//!    而 libxc 用全精度 → `LDA_C_VWN_RPA` / `GGA_X_PBE` 的残留差在 1e-7～2e-6 量级；
//! 2. `svwn5.yaml` 的参数集实际上是 libxc 的 **VWN RPA**（`LDA_C_VWN_RPA`），
//!    不是 libxc 叫 `LDA_C_VWN` 的那个（通常称 VWN5）：两者在 rs ≈ 0.7 处差 0.02 Ha/e。
//!    本测试钉住的是"我们的实现与它声明的那个 libxc 条目一致"，命名差异记在
//!    `docs/choices.md` 五十；
//! 3. ζ = ±1 的点上 `φ'(ζ)` 真实发散，`gga_spin` 用 ζ 的数值夹紧给出有界替代值，
//!    所以那些点只断言能量、不断言势（`strict_polarized = false`）。

use qwb_xc::{
    DensityBatch, FunctionalTerm, LdaSlaterX, LdaSvwn5, SpinDensityBatch, SpinXcOutput,
    gga::{PbeCorrelation, PbeExchange},
};

/// 与 `qwb-data/functional/svwn5.yaml` 一致（libxc 的 VWN RPA 参数集）。
const VWN_PARA: (f64, f64, f64, f64) = (0.031_090_7, -0.409_286, 13.072, 42.719_8);
const VWN_FERRO: (f64, f64, f64, f64) = (0.015_545_3, -0.743_294, 20.123_1, 101.578);
const VWN_ALPHA: (f64, f64, f64, f64) =
    (-0.053_051_647_697_298_45, -0.004_758_4, 1.131_07, 13.004_5);

/// 与 `qwb-data/functional/pbe.yaml` 一致。
const PBE_BETA: f64 = 0.066_724_550_603_149_22;
const PBE_GAMMA: f64 = 0.031_090_690_869_654_9;
const PBE_KAPPA: f64 = 0.804;
/// PBE 1996 的 `mu = beta*pi^2/3`（与 `qwb-data/functional/pbe.yaml` 一致，
/// 17 位；旧的 `0.21951` 只有 6 位，会让 GGA_X_PBE 的残差停在 2e-6）。
const PBE_MU: f64 = 0.219_514_972_764_517_04;

fn pbe_correlation() -> PbeCorrelation {
    PbeCorrelation::new(VWN_PARA, PBE_BETA, PBE_GAMMA)
}

fn reference_rows() -> Vec<Vec<String>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/reference/spin_xc_pointwise.csv"
    );
    let text = std::fs::read_to_string(path).expect("read spin_xc_pointwise.csv");
    let mut rows = Vec::new();
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(line.split(',').map(str::to_string).collect::<Vec<_>>());
    }
    rows
}

fn field(row: &[String], index: usize) -> f64 {
    row[index].parse::<f64>().expect("numeric csv field")
}

/// 单点求值：（每体积能量密度、v_α、v_β、∂e/∂σ_αα、∂e/∂σ_ββ、∂e/∂σ_αβ）。
fn evaluate(
    term: &dyn FunctionalTerm,
    rho_alpha: f64,
    rho_beta: f64,
    sigma_alpha: f64,
    sigma_beta: f64,
    sigma_alpha_beta: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let rho_a = [rho_alpha];
    let rho_b = [rho_beta];
    let sigma_aa = [sigma_alpha];
    let sigma_bb = [sigma_beta];
    let sigma_ab = [sigma_alpha_beta];
    let batch = SpinDensityBatch::with_sigma(&rho_a, &rho_b, &sigma_aa, &sigma_bb, &sigma_ab);
    let (mut energy, mut v_alpha, mut v_beta) = ([0.0_f64], [0.0_f64], [0.0_f64]);
    let (mut d_aa, mut d_bb, mut d_ab) = ([0.0_f64], [0.0_f64], [0.0_f64]);
    let mut out = SpinXcOutput::new(
        &mut energy,
        &mut v_alpha,
        &mut v_beta,
        &mut d_aa,
        &mut d_bb,
        &mut d_ab,
    );
    term.evaluate_spin(&batch, &mut out)
        .unwrap_or_else(|err| panic!("evaluate_spin failed: {err}"));
    (energy[0], v_alpha[0], v_beta[0], d_aa[0], d_bb[0], d_ab[0])
}

/// 容差与严格性。
#[derive(Debug, Clone, Copy)]
struct Tolerances {
    energy: f64,
    potential: f64,
    sigma: f64,
    /// 相对容差：`|差| ≤ abs + relative·|参考值|`。
    /// 数据参数只有 6–7 位有效数字，而势与 σ 导数的量级可以到 O(1)，
    /// 所以必须给相对容差，否则"数据精度"会被误判成"公式错"。
    relative: f64,
    strict_polarized: bool,
}

fn check(functional: &str, term: &dyn FunctionalTerm, tolerances: Tolerances) {
    let rows = reference_rows();
    let mut checked = 0usize;
    let mut skipped_polarized = 0usize;
    for row in rows.iter().filter(|row| row[0] == functional) {
        let rho_alpha = field(row, 1);
        let rho_beta = field(row, 2);
        let sigma_alpha = field(row, 3);
        let sigma_beta = field(row, 4);
        let sigma_alpha_beta = field(row, 5);
        let expected = (
            field(row, 6),
            field(row, 7),
            field(row, 8),
            field(row, 9),
            field(row, 10),
            field(row, 11),
        );
        let got = evaluate(
            term,
            rho_alpha,
            rho_beta,
            sigma_alpha,
            sigma_beta,
            sigma_alpha_beta,
        );
        let case = format!(
            "{functional} ρ=({rho_alpha},{rho_beta}) σ=({sigma_alpha},{sigma_beta},{sigma_alpha_beta})"
        );
        assert!(
            (got.0 - expected.0).abs() < tolerances.energy,
            "{case}: 能量密度 {:.15} vs libxc {:.15}",
            got.0,
            expected.0
        );
        let polarized = (rho_alpha - rho_beta).abs() >= (rho_alpha + rho_beta) - 1.0e-12;
        if polarized && !tolerances.strict_polarized {
            skipped_polarized += 1;
            checked += 1;
            continue;
        }
        let close = |value: f64, reference: f64, absolute: f64| {
            (value - reference).abs() <= absolute + tolerances.relative * reference.abs()
        };
        assert!(
            close(got.1, expected.1, tolerances.potential),
            "{case}: v_α {:.15} vs libxc {:.15}",
            got.1,
            expected.1
        );
        assert!(
            close(got.2, expected.2, tolerances.potential),
            "{case}: v_β {:.15} vs libxc {:.15}",
            got.2,
            expected.2
        );
        // `sigma = 0.0` 表示这一项没有 σ 通道（LDA），跳过而不是让断言恒假。
        if tolerances.sigma > 0.0 {
            for (what, value, reference) in [
                ("∂e/∂σ_αα", got.3, expected.3),
                ("∂e/∂σ_ββ", got.4, expected.4),
                ("∂e/∂σ_αβ", got.5, expected.5),
            ] {
                assert!(
                    close(value, reference, tolerances.sigma),
                    "{case}: {what} {value:.15} vs libxc {reference:.15}"
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 10, "{functional}: 只有 {checked} 行参考数据");
    println!("{functional}: {checked} 行一致（其中 {skipped_polarized} 行全极化点只查能量）");
}

#[test]
fn slater_exchange_matches_libxc() {
    // Slater α = 2/3（数据里的 slater_alpha）
    let term = LdaSlaterX::new(2.0 / 3.0);
    check(
        "LDA_X",
        &term,
        Tolerances {
            energy: 1.0e-12,
            potential: 1.0e-12,
            sigma: 0.0,
            relative: 1.0e-12,
            strict_polarized: true,
        },
    );
}

#[test]
fn vwn_correlation_matches_libxc_rpa() {
    let term = LdaSvwn5::new(VWN_PARA).with_spin_parameters(VWN_FERRO, VWN_ALPHA);
    // 容差 1e-6：数据里 VWN 参数只有 7 位有效数字，残留差来自参数精度（公式差为 0）。
    check(
        "LDA_C_VWN_RPA",
        &term,
        Tolerances {
            energy: 1.0e-6,
            potential: 1.0e-6,
            sigma: 0.0,
            relative: 1.0e-4,
            strict_polarized: true,
        },
    );
}

#[test]
fn pbe_correlation_at_zero_sigma_matches_pw92() {
    // σ = 0 时 PBE 相关的梯度修正 H = 0，剩下的就是 PW92 的 LDA 基。
    let term = pbe_correlation();
    check(
        "LDA_C_PW_MOD",
        &term,
        Tolerances {
            energy: 1.0e-12,
            potential: 1.0e-12,
            sigma: 0.0,
            relative: 1.0e-12,
            strict_polarized: false,
        },
    );
}

#[test]
fn pbe_exchange_matches_libxc() {
    let term = PbeExchange::new(PBE_KAPPA, PBE_MU);
    // μ 已经换成精确的 βπ²/3，所以这里可以按浮点精度要求（原来的 1e-6 是因为
    // 数据里 `pbe_mu = 0.21951` 只有 6 位有效数字）。
    check(
        "GGA_X_PBE",
        &term,
        Tolerances {
            energy: 1.0e-12,
            potential: 1.0e-12,
            sigma: 1.0e-12,
            relative: 1.0e-12,
            strict_polarized: true,
        },
    );
}

#[test]
fn pbe_correlation_matches_libxc() {
    let term = pbe_correlation();
    check(
        "GGA_C_PBE",
        &term,
        Tolerances {
            energy: 1.0e-9,
            potential: 1.0e-9,
            sigma: 1.0e-9,
            relative: 1.0e-9,
            strict_polarized: false,
        },
    );
}

/// 闭壳层退化：`ρ_α = ρ_β`、`σ_αα = σ_ββ = σ_αβ = σ/4` 时，
/// 自旋版本的能量密度与闭壳层逐位相同（UKS ≡ RKS 的基础）。
///
/// σ 通道的退化**不是逐分量相等**：`∂e/∂σ_αα` 是"固定 σ_αβ"的偏导，
/// 而闭壳层 `∂e/∂σ` 是沿 `(σ/4, σ/4, σ/4)` 方向的全微分。两者的关系是
/// `(d_αα + d_ββ + d_αβ)/4 = ρ dε/dσ`（三个 σ 分量对 σ 的偏导都是 1/4）。
#[test]
fn unpolarized_spin_matches_the_closed_shell_path() {
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![
        Box::new(LdaSlaterX::new(2.0 / 3.0)),
        Box::new(LdaSvwn5::new(VWN_PARA).with_spin_parameters(VWN_FERRO, VWN_ALPHA)),
        Box::new(PbeExchange::new(PBE_KAPPA, PBE_MU)),
        Box::new(pbe_correlation()),
    ];
    for rho in [0.02, 0.2, 1.0, 3.0] {
        for sigma in [0.0, 0.01, 0.3] {
            let rho_holder = [rho];
            let sigma_holder = [sigma];
            let closed_batch = DensityBatch::new(&rho_holder, &sigma_holder);
            let (mut closed_epsilon, mut closed_d_rho, mut closed_d_sigma) =
                ([0.0_f64], [0.0_f64], [0.0_f64]);
            for term in &terms {
                term.energy_density(&closed_batch, &mut closed_epsilon);
                term.potential_derivatives(&closed_batch, &mut closed_d_rho, &mut closed_d_sigma);
                let quarter = 0.25 * sigma;
                let got = evaluate(
                    term.as_ref(),
                    0.5 * rho,
                    0.5 * rho,
                    quarter,
                    quarter,
                    quarter,
                );
                // 每体积能量密度：闭壳层给每电子 ε，所以 e = ρ·ε。
                assert!(
                    (got.0 - rho * closed_epsilon[0]).abs() < 1.0e-12,
                    "ρ={rho} σ={sigma} {}: 自旋 {:.15} vs 闭壳层 {:.15}",
                    term.id(),
                    got.0,
                    rho * closed_epsilon[0]
                );
                // ζ = 0 时两个通道的势相同，且等于闭壳层的 v = ε + ρ dε/dρ。
                let expected_v = closed_epsilon[0] + rho * closed_d_rho[0];
                for (what, value) in [("v_α", got.1), ("v_β", got.2)] {
                    assert!(
                        (value - expected_v).abs() < 1.0e-10,
                        "ρ={rho} σ={sigma} {}: {what} {value:.15} vs 闭壳层 {expected_v:.15}",
                        term.id()
                    );
                }
                // σ 通道的投影关系（见上面的文档注释）。
                let projected = 0.25 * (got.3 + got.4 + got.5);
                let expected_sigma = rho * closed_d_sigma[0];
                assert!(
                    (projected - expected_sigma).abs() < 1.0e-10,
                    "ρ={rho} σ={sigma} {}: σ 投影 {projected:.15} vs 闭壳层 {expected_sigma:.15}",
                    term.id()
                );
            }
        }
    }
}
