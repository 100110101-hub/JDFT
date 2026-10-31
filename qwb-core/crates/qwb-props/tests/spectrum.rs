//! PR-04 验收：展宽谱的**定量**性质。
//!
//! 三条恒等式（见 `qwb_props::spectrum` 的模块文档）：
//!
//! 1. 面积：`Σ ε·ΔE = Σf·HWHM·√(π/ln2)`（对称高斯核的解析面积，与网格无关）；
//! 2. 重心：`Σ E·ε / Σ ε = Σ f·E / Σ f`（对称核的一阶矩）；
//! 3. 峰位：孤立跃迁的极大值落在它的能量上（一个网格步长以内）。
//!
//! 再加两条端到端：数据驱动的默认值可读、双峰可分辨、CSV 可解析。

use std::path::PathBuf;

use qwb_props::spectrum::{Spectrum, SpectrumOptions, broadened_spectrum, gaussian_kernel};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// 一组典型的 CIS/TDA 跃迁（水：7.4 eV 与 9.9 eV）。
fn water_like() -> Vec<(f64, f64)> {
    vec![(7.41, 0.052), (9.88, 0.181), (12.31, 0.041)]
}

#[test]
fn gaussian_kernel_is_half_height_at_the_hwhm() {
    let hwhm = 0.4;
    assert!((gaussian_kernel(0.0, hwhm) - 1.0).abs() < 1e-15);
    assert!(
        (gaussian_kernel(hwhm, hwhm) - 0.5).abs() < 1e-14,
        "HWHM 处必须是半高：{}",
        gaussian_kernel(hwhm, hwhm)
    );
    assert!((gaussian_kernel(-hwhm, hwhm) - 0.5).abs() < 1e-14);
}

#[test]
fn sampled_area_matches_the_analytic_normalization() {
    let transitions = water_like();
    let options = SpectrumOptions::new(0.4, 0.01, 8.0);
    let spectrum = broadened_spectrum(&transitions, &options).expect("spectrum");
    let oscillator_sum: f64 = transitions.iter().map(|(_, f)| f).sum();
    let analytic = Spectrum::analytic_area(oscillator_sum, options.gaussian_hwhm_ev);
    let numeric = spectrum.area();
    let relative = (numeric - analytic).abs() / analytic;
    assert!(
        relative < 1e-9,
        "面积 {numeric} 与解析归一化 {analytic} 不符（相对差 {relative:.2e}）"
    );
}

#[test]
fn centroid_equals_the_f_weighted_mean_energy() {
    let transitions = water_like();
    let options = SpectrumOptions::new(0.3, 0.01, 8.0);
    let spectrum = broadened_spectrum(&transitions, &options).expect("spectrum");
    let total_f: f64 = transitions.iter().map(|(_, f)| f).sum();
    let expected: f64 = transitions.iter().map(|(e, f)| e * f).sum::<f64>() / total_f;
    let centroid = spectrum.centroid_ev().expect("centroid");
    assert!(
        (centroid - expected).abs() < 1e-9,
        "重心 {centroid} vs f 加权平均 {expected}"
    );
}

#[test]
fn isolated_transition_peaks_at_its_own_energy() {
    let options = SpectrumOptions::new(0.4, 0.02, 6.0);
    for energy in [4.0, 7.41, 11.25] {
        let spectrum = broadened_spectrum(&[(energy, 1.0)], &options).expect("spectrum");
        let peaks = spectrum.peak_energies_ev();
        assert_eq!(peaks.len(), 1, "单峰情形应只有一个局部极大值：{peaks:?}");
        assert!(
            (peaks[0] - energy).abs() <= 0.5 * options.grid_step_ev,
            "峰位 {} 偏离跃迁能量 {energy} 超过半步长",
            peaks[0]
        );
    }
}

#[test]
fn two_well_separated_states_give_two_peaks_at_their_energies() {
    let options = SpectrumOptions::new(0.25, 0.01, 6.0);
    let transitions = vec![(6.0, 0.5), (8.0, 0.7)];
    let spectrum = broadened_spectrum(&transitions, &options).expect("spectrum");
    let peaks = spectrum.peak_energies_ev();
    assert_eq!(peaks.len(), 2, "应分辨出两个峰：{peaks:?}");
    assert!((peaks[0] - 6.0).abs() <= 0.5 * options.grid_step_ev);
    assert!((peaks[1] - 8.0).abs() <= 0.5 * options.grid_step_ev);
}

#[test]
fn defaults_come_from_the_pinned_data() {
    let registry = Registry::open(data_root()).expect("registry");
    let options = SpectrumOptions::from_registry(&registry).expect("spectrum defaults");
    // 数据里写的值（`qwb-data/numerics/numerics.yaml` 的 `spectrum_defaults`）。
    assert!((options.gaussian_hwhm_ev - 0.4).abs() < 1e-15);
    assert!((options.grid_step_ev - 0.05).abs() < 1e-15);
    assert!((options.padding_hwhm - 6.0).abs() < 1e-15);
    options.validate().expect("defaults must be valid");
}

#[test]
fn invalid_options_are_rejected() {
    let transitions = vec![(7.0, 1.0)];
    // 步长大于 HWHM：会漏掉峰。
    let bad = SpectrumOptions::new(0.1, 0.5, 6.0);
    assert!(broadened_spectrum(&transitions, &bad).is_err());
    // 负的 HWHM。
    let bad = SpectrumOptions::new(-0.4, 0.05, 6.0);
    assert!(broadened_spectrum(&transitions, &bad).is_err());
    // 空跃迁表。
    let ok = SpectrumOptions::default();
    assert!(broadened_spectrum(&[], &ok).is_err());
}

#[test]
fn csv_round_trips_through_parsing() {
    let transitions = water_like();
    let options = SpectrumOptions::new(0.4, 0.05, 6.0);
    let spectrum = broadened_spectrum(&transitions, &options).expect("spectrum");
    let csv = spectrum.to_csv();
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some("energy_ev,intensity"));
    let mut count = 0;
    let mut previous = f64::NEG_INFINITY;
    for line in lines {
        let (energy, intensity) = line.split_once(',').expect("comma");
        let energy: f64 = energy.parse().expect("energy");
        let intensity: f64 = intensity.parse().expect("intensity");
        assert!(energy > previous, "能量必须递增");
        assert!(intensity >= 0.0 && intensity.is_finite());
        previous = energy;
        count += 1;
    }
    assert_eq!(count, spectrum.points.len());
}
