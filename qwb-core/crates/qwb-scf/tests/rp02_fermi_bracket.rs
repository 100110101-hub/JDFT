//! RP-02：费米展宽的化学势二分必须把根**夹住**，否则占据数会全部塌掉。
//!
//! 引擎的退火退路（`qwb-engine` 的第三级）在前 `smearing_steps` 步用费米权重占据，
//! 之后切回整数占据。权重来自 [`qwb_scf::rks::fermi_occupations`] 的一次二分：
//! `Σ_i f_i(μ) = n_electrons`。`fill(μ)` 对 μ 单调递增，所以「`fill(mid) > n` ⇒ 根在
//! 左边」；这两个分支原先写反了，二分于是**发散**到上界 `max + 40T`，占据数全部变成 0
//! （实测 `sum(f) = 0.000000`，T = 0.3、20 步退火），密度塌成零矩阵。
//!
//! 那个状态**看起来**像收敛：能量不再变、密度不再动 —— 但它不是自洽解，它连电子数
//! 都不对。这是 RP-02 的根因（PBE/def2-SVP 的 90° 弯 HCN 因此报了 -81.7122547541 Ha，
//! PySCF 是 -93.1721319955 Ha）。
//!
//! 这里钉两件事：
//!
//! 1. 一组 HCN/def2-SVP 形状的能级上，占据数之和**必须**等于电子数（T = 0.05–0.5）；
//! 2. 简并到极致的能级表上也一样（退化情形不该让二分跑偏）。
#![allow(clippy::unreadable_literal)]

use qwb_scf::rks::fermi_occupations;

/// HCN/def2-SVP 形状的能级表（占据区与虚区之间有 ~1.3 Ha 的 gap）。
fn hcn_levels() -> [f64; 20] {
    [
        -15.62, -11.03, -1.42, -0.86, -0.61, -0.53, -0.47, 0.12, 0.19, 0.35, 0.51, 0.62, 0.71,
        0.80, 1.02, 1.15, 1.31, 1.44, 1.60, 1.75,
    ]
}

#[test]
fn fermi_occupations_conserve_the_electron_count() {
    let levels = hcn_levels();
    for temperature in [0.05, 0.1, 0.3, 0.5] {
        let occupations = fermi_occupations(&levels, 14.0, temperature).expect("occupations");
        let total: f64 = occupations.iter().sum();
        println!(
            "T = {temperature:<5} sum(occ) = {total:.10}  occ = {:?}",
            &occupations[..8]
        );
        assert!(
            (total - 14.0).abs() < 1.0e-6,
            "T = {temperature}: Fermi occupations sum to {total}, not 14 electrons \
             (0 means the bisection walked mu into the empty region)"
        );
        // 每个权重都是合法的费米占据，落在 [0, 2]。
        for value in &occupations {
            assert!(
                value.is_finite() && *value >= 0.0 && *value <= 2.0,
                "occupation {value} is outside [0, 2]"
            );
        }
        // 这些温度下 1.3 Ha 的 gap 必须让最低能级基本是双占据。
        assert!(
            occupations[0] > 1.999,
            "T = {temperature}: the lowest level must be essentially doubly occupied, got {}",
            occupations[0]
        );
    }
}

#[test]
fn fermi_occupations_brackets_mu_for_a_degenerate_level_list() {
    // 紧致的极小值：没有正确的夹逼，二分会跑到空的那一端。
    let levels = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 5.0];
    let occupations = fermi_occupations(&levels, 8.0, 0.3).expect("occupations");
    let total: f64 = occupations.iter().sum();
    println!("sum(occ) = {total:.10}  occ = {occupations:?}");
    assert!(
        (total - 8.0).abs() < 1.0e-6,
        "Fermi occupations sum to {total}, not 8 electrons"
    );
}
