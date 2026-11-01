//! DA-06 的 **每个元素对都有测试**（`project-phase2.md` §6 的 DA-06 判据第 ② 条）。
//!
//! D3 那半的「元素对」是数据表里的 666 个 `C6` 键，所以测试可以直接逐个 diff；D4 的 C6
//! **不是**元素对常量 —— 它由每个参考体系的 `alpha(iω)` 现算、再按 CN/电荷加权的参考权重
//! 组合。所以"D4 的每个元素对都有测试"的含义是：**对 H–Kr 的每一对元素，中间量链路都跑得通、
//! 给得出有限的配位数、电荷与正的 C6，并且 C6 对称**。
//!
//! 这是一个"覆盖面"测试而不是精度测试（精度由 `da06_d4_properties.rs` 对官方中间量钉住）：
//! 它的价值在于**任何一对元素**（包括只有单条参考体系的稀有气体、以及 `z_i == z_j` 的
//! 同核对）都不会掉进"未覆盖 / 索引错位 / EEQ 系统奇异"的分支里。D4 的参考体系条数是逐元素
//! 变化的（2 ~ 7 条），所以 `nref_i × nref_j` 的形状每对都不同 —— 这类错误只会在某些对上出现。

use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_registry::elements::atomic_number;
use qwb_scf::d4;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

#[test]
fn every_element_pair_of_the_d4_data_produces_finite_intermediates() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_d4_dispersion("d4").expect("d4 data");
    let symbols: Vec<&str> = data.elements.iter().map(String::as_str).collect();
    assert_eq!(symbols.len(), 36, "D4 is staged for H-Kr");

    // 距离取 5 Bohr：远大于任何一条共价半径之和，所以 CN 接近 0（这是**最吃力**的输入 ——
    // 参考权重的高斯几乎全落在第一个参考体系上，`zeta` 也接近纯指数）。
    let distance_bohr = 5.0_f64;
    let mut pairs = 0usize;
    let mut smallest_c6 = f64::INFINITY;
    let mut largest_c6 = 0.0_f64;
    let mut worst_charge_sum = 0.0_f64;
    for (i, left) in symbols.iter().enumerate() {
        for right in symbols.iter().skip(i) {
            // 用数据自己的元素表反查 Z，确保索引与 `eeq.*` / `reference` 的排序一致。
            let zi = atomic_number(left).expect("known element");
            let zj = atomic_number(right).expect("known element");
            assert_eq!(zi as usize - 1, i, "elements must be sorted by Z");
            let positions = [[0.0, 0.0, 0.0], [distance_bohr, 0.0, 0.0]];
            let kinds = [*left, *right];
            let properties = d4::properties(data, &kinds, &positions, 0.0)
                .unwrap_or_else(|error| panic!("{left}-{right} (Z {zi}-{zj}): {error}"));

            for value in &properties.coordination_numbers {
                assert!(
                    value.is_finite() && *value >= 0.0,
                    "{left}-{right}: CN = {value} is not a finite non-negative number"
                );
            }
            let total: f64 = properties.charges.iter().sum();
            worst_charge_sum = worst_charge_sum.max(total.abs());
            assert!(
                properties.charges.iter().all(|q| q.is_finite()),
                "{left}-{right}: non-finite EEQ charge"
            );

            let c6 = properties.c6[0][1];
            assert!(
                c6.is_finite() && c6 > 0.0,
                "{left}-{right} (Z {zi}-{zj}): C6 = {c6} must be finite and positive"
            );
            // 对称性：C6 是元素对的函数，与哪个原子在第一位无关。
            let symmetric = d4::properties(data, &[right, left], &positions, 0.0)
                .unwrap_or_else(|error| panic!("{right}-{left}: {error}"))
                .c6[0][1];
            assert!(
                (c6 - symmetric).abs() <= 1.0e-12 * c6.abs().max(1.0),
                "{left}-{right}: C6 is not symmetric ({c6} vs {symmetric})"
            );

            smallest_c6 = smallest_c6.min(c6);
            largest_c6 = largest_c6.max(c6);
            pairs += 1;
        }
    }
    println!(
        "D4 element pairs covered: {pairs} (expected 666 for Z = 1..36); C6 range at 5 Bohr: \
         {smallest_c6:.6e} .. {largest_c6:.6e} Ha*Bohr^6; worst |Σq| = {worst_charge_sum:.2e}"
    );
    assert_eq!(
        pairs,
        36 * 37 / 2,
        "every i <= j pair of H-Kr must be covered"
    );
    assert!(
        worst_charge_sum < 1.0e-9,
        "neutral pairs must have Σq = 0, worst |Σq| = {worst_charge_sum:.3e}"
    );
}

#[test]
fn d4_rejects_elements_outside_the_staged_range() {
    // 数据覆盖 H-Kr；Rb 及以后必须**显式报错**，而不是给出一个 0 或者静默跳过。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_d4_dispersion("d4").expect("d4 data");
    let error = d4::properties(data, &["H", "Rb"], &[[0.0; 3], [5.0, 0.0, 0.0]], 0.0)
        .expect_err("Rb is outside the staged D4 range");
    println!("Rb: {error}");
    assert!(
        error.contains("Rb") && error.contains("outside"),
        "the error must name the element and say it is outside the data range: {error}"
    );
}
