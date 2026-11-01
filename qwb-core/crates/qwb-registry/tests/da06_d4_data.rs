//! DA-06 的 **D4 数据层**验收：文件能加载、形状与记录一致、派生量自洽。
//!
//! 这一条守的是"数据先钉住"那一步：`qwb data verify` 已经查过 schema 与结构，这里再查
//! ①`Registry` 的加载路径真的能把它读进来（不是只有 `data verify` 认），②**数量**与上游一致
//! （36 个元素 / 119 个参考体系 / 5 个泛函），③派生量 `r4r2 = sqrt(0.5·<r⁴>/<r²>·sqrt(Z))`
//! 与文件里两个字段自洽，④上游缺参数的泛函**有记录**（`m06-2x`），而不是悄悄缺席。
//!
//! 为什么值得一条独立测试：D4 的数据文件是生成器产物 + `qwb-stamp-data` 重新序列化过，
//! 一旦有人在生成器里改错下标（例如把 `reference` 与 `elements` 错一位），`data verify` 的
//! 结构检查**看不出来**（长度一样、每项都合法），而这条测试会红。

use std::path::PathBuf;

use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

#[test]
fn d4_data_loads_with_the_documented_shape() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let d4 = registry
        .get_d4_dispersion("d4")
        .expect("the registry must index the D4 data set (d4_dispersions in registry.yaml)");

    assert_eq!(d4.name, "d4");
    assert_eq!(d4.kind, "d4");
    assert_eq!(d4.elements.len(), 36, "D4 is staged for H-Kr");
    assert_eq!(d4.elements.first().map(String::as_str), Some("H"));
    assert_eq!(d4.elements.last().map(String::as_str), Some("Kr"));

    // 逐元素的四个 EEQ 数组与 elements 同序同长。
    for (label, values) in [
        ("chi", &d4.eeq.chi),
        ("eta", &d4.eeq.eta),
        ("kcnchi", &d4.eeq.kcnchi),
        ("rad", &d4.eeq.rad),
    ] {
        assert_eq!(values.len(), d4.elements.len(), "eeq.{label}");
    }

    // 上游 v4.3.1 的参考体系总数（`reference.inc` 里 36 个元素合计 119 条）。
    let reference_systems: usize = d4
        .reference
        .iter()
        .map(|element| element.reference_systems.len())
        .sum();
    assert_eq!(reference_systems, 119, "reference systems across H-Kr");

    // 每个参考体系的 alpha(iω) 点数等于频率网格点数。
    let points = d4.frequency_grid.hartree.len();
    assert_eq!(points, d4.frequency_grid.trapzd_weights.len());
    for element in &d4.reference {
        for system in &element.reference_systems {
            assert_eq!(system.alpha_iw.len(), points, "{}", element.element);
            assert_eq!(system.alpha_iw_eeq.len(), points, "{}", element.element);
        }
    }

    // `r4r2` 是派生量：sqrt(0.5 * <r4>/<r2> * sqrt(Z))，与 `r4_over_r2` 必须自洽。
    for (index, element) in d4.reference.iter().enumerate() {
        let z = (index + 1) as f64;
        let expected = (0.5 * element.r4_over_r2 * z.sqrt()).sqrt();
        assert!(
            (element.r4r2 - expected).abs() < 1.0e-12,
            "{}: r4r2 = {} but sqrt(0.5*r4_over_r2*sqrt(Z)) = {expected}",
            element.element,
            element.r4r2
        );
    }

    // 五个泛函有阻尼参数；上游没有参数的那个**有记录**，且不出现在参数表里。
    let mut names: Vec<&str> = d4.damping_parameters.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(names, ["b3lyp", "hf", "m06-l", "pbe", "pbe0"]);
    assert!(
        d4.unsupported_functionals.contains_key("m06-2x"),
        "M06-2X has no D4 parameters upstream and must be recorded, not invented"
    );
    assert!(!d4.damping_parameters.contains_key("m06-2x"));
    for parameters in d4.damping_parameters.values() {
        assert!(parameters.s6.is_finite() && parameters.s8.is_finite());
        assert!(parameters.a1.is_finite() && parameters.a2.is_finite());
    }

    // ATM 的 C6 用 q = 0 重建（上游 disp.f90:139-141）；这条被写进了 `check_structure`。
    assert_eq!(d4.dispersion.atm.charge_for_c6, 0.0);
    assert!(d4.dispersion.atm.enabled_by_default);

    // 溯源：47 个上游文件 + 三个仓库的 tag/commit。
    assert!(d4.provenance.upstream.len() >= 40, "27-ish files at least");
    assert!(d4.provenance.pinned.contains_key("dftd4/dftd4"));

    let issues = d4.check();
    assert!(issues.is_empty(), "{issues:?}");
}

#[test]
fn d4_data_is_a_separate_kind_from_d3() {
    // D3 的两个文件仍然按 D3 的类型加载；D4 不被当成 D3（`kind` 不同、类型不同）。
    let registry = Registry::open(data_root()).expect("qwb-data");
    assert!(registry.get_dispersion("d3bj").is_ok());
    assert!(registry.get_dispersion("d3zero").is_ok());
    assert!(
        registry.get_dispersion("d4").is_err(),
        "D4 must not be reachable through the D3 accessor"
    );
    assert!(
        registry.get_d4_dispersion("d3bj").is_err(),
        "D3 must not be reachable through the D4 accessor"
    );
}
