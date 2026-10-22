//! D-09 验收（project.md §6.2）：`SVWN5` 正式数据。
//!
//! 验收原文要求 "X-02 有限差分导数 < 1e-9；参数逐个带 citation"。X-02 的导数门禁属于
//! X 轨（依赖 X-01 的 `FunctionalTerm`），这里先做**数据侧**能做的全部：
//!
//! 1. 每个参数都有 `parameter_sources` 条目（逐个带 citation）；
//! 2. 13 个常数逐位等于 VWN5（parameterization V）的权威值 —— 抓转录笔误（R-04）；
//! 3. 用数据文件里的常数**独立求值**，与 Ceperley-Alder 顺磁均匀电子气的相关能
//!    对照（rs = 1 处 -0.0799 Ha）；同时做**负对照**：把 VWN3/RPA 那套常数代进去
//!    必须**明显偏出**容差 —— 证明本测试真的能分辨 VWN5 与 VWN3；
//! 4. 参数ization 的数值指纹：`Q² = 4c - b²` 对 parameterization V 极小但为正。
//!
//! §7.0 说明：第 3 条用的是教科书常引的 C-A 值（外部引用），容差 5 mHa；
//! 更严格的门禁（有限差分导数 < 1e-9）由 X-02 在 X 轨执行，不在此处伪造。

use std::path::{Path, PathBuf};

use qwb_registry::Registry;
use qwb_registry::functional::Functional;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn data_file() -> PathBuf {
    repo_root().join("qwb-data/functional/svwn5.yaml")
}

fn load() -> Functional {
    let text =
        std::fs::read_to_string(data_file()).expect("qwb-data/functional/svwn5.yaml 必须存在");
    let functional: Functional = serde_yaml::from_str(&text).expect("合法 K1 泛函文件");
    assert!(
        functional.check().is_empty(),
        "必须过完整校验（G2 + 结构）：{:?}",
        functional.check()
    );
    functional
}

fn param(functional: &Functional, name: &str) -> f64 {
    *functional
        .parameters
        .get(name)
        .unwrap_or_else(|| panic!("缺参数 {name}"))
}

/// VWN 相关能（每电子，Ha），按参数自行求值 —— 测试侧的独立实现，不是产品代码。
fn vwn_eps_c(rs: f64, a: f64, x0: f64, b: f64, c: f64) -> f64 {
    let x = rs.sqrt();
    let xpoly = |t: f64| t * t + b * t + c;
    let q = (4.0 * c - b * b).sqrt();
    a * ((x * x / xpoly(x)).ln() - (b * x0 / xpoly(x0)) * (((x - x0) * (x - x0)) / xpoly(x)).ln()
        + (2.0 * b * (c - x0 * x0)) / (xpoly(x0) * q) * (q / (2.0 * x + b)).atan())
}

fn set(functional: &Functional, prefix: &str) -> (f64, f64, f64, f64) {
    (
        param(functional, &format!("vwn5_{prefix}_A")),
        param(functional, &format!("vwn5_{prefix}_x0")),
        param(functional, &format!("vwn5_{prefix}_b")),
        param(functional, &format!("vwn5_{prefix}_c")),
    )
}

// ─────────────────── 1. 参数逐个带 citation ───────────────────

#[test]
fn every_parameter_has_a_source() {
    let functional = load();
    assert!(
        !functional.parameters.is_empty(),
        "SVWN5 必须有参数（否则就是纯 Slater 交换）"
    );
    for name in functional.parameters.keys() {
        let source = functional.parameter_sources.get(name).unwrap_or_else(|| {
            panic!("参数 {name} 没有 parameter_sources 条目（D-09 要求逐个带 citation）")
        });
        assert!(
            source.len() > 10,
            "参数 {name} 的来源太短，不可核查: {source:?}"
        );
    }
    // 反向：不能有悬空来源。
    for name in functional.parameter_sources.keys() {
        assert!(
            functional.parameters.contains_key(name),
            "parameter_sources.{name} 指向不存在的参数"
        );
    }
    assert_eq!(
        functional.parameters.len(),
        13,
        "1 个交换系数 + 3x4 个 VWN5 常数"
    );
}

// ─────────────── 2. 常数逐位等于 VWN5 (parameterization V) ───────────────

#[test]
fn constants_are_bitwise_the_vwn5_parameterization() {
    let functional = load();

    // 顺磁（para）/ 铁磁（ferro）/ 自旋刚度（alpha）三套，parameterization V。
    let expected = [
        ("p", 0.0310907, -0.409286, 13.0720, 42.7198),
        ("f", 0.0155453, -0.743294, 20.1231, 101.578),
        ("a", -0.053_051_647_697_298_45, -0.0047584, 1.13107, 13.0045),
    ];
    for (prefix, a, x0, b, c) in expected {
        let (a_actual, x0_actual, b_actual, c_actual) = set(&functional, prefix);
        assert_eq!(a_actual, a, "vwn5_{prefix}_A");
        assert_eq!(x0_actual, x0, "vwn5_{prefix}_x0");
        assert_eq!(b_actual, b, "vwn5_{prefix}_b");
        assert_eq!(c_actual, c, "vwn5_{prefix}_c");
    }

    // Slater 交换系数恰好 2/3。
    assert_eq!(param(&functional, "slater_alpha"), 2.0 / 3.0);

    // 组成项：交换 + VWN5 相关，权重都是 1。
    let ids: Vec<&str> = functional
        .components
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    assert_eq!(ids, vec!["LdaSlaterX", "LdaSvwn5"]);
    for component in &functional.components {
        assert_eq!(component.weight, 1.0, "{}", component.id);
    }

    // 纯 LDA：只要 rho，不需要 sigma/laplacian。
    assert!(functional.requires.rho);
    assert!(!functional.requires.sigma, "SVWN5 没有 sigma 依赖");
    assert!(!functional.requires.laplacian);
}

// ─────────── 3. 物理对照（含 VWN3 负对照，证明测试有分辨力）───────────

#[test]
fn vwn5_parameters_reproduce_the_ceperley_alder_correlation_energy() {
    let functional = load();
    let (a, x0, b, c) = set(&functional, "p");

    // Ceperley & Alder 1980 顺磁均匀电子气 related energy at rs = 1（Ha/electron）。
    // 教科书常引值 -0.0799；§7.0：外部引用 + 5 mHa 容差。
    let reference = -0.0799_f64;
    let computed = vwn_eps_c(1.0, a, x0, b, c);
    assert!(
        (computed - reference).abs() < 5.0e-3,
        "VWN5 参数在 rs=1 处给出 {computed:.6}，与 C-A {reference:.4} 差 {:.2e} Ha",
        (computed - reference).abs()
    );

    // 负对照：若有人把 VWN3/RPA 那套常数抄进来（网上有些表把它标成 VWN5），
    // 本测试必须失败。用同一个参考值检查。
    let vwn3 = (0.0310907_f64, -0.10498_f64, 3.72744_f64, 12.9352_f64);
    let wrong = vwn_eps_c(1.0, vwn3.0, vwn3.1, vwn3.2, vwn3.3);
    assert!(
        (wrong - reference).abs() > 1.0e-2,
        "负对照失效：VWN3 常数居然也落在容差内（{wrong:.6}），本测试失去分辨力"
    );

    // 参数ization 的数值指纹：parameterization V 的 Q² 极小但为正。
    for prefix in ["p", "f", "a"] {
        let (_, _, b, c) = set(&functional, prefix);
        let q_squared = 4.0 * c - b * b;
        assert!(q_squared > 0.0, "vwn5_{prefix}: Q^2 = {q_squared} 必须 > 0");
    }
    let (_, _, b_p, c_p) = set(&functional, "p");
    assert!(
        (4.0 * c_p - b_p * b_p) < 1.0e-2,
        "parameterization V 的顺磁 Q^2 应当极小（VWN3 是 ~37.8）"
    );

    // 铁磁/顺磁的物理顺序：完全极化时相关能更小（更不负）。
    let (af, x0f, bf, cf) = set(&functional, "f");
    assert!(
        vwn_eps_c(1.0, af, x0f, bf, cf) > vwn_eps_c(1.0, a, x0, b, c),
        "eps_c(ferro) 应比 eps_c(para) 更接近 0"
    );
}

// ─────────────────── 4. 能被 Registry 按生产名取到 ───────────────────

#[test]
fn svwn5_is_registered_under_its_production_name() {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data must load");
    let functional = registry
        .get_functional("svwn5")
        .expect("生产名 svwn5 必须可加载");
    assert_eq!(functional.components.len(), 2);
    assert_eq!(functional.parameters.len(), 13);
    // 稳定性截断规则必须显式（§8.3 第三道门禁）。
    assert!(functional.stability.density_floor > 0.0);
    assert!(functional.stability.sigma_ceiling > 0.0);
}
