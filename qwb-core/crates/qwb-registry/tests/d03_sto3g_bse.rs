//! D-03 验收（project.md §6.2）：`STO-3G`（H,C,N,O,F）正式数据。
//!
//! ① 与 **BSE 导出**逐项 diff < 1e-12（相对）——`fixtures/bse/sto-3g.json` 是
//!    BSE API 的原始导出（committed 的 ground truth），数据文件由
//!    `scripts/bse_to_qwb_data.py` 从它生成；
//! ② 每元素基函数数 H=1、C/N/O/F=5；
//! ③ 球谐；
//! ④ 数据必须能被 `Registry` 按生产名 `sto-3g` 取到（不是 `example-` 前缀的样例）。

use std::path::{Path, PathBuf};

use qwb_registry::Registry;
use qwb_registry::basis::{Angular, BasisSet};
use qwb_registry::elements::atomic_number;

/// BSE 的原始导出（2018-06-19 revision，"Data from Gaussian09"）。
const BSE_EXPORT: &str = include_str!("fixtures/bse/sto-3g.json");

const ELEMENTS: [&str; 5] = ["H", "C", "N", "O", "F"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn data_file() -> PathBuf {
    repo_root().join("qwb-data/basis/sto-3g.yaml")
}

fn load() -> BasisSet {
    let text = std::fs::read_to_string(data_file()).expect("qwb-data/basis/sto-3g.yaml 必须存在");
    let basis: BasisSet = serde_yaml::from_str(&text).expect("必须是合法的 K1 基组文件");
    assert!(
        basis.check().is_empty(),
        "数据必须过完整校验（含 G2）：{:?}",
        basis.check()
    );
    basis
}

/// 一个收缩函数：`(l, exponents, coefficients)`。
type Triple = (u32, Vec<f64>, Vec<f64>);

/// 把 BSE 的 `electron_shells` 展开成与 K1 同形的 triple 列表。
fn bse_triples(shells: &serde_json::Value) -> Vec<Triple> {
    let mut out: Vec<Triple> = Vec::new();
    for shell in shells.as_array().expect("electron_shells 是数组") {
        assert_eq!(shell["function_type"], "gto", "只支持 gto");

        let angular: Vec<u32> = shell["angular_momentum"]
            .as_array()
            .expect("angular_momentum")
            .iter()
            .map(|value| u32::try_from(value.as_u64().expect("l 是整数")).expect("l fits u32"))
            .collect();
        let exponents: Vec<f64> = shell["exponents"]
            .as_array()
            .expect("exponents")
            .iter()
            .map(|value| parse_bse_number(value.as_str().expect("BSE 数字是字符串")))
            .collect();
        let columns: Vec<Vec<f64>> = shell["coefficients"]
            .as_array()
            .expect("coefficients")
            .iter()
            .map(|column| {
                column
                    .as_array()
                    .expect("coefficient 列")
                    .iter()
                    .map(|value| parse_bse_number(value.as_str().expect("BSE 数字是字符串")))
                    .collect()
            })
            .collect();

        if columns.len() == angular.len() {
            for (l, column) in angular.iter().zip(columns) {
                out.push((*l, exponents.clone(), column));
            }
        } else {
            assert_eq!(angular.len(), 1, "通用收缩只能有一个角动量：{angular:?}");
            for column in columns {
                out.push((angular[0], exponents.clone(), column));
            }
        }
    }
    out
}

/// 我们数据里展开成同形的 triple 列表。
fn qwb_triples(basis: &BasisSet, element: &str) -> Vec<Triple> {
    let block = basis
        .shells_for(element)
        .unwrap_or_else(|| panic!("基组缺元素 {element}"));
    let mut out: Vec<Triple> = Vec::new();
    for shell in &block.shells {
        for column in &shell.coefficients {
            out.push((shell.l, shell.exponents.clone(), column.clone()));
        }
    }
    out
}

/// BSE 的字符串数字（如 `0.3425250914E+01`）。
fn parse_bse_number(text: &str) -> f64 {
    text.parse::<f64>()
        .unwrap_or_else(|err| panic!("BSE 数字无法解析: {text:?}: {err}"))
}

fn sort_canonical(triples: &mut [Triple]) {
    triples.sort_by(|a, b| {
        a.0.cmp(&b.0).then_with(|| {
            b.1.first()
                .partial_cmp(&a.1.first())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
}

fn assert_rel_eq(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: 长度不同");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let scale = a.abs().max(e.abs());
        let diff = (a - e).abs();
        let relative = if scale == 0.0 { diff } else { diff / scale };
        assert!(
            relative <= 1e-12,
            "{what}[{index}]: {a} vs BSE {e}（相对差 {relative:e} > 1e-12）"
        );
    }
}

// ───────────────────── ① 与 BSE 逐项 diff < 1e-12 ─────────────────────

#[test]
fn sto3g_matches_the_bse_export_element_by_element() {
    let basis = load();
    let bse: serde_json::Value = serde_json::from_str(BSE_EXPORT).expect("BSE 导出是 JSON");

    assert_eq!(
        basis.elements,
        ELEMENTS.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "元素集合必须是 H,C,N,O,F"
    );

    for element in ELEMENTS {
        let z = atomic_number(element).expect("元素在第一阶段表里");
        let shells = &bse["elements"][z.to_string()]["electron_shells"];

        let mut expected = bse_triples(shells);
        let mut actual = qwb_triples(&basis, element);
        sort_canonical(&mut expected);
        sort_canonical(&mut actual);

        assert_eq!(
            actual.len(),
            expected.len(),
            "{element}: 收缩函数个数不同（K1 {} vs BSE {}）",
            actual.len(),
            expected.len()
        );

        for (index, ((l_actual, exp_actual, c_actual), (l_expected, exp_expected, c_expected))) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(
                l_actual, l_expected,
                "{element}[{index}]: 角动量不同（K1 {l_actual} vs BSE {l_expected}）"
            );
            assert_rel_eq(
                exp_actual,
                exp_expected,
                &format!("{element}[{index}] l={l_actual} exponents"),
            );
            assert_rel_eq(
                c_actual,
                c_expected,
                &format!("{element}[{index}] l={l_actual} coefficients"),
            );
        }
    }
}

// ─────────────────── ② 每元素基函数数（H=1，C/N/O/F=5）───────────────────

#[test]
fn per_element_basis_function_counts() {
    let basis = load();
    let expected = [("H", 1_usize), ("C", 5), ("N", 5), ("O", 5), ("F", 5)];
    for (element, count) in expected {
        assert_eq!(
            basis.n_basis_functions_for(element),
            Some(count),
            "{element}/STO-3G 应有 {count} 个基函数"
        );
    }
    // 五个元素加起来 21。
    assert_eq!(basis.n_basis_functions(), 21);
    // H 只有一个壳层；其余是 1s + 2s + 2p 三个壳层。
    assert_eq!(basis.shells_for("H").expect("H").shells.len(), 1);
    for element in ["C", "N", "O", "F"] {
        let shells = &basis.shells_for(element).expect(element).shells;
        assert_eq!(shells.len(), 3, "{element}: 1s + 2s + 2p");
        assert_eq!(
            shells.iter().map(|shell| shell.l).collect::<Vec<_>>(),
            vec![0, 0, 1],
            "{element}: 壳层顺序（按 l 升序，同 l 保持来源顺序）"
        );
    }
}

// ───────────────────────── ③ 球谐 ─────────────────────────

#[test]
fn sto3g_declares_spherical() {
    let basis = load();
    assert_eq!(basis.angular, Angular::Spherical, "STO-3G 必须声明球谐");
    assert_eq!(basis.role, qwb_registry::basis::BasisRole::Orbital);
    assert_eq!(basis.normalization, qwb_registry::basis::Normalization::L2);

    // STO-3G 最大 l=1；l<=1 时球谐与笛卡尔函数个数相同（1 个 s、3 个 p），
    // 所以 `angular` 的取值在物理上不产生歧义（但字段仍然必须显式声明，R-06）。
    let max_l = basis
        .shells
        .iter()
        .flat_map(|block| block.shells.iter())
        .map(|shell| shell.l)
        .max()
        .expect("at least one shell");
    assert_eq!(max_l, 1, "STO-3G 只到 p 函数");
}

// ─────────────── ④ 生产名 `sto-3g` 可被 Registry 取到 ───────────────

#[test]
fn sto3g_is_registered_under_its_production_name() {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data must load");
    let basis = registry
        .get_basis("sto-3g")
        .expect("生产名 sto-3g 必须可加载");
    assert_eq!(basis.elements.len(), 5);
    assert_eq!(basis.n_basis_functions(), 21);

    // 样例文件仍在磁盘上（D-02 的回归夹具按路径用它），但**不进**生产清单，
    // 否则 grid/functional 之类会出现两个同义定义。
    assert!(
        repo_root()
            .join("qwb-data/examples/basis/example-sto-3g-H.yaml")
            .is_file(),
        "样例文件应保留"
    );
    assert!(
        registry.get_basis("example-sto-3g-H").is_err(),
        "样例不应出现在生产注册表里"
    );
}
