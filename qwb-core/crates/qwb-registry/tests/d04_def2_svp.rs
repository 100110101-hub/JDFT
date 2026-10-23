//! D-04 验收（project.md §6.2）：`def2-SVP`（H,C,N,O）正式数据。
//!
//! 与 D-03 同一套纪律：数据由 `scripts/bse_to_qwb_data.py` 从 **BSE 原始导出**
//! 机器转录，逐项 diff < 1e-12；每元素基函数数**从导出文件算出来**，不靠记忆。
//!
//! def2-SVP：H = [2s1p] → 5；C/N/O = [3s2p1d] → 3+6+5 = 14（球谐 d = 5）。

use std::path::{Path, PathBuf};

use qwb_registry::Registry;
use qwb_registry::basis::{Angular, BasisSet};
use qwb_registry::elements::atomic_number;

const BSE_EXPORT: &str = include_str!("fixtures/bse/def2-svp.json");
const ELEMENTS: [&str; 4] = ["H", "C", "N", "O"];

type Triple = (u32, Vec<f64>, Vec<f64>);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn load() -> BasisSet {
    let text = std::fs::read_to_string(repo_root().join("qwb-data/basis/def2-svp.yaml"))
        .expect("qwb-data/basis/def2-svp.yaml 必须存在");
    let basis: BasisSet = serde_yaml::from_str(&text).expect("合法 K1 基组");
    assert!(basis.check().is_empty(), "{:?}", basis.check());
    basis
}

fn parse_bse_number(text: &str) -> f64 {
    text.parse::<f64>()
        .unwrap_or_else(|err| panic!("BSE 数字 {text:?}: {err}"))
}

fn bse_triples(shells: &serde_json::Value) -> Vec<Triple> {
    let mut out: Vec<Triple> = Vec::new();
    for shell in shells.as_array().expect("array") {
        let angular: Vec<u32> = shell["angular_momentum"]
            .as_array()
            .expect("angular_momentum")
            .iter()
            .map(|v| u32::try_from(v.as_u64().expect("l")).expect("fits"))
            .collect();
        let exponents: Vec<f64> = shell["exponents"]
            .as_array()
            .expect("exponents")
            .iter()
            .map(|v| parse_bse_number(v.as_str().expect("string")))
            .collect();
        let columns: Vec<Vec<f64>> = shell["coefficients"]
            .as_array()
            .expect("coefficients")
            .iter()
            .map(|column| {
                column
                    .as_array()
                    .expect("column")
                    .iter()
                    .map(|v| parse_bse_number(v.as_str().expect("string")))
                    .collect()
            })
            .collect();

        if columns.len() == angular.len() {
            for (l, column) in angular.iter().zip(columns) {
                out.push((*l, exponents.clone(), column));
            }
        } else {
            assert_eq!(angular.len(), 1, "通用收缩只能有一个 l：{angular:?}");
            for column in columns {
                out.push((angular[0], exponents.clone(), column));
            }
        }
    }
    out
}

fn qwb_triples(basis: &BasisSet, element: &str) -> Vec<Triple> {
    let block = basis.shells_for(element).expect(element);
    let mut out: Vec<Triple> = Vec::new();
    for shell in &block.shells {
        for column in &shell.coefficients {
            out.push((shell.l, shell.exponents.clone(), column.clone()));
        }
    }
    out
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
            "{what}[{index}]: {a} vs BSE {e}（相对差 {relative:e}）"
        );
    }
}

#[test]
fn def2_svp_matches_the_bse_export() {
    let basis = load();
    let bse: serde_json::Value = serde_json::from_str(BSE_EXPORT).expect("JSON");
    assert_eq!(
        basis.elements,
        ELEMENTS.iter().map(ToString::to_string).collect::<Vec<_>>()
    );

    for element in ELEMENTS {
        let z = atomic_number(element).expect("H..Kr");
        let shells = &bse["elements"][z.to_string()]["electron_shells"];

        let mut expected = bse_triples(shells);
        let mut actual = qwb_triples(&basis, element);
        sort_canonical(&mut expected);
        sort_canonical(&mut actual);
        assert_eq!(actual.len(), expected.len(), "{element}: 收缩函数数不同");

        for (index, ((la, ea, ca), (le, ee, ce))) in actual.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(la, le, "{element}[{index}]: 角动量不同");
            assert_rel_eq(ea, ee, &format!("{element}[{index}] l={la} exponents"));
            assert_rel_eq(ca, ce, &format!("{element}[{index}] l={la} coefficients"));
        }
    }
}

#[test]
fn per_element_basis_function_counts_come_from_the_export() {
    let basis = load();
    let bse: serde_json::Value = serde_json::from_str(BSE_EXPORT).expect("JSON");
    assert_eq!(basis.angular, Angular::Spherical);

    for element in ELEMENTS {
        let z = atomic_number(element).expect("element");
        let shells = &bse["elements"][z.to_string()]["electron_shells"];

        // 从导出文件本身算基函数数：每个收缩函数贡献 (2l+1) 个球谐函数。
        let expected: usize = bse_triples(shells)
            .iter()
            .map(|(l, _, _)| 2 * (*l as usize) + 1)
            .sum();
        assert_eq!(
            basis.n_basis_functions_for(element),
            Some(expected),
            "{element}/def2-SVP 基函数数应与 BSE 导出一致"
        );
    }

    // 具体数字（与上面的导出计算独立）：H = [2s1p] = 5；C/N/O = [3s2p1d] = 14。
    assert_eq!(basis.n_basis_functions_for("H"), Some(5));
    for element in ["C", "N", "O"] {
        assert_eq!(basis.n_basis_functions_for(element), Some(14), "{element}");
    }
    // C/N/O 必须真的有 d 壳层（这是 def2-SVP 与 STO-3G 的关键差别）。
    for element in ["C", "N", "O"] {
        let has_d = basis
            .shells_for(element)
            .expect(element)
            .shells
            .iter()
            .any(|shell| shell.l == 2);
        assert!(has_d, "{element} 应有 d 壳层");
    }
    assert_eq!(basis.n_basis_functions(), 5 + 14 * 3);
}

#[test]
fn def2_svp_is_registered_under_its_production_name() {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");
    let basis = registry.get_basis("def2-svp").expect("生产名 def2-svp");
    assert_eq!(basis.n_basis_functions_for("O"), Some(14));
    // STO-3G 仍在，两者不冲突。
    assert!(registry.get_basis("sto-3g").is_ok());
}

#[test]
fn internal_basis_handles_d_shells() {
    use qwb_basis::InternalBasis;
    use qwb_engine_api::{Atom, Molecule};

    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");
    let basis_k1 = registry.get_basis("def2-svp").expect("def2-svp");
    // 单个碳：数据层（球谐）14 个 AO = 3s + 2p + 1d(5)。
    assert_eq!(basis_k1.n_basis_functions_for("C"), Some(14));

    let carbon = Molecule::new(vec![Atom::new("C", [0.0, 0.0, 0.0])], 0, 3);
    let internal = InternalBasis::build(basis_k1, &carbon).expect("build");

    // `InternalBasis` 现在按数据的 `Angular` 报 AO 数（球谐 d = 5），所以 C 是
    // 3+6+5 = 14，与数据层的 `n_basis_functions_for` 一致（R-06 的计数缺口已修）。
    assert_eq!(
        internal.n_functions(),
        14,
        "球谐下 C 是 3s + 2p(6) + 1d(5) = 14"
    );
    assert!(
        internal
            .shells()
            .iter()
            .all(|shell| shell.n_functions() <= 5)
    );

    // 但**积分核**仍是直角坐标实现（d = 6），所以 l>=2 必须显式拒绝，
    // 不能给出索引对不上的结果。等 cartesian->spherical 变换接上后改成正常计算。
    // （积分层在 qwb-integrals 的测试里覆盖，这里只验证基组层。）
    // 壳层顺序由数据文件决定；这里断言**组成**（3 个 s、2 个 p、1 个 d），
    // 而不是硬编码顺序。
    let mut counts = std::collections::BTreeMap::new();
    for shell in internal.shells() {
        *counts.entry(shell.l).or_insert(0usize) += 1;
    }
    assert_eq!(counts.get(&0), Some(&3), "3 个 s 壳层");
    assert_eq!(counts.get(&1), Some(&2), "2 个 p 壳层");
    assert_eq!(counts.get(&2), Some(&1), "1 个 d 壳层");
}
