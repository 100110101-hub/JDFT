//! D-07 验收（project.md §6.2）：`LANL2DZ` 价基 + ECP 的**正式数据**（Cu、Zn）。
//!
//! 纪律与 D-03/D-04 相同：数据由 `scripts/bse_to_qwb_data.py` 与
//! `scripts/bse_ecp_to_qwb_data.py` 从 **BSE 原始导出**机器转录，测试对导出逐项 diff
//! （相对差 < 1e-12），基函数数**从导出文件算出来**，不靠记忆。
//!
//! # 两条被更正过的 spec 判据（2026-10-26，用户批准）
//!
//! T5 原文写「Cu `n_core_electrons == 28`、`z_eff == 1.0`」—— 与真实 LANL2DZ 不符：
//! BSE 的 LANL2DZ 对 Cu 是 **10 芯电子**（`[Ne]` 芯，3s/3p 留在价层，Hay–Wadt 的
//! "K to Au including the outermost core orbitals" 一文），所以 `z_eff = 19`。
//! 28 芯会只剩 4s 一个价电子，与这套 s/p/d 价基自相矛盾。
//!
//! T5 原文的 ZnO 用 `def2-ECP` —— def2 系列对 3d 元素是**全电子**，BSE 的 def2-ecp
//! 从 Rb（Z=37）才开始，Zn 的 def2-ECP 不存在。经用户批准，ZnO 改用同族的
//! LANL2DZ（Zn：18 芯，`[Ar]` 芯，`z_eff = 12`）。
//!
//! 这两条更正写进 `docs/status.yaml` 的 D-07/T5 与本文件的文档注释里，不藏着。

use std::path::{Path, PathBuf};

use qwb_registry::Registry;
use qwb_registry::basis::{Angular, BasisSet};
use qwb_registry::ecp::{Ecp, RPowerConvention};
use qwb_registry::elements::atomic_number;

const BSE_EXPORT: &str = include_str!("fixtures/bse/lanl2dz.json");
const ELEMENTS: [&str; 4] = ["H", "O", "Cu", "Zn"];

type Triple = (u32, Vec<f64>, Vec<f64>);
/// 一个 ECP 通道：(l, [(power, exponent, coefficient)])。
type Channel = (u32, Vec<(i32, f64, f64)>);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn load_basis() -> BasisSet {
    let text = std::fs::read_to_string(repo_root().join("qwb-data/basis/lanl2dz.yaml"))
        .expect("qwb-data/basis/lanl2dz.yaml 必须存在");
    let basis: BasisSet = serde_yaml::from_str(&text).expect("合法 K1 基组");
    assert!(basis.check().is_empty(), "{:?}", basis.check());
    basis
}

fn load_ecp(name: &str) -> Ecp {
    let text = std::fs::read_to_string(repo_root().join(format!("qwb-data/ecp/{name}.yaml")))
        .unwrap_or_else(|err| panic!("qwb-data/ecp/{name}.yaml: {err}"));
    let ecp: Ecp = serde_yaml::from_str(&text).expect("合法 K1 ECP");
    assert!(ecp.check().is_empty(), "{:?}", ecp.check());
    ecp
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
fn lanl2dz_basis_matches_the_bse_export() {
    let basis = load_basis();
    let bse: serde_json::Value = serde_json::from_str(BSE_EXPORT).expect("JSON");
    assert_eq!(
        basis.elements,
        ELEMENTS.iter().map(ToString::to_string).collect::<Vec<_>>()
    );
    assert_eq!(basis.angular, Angular::Spherical);

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
    let basis = load_basis();
    let bse: serde_json::Value = serde_json::from_str(BSE_EXPORT).expect("JSON");
    for element in ELEMENTS {
        let z = atomic_number(element).expect("element");
        let expected: usize = bse_triples(&bse["elements"][z.to_string()]["electron_shells"])
            .iter()
            .map(|(l, _, _)| 2 * (*l as usize) + 1)
            .sum();
        assert_eq!(
            basis.n_basis_functions_for(element),
            Some(expected),
            "{element}/LANL2DZ 基函数数应与 BSE 导出一致"
        );
    }
    // 具体数字（与上面的导出计算独立）：
    // H = [2s] = 2；O = [3s2p] = 3 + 6 = 9；Cu = [3s3p2d] = 3 + 9 + 10 = 22；
    // Zn = [2s2p2d] = 2 + 6 + 10 = 18。
    assert_eq!(basis.n_basis_functions_for("H"), Some(2));
    assert_eq!(basis.n_basis_functions_for("O"), Some(9));
    assert_eq!(basis.n_basis_functions_for("Cu"), Some(22));
    assert_eq!(basis.n_basis_functions_for("Zn"), Some(18));
    // 两个 ECP 元素必须有 d 壳层（3d 价电子），否则 ECP 的 d 通道无从谈起。
    for element in ["Cu", "Zn"] {
        assert!(
            basis
                .shells_for(element)
                .expect(element)
                .shells
                .iter()
                .any(|shell| shell.l == 2),
            "{element} 应有 d 壳层"
        );
    }
    assert_eq!(basis.n_basis_functions(), 2 + 9 + 22 + 18);
}

/// BSE 的一个 `ecp_potentials[]` 块 → `(l, [(power, exponent, coefficient)])`。
fn bse_channels(entry: &serde_json::Value) -> Vec<Channel> {
    let mut out = Vec::new();
    for block in entry["ecp_potentials"].as_array().expect("ecp_potentials") {
        assert_eq!(block["ecp_type"], "scalar_ecp", "只支持标量相对论 ECP");
        let angular = block["angular_momentum"].as_array().expect("angular");
        assert_eq!(angular.len(), 1, "期望每个块一个 l");
        let l = u32::try_from(angular[0].as_u64().expect("l")).expect("fits");
        let powers: Vec<i32> = block["r_exponents"]
            .as_array()
            .expect("r_exponents")
            .iter()
            .map(|v| i32::try_from(v.as_i64().expect("n")).expect("fits"))
            .collect();
        let exponents: Vec<f64> = block["gaussian_exponents"]
            .as_array()
            .expect("gaussian_exponents")
            .iter()
            .map(|v| parse_bse_number(v.as_str().expect("string")))
            .collect();
        let coefficients: Vec<f64> = block["coefficients"][0]
            .as_array()
            .expect("coefficients")
            .iter()
            .map(|v| parse_bse_number(v.as_str().expect("string")))
            .collect();
        assert_eq!(powers.len(), exponents.len());
        assert_eq!(powers.len(), coefficients.len());
        out.push((
            l,
            powers
                .into_iter()
                .zip(exponents)
                .zip(coefficients)
                .map(|((power, exponent), coefficient)| (power, exponent, coefficient))
                .collect(),
        ));
    }
    out
}

fn qwb_terms(terms: &[qwb_registry::ecp::EcpTerm]) -> Vec<(i32, f64, f64)> {
    terms
        .iter()
        .map(|term| (term.power, term.exponent, term.coefficient))
        .collect()
}

#[test]
fn ecp_matches_the_bse_export() {
    let bse: serde_json::Value = serde_json::from_str(BSE_EXPORT).expect("JSON");
    for (element, name, expected_n_core) in [("Cu", "lanl2dz-cu", 10_u32), ("Zn", "lanl2dz-zn", 18)]
    {
        let z = atomic_number(element).expect("element");
        let entry = &bse["elements"][z.to_string()];
        let ecp = load_ecp(name);

        assert_eq!(ecp.element, element, "{name}: element");
        assert_eq!(
            ecp.n_core_electrons,
            u32::try_from(entry["ecp_electrons"].as_u64().expect("ecp_electrons")).expect("fits"),
            "{name}: n_core_electrons 应来自 BSE 的 ecp_electrons"
        );
        assert_eq!(ecp.n_core_electrons, expected_n_core, "{name}: n_core");
        assert_eq!(
            ecp.z_eff,
            f64::from(z - expected_n_core),
            "{name}: z_eff = Z − n_core"
        );
        ecp.z_eff_consistent().expect("z_eff 必须自洽");

        // §8.2 的陷阱必须在数据里显式声明；BSE 的 r_exponents 就是 `r^n`。
        assert_eq!(
            ecp.r_power_convention,
            RPowerConvention::Rn,
            "{name}: BSE 的 r_exponents 是 r^n 约定"
        );

        let channels = bse_channels(entry);
        let local_l = channels.iter().map(|(l, _)| *l).max().expect("有通道");
        let local_terms = channels
            .iter()
            .find(|(l, _)| *l == local_l)
            .map(|(_, terms)| terms.clone())
            .expect("局域通道");
        assert_eq!(
            qwb_terms(&ecp.local_part.terms),
            local_terms,
            "{name}: 局域（l={local_l}）项应与 BSE 逐项一致"
        );

        let mut nonlocal_expected: Vec<Channel> = channels
            .into_iter()
            .filter(|(l, _)| *l != local_l)
            .collect();
        nonlocal_expected.sort_by_key(|(l, _)| *l);
        let mut nonlocal_actual: Vec<Channel> = ecp
            .nonlocal_parts
            .iter()
            .map(|part| (part.l, qwb_terms(&part.terms)))
            .collect();
        nonlocal_actual.sort_by_key(|(l, _)| *l);
        assert_eq!(
            nonlocal_actual, nonlocal_expected,
            "{name}: 非局域项应与 BSE 逐项一致"
        );
    }
}

#[test]
fn z_eff_and_n_core_use_the_corrected_t5_numbers() {
    // T5 的原始 spec 写 Cu 28 / 1.0；真实 LANL2DZ 是 10 / 19（见文件头）。
    let cu = load_ecp("lanl2dz-cu");
    assert_eq!((cu.n_core_electrons, cu.z_eff), (10, 19.0));
    let zn = load_ecp("lanl2dz-zn");
    assert_eq!((zn.n_core_electrons, zn.z_eff), (18, 12.0));
}

#[test]
fn load_pair_is_the_only_pairing_entry_point() {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");

    let (basis, ecp) = registry
        .load_pair("lanl2dz", "lanl2dz-cu")
        .expect("Cu 配对");
    assert_eq!(ecp.element, "Cu");
    assert!(
        basis.elements.iter().any(|element| element == "Cu"),
        "基组必须覆盖 ECP 的元素"
    );
    let (_, ecp_zn) = registry
        .load_pair("lanl2dz", "lanl2dz-zn")
        .expect("Zn 配对");
    assert_eq!(ecp_zn.element, "Zn");

    // 配对必须**按元素**校验：def2-SVP 的正式数据里没有 Cu，拿它配 lanl2dz-cu 必须报错。
    let error = registry
        .load_pair("def2-svp", "lanl2dz-cu")
        .expect_err("def2-SVP 没有 Cu，配 lanl2dz-cu 应失败");
    let message = format!("{error}");
    assert!(
        message.contains("Cu") && message.contains("def2-svp"),
        "错误信息要指名元素与基组：{message}"
    );

    // 反过来：不存在的 ECP 名字必须是数据错误。
    assert!(matches!(
        registry.get_ecp("no-such-ecp"),
        Err(qwb_engine_api::EngineError::Data(_))
    ));
}
