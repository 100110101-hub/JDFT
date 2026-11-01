//! SV-03 的**元素级 SMD 数据**验收：能加载、形状与记录一致、关键数值对得上。
//!
//! 数据本身的可信度来自 `scripts/validate_smd_cds.py`（对 PySCF 2.14.0 的 `libsolvent`
//! 验到水相 σ 1.6e-5 cal/(mol·Å²)、五个非水溶剂 E_CDS 3.8e-7 kcal/mol）。这条测试守的是
//! **落盘与加载**这一环：`elements` 与各表同序同长、面积偏移是 0.4、σ 的已知非零项没丢、
//! 非水表被截到 36 项（上游是 150 项，超过 Kr 的不在本文件范围）。

use std::path::PathBuf;

use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

#[test]
fn smd_atomic_data_loads_with_the_documented_shape() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry
        .get_smd_atomic("smd-atomic")
        .expect("the registry must index the element-level SMD data");

    assert_eq!(data.name, "smd-atomic");
    assert_eq!(data.kind, "smd-atomic");
    assert_eq!(data.elements.len(), 36, "H-Kr");
    assert_eq!(data.elements.first().map(String::as_str), Some("H"));
    assert_eq!(data.elements.last().map(String::as_str), Some("Kr"));

    // 面积约定：0.4 Å 的偏移是**反推并验证**出来的（不是从论文抄的），所以它是一条断言。
    assert!(
        (data.cds.area_offset_angstrom - 0.4).abs() < 1.0e-12,
        "the CDS area offset must be 0.4 A, got {}",
        data.cds.area_offset_angstrom
    );
    assert!(data.cds.area_convention.contains("4*pi"));

    // 水相 σ：已知的非零项逐个钉住（上游 N/O 为 0，是原值不是缺项）。
    let sigma = &data.cds.water.sigma_cal_per_mol_angstrom2;
    assert_eq!(sigma.len(), 36);
    for (symbol, expected) in [
        ("H", 48.69),
        ("C", 129.74),
        ("F", 38.18),
        ("S", -9.10),
        ("Cl", 9.82),
        ("Br", -8.72),
        ("N", 0.0),
        ("O", 0.0),
    ] {
        let value = sigma
            .get(symbol)
            .unwrap_or_else(|| panic!("missing sigma for {symbol}"));
        assert!(
            (value - expected).abs() < 1.0e-9,
            "{symbol}: sigma = {value}, expected {expected}"
        );
    }

    // 非水：三张表各 36 项 + 四个常数项 + 六个描述符名。
    let nonaqueous = &data.cds.nonaqueous;
    for (label, values) in [
        ("sigma_n", &nonaqueous.sigma_n),
        ("sigma_a", &nonaqueous.sigma_a),
        ("sigma_b", &nonaqueous.sigma_b),
        ("hsigma_n", &nonaqueous.hsigma_n),
        ("hsigma_a", &nonaqueous.hsigma_a),
        ("hsigma_b", &nonaqueous.hsigma_b),
    ] {
        assert_eq!(values.len(), 36, "{label}");
    }
    assert_eq!(nonaqueous.sigma_mol.len(), 4);
    assert_eq!(nonaqueous.sigma_mol[0], 0.35);
    assert_eq!(nonaqueous.sigma_mol[2], -4.19);
    assert_eq!(nonaqueous.sigma_mol[3], -6.68);
    assert_eq!(
        nonaqueous.descriptor_names,
        ["SOLA", "SOLB", "SOLC", "SOLG", "SOLH", "SOLN"]
    );

    // CDS 的 vdW 半径与 SMD 的参数化半径**不是**一套：C 在 CDS 里是 1.70，参数化半径是 1.85。
    let radii = &data.cds.vdw_radii_angstrom;
    assert_eq!(radii.len(), 36);
    assert_eq!(radii["H"], 1.20);
    assert_eq!(radii["C"], 1.70);
    assert_eq!(radii["N"], 1.55);
    assert_eq!(radii["O"], 1.52);

    assert!(data.provenance.upstream[0].file.contains("mnsol"));
    let issues = data.check();
    assert!(issues.is_empty(), "{issues:?}");
}

#[test]
fn the_bond_contact_rules_match_upstream() {
    // `SMXCDS` 的成键权重 `COT = exp(Δ/(d − (R + Δ)))` 需要 15×15 的 `RKKVAL` 与
    // `NATCNV` 的元素类型。这里钉住几处手抄过的值：H–C 1.55、C–C 1.84、O–C 1.84、
    // O–O 2.75（`ISTS = 6` 时被特例覆盖成 1.80，特例在实现侧）、O–S 1.71（特例 1.330）。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    let bonds = &data.cds.bonds;
    assert_eq!(bonds.table, "RKKVAL");
    assert_eq!(bonds.rkkval_angstrom.len(), 15);
    assert!(bonds.rkkval_angstrom.iter().all(|row| row.len() == 15));
    // 类型编号（上游 `NATCNV`）：H 1、C 2、N 3、O 4、F 5、S 6、Cl 7、Br 8、P 9、Si 11。
    let expected_types = [
        ("H", 1usize),
        ("C", 2),
        ("N", 3),
        ("O", 4),
        ("F", 5),
        ("S", 6),
        ("Cl", 7),
        ("Br", 8),
        ("P", 9),
        ("Si", 11),
    ];
    for (symbol, kind) in expected_types {
        assert_eq!(bonds.element_types[symbol], kind, "{symbol}");
    }
    let r = |a: &str, b: &str| {
        bonds.rkkval_angstrom[bonds.element_types[a] - 1][bonds.element_types[b] - 1]
    };
    assert_eq!(r("H", "C"), 1.55);
    assert_eq!(r("C", "C"), 1.84);
    assert_eq!(r("N", "C"), 1.84);
    assert_eq!(r("O", "C"), 1.84);
    assert_eq!(r("O", "N"), 1.50);
    assert_eq!(r("O", "O"), 2.75);
    assert_eq!(r("O", "S"), 1.71);
    // 本文件覆盖的 36 个元素里有 25 个类型是 0（不参与成键权重），例如 He/Ne/Ar。
    assert_eq!(bonds.element_types.len(), 36);
    let inactive = bonds
        .element_types
        .values()
        .filter(|kind| **kind == 0)
        .count();
    assert_eq!(inactive, 26, "26 of the 36 elements carry type 0");
    assert_eq!(bonds.element_types["Ne"], 0);
}

#[test]
fn the_smd_atomic_kind_is_separate_from_the_solvent_kind() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    // 溶剂文件仍然走 `Solvent`（`radii_set: bondi` / `cavity_model: pcm`）。
    let water = registry.get_solvent("water").expect("water");
    assert!(water.epsilon > 70.0);
    assert!(registry.get_solvent("smd-atomic").is_err());
    assert!(registry.get_smd_atomic("water").is_err());
}

#[test]
fn the_pair_sigma_blocks_match_upstream() {
    // 成对修正 `SIGMA(101..150)`：水相六个非零项、非水三张表各几项。这些数是从
    // `mnsol.F` 的 `SIGMA_DATA` / `SIGMA_N/A/B` 逐字提的，`smd_sigma` 的成对分支靠它们。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    let water = &data.cds.water.pair_sigma_cal_per_mol_angstrom2;
    assert_eq!(water.len(), 50);
    // pair 编号 p ↔ 下标 p − 1 ↔ 上游 `SIGMA(100 + p)`。
    for (pair, value) in [
        (1usize, -72.95),
        (3, 68.69),
        (5, -48.22),
        (6, 121.98),
        (14, 68.85),
        (16, 84.10),
    ] {
        assert_eq!(water[pair - 1], value, "aqueous SIGMA({})", 100 + pair);
    }
    let table = &data.cds.nonaqueous;
    for values in [
        &table.pair_sigma_n,
        &table.pair_sigma_a,
        &table.pair_sigma_b,
    ] {
        assert_eq!(values.len(), 50);
    }
    assert_eq!(table.pair_sigma_n[0], -62.05, "SIGMA_N(101)");
    assert_eq!(table.pair_sigma_n[2], -15.70, "SIGMA_N(103)");
    assert_eq!(table.pair_sigma_n[9], -99.76, "SIGMA_N(110)");
    assert_eq!(table.pair_sigma_a[2], 95.99, "SIGMA_A(103)");
    assert_eq!(table.pair_sigma_a[4], -41.00, "SIGMA_A(105)");
    assert_eq!(table.pair_sigma_a[9], 152.20, "SIGMA_A(110)");
    assert_eq!(table.pair_sigma_b[3], -128.16, "SIGMA_B(104)");
    assert_eq!(table.pair_sigma_b[5], 79.13, "SIGMA_B(106)");
    let issues = data.check();
    assert!(issues.is_empty(), "{issues:?}");
}

#[test]
fn the_parameterised_radii_are_the_second_radius_family() {
    // SMD 的**参数化半径**（eq. 16）与 CDS 的 vdW 半径是两套：C 1.85 vs 1.70、N 1.89 vs 1.55。
    // 数据里它们并列存在、各自有名字，这条测试钉住几个关键值并断言"不许相等"。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    let radii = &data.parameterised_radii;
    assert_eq!(radii.base_angstrom.len(), 36);
    assert_eq!(radii.base_angstrom["H"], 1.20);
    assert_eq!(radii.base_angstrom["He"], 1.40);
    assert_eq!(radii.overrides_angstrom["H"], 1.20);
    assert_eq!(radii.overrides_angstrom["C"], 1.85);
    assert_eq!(radii.overrides_angstrom["N"], 1.89);
    assert_eq!(radii.overrides_angstrom["F"], 1.73);
    assert_eq!(radii.overrides_angstrom["S"], 2.49);
    assert_eq!(radii.overrides_angstrom["Cl"], 2.38);
    assert_eq!(radii.overrides_angstrom["Br"], 2.60);
    // O 由 α 决定，不在覆盖表里（那是常数表解释不了的部分）。
    assert!(!radii.overrides_angstrom.contains_key("O"));
    assert_eq!(radii.oxygen.threshold_alpha, 0.43);
    assert_eq!(radii.oxygen.radius_high_angstrom, 1.52);
    assert_eq!(radii.oxygen.slope_angstrom, 1.8);
    // 两套半径在 C/N 上必须不同。
    assert!((radii.overrides_angstrom["C"] - data.cds.vdw_radii_angstrom["C"]).abs() > 0.1);
    assert!((radii.overrides_angstrom["N"] - data.cds.vdw_radii_angstrom["N"]).abs() > 0.3);
    let issues = data.check();
    assert!(issues.is_empty(), "{issues:?}");
}
