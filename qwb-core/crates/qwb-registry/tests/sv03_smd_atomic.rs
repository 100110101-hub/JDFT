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
fn the_smd_atomic_kind_is_separate_from_the_solvent_kind() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    // 溶剂文件仍然走 `Solvent`（`radii_set: bondi` / `cavity_model: pcm`）。
    let water = registry.get_solvent("water").expect("water");
    assert!(water.epsilon > 70.0);
    assert!(registry.get_solvent("smd-atomic").is_err());
    assert!(registry.get_smd_atomic("water").is_err());
}
