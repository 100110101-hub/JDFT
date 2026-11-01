//! DA-06 的 D4 **中间量层**验收：配位数、EEQ 电荷、电荷相关的 C6。
//!
//! 对照物是 `tests/reference/d4_properties.csv` —— 由官方 dftd4 4.3.1 的
//! `get_properties()` 生成（`scripts/dftd4_dispersion_reference.py`，逐行带来源），
//! 覆盖 9 个二聚体（`d3_dimers.csv` 的几何，Å 坐标）。
//!
//! 为什么要分三层验收：D3 那轮只有一个总能量，错了只能二分；D4 的 C6 来自
//! `alpha(iω)` 的积分、权重又带电荷缩放 —— 一旦能量对不上，"CN 错"与"C6 错"从总能量上
//! 看不出区别。这里把 CN / 电荷 / C6 分别钉住，能量那层（下一段）就只剩下两体与 ATM 的
//! 装配问题。
//!
//! **长度常数**：几何用 K2 的 `BOHR_PER_ANGSTROM`（与参考值生成脚本同一套），而 D4 内部把
//! 半径从 Å 换到 Bohr 用的是上游 mctc-lib 的 `aatoau`。两者相对差 1.9e-8，所以这里不可能
//! 逐位相同 —— 容差按"上游常数差异 + 参考值本身 7 位小数"来定，实测值打印在测试里。

use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_scf::d4;

/// K2 的 Å→Bohr（与 `scripts/dftd4_dispersion_reference.py` 一致）。
const BOHR_PER_ANGSTROM: f64 = 1.889_726_089_838_572;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn read_csv(path: &std::path::Path) -> (Vec<String>, Vec<BTreeMap<String, String>>) {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .expect("header")
        .split(',')
        .map(str::trim)
        .map(String::from)
        .collect();
    let rows = lines
        .map(|line| {
            line.split(',')
                .map(str::trim)
                .zip(header.iter())
                .map(|(value, key)| (key.clone(), value.to_string()))
                .collect()
        })
        .collect();
    (header, rows)
}

fn field<'a>(row: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    row.get(key)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("missing column `{key}`"))
}

#[test]
fn d4_intermediate_quantities_match_the_official_reference() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let d4_data = registry.get_d4_dispersion("d4").expect("d4 data");

    // 几何：`d3_dimers.csv`（与 D3 参考同一个几何源）。
    let (_, geometry_rows) = read_csv(&reference_dir().join("d3_dimers.csv"));
    let mut molecules: BTreeMap<String, Vec<(String, [f64; 3])>> = BTreeMap::new();
    for row in &geometry_rows {
        let position = [
            field(row, "x_angstrom").parse::<f64>().expect("x") * BOHR_PER_ANGSTROM,
            field(row, "y_angstrom").parse::<f64>().expect("y") * BOHR_PER_ANGSTROM,
            field(row, "z_angstrom").parse::<f64>().expect("z") * BOHR_PER_ANGSTROM,
        ];
        molecules
            .entry(field(row, "dimer").to_string())
            .or_default()
            .push((field(row, "element").to_string(), position));
    }

    // 参考值：官方 `get_properties()`。
    let (_, property_rows) = read_csv(&reference_dir().join("d4_properties.csv"));
    let mut expected: BTreeMap<String, Vec<BTreeMap<String, String>>> = BTreeMap::new();
    for row in &property_rows {
        expected
            .entry(field(row, "dimer").to_string())
            .or_default()
            .push(row.clone());
    }
    assert_eq!(molecules.len(), expected.len());
    assert!(
        molecules.len() >= 9,
        "expected at least the nine D3 dimers, got {}",
        molecules.len()
    );

    let mut worst_cn = 0.0_f64;
    let mut worst_charge = 0.0_f64;
    let mut worst_c6 = 0.0_f64;
    let mut worst_charge_sum = 0.0_f64;
    for (name, atoms) in &molecules {
        let symbols: Vec<&str> = atoms.iter().map(|(symbol, _)| symbol.as_str()).collect();
        let positions: Vec<[f64; 3]> = atoms.iter().map(|(_, position)| *position).collect();
        let properties = d4::properties(d4_data, &symbols, &positions, 0.0)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let rows = expected
            .get(name)
            .unwrap_or_else(|| panic!("{name}: no reference rows"));
        assert_eq!(rows.len(), atoms.len(), "{name}: atom count");
        // 参考 CSV 的 `atom_index` 是 1-based，行序与 `d3_dimers.csv` 的原子序一致。
        for (index, row) in rows.iter().enumerate() {
            let cn_reference = field(row, "coordination_number")
                .parse::<f64>()
                .expect("cn");
            let charge_reference = field(row, "eeq_charge").parse::<f64>().expect("charge");
            let c6_reference = field(row, "c6_to_first_atom").parse::<f64>().expect("c6");
            worst_cn = worst_cn.max((properties.coordination_numbers[index] - cn_reference).abs());
            worst_charge = worst_charge.max((properties.charges[index] - charge_reference).abs());
            worst_c6 = worst_c6.max((properties.c6[index][0] - c6_reference).abs());
        }
        let total: f64 = properties.charges.iter().sum();
        worst_charge_sum = worst_charge_sum.max((total - 0.0).abs());
        println!(
            "{name:24} atoms = {:2}  charge sum = {:+.2e}  CN[0] = {:.8} (ref {:.8})",
            atoms.len(),
            total,
            properties.coordination_numbers[0],
            rows[0]
                .get("coordination_number")
                .map_or(f64::NAN, |_| field(&rows[0], "coordination_number")
                    .parse()
                    .unwrap_or(f64::NAN)),
        );
    }
    println!(
        "worst |ΔCN| = {worst_cn:.3e}, |Δq| = {worst_charge:.3e}, |ΔC6| = {worst_c6:.3e}, \
         |Σq| = {worst_charge_sum:.3e}"
    );
    // 判据：CN/电荷 1e-6（上游 `alpha` 只有 7 位小数、长度常数差 1.9e-8）；C6 1e-5（Ha·Bohr⁶）。
    assert!(worst_cn < 1.0e-6, "CN differs by {worst_cn:.3e}");
    assert!(
        worst_charge < 1.0e-6,
        "EEQ charge differs by {worst_charge:.3e}"
    );
    assert!(worst_c6 < 1.0e-5, "C6 differs by {worst_c6:.3e} Ha*Bohr^6");
    assert!(
        worst_charge_sum < 1.0e-9,
        "neutral molecules must sum to zero charge, got {worst_charge_sum:.3e}"
    );
}

#[test]
fn d4_charges_respond_to_the_total_charge() {
    // 电荷依赖是 D4 与 D3 最大的不同：同一几何换总电荷，EEQ 电荷与 C6 都要变。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let d4_data = registry.get_d4_dispersion("d4").expect("d4 data");
    let molecules = {
        let (_, rows) = read_csv(&reference_dir().join("d3_dimers.csv"));
        let mut molecules: BTreeMap<String, Vec<(String, [f64; 3])>> = BTreeMap::new();
        for row in &rows {
            let position = [
                field(row, "x_angstrom").parse::<f64>().expect("x") * BOHR_PER_ANGSTROM,
                field(row, "y_angstrom").parse::<f64>().expect("y") * BOHR_PER_ANGSTROM,
                field(row, "z_angstrom").parse::<f64>().expect("z") * BOHR_PER_ANGSTROM,
            ];
            molecules
                .entry(field(row, "dimer").to_string())
                .or_default()
                .push((field(row, "element").to_string(), position));
        }
        molecules
    };
    let atoms = molecules.get("water_dimer").expect("water_dimer");
    let symbols: Vec<&str> = atoms.iter().map(|(symbol, _)| symbol.as_str()).collect();
    let positions: Vec<[f64; 3]> = atoms.iter().map(|(_, position)| *position).collect();
    let neutral = d4::properties(d4_data, &symbols, &positions, 0.0).expect("neutral");
    let cation = d4::properties(d4_data, &symbols, &positions, 1.0).expect("cation");
    let anion = d4::properties(d4_data, &symbols, &positions, -1.0).expect("anion");
    for (label, value, target) in [
        ("neutral", &neutral, 0.0),
        ("cation", &cation, 1.0),
        ("anion", &anion, -1.0),
    ] {
        let total: f64 = value.charges.iter().sum();
        println!(
            "{label:8} Σq = {total:+.10}  q[0] = {:+.6}",
            value.charges[0]
        );
        assert!((total - target).abs() < 1.0e-9, "{label}: Σq = {total}");
    }
    assert!(
        (neutral.charges[0] - cation.charges[0]).abs() > 0.05,
        "EEQ charges must respond to the total charge"
    );
    assert!(
        (neutral.c6[0][1] - cation.c6[0][1]).abs() > 1.0e-3,
        "C6 must respond to the total charge (through the reference weights)"
    );
}
