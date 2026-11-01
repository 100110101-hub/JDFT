//! DA-06 的 D4 **能量梯度**验收：`dispersion_gradient`（两体 + ATM）。
//!
//! # 三条判据
//!
//! * **(a) 紧判据**：`dispersion_energy` 的中心差分（Richardson 外推，
//!   `(4·D(h) − D(2h))/3 = f + O(h⁴)`，`h = 1e-3` Bohr）对 `water_dimer`、
//!   `methane_dimer`、`benzene_dimer_stacked` 三个几何逐个分量比对，
//!   判据 `max |analytic − richardson| < 1e-8` Ha/Bohr。
//! * **(b) 上游约定**：`d4_dispersion.csv` 的 60 行里，`gradient_max_abs_hartree_per_bohr`
//!   是官方 dftd4 4.3.1 对**同一几何**算出的最大绝对梯度分量。这里比对
//!   `max |gradient component|`，要求相对一致优于 `1e-6`。这一列只能验证"约定"：
//!   最大分量丢掉了方向和其余分量，所以 (a) 才是紧检查，(b) 只确认上游的
//!   长度/符号/电荷约定（Bohr、`q` 的符号、ATM 的 C6 用 q=0）没有搞反。
//! * **(c) 平移不变性**：`Σ_atoms gradient ≈ 0`（`< 1e-10` Ha/Bohr）。
//!
//! 另外核对 `dispersion_gradient` 的三个能量分量与 `dispersion_energy` **逐位相同**。
#![deny(unsafe_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_scf::d4;

/// K2 的 Å→Bohr（与 `da06_d4_properties.rs`、参考脚本一致）。
const BOHR_PER_ANGSTROM: f64 = 1.889_726_089_838_572;

/// 中心差分步长（Bohr）。
const STEP: f64 = 1.0e-3;

type Row = BTreeMap<String, String>;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn read_csv(path: &std::path::Path) -> Vec<Row> {
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
    lines
        .map(|line| {
            line.split(',')
                .map(str::trim)
                .zip(header.iter())
                .map(|(value, key)| (key.clone(), value.to_string()))
                .collect()
        })
        .collect()
}

fn field<'a>(row: &'a Row, key: &str) -> &'a str {
    row.get(key)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("missing column `{key}`"))
}

/// 上游（dftd4）的泛函键名 → 本项目数据文件的键名。
fn canonical_method(name: &str) -> &str {
    match name {
        "m06l" => "m06-l",
        other => other,
    }
}

fn geometries() -> BTreeMap<String, (Vec<String>, Vec<[f64; 3]>)> {
    let rows = read_csv(&reference_dir().join("d3_dimers.csv"));
    let mut molecules: BTreeMap<String, (Vec<String>, Vec<[f64; 3]>)> = BTreeMap::new();
    for row in &rows {
        let entry = molecules
            .entry(field(row, "dimer").to_string())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(field(row, "element").to_string());
        entry.1.push([
            field(row, "x_angstrom").parse::<f64>().expect("x") * BOHR_PER_ANGSTROM,
            field(row, "y_angstrom").parse::<f64>().expect("y") * BOHR_PER_ANGSTROM,
            field(row, "z_angstrom").parse::<f64>().expect("z") * BOHR_PER_ANGSTROM,
        ]);
    }
    molecules
}

fn displaced(positions: &[[f64; 3]], atom: usize, axis: usize, delta: f64) -> Vec<[f64; 3]> {
    let mut shifted = positions.to_vec();
    shifted[atom][axis] += delta;
    shifted
}

/// Richardson 外推的中心差分，`h` 与 `2h` 两档：`(4·D(h) − D(2h))/3 = f + O(h⁴)`。
fn richardson<F>(positions: &[[f64; 3]], atom: usize, axis: usize, evaluate: &F) -> f64
where
    F: Fn(&[[f64; 3]]) -> f64,
{
    let difference = |step: f64| {
        let forward = evaluate(&displaced(positions, atom, axis, step));
        let backward = evaluate(&displaced(positions, atom, axis, -step));
        (forward - backward) / (2.0 * step)
    };
    (4.0 * difference(STEP) - difference(2.0 * STEP)) / 3.0
}

/// 三个几何上逐个分量比对解析梯度与 `dispersion_energy` 的中心差分。
#[test]
fn d4_gradient_matches_central_differences_of_the_energy() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_d4_dispersion("d4").expect("d4 data");
    let molecules = geometries();

    let mut worst = 0.0_f64;
    let mut worst_row = String::new();
    let mut worst_translation = 0.0_f64;
    for (name, functional) in [
        ("water_dimer", "b3lyp"),
        ("methane_dimer", "b3lyp"),
        ("benzene_dimer_stacked", "b3lyp"),
    ] {
        let (symbols, positions) = molecules
            .get(name)
            .unwrap_or_else(|| panic!("{name}: geometry missing"));
        let refs: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let n = symbols.len();

        let analytic = d4::dispersion_gradient(data, functional, &refs, positions, 0.0)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(analytic.gradient.len(), n, "{name}: gradient length");

        // 能量分量必须与 `dispersion_energy` 逐位相同（同一段代码）。
        let energy = d4::dispersion_energy(data, functional, &refs, positions, 0.0)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            analytic.energy_hartree, energy.energy_hartree,
            "{name}: total energy must be bit-identical"
        );
        assert_eq!(
            analytic.two_body_hartree, energy.two_body_hartree,
            "{name}: two-body energy must be bit-identical"
        );
        assert_eq!(
            analytic.atm_hartree, energy.atm_hartree,
            "{name}: ATM energy must be bit-identical"
        );

        let mut molecule_worst = 0.0_f64;
        for atom in 0..n {
            for axis in 0..3 {
                let numeric = richardson(positions, atom, axis, &|shifted| {
                    d4::dispersion_energy(data, functional, &refs, shifted, 0.0)
                        .expect("dispersion energy")
                        .energy_hartree
                });
                let delta = (analytic.gradient[atom][axis] - numeric).abs();
                if delta > molecule_worst {
                    molecule_worst = delta;
                    worst_row = format!("{name}/{functional} atom {atom} axis {axis}");
                }
            }
        }
        let mut translation = [0.0_f64; 3];
        for entry in &analytic.gradient {
            for (axis, value) in translation.iter_mut().enumerate() {
                *value += entry[axis];
            }
        }
        let translation = translation
            .iter()
            .fold(0.0_f64, |worst, value| worst.max(value.abs()));
        println!(
            "{name:24} atoms = {n:2}  E = {:+.12} Ha  max |Δg| = {molecule_worst:.3e} Ha/Bohr  \
             |Σg| = {translation:.3e}",
            analytic.energy_hartree
        );
        worst = worst.max(molecule_worst);
        worst_translation = worst_translation.max(translation);
    }

    println!("(a) worst |analytic − richardson| = {worst:.3e} Ha/Bohr ({worst_row})");
    println!("(c) worst |Σ_atoms gradient| = {worst_translation:.3e} Ha/Bohr");
    assert!(
        worst < 1.0e-8,
        "the D4 gradient differs from finite differences by {worst:.3e} Ha/Bohr at {worst_row}"
    );
    assert!(
        worst_translation < 1.0e-10,
        "Σ_atoms gradient must vanish, got {worst_translation:.3e} Ha/Bohr"
    );
}

/// (b) 上游 oracle 列：`gradient_max_abs_hartree_per_bohr`。
#[test]
fn d4_gradient_max_component_matches_the_official_oracle() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_d4_dispersion("d4").expect("d4 data");
    let molecules = geometries();
    let rows = read_csv(&reference_dir().join("d4_dispersion.csv"));
    assert_eq!(rows.len(), 60, "expected the 60 oracle rows");

    let mut worst = 0.0_f64;
    let mut worst_row = String::new();
    for row in &rows {
        let dimer = field(row, "dimer");
        let method = canonical_method(field(row, "method"));
        let charge = field(row, "charge").parse::<f64>().expect("charge");
        let expected = field(row, "gradient_max_abs_hartree_per_bohr")
            .parse::<f64>()
            .expect("gradient");
        let (symbols, positions) = molecules
            .get(dimer)
            .unwrap_or_else(|| panic!("{dimer}: geometry missing"));
        let refs: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let result = d4::dispersion_gradient(data, method, &refs, positions, charge)
            .unwrap_or_else(|error| panic!("{dimer}/{method}: {error}"));
        let max_component = result
            .gradient
            .iter()
            .flatten()
            .fold(0.0_f64, |worst, value| worst.max(value.abs()));
        let relative = (max_component - expected).abs() / expected.abs();
        if relative > worst {
            worst = relative;
            worst_row = format!("{dimer}/{method}/q={charge:+}");
        }
        println!(
            "{dimer:24} {method:6} q={charge:+} max|g| = {max_component:.12e} \
             (ref {expected:.12e}) rel = {relative:.2e}"
        );
    }
    println!("(b) worst relative |Δmax|g|| = {worst:.3e} ({worst_row})");
    // (a) 才是紧检查（逐分量 1e-8）；这一列只有最大分量、没有方向，只够确认上游约定
    // （Bohr、总电荷的符号、ATM 的 C6 用 q = 0），所以判据按相对 1e-6。
    assert!(
        worst < 1.0e-6,
        "the max gradient component differs from the oracle by {worst:.3e} relative at {worst_row}"
    );
}

/// 线性分子的平移/旋转不变性之外，再钉一条：能量对整体平移必须严格不变，
/// 而梯度之和必须为 0（已经由 (c) 覆盖）——这里检查 `dispersion_gradient` 的
/// 能量与梯度在 `q = ±1` 时也自洽（电荷相关的通路不能只在 q = 0 时正确）。
#[test]
fn d4_gradient_handles_charged_systems() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_d4_dispersion("d4").expect("d4 data");
    let molecules = geometries();
    let (symbols, positions) = molecules.get("water_dimer").expect("water_dimer");
    let refs: Vec<&str> = symbols.iter().map(String::as_str).collect();
    let n = symbols.len();
    let mut worst = 0.0_f64;
    for charge in [1.0_f64, -1.0] {
        let analytic = d4::dispersion_gradient(data, "b3lyp", &refs, positions, charge)
            .unwrap_or_else(|error| panic!("q={charge:+}: {error}"));
        for atom in 0..n {
            for axis in 0..3 {
                let numeric = richardson(positions, atom, axis, &|shifted| {
                    d4::dispersion_energy(data, "b3lyp", &refs, shifted, charge)
                        .expect("dispersion energy")
                        .energy_hartree
                });
                worst = worst.max((analytic.gradient[atom][axis] - numeric).abs());
            }
        }
        let mut translation = [0.0_f64; 3];
        for entry in &analytic.gradient {
            for (axis, value) in translation.iter_mut().enumerate() {
                *value += entry[axis];
            }
        }
        println!(
            "water_dimer q={charge:+}  E = {:+.12}  max |Σg| = {:.3e}",
            analytic.energy_hartree,
            translation
                .iter()
                .fold(0.0_f64, |worst, value| worst.max(value.abs()))
        );
    }
    println!("charged: worst |analytic − richardson| = {worst:.3e} Ha/Bohr");
    assert!(
        worst < 1.0e-8,
        "the charged-system gradient differs from finite differences by {worst:.3e} Ha/Bohr"
    );
}
