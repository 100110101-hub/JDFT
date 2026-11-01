//! FX-04 / DA-06：**零阻尼**（D3(0)）的能量与解析梯度对官方 s-dftd3。
//!
//! 为什么单独一份数据与一组测试：M06-L 与 M06-2X 在上游**只有零阻尼参数**
//! （`rs6 = 1.581 / 1.619`、`s8 = 0`），所以 M2 那条「M06-2X-D3」断言走的必须是这条路。
//! 零阻尼与 BJ 的两点式不同：`R0_ij` 来自**成对** vdW 半径表（不是两原子半径相加），
//! 阻尼函数是 `f = 1/(1 + 6·(rs·R0/r)^alp)` 而不是 `1/(r⁶+R0⁶)`。
//!
//! 参考值来自官方实现（`ZeroDampingParam`，黑盒），生成器
//! `scripts/dftd3_dispersion_reference.py`，文件
//! `tests/reference/d3zero_dispersion.csv`（每行带 `source_doi`/`source_url`）。

use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_scf::dispersion::dispersion_energy;

const HARTREE_PER_KCAL: f64 = 627.509_474_063_1;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

type Rows = (Vec<String>, Vec<Vec<String>>);

fn read_csv(path: &std::path::Path) -> Rows {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .expect("header")
        .split(',')
        .map(str::trim)
        .map(String::from)
        .collect();
    let rows = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split(',').map(str::trim).map(String::from).collect())
        .collect();
    (header, rows)
}

fn column(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|field| field == name)
        .unwrap_or_else(|| panic!("missing column `{name}`"))
}

fn dimers() -> BTreeMap<String, Vec<(String, [f64; 3])>> {
    let (header, rows) = read_csv(&reference_dir().join("d3_dimers.csv"));
    let dimer_col = column(&header, "dimer");
    let element_col = column(&header, "element");
    let x_col = column(&header, "x_angstrom");
    let y_col = column(&header, "y_angstrom");
    let z_col = column(&header, "z_angstrom");
    let mut map: BTreeMap<String, Vec<(String, [f64; 3])>> = BTreeMap::new();
    for row in rows {
        let position = [
            row[x_col].parse::<f64>().expect("x"),
            row[y_col].parse::<f64>().expect("y"),
            row[z_col].parse::<f64>().expect("z"),
        ];
        map.entry(row[dimer_col].clone())
            .or_default()
            .push((row[element_col].clone(), position));
    }
    map
}

/// (二聚体, 方法) → (能量, 逐原子梯度)。
type OfficialEntry = (f64, Vec<[f64; 3]>);

fn official() -> BTreeMap<(String, String), OfficialEntry> {
    let (header, rows) = read_csv(&reference_dir().join("d3zero_dispersion.csv"));
    let dimer_col = column(&header, "dimer");
    let method_col = column(&header, "method");
    let energy_col = column(&header, "energy_hartree");
    let gx = column(&header, "gradient_x");
    let gy = column(&header, "gradient_y");
    let gz = column(&header, "gradient_z");
    let mut map: BTreeMap<(String, String), OfficialEntry> = BTreeMap::new();
    for row in rows {
        let key = (row[dimer_col].clone(), row[method_col].clone());
        let gradient = [
            row[gx].parse::<f64>().expect("gx"),
            row[gy].parse::<f64>().expect("gy"),
            row[gz].parse::<f64>().expect("gz"),
        ];
        let entry = map
            .entry(key)
            .or_insert_with(|| (row[energy_col].parse::<f64>().expect("energy"), Vec::new()));
        entry.1.push(gradient);
    }
    map
}

fn positions_bohr(atoms: &[(String, [f64; 3])]) -> Vec<[f64; 3]> {
    let bohr = qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
    atoms
        .iter()
        .map(|(_, [x, y, z])| [x * bohr, y * bohr, z * bohr])
        .collect()
}

#[test]
fn d3zero_matches_the_official_implementation() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_dispersion("d3zero").expect("d3zero");
    assert!(data.check().is_empty(), "{:?}", data.check());
    let dimers = dimers();
    let official = official();

    let mut worst_energy = (0.0f64, String::new());
    let mut worst_gradient = (0.0f64, String::new());
    let mut checked = 0usize;
    for (dimer, atoms) in &dimers {
        let symbols: Vec<&str> = atoms.iter().map(|(symbol, _)| symbol.as_str()).collect();
        let positions = positions_bohr(atoms);
        for method in ["b3lyp", "pbe0", "m06-l", "m06-2x"] {
            let key = (dimer.clone(), method.to_string());
            let Some((expected_energy, expected_gradient)) = official.get(&key) else {
                continue;
            };
            let result = dispersion_energy(data, method, &symbols, &positions)
                .unwrap_or_else(|err| panic!("{dimer}/{method}: {err}"));
            let delta_kcal = (result.energy_hartree - expected_energy).abs() * HARTREE_PER_KCAL;
            println!(
                "{dimer:32} {method:6} ours={:+.12e} official={expected_energy:+.12e} \
                 |d|={delta_kcal:.3e} kcal/mol",
                result.energy_hartree
            );
            if delta_kcal > worst_energy.0 {
                worst_energy = (delta_kcal, format!("{dimer}/{method}"));
            }
            for (index, expected) in expected_gradient.iter().enumerate() {
                for (axis, component) in expected.iter().enumerate() {
                    let delta = (result.gradient[index][axis] - component).abs();
                    if delta > worst_gradient.0 {
                        worst_gradient = (
                            delta,
                            format!("{dimer}/{method}/atom{}/axis{axis}", index + 1),
                        );
                    }
                }
            }
            checked += 1;
        }
    }
    println!(
        "checked {checked} (dimer, functional) combinations: worst energy {:.3e} kcal/mol ({}), \
         worst gradient {:.3e} Ha/Bohr ({})",
        worst_energy.0, worst_energy.1, worst_gradient.0, worst_gradient.1
    );
    assert_eq!(checked, 40, "4 methods x 10 dimers");
    assert!(
        worst_energy.0 < 1.0e-3,
        "criterion < 1e-3 kcal/mol, worst {:.3e} at {}",
        worst_energy.0,
        worst_energy.1
    );
    assert!(
        worst_gradient.0 < 1.0e-6,
        "gradient should stay below 1e-6 Ha/Bohr, worst {:.3e} at {}",
        worst_gradient.0,
        worst_gradient.1
    );
}

#[test]
fn d3zero_analytic_gradient_matches_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_dispersion("d3zero").expect("d3zero");
    let dimers = dimers();
    let atoms = dimers.get("benzene_dimer_stacked").expect("benzene dimer");
    let symbols: Vec<&str> = atoms.iter().map(|(symbol, _)| symbol.as_str()).collect();
    let base = positions_bohr(atoms);
    let analytic = dispersion_energy(data, "m06-2x", &symbols, &base).expect("analytic");

    let step = 1.0e-5;
    let mut worst = 0.0f64;
    for index in 0..base.len() {
        for axis in 0..3 {
            let mut plus = base.clone();
            plus[index][axis] += step;
            let mut minus = base.clone();
            minus[index][axis] -= step;
            let e_plus = dispersion_energy(data, "m06-2x", &symbols, &plus)
                .expect("plus")
                .energy_hartree;
            let e_minus = dispersion_energy(data, "m06-2x", &symbols, &minus)
                .expect("minus")
                .energy_hartree;
            let numeric = (e_plus - e_minus) / (2.0 * step);
            worst = worst.max((numeric - analytic.gradient[index][axis]).abs());
        }
    }
    println!("d3zero worst analytic-vs-FD difference: {worst:.3e} Ha/Bohr");
    assert!(
        worst < 1.0e-6,
        "criterion < 1e-6 Ha/Bohr, worst {worst:.3e}"
    );
}

#[test]
fn d3zero_and_d3bj_are_separate_data_sets() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let bj = registry.get_dispersion("d3bj").expect("d3bj");
    let zero = registry.get_dispersion("d3zero").expect("d3zero");

    // ① 两份数据共享同一套参考 C6/CN/r4r2（同一个生成器写的，逐位相同）。
    assert_eq!(bj.reference_c6, zero.reference_c6);
    assert_eq!(bj.reference_cn, zero.reference_cn);
    assert_eq!(bj.r4r2, zero.r4r2);

    // ② 参数面不同：BJ 只有 4 个泛函，零阻尼有 6 个（多 M06-L / M06-2X）。
    assert!(bj.damping("m06-2x").is_none());
    let m062x = zero.zero_damping("m06-2x").expect("m06-2x zero");
    assert!((m062x.rs6 - 1.619).abs() < 1.0e-12);
    assert!((m062x.s8 - 0.0).abs() < 1.0e-12);
    assert!((m062x.alp - 14.0).abs() < 1.0e-12);

    // ③ 成对 vdW 半径：666 个元素对，且 H-H 是表的第一个值。
    assert_eq!(zero.vdw_radii_angstrom.len(), 666);
    assert!((zero.vdw_radii_angstrom["H-H"] - 2.1823).abs() < 1.0e-12);
    let hh = zero.vdw_radius_bohr("H", "H").expect("H-H radius");
    let expected = 2.1823 * qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
    assert!((hh - expected).abs() < 1.0e-12);

    // ④ 用 BJ 那份跑 M06-2X 必须报错（它没有 BJ 参数），用零阻尼那份才行。
    let symbols = ["O", "H", "H"];
    let positions = [[0.0, 0.0, 0.0], [1.4, 1.1, 0.0], [-1.4, 1.1, 0.0]];
    let error = dispersion_energy(bj, "m06-2x", &symbols, &positions)
        .expect_err("BJ data has no m06-2x parameters");
    assert!(error.contains("no D3(BJ) parameters"), "{error}");
    let result = dispersion_energy(zero, "m06-2x", &symbols, &positions).expect("zero m06-2x");
    assert!(result.energy_hartree < 0.0);
}
