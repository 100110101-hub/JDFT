//! FX-04 验收：D3(BJ) 的能量与**解析梯度**对官方 s-dftd3。
//!
//! 判据（§6, FX-04）：「D3 梯度 vs 有限差分 < 1e-6 Ha/Bohr；与官方参考实现（黑盒脚本）
//! 在 10 个二聚体上差 < 1e-3 kcal/mol；`dispersion_energy_hartree` 进 `result.json`」。
//!
//! 本文件覆盖前两条（第三条在 `qwb-engine` 的端到端测试里）：
//!
//! 1. **对官方实现**：`tests/reference/d3_dimers.csv`（10 个二聚体的理想化几何）与
//!    `d3_dispersion.csv`（官方 s-dftd3 给的每体系能量与每原子梯度）—— 40 个
//!    (二聚体, 泛函) 组合逐一比较，判据 1e-3 kcal/mol；
//! 2. **梯度自洽**：解析梯度 vs 中心差分 < 1e-6 Ha/Bohr；
//! 3. 没有 BJ 参数的泛函（M06-L / M06-2X）必须显式报错，而不是静默返回 0；
//! 4. 两次调用逐位相同（色散是纯函数，没有随机性也没有并行归约）。
//!
//! 参考值的来源是**官方实现本身**（黑盒），生成器是
//! `scripts/dftd3_dispersion_reference.py`（逐行带 `source_doi`/`source_url`）。

use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_scf::dispersion::dispersion_energy;

/// 1 kcal/mol 等于多少 Hartree（CODATA：1 Ha = 627.5094740631 kcal/mol）。
const HARTREE_PER_KCAL: f64 = 1.0 / 627.509_474_063_1;

/// 官方参考的能量与逐原子梯度（键是「二聚体, 泛函」）。
type OfficialEntry = (f64, Vec<[f64; 3]>);

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// 从 CSV 里按列名取值（仓库里的参考 CSV 都是 `key,value` 扁平行）。
fn read_csv(path: &std::path::Path) -> (Vec<String>, Vec<Vec<String>>) {
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
        .unwrap_or_else(|| panic!("missing column `{name}` in {header:?}"))
}

/// 二聚体 → [(元素, [x, y, z]_Å)]（保持 CSV 的行序）。
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

/// (二聚体, 泛函) → (能量, 逐原子梯度)。
fn official() -> BTreeMap<(String, String), OfficialEntry> {
    let (header, rows) = read_csv(&reference_dir().join("d3_dispersion.csv"));
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

/// 把 Å 转成 Bohr（引擎内部单位；参考 CSV 的两个代码都用这个常数换算）。
fn positions_bohr(atoms: &[(String, [f64; 3])]) -> Vec<[f64; 3]> {
    let bohr = qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
    atoms
        .iter()
        .map(|(_, [x, y, z])| [x * bohr, y * bohr, z * bohr])
        .collect()
}

#[test]
fn fx04_dispersion_matches_the_official_implementation() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_dispersion("d3bj").expect("d3bj");
    let dimers = dimers();
    let official = official();
    assert_eq!(dimers.len(), 10, "the reference set must hold 10 dimers");

    let mut worst_energy = (0.0f64, String::new());
    let mut worst_gradient = (0.0f64, String::new());
    let mut checked = 0usize;
    for (dimer, atoms) in &dimers {
        let symbols: Vec<&str> = atoms.iter().map(|(symbol, _)| symbol.as_str()).collect();
        let positions = positions_bohr(atoms);
        for method in ["b3lyp", "pbe", "pbe0", "hf"] {
            let key = (dimer.clone(), method.to_string());
            let (expected_energy, expected_gradient) = official
                .get(&key)
                .unwrap_or_else(|| panic!("no official reference for {dimer}/{method}"));
            assert_eq!(expected_gradient.len(), atoms.len());
            let result = dispersion_energy(data, method, &symbols, &positions)
                .unwrap_or_else(|err| panic!("{dimer}/{method}: {err}"));
            let delta_energy = (result.energy_hartree - expected_energy).abs();
            let delta_kcal = delta_energy * 627.509_474_063_1;
            println!(
                "{dimer:32} {method:6} ours={:+.12e} official={expected_energy:+.12e} \
                 |d|={delta_kcal:.3e} kcal/mol",
                result.energy_hartree
            );
            if delta_kcal > worst_energy.0 {
                worst_energy = (delta_kcal, format!("{dimer}/{method}"));
            }
            for (index, expected) in expected_gradient.iter().enumerate() {
                for (axis, expected_component) in expected.iter().enumerate() {
                    let delta = (result.gradient[index][axis] - expected_component).abs();
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
        "checked {checked} (dimer, functional) combinations: worst energy {:.3e} kcal/mol \
         ({}), worst gradient {:.3e} Ha/Bohr ({})",
        worst_energy.0, worst_energy.1, worst_gradient.0, worst_gradient.1
    );
    assert!(
        worst_energy.0 < 1.0e-3,
        "FX-04 criterion is < 1e-3 kcal/mol, worst was {:.3e} at {}",
        worst_energy.0,
        worst_energy.1
    );
    // 梯度判据是「vs 有限差分 < 1e-6 Ha/Bohr」，所以对官方实现的梯度用同量级的容差。
    assert!(
        worst_gradient.0 < 1.0e-6,
        "gradient vs the official implementation should be below 1e-6 Ha/Bohr, worst {:.3e} at {}",
        worst_gradient.0,
        worst_gradient.1
    );
}

#[test]
fn fx04_analytic_gradient_matches_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_dispersion("d3bj").expect("d3bj");
    let dimers = dimers();
    let atoms = dimers.get("benzene_dimer_stacked").expect("benzene dimer");
    let symbols: Vec<&str> = atoms.iter().map(|(symbol, _)| symbol.as_str()).collect();
    let base = positions_bohr(atoms);
    let analytic = dispersion_energy(data, "b3lyp", &symbols, &base).expect("analytic");

    let step = 1.0e-5;
    let mut worst = 0.0f64;
    for index in 0..base.len() {
        for axis in 0..3 {
            let mut plus = base.clone();
            plus[index][axis] += step;
            let mut minus = base.clone();
            minus[index][axis] -= step;
            let e_plus = dispersion_energy(data, "b3lyp", &symbols, &plus)
                .expect("plus")
                .energy_hartree;
            let e_minus = dispersion_energy(data, "b3lyp", &symbols, &minus)
                .expect("minus")
                .energy_hartree;
            let numeric = (e_plus - e_minus) / (2.0 * step);
            let delta = (numeric - analytic.gradient[index][axis]).abs();
            if delta > worst {
                worst = delta;
            }
        }
    }
    println!("worst analytic-vs-FD difference: {worst:.3e} Ha/Bohr");
    assert!(
        worst < 1.0e-6,
        "FX-04 criterion is < 1e-6 Ha/Bohr, worst was {worst:.3e}"
    );
}

#[test]
fn fx04_functionals_without_bj_parameters_are_rejected() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_dispersion("d3bj").expect("d3bj");
    let symbols = ["O", "H", "H"];
    let positions = [[0.0, 0.0, 0.0], [1.4, 1.1, 0.0], [-1.4, 1.1, 0.0]];
    for method in ["m06-l", "m06-2x", "b97-d"] {
        let error = dispersion_energy(data, method, &symbols, &positions)
            .expect_err("functionals without BJ parameters must be rejected");
        println!("{method}: {error}");
        assert!(
            error.contains("no D3(BJ) parameters"),
            "unexpected error for {method}: {error}"
        );
    }
    // 有参数的那四个都要能算。
    for method in ["b3lyp", "pbe", "pbe0", "hf"] {
        let result = dispersion_energy(data, method, &symbols, &positions).expect(method);
        assert!(result.energy_hartree < 0.0, "{method}");
        assert_eq!(result.gradient.len(), 3);
    }
}

#[test]
fn fx04_is_deterministic_and_reports_pairs() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_dispersion("d3bj").expect("d3bj");
    let dimers = dimers();
    let atoms = dimers.get("water_dimer").expect("water dimer");
    let symbols: Vec<&str> = atoms.iter().map(|(symbol, _)| symbol.as_str()).collect();
    let positions = positions_bohr(atoms);
    let first = dispersion_energy(data, "b3lyp", &symbols, &positions).expect("first");
    let second = dispersion_energy(data, "b3lyp", &symbols, &positions).expect("second");
    assert_eq!(first.energy_hartree, second.energy_hartree);
    assert_eq!(first.gradient, second.gradient);
    // 逐对能量之和必须等于总能量（能量是「对之和」，没有三体项）。
    let sum: f64 = first.pairs.iter().map(|(_, _, energy)| energy).sum();
    assert!(
        (sum - first.energy_hartree).abs() < 1.0e-14,
        "sum of pair energies {sum} != total {}",
        first.energy_hartree
    );
    // 6 个原子 → 15 个对。
    assert_eq!(first.pairs.len(), 15);
    // 能量量级：水二聚体的 D3(BJ) 在 B3LYP 下是 −2.19e-3 Ha 量级。
    let kcal = first.energy_hartree / HARTREE_PER_KCAL;
    println!(
        "water dimer D3(BJ)/B3LYP = {} Ha = {kcal} kcal/mol",
        first.energy_hartree
    );
    assert!(kcal < -1.0 && kcal > -2.0, "{kcal} kcal/mol out of range");
}
