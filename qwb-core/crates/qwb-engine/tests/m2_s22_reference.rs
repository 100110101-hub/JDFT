//! M2 ④ 的数据侧验收：**S22 参考集**完整、可读，且苯二聚体的构象参数被量出来。
//!
//! 背景（`docs/status-v2.yaml` 的 M2）：第 ④ 条判据要"苯二聚体的相互作用能与文献差
//! < 1 kcal/mol"，而 `examples/benzene_dimer_d3.yaml` 用的是**理想化**几何（环间距 3.5 Å、
//! 错开 1.4 Å），不能直接与 S22 的文献值比。这一轮把 S22 的几何与参考能取回并按 sha256 钉住
//! （`scripts/gen_s22_reference.py`，数据来自 ASE 的 `ase/data/s22.py`），这条测试负责：
//!
//! 1. 数据集完整（22 个二聚体、414 个原子、每个体系的原子数与行数一致）；
//! 2. 参考能自洽（eV 与 kcal/mol 两列按 23.060548867 换算，误差 < 1e-6）；
//! 3. **把两个几何的"环平面间距 / 滑移"都算出来** —— 于是"能不能直接比"从一个说法
//!    变成两个数（S22 约 3.36 / 1.70 Å vs 例子 3.50 / 1.40 Å）。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::Atom;

/// eV → kcal/mol（1 cal = 4.184 J 精确，1 eV = 1.602176634e-19 J）。
const EV_TO_KCAL: f64 = 23.060_548_867;

struct Dimer {
    name: String,
    symbols: Vec<String>,
    positions: Vec<[f64; 3]>,
    n_atoms: usize,
    energy_ev: f64,
    energy_kcal: f64,
}

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_s22() -> Vec<Dimer> {
    let path = crate_dir().join("tests/reference/s22_dimers.csv");
    let text = std::fs::read_to_string(&path).expect("s22_dimers.csv");
    let mut order: Vec<String> = Vec::new();
    let mut dimers: BTreeMap<String, Dimer> = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 11, "line {}: expected 11 columns", index + 1);
        let name = fields[0].to_string();
        let n_atoms = fields[1].parse::<usize>().expect("n_atoms");
        let symbol = fields[3].to_string();
        let position = [
            fields[4].parse::<f64>().expect("x"),
            fields[5].parse::<f64>().expect("y"),
            fields[6].parse::<f64>().expect("z"),
        ];
        let energy_ev = fields[7].parse::<f64>().expect("e_int_ev");
        let energy_kcal = fields[8].parse::<f64>().expect("e_int_kcal");
        let entry = dimers.entry(name.clone()).or_insert_with(|| {
            order.push(name.clone());
            Dimer {
                name: name.clone(),
                symbols: Vec::new(),
                positions: Vec::new(),
                n_atoms,
                energy_ev,
                energy_kcal,
            }
        });
        assert_eq!(entry.n_atoms, n_atoms, "{name}: n_atoms must be constant");
        entry.symbols.push(symbol);
        entry.positions.push(position);
    }
    order
        .into_iter()
        .map(|name| dimers.remove(&name).expect("dimer"))
        .collect()
}

/// 读 `examples/benzene_dimer_d3.yaml` 的原子（`- symbol: X` + `position_angstrom: [...]`）。
fn read_example() -> Vec<(String, [f64; 3])> {
    let path = crate_dir().join("../../examples/benzene_dimer_d3.yaml");
    let text = std::fs::read_to_string(&path).expect("benzene_dimer_d3.yaml");
    let mut atoms = Vec::new();
    let mut pending: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("- symbol:") {
            pending = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("position_angstrom:") {
            let body = rest.trim().trim_start_matches('[').trim_end_matches(']');
            let values: Vec<f64> = body
                .split(',')
                .map(|value| value.trim().parse::<f64>().expect("coordinate"))
                .collect();
            assert_eq!(values.len(), 3, "position_angstrom needs three values");
            let symbol = pending.take().expect("symbol before position");
            atoms.push((symbol, [values[0], values[1], values[2]]));
        }
    }
    atoms
}

/// 两半（各 12 个原子）的**环平面间距**与**滑移**：`Δ = c2 − c1` 在环法向上的分量与面内分量。
fn stacking(atoms: &[(String, [f64; 3])], split: usize) -> (f64, f64) {
    assert_eq!(atoms.len(), 2 * split, "each half must have `split` atoms");
    let centroid = |range: std::ops::Range<usize>| -> [f64; 3] {
        let carbons: Vec<[f64; 3]> = range
            .clone()
            .filter(|index| atoms[*index].0 == "C")
            .map(|index| atoms[index].1)
            .collect();
        assert_eq!(carbons.len(), 6, "each benzene has six carbons");
        let mut sum = [0.0_f64; 3];
        for position in &carbons {
            for axis in 0..3 {
                sum[axis] += position[axis];
            }
        }
        [sum[0] / 6.0, sum[1] / 6.0, sum[2] / 6.0]
    };
    let c1 = centroid(0..split);
    let c2 = centroid(split..2 * split);
    // 环 1 的法向：(p1 − p0) × (p2 − p1)
    let carbon = |offset: usize| atoms[offset].1;
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let mut normal = cross(sub(carbon(1), carbon(0)), sub(carbon(2), carbon(1)));
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    for component in &mut normal {
        *component /= length;
    }
    let delta = sub(c2, c1);
    let separation = (delta[0] * normal[0] + delta[1] * normal[1] + delta[2] * normal[2]).abs();
    let total = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    let slip = (total * total - separation * separation).max(0.0).sqrt();
    (separation, slip)
}

#[test]
fn the_s22_set_is_complete_and_self_consistent() {
    let dimers = read_s22();
    assert_eq!(dimers.len(), 22, "the S22 set has 22 dimers");
    let mut total_atoms = 0;
    for dimer in &dimers {
        assert_eq!(
            dimer.symbols.len(),
            dimer.n_atoms,
            "{}: row count must equal n_atoms",
            dimer.name
        );
        total_atoms += dimer.n_atoms;
        assert!(
            dimer.energy_ev < 0.0,
            "{}: the reference interaction energy is attractive",
            dimer.name
        );
        let converted = dimer.energy_ev * EV_TO_KCAL;
        assert!(
            (converted - dimer.energy_kcal).abs() < 1.0e-6,
            "{}: eV/kcal columns disagree ({} vs {})",
            dimer.name,
            converted,
            dimer.energy_kcal
        );
    }
    assert_eq!(total_atoms, 414, "the S22 set has 414 atoms in total");
    // 每个原子都必须是真实元素（拒绝空行/坏行被静默跳过）。
    for dimer in &dimers {
        for symbol in &dimer.symbols {
            assert!(
                qwb_registry::elements::atomic_number(symbol).is_some(),
                "{}: unknown symbol {symbol}",
                dimer.name
            );
        }
    }
}

#[test]
fn the_benzene_dimer_geometry_is_the_slipped_parallel_stack() {
    let dimers = read_s22();
    let benzene = dimers
        .iter()
        .find(|dimer| dimer.name == "Benzene_dimer_parallel_displaced")
        .expect("the benzene dimer is part of S22");
    assert_eq!(benzene.n_atoms, 24, "C6H6 dimer = 24 atoms");
    let atoms: Vec<(String, [f64; 3])> = benzene
        .symbols
        .iter()
        .cloned()
        .zip(benzene.positions.iter().copied())
        .collect();
    // S22 的 `dimer atoms` 是 [12, 12]：两个单体各 12 个原子。
    let (separation, slip) = stacking(&atoms, 12);
    println!(
        "S22 benzene dimer: reference {:.6} kcal/mol, plane separation {separation:.4} A, \
         slip {slip:.4} A",
        benzene.energy_kcal
    );
    assert!(
        (3.30..3.45).contains(&separation),
        "the S22 benzene dimer stacks at ~3.36 A, got {separation:.4}"
    );
    assert!(
        (1.60..1.80).contains(&slip),
        "the S22 benzene dimer is slipped by ~1.70 A, got {slip:.4}"
    );
    // 最小跨单体距离：不是 3.36（那是平面间距），而是 ~3.37。
    let mut minimum = f64::INFINITY;
    for (_, left) in atoms.iter().take(12) {
        for (_, right) in atoms.iter().skip(12) {
            let distance = ((left[0] - right[0]).powi(2)
                + (left[1] - right[1]).powi(2)
                + (left[2] - right[2]).powi(2))
            .sqrt();
            minimum = minimum.min(distance);
        }
    }
    println!("minimum inter-monomer distance {minimum:.4} A");
    assert!((3.30..3.45).contains(&minimum));
}

#[test]
fn the_idealised_example_differs_from_the_s22_geometry() {
    // 这条断言是 M2 ④ 的"为什么现在不能比"的**可执行版本**：例子用的是 3.5 / 1.4 的理想化
    // 构象，S22 的平衡构象是 ~3.36 / 1.70。差值本身不大，但相互作用能在 def2-SVP 上对
    // 间距极敏感（色散 ~R^-6），所以两者不能互相冒充。
    let example = read_example();
    assert_eq!(example.len(), 24, "the example is a C6H6 dimer too");
    let (example_separation, example_slip) = stacking(&example, 12);
    let dimers = read_s22();
    let benzene = dimers
        .iter()
        .find(|dimer| dimer.name == "Benzene_dimer_parallel_displaced")
        .expect("the benzene dimer is part of S22");
    let atoms: Vec<(String, [f64; 3])> = benzene
        .symbols
        .iter()
        .cloned()
        .zip(benzene.positions.iter().copied())
        .collect();
    let (separation, slip) = stacking(&atoms, 12);
    println!(
        "example {example_separation:.4} / {example_slip:.4} A vs S22 {separation:.4} / \
         {slip:.4} A"
    );
    assert!(
        (example_separation - 3.5).abs() < 1.0e-6,
        "the example stacks at 3.5 A by construction"
    );
    assert!(
        (example_slip - 1.4).abs() < 1.0e-6,
        "the example is slipped by 1.4 A by construction"
    );
    assert!(
        (example_separation - separation).abs() > 0.1,
        "the example must not be the S22 geometry"
    );
}

#[test]
fn the_benzene_monomer_reference_is_the_atom_set_we_use() {
    // 一条廉价的一致性检查：S22 里第一个苯环的 12 个原子与例子里的第一个苯环**不是**同一份
    // 坐标（一个是 MP2/cc-pVTZ 优化、一个是理想化），但元素组成必须一样。
    let dimers = read_s22();
    let benzene = dimers
        .iter()
        .find(|dimer| dimer.name == "Benzene_dimer_parallel_displaced")
        .expect("benzene dimer");
    let monomer: Vec<&String> = benzene.symbols.iter().take(12).collect();
    let carbons = monomer.iter().filter(|symbol| ***symbol == "C").count();
    let hydrogens = monomer.iter().filter(|symbol| ***symbol == "H").count();
    assert_eq!((carbons, hydrogens), (6, 6), "one monomer is C6H6");
    // `Atom` 构造器也要接受这些符号（引擎侧的入口）。
    let molecule: Vec<Atom> = benzene
        .symbols
        .iter()
        .zip(&benzene.positions)
        .map(|(symbol, position)| Atom::new(symbol, *position))
        .collect();
    assert_eq!(molecule.len(), 24);
}
