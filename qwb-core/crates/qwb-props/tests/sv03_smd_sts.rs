//! SV-03 第三半验收：**每原子 CDS 表面张力 `STS`**（上游 `SMXCDS`）对上官方 `E_CDS`。
//!
//! 这一层把三块拼起来：零阶 σ 表 + H 的 `HSIGMA` 修正 + 成对修正 `SIGMA(100+pair)`，
//! 权重是 `COT = exp(Δ/(d − (R + Δ)))`。表面面积仍来自 [`smd_surface`]（`w = 0`）。
//!
//! 参考值来自 `scripts/smd_cds_legacy_reference.py`（PySCF 2.14.0 的 `get_cds_legacy`），
//! 八个体系：四个"零阶即全部"的（水/HF/HCl/H₂S）只测面积，另外四个（甲烷/甲醇/丙酮/乙腈）
//! 分别触发 H–C、O–C（含 Δ = 0.10 / R = 1.330 特例）、O–C×2、N–C 那套软核与三键。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_props::smd::{SmdDescriptors, cds_constant_term};
use qwb_props::smd_sigma::{cds_energy_kcal_from_sts, smd_sts};
use qwb_props::smd_surface::smd_surface;
use qwb_registry::Registry;
use qwb_registry::smd_atomic::SmdAtomic;

/// 一个来自参考 CSV 的体系。
struct OracleSystem {
    name: String,
    solvent: String,
    atoms: Vec<(String, [f64; 3])>,
    energy_kcal: f64,
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn smd() -> SmdAtomic {
    let registry = Registry::open(data_root()).expect("qwb-data");
    registry
        .get_smd_atomic("smd-atomic")
        .expect("smd-atomic")
        .clone()
}

fn read_oracle_systems() -> Vec<OracleSystem> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/smd_cds_legacy.csv");
    let text = std::fs::read_to_string(&path).expect("smd_cds_legacy.csv");
    let mut systems: Vec<OracleSystem> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 13, "line {}: expected 13 columns", index + 1);
        let name = fields[0].to_string();
        let solvent = fields[1].to_string();
        let symbol = fields[3].to_string();
        let position = [
            fields[4].parse::<f64>().expect("x"),
            fields[5].parse::<f64>().expect("y"),
            fields[6].parse::<f64>().expect("z"),
        ];
        let energy_kcal = fields[7].parse::<f64>().expect("e_cds");
        match systems.last_mut() {
            Some(last) if last.name == name && last.solvent == solvent => {
                last.atoms.push((symbol, position));
            }
            _ => systems.push(OracleSystem {
                name,
                solvent,
                atoms: vec![(symbol, position)],
                energy_kcal,
            }),
        }
    }
    assert!(
        systems.len() >= 8,
        "the oracle file must cover all eight systems"
    );
    systems
}

fn molecule_of(system: &OracleSystem) -> Molecule {
    let atoms = system
        .atoms
        .iter()
        .map(|(symbol, position)| Atom::new(symbol, *position))
        .collect();
    Molecule::new(atoms, 0, 1)
}

/// 从**本仓库**的 `qwb-data/solvent/<name>.yaml` 取六个 SMD 描述符。
///
/// 刻意不用 PySCF 的 `solvent_db`：参考值那边用 PySCF 自己的溶剂表，我们这边用仓库的
/// 六个数 —— 于是这条对照同时验证"数据文件里的描述符"与"非水 σ 的拼装规则"。
fn descriptors_for(name: &str) -> SmdDescriptors {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let solvent = registry.get_solvent(name).expect("solvent");
    let smd = solvent.smd.as_ref().expect("the SMD block");
    SmdDescriptors::new(
        smd.abraham_alpha.expect("abraham_alpha"),
        smd.abraham_beta.expect("abraham_beta"),
        smd.phi_sq.expect("phi_sq"),
        smd.gamma_cal_per_mol_angstrom2
            .expect("gamma_cal_per_mol_angstrom2"),
        smd.psi_sq.expect("psi_sq"),
        smd.refractive_index_20c.expect("refractive_index_20c"),
    )
}

#[test]
fn the_per_atom_surface_tension_reproduces_the_upstream_cds_energy() {
    let data = smd();
    // 8192 点时水相最差 1.63e-3，32768 点降到 8.4e-4，与上一半实测的 `O(1/√N)` 采样噪声
    // 同量级。判据定在 2e-3。
    let points = 32768;
    let mut worst = 0.0_f64;
    for system in read_oracle_systems() {
        let water = system.solvent == "water";
        let descriptors = if water {
            None
        } else {
            Some(descriptors_for(&system.solvent))
        };
        let molecule = molecule_of(&system);
        let surface = smd_surface(&data, &molecule, points, 0.0).expect("surface");
        let sts = smd_sts(&data, &molecule, water, descriptors.as_ref()).expect("sts");
        let cssigm = cds_constant_term(&data, water, descriptors.as_ref());
        let energy =
            cds_energy_kcal_from_sts(&sts, cssigm, &surface.areas_angstrom2).expect("cds energy");
        let difference = (energy - system.energy_kcal).abs() / system.energy_kcal.abs();
        // 非水溶剂的 σ 有正有负，`E_CDS` 常常是几项相消后的小数（甲苯里的甲烷：各项
        // 绝对值合计 ~5.7 kcal/mol，而 `E_CDS` 只有 0.53），所以纯相对差会被放大约 13 倍。
        // 判据因此用"对各项绝对值之和"的相对差 —— 对相消不敏感，量纲仍是能量。
        let scale: f64 = sts
            .iter()
            .zip(&surface.areas_angstrom2)
            .map(|(sigma, area)| ((sigma + cssigm) * area).abs())
            .sum::<f64>()
            / 1000.0;
        let deviation = (energy - system.energy_kcal).abs() / scale.max(1.0e-12);
        println!(
            "{:20} {:12} ours {energy:.6} vs oracle {:.6} kcal/mol \
             (rel {difference:.2e}, vs sum of |terms| {deviation:.2e})",
            system.name, system.solvent, system.energy_kcal
        );
        worst = worst.max(deviation);
    }
    println!("worst deviation (normalised by the sum of |terms|): {worst:.3e}");
    assert!(
        worst < 1.0e-3,
        "the per-atom sigma assembly must reproduce the oracle, got {worst:.3e}"
    );
}

#[test]
fn the_hand_checked_surface_tensions_match_the_fortran_branches() {
    let data = smd();

    // 水：O 的零阶 σ = 0、H 只挂 HSIGMA(O) = 0，成对项一条都不触发。
    let water = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    );
    let sts = smd_sts(&data, &water, true, None).expect("sts");
    assert!((sts[0] - 0.0).abs() < 1.0e-12, "O has sigma 0");
    for hydrogen in &sts[1..] {
        assert!((hydrogen - 48.69).abs() < 1.0e-12, "H has sigma 48.69");
    }

    // 甲烷：C–H 1.087 Å，RKKVAL(H,C) = 1.55、Δ = 0.30 → COT = exp(0.30/(1.087 − 1.85))。
    let methane = Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.6276, 0.6276, 0.6276]),
            Atom::new("H", [-0.6276, -0.6276, 0.6276]),
            Atom::new("H", [-0.6276, 0.6276, -0.6276]),
            Atom::new("H", [0.6276, -0.6276, -0.6276]),
        ],
        0,
        1,
    );
    let sts = smd_sts(&data, &methane, true, None).expect("sts");
    let bond = 0.6276 * 3.0_f64.sqrt();
    let cot = (0.30 / (bond - 1.85)).exp();
    let expected_h = 48.69 - 60.77 * cot;
    println!("methane: COT = {cot:.6}, STS(H) = {:.6}", sts[1]);
    assert!(
        (sts[0] - 129.74).abs() < 1.0e-12,
        "C keeps its zero-order sigma"
    );
    for hydrogen in &sts[1..] {
        assert!(
            (hydrogen - expected_h).abs() < 1.0e-9,
            "H follows HSIGMA(C)"
        );
    }

    // 甲醇：O 与 C 成键、距离 1.40 Å，**O–C 特例** R = 1.330、Δ = 0.10
    // → COT = exp(0.10/(1.40 − 1.43))，STS(O) = COT·68.69。若漏了特例（R = 1.84、Δ = 0.30），
    // COT 会变成 0.667、STS(O) 变成 45.8 —— 差一个数量级，所以这条能钉住特例。
    let methanol = Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("O", [1.4, 0.0, 0.0]),
            Atom::new("H", [-0.3636, 0.9382, 0.0]),
            Atom::new("H", [-0.3636, -0.4691, 0.8126]),
            Atom::new("H", [-0.3636, -0.4691, -0.8126]),
            Atom::new("H", [1.76, 0.83, 0.0]),
        ],
        0,
        1,
    );
    let sts = smd_sts(&data, &methanol, true, None).expect("sts");
    let special = (0.10_f64 / (1.4 - 1.43)).exp();
    println!(
        "methanol: STS(O) = {:.6} (special COT = {special:.6})",
        sts[1]
    );
    assert!(
        (sts[1] - 68.69 * special).abs() < 1.0e-9,
        "the O-C branch must use the R = 1.330 / delta = 0.10 special case"
    );
}

#[test]
fn the_unimplemented_pair_branches_are_zero_in_the_data() {
    // 上游还有 O–Si(117)、O–S(118)、S–S(107)、S–P(115)、H–N(2)(112)、H–O(2)(113)、
    // N–C(2)(111) 七条分支。本仓库的数据里它们是 0，所以 `smd_sigma` 不写这七条 ——
    // 这条测试是那个决定的**守卫**：上游数据或非水溶剂描述符一旦让它们非零，就红。
    let data = smd();
    let water = &data.cds.water.pair_sigma_cal_per_mol_angstrom2;
    let zero_pairs = [2usize, 4, 7, 11, 12, 13, 15, 17, 18];
    for pair in zero_pairs {
        assert_eq!(
            water[pair - 1],
            0.0,
            "aqueous SIGMA({}) must be zero",
            100 + pair
        );
    }
    // 非水相：那七条分支在 σ_N/A/B 三张表里合起来必须是 0。
    let table = &data.cds.nonaqueous;
    for pair in [7usize, 11, 12, 13, 15, 17, 18] {
        let slot = pair - 1;
        assert_eq!(
            table.pair_sigma_n[slot],
            0.0,
            "nonaqueous SIGMA_N({})",
            100 + pair
        );
        assert_eq!(
            table.pair_sigma_a[slot],
            0.0,
            "nonaqueous SIGMA_A({})",
            100 + pair
        );
        assert_eq!(
            table.pair_sigma_b[slot],
            0.0,
            "nonaqueous SIGMA_B({})",
            100 + pair
        );
    }
}

#[test]
fn the_nonaqueous_path_needs_the_solvent_descriptors() {
    let data = smd();
    let methane = Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.6276, 0.6276, 0.6276]),
        ],
        0,
        1,
    );
    // 非水相：σ 与 HSIGMA 都要 Abraham 描述符，缺了就显式报错（不静默用 0）。
    let error = smd_sts(&data, &methane, false, None).expect_err("descriptors are required");
    println!("missing descriptors -> {error}");
    // 给了描述符就能算（甲苯的六个数来自本仓库的 `solvent/toluene.yaml`）。
    let descriptors = SmdDescriptors::new(0.0, 0.14, 0.0, 38.14, 0.0, 1.4969);
    let sts = smd_sts(&data, &methane, false, Some(&descriptors)).expect("nonaqueous sts");
    println!("nonaqueous methane STS = {:?}", sts);
    assert!(sts.iter().all(|value| value.is_finite()));
}
