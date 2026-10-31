//! RS-04 的第一块验收：**抗磁屏蔽**。
//!
//! `σ_dia(A) = 1/(3c²) Σ_μν P_μν ⟨μ| 1/|r − R_A| |ν⟩`（ppm，核在原点，Ramsey 抗磁项）。
//!
//! 三条检查：
//! 1. 与 PySCF 的**独立**实现逐位对照（水 + 甲烷 + 甲醇，14 个核，STO-3G）。基准由
//!    `scripts/pyscf_nmr_shieldings_reference.py` 生成 —— 那边是 PySCF 自己的
//!    `int1e_rinv`（把分子平移到核在原点后取 `1/|r|`），这边是高斯递推的核吸引矩阵
//!    （单位电荷技巧）。两条不同的积分路径。
//!
//!    **几何从 `nmr_molecules.csv` 读**，不在测试里硬编码：同一份几何同时喂 PySCF 生成器
//!    与这里，"同一个几何"才成立（DA-08）。参考文件每行还带 `source_doi`/`source_url`
//!    （PySCF 论文 DOI，Crossref 核对过卷期），门禁 `scripts/check_reference_sources.py` 查。
//! 2. **旋转不变性**（精确）：`1/r_A` 是标量算符，把分子整体转一下，每个核的 σ 必须不变。
//! 3. **自由原子极限**：20 bohr 处的一对 H 原子，每个核只剩自己的一个电子，
//!    σ 必须回到「单个电子 + STO-3G 的 ⟨1/r_H⟩ = 1.2266」→ 21.77 ppm。
//!
//! **这一块不等于化学位移**：顺磁项（`⟨μ|L_A/r_A³|ν⟩` + CPHF）还没做，而 RS-04 的判据
//! 要求 GIAO/London 轨道（规范不变性 < 1 ppm）。所以本测试只钉「抗磁项算对了」。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_props::diamagnetic_shielding_ppm;
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join(name)
}

/// 从 `nmr_molecules.csv` 读几何（列：system,atom_index,symbol,x,y,z,source_doi,source_url）。
///
/// 分子按出现顺序组装；`atom_index` 必须从 0 连续递增 —— 否则说明文件被改坏了。
fn reference_molecules() -> Vec<(String, Molecule)> {
    let text = std::fs::read_to_string(reference_path("nmr_molecules.csv"))
        .expect("nmr_molecules.csv（DA-08 的几何真值源）");
    let mut out: Vec<(String, Molecule)> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 6,
            "row {lineno} has {} fields",
            fields.len()
        );
        let name = fields[0].trim().to_string();
        let index: usize = fields[1].trim().parse().expect("atom_index");
        let symbol = fields[2].trim();
        let mut position = [0.0_f64; 3];
        for (slot, field) in position.iter_mut().zip(&fields[3..6]) {
            *slot = field.trim().parse().expect("coordinate");
        }
        match out.last_mut() {
            Some((label, molecule)) if *label == name => {
                assert_eq!(
                    molecule.atoms.len(),
                    index,
                    "{name}: atom_index must be dense and ascending"
                );
                molecule.atoms.push(Atom::new(symbol, position));
            }
            _ => {
                assert_eq!(index, 0, "{name}: the first atom must be index 0");
                out.push((name, Molecule::new(vec![Atom::new(symbol, position)], 0, 1)));
            }
        }
    }
    out
}

/// `nmr_molecules.csv` 里的某个体系（不变性测试要与参考集用同一个几何）。
fn reference_molecule(label: &str) -> Molecule {
    reference_molecules()
        .into_iter()
        .find(|(name, _)| name == label)
        .map(|(_, molecule)| molecule)
        .unwrap_or_else(|| panic!("{label} in nmr_molecules.csv"))
}

/// 绕 `(1,1,1)/√3` 转 30°，用 Rodrigues 公式手写（不引罗盘依赖）。
fn rotation(angle: f64) -> [[f64; 3]; 3] {
    let n = 1.0 / 3.0_f64.sqrt();
    let k = [n, n, n];
    let (s, c) = angle.sin_cos();
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = c * f64::from(u8::from(i == j)) + (1.0 - c) * k[i] * k[j];
        }
    }
    // + sin(angle) · ε_ijk k_k
    let e = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] += s * e[i][j];
        }
    }
    r
}

fn rotate(molecule: &Molecule, r: &[[f64; 3]; 3]) -> Molecule {
    let atoms: Vec<Atom> = molecule
        .atoms
        .iter()
        .map(|atom| {
            let p = atom.position_bohr();
            let mut q = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    q[i] += r[i][j] * p[j];
                }
                // `Atom::new` 吃 Å，`position_bohr` 吐 bohr。
                q[i] /= qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
            }
            Atom::new(&atom.symbol, q)
        })
        .collect();
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

/// 一次闭壳层（RKS 走空泛函 = HF）求解，返回密度矩阵。
fn solve_density(
    registry: &Registry,
    molecule: &Molecule,
) -> (InternalBasis, qwb_linalg::Matrix, f64) {
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), molecule).unwrap();
    let tensor = two_electron(&basis, molecule).unwrap();
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), molecule).unwrap();
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options).unwrap();
    (basis, result.density, result.energy_hartree)
}

#[test]
fn rs04_diamagnetic_shielding_matches_the_pyscf_reference() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecules = reference_molecules();
    let labels: Vec<&str> = molecules.iter().map(|(label, _)| label.as_str()).collect();
    assert_eq!(
        labels,
        ["water", "methane", "methanol"],
        "DA-08 的 NMR 家族是这三个体系"
    );
    // 列：system,basis,atom_index,symbol,inverse_r_bohr,sigma_dia_ppm,source_doi,source_url
    let text = std::fs::read_to_string(reference_path("nmr_diamagnetic.csv"))
        .expect("nmr_diamagnetic.csv");
    let mut checked = 0usize;
    let mut worst = 0.0_f64;
    let mut sources = std::collections::BTreeSet::new();
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 8, "nmr_diamagnetic.csv 每行 8 列：{fields:?}");
        let name = fields[0].trim();
        let index: usize = fields[2].trim().parse().expect("atom index");
        let symbol = fields[3].trim();
        let reference: f64 = fields[5].trim().parse().expect("reference");
        sources.insert((fields[6].trim().to_string(), fields[7].trim().to_string()));
        let molecule = molecules
            .iter()
            .find(|(label, _)| label == name)
            .map(|(_, molecule)| molecule)
            .unwrap_or_else(|| panic!("unknown system {name}"));
        let (basis, density, _energy) = solve_density(&registry, molecule);
        let shieldings =
            diamagnetic_shielding_ppm(&basis, molecule, &density).expect("diamagnetic shielding");
        let ours = shieldings[index];
        let relative = (ours - reference).abs() / reference.abs();
        worst = worst.max(relative);
        checked += 1;
        println!(
            "  {name}[{index}] {symbol}: ours {ours:.6} ppm  PySCF {reference:.6}  相对差 {relative:.2e}"
        );
    }
    println!("checked {checked} nuclei, worst relative difference {worst:.2e}");
    assert_eq!(
        checked, 14,
        "expected the full reference set (3+5+6 nuclei)"
    );
    assert!(
        worst < 1.0e-6,
        "diamagnetic shielding differs from the PySCF reference by {worst:.3e}"
    );
    assert_eq!(sources.len(), 1, "每行都该带同一个来源：{sources:?}");
}

#[test]
fn rs04_diamagnetic_shielding_is_rotation_invariant() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = reference_molecule("water");
    let (basis, density, energy) = solve_density(&registry, &molecule);
    let base = diamagnetic_shielding_ppm(&basis, &molecule, &density).unwrap();

    let r = rotation(std::f64::consts::PI / 6.0); // 30°
    let turned = rotate(&molecule, &r);
    let (basis2, density2, energy2) = solve_density(&registry, &turned);
    let moved = diamagnetic_shielding_ppm(&basis2, &turned, &density2).unwrap();

    let mut worst = 0.0_f64;
    for (a, b) in base.iter().zip(&moved) {
        worst = worst.max((a - b).abs() / a.abs());
    }
    println!("water energy {energy:.10} vs rotated {energy2:.10}");
    println!(
        "water: {:?} vs rotated {:?}, worst relative {worst:.2e}",
        base, moved
    );
    assert!(
        worst < 1.0e-8,
        "sigma_dia should be rotation invariant, got {worst:.3e}"
    );
}

#[test]
fn rs04_diamagnetic_shielding_reproduces_the_free_atom_value() {
    // 单个 H 原子、一个电子（P = 2 表示两个自旋都占）。密度矩阵直接给，
    // 不走 SCF：这样 σ 只依赖积分与常数。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = Molecule::new(vec![Atom::new("H", [0.0, 0.0, 0.0])], 0, 2);
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule).unwrap();
    assert_eq!(basis.n_functions(), 1, "STO-3G H is one s function");
    let mut density = qwb_linalg::Matrix::zeros(1, 1);
    density.set(0, 0, 2.0);
    let shieldings = diamagnetic_shielding_ppm(&basis, &molecule, &density).unwrap();
    // PySCF 的独立积分：STO-3G 的 ⟨1/r_H⟩ = 1.2266137331（每个电子）。
    let inverse_r = 1.226_613_733_124_f64;
    let expected = 2.0 * inverse_r * 1.0e6 / (3.0 * 137.035_999_084_f64.powi(2));
    println!(
        "H atom: ours {:.9} ppm, expected {expected:.9} ppm",
        shieldings[0]
    );
    let relative = (shieldings[0] - expected).abs() / expected;
    assert!(
        relative < 1.0e-6,
        "free H atom sigma_dia should be {expected:.6} ppm, got {:.6}",
        shieldings[0]
    );
    // 文献锚点：氢原子 1s 精确解（⟨1/r⟩ = 1）→ 2/(3c²) = 35.50 ppm。
    let exact_atom = 2.0 * 1.0e6 / (3.0 * 137.035_999_084_f64.powi(2));
    println!("exact H atom (⟨1/r⟩ = 1): {exact_atom:.4} ppm");
    assert!(
        (exact_atom - 35.5025).abs() < 0.01,
        "textbook hydrogen diamagnetic shielding is 35.50 ppm, got {exact_atom:.4}"
    );
}
