//! RS-04 的第一块验收：**抗磁屏蔽**。
//!
//! `σ_dia(A) = 1/(3c²) Σ_μν P_μν ⟨μ| 1/|r − R_A| |ν⟩`（ppm，核在原点，Ramsey 抗磁项）。
//!
//! 三条检查：
//! 1. 与 PySCF 的**独立**实现逐位对照（水 + 甲烷，8 个核，STO-3G）。基准由
//!    `scripts/pyscf_nmr_diamagnetic_reference.py` 生成 —— 那边是 PySCF 自己的
//!    `int1e_rinv`（把分子平移到核在原点后取 `1/|r|`），这边是高斯递推的核吸引矩阵
//!    （单位电荷技巧）。两条不同的积分路径。
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

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    )
}

fn methane() -> Molecule {
    let d = 0.629_0_f64;
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [d, d, d]),
            Atom::new("H", [-d, -d, d]),
            Atom::new("H", [-d, d, -d]),
            Atom::new("H", [d, -d, -d]),
        ],
        0,
        1,
    )
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
    let molecules = [("water", water()), ("methane", methane())];
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/nmr_diamagnetic.csv");
    let text = std::fs::read_to_string(&path).expect("nmr_diamagnetic.csv");
    let mut checked = 0usize;
    let mut worst = 0.0_f64;
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let name = fields[0].trim();
        let index: usize = fields[1].trim().parse().expect("atom index");
        let symbol = fields[2].trim();
        let reference: f64 = fields[4].trim().parse().expect("reference");
        let molecule = molecules
            .iter()
            .find(|(label, _)| *label == name)
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
    assert_eq!(checked, 8, "expected the full reference set");
    assert!(
        worst < 1.0e-6,
        "diamagnetic shielding differs from the PySCF reference by {worst:.3e}"
    );
}

#[test]
fn rs04_diamagnetic_shielding_is_rotation_invariant() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
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
