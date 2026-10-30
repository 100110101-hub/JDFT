//! RS-04 顺磁项的第一块：**角动量积分** `⟨μ| (r−O) × ∇ |ν⟩`（`L = −i N`）。
//!
//! 两条检查：
//! 1. 与 PySCF 的 `int1e_irp`（`⟨μ| i r_β p_γ |ν⟩` 的 9 个分量）逐位对照，
//!    两个规范原点各 3×7×7 个元素（水、STO-3G）。基准由
//!    `scripts/pyscf_angmom_reference.py` 生成。
//! 2. **反对称性**（精确）：`∫φ_μ (r×∇) φ_ν = −∫φ_ν (r×∇) φ_μ`（分部积分，`∇×r = 0`）。
//!
//! 注意：角动量代数 `[N_x, N_y] = −N_z` 在**有限基**下只是近似的（`L|ν⟩` 会漏到基外），
//! 所以它只用来确定 `int1e_irp` 的分量顺序（残差 0.117 vs 2.117，见脚本），
//! 不能当判据 —— 判据是 PySCF 的逐元素对照。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::angmom::angular_momentum_integrals;
use qwb_registry::Registry;

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

fn basis() -> InternalBasis {
    let registry = Registry::open(data_root()).expect("qwb-data");
    InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &water()).unwrap()
}

/// 核电荷重心（水：O 权重 8）。
fn charge_centroid(molecule: &Molecule) -> [f64; 3] {
    let weights = [8.0, 1.0, 1.0];
    let mut total = 0.0;
    let mut sum = [0.0; 3];
    for (atom, weight) in molecule.atoms.iter().zip(weights) {
        let position = atom.position_bohr();
        for axis in 0..3 {
            sum[axis] += weight * position[axis];
        }
        total += weight;
    }
    sum.map(|value| value / total)
}

#[test]
fn rs04_angular_momentum_integrals_match_the_pyscf_reference() {
    let molecule = water();
    let basis = basis();
    let origins = [
        ("O_nucleus", molecule.atoms[0].position_bohr()),
        ("charge_centroid", charge_centroid(&molecule)),
    ];
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/angmom.csv");
    let text = std::fs::read_to_string(&path).expect("angmom.csv");
    let mut checked = 0usize;
    let mut worst = 0.0_f64;
    for (label, origin) in origins {
        let matrices =
            angular_momentum_integrals(&basis, &molecule, origin).expect("angular momentum");
        let mut label_worst = 0.0_f64;
        for line in text.lines().skip(1) {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields[1].trim() != label {
                continue;
            }
            let component: usize = fields[2].trim().parse().expect("component");
            let row: usize = fields[3].trim().parse().expect("row");
            let column: usize = fields[4].trim().parse().expect("column");
            let reference: f64 = fields[5].trim().parse().expect("value");
            let ours = matrices[component][(row, column)];
            let difference = (ours - reference).abs();
            label_worst = label_worst.max(difference);
            checked += 1;
        }
        println!("{label}: worst absolute difference {label_worst:.3e}");
        worst = worst.max(label_worst);
    }
    println!("checked {checked} elements, worst absolute difference {worst:.3e}");
    assert_eq!(checked, 2 * 3 * 7 * 7, "expected the full reference set");
    // 实测 2.0e-11：两侧的收缩归一化常数是各自独立按双精度算的（基组指数同一份
    // qwb-data 数据，已经排除了 PySCF 内建表只有 8 位有效数字这个干扰）。
    assert!(
        worst < 1.0e-9,
        "angular momentum integrals differ from PySCF by {worst:.3e}"
    );
}

#[test]
fn rs04_angular_momentum_integrals_are_antisymmetric() {
    let molecule = water();
    let basis = basis();
    for (label, origin) in [
        ("origin", [0.0, 0.0, 0.0]),
        ("O nucleus", molecule.atoms[0].position_bohr()),
        ("off-centre", [0.7, -0.35, 1.1]),
    ] {
        let matrices = angular_momentum_integrals(&basis, &molecule, origin).unwrap();
        let n = basis.n_functions();
        let mut worst = 0.0_f64;
        for matrix in &matrices {
            for row in 0..n {
                for column in 0..n {
                    worst = worst.max((matrix[(row, column)] + matrix[(column, row)]).abs());
                }
            }
        }
        println!("{label}: worst antisymmetry violation {worst:.3e}");
        assert!(
            worst < 1.0e-14,
            "N + N^T should vanish at {label}, got {worst:.3e}"
        );
    }
}
