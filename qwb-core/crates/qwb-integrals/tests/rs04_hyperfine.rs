//! RS-04 顺磁项的第二块：**轨道超精细** `⟨μ| ((r−R_A) × ∇)_α / |r−R_A|³ |ν⟩`
//! （`L_A/r_A³ = −i ×` 这一层）。
//!
//! 判据：
//! 1. 与 cint 的 `int1e_prinvxp`（`ε_αβγ⟨∂_βμ|1/r|∂_γν⟩` 的三个分量）逐元素对照，
//!    O 核与 H 核各 3×7×7 个元素（水、STO-3G）。基准由
//!    `scripts/pyscf_hyperfine_reference.py` 生成；两侧用的是同一份 `qwb-data` 指数。
//! 2. **反对称性**（精确）：算符 `L_A/r_A³` 厄米且实基下纯虚 ⇒ 实矩阵满足 `M = −Mᵀ`。
//! 3. **符号关系**：本实现的 `M_α = −ε_αβγ⟨∂_βμ|1/r_A|∂_γν⟩`，因此应当与
//!    `int1e_prinvxp` 差一个整体符号 —— 测试会同时试 `+`/`−` 并断言其中一个是
//!    1e-9 量级、另一个是 O(1)，这样符号约定被钉住而不是靠猜。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::angmom::hyperfine_integrals;
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

#[test]
fn rs04_hyperfine_integrals_match_the_cint_reference() {
    let molecule = water();
    let basis = basis();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/hyperfine.csv");
    let text = std::fs::read_to_string(&path).expect("hyperfine.csv");
    let mut checked = 0usize;
    let mut worst_same = 0.0_f64;
    let mut worst_flipped = 0.0_f64;
    for (nucleus_index, label) in [(0_usize, "O0"), (1_usize, "H1")] {
        let matrices = hyperfine_integrals(
            &basis,
            &molecule,
            molecule.atoms[nucleus_index].position_bohr(),
        )
        .expect("hyperfine");
        let mut same = 0.0_f64;
        let mut flipped = 0.0_f64;
        for line in text.lines().skip(1) {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            let index: usize = fields[0].trim().parse().expect("nucleus index");
            if index != nucleus_index {
                continue;
            }
            let component: usize = fields[1].trim().parse().expect("component");
            let row: usize = fields[2].trim().parse().expect("row");
            let column: usize = fields[3].trim().parse().expect("column");
            let reference: f64 = fields[4].trim().parse().expect("value");
            let ours = matrices[component][(row, column)];
            same = same.max((ours - reference).abs());
            flipped = flipped.max((ours + reference).abs());
            checked += 1;
        }
        println!("{label}: |ours - cint| max {same:.3e}, |ours + cint| max {flipped:.3e}");
        worst_same = worst_same.max(same);
        worst_flipped = worst_flipped.max(flipped);
    }
    println!(
        "checked {checked} elements: same-sign worst {worst_same:.3e}, flipped worst {worst_flipped:.3e}"
    );
    assert_eq!(checked, 2 * 3 * 7 * 7, "expected the full reference set");
    // 只允许一个符号关系成立：M = −⟨∂_βμ|1/r|∂_γν⟩·ε
    assert!(
        worst_flipped < 1.0e-9,
        "hyperfine integrals should equal minus cint's int1e_prinvxp (worst {worst_flipped:.3e})"
    );
    assert!(
        worst_same > 1.0,
        "the same-sign match should fail loudly if the convention changes (got {worst_same:.3e})"
    );
}

#[test]
fn rs04_hyperfine_integrals_are_antisymmetric() {
    let molecule = water();
    let basis = basis();
    for (label, nucleus) in [
        ("O nucleus", molecule.atoms[0].position_bohr()),
        ("H nucleus", molecule.atoms[1].position_bohr()),
        ("off-nucleus", [0.4, -0.25, 0.9]),
    ] {
        let matrices = hyperfine_integrals(&basis, &molecule, nucleus).unwrap();
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
            worst < 1.0e-12,
            "M + M^T should vanish at {label}, got {worst:.3e}"
        );
    }
}
