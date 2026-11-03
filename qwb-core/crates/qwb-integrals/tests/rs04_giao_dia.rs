//! RS-04 抗磁（dia）项的两块 GIAO 积分：libcint 的 `int1e_giao_a11part` 与 `int1e_a01gp`。
//!
//! 判据分三层：
//!
//! 1. **逐元素**与 libcint 对照（`tests/reference/giao_dia.csv`，由
//!    `scripts/pyscf_giao_dia_reference.py` 生成）：水 × {STO-3G, LANL2DZ} × 三个核 × 9 个
//!    (α,β) 块 × 两个算子 = 11772 个元素。这个数是 assert 死的（`expect` 出来的元素个数）。
//! 2. **结构性质**（不需要参考文件也能测，所以换一种实现写错时会被抓到）：
//!    * `a11part` 的第二因子是 `(r−R_ν)`（ket 中心），所以它**不是** α↔β 对称的：
//!      实测 `max|A_αβ − A_βα| / max|A| = 0.14`（O，STO-3G）～`0.36`（H，LANL2DZ）。
//!      一个「看起来对称」的错误实现（把第二因子写成 `(r−R_A)`）会在这里给出 0。
//!    * `a01gp` 带 `(R_ν−R_μ)` 因子，于是**每个块的对角元精确为零**（参考文件里 540 个
//!      对角元素全是 `0.0`）。
//!    * `a11part` 不含绝对坐标 ⇒ 整体平移不变；`a01gp` 含绝对 `r` ⇒ 平移会变。
//!      这一条把「(r−R_A) 还是 r」这个最容易搞错的约定钉住了。
//! 3. 与参考文件无关的非平凡性：矩阵最大元是 O(1)（`max|a11| = 1.27`），
//!    而不是一个零矩阵。
//!
//! 容差为什么按基组分开：STO-3G 那一列的残差（6.1e-11）本来就来自**基组数据交接**，不来自
//! 积分本身 —— 同一对基函数在两边算出的重叠矩阵差 7.0e-11（`qwb-data` 的 STO-3G 系数是
//! 8 位表），所以那一列只能给 1e-9。LANL2DZ 的交接差 6.7e-16，于是两个算子给 3e-14。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::giao_dia::{giao_a01_integrals, giao_a11_integrals};
use qwb_linalg::Matrix;
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// 与 `scripts/pyscf_giao_dia_reference.py` 里的 `GEOMETRY` 是同一份（Å）。
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

fn basis(name: &str) -> InternalBasis {
    basis_for(name, &water())
}

fn basis_for(name: &str, molecule: &Molecule) -> InternalBasis {
    let registry = Registry::open(data_root()).expect("qwb-data");
    InternalBasis::build(registry.get_basis(name).unwrap(), molecule).unwrap()
}

/// 参考文件里一个 (basis, nucleus, operator) 的 9 个块。
fn read_reference(name: &str, nucleus: usize, operator: &str, nao: usize) -> [[Matrix; 3]; 3] {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/giao_dia.csv");
    let text = std::fs::read_to_string(&path).expect("giao_dia.csv");
    let mut out: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(nao, nao)));
    let mut seen = 0usize;
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.trim().split(',').collect();
        if fields[1] != name || fields[3] != operator {
            continue;
        }
        let index: usize = fields[2].trim().parse().expect("nucleus index");
        if index != nucleus {
            continue;
        }
        let alpha: usize = fields[4].trim().parse().expect("alpha");
        let beta: usize = fields[5].trim().parse().expect("beta");
        let row: usize = fields[6].trim().parse().expect("row");
        let column: usize = fields[7].trim().parse().expect("column");
        let value: f64 = fields[8].trim().parse().expect("value");
        out[alpha][beta][(row, column)] = value;
        seen += 1;
    }
    assert_eq!(
        seen,
        9 * nao * nao,
        "{name}/{operator}/nucleus {nucleus}: 只读到 {seen} 个参考元素"
    );
    out
}

/// `(最大绝对偏差, 参考矩阵最大元, 元素个数)`。
fn deviation(
    ours: &[[Matrix; 3]; 3],
    reference: &[[Matrix; 3]; 3],
    nao: usize,
) -> (f64, f64, usize) {
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    let mut count = 0usize;
    for alpha in 0..3 {
        for beta in 0..3 {
            for row in 0..nao {
                for column in 0..nao {
                    let expected = reference[alpha][beta][(row, column)];
                    scale = scale.max(expected.abs());
                    worst = worst.max((ours[alpha][beta][(row, column)] - expected).abs());
                    count += 1;
                }
            }
        }
    }
    (worst, scale, count)
}

#[test]
fn rs04_giao_dia_matches_the_cint_reference() {
    let molecule = water();
    // (基组, 判据, 每个 (核, 算子) 的元素数)
    let cases = [("sto-3g", 1.0e-9_f64), ("lanl2dz", 3.0e-14_f64)];
    let mut total = 0usize;
    for (name, tolerance) in cases {
        let basis = basis(name);
        let nao = basis.n_functions();
        for nucleus in 0..3 {
            let nucleus_bohr = molecule.atoms[nucleus].position_bohr();
            let ours = [
                (
                    "a11part",
                    giao_a11_integrals(&basis, &molecule, nucleus_bohr).expect("a11"),
                ),
                (
                    "a01gp",
                    giao_a01_integrals(&basis, &molecule, nucleus_bohr).expect("a01"),
                ),
            ];
            for (operator, matrices) in ours {
                let reference = read_reference(name, nucleus, operator, nao);
                let (worst, scale, count) = deviation(&matrices, &reference, nao);
                total += count;
                println!(
                    "{name}/{operator}/nucleus {nucleus} ({}): max|Δ| = {worst:.3e}, \
                     max|ref| = {scale:.3e}, {count} elements",
                    molecule.atoms[nucleus].symbol
                );
                assert!(
                    scale > 1.0e-2,
                    "{name}/{operator}: 参考矩阵太小（max|ref| = {scale:.3e}），尺子是空的"
                );
                assert!(
                    worst < tolerance,
                    "{name}/{operator}/nucleus {nucleus}: 与 libcint 差 {worst:.3e}\
                     （max|ref| = {scale:.3e}，判据 {tolerance:e}）"
                );
            }
        }
    }
    // 水 × 两个基组 × 3 核 × 9 块 × 两个算子：(7² + 13²) × 9 × 3 × 2 = 11772。
    assert_eq!(total, (49 + 169) * 9 * 3 * 2, "参考元素个数不对");
    println!("共对照 {total} 个元素");
}

#[test]
fn rs04_giao_dia_keeps_its_measured_structure() {
    let molecule = water();
    for name in ["sto-3g", "lanl2dz"] {
        let basis = basis(name);
        let nao = basis.n_functions();
        for nucleus in 0..3 {
            let nucleus_bohr = molecule.atoms[nucleus].position_bohr();
            let a11 = giao_a11_integrals(&basis, &molecule, nucleus_bohr).expect("a11");
            let a01 = giao_a01_integrals(&basis, &molecule, nucleus_bohr).expect("a01");

            let mut scale = 0.0_f64;
            let mut asymmetry = 0.0_f64;
            let mut worst_antisymmetry = 0.0_f64;
            let mut diagonal = 0.0_f64;
            for alpha in 0..3 {
                for beta in 0..3 {
                    for row in 0..nao {
                        for column in 0..nao {
                            let value = a11[alpha][beta][(row, column)];
                            scale = scale.max(value.abs());
                            asymmetry =
                                asymmetry.max((value - a11[beta][alpha][(row, column)]).abs());
                            worst_antisymmetry = worst_antisymmetry
                                .max((value + a11[beta][alpha][(row, column)]).abs());
                        }
                        diagonal = diagonal.max(a01[alpha][beta][(row, row)].abs());
                    }
                }
            }
            println!(
                "{name}/nucleus {nucleus}: max|a11| = {scale:.9}, \
                 max|a11_ab - a11_ba| = {asymmetry:.9} \
                 ({:.1}% of scale), max|a11_ab + a11_ba| = {worst_antisymmetry:.3e}, \
                 max|a01 diagonal| = {diagonal:.3e}",
                100.0 * asymmetry / scale
            );
            // 非空性：O 核上 max|a11| ≈ 1.27，H 核上 ≈ 0.2。
            assert!(
                scale > 0.2,
                "{name}: max|a11| 只有 {scale:.3e}，检查是不是算成了零矩阵"
            );
            // a11 用的是 `(r−R_ν)`，所以 α↔β **不**对称 —— 一个对称化的错误实现会给出 0。
            assert!(
                asymmetry > 0.1 * scale,
                "{name}/nucleus {nucleus}: max|a11_ab − a11_ba| = {asymmetry:.3e} 太小\
                 （max|a11| = {scale:.3e}）；第二因子应当是 (r−R_ν)，不是 (r−R_A)"
            );
            // a01 带 (R_ν − R_μ)，同一 AO 上（行 = 列）恒为零。
            assert!(
                diagonal < 1.0e-25,
                "{name}/nucleus {nucleus}: a01 的对角元应当是精确零，实测 {diagonal:.3e}"
            );
        }
    }
}

#[test]
fn rs04_giao_a01_shifts_with_the_origin_but_a11_does_not() {
    let molecule = water();
    let shifted = Molecule::new(
        molecule
            .atoms
            .iter()
            .map(|atom| {
                let p = atom.position_angstrom;
                Atom::new(atom.symbol.clone(), [p[0] + 0.3, p[1] - 0.25, p[2] + 0.42])
            })
            .collect(),
        molecule.charge,
        molecule.multiplicity,
    );
    let basis_before = basis_for("sto-3g", &molecule);
    let basis_after = basis_for("sto-3g", &shifted);
    let nao = basis_before.n_functions();
    let before = molecule.atoms[0].position_bohr();
    let after = shifted.atoms[0].position_bohr();
    let a11_before = giao_a11_integrals(&basis_before, &molecule, before).expect("a11");
    let a11_after = giao_a11_integrals(&basis_after, &shifted, after).expect("a11");
    let a01_before = giao_a01_integrals(&basis_before, &molecule, before).expect("a01");
    let a01_after = giao_a01_integrals(&basis_after, &shifted, after).expect("a01");
    let mut a11_change = 0.0_f64;
    let mut a01_change = 0.0_f64;
    let mut a01_scale = 0.0_f64;
    for alpha in 0..3 {
        for beta in 0..3 {
            for row in 0..nao {
                for column in 0..nao {
                    a11_change = a11_change.max(
                        (a11_before[alpha][beta][(row, column)]
                            - a11_after[alpha][beta][(row, column)])
                            .abs(),
                    );
                    a01_scale = a01_scale.max(a01_before[alpha][beta][(row, column)].abs());
                    a01_change = a01_change.max(
                        (a01_before[alpha][beta][(row, column)]
                            - a01_after[alpha][beta][(row, column)])
                            .abs(),
                    );
                }
            }
        }
    }
    println!(
        "平移 (0.3, −0.25, 0.42) Å：max|Δa11| = {a11_change:.3e}，\
         max|Δa01| = {a01_change:.3e}（max|a01| = {a01_scale:.3e}）"
    );
    // a11 只用相对坐标 (r−R_A)、(r−R_ν) ⇒ 平移不变。
    assert!(
        a11_change < 1.0e-12,
        "a11part 应当对整体平移不变，实测 max|Δ| = {a11_change:.3e}"
    );
    // a01 含绝对 r ⇒ 平移必须让它变（实测 4.1e-2，约 23% 的 max|a01|）。
    assert!(
        a01_change > 0.05 * a01_scale,
        "a01gp 应当随绝对原点变化，实测 max|Δ| = {a01_change:.3e}\
         （max|a01| = {a01_scale:.3e}）；若为 0，说明把 r 写成了 (r−R_A)"
    );
}
