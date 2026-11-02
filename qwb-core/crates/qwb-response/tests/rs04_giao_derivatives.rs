//! RS-04（GIAO）：**逐元素**验收 GIAO 的两个一阶导数矩阵 —— 重叠导数与单电子核心导数。
//!
//! 参考值来自 `scripts/pyscf_giao_nmr_reference.py`（PySCF 的 libcint GIAO 积分 +
//! `pyscf/prop/nmr/rhf.py` 的公式），水分子、STO-3G、收敛的 RHF 态：
//!
//! * `overlap_derivative`：`∂S/∂B_k = i·S1_k` —— 钉
//!   [`qwb_integrals::giao::giao_overlap_derivative`] 的 `i/2` 约定（那个函数返回 `C_k`，
//!   `∂S/∂B = (i/2)C_k`）；
//! * `core_field_derivative`：`∂H/∂B_k = i·X_k` —— 钉
//!   [`qwb_integrals::giao::giao_core_field_derivative`]（London 相位 + 原点在 `R_ν` 的
//!   Zeeman 项）。
//!
//! # 判据 1e-6 的来历（不是"放宽"，是量出来的）
//!
//! 两边几何用同一个 Å 数，但 **Å→Bohr 常数不同**：`qwb` 用 `1.889726089838572`
//! （CODATA 2018），PySCF 2.14 用 `1.8897261245650618`，相对差 **1.84e-8**。于是所有积分
//! 矩阵都有同量级的相对差：实测 `∂S/∂B` 绝对差 8.1e-9（量级 3.2e-1）、`∂H/∂B` 绝对差
//! 3.9e-8（量级 1.6e0）—— 恰好是 1.84e-8 相对。判据取 1e-6（相对 ~1e-6）把这条常数差排除
//! 在外，而比"公式写错"（那会是 1e-2 以上）小四个数量级。
//!
//! `two_electron_field_derivative` 那一列也在 CSV 里，但**本轮没有做到能对它** ——
//! 见 `qwb_integrals::giao_2e` 的模块文档（交付边界）。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/giao_fock_derivative.csv")
}

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

/// 参考 CSV 里的量：`quantity -> [k][row][col]`。
fn references() -> BTreeMap<String, Vec<Vec<Vec<f64>>>> {
    let text = std::fs::read_to_string(reference_path()).expect("GIAO 参考 CSV");
    let mut out: BTreeMap<String, Vec<Vec<Vec<f64>>>> = BTreeMap::new();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split(',').collect();
    let position = |name: &str| {
        header
            .iter()
            .position(|item| *item == name)
            .unwrap_or_else(|| panic!("column {name}"))
    };
    let (q_at, k_at, r_at, c_at, v_at) = (
        position("quantity"),
        position("k"),
        position("row"),
        position("column"),
        position("value"),
    );
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let quantity = fields[q_at].to_string();
        let k: usize = fields[k_at].parse().expect("k");
        let row: usize = fields[r_at].parse().expect("row");
        let column: usize = fields[c_at].parse().expect("column");
        let value: f64 = fields[v_at].parse().expect("value");
        let entry = out.entry(quantity).or_default();
        while entry.len() <= k {
            entry.push(Vec::new());
        }
        while entry[k].len() <= row {
            entry[k].push(Vec::new());
        }
        while entry[k][row].len() <= column {
            entry[k][row].push(0.0);
        }
        entry[k][row][column] = value;
    }
    out
}

fn converged_state() -> (qwb_basis::InternalBasis, Molecule) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    (basis, molecule)
}

#[allow(clippy::needless_range_loop)] // k/row/column 同时用作下标与打印
fn compare(name: &str, ours: &[qwb_linalg::Matrix], expected: &[Vec<Vec<f64>>]) {
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    let mut where_worst = (0usize, 0usize, 0usize);
    for (k, matrix) in ours.iter().enumerate() {
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                let reference = expected[k][row][column];
                let difference = (matrix.get(row, column) - reference).abs();
                if difference > worst {
                    worst = difference;
                    where_worst = (k, row, column);
                }
                scale = scale.max(reference.abs());
            }
        }
    }
    let (k, row, column) = where_worst;
    println!(
        "{name}: max|ours − PySCF| = {worst:.3e}（max|ref| = {scale:.3e}）；\
         worst at k={k} row={row} col={column}: ours {:.12e} vs ref {:.12e}",
        ours[k].get(row, column),
        expected[k][row][column]
    );
    assert!(
        worst < 1.0e-6,
        "{name}: GIAO 导数与 PySCF 参考差 {worst:.3e}（判据 1e-6，来历见模块文档）"
    );
}

/// `∂S/∂B_k = i·S1_k`，`S1 = ½C`（[`qwb_integrals::giao::giao_overlap_derivative`] 返回 `C`）。
#[test]
fn giao_overlap_derivative_matches_the_pyscf_reference() {
    let (basis, molecule) = converged_state();
    let c = qwb_integrals::giao::giao_overlap_derivative(&basis, &molecule).expect("overlap");
    let ours: Vec<qwb_linalg::Matrix> = c
        .iter()
        .map(|matrix| {
            let mut half = qwb_linalg::Matrix::zeros(matrix.nrows(), matrix.ncols());
            for row in 0..matrix.nrows() {
                for column in 0..matrix.ncols() {
                    half[(row, column)] = 0.5 * matrix.get(row, column);
                }
            }
            half
        })
        .collect();
    compare("∂S/∂B", &ours, &references()["overlap_derivative"]);
}

/// 单电子 GIAO 核心导数 `∂H/∂B_k = i·X_k`。
#[test]
fn giao_core_field_derivative_matches_the_pyscf_reference() {
    let (basis, molecule) = converged_state();
    let ours = qwb_integrals::giao::giao_core_field_derivative(&basis, &molecule)
        .expect("core field derivative");
    compare("∂H/∂B", &ours, &references()["core_field_derivative"]);
}
