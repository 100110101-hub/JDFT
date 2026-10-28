//! QWB 性质（S-06，v2.1 新增的 `qwb-props`）：Mulliken / Löwdin 布居。
//!
//! # 定义与为什么它俩的和必须精确等于电子数
//!
//! Mulliken：`q_μ = (P S)_μμ`，按原子求和得该原子的电子布居。
//! Löwdin：先用对称正交化基 `S^{1/2}` 变换密度，`q_μ = (S^{1/2} P S^{1/2})_μμ`。
//!
//! 两者都满足 `Σ_μ q_μ = Tr(PS) = N`（Mulliken 直接是迹；Löwdin 因为
//! `Tr(S^{1/2}PS^{1/2}) = Tr(PS)`）。所以「各自求和 == 总电荷」是**恒等式级别**的
//! 校验，任何矩阵装配错误都会被它抓到——这也是 S-06 的验收判据之一。
//!
//! 注意：Mulliken 布居对基组敏感（甚至可能为负、可能 > 2），Löwdin 更稳；
//! 两者数值不同但**和相同**。

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod cavity;

use qwb_basis::InternalBasis;
use qwb_engine_api::Molecule;
use qwb_linalg::{LinalgError, Matrix, matmul, symmetric_eigen};

use qwb_registry::elements::atomic_number;

/// 一个分子的布居分析结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Populations {
    /// 每个 AO 的 Mulliken 布居。
    pub mulliken_ao: Vec<f64>,
    /// 每个原子的 Mulliken 电子布居（按基函数所属原子求和）。
    pub mulliken_atoms: Vec<f64>,
    /// 每个 AO 的 Löwdin 布居。
    pub lowdin_ao: Vec<f64>,
    /// 每个原子的 Löwdin 电子布居。
    pub lowdin_atoms: Vec<f64>,
    /// 每个原子的 Mulliken 净电荷 `Z − q`。
    pub mulliken_charges: Vec<f64>,
    /// 每个原子的 Löwdin 净电荷 `Z − q`。
    pub lowdin_charges: Vec<f64>,
}

impl Populations {
    /// Mulliken 布居总和（应等于电子数）。
    #[must_use]
    pub fn mulliken_total(&self) -> f64 {
        self.mulliken_ao.iter().sum()
    }

    /// Löwdin 布居总和（应等于电子数）。
    #[must_use]
    pub fn lowdin_total(&self) -> f64 {
        self.lowdin_ao.iter().sum()
    }

    /// 净电荷总和（应等于分子总电荷）。
    #[must_use]
    pub fn charge_total(&self) -> f64 {
        self.mulliken_charges.iter().sum()
    }
}

/// 把 AO 布居按原子求和。
fn per_atom(basis: &InternalBasis, ao_values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; basis.n_atoms()];
    let mut offset = 0;
    for (atom_index, n) in (0..basis.n_atoms()).map(|a| (a, basis.n_functions_of_atom(a))) {
        out[atom_index] = ao_values[offset..offset + n].iter().sum();
        offset += n;
    }
    out
}

/// 对称正交化矩阵 `S^{1/2} = U diag(√λ) Uᵀ`。
///
/// # Errors
///
/// 重叠矩阵不是方阵/维数不匹配，或特征分解失败。
fn sqrt_overlap(overlap: &Matrix) -> Result<Matrix, LinalgError> {
    let n = overlap.nrows();
    let (values, vectors) = symmetric_eigen(overlap)?;
    let mut root = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vectors[(i, k)] * values.get(k).max(0.0).sqrt() * vectors[(j, k)];
            }
            root[(i, j)] = sum;
        }
    }
    Ok(root)
}

/// 做一次 Mulliken + Löwdin 布居分析。
///
/// # Errors
///
/// 形状不匹配（`density` 与 `basis` 维数不同）或 `S^{1/2}` 计算失败。
pub fn populations(
    basis: &InternalBasis,
    molecule: &Molecule,
    overlap: &Matrix,
    density: &Matrix,
) -> Result<Populations, LinalgError> {
    let n = basis.n_functions();
    if overlap.nrows() != n || density.nrows() != n {
        return Err(LinalgError::Dimension {
            context: "populations: basis/overlap/density size mismatch",
            left_rows: density.nrows(),
            left_cols: density.ncols(),
            right_rows: n,
            right_cols: n,
        });
    }

    // Mulliken: (P S) 的对角元
    let ps = matmul(density, overlap)?;
    let mut mulliken_ao = vec![0.0; n];
    for i in 0..n {
        mulliken_ao[i] = ps[(i, i)];
    }

    // Löwdin: S^{1/2} P S^{1/2} 的对角元
    let root = sqrt_overlap(overlap)?;
    let left = matmul(&root, density)?;
    let transformed = matmul(&left, &root)?;
    let mut lowdin_ao = vec![0.0; n];
    for i in 0..n {
        lowdin_ao[i] = transformed[(i, i)];
    }

    let mulliken_atoms = per_atom(basis, &mulliken_ao);
    let lowdin_atoms = per_atom(basis, &lowdin_ao);

    let mut mulliken_charges = Vec::with_capacity(molecule.atoms.len());
    let mut lowdin_charges = Vec::with_capacity(molecule.atoms.len());
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let z = f64::from(atomic_number(&atom.symbol).unwrap_or(0));
        mulliken_charges.push(z - mulliken_atoms[index]);
        lowdin_charges.push(z - lowdin_atoms[index]);
    }

    Ok(Populations {
        mulliken_ao,
        mulliken_atoms,
        lowdin_ao,
        lowdin_atoms,
        mulliken_charges,
        lowdin_charges,
    })
}

/// 偶极矩，单位 Debye（原子单位 → Debye 用 2.5417466320311）。
///
/// `μ = Σ_A Z_A R_A − Tr(P · D)`，其中 `D` 是偶极积分矩阵（原子单位 `e·bohr`）。
/// 对**中性**分子，`μ` 必须与原点无关——这是一条很强的自检，见测试。
///
/// # Errors
///
/// 偶极积分计算失败（球谐变换未实现等）。
pub fn dipole_debye(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
) -> Result<[f64; 3], qwb_engine_api::EngineError> {
    let dipoles = qwb_integrals::dipole::dipole_integrals(basis, molecule)?;
    let n = basis.n_functions();
    let mut out = [0.0_f64; 3];
    for axis in 0..3 {
        let mut electronic = 0.0;
        for i in 0..n {
            for j in 0..n {
                electronic += density[(i, j)] * dipoles[axis][(i, j)];
            }
        }
        let mut nuclear = 0.0;
        for atom in &molecule.atoms {
            let z = f64::from(atomic_number(&atom.symbol).unwrap_or(0));
            nuclear += z * atom.position_bohr()[axis];
        }
        out[axis] = (nuclear - electronic) * 2.541_746_632_031_1;
    }
    Ok(out)
}
