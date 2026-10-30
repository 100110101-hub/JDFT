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
pub mod cosmo;

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

/// **抗磁（diamagnetic）NMR 屏蔽**，单位 ppm，每个原子一个。
///
/// Ramsey 的抗磁项，规范取「核在原点」的共原（common-origin）形式：
///
/// ```text
/// σ_dia(A) = 1/(3c²) · Σ_μν P_μν ⟨μ| 1/|r − R_A| |ν⟩
/// ```
///
/// 来源是 `A_ext · A_μ` 交叉项 `A_ext = ½ B×(r−R_O)`、`A_μ = μ×(r−R_A)/|r−R_A|³`：
///
/// ```text
/// σ^dia_αβ(A) = 1/(2c²) Σ P ⟨ (r_O·r_A)δ_αβ − r_Oα r_Aβ | / r_A³ ⟩,  r_O = r − R_O, r_A = r − R_A
/// ```
///
/// 取迹、并把规范原点放在核上（`R_O = R_A`）就得到上面的 `⟨1/r_A⟩` 形式
/// （`tr = 2/r_A`，`σ_iso = ⅓·tr`）。**锚点**：氢原子（1s 精确解，`⟨1/r⟩ = 1`、
/// 两个电子）→ `2/(3c²) = 35.50 ppm`，正是教科书的氢原子抗磁屏蔽。
///
/// 积分用 [`qwb_integrals::one_electron::one_electron_with_charges`] 的**单位电荷**
/// 技巧：对目标核那一列放 `Z = 1`、其余放 0，得到的核吸引矩阵就是该核的
/// `−⟨μ|1/r_A|ν⟩`。这样不必另写一套积分（同一段经过验证的球谐变换）。
///
/// **范围**：只做抗磁项。顺磁项（`⟨μ|L_A/r_A³|ν⟩` + CPHF 响应）还没做 —— 注意抗磁项
/// 本身就是几百 ppm 的量级（水 ¹H 在这里是 103 ppm，因为氧的 8 个电子都在
/// `1/r_H` 里贡献了 4.6），**不能**当化学位移用；`docs/choices.md` 七十九 记了这次
/// 公式订正（先写成 `⟨|r−R|²⟩`，那是错的）。
///
/// # Errors
///
/// 积分计算失败（球谐变换未实现等）。
pub fn diamagnetic_shielding_ppm(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
) -> Result<Vec<f64>, qwb_engine_api::EngineError> {
    /// 原子单位下的光速（CODATA 2018）。
    const SPEED_OF_LIGHT: f64 = 137.035_999_084;
    /// 1/(3c²) 换成 ppm 的因子：每个 `⟨1/r⟩` 单位（每个电子）17.7512 ppm。
    const PPM_PER_UNIT_INVERSE_R: f64 = 1.0e6 / (3.0 * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
    let n_atoms = molecule.n_atoms();
    let mut shielding = Vec::with_capacity(n_atoms);
    for target in 0..n_atoms {
        // 单位电荷技巧：只保留目标核的 1/|r − R_A| 吸引。
        let mut charges = vec![0.0; n_atoms];
        charges[target] = 1.0;
        let attraction =
            qwb_integrals::one_electron::one_electron_with_charges(basis, molecule, &charges)?
                .nuclear;
        // `nuclear` 带吸引符号（负），所以 Σ P·V 是 −⟨1/r_A⟩。
        let n = basis.n_functions();
        let mut expectation = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                expectation -= density[(mu, nu)] * attraction[(mu, nu)];
            }
        }
        shielding.push(expectation * PPM_PER_UNIT_INVERSE_R);
    }
    Ok(shielding)
}

/// 相对某个原点的**无迹四极矩** `Θ_ij = ½ Σ_a q_a (3 r_i r_j − r² δ_ij)`，单位 Debye·Å。
///
/// 二阶矩用「相对原点」的形式从**固定积分**拼出来（积分本身只算一次）：
///
/// ```text
/// ∫φφ (R_i − O_i)(R_j − O_j) = S^(2)_ij − O_i D_j − O_j D_i + O_i O_j S
/// ```
///
/// 然后按电荷加权求和（核 +Z、电子由 `P` 收缩并带负号），再做无迹组合
/// `Θ_ij = ½[3 M_ij − tr(M) δ_ij]`（这样迹恒为零）。
///
/// **原点敏感性**：`Θ` 只在中性且偶极矩为零时才与原点无关；一般情形满足
/// `Θ(O+R) − Θ(O) = −½[3(R_i μ_j + R_j μ_i) − 2(R·μ)δ_ij]`。两条性质都在测试里钉住。
///
/// # Errors
///
/// 积分计算失败（球谐变换未实现等）。
pub fn quadrupole_traceless_debye_angstrom(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
    origin_angstrom: [f64; 3],
) -> Result<[[f64; 3]; 3], qwb_engine_api::EngineError> {
    const DEBYE_PER_BOHR: f64 = 2.541_746_632_031_1;
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let bohr_per_angstrom = qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
    let moments = qwb_integrals::dipole::second_moment_integrals(basis, molecule)?;
    let dipoles = qwb_integrals::dipole::dipole_integrals(basis, molecule)?;
    let overlap = qwb_integrals::one_electron::one_electron(basis, molecule)?.overlap;
    let n = basis.n_functions();
    let origin = [
        origin_angstrom[0] * bohr_per_angstrom,
        origin_angstrom[1] * bohr_per_angstrom,
        origin_angstrom[2] * bohr_per_angstrom,
    ];
    let components = [(0usize, 0usize), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let mut electronic = [[0.0_f64; 3]; 3];
    for (component, (i, j)) in components.into_iter().enumerate() {
        let mut value = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                let reduced = moments[component][(mu, nu)]
                    - origin[i] * dipoles[j][(mu, nu)]
                    - origin[j] * dipoles[i][(mu, nu)]
                    + origin[i] * origin[j] * overlap[(mu, nu)];
                value += density[(mu, nu)] * reduced;
            }
        }
        electronic[i][j] = value;
        electronic[j][i] = value;
    }
    let mut nuclear = [[0.0_f64; 3]; 3];
    for atom in &molecule.atoms {
        let z = f64::from(atomic_number(&atom.symbol).unwrap_or(0));
        let r = [
            atom.position_bohr()[0] - origin[0],
            atom.position_bohr()[1] - origin[1],
            atom.position_bohr()[2] - origin[2],
        ];
        for i in 0..3 {
            for j in 0..3 {
                nuclear[i][j] += z * r[i] * r[j];
            }
        }
    }
    let mut moment = [[0.0_f64; 3]; 3];
    let mut trace = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            moment[i][j] = nuclear[i][j] - electronic[i][j];
        }
        trace += moment[i][i];
    }
    let mut quadrupole = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let operator = 3.0 * moment[i][j] - if i == j { trace } else { 0.0 };
            quadrupole[i][j] = 0.5 * operator * DEBYE_PER_BOHR * ANGSTROM_PER_BOHR;
        }
    }
    Ok(quadrupole)
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
