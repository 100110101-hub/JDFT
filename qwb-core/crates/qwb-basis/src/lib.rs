//! I-01：把 K1 的 [`qwb_registry::basis::BasisSet`] 变成**以原子为中心的内部 AO 基组**。
//!
//! # 职责边界
//!
//! 本 crate 只做"数据 → 内部表示"的转换与**归一化**，不做任何积分。
//! 积分在 `qwb-integrals`（I-02）里，读 [`InternalBasis`]。
//!
//! # 单位
//!
//! 数据文件的指数是 **Bohr⁻²**；分子坐标进来是 **Å**，这里一次性转成 Bohr
//! （§12.2：引擎内部一律原子单位，边界才用 Å）。
//!
//! # 归一化只有一处（§8.1）
//!
//! `BasisSet.normalization` 有两个取值：
//!
//! - `L2`：数据声称每个收缩函数已归一。**这里会真的算一遍自重叠验证**，
//!   偏差超过 [`NORMALIZATION_TOLERANCE`] 就报 `Data` 错 —— 这样"归一声明"不是
//!   信任，而是被检查的。（STO-3G 的 BSE 导出在归一化原函数下 ⟨f|f⟩ = 1 ± 1e-10。）
//! - `None`：数据未归一，这里按 `1/sqrt(⟨f|f⟩)` 缩放一次。
//!
//! 运行时不转换（§8.1）。
//!
//! # 原函数约定
//!
//! 归一化是指"**归一化原函数**（normalized primitive）乘系数再收缩"。这是 BSE / STO-3G
//! 表的惯例，也是 [`primitive_self_overlap`] 的判据。

#![deny(unsafe_code)]
#![deny(missing_docs)]

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::basis::{Angular, BasisSet, Normalization};

/// 归一声明的容差。BSE 的 STO-3G 实测 1 ± 1e-10，1e-6 足够宽松但能抓转录错误。
pub const NORMALIZATION_TOLERANCE: f64 = 1.0e-6;

/// 一个直角坐标分量的角量子数计数 `(lx, ly, lz)`。
pub type CartesianPowers = (u32, u32, u32);

/// 每个壳层贡献的直角坐标分量（顺序固定，保证可复现）。
///
/// l<=1 时球谐与直角坐标函数个数相同（1 个 s、3 个 p），因此第一阶段直接用直角坐标
/// 表示；l>=2 时球谐转换在 X/SCF 轨补（R-06 的显式入口仍在 `BasisSet.angular`）。
#[must_use]
pub fn cartesian_components(l: u32) -> Vec<CartesianPowers> {
    const P1: [CartesianPowers; 3] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)];
    const P2: [CartesianPowers; 6] = [
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    ];
    match l {
        0 => vec![(0, 0, 0)],
        1 => P1.to_vec(),
        2 => P2.to_vec(),
        other => vec![(other, 0, 0)],
    }
}

/// 两个**归一化原函数**（同一中心、同一 `l`）的自重叠。
///
/// 闭式：`(2 sqrt(a b) / (a + b))^(l + 3/2)`。
#[must_use]
pub fn primitive_self_overlap(a: f64, b: f64, l: u32) -> f64 {
    let ratio = 2.0 * (a * b).sqrt() / (a + b);
    ratio.powf(f64::from(l) + 1.5)
}

/// 一个收缩壳层（内部表示）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct InternalShell {
    /// 所属原子序号（`Molecule.atoms` 的下标）。
    pub atom_index: usize,
    /// 角量子数。
    pub l: u32,
    /// 球心，Bohr。
    pub center_bohr: [f64; 3],
    /// 球谐还是直角坐标——**来自数据**（R-06 的唯一入口）。
    pub angular: Angular,
    /// 指数，Bohr⁻²（已排序：大 → 小）。
    pub exponents: Vec<f64>,
    /// **已归一化**的收缩系数（与 `exponents` 同长）。
    pub coefficients: Vec<f64>,
    /// 本壳层第一个 AO 在 [`InternalBasis`] 里的全局下标。
    pub first_function: usize,
}

impl InternalShell {
    /// 本壳层贡献的 AO 个数。
    ///
    /// 球谐时是 `2l+1`；直角坐标时是 `(l+1)(l+2)/2`。两者在 `l <= 1` 时相同
    /// （1 个 s、3 个 p），所以 STO-3G 不受影响；`l >= 2` 才有差别（d：5 vs 6）。
    #[must_use]
    pub fn n_functions(&self) -> usize {
        match self.angular {
            Angular::Spherical => 2 * self.l as usize + 1,
            Angular::Cartesian => cartesian_components(self.l).len(),
            // `Angular` 是 non_exhaustive：不认识就按球谐算，并在积分层显式拒绝 l>=2。
            _ => 2 * self.l as usize + 1,
        }
    }

    /// 第 `component` 个分量的角动量计数。
    #[must_use]
    pub fn component(&self, component: usize) -> CartesianPowers {
        cartesian_components(self.l)[component]
    }
}

/// 内部 AO 基组。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct InternalBasis {
    /// 球谐 / 笛卡尔——**只来自数据**（R-06：唯一入口）。
    pub angular: Angular,
    /// 原子序号 → 该原子上的壳层下标。
    atom_shells: Vec<Vec<usize>>,
    /// 全部壳层（按原子顺序、原子内的壳层顺序）。
    shells: Vec<InternalShell>,
    /// AO 总数。
    n_functions: usize,
}

impl InternalBasis {
    /// 由 K1 基组 + 分子构建内部基组。
    ///
    /// # Errors
    ///
    /// - 基组不覆盖分子里的某个元素 → `Data`；
    /// - 数据声称 `L2` 归一，但实测自重叠偏离 1 超过 [`NORMALIZATION_TOLERANCE`] → `Data`；
    /// - 指数/系数长度不一致或出现非有限值 → `Data`。
    pub fn build(basis: &BasisSet, molecule: &Molecule) -> Result<Self, EngineError> {
        let mut shells: Vec<InternalShell> = Vec::new();
        let mut atom_shells: Vec<Vec<usize>> = Vec::with_capacity(molecule.n_atoms());
        let mut n_functions = 0usize;

        for (atom_index, atom) in molecule.atoms.iter().enumerate() {
            let block = basis.shells_for(&atom.symbol).ok_or_else(|| {
                EngineError::Data(format!(
                    "basis `{}` does not cover element {} (atom {}): available elements [{}]",
                    basis.name,
                    atom.symbol,
                    atom_index,
                    basis.elements.join(", ")
                ))
            })?;

            let center_bohr = atom.position_bohr();
            let mut indices = Vec::new();

            for shell in &block.shells {
                for (contraction_index, column) in shell.coefficients.iter().enumerate() {
                    if column.len() != shell.exponents.len() {
                        return Err(EngineError::Data(format!(
                            "basis `{}` {}: shells[l={}].coefficients[{}] has {} values but exponents has {}",
                            basis.name,
                            atom.symbol,
                            shell.l,
                            contraction_index,
                            column.len(),
                            shell.exponents.len()
                        )));
                    }
                    if shell.exponents.iter().any(|e| !e.is_finite() || *e <= 0.0) {
                        return Err(EngineError::Data(format!(
                            "basis `{}` {}: shells[l={}].exponents must be positive finite",
                            basis.name, atom.symbol, shell.l
                        )));
                    }

                    // 归一化（§8.1：只有这一处）。
                    let mut self_overlap = 0.0_f64;
                    for (i, a) in shell.exponents.iter().enumerate() {
                        for (j, b) in shell.exponents.iter().enumerate() {
                            self_overlap +=
                                column[i] * column[j] * primitive_self_overlap(*a, *b, shell.l);
                        }
                    }

                    if !(self_overlap.is_finite() && self_overlap > 0.0) {
                        return Err(EngineError::Data(format!(
                            "basis `{}` {}: shells[l={}] has non-positive self-overlap {self_overlap}",
                            basis.name, atom.symbol, shell.l
                        )));
                    }

                    let coefficients: Vec<f64> = match basis.normalization {
                        Normalization::L2 => {
                            if (self_overlap - 1.0).abs() > NORMALIZATION_TOLERANCE {
                                return Err(EngineError::Data(format!(
                                    "basis `{}` {}: shells[l={}] claims `normalization: l2` but \
                                     <f|f> = {self_overlap:.12} (tolerance {NORMALIZATION_TOLERANCE:e}); \
                                     either fix the coefficients or declare `normalization: none`",
                                    basis.name, atom.symbol, shell.l
                                )));
                            }
                            column.clone()
                        }
                        Normalization::None => {
                            let scale = 1.0 / self_overlap.sqrt();
                            column.iter().map(|c| c * scale).collect()
                        }
                        // `Normalization` 是 `#[non_exhaustive]`：未来可能新增取值。
                        // 加载器不认识就必须报错，不能猜（§8.1：归一化只有一处）。
                        _ => {
                            return Err(EngineError::Data(format!(
                                "basis `{}`: unsupported normalization value from data",
                                basis.name
                            )));
                        }
                    };

                    let first_function = n_functions;
                    n_functions += match basis.angular {
                        Angular::Cartesian => cartesian_components(shell.l).len(),
                        _ => 2 * shell.l as usize + 1,
                    };
                    indices.push(shells.len());
                    shells.push(InternalShell {
                        atom_index,
                        l: shell.l,
                        center_bohr,
                        angular: basis.angular,
                        exponents: shell.exponents.clone(),
                        coefficients,
                        first_function,
                    });
                }
            }
            atom_shells.push(indices);
        }

        Ok(Self {
            angular: basis.angular,
            atom_shells,
            shells,
            n_functions,
        })
    }

    /// AO 总数。
    #[must_use]
    pub fn n_functions(&self) -> usize {
        self.n_functions
    }

    /// 全部壳层。
    #[must_use]
    pub fn shells(&self) -> &[InternalShell] {
        &self.shells
    }

    /// 某原子上的壳层（迭代器，本身已 `#[must_use]`）。
    pub fn shells_of_atom(&self, atom_index: usize) -> impl Iterator<Item = &InternalShell> {
        self.atom_shells
            .get(atom_index)
            .into_iter()
            .flatten()
            .map(|index| &self.shells[*index])
    }

    /// 原子数。
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atom_shells.len()
    }

    /// 某原子上的 AO 个数。
    #[must_use]
    pub fn n_functions_of_atom(&self, atom_index: usize) -> usize {
        self.shells_of_atom(atom_index)
            .map(InternalShell::n_functions)
            .sum()
    }
}
