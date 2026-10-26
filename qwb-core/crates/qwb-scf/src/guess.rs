//! 初猜：**原子密度叠加**（SAD, superposition of atomic densities）。
//!
//! # 为什么需要它
//!
//! O2 的 UHF 有两个 SCF 解：`E = −147.378901` 与 `E = −147.633970`（后者低 0.255 Ha）。
//! 纯 `H_core`（core guess）会稳定地落到**较高**的那个 —— 这不是我们实现的 bug：
//! PySCF 用 `init_guess='hcore'` 也落在同一个数上（实测差 < 1e-8）。
//! 原因是 core guess 从分子轨道出发，数值噪声会轻微破坏 π_g 的简并，SCF 随后把这个
//! 破缺放大；而**原子密度**是球对称的，不会破坏对称性，于是收敛到真正的变分极小。
//!
//! 这也是所有生产级代码的默认做法（Jaguar/Gaussian/PySCF 的默认初猜都不是 core）。
//!
//! # 做法
//!
//! 对分子里的每个原子：
//!
//! 1. 用**同一套基组**建一个单原子片段（放在原点，中性，取该元素的基态多重度）；
//! 2. 用同一个 `uhf_scf`（多重度 1 时它自然退化成 RHF）算这个片段；
//! 3. 把片段密度**按 AO 块**搬到分子的 AO 基里（片段的壳层顺序与分子里该原子的壳层
//!    顺序一致，所以只是块对角搬运）；
//! 4. 所有原子相加得到 `P_α`、`P_β`。
//!
//! 片段 SCF 本身用 **core guess**（避免递归）。片段失败时退回 core guess 并**打一条警告**
//! —— 初猜只是初猜，不该让整个计算失败，但也不该悄悄换策略。

use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_linalg::Matrix;

use crate::uhf::{UhfOptions, uhf_scf};

/// 元素的中性基态自旋多重度（H–Kr）。
///
/// 只用于**初猜**的原子片段（不参与任何能量/性质计算），所以这里写的是"元素表"级别的
/// 常识数据：H 2、He 1、…、Cr 7、Mn 6、Fe 5、Cu 2、Zn 1。
#[must_use]
pub fn ground_state_multiplicity(symbol: &str) -> Option<u32> {
    const TABLE: [(&str, u32); 36] = [
        ("H", 2),
        ("He", 1),
        ("Li", 2),
        ("Be", 1),
        ("B", 2),
        ("C", 3),
        ("N", 4),
        ("O", 3),
        ("F", 2),
        ("Ne", 1),
        ("Na", 2),
        ("Mg", 1),
        ("Al", 2),
        ("Si", 3),
        ("P", 4),
        ("S", 3),
        ("Cl", 2),
        ("Ar", 1),
        ("K", 2),
        ("Ca", 1),
        ("Sc", 2),
        ("Ti", 3),
        ("V", 4),
        ("Cr", 7),
        ("Mn", 6),
        ("Fe", 5),
        ("Co", 4),
        ("Ni", 3),
        ("Cu", 2),
        ("Zn", 1),
        ("Ga", 2),
        ("Ge", 3),
        ("As", 4),
        ("Se", 3),
        ("Br", 2),
        ("Kr", 1),
    ];
    TABLE
        .iter()
        .find(|(element, _)| *element == symbol)
        .map(|(_, multiplicity)| *multiplicity)
}

/// 一个原子的 AO 块：`(分子里的起止下标, 片段里的起止下标)`。
///
/// 片段基只包含这个原子的壳层，所以两边的块大小相同、内部顺序相同。
fn atom_blocks(basis: &qwb_basis::InternalBasis, atom_index: usize) -> Vec<(usize, usize, usize)> {
    let mut blocks = Vec::new();
    let mut local = 0usize;
    for shell in basis.shells_of_atom(atom_index) {
        let width = shell.n_functions();
        blocks.push((shell.first_function, local, width));
        local += width;
    }
    blocks
}

/// 原子密度叠加初猜：返回分子 AO 基上的 `(P_α, P_β)`。
///
/// # Errors
///
/// 片段基重建失败、元素没有基态多重度、或片段 SCF 出现非收敛类错误时返回错误。
/// 片段**不收敛**时退回 core guess 并打一条警告（初猜失败不该杀死整个计算）。
pub fn sad_guess(
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
) -> Result<(Matrix, Matrix), EngineError> {
    let n = basis.n_functions();
    let mut density_alpha = Matrix::zeros(n, n);
    let mut density_beta = Matrix::zeros(n, n);
    for (atom_index, atom) in molecule.atoms.iter().enumerate() {
        let multiplicity = ground_state_multiplicity(&atom.symbol).ok_or_else(|| {
            EngineError::Data(format!(
                "no ground-state multiplicity for element `{}` (SAD guess)",
                atom.symbol
            ))
        })?;
        let fragment = Molecule::new(
            vec![Atom::new(&atom.symbol, [0.0, 0.0, 0.0])],
            0,
            multiplicity,
        );
        let fragment_basis = match basis.sub_basis_of_atom(atom_index) {
            Ok(fragment) => fragment,
            Err(error) => {
                eprintln!(
                    "warning: SAD guess unavailable ({error}); falling back to the core guess"
                );
                return Ok((Matrix::zeros(n, n), Matrix::zeros(n, n)));
            }
        };
        let tensor = qwb_integrals::two_electron::two_electron(&fragment_basis, &fragment)?;
        // 片段 SCF 用 core guess（避免递归）。
        let options = UhfOptions {
            initial_guess: crate::uhf::UhfInitialGuess::Core,
            max_iter: 200,
            ..UhfOptions::default()
        };
        let outcome = uhf_scf(&fragment_basis, &fragment, &tensor, options);
        let fragment_result = match outcome {
            Ok(result) => result,
            Err(error) => {
                eprintln!(
                    "warning: atomic SCF for {} failed ({error}); falling back to the core guess",
                    atom.symbol
                );
                return Ok((Matrix::zeros(n, n), Matrix::zeros(n, n)));
            }
        };
        // **自旋平均**：把片段的总密度对半分给 α/β。这是经典 SAD 的做法
        // （Van Lenthe & Baerends），也是它保住对称性的原因 —— 片段的"哪个 p 轨道
        // 是单占"这种细节不该带进分子的初猜。
        for (global_start, local_start, width) in atom_blocks(basis, atom_index) {
            for i in 0..width {
                for j in 0..width {
                    let total = fragment_result.density_alpha[(local_start + i, local_start + j)]
                        + fragment_result.density_beta[(local_start + i, local_start + j)];
                    density_alpha[(global_start + i, global_start + j)] += 0.5 * total;
                    density_beta[(global_start + i, global_start + j)] += 0.5 * total;
                }
            }
        }
    }
    Ok((density_alpha, density_beta))
}

#[cfg(test)]
mod tests {
    use super::ground_state_multiplicity;

    #[test]
    fn multiplicities_match_the_periodic_table_ground_states() {
        assert_eq!(ground_state_multiplicity("H"), Some(2));
        assert_eq!(ground_state_multiplicity("C"), Some(3));
        assert_eq!(ground_state_multiplicity("O"), Some(3));
        assert_eq!(ground_state_multiplicity("Cr"), Some(7));
        assert_eq!(ground_state_multiplicity("Zn"), Some(1));
        assert_eq!(ground_state_multiplicity("Xx"), None);
    }
}
