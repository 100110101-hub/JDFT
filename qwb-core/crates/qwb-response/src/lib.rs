//! `qwb-response`：**CPHF / UCPHF 求解器**与静态极化率（阶段二 RS-01）。
//!
//! # 为什么单独一个 crate
//!
//! `project-phase2.md` §10.5 把 `RS-01`（CPHF）列为 `RS-02`（TDA/TDDFT）、`RS-04`（NMR
//! GIAO）、`RS-05`（解析 Hessian）、`RS-06`（Raman）的**公共前置**。它的形状与
//! "能量 + 梯度"那套不同（解的是线性响应方程），所以独立成 crate，而不是塞进
//! `qwb-scf`：SCF 只管基态，响应方程的解是另一层。
//!
//! # 这一层的数学骨架
//!
//! 闭壳层 RHF 的**静态**（ω = 0）线性响应方程（推导见
//! [`cphf::orbital_hessian_apply`]）：
//!
//! ```text
//! (A + B)_{ai,bj} = (ε_a − ε_i) δ_ij δ_ab + 4(ai|bj) − (ab|ij) − (aj|ib)
//! ```
//!
//! 其中 `i,j` 占占据、`a,b` 跑虚轨道，全部是**空间轨道**（自旋已配对进因子 4）。
//! 扰动 `v`（例如均匀电场 `v_{ai} = F·⟨a|r|i⟩`）下解
//!
//! ```text
//! (A + B) X = −v
//! ```
//!
//! 再对解做一阶期望。静态极化率就是电场扰动的解与偶极矩阵元的双线性型：
//!
//! ```text
//! α_ij = 4 Σ_{ai,bj} d_i^{ai} [(A+B)⁻¹]_{(ai),(bj)} d_j^{bj}
//! ```
//!
//! （`d_k^{ai} = ⟨a|r_k|i⟩`，即 [`qwb_integrals::dipole`] 给的矩阵元。）
//!
//! # 求解器：矩阵自由的迭代，不是把 (A+B) 稠密化
//!
//! 判据要的是「迭代、2e 驱动」（`project-phase2.md` §6.6）。所以 `(A+B)` 只以
//! **算子**形式出现（[`cphf::orbital_hessian_apply`]），迭代用 Jacobi 预条件的
//! 共轭梯度（(A+B) 是 SPD），收敛判据是残差的 2-范数。
//!
//! 索引约定：把 `(ai)` 展平成下标 `p = i·n_vir + a`（占据在外层），与
//! `qwb-scf` 里 `(ia|jb)` 的展开方式一致，便于交叉核对。
#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod cphf;
pub mod mo_integrals;
pub mod polarizability;

pub use cphf::{CphfOptions, CphfResult, orbital_hessian_apply, solve_cphf};
pub use polarizability::{Polarizability, static_polarizability};
