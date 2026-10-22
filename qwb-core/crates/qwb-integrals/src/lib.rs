//! I-02：**一电子积分**（重叠 S、动能 T、核吸引 V）。
//!
//! # 方法：McMurchie–Davidson 的 Hermite 展开
//!
//! 直角坐标高斯 `(x-A)^i (y-A)^j (z-A)^k exp(-α|r-A|²)` 两两相乘后是单个高斯
//! （Gaussian product theorem）。Hermite 系数 `E_t^{i,j}` 完全由递推给出，于是
//!
//! - `S = E_AB (π/p)^{3/2} E_0^{i_x,j_x} E_0^{i_y,j_y} E_0^{i_z,j_z}`
//! - `T = (1/2) ∫ (∇a)·(∇b)`：把 `∂/∂x` 展开成 `[i(x-A)^{i-1} − 2α(x-A)^{i+1}]`，
//!   于是动能**只是若干次错位重叠**，不需要另写一套核函数（少一处出错机会）。
//! - `V = −Z (2π/p) E_AB Σ_{tuv} E_t^{i,j} E_u^{k,l} E_v^{m,n} R^{t+u+v}`，
//!   其中 Hermite Coulomb 积分 `R^n = (−2p)^n F_n(p|P−C|²)`，`F_n` 是 Boys 函数。
//!
//! # 原函数归一化
//!
//! 收缩系数乘的是**归一化原函数**（与 `qwb-basis` 的归一化判据一致）：
//! `N(L,α) = (2α/π)^{3/4} (4α)^{L/2} / sqrt((2L−1)!!)`。
//!
//! # 独立验证（§7.0 允许的"内部独立路径"）
//!
//! 测试里的对照全是**解析闭式**，与本节实现走的是不同路径：
//! 同中心归一化原函数的 `S = 1`、`T = (2L+3)α/2`、`V = −Z·2√(2α/π)`（l=0）、
//! `V = 0`（l=1，对称性）、`V = −Z N²π/(3α)`（l=2 的 xx 分量），
//! 以及 H2/STO-3G 的 s-s 闭式。

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod boys;
pub mod one_electron;
pub mod two_electron;

pub use boys::boys_functions;
pub use one_electron::{OneElectronIntegrals, one_electron};
pub use two_electron::{TwoElectronIntegrals, primitive_ssss, two_electron};
