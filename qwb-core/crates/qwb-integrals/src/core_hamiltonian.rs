//! 核心哈密顿量对磁场的导数 `∂H_core/∂B_k = i X_k`（GIAO 层的第二个积分）。
//!
//! # 公式
//!
//! London（GIAO）轨道 `χ_μ^B(r) = exp(−i K_μ·r) χ_μ(r)`，其中
//! `K_μ = (1/2) B × (R_μ − O)`、`O` 是规范原点。核心哈密顿量矩阵与它在 `B = 0` 的
//! 一阶导数：
//!
//! ```text
//! H_μν(B) = ⟨χ_μ^B| T + V |χ_ν^B⟩
//!
//! ∂H/∂B_k |_{B=0} = i X_k
//!
//! X_k[μ,ν] = (1/2) Σ_l { [e_k × (R_μ − O)]_l M_l[μ,ν]
//!                      − [e_k × (R_ν − O)]_l M_l[ν,μ] }
//! ```
//!
//! `M_l = A_l + B_l` 是 [`crate::giao::kinetic_moment_integrals`] 与
//! [`crate::giao::nuclear_moment_integrals`] 之和（`⟨μ| r_l (T+V) |ν⟩`），**不需要**任何
//! 新积分。
//!
//! # 三个容易写错的点（都由测试钉住）
//!
//! 1. 第二项用的是 `M_l[ν,μ]`，即第一项的**转置**——不是另一个积分。实基 + 厄米 `W` +
//!    实乘性 `r_l` 给出 `⟨μ|W r_l|ν⟩ = ⟨ν| r_l W|μ⟩`。
//! 2. 叉积符号由 `∂(B × a)/∂B_k = e_k × a` 定出：`(e_k × a)_l = ε_{lkm} a_m`，例如
//!    `e_0 × a = (0, −a_z, a_y)`（不是 `(0, a_z, −a_y)`）。
//! 3. 规范原点 `O` **不**在这张矩阵里抵消（只在进入可观测量的比值里抵消）。所以 `X_k`
//!    依赖 `O`，并精确满足
//!    `X_k(O+δ) − X_k(O) = −(1/2) Σ_l [e_k × δ]_l p_l[μ,ν]`，
//!    其中 `p_l` 是 [`crate::momentum::momentum_integrals`]。
//!
//! 返回的是**实**的 `X_k`（crate 的 [`Matrix`] 是实 `f64` 矩阵）；约定与
//! [`crate::giao::giao_overlap_derivative`] 相同：虚数因子 `i` 由调用方乘进去。
#![deny(unsafe_code)]

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;

use crate::giao::nuclear_moment_integrals_with_charges;
use crate::giao::{kinetic_moment_integrals, nuclear_moment_integrals};

/// 三个**实**矩阵 `X_k`，满足 `∂H_core/∂B_k |_{B=0} = i X_k`。
///
/// `origin` 是规范原点 `O`，单位 **Bohr**（与 [`qwb_basis::InternalShell::center_bohr`]
/// 同一坐标系）。核电荷取元素表里的 `Z`；有 ECP 的体系用
/// [`core_hamiltonian_field_derivative_with_charges`]。
///
/// # Errors
///
/// 分子里出现元素表之外的符号 → [`EngineError::Internal`]（来自核磁矩积分）；
/// 球谐变换不可用 → [`EngineError::Unsupported`]；矩阵维度不一致 → [`EngineError::Internal`]。
pub fn core_hamiltonian_field_derivative(
    basis: &InternalBasis,
    molecule: &Molecule,
    origin: [f64; 3],
) -> Result<[Matrix; 3], EngineError> {
    let kinetic = kinetic_moment_integrals(basis)?;
    let nuclear = nuclear_moment_integrals(basis, molecule)?;
    assemble_core(basis, &kinetic, &nuclear, origin)
}

/// 与 [`core_hamiltonian_field_derivative`] 相同，但核电荷由调用方给（ECP 的 `z_eff`，
/// 与 [`crate::one_electron::one_electron_with_charges`] 同一约定）。
///
/// # Errors
///
/// `charges` 长度与原子数不符 → [`EngineError::Internal`]；其余同
/// [`core_hamiltonian_field_derivative`]。
pub fn core_hamiltonian_field_derivative_with_charges(
    basis: &InternalBasis,
    molecule: &Molecule,
    charges: &[f64],
    origin: [f64; 3],
) -> Result<[Matrix; 3], EngineError> {
    let kinetic = kinetic_moment_integrals(basis)?;
    let nuclear = nuclear_moment_integrals_with_charges(basis, molecule, charges)?;
    assemble_core(basis, &kinetic, &nuclear, origin)
}

/// 把动能磁矩与核磁矩加成 `M_l`，再交给 [`assemble_field_derivative`]。
fn assemble_core(
    basis: &InternalBasis,
    kinetic: &[Matrix; 3],
    nuclear: &[Matrix; 3],
    origin: [f64; 3],
) -> Result<[Matrix; 3], EngineError> {
    let n = basis.n_functions();
    for matrix in kinetic.iter().chain(nuclear.iter()) {
        if matrix.nrows() != n || matrix.ncols() != n {
            return Err(EngineError::Internal(format!(
                "core_hamiltonian_field_derivative: moment matrix is {}x{}, expected {n}x{n}",
                matrix.nrows(),
                matrix.ncols()
            )));
        }
    }
    let mut core_moment: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    for (axis, moment) in core_moment.iter_mut().enumerate() {
        for row in 0..n {
            for column in 0..n {
                moment[(row, column)] =
                    kinetic[axis].get(row, column) + nuclear[axis].get(row, column);
            }
        }
    }
    assemble_field_derivative(basis, &core_moment, origin)
}

/// 由任意一组 `M_l[μ,ν]` 按 `X_k[μ,ν] = (1/2) Σ_l {c_μ,l M_l[μ,ν] − c_ν,l M_l[ν,μ]}` 组装。
///
/// `pub(crate)` 只是为了让 `R3` 那条"绝对尺子"能用**同一份**组装代码、但只喂动能磁矩
/// `A_l`（有限差分独立给出 `∂H^T/∂B_k = i X_k`）——这样被检验的是生产路径本身。
#[allow(clippy::needless_range_loop)]
pub(crate) fn assemble_field_derivative(
    basis: &InternalBasis,
    moment: &[Matrix; 3],
    origin: [f64; 3],
) -> Result<[Matrix; 3], EngineError> {
    let n = basis.n_functions();
    for matrix in moment {
        if matrix.nrows() != n || matrix.ncols() != n {
            return Err(EngineError::Internal(format!(
                "core_hamiltonian_field_derivative: moment matrix is {}x{}, expected {n}x{n}",
                matrix.nrows(),
                matrix.ncols()
            )));
        }
    }

    // 每个基函数的中心（同一壳层的所有分量共享中心）。
    let mut centers: Vec<[f64; 3]> = Vec::with_capacity(n);
    for shell in basis.shells() {
        for _ in 0..shell.n_functions() {
            centers.push(shell.center_bohr);
        }
    }
    if centers.len() != n {
        return Err(EngineError::Internal(format!(
            "core_hamiltonian_field_derivative: {n} basis functions but {} centers",
            centers.len()
        )));
    }

    // c[k][l] = [e_k × (R_μ − O)]_l。
    let crosses: Vec<[[f64; 3]; 3]> = centers
        .iter()
        .map(|center| {
            cross_unit([
                center[0] - origin[0],
                center[1] - origin[1],
                center[2] - origin[2],
            ])
        })
        .collect();

    let mut out: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    for mu in 0..n {
        for nu in 0..n {
            for (k, matrix) in out.iter_mut().enumerate() {
                let mut value = 0.0_f64;
                for l in 0..3 {
                    value += crosses[mu][k][l] * moment[l].get(mu, nu)
                        - crosses[nu][k][l] * moment[l].get(nu, mu);
                }
                matrix[(mu, nu)] = 0.5 * value;
            }
        }
    }
    Ok(out)
}

/// `[e_k × a]_l` 的三个 `k` 切片：`(e_k × a)_l = ε_{lkm} a_m`。
///
/// 符号来源：`∂(B × a)/∂B_k = e_k × a`，即 `e_0 × a = (0, −a_z, a_y)`。
fn cross_unit(a: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
}

#[cfg(test)]
mod tests {
    use super::{assemble_field_derivative, core_hamiltonian_field_derivative, cross_unit};
    use crate::giao::kinetic_moment_integrals;
    use crate::momentum::{momentum_integrals, power_along, with_power};
    use crate::one_electron::one_electron;
    use qwb_basis::InternalBasis;
    use qwb_engine_api::{Atom, Molecule};
    use qwb_linalg::Matrix;
    use qwb_registry::Registry;
    use std::f64::consts::PI;
    use std::path::PathBuf;

    fn data_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
    }

    /// 水，三个中心全都离开原点（`X_k` 与 `O` 的依赖不会意外退化）。
    fn displaced_water() -> Molecule {
        Molecule::new(
            vec![
                Atom::new("O", [0.30, -0.25, 0.42]),
                Atom::new("H", [0.30, 0.507, 1.006]),
                Atom::new("H", [0.30, -1.007, 1.006]),
            ],
            0,
            1,
        )
    }

    /// 水，**三个原子中心全在同一点**（`R1`：所有 `R_μ − O = 0`）。
    fn one_centre_water(point_angstrom: [f64; 3]) -> Molecule {
        Molecule::new(
            vec![
                Atom::new("O", point_angstrom),
                Atom::new("H", point_angstrom),
                Atom::new("H", point_angstrom),
            ],
            0,
            1,
        )
    }

    fn basis(name: &str, molecule: &Molecule) -> InternalBasis {
        let registry = Registry::open(data_root()).expect("qwb-data");
        InternalBasis::build(registry.get_basis(name).expect("basis"), molecule)
            .expect("basis build")
    }

    fn max_abs(matrices: &[Matrix; 3]) -> f64 {
        let mut worst = 0.0_f64;
        for matrix in matrices {
            for row in 0..matrix.nrows() {
                for column in 0..matrix.ncols() {
                    worst = worst.max(matrix.get(row, column).abs());
                }
            }
        }
        worst
    }

    // ===== 复数的极小封装（不引入新依赖）=====

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct C {
        re: f64,
        im: f64,
    }

    impl C {
        const ZERO: C = C { re: 0.0, im: 0.0 };
        const ONE: C = C { re: 1.0, im: 0.0 };

        fn new(re: f64, im: f64) -> Self {
            Self { re, im }
        }

        fn magnitude(self) -> f64 {
            (self.re * self.re + self.im * self.im).sqrt()
        }
    }

    impl std::ops::Add for C {
        type Output = C;

        fn add(self, other: C) -> C {
            C::new(self.re + other.re, self.im + other.im)
        }
    }

    impl std::ops::Sub for C {
        type Output = C;

        fn sub(self, other: C) -> C {
            C::new(self.re - other.re, self.im - other.im)
        }
    }

    impl std::ops::Mul for C {
        type Output = C;

        fn mul(self, other: C) -> C {
            C::new(
                self.re * other.re - self.im * other.im,
                self.re * other.im + self.im * other.re,
            )
        }
    }

    impl std::ops::Mul<f64> for C {
        type Output = C;

        fn mul(self, factor: f64) -> C {
            C::new(self.re * factor, self.im * factor)
        }
    }

    /// `(u + shift)^power` 按 `u^k` 升幂的复系数。
    fn shifted_poly(power: u32, shift: C) -> Vec<C> {
        let mut poly = vec![C::ONE];
        for _ in 0..power {
            let mut next = vec![C::ZERO; poly.len() + 1];
            for (k, coefficient) in poly.iter().enumerate() {
                next[k] = next[k] + *coefficient * shift;
                next[k + 1] = next[k + 1] + *coefficient;
            }
            poly = next;
        }
        poly
    }

    fn poly_mul(a: &[C], b: &[C]) -> Vec<C> {
        let mut out = vec![C::ZERO; a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            if ai.magnitude() == 0.0 {
                continue;
            }
            for (j, bj) in b.iter().enumerate() {
                out[i + j] = out[i + j] + *ai * *bj;
            }
        }
        out
    }

    /// `∫ u^k e^{−p u²} du`。
    fn gaussian_moment_1d(k: usize, p: f64) -> f64 {
        if k % 2 == 1 {
            return 0.0;
        }
        let mut double_factorial = 1.0_f64;
        let mut j = 1usize;
        while j < k {
            double_factorial *= j as f64;
            j += 2;
        }
        double_factorial / (2.0 * p).powi((k / 2) as i32) * (PI / p).sqrt()
    }

    /// `∫ e^{i c x} (x−A)^a (x−B)^b e^{−α(x−A)²−β(x−B)²} dx`（**未归一化**原函数）。
    ///
    /// 配方：`−p(x−P)² + i c x = −p(u)² + i c P − c²/(4p)`，`u = x − P − i c/(2p)`，
    /// 于是多项式在复中心 `P + i c/(2p)` 处取值，逐幂次用实高斯矩收缩。
    fn phased_overlap_1d(
        alpha: f64,
        a: u32,
        center_a: f64,
        beta: f64,
        b: u32,
        center_b: f64,
        c: f64,
    ) -> C {
        let p = alpha + beta;
        let product = (alpha * center_a + beta * center_b) / p;
        let e_ab = (-alpha * beta / p * (center_a - center_b).powi(2)).exp();
        let z = C::new(0.0, c / (2.0 * p));
        let bra = shifted_poly(a, C::new(product - center_a, 0.0) + z);
        let ket = shifted_poly(b, C::new(product - center_b, 0.0) + z);
        let poly = poly_mul(&bra, &ket);
        let mut sum = C::ZERO;
        for (k, coefficient) in poly.iter().enumerate() {
            let moment = gaussian_moment_1d(k, p);
            if moment != 0.0 {
                sum = sum + *coefficient * moment;
            }
        }
        let damping = (-(c * c) / (4.0 * p)).exp();
        let phase = C::new((c * product).cos(), (c * product).sin());
        sum * (e_ab * damping) * phase
    }

    /// 三维相因子重叠：三个轴相乘（`e^{i c·r}` 与高斯都是按轴分离的）。
    fn phased_overlap(
        powers_a: (u32, u32, u32),
        center_a: [f64; 3],
        alpha: f64,
        powers_b: (u32, u32, u32),
        center_b: [f64; 3],
        beta: f64,
        c: [f64; 3],
    ) -> C {
        let mut out = C::ONE;
        for axis in 0..3 {
            out = out
                * phased_overlap_1d(
                    alpha,
                    power_along(powers_a, axis),
                    center_a[axis],
                    beta,
                    power_along(powers_b, axis),
                    center_b[axis],
                    c[axis],
                );
        }
        out
    }

    /// 闭合形式的**动能半**：`⟨μ| e^{iK_μ·r} T e^{−iK_ν·r} |ν⟩`（未乘收缩系数与 `N_a`）。
    ///
    /// `T g = −(1/2) e^{−iK_ν·r}[∇²χ_ν − 2i K_ν·∇χ_ν − |K_ν|²χ_ν]`，括号里每一项都是
    /// 同中心、同指数的直角高斯（幂次在某一个轴上 ±1 / ±2），于是每个矩阵元都是若干
    /// "相因子重叠"的线性组合。
    fn giao_kinetic_primitive(
        powers_a: (u32, u32, u32),
        center_a: [f64; 3],
        alpha: f64,
        powers_b: (u32, u32, u32),
        center_b: [f64; 3],
        beta: f64,
        k_mu: [f64; 3],
        k_nu: [f64; 3],
    ) -> C {
        let c = [k_mu[0] - k_nu[0], k_mu[1] - k_nu[1], k_mu[2] - k_nu[2]];
        let k_squared = k_nu[0] * k_nu[0] + k_nu[1] * k_nu[1] + k_nu[2] * k_nu[2];
        let plain = phased_overlap(powers_a, center_a, alpha, powers_b, center_b, beta, c);
        let mut total = plain * -k_squared;
        for (axis, k_nu_axis) in k_nu.iter().enumerate() {
            let bi = power_along(powers_b, axis);
            // ∇² 的 −2β(2b+1) 项（每个轴一次）。
            total = total + plain * (-2.0 * beta * (2.0 * f64::from(bi) + 1.0));
            // ∇² 的 b(b−1) 项（降 2 次）。
            if bi >= 2 {
                let lowered = with_power(powers_b, axis, bi - 2);
                total = total
                    + phased_overlap(powers_a, center_a, alpha, lowered, center_b, beta, c)
                        * (f64::from(bi) * f64::from(bi - 1));
            }
            // ∇² 的 4β² 项（升 2 次）。
            let raised = with_power(powers_b, axis, bi + 2);
            total = total
                + phased_overlap(powers_a, center_a, alpha, raised, center_b, beta, c)
                    * (4.0 * beta * beta);
            // −2i K_ν·∇ 的 b 项（降 1 次）：系数 −2i K b。
            if bi >= 1 {
                let lowered = with_power(powers_b, axis, bi - 1);
                total = total
                    + phased_overlap(powers_a, center_a, alpha, lowered, center_b, beta, c)
                        * C::new(0.0, -2.0 * k_nu_axis * f64::from(bi));
            }
            // −2i K_ν·∇ 的 −2β 项（升 **1** 次，与 ∇² 的升 2 次不同）：系数 +4i β K。
            let raised_once = with_power(powers_b, axis, bi + 1);
            total = total
                + phased_overlap(powers_a, center_a, alpha, raised_once, center_b, beta, c)
                    * C::new(0.0, 4.0 * beta * k_nu_axis);
        }
        total * -0.5
    }

    /// 组装球谐基下的闭合形式动能矩阵（实部、虚部两个 `Matrix`）。
    fn giao_kinetic_spherical(
        basis: &InternalBasis,
        field: [f64; 3],
        origin: [f64; 3],
    ) -> (Matrix, Matrix) {
        let mut transforms = Vec::with_capacity(basis.shells().len());
        for shell in basis.shells() {
            transforms.push(
                qwb_basis::spherical::spherical_transform(shell.l, shell.angular)
                    .expect("spherical transform"),
            );
        }
        let mut cart_offsets = Vec::with_capacity(transforms.len());
        let mut n_cartesian = 0usize;
        for transform in &transforms {
            cart_offsets.push(n_cartesian);
            n_cartesian += transform.n_cartesian;
        }
        let shells: Vec<_> = basis.shells().iter().collect();
        let mut real = Matrix::zeros(n_cartesian, n_cartesian);
        let mut imaginary = Matrix::zeros(n_cartesian, n_cartesian);
        for (ia, shell_a) in shells.iter().enumerate() {
            let k_mu = [
                0.5 * (field[1] * (shell_a.center_bohr[2] - origin[2])
                    - field[2] * (shell_a.center_bohr[1] - origin[1])),
                0.5 * (field[2] * (shell_a.center_bohr[0] - origin[0])
                    - field[0] * (shell_a.center_bohr[2] - origin[2])),
                0.5 * (field[0] * (shell_a.center_bohr[1] - origin[1])
                    - field[1] * (shell_a.center_bohr[0] - origin[0])),
            ];
            for (ib, shell_b) in shells.iter().enumerate() {
                let k_nu = [
                    0.5 * (field[1] * (shell_b.center_bohr[2] - origin[2])
                        - field[2] * (shell_b.center_bohr[1] - origin[1])),
                    0.5 * (field[2] * (shell_b.center_bohr[0] - origin[0])
                        - field[0] * (shell_b.center_bohr[2] - origin[2])),
                    0.5 * (field[0] * (shell_b.center_bohr[1] - origin[1])
                        - field[1] * (shell_b.center_bohr[0] - origin[0])),
                ];
                let components_a = qwb_basis::cartesian_components(shell_a.l);
                let components_b = qwb_basis::cartesian_components(shell_b.l);
                for (ca, powers_a) in components_a.iter().enumerate() {
                    for (cb, powers_b) in components_b.iter().enumerate() {
                        let mut sum = C::ZERO;
                        for (alpha, coefficient_a) in
                            shell_a.exponents.iter().zip(&shell_a.coefficients)
                        {
                            let norm_a = crate::one_electron::primitive_norm(*powers_a, *alpha);
                            for (beta, coefficient_b) in
                                shell_b.exponents.iter().zip(&shell_b.coefficients)
                            {
                                let norm_b = crate::one_electron::primitive_norm(*powers_b, *beta);
                                let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                                sum = sum
                                    + giao_kinetic_primitive(
                                        *powers_a,
                                        shell_a.center_bohr,
                                        *alpha,
                                        *powers_b,
                                        shell_b.center_bohr,
                                        *beta,
                                        k_mu,
                                        k_nu,
                                    ) * weight;
                            }
                        }
                        let row = cart_offsets[ia] + ca;
                        let column = cart_offsets[ib] + cb;
                        real[(row, column)] = sum.re;
                        imaginary[(row, column)] = sum.im;
                    }
                }
            }
        }

        // 球谐变换（实 `T`：实部/虚部分开乘 `Tᵀ M T`）。
        let n_spherical = basis.n_functions();
        let mut global = Matrix::zeros(n_cartesian, n_spherical);
        for (shell_index, shell) in shells.iter().enumerate() {
            let transform = &transforms[shell_index];
            for row in 0..transform.n_cartesian {
                for column in 0..transform.n_spherical {
                    global[(
                        cart_offsets[shell_index] + row,
                        shell.first_function + column,
                    )] = transform.get(row, column);
                }
            }
        }
        let mut out_real = Matrix::zeros(n_spherical, n_spherical);
        let mut out_imag = Matrix::zeros(n_spherical, n_spherical);
        for mu in 0..n_spherical {
            for nu in 0..n_spherical {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for a in 0..n_cartesian {
                    let ta = global.get(a, mu);
                    if ta == 0.0 {
                        continue;
                    }
                    for b in 0..n_cartesian {
                        let tb = global.get(b, nu);
                        if tb == 0.0 {
                            continue;
                        }
                        let both = ta * tb;
                        re += both * real.get(a, b);
                        im += both * imaginary.get(a, b);
                    }
                }
                out_real[(mu, nu)] = re;
                out_imag[(mu, nu)] = im;
            }
        }
        (out_real, out_imag)
    }

    // ===== R1：所有中心都在规范原点上 ⇒ X_k 必须恒为零（而 M_l 非零）=====

    #[test]
    fn r1_all_centres_on_the_origin_give_exactly_zero() {
        let point = [0.31, -0.22, 0.47]; // Å
        let molecule = one_centre_water(point);
        let origin = molecule.atoms[0].position_bohr();
        let basis = basis("sto-3g", &molecule);

        let x = core_hamiltonian_field_derivative(&basis, &molecule, origin).expect("X");
        let kinetic = kinetic_moment_integrals(&basis).expect("kinetic moment");
        let nuclear = crate::giao::nuclear_moment_integrals(&basis, &molecule).expect("nuclear");
        let mut moment: [Matrix; 3] =
            std::array::from_fn(|_| Matrix::zeros(basis.n_functions(), basis.n_functions()));
        for axis in 0..3 {
            for row in 0..basis.n_functions() {
                for column in 0..basis.n_functions() {
                    moment[axis][(row, column)] =
                        kinetic[axis].get(row, column) + nuclear[axis].get(row, column);
                }
            }
        }
        let x_scale = max_abs(&x);
        let moment_scale = max_abs(&moment);
        // 同一个几何、同一个基组，但规范原点换到远离所有中心的地方 ⇒ X 必须非零
        // （证明零是"O 取在中心"造成的，不是 M_l 或组装退化了）。
        let x_elsewhere =
            core_hamiltonian_field_derivative(&basis, &molecule, [5.0, -4.0, 3.0]).expect("X");
        let x_elsewhere_scale = max_abs(&x_elsewhere);
        println!(
            "R1: max|X(O=中心)| = {x_scale:.3e}；max|M_l| = {moment_scale:.3e}；\
             max|X(O=(5,−4,3))| = {x_elsewhere_scale:.3e}"
        );
        assert_eq!(x_scale, 0.0, "R1: 所有中心都在 O 上时 X_k 必须精确为零");
        assert!(
            moment_scale > 1.0,
            "R1: max|M_l| 只有 {moment_scale:.3e}，尺子是空的"
        );
        assert!(
            x_elsewhere_scale > 1.0e-3,
            "R1: 换一个规范原点后 X 仍只有 {x_elsewhere_scale:.3e}，零不是 O 造成的"
        );
    }

    /// `_with_charges` 变体与元素表路径必须逐位一致（ECP 的 `z_eff` 走的正是这条），
    /// 并且核电荷个数不符要报错。
    #[test]
    fn the_charge_variant_matches_the_element_table_variant() {
        let molecule = displaced_water();
        let basis = basis("sto-3g", &molecule);
        let origin = [0.21, -0.33, 0.44];
        // 用 crate 根的重导出，顺带验证 `pub use core_hamiltonian::...`。
        let by_table = crate::core_hamiltonian_field_derivative(&basis, &molecule, origin)
            .expect("element table");
        let by_charge = crate::core_hamiltonian_field_derivative_with_charges(
            &basis,
            &molecule,
            &[8.0, 1.0, 1.0],
            origin,
        )
        .expect("charges");
        let mut worst = 0.0_f64;
        for axis in 0..3 {
            for row in 0..basis.n_functions() {
                for column in 0..basis.n_functions() {
                    worst = worst.max(
                        (by_table[axis].get(row, column) - by_charge[axis].get(row, column)).abs(),
                    );
                }
            }
        }
        println!("_with_charges 与元素表路径最差 {worst:.3e}");
        assert_eq!(worst, 0.0, "_with_charges 与元素表路径必须逐位一致");
        assert!(
            crate::core_hamiltonian_field_derivative_with_charges(
                &basis,
                &molecule,
                &[8.0],
                origin
            )
            .is_err(),
            "核电荷个数不符必须报错"
        );
    }

    // ===== R2：规范恒等式 X_k(O+δ) − X_k(O) = −(1/2) Σ_l [e_k × δ]_l p_l =====

    #[test]
    fn r2_the_gauge_identity_holds_against_the_momentum_integrals() {
        let molecule = displaced_water();
        let origin = [0.13, -0.41, 0.27];
        let delta = [0.31, -0.17, 0.23];
        let shifted_origin = [
            origin[0] + delta[0],
            origin[1] + delta[1],
            origin[2] + delta[2],
        ];
        for name in ["sto-3g", "def2-svp"] {
            let basis = basis(name, &molecule);
            let x = core_hamiltonian_field_derivative(&basis, &molecule, origin).expect("X(O)");
            let x_shifted = core_hamiltonian_field_derivative(&basis, &molecule, shifted_origin)
                .expect("X(O+δ)");
            let momentum = momentum_integrals(&basis).expect("momentum");
            let n = basis.n_functions();
            let cross = cross_unit(delta);
            let mut worst = 0.0_f64;
            let mut predicted = 0.0_f64;
            let mut scale = 0.0_f64;
            for k in 0..3 {
                for mu in 0..n {
                    for nu in 0..n {
                        let mut expected = 0.0_f64;
                        for l in 0..3 {
                            expected -= 0.5 * cross[k][l] * momentum[l].get(mu, nu);
                        }
                        let measured = x_shifted[k].get(mu, nu) - x[k].get(mu, nu);
                        worst = worst.max((measured - expected).abs());
                        predicted = predicted.max(expected.abs());
                        scale = scale.max(x[k].get(mu, nu).abs());
                    }
                }
            }
            println!(
                "R2 {name}: max|X| = {scale:.3e}，max|Δ预测| = {predicted:.3e}，\
                 规范恒等式最差 {worst:.3e}（相对 {:.3e}）",
                worst / scale
            );
            assert!(
                predicted > 1.0e-3,
                "R2 {name}: 预测的差值整体为零，尺子是空的"
            );
            assert!(
                worst / scale < 1.0e-12,
                "R2 {name}: 规范恒等式差 {worst:.3e}（max|X| = {scale:.3e}，判据 1e-12 相对）"
            );
        }
    }

    // ===== R3：闭合形式动能半的中心差分（独立的**绝对**尺子）=====

    /// `∂H^T/∂B_k` 的中心差分（实部、虚部）。
    fn kinetic_derivative_fd(
        basis: &InternalBasis,
        origin: [f64; 3],
        axis: usize,
        h: f64,
    ) -> (Matrix, Matrix) {
        let mut plus = [0.0_f64; 3];
        let mut minus = [0.0_f64; 3];
        plus[axis] = h;
        minus[axis] = -h;
        let (plus_real, plus_imag) = giao_kinetic_spherical(basis, plus, origin);
        let (minus_real, minus_imag) = giao_kinetic_spherical(basis, minus, origin);
        let n = plus_real.nrows();
        let mut real = Matrix::zeros(n, n);
        let mut imaginary = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                real[(mu, nu)] = (plus_real.get(mu, nu) - minus_real.get(mu, nu)) / (2.0 * h);
                imaginary[(mu, nu)] = (plus_imag.get(mu, nu) - minus_imag.get(mu, nu)) / (2.0 * h);
            }
        }
        (real, imaginary)
    }

    /// `max |FD − i X|`（`X` 是实的，故 `iX` 只有虚部）、`max|FD|`、`max|Re FD|`。
    fn complex_deviation(fd_real: &Matrix, fd_imag: &Matrix, x: &Matrix) -> (f64, f64, f64) {
        let n = x.nrows();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut real_part = 0.0_f64;
        for mu in 0..n {
            for nu in 0..n {
                let dr = fd_real.get(mu, nu);
                let di = fd_imag.get(mu, nu) - x.get(mu, nu);
                worst = worst.max((dr * dr + di * di).sqrt());
                scale =
                    scale.max((fd_real.get(mu, nu).powi(2) + fd_imag.get(mu, nu).powi(2)).sqrt());
                real_part = real_part.max(dr.abs());
            }
        }
        (worst, scale, real_part)
    }

    #[test]
    fn r3_the_kinetic_half_matches_a_finite_difference_of_the_closed_form() {
        let molecule = displaced_water();
        let bases = ["sto-3g", "def2-svp"];
        let origins = [[0.0, 0.0, 0.0], [0.37, -0.21, 0.53]];
        let step = 5.0e-5_f64;
        for name in bases {
            let basis = basis(name, &molecule);
            let n = basis.n_functions();
            // 先确认闭合形式在 B = 0 就是普通动能矩阵（否则后面的差分没有意义）。
            let (zero_real, zero_imag) = giao_kinetic_spherical(&basis, [0.0; 3], [0.0; 3]);
            let reference = one_electron(&basis, &molecule).expect("one electron");
            let mut zero_deviation = 0.0_f64;
            let mut zero_imaginary = 0.0_f64;
            for mu in 0..n {
                for nu in 0..n {
                    zero_deviation = zero_deviation
                        .max((zero_real.get(mu, nu) - reference.kinetic.get(mu, nu)).abs());
                    zero_imaginary = zero_imaginary.max(zero_imag.get(mu, nu).abs());
                }
            }
            println!(
                "R3 {name}: B=0 闭合形式 vs 普通动能最差 {zero_deviation:.3e}，\
                 max|Im| = {zero_imaginary:.3e}"
            );
            assert!(
                zero_deviation < 1.0e-10,
                "R3 {name}: B=0 闭合形式与普通动能差 {zero_deviation:.3e}"
            );
            assert!(
                zero_imaginary < 1.0e-12,
                "R3 {name}: B=0 时闭合形式有虚部 {zero_imaginary:.3e}"
            );

            let kinetic_moment = kinetic_moment_integrals(&basis).expect("kinetic moment");
            for origin in origins {
                let analytic =
                    assemble_field_derivative(&basis, &kinetic_moment, origin).expect("X^T");
                for (axis, analytic_axis) in analytic.iter().enumerate() {
                    let (fd_real, fd_imag) = kinetic_derivative_fd(&basis, origin, axis, step);
                    let (half_real, half_imag) =
                        kinetic_derivative_fd(&basis, origin, axis, 0.5 * step);
                    let (worst, scale, real_part) =
                        complex_deviation(&fd_real, &fd_imag, analytic_axis);
                    let (worst_half, _, _) =
                        complex_deviation(&half_real, &half_imag, analytic_axis);
                    let mut convergence = 0.0_f64;
                    for mu in 0..n {
                        for nu in 0..n {
                            let dr = fd_real.get(mu, nu) - half_real.get(mu, nu);
                            let di = fd_imag.get(mu, nu) - half_imag.get(mu, nu);
                            convergence = convergence.max((dr * dr + di * di).sqrt());
                        }
                    }
                    println!(
                        "R3 {name} O={origin:?} k={axis}: max|dH^T/dB| = {scale:.3e}，\
                         max|FD − iX| = {worst:.3e}（h={step:.0e}）、{worst_half:.3e}（h/2）；\
                         max|Re FD| = {real_part:.3e}；h vs h/2 差 {convergence:.3e}"
                    );
                    assert!(
                        scale > 1.0e-2,
                        "R3 {name}: max|dH^T/dB| 只有 {scale:.3e}，尺子是空的"
                    );
                    assert!(
                        worst < 1.0e-8,
                        "R3 {name} k={axis}: 闭合形式差分与 iX 差 {worst:.3e}（判据 1e-8）"
                    );
                    assert!(
                        real_part < 1.0e-8,
                        "R3 {name} k={axis}: iX 是纯虚的，但差分的实部有 {real_part:.3e}"
                    );
                    assert!(
                        convergence < 1.0e-6,
                        "R3 {name} k={axis}: h 与 h/2 不收敛（差 {convergence:.3e}）"
                    );
                    // 中心差分的截断是 O(h²)：h → h/2 时偏差应掉约 4 倍。
                    let ratio = worst / worst_half;
                    assert!(
                        (3.0..=5.0).contains(&ratio),
                        "R3 {name} k={axis}: h vs h/2 偏差比 {ratio:.3e} 不像 O(h²) 截断"
                    );
                }
            }
        }
    }
}
