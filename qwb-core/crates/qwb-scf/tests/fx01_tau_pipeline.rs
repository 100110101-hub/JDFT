//! FX-01：**网格上的 τ 管线**验收（`qwb-scf` 侧）。
//!
//! 核心判据是**闭式可算**的一条。若 XC 项的**每电子**能量密度是 `ε = c·τ/ρ`
//! （即 `E_xc = c∫τ dV`），那么
//!
//! ```text
//! ∫τ = ½ Σ_μν P_μν ∫∇φ_μ·∇φ_ν = Σ_μν P_μν K_μν = Tr[P·K]
//! V_xc,μν = ½ ρ (∂ε/∂τ) ∇φ_μ·∇φ_ν = ½ c ∇φ_μ·∇φ_ν = c·K_μν
//! ```
//!
//! 其中 `K_μν = ½∫∇φ_μ·∇φ_ν` 正是 `qwb-integrals` 存的动能矩阵
//! （闭壳层约定：自旋求和密度下 `E_kin = Tr[P·K]`）。
//!
//! 这条式子同时钉住"网格上的 τ 算对了"与"矩阵元组装对了"，不需要任何外部参考值。
//! 实测锚点：`∫|∇φ₀|²` 网格上 = 1.52006、`K₀₀ = 0.76003`（正好一半），
//! `∫φ₀²` 两边都是 1.0000 —— 说明网格与权重没问题，差的是矩阵的约定
//! （第一版按 `K = ∫∇φ∇φ` 写，能量差 2 倍）。
//!
//! 另外两条：非线性的 τ 项要能跑完一次真 SCF；开壳层的 meta-GGA（τ_α/τ_β）
//! 还没做，必须**显式拒绝**。

use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::rks::{GridAoCached, RksOptions, evaluate_xc_cached, rks_scf};
use qwb_xc::{DensityBatch, FunctionalTerm, Requirements, SpinDensityBatch, SpinXcOutput, XcError};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    )
}

fn o2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.208]),
        ],
        0,
        3,
    )
}

/// 线性 τ 项（**每电子** `ε = c·τ/ρ`）—— 换算见文件头。
struct TauLinear {
    c: f64,
}

impl FunctionalTerm for TauLinear {
    fn id(&self) -> &'static str {
        "TestTauLinear"
    }

    fn requires(&self) -> Requirements {
        Requirements::meta_gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        assert!(batch.has_tau(), "TauLinear 需要 τ");
        for (index, slot) in out.iter_mut().enumerate() {
            let rho = batch.rho[index].max(1.0e-12);
            *slot = self.c * batch.tau[index] / rho;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        for index in 0..batch.len() {
            let rho = batch.rho[index].max(1.0e-12);
            d_rho[index] = -self.c * batch.tau_at(index) / (rho * rho);
            d_sigma[index] = 0.0;
        }
    }

    fn potential_derivatives_full(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
        d_tau: &mut [f64],
    ) {
        self.potential_derivatives(batch, d_rho, d_sigma);
        d_laplacian.fill(0.0);
        for (index, slot) in d_tau.iter_mut().enumerate() {
            *slot = self.c / batch.rho[index].max(1.0e-12);
        }
    }
}

/// 线性 τ 项的**自旋**版本：`e = c·(τ_α + τ_β)`（每体积）。
///
/// 闭壳层侧对应 `ε = c·τ/ρ`（即 `e = c·τ`，与 [`TauLinear`] 同一规范），所以它有闭式参考：
///
/// ```text
/// E = c(Tr[P_α·K] + Tr[P_β·K]),   V_α = V_β = c·K,   K_μν = ½∫∇φ_μ·∇φ_ν
/// ```
///
/// 这条断言同时钉住「`τ_σ` 从 `P^σ` 积累对了」与「τ 的 Fock 矩阵元 `½(∂e/∂τ_σ)∇φ∇φ` 组装对了」，
/// 而且两个自旋用**不同**的密度矩阵，α/β 通道写混了就会红。
struct TauSpinCapable {
    c: f64,
}

impl FunctionalTerm for TauSpinCapable {
    fn id(&self) -> &'static str {
        "TestTauSpinCapable"
    }

    fn requires(&self) -> Requirements {
        Requirements::meta_gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn spin_supported(&self) -> bool {
        true
    }

    fn spin_meta_supported(&self) -> bool {
        true
    }

    fn evaluate_spin(
        &self,
        batch: &SpinDensityBatch<'_>,
        out: &mut SpinXcOutput<'_>,
    ) -> Result<(), XcError> {
        let points = batch.len();
        assert!(
            !batch.tau_alpha.is_empty() && !batch.tau_beta.is_empty(),
            "TauSpinCapable 需要 τ_α/τ_β"
        );
        for index in 0..points {
            out.energy[index] = self.c * (batch.tau_alpha[index] + batch.tau_beta[index]);
            out.v_alpha[index] = 0.0;
            out.v_beta[index] = 0.0;
            out.d_sigma_alpha[index] = 0.0;
            out.d_sigma_beta[index] = 0.0;
            out.d_sigma_alpha_beta[index] = 0.0;
            out.d_tau_alpha[index] = self.c;
            out.d_tau_beta[index] = self.c;
        }
        Ok(())
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        assert!(batch.has_tau(), "TauSpinCapable 需要 τ");
        for (index, slot) in out.iter_mut().enumerate() {
            let rho = batch.rho[index].max(1.0e-12);
            *slot = self.c * batch.tau[index] / rho;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        for index in 0..batch.len() {
            let rho = batch.rho[index].max(1.0e-12);
            d_rho[index] = -self.c * batch.tau_at(index) / (rho * rho);
            d_sigma[index] = 0.0;
        }
    }

    fn potential_derivatives_full(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
        d_tau: &mut [f64],
    ) {
        self.potential_derivatives(batch, d_rho, d_sigma);
        d_laplacian.fill(0.0);
        for (index, slot) in d_tau.iter_mut().enumerate() {
            *slot = self.c / batch.rho[index].max(1.0e-12);
        }
    }
}

/// 非线性 τ 项（只为跑通一次真 SCF）：每电子 `ε = −a·τ²/(ρ + ρ₀)`。
struct TauQuadratic {
    a: f64,
}

impl FunctionalTerm for TauQuadratic {
    fn id(&self) -> &'static str {
        "TestTauQuadratic"
    }

    fn requires(&self) -> Requirements {
        Requirements::meta_gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        assert!(batch.has_tau(), "TauQuadratic 需要 τ");
        for (index, slot) in out.iter_mut().enumerate() {
            let rho = batch.rho[index].max(1.0e-12) + 0.1;
            *slot = -self.a * batch.tau[index] * batch.tau[index] / rho;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        for index in 0..batch.len() {
            let rho = batch.rho[index].max(1.0e-12) + 0.1;
            let tau = batch.tau_at(index);
            d_rho[index] = self.a * tau * tau / (rho * rho);
            d_sigma[index] = 0.0;
        }
    }

    fn potential_derivatives_full(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
        d_tau: &mut [f64],
    ) {
        self.potential_derivatives(batch, d_rho, d_sigma);
        d_laplacian.fill(0.0);
        for (index, slot) in d_tau.iter_mut().enumerate() {
            let rho = batch.rho[index].max(1.0e-12) + 0.1;
            *slot = -2.0 * self.a * batch.tau_at(index) / rho;
        }
    }
}

#[test]
fn tau_linear_term_reproduces_the_kinetic_matrix() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    let basis = qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule)
        .expect("basis");
    // veryfine：这条断言比的是**精确积分**，网格的求积误差会直接进来
    // （grid-normal 上 ∫|∇φ₀|² 的误差 ~1e-7，相对 0.576 就是 5e-9）。
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-veryfine").unwrap(), &molecule)
        .expect("grid");
    let n = basis.n_functions();

    // 非平凡但合法的密度矩阵（对称、量级正常）
    let mut density = Matrix::zeros(n, n);
    density[(0, 0)] = 1.1;
    density[(1, 1)] = 0.9;
    density[(0, 1)] = 0.08;
    density[(1, 0)] = 0.08;

    let c = 0.37;
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![Box::new(TauLinear { c })];
    let cache = GridAoCached::build(&basis, &grid, false, false, true).expect("cache");
    let (xc_energy, v_xc) =
        evaluate_xc_cached(&basis, &grid, &cache, &density, &terms).expect("xc");

    // 参考矩阵：K_μν = ½∫∇φ_μ·∇φ_ν
    let integrals = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let kinetic = &integrals.kinetic;
    let mut expected_energy = 0.0;
    for i in 0..n {
        for j in 0..n {
            expected_energy += density[(i, j)] * kinetic[(i, j)];
        }
    }
    expected_energy *= c;
    assert!(
        (xc_energy - expected_energy).abs() < 1.0e-9,
        "E_xc = {xc_energy:.12}，闭式 c·Tr[P·K] = {expected_energy:.12}"
    );

    // V_xc = c·K 逐元素
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((v_xc[(i, j)] - c * kinetic[(i, j)]).abs());
        }
    }
    assert!(
        worst < 1.0e-9,
        "V_xc 与 c·K 的最大偏差 {worst:.3e}（τ 通道的矩阵元组装不对）"
    );
    let scale = (0..n).map(|i| v_xc[(i, i)].abs()).fold(0.0_f64, f64::max);
    assert!(
        scale > 1.0e-3,
        "V_xc 的对角量级太小（{scale:.3e}），τ 通道没进去"
    );
}

#[test]
fn nonlinear_tau_term_converges_a_real_scf() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    let basis = qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule)
        .expect("basis");
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), &molecule)
        .expect("grid");
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![Box::new(TauQuadratic { a: 0.05 })];
    let mut options = RksOptions::default();
    options.max_iter = 200;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options)
        .expect("非线性 τ 项也应当能跑完 SCF");
    println!(
        "H2 + TestTauQuadratic: E = {:.10} Ha, E_xc = {:.10} Ha, iters = {}",
        result.energy_hartree, result.exchange_correlation_hartree, result.iterations
    );
    assert!(result.scf_converged);
    assert!(result.energy_hartree.is_finite());
    assert!(
        result.exchange_correlation_hartree < 0.0,
        "e = −a τ²/(ρ+ρ₀) 必须给负的 E_xc，实测 {}",
        result.exchange_correlation_hartree
    );
}

#[test]
fn open_shell_tau_channel_matches_the_kinetic_matrix() {
    // 自旋版 τ 管线的闭式判据（换算见 `TauSpinCapable` 的注释）：
    //   E = c(Tr[P_α·K] + Tr[P_β·K]),  V_α = V_β = c·K
    // 两个自旋故意用**不同**的密度矩阵 —— α/β 写混时这条会红，能量差值也能看出来。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = o2();
    let basis = qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule)
        .expect("basis");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-veryfine").unwrap(), &molecule)
        .expect("grid");
    let n = basis.n_functions();

    let mut density_alpha = Matrix::zeros(n, n);
    density_alpha[(0, 0)] = 1.05;
    density_alpha[(1, 1)] = 0.95;
    density_alpha[(0, 1)] = 0.05;
    density_alpha[(1, 0)] = 0.05;
    let mut density_beta = Matrix::zeros(n, n);
    density_beta[(0, 0)] = 0.85;
    density_beta[(2, 2)] = 0.55;
    density_beta[(0, 2)] = -0.03;
    density_beta[(2, 0)] = -0.03;

    let c = 0.31;
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![Box::new(TauSpinCapable { c })];
    let cache = GridAoCached::build(&basis, &grid, true, false, true).expect("cache");
    let evaluation = qwb_scf::uks::evaluate_xc_spin_cached(
        &basis,
        &grid,
        &cache,
        &density_alpha,
        &density_beta,
        &terms,
    )
    .expect("spin-resolved τ evaluation");

    let kinetic = &qwb_integrals::one_electron::one_electron(&basis, &molecule)
        .expect("1e")
        .kinetic;
    let mut trace_alpha = 0.0;
    let mut trace_beta = 0.0;
    for i in 0..n {
        for j in 0..n {
            trace_alpha += density_alpha[(i, j)] * kinetic[(i, j)];
            trace_beta += density_beta[(i, j)] * kinetic[(i, j)];
        }
    }
    let expected_energy = c * (trace_alpha + trace_beta);
    println!(
        "O2/STO-3G τ 通道: E = {:.12}, 闭式 c(Tr[P_α K]+Tr[P_β K]) = {:.12}",
        evaluation.energy_hartree, expected_energy
    );
    assert!(
        (evaluation.energy_hartree - expected_energy).abs() < 1.0e-9,
        "E = {:.12}，闭式 {:.12}",
        evaluation.energy_hartree,
        expected_energy
    );
    let mut worst_alpha = 0.0_f64;
    let mut worst_beta = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst_alpha = worst_alpha.max((evaluation.v_alpha[(i, j)] - c * kinetic[(i, j)]).abs());
            worst_beta = worst_beta.max((evaluation.v_beta[(i, j)] - c * kinetic[(i, j)]).abs());
        }
    }
    println!("V_α 与 c·K 的最大偏差 {worst_alpha:.3e}、V_β {worst_beta:.3e}");
    assert!(
        worst_alpha < 1.0e-9 && worst_beta < 1.0e-9,
        "τ 通道的矩阵元组装不对：V_α {worst_alpha:.3e}、V_β {worst_beta:.3e}"
    );
}

#[test]
fn open_shell_meta_gga_runs_a_real_uks_scf() {
    // 接线之后 UKS + τ 项必须**跑得通**（SC-06 的判据是「开壳层 + meta-GGA 可用」）：
    // 用同一个线性 τ 项在 O2 三重态上跑一次完整 SCF，断言收敛且能量有限。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = o2();
    let basis = qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule)
        .expect("basis");
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), &molecule)
        .expect("grid");
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![Box::new(TauSpinCapable { c: 0.1 })];
    let mut options = qwb_scf::uks::UksOptions::default();
    options.max_iter = 200;
    let result = qwb_scf::uks::uks_scf(&basis, &molecule, &tensor, &grid, &terms, options)
        .expect("UKS + τ 项必须跑通");
    println!(
        "O2 + TestTauSpinCapable: E = {:.10} Ha, E_xc = {:.10} Ha, iters = {}, converged = {}",
        result.energy_hartree, result.xc_energy_hartree, result.iterations, result.scf_converged
    );
    assert!(result.scf_converged, "UKS + τ 项没有收敛");
    assert!(result.energy_hartree.is_finite());
    // e = c(τ_α+τ_β) > 0 ⇒ E_xc > 0；为 0 说明 τ 通道根本没进去。
    assert!(
        result.xc_energy_hartree > 0.0,
        "线性 τ 项必须给正的 E_xc，实测 {}",
        result.xc_energy_hartree
    );
}

#[test]
fn open_shell_meta_gga_without_a_spin_implementation_is_rejected() {
    // 拒绝点从「所有 meta 项」收到「**没有自旋+τ 实现**的 meta 项」：
    // 只实现了闭壳层 `energy_density`/`potential_derivatives_full` 的项（`TauLinear`）
    // 仍然必须被显式拒绝，而不是让 UKS 喂一个 τ 进去拿到静默的假势。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = o2();
    let basis = qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule)
        .expect("basis");
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), &molecule)
        .expect("grid");
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![Box::new(TauLinear { c: 0.1 })];
    let error = qwb_scf::uks::uks_scf(
        &basis,
        &molecule,
        &tensor,
        &grid,
        &terms,
        qwb_scf::uks::UksOptions::default(),
    )
    .expect_err("没有自旋实现 + τ 的项必须被拒");
    let message = error.to_string();
    assert!(
        message.contains("spin-resolved implementation"),
        "错误信息要点名缺的是自旋实现：{message}"
    );
}
