//! SC-02（UKS 半）的**可判定**验收：闭壳层极限下，UKS 梯度装配必须与
//! `rks_analytic_gradient_with_xc` 逐位一致。
//!
//! 为什么用这条而不是「OH 开壳层解析 vs 数值」：OH/SVWN5/STO-3G 的 UKS SCF 在
//! DIIS/阻尼/能级移动/初猜缩放全试过的情况下 400 步不收敛（`docs/choices.md` 六十七
//! 记了实测），拿不到收敛密度就没法谈梯度。闭壳层极限（`P^α = P^β = P/2`）能把
//! UKS 装配里的三件事一起钉死：
//!
//! 1. `J` 用总密度、`K` 按自旋各取一半（闭壳层极限下 `½[K(P/2)+K(P/2)]` 要回到 `½K(P)`）；
//! 2. `W^σ = P^σ F^σ P^σ` 的和要等于闭壳层的 `½·P F P`（`pulay_w` 的约定）；
//! 3. `v_xc^σ` 在闭壳层极限下各占一半，Pulay 项才与闭壳层一致。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::analytic_gradient::uks_analytic_gradient;
use qwb_scf::rks::{RksOptions, rks_analytic_gradient_with_xc, rks_scf};
use qwb_scf::uks::{UksOptions, uks_scf};

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

#[test]
fn sc02_uks_gradient_reduces_to_the_closed_shell_path() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule).unwrap();
    let one = one_electron(&basis, &molecule).unwrap();
    let tensor = two_electron(&basis, &molecule).unwrap();
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), &molecule).unwrap();
    let (terms, hf_exchange) = qwb_engine::terms_and_hf_exchange(&registry, "svwn5").unwrap();

    // 闭壳层参照
    let mut rks_options = RksOptions::default();
    rks_options.max_iter = 400;
    rks_options.energy_threshold = 1.0e-10;
    rks_options.hf_exchange = hf_exchange;
    let closed = rks_scf(&basis, &molecule, &tensor, &grid, &terms, rks_options).unwrap();
    assert!(closed.scf_converged, "closed-shell SCF");
    let zeros = vec![[0.0_f64; 3]; molecule.atoms.len()];
    let closed_gradient = rks_analytic_gradient_with_xc(
        &basis,
        &molecule,
        &grid,
        &one.core_hamiltonian,
        &tensor,
        &closed.density,
        &terms,
        hf_exchange,
        &zeros,
    )
    .unwrap();

    // UKS：同样的分子当“等自旋”体系跑（n_α = n_β = P/2）
    let mut uks_options = UksOptions::default();
    uks_options.max_iter = 400;
    uks_options.energy_threshold = 1.0e-10;
    uks_options.hf_exchange = hf_exchange;
    let open = uks_scf(&basis, &molecule, &tensor, &grid, &terms, uks_options).unwrap();
    assert!(open.scf_converged, "UKS SCF on a closed-shell molecule");
    println!(
        "E(RKS) = {:.10}  E(UKS) = {:.10}  diff = {:.3e}",
        closed.energy_hartree,
        open.energy_hartree,
        (closed.energy_hartree - open.energy_hartree).abs()
    );
    let uks_gradient = uks_analytic_gradient(
        &basis,
        &molecule,
        &grid,
        &one.core_hamiltonian,
        &tensor,
        &open.density_alpha,
        &open.density_beta,
        &terms,
        hf_exchange,
        &zeros,
    )
    .unwrap();
    let mut worst = 0.0_f64;
    for atom in 0..closed_gradient.len() {
        for axis in 0..3 {
            let difference = (closed_gradient[atom][axis] - uks_gradient[atom][axis]).abs();
            worst = worst.max(difference);
            println!(
                "  atom {atom} axis {axis}: RKS {:>13.9}  UKS {:>13.9}  差 {difference:.2e}",
                closed_gradient[atom][axis], uks_gradient[atom][axis]
            );
        }
    }
    println!("  最大差 {worst:.3e} Ha/Bohr");
    // 两条路径解的是同一个泛函；差异只来自两条 SCF 收敛到略有差别的密度
    // （能量差 2.9e-9 Ha），所以容差取 1e-8 而不是机器精度。
    assert!(
        worst < 1.0e-8,
        "UKS gradient must reduce to the closed-shell path, worst {worst:.3e}"
    );
    let _ = Matrix::zeros(1, 1);
}
