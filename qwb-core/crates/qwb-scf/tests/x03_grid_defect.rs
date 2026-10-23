//! KNOWN DEFECT（记录用）：重原子（O 的 1s，ζ≈130）的 XC 网格求积**没有收敛**。
//!
//! 这条测试把实测数字钉在这里，不让「水的 PBE/X 能量」被当成可用数。
//! 它不是验收判据（当前行为是错的），所以 `#[ignore]`；修好后应改成断言收敛。
//!
//! 实测（H2O/STO-3G，同一几何，只换网格等级，能量单位 Ha）：
//!
//! | 网格 | PBE 总能量 | PBE E_xc | SVWN5 总能量 | SVWN5 E_xc |
//! |---|---|---|---|---|
//! | grid-coarse | −74.27601950 | −8.38914217 | −73.67328369 | −7.79747495 |
//! | grid-normal | −77.43625901 | −11.56548869 | −76.74534823 | −10.87705179 |
//! | grid-fine | −74.44047360 | −8.57632001 | −73.98070008 | −8.12047392 |
//! | grid-veryfine | −73.65429773 | −7.78782332 | （太慢，未测） | — |
//!
//! 对照：同一套网格在 **H2** 上是收敛的（PBE：−1.18117152 / −1.18118188 / −1.18118270 /
//! −1.18118271，相邻等级差依次 1.0e-5、8e-7、1e-9）。所以问题不在网格代码本身，
//! 而在「网格分辨不出重原子核心」：O 的 1s 宽度约 1/130 ≈ 0.008 bohr，径向映射的节点
//! 间距在这个尺度上不够，`∫ρ^{4/3}` 的核区尖峰被aliasing成随等级乱跳的数。
//!
//! 这与 docs/choices.md §十二记录的「cutoff 是尺度不是边界」是同一族问题：
//! 径向映射 `r = a·t²/(1−t²)`、`a = cutoff/8` 只保证外层覆盖，不保证内层分辨率。
//!
//! 影响面：**所有含重原子的 XC / PBE / LDA 绝对能量都不可引用**（HF 不经过网格，
//! 所以 H2O/STO-3G 的 HF = −74.9629 与外部参照一致是成立的）。后续要做的是换一个
//! 带核心标度的径向映射（Mura–Knowles 或 Treutler–Ahlrichs），并重新标定 T1 的参考值。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn energy(functional_name: &str, grid_name: &str, molecule: &Molecule) -> (f64, f64) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let basis = qwb_basis::InternalBasis::build(basis_k1, molecule).expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name).expect("grid"), molecule)
        .expect("grid");
    let functional = registry
        .get_functional(functional_name)
        .expect("functional");
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &terms, options).expect("scf");
    (result.energy_hartree, result.exchange_correlation_hartree)
}

#[test]
#[ignore = "KNOWN DEFECT: 水的 XC 能量随网格等级乱跳（PBE −74.276/−77.436/−74.440/−73.654，SVWN5 −73.673/−76.745/−73.981），H2 同期收敛到 1e-9。见 docs/error-budget.md §2。"]
fn water_xc_is_not_grid_converged() {
    let water = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    );
    for name in ["grid-coarse", "grid-normal", "grid-fine"] {
        for functional_name in ["pbe", "svwn5"] {
            let (total, xc) = energy(functional_name, name, &water);
            println!("{name:<12} {functional_name:<6} E = {total:.8}  E_xc = {xc:.8}");
        }
    }
}
