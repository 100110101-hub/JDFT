//! 网格 XC 路径的**位级锚点**（EX-03 的 CPU 侧回归）：`evaluate_xc` 的能量与 `v_xc`
//! 必须与下面钉住的 bit pattern 完全一致。
//!
//! 为什么要有这条：EX-03 之后会动这一条路径（去掉逐点 `clone`、以及将来可能的 GPU 收缩），
//! 而附录 A 的硬要求是「**CPU 路径的位模式不得改变**」。任何"看起来等价"的重写（换求和顺序、
//! 合并循环、用 gemm 做收缩）都会改变最后几位；这条测试会在那一步立刻红，而不是等到某个
//! 能量容差测试恰好没抓到。
//!
//! 数值来源：本仓库 2026-11-08 的实现，H2O/STO-3G、grid-coarse、svwn5、单位密度
//! （`P = I`）—— 与 SCF 无关，纯粹测网格 XC 这一层。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::evaluate_xc;

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
fn the_grid_xc_bit_pattern_is_frozen() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    // SVWN5 = Slater LDA 交换 + VWN5 相关（`qwb-data/functional/svwn5.yaml` 的组件与参数）。
    // 直接构造，避免依赖 registry → terms 的私有路径。
    let functional = registry.get_functional("svwn5").expect("svwn5");
    let mut parameters = std::collections::BTreeMap::new();
    for (key, value) in functional.parameters.iter() {
        parameters.insert(key.clone(), *value);
    }
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        if component.weight.abs() <= 1e-12 {
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    // 单位密度：与 SCF 无关，只测"网格 + AO + 泛函"这一层。
    let n = basis.n_functions();
    let mut density = qwb_linalg::Matrix::zeros(n, n);
    for i in 0..n {
        density.set(i, i, 1.0);
    }
    let (energy, vxc) = evaluate_xc(&basis, &grid, &density, &terms).expect("evaluate_xc");
    println!("E_xc = {energy:.17e} (bits {:#018x})", energy.to_bits());
    println!("grid points = {}", grid.len());
    let probes = [(0usize, 0usize), (1, 1), (2, 5), (4, 4)];
    for (i, j) in probes {
        println!(
            "  v_xc[{i}][{j}] = {:.17e} (bits {:#018x})",
            vxc[(i, j)],
            vxc[(i, j)].to_bits()
        );
    }
    // 下面这些字面量是 **2026-11-08 重构之前**（逐点 clone 的版本）的 bit pattern：
    // 去掉 clone 只删拷贝、不动算术，所以必须逐位不变。任何一位变了都说明重构顺手改了运算。
    assert_eq!(
        energy.to_bits(),
        0xc013_1c6c_993b_7753_u64,
        "E_xc bit pattern changed"
    );
    let expected = [
        (0usize, 0usize, 0xc002_ea8b_f938_df99_u64),
        (1, 1, 0xbfe7_20fc_a667_4b92_u64),
        (2, 5, 0x3c6d_ea03_d3ac_96bd_u64),
        (4, 4, 0xbfe7_0df7_edb8_df77_u64),
    ];
    for (i, j, bits) in expected {
        assert_eq!(vxc[(i, j)].to_bits(), bits, "v_xc[{i}][{j}] bits changed");
    }
}
