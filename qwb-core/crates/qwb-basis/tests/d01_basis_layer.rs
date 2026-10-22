//! I-01 验收（project.md §6.3）：基组加载器 → 内部 AO。
//!
//! ① H2/STO-3G 的 AO 数 = 2；
//! ② C/STO-3G 的 AO 数 = 5；
//! ③ `angular` 只从数据读（R-06：不允许隐式默认）；
//! ④ 归一化在加载时一次性完成，且 `normalization: l2` 的声明被**实测验证**。

use std::path::{Path, PathBuf};

use qwb_basis::{InternalBasis, cartesian_components, primitive_self_overlap};
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_registry::basis::{Angular, Normalization};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn registry() -> Registry {
    Registry::open(repo_root().join("qwb-data")).expect("qwb-data must load")
}

fn molecule(atoms: &[(&str, [f64; 3])]) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|(symbol, position)| Atom::new(*symbol, *position))
            .collect(),
        0,
        1,
    )
}

// ───────────────────── ①② AO 数（H2 = 2，C = 5）─────────────────────

#[test]
fn h2_sto3g_has_two_basis_functions() {
    let registry = registry();
    let basis = registry.get_basis("sto-3g").expect("sto-3g");
    let h2 = molecule(&[("H", [0.0, 0.0, -0.37]), ("H", [0.0, 0.0, 0.37])]);

    let internal = InternalBasis::build(basis, &h2).expect("build");
    assert_eq!(internal.n_functions(), 2, "H2/STO-3G 有 2 个 AO");
    assert_eq!(internal.n_atoms(), 2);
    assert_eq!(internal.n_functions_of_atom(0), 1, "每个 H 只有一个 1s");
    assert_eq!(internal.n_functions_of_atom(1), 1);
    assert_eq!(internal.shells().len(), 2, "两个收缩 1s 壳层");
    for shell in internal.shells() {
        assert_eq!(shell.l, 0);
        assert_eq!(shell.n_functions(), 1);
        assert_eq!(shell.exponents.len(), 3, "STO-3G 每个壳层 3 个原函数");
    }
}

#[test]
fn carbon_sto3g_has_five_basis_functions() {
    let registry = registry();
    let basis = registry.get_basis("sto-3g").expect("sto-3g");
    let carbon = molecule(&[("C", [0.0, 0.0, 0.0])]);

    let internal = InternalBasis::build(basis, &carbon).expect("build");
    assert_eq!(internal.n_functions(), 5, "C/STO-3G = 1s + 2s + 2p(3) = 5");
    assert_eq!(internal.shells().len(), 3, "1s、2s、2p 三个收缩壳层");

    let shells: Vec<_> = internal.shells_of_atom(0).collect();
    assert_eq!(
        shells.iter().map(|s| s.l).collect::<Vec<_>>(),
        vec![0, 0, 1]
    );
    assert_eq!(shells[2].n_functions(), 3, "p 壳层 3 个分量");
    // 全局 AO 下标必须连续且不重叠。
    assert_eq!(shells[0].first_function, 0);
    assert_eq!(shells[1].first_function, 1);
    assert_eq!(shells[2].first_function, 2);
}

// ─────────────────── ③ angular 只来自数据 ───────────────────

#[test]
fn angular_comes_only_from_the_data() {
    let registry = registry();
    let basis = registry.get_basis("sto-3g").expect("sto-3g");
    assert_eq!(basis.angular, Angular::Spherical);

    let water = molecule(&[
        ("O", [0.0, 0.0, 0.0]),
        ("H", [0.0, 0.757, 0.586]),
        ("H", [0.0, -0.757, 0.586]),
    ]);
    let internal = InternalBasis::build(basis, &water).expect("build");
    // O: 1s+2s+3p = 5，每个 H: 1s = 1 → 7
    assert_eq!(internal.n_functions(), 7);
    assert_eq!(
        internal.angular, basis.angular,
        "内部基组沿用数据里的 angular"
    );
}

// ────────────── ④ 归一化被实测验证（不是信任声明）──────────────

#[test]
fn every_contraction_is_normalized_to_one() {
    let registry = registry();
    let basis = registry.get_basis("sto-3g").expect("sto-3g");
    let all = molecule(&[
        ("H", [0.0, 0.0, 0.0]),
        ("C", [2.0, 0.0, 0.0]),
        ("N", [4.0, 0.0, 0.0]),
        ("O", [6.0, 0.0, 0.0]),
        ("F", [8.0, 0.0, 0.0]),
    ]);
    let internal = InternalBasis::build(basis, &all).expect("build");

    for shell in internal.shells() {
        let mut self_overlap = 0.0;
        for (i, (a, ca)) in shell.exponents.iter().zip(&shell.coefficients).enumerate() {
            for (b, cb) in shell.exponents.iter().zip(&shell.coefficients).skip(i) {
                let contribution = ca * cb * primitive_self_overlap(*a, *b, shell.l);
                self_overlap += if std::ptr::eq(a, b) {
                    contribution
                } else {
                    2.0 * contribution
                };
            }
        }
        assert!(
            (self_overlap - 1.0).abs() < 1e-9,
            "l={} 收缩壳层 <f|f> = {self_overlap}（应为 1）",
            shell.l
        );
    }
}

#[test]
fn l2_claim_is_checked_not_trusted() {
    // 拿真实数据（它声称 L2）把系数整体乘 2（→ <f|f> = 4），加载必须报 Data。
    let registry = registry();
    let mut broken = registry.get_basis("sto-3g").expect("sto-3g").clone();
    broken.normalization = Normalization::L2;
    for block in &mut broken.shells {
        for shell in &mut block.shells {
            for column in &mut shell.coefficients {
                for coefficient in column.iter_mut() {
                    *coefficient *= 2.0;
                }
            }
        }
    }
    let h = molecule(&[("H", [0.0, 0.0, 0.0])]);
    let error = InternalBasis::build(&broken, &h).expect_err("必须拒绝错误的 L2 声明");
    let message = error.to_string();
    assert!(message.contains("normalization: l2"), "{message}");
    assert!(message.contains("claims"), "{message}");

    // 同样的系数声明 `normalization: none` → 加载器自己归一，结果 <f|f> = 1。
    let mut fixed = broken;
    fixed.normalization = Normalization::None;
    let internal = InternalBasis::build(&fixed, &h).expect("none 由加载器归一");
    let shell = &internal.shells()[0];
    let mut self_overlap = 0.0;
    for (i, a) in shell.exponents.iter().enumerate() {
        for (j, b) in shell.exponents.iter().enumerate() {
            self_overlap += shell.coefficients[i]
                * shell.coefficients[j]
                * primitive_self_overlap(*a, *b, shell.l);
        }
    }
    assert!((self_overlap - 1.0).abs() < 1e-12, "{self_overlap}");
}

// ─────────────────── 其它结构检查 ───────────────────

#[test]
fn uncovered_element_is_a_data_error() {
    let registry = registry();
    let basis = registry.get_basis("sto-3g").expect("sto-3g");
    // Zn 不在 STO-3G 的五元素里。
    let zinc = molecule(&[("Zn", [0.0, 0.0, 0.0])]);
    let error = InternalBasis::build(basis, &zinc).expect_err("must reject");
    let message = error.to_string();
    assert!(message.contains("does not cover element Zn"), "{message}");
}

#[test]
fn centers_are_converted_to_bohr() {
    let registry = registry();
    let basis = registry.get_basis("sto-3g").expect("sto-3g");
    let h2 = molecule(&[("H", [0.0, 0.0, -0.37]), ("H", [0.0, 0.0, 0.37])]);
    let internal = InternalBasis::build(basis, &h2).expect("build");
    let bohr = qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
    assert!((internal.shells()[0].center_bohr[2] + 0.37 * bohr).abs() < 1e-12);
    assert!((internal.shells()[1].center_bohr[2] - 0.37 * bohr).abs() < 1e-12);
}

#[test]
fn cartesian_component_ordering_is_fixed() {
    assert_eq!(cartesian_components(0), vec![(0, 0, 0)]);
    assert_eq!(
        cartesian_components(1),
        vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    );
    assert_eq!(cartesian_components(2).len(), 6);
}
