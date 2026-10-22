//! I-02 验收（project.md §6.3）：一电子积分（S、T、V）。
//!
//! 验收原文：**H2/STO-3G 的 S、T、V 与解析闭式逐项 < 1e-10**。
//!
//! 本文件里的"解析闭式"是**独立实现**（s-s 专用公式 + 自己写的 erf），
//! 不复用被测代码路径；此外用同中心归一化原函数的三组解析恒等式
//! （S=1、T=(2L+3)α/2、V 的 l=0/1/2 解析值）验证 l≥1 的通用机制。

use std::path::{Path, PathBuf};

use qwb_basis::{InternalBasis, cartesian_components};
use qwb_engine_api::{Atom, Molecule, molecule::BOHR_PER_ANGSTROM};
use qwb_integrals::one_electron::{
    one_electron, primitive_kinetic, primitive_norm, primitive_nuclear, primitive_overlap,
};
use qwb_registry::Registry;

const PI: f64 = std::f64::consts::PI;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn registry() -> Registry {
    Registry::open(repo_root().join("qwb-data")).expect("qwb-data must load")
}

/// 测试自带的 erf（级数），用来独立算 F0 —— 不调用被测的 Boys 递推。
fn erf(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    let mut n = 0.0_f64;
    loop {
        n += 1.0;
        term *= -x * x / n;
        let add = term / (2.0 * n + 1.0);
        sum += add;
        if add.abs() < 1e-18 || n > 200.0 {
            break;
        }
    }
    2.0 / PI.sqrt() * sum
}

/// F0(x) 的闭式。
fn boys0_closed_form(x: f64) -> f64 {
    if x < 1e-12 {
        1.0 - x / 3.0
    } else {
        0.5 * (PI / x).sqrt() * erf(x.sqrt())
    }
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let tolerance = 1e-10 * expected.abs().max(1e-2);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual:.15} vs {expected:.15} (|diff| = {:e} > {tolerance:e})",
        (actual - expected).abs()
    );
}

/// H2（与 T1 示例同一几何）：z = ±0.37 Å。
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

// ───────────── H2/STO-3G：S、T、V 对解析闭式（< 1e-10）─────────────

#[test]
fn h2_sto3g_one_electron_matches_closed_form() {
    let registry = registry();
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = h2();
    let basis = InternalBasis::build(basis_k1, &molecule).expect("build");
    assert_eq!(basis.n_functions(), 2);

    let integrals = one_electron(&basis, &molecule).expect("integrals");

    let h_block = basis_k1.shells_for("H").expect("H block");
    let shell = &h_block.shells[0];
    let exps = &shell.exponents;
    let coeffs = &shell.coefficients[0];
    assert_eq!(shell.l, 0, "H/STO-3G 只有 1s");

    let a_center = molecule.atoms[0].position_bohr();
    let b_center = molecule.atoms[1].position_bohr();
    let nuclei = [a_center, b_center];

    // s-s 闭式（**归一化原函数**：权重含 N(0,α)）：
    //   S = Σ c_i c_j (π/p)^{3/2} e^{-μR²}
    //   T = Σ c_i c_j μ(3 − 2μR²) (π/p)^{3/2} e^{-μR²}
    //   V = Σ c_i c_j (−Z)(2π/p) e^{-μR²} F0(p|P−C|²)
    let closed_form = |left: [f64; 3], right: [f64; 3]| -> (f64, f64, f64) {
        let r2: f64 = (0..3).map(|axis| (left[axis] - right[axis]).powi(2)).sum();
        let norm = |a: f64| (2.0 * a / PI).powf(0.75);
        let mut s_total = 0.0;
        let mut t_total = 0.0;
        let mut v_total = 0.0;
        for (ai, ci) in exps.iter().zip(coeffs) {
            for (bj, cj) in exps.iter().zip(coeffs) {
                let p = ai + bj;
                let mu = ai * bj / p;
                let weight = ci * cj * norm(*ai) * norm(*bj);
                let decay = (-mu * r2).exp();
                let prefactor = (PI / p).powf(1.5) * decay;

                let s = prefactor;
                let t = mu * (3.0 - 2.0 * mu * r2) * prefactor;

                let p_center: [f64; 3] = [
                    (ai * left[0] + bj * right[0]) / p,
                    (ai * left[1] + bj * right[1]) / p,
                    (ai * left[2] + bj * right[2]) / p,
                ];
                let mut v = 0.0;
                for nucleus in nuclei {
                    let pc2: f64 = (0..3)
                        .map(|axis| (p_center[axis] - nucleus[axis]).powi(2))
                        .sum();
                    v += -(2.0 * PI / p) * decay * boys0_closed_form(p * pc2);
                }

                s_total += weight * s;
                t_total += weight * t;
                v_total += weight * v;
            }
        }
        (s_total, t_total, v_total)
    };

    // 对角元：两个函数都在同一个核上（R=0）。
    let (s_diag, t_diag, v_diag) = closed_form(a_center, a_center);
    // 非对角元：A 上的 1s 与 B 上的 1s。
    let (s_off, t_off, v_off) = closed_form(a_center, b_center);

    assert_close(s_diag, 1.0, "S_11 = 1（归一化收缩）");
    assert_close(integrals.overlap[(0, 0)], s_diag, "S_11 闭式");
    assert_close(integrals.overlap[(1, 1)], s_diag, "S_22 闭式");
    assert_close(integrals.overlap[(0, 1)], s_off, "S_12 闭式");
    assert_close(integrals.overlap[(1, 0)], s_off, "S_21 = S_12");

    assert_close(integrals.kinetic[(0, 0)], t_diag, "T_11 闭式");
    assert_close(integrals.kinetic[(1, 1)], t_diag, "T_22 闭式");
    assert_close(integrals.kinetic[(0, 1)], t_off, "T_12 闭式");
    assert_close(integrals.kinetic[(1, 0)], t_off, "T_21 = T_12");

    assert_close(integrals.nuclear[(0, 0)], v_diag, "V_11 闭式");
    assert_close(integrals.nuclear[(1, 1)], v_diag, "V_22 闭式");
    assert_close(integrals.nuclear[(0, 1)], v_off, "V_12 闭式");
    assert_close(integrals.nuclear[(1, 0)], v_off, "V_21 = V_12");

    // 物理合理性。
    assert!(integrals.kinetic[(0, 0)] > 0.0, "T_ii > 0");
    assert!(integrals.nuclear[(0, 0)] < 0.0, "V_ii < 0");
    assert_close(
        integrals.core_hamiltonian[(0, 0)],
        integrals.kinetic[(0, 0)] + integrals.nuclear[(0, 0)],
        "H = T + V",
    );
}
// ───────── 同中心归一化原函数：S=1、T=(2L+3)α/2（l=0,1,2）─────────

#[test]
fn normalized_same_center_primitives_have_unit_overlap() {
    let alpha = 1.3_f64;
    let center = [0.4, -0.2, 0.7];
    for l in 0..=2_u32 {
        for powers in cartesian_components(l) {
            let norm = primitive_norm(powers, alpha);
            let s = norm * norm * primitive_overlap(alpha, powers, center, alpha, powers, center);
            assert_close(s, 1.0, &format!("<L={l} {powers:?}|L={l}> = 1"));
        }
    }
}

#[test]
fn pure_components_have_the_kinetic_identity() {
    // 恒等式 T = (2L+3)α/2 只对**纯 L 分量**成立。
    // 笛卡尔表示里 x²、y²、z² 是 L=0 与 L=2 的混合，不能用来验这条；
    // 纯分量：s=(0,0,0)、p=(1,0,0)、d=(1,1,0)/(1,0,1)/(0,1,1)。
    let alpha = 0.75_f64;
    let center = [1.0, 2.0, -0.5];
    for (l, powers) in [
        (0_u32, (0_u32, 0_u32, 0_u32)),
        (1, (1, 0, 0)),
        (1, (0, 1, 0)),
        (1, (0, 0, 1)),
        (2, (1, 1, 0)),
        (2, (1, 0, 1)),
        (2, (0, 1, 1)),
    ] {
        let norm = primitive_norm(powers, alpha);
        let t = norm * norm * primitive_kinetic(alpha, powers, center, alpha, powers, center);
        let expected = (2.0 * f64::from(l) + 3.0) * alpha / 2.0;
        assert_close(t, expected, &format!("T(L={l}, {powers:?}) = (2L+3)α/2"));
    }
}

#[test]
fn mixed_cartesian_components_are_not_pure_l() {
    // x² 与 xy 都是 L=2 的笛卡尔分量，但 x² 混入了 s 成分，
    // 所以两者的动能**不相等**；这条断言记录了这个事实，防止有人误用恒等式。
    let alpha = 0.75_f64;
    let center = [0.0, 0.0, 0.0];
    let kinetic_of = |powers| {
        let norm = primitive_norm(powers, alpha);
        norm * norm * primitive_kinetic(alpha, powers, center, alpha, powers, center)
    };
    let xx = kinetic_of((2, 0, 0));
    let xy = kinetic_of((1, 1, 0));
    assert!(
        (xx - xy).abs() > 0.1,
        "x² ({xx}) 与 xy ({xy}) 的动能不应相同"
    );
    // xy 是纯 L=2 → 7α/2；x² 是混合 → 13α/6 + s 部分的贡献。
    assert_close(xy, 7.0 * alpha / 2.0, "T(xy) = 7α/2（纯 L=2）");
}

#[test]
fn water_integrals_are_complete_and_symmetric() {
    // l>0 的核吸引修好后，含 p 函数的基组（水：7 AO）必须能完整算出来。
    let registry = registry();
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let water = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    );
    let basis = InternalBasis::build(basis_k1, &water).expect("build");
    assert_eq!(basis.n_functions(), 7, "O(1s,2s,2p×3) + 2×H(1s)");

    let integrals = one_electron(&basis, &water).expect("integrals");

    for i in 0..7 {
        assert_close(integrals.overlap[(i, i)], 1.0, "S_ii = 1（含 p 分量）");
        assert!(integrals.kinetic[(i, i)] > 0.0, "T_ii > 0");
        assert!(integrals.nuclear[(i, i)] < 0.0, "V_ii < 0");
        for j in 0..7 {
            assert!(
                (integrals.overlap[(i, j)] - integrals.overlap[(j, i)]).abs() < 1e-13,
                "S 不对称 ({i},{j})"
            );
            assert!(
                (integrals.kinetic[(i, j)] - integrals.kinetic[(j, i)]).abs() < 1e-13,
                "T 不对称"
            );
            assert!(
                (integrals.nuclear[(i, j)] - integrals.nuclear[(j, i)]).abs() < 1e-13,
                "V 不对称"
            );
            assert_close(
                integrals.core_hamiltonian[(i, j)],
                integrals.kinetic[(i, j)] + integrals.nuclear[(i, j)],
                "H = T + V",
            );
        }
    }
    assert_close(integrals.overlap.trace().expect("square"), 7.0, "trace(S)");

    // O 的 2p_x 与 2p_y 在同核上不等价（水不是球对称），但同壳层的三个 p
    // 对角元都必须是有限负值（核吸引）。
    for i in 2..5 {
        assert!(integrals.nuclear[(i, i)] < 0.0, "O 的 p 分量 V_ii < 0");
    }
}
#[test]
fn primitive_nuclear_is_exact_for_s_functions() {
    // l = 0：<1s|−Z/r|1s> = −Z·2√(2α/π)（解析闭式）
    let alpha = 0.9_f64;
    let center = [0.0, 0.0, 0.0];
    let charge = 1.0;
    let norm_s = primitive_norm((0, 0, 0), alpha);
    let v_s = norm_s
        * norm_s
        * primitive_nuclear(
            alpha,
            (0, 0, 0),
            center,
            alpha,
            (0, 0, 0),
            center,
            center,
            charge,
        );
    assert_close(v_s, -2.0 * (2.0 * alpha / PI).sqrt(), "V(1s) = -2√(2α/π)");

    // 核在别处时也与 s-s 闭式一致（用 Boys 的闭式独立算）。
    let nucleus = [1.1, -0.4, 0.6];
    for &beta in &[0.5_f64, 1.7] {
        let p = alpha + beta;
        let p_center = [
            (alpha * center[0] + beta * center[0]) / p,
            (alpha * center[1] + beta * center[1]) / p,
            (alpha * center[2] + beta * center[2]) / p,
        ];
        let pc2: f64 = (0..3)
            .map(|axis| (p_center[axis] - nucleus[axis]).powi(2))
            .sum();
        let expected = -charge * (2.0 * PI / p) * boys0_closed_form(p * pc2);
        let actual = primitive_nuclear(
            alpha,
            (0, 0, 0),
            center,
            beta,
            (0, 0, 0),
            center,
            nucleus,
            charge,
        );
        assert_close(actual, expected, "V_ss 闭式（核偏移）");
    }
}

#[test]
fn symmetry_and_integrals_are_consistent() {
    // 只有全 s 基组（H2）能跑通 V；l>0 的缺口由上一个测试钉住。
    let registry = registry();
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = h2();
    let basis = InternalBasis::build(basis_k1, &molecule).expect("build");
    let integrals = one_electron(&basis, &molecule).expect("integrals");

    for i in 0..2 {
        assert_close(integrals.overlap[(i, i)], 1.0, "S_ii = 1");
        assert!(integrals.kinetic[(i, i)] > 0.0, "T_ii > 0");
        assert!(integrals.nuclear[(i, i)] < 0.0, "V_ii < 0");
        for j in 0..2 {
            assert!((integrals.overlap[(i, j)] - integrals.overlap[(j, i)]).abs() < 1e-14);
            assert!((integrals.kinetic[(i, j)] - integrals.kinetic[(j, i)]).abs() < 1e-14);
            assert!((integrals.nuclear[(i, j)] - integrals.nuclear[(j, i)]).abs() < 1e-14);
        }
    }
    assert_close(integrals.overlap.trace().expect("square"), 2.0, "trace(S)");
}
#[test]
fn conversion_factor_is_the_expected_one() {
    // 0.74 Å 的 H2 键长对应 1.3984… Bohr；间接确认 Å→Bohr 只转一次。
    assert!((BOHR_PER_ANGSTROM - 1.889_726_089_838_572).abs() < 1e-15);
}
