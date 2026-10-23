//! I-03：任意角动量 2e 积分 vs **独立数值路径**（闭式 `(ss|ss)` 的中心导数）。
//!
//! 生产路径在 `qwb_integrals::hermite2e`（McMurchie–Davidson 的 Hermite 展开，
//! Hermite Coulomb 用「按定义的精确符号求导」得到，不依赖记忆里的递推系数）。
//!
//! 这里把两者逐分量对照：s/p/d 各组合、三组不同几何。参照值只由 `primitive_ssss`
//! 的中心导数产生，与生产路径没有任何共享代码，所以「两边一致」是真正的交叉验证。

use qwb_integrals::hermite2e::primitive_component;

#[path = "common/oracle.rs"]
mod oracle;

type Center = [f64; 3];

use oracle::{Function, Powers, reference};
// ───────────────────────────── 待验证的生产路径 ─────────────────────────────

fn production(functions: [Function; 4]) -> f64 {
    primitive_component(
        functions[0].powers,
        functions[0].center,
        functions[0].exponent,
        functions[1].powers,
        functions[1].center,
        functions[1].exponent,
        functions[2].powers,
        functions[2].center,
        functions[2].exponent,
        functions[3].powers,
        functions[3].center,
        functions[3].exponent,
    )
}

fn quartet(centers: [Center; 4], exponents: [f64; 4], powers: [Powers; 4]) -> [Function; 4] {
    [
        Function {
            center: centers[0],
            exponent: exponents[0],
            powers: powers[0],
        },
        Function {
            center: centers[1],
            exponent: exponents[1],
            powers: powers[1],
        },
        Function {
            center: centers[2],
            exponent: exponents[2],
            powers: powers[2],
        },
        Function {
            center: centers[3],
            exponent: exponents[3],
            powers: powers[3],
        },
    ]
}

fn geometries() -> Vec<(&'static str, [Center; 4], [f64; 4])> {
    vec![
        (
            "H2-like",
            [
                [0.0, 0.0, -0.699_2],
                [0.0, 0.0, 0.699_2],
                [0.5, 0.1, 0.3],
                [-0.2, 0.4, 0.7],
            ],
            [1.2, 0.9, 1.1, 0.8],
        ),
        (
            "水分子一角",
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.430_5, 1.107_4],
                [0.0, -1.430_5, 1.107_4],
                [0.6, 0.9, -0.4],
            ],
            [1.5, 0.7, 1.0, 0.55],
        ),
        (
            "共线",
            [
                [0.0, 0.0, -1.4],
                [0.0, 0.0, 1.4],
                [0.3, -0.2, 0.0],
                [-0.1, 0.5, 0.0],
            ],
            [2.0, 0.45, 1.3, 0.6],
        ),
    ]
}

/// 逐分量比对，返回最大的相对偏差。
fn check(powers: [Powers; 4], label: &str) -> f64 {
    let mut worst = 0.0_f64;
    for (name, centers, exponents) in geometries() {
        let functions = quartet(centers, exponents, powers);
        let expected = reference(functions);
        let actual = production(functions);
        let scale = expected.abs().max(1.0e-12);
        let absolute = (actual - expected).abs();
        let relative = absolute / scale;
        println!(
            "{label:<10} {name:<12} oracle {expected:>16.10e}  production {actual:>16.10e}  abs {absolute:.1e}  rel {relative:.2e}"
        );
        // 判据：绝对差 < 1e-8 **或** 相对差 < 1e-8。
        // 参照路径是「嵌套的 Richardson 差分」，阶数越高噪声越大——三阶及以上
        // 导数的绝对噪声约 1e-9（例如 pxpx|pxpx 的 0.0271 差 1.9e-9）。任何真正的
        // 公式/符号错误都是 1e-2 量级，所以这个判据依然很紧。
        assert!(
            absolute < 1.0e-8 || relative < 1.0e-8,
            "{label} / {name}: oracle {expected} vs production {actual}，绝对差 {absolute:.2e}、相对差 {relative:.2e}"
        );
        worst = worst.max(relative);
        assert!(
            expected.abs() > 1.0e-8,
            "{label} / {name} 的参照值太小（{expected}），这条用例抓不住错"
        );
    }
    worst
}

#[test]
fn ssss_matches_the_closed_form() {
    check([[0, 0, 0]; 4], "ss|ss");
}

#[test]
fn p_type_integrals_match() {
    let cases: [([Powers; 4], &str); 6] = [
        ([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], "px_s|s_s"),
        ([[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]], "pz_s|s_s"),
        ([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], "s_px|s_s"),
        ([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]], "ss|py_s"),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]], "ss|s_pz"),
        ([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]], "px_pz|py_pz"),
    ];
    for (powers, label) in cases {
        check(powers, label);
    }
}

#[test]
fn d_type_integrals_match() {
    let cases: [([Powers; 4], &str); 5] = [
        ([[2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], "dxx_s|s_s"),
        ([[1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], "dxy_s|s_s"),
        ([[0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]], "dzz_s|s_s"),
        ([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], "pxpx|pxpx"),
        ([[2, 0, 0], [0, 0, 0], [0, 0, 2], [0, 0, 0]], "dxx_s|dzz_s"),
    ];
    for (powers, label) in cases {
        check(powers, label);
    }
}

#[test]
fn extreme_exponent_ratios_match_the_oracle() {
    // 深核（O 1s 的 ζ≈130）与价层（ζ≈0.5）混在一起时，Hermite 系数里的指数比
    // 最悬殊。水/STO-3G 的 2 Ha 缺口就在这个方向，所以单独钉住。
    let cases: [([f64; 4], &str); 4] = [
        ([130.0, 130.0, 1.5, 1.5], "tight-tight"),
        ([130.0, 1.5, 1.5, 0.5], "tight-valence"),
        ([300.0, 300.0, 300.0, 300.0], "all-tight"),
        ([1.5, 0.5, 130.0, 1.5], "mixed-bra-ket"),
    ];
    let centers = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.28],
        [0.0, 0.0, -0.28],
        [0.1, 0.2, 0.0],
    ];
    for (exponents, label) in cases {
        for powers in [
            [[0, 0, 0]; 4],
            [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ] {
            let functions = quartet(centers, exponents, powers);
            let expected = reference(functions);
            let actual = production(functions);
            let absolute = (actual - expected).abs();
            let relative = absolute / expected.abs().max(1e-30);
            println!(
                "{label:<14} {powers:?} oracle {expected:>14.8e} prod {actual:>14.8e} rel {relative:.2e}"
            );
            assert!(
                absolute < 1.0e-9 || relative < 1.0e-7,
                "{label} {powers:?}: oracle {expected} vs production {actual}，相对差 {relative:.2e}"
            );
        }
    }
}

#[test]
fn production_satisfies_the_pair_symmetry() {
    // (ab|cd) = (cd|ab)，用生产路径自身验证（与独立路径无关的一条额外约束）。
    for (name, centers, exponents) in geometries() {
        let forward = quartet(
            centers,
            exponents,
            [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]],
        );
        let backward = [forward[2], forward[3], forward[0], forward[1]];
        let a = production(forward);
        let b = production(backward);
        assert!(
            (a - b).abs() < 1.0e-10 * a.abs().max(1.0),
            "{name}: (ab|cd) = {a} ≠ (cd|ab) = {b}"
        );
    }
}
