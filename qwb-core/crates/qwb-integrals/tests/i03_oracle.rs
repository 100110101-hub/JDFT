//! I-03 的独立数值路径：用**闭式 `(ss|ss)` 的中心导数**生成任意 l 的参照值。
//!
//! 原理（不需要任何 2e 递推）：笛卡尔高斯对中心的导数还是高斯乘多项式，
//!
//! ```text
//! ∂/∂A_x e^{−α|r−A|²} = 2α (x−A_x) e^{−α|r−A|²}
//! ⇒ (x−A_x)^a e^{…} = (1/(2α))^a ∂^a/∂A_x^a e^{…}
//! ```
//!
//! 于是任意 `(a b|c d)`（任意角动量）都等于**闭式 `(ss|ss)` 对中心的高阶导数**：
//!
//! ```text
//! (a b|c d) = Π_k (1/(2α_k))^{n_k} ∂^{n_k}/∂c_k^{n_k} (ss|ss)
//! ```
//!
//! 这条路径与 McMurchie–Davidson / Obara–Saika **没有任何共享代码**（只调用
//! `primitive_ssss` 这个 l=0 的闭式），所以能独立判定任何 2e 实现的公式与符号
//! 对不对。精度受差分步长限制（Richardson 外推后相对误差 ~1e-9），比生产路径的
//! 1e-12 低，但足以抓住系数/符号错误——这就是 I-03 要求的「另实现一条数值路径
//! 交叉验证」。
//!
//! 本文件现在做的是**自证**：证明这条路径自己是对的（l=0 与闭式逐位相同、满足
//! 交叉对称、刚体平移下为零、共线体系里 x-奇分量必为零）。等 OS 垂直递推落地，
//! 把待测值换成 `two_electron` 的输出即可完成交叉验证。

use qwb_integrals::two_electron::primitive_ssss;

type Center = [f64; 3];
/// 笛卡尔角动量分量 `(x, y, z)` 的幂次。
type Powers = (u32, u32, u32);

/// 一个未归一化的笛卡尔高斯原函数。
#[derive(Debug, Clone, Copy)]
struct Function {
    center: Center,
    exponent: f64,
    powers: Powers,
}

impl Function {
    fn s(center: Center, exponent: f64) -> Self {
        Self {
            center,
            exponent,
            powers: (0, 0, 0),
        }
    }

    fn with_powers(self, powers: Powers) -> Self {
        Self { powers, ..self }
    }
}

const STEP: f64 = 1.0e-2;

/// 把 `pending` 里的导数逐条施加在「四中心 s 型闭式」上。
fn evaluate(functions: &[Function; 4], pending: &[(usize, usize)]) -> f64 {
    let Some((&(which, axis), rest)) = pending.split_first() else {
        return primitive_ssss(
            functions[0].exponent,
            functions[0].center,
            functions[1].exponent,
            functions[1].center,
            functions[2].exponent,
            functions[2].center,
            functions[3].exponent,
            functions[3].center,
        );
    };
    let target = functions[which];
    let origin = target.center[axis];
    let shifted = |t: f64| {
        let mut moved = *functions;
        moved[which].center[axis] = t;
        evaluate(&moved, rest)
    };
    let central = |step: f64| (shifted(origin + step) - shifted(origin - step)) / (2.0 * step);
    let raw = (4.0 * central(STEP * 0.5) - central(STEP)) / 3.0;
    raw / (2.0 * target.exponent)
}

/// 任意角动量的参照值 `(a b|c d)`。
fn reference(functions: [Function; 4]) -> f64 {
    let mut pending = Vec::new();
    for (which, function) in functions.iter().enumerate() {
        for (axis, order) in [function.powers.0, function.powers.1, function.powers.2]
            .iter()
            .enumerate()
        {
            for _ in 0..*order {
                pending.push((which, axis));
            }
        }
    }
    evaluate(&functions, &pending)
}

fn exponents() -> [f64; 4] {
    [1.2, 0.9, 1.1, 0.8]
}

fn centers_h2_like() -> [Center; 4] {
    [
        [0.0, 0.0, -0.699_2],
        [0.0, 0.0, 0.699_2],
        [0.5, 0.1, 0.3],
        [-0.2, 0.4, 0.7],
    ]
}

fn centers_collinear() -> [Center; 4] {
    [
        [0.0, 0.0, -0.7],
        [0.0, 0.0, 0.7],
        [0.0, 0.0, -0.7],
        [0.0, 0.0, 0.7],
    ]
}

fn build(centers: [Center; 4]) -> [Function; 4] {
    let zeta = exponents();
    [
        Function::s(centers[0], zeta[0]),
        Function::s(centers[1], zeta[1]),
        Function::s(centers[2], zeta[2]),
        Function::s(centers[3], zeta[3]),
    ]
}

fn with_powers(functions: [Function; 4], powers: [Powers; 4]) -> [Function; 4] {
    [
        functions[0].with_powers(powers[0]),
        functions[1].with_powers(powers[1]),
        functions[2].with_powers(powers[2]),
        functions[3].with_powers(powers[3]),
    ]
}

#[test]
fn oracle_matches_the_closed_form_for_s_functions() {
    let functions = build(centers_h2_like());
    let closed = primitive_ssss(
        functions[0].exponent,
        functions[0].center,
        functions[1].exponent,
        functions[1].center,
        functions[2].exponent,
        functions[2].center,
        functions[3].exponent,
        functions[3].center,
    );
    let from_oracle = reference(functions);
    assert!(
        (closed - from_oracle).abs() < 1.0e-15,
        "l=0 时数值路径必须与闭式一致：{closed} vs {from_oracle}"
    );
    assert!(closed > 0.0, "(ss|ss) 必须为正");
}

#[test]
fn oracle_satisfies_the_permutational_symmetries() {
    let base = with_powers(
        build(centers_h2_like()),
        [(1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 0)],
    );
    let value = reference(base);
    assert!(value.abs() > 1.0e-6, "测试值太小，抓不住符号错误：{value}");

    // bra 内部交换：(a b|c d) = (b a|c d)，函数跟着自己的中心一起换位。
    let swapped_bra = [base[1], base[0], base[2], base[3]];
    // 两对交换：(a b|c d) = (c d|a b)。
    let swapped_pairs = [base[2], base[3], base[0], base[1]];

    for (label, permuted) in [("(ba|cd)", swapped_bra), ("(cd|ab)", swapped_pairs)] {
        let other = reference(permuted);
        assert!(
            (value - other).abs() < 1.0e-6 * value.abs().max(1.0),
            "(ab|cd) ≠ {label}：{value} vs {other}"
        );
    }
}

#[test]
fn oracle_vanishes_under_a_rigid_translation() {
    // 把四个中心一起平移不改变积分 ⇒ Σ_k ∂_{c_k,x}(ss|ss) = 0。
    // 把每个 ∂ 换成 p 型积分要除以 2α_k，所以**带权**形式和才为零：
    //     Σ_k 2α_k·(p_x,k s|ss) = 0
    // 这里踩过两次：① 只平移 bra 一对不是对称操作（P−Q 会变）；
    // ② 忘了 α 权重时和不为零——那是记错公式，不是积分有问题。
    let functions = build(centers_h2_like());
    let mut total = 0.0;
    for which in 0..4 {
        let mut moved = functions;
        moved[which].powers = (1, 0, 0);
        let term = reference(moved);
        total += 2.0 * functions[which].exponent * term;
    }
    assert!(
        total.abs() < 1.0e-6,
        "刚体平移下 Σ 2α_k·(p_k s|ss) 应为零，实得 {total}"
    );
}

#[test]
fn odd_components_vanish_for_a_collinear_symmetric_system() {
    // 共线、关于 x 偶：含奇数个 x 幂次的分量必须为零。
    let functions = build(centers_collinear());
    let px = reference(with_powers(
        functions,
        [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
    ));
    let pz = reference(with_powers(
        functions,
        [(0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
    ));
    let dxy = reference(with_powers(
        functions,
        [(1, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
    ));
    let dzz = reference(with_powers(
        functions,
        [(0, 0, 2), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
    ));

    assert!(px.abs() < 1.0e-10, "共线体系的 p_x 分量应为零，实得 {px}");
    assert!(dxy.abs() < 1.0e-10, "d_xy 应为零，实得 {dxy}");
    assert!(pz.abs() > 1.0e-4, "p_z 分量不该为零，实得 {pz}");
    assert!(dzz.abs() > 1.0e-4, "d_zz 不该为零，实得 {dzz}");
}

#[test]
fn oracle_is_isotropic_about_the_bond_axis() {
    // 共线体系里绕 z 轴旋转 90° 把 x 换成 y ⇒ (p_x p_x|ss) = (p_y p_y|ss)。
    let functions = build(centers_collinear());
    let pxx = reference(with_powers(
        functions,
        [(1, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)],
    ));
    let pyy = reference(with_powers(
        functions,
        [(0, 1, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0)],
    ));
    let pzz = reference(with_powers(
        functions,
        [(0, 0, 1), (0, 0, 1), (0, 0, 0), (0, 0, 0)],
    ));
    assert!(pxx.abs() > 1.0e-4, "测试值太小：{pxx}");
    assert!(
        (pxx - pyy).abs() < 1.0e-6 * pxx.abs(),
        "(p_xp_x|ss) 与 (p_yp_y|ss) 必须相等：{pxx} vs {pyy}"
    );
    assert!(
        (pxx - pzz).abs() > 1.0e-6,
        "p_zz 与 p_xx 不该相等（轴向不同）：{pzz} vs {pxx}"
    );
}
