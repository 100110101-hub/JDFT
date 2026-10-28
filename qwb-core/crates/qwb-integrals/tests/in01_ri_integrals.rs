//! IN-01 的验收：RI 积分（3 中心 `(ij|P)`、2 中心库仑 `(P|Q)`）。
//!
//! # 判据（`project-phase2.md` §6.3，IN-01）
//!
//! - 张量对称性 < 1e-12：`(ij|P) = (ji|P)`、`V_PQ = V_QP`、`V` 正定；
//! - **外部参考**：`V` 的特征值、3 中心张量的奇异值与 PySCF（libcint）一致；
//! - RI 恒等式的残差 = 拟合误差，随辅助基变好而单调下降，且不为零；
//! - RI-J 能量对 H2O/def2-SVP 与直接 4 中心的差 < 1e-6 Ha
//!   （在 `qwb-scf/tests/in01_ri_fit.rs`，那里能拿到收敛密度）。
//!
//! # 为什么用"不变量"当外部参考
//!
//! 两边的球谐分量排序可能不同（我们的直角→球谐是一套 Gram–Schmidt，libcint 是
//! 另一套），逐元素对比会假失败。而
//!
//! * `V` 的特征值在 `V → U V Uᵀ`（任意正交 `U`）下不变；
//! * 3 中心张量作为 `(n_aux × n_orb²)` 矩阵的奇异常值对辅助侧与轨道侧的
//!   "壳层内正交混合"都不变
//!
//! 所以它们是**完全等价**的对照量，且一个数字就能覆盖整张张量。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::ri::{inverse_sqrt, symmetric_eigenvalues, three_center, two_center_coulomb};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("ri_invariants.csv")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    )
}

/// `(system, basis, aux) -> quantity -> 升序值`
fn load_reference() -> BTreeMap<(String, String, String), BTreeMap<String, Vec<f64>>> {
    let text = std::fs::read_to_string(reference_path()).expect("ri_invariants.csv");
    let mut out: BTreeMap<(String, String, String), BTreeMap<String, Vec<f64>>> = BTreeMap::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 6,
            "row {lineno} has {} fields",
            fields.len()
        );
        let key = (
            fields[0].to_string(),
            fields[1].to_string(),
            fields[2].to_string(),
        );
        let value: f64 = fields[5].parse().expect("value");
        let entry = out.entry(key).or_default();
        entry.entry(fields[3].to_string()).or_default().push(value);
    }
    assert!(!out.is_empty(), "empty RI reference table");
    out
}

#[test]
fn three_center_is_symmetric_in_the_orbital_pair() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp").expect("def2-svp"),
        &molecule,
    )
    .expect("orbital basis");
    let aux = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp-ri").expect("def2-svp-ri"),
        &molecule,
    )
    .expect("aux basis");
    let three = three_center(&basis, &aux, &molecule).expect("3-center");
    assert_eq!(three.n_orbital, basis.n_functions());
    assert_eq!(three.n_auxiliary, aux.n_functions());
    println!(
        "H2O/def2-SVP: n_orb = {}, n_aux(def2-SVP-RI) = {}",
        three.n_orbital, three.n_auxiliary
    );

    let mut worst = 0.0_f64;
    for p in 0..three.n_auxiliary {
        for i in 0..three.n_orbital {
            for j in 0..three.n_orbital {
                worst = worst.max((three.get(i, j, p) - three.get(j, i, p)).abs());
            }
        }
    }
    println!("(ij|P) 的 i↔j 最大不对称 {worst:.2e}");
    assert!(worst < 1.0e-12, "(ij|P) 对 i,j 不对称：{worst:.2e}");
}

#[test]
fn two_center_coulomb_is_symmetric_and_positive_definite() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let aux = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp-ri").expect("def2-svp-ri"),
        &molecule,
    )
    .expect("aux basis");
    let v = two_center_coulomb(&aux, &molecule).expect("(P|Q)");
    let n = v.nrows();
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((v[(i, j)] - v[(j, i)]).abs());
        }
    }
    println!("V 的对称性最大偏差 {worst:.2e}；V_00 = {:.10}", v[(0, 0)]);
    assert!(worst < 1.0e-12, "V 不对称：{worst:.2e}");

    let values = symmetric_eigenvalues(&v).expect("eigenvalues");
    println!(
        "V 的特征值：最小 {:.6e}，最大 {:.6e}，条件数 {:.3e}",
        values[0],
        values[n - 1],
        values[n - 1] / values[0]
    );
    for (index, value) in values.iter().enumerate() {
        assert!(*value > 0.0, "V 的第 {index} 个特征值非正：{value:.3e}");
    }

    // `V^{-1/2}` 是 RI 的白化变换：`W V W` 必须在非奇异方向上恰好是单位阵。
    let w = inverse_sqrt(&v, 1.0e-10).expect("V^{-1/2}");
    let mut worst_product = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                let mut vw = 0.0;
                for l in 0..n {
                    vw += v[(k, l)] * w[(l, j)];
                }
                sum += w[(i, k)] * vw;
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            worst_product = worst_product.max((sum - expected).abs());
        }
    }
    println!("max|W V W − I| = {worst_product:.3e}");
    assert!(
        worst_product < 1.0e-8,
        "V^(−1/2) 不合格：max|W V W − I| = {worst_product:.3e}"
    );
}

/// 外部参考：`V` 的特征值与 3 中心张量的奇异值对 PySCF（libcint）。
#[test]
fn ri_invariants_match_pyscf() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let reference = load_reference();
    let molecule = water();

    let cases = [
        ("H2O", "def2-svp", "def2-svp-ri"),
        ("H2O", "sto-3g", "sto-3g"),
    ];
    for (system, basis_name, aux_name) in cases {
        let key = (
            system.to_string(),
            basis_name.to_string(),
            aux_name.to_string(),
        );
        let expected = reference
            .get(&key)
            .unwrap_or_else(|| panic!("no reference for {system}/{basis_name}/{aux_name}"));
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis(basis_name).expect(basis_name),
            &molecule,
        )
        .expect("orbital basis");
        let aux = qwb_basis::InternalBasis::build(
            registry.get_basis(aux_name).expect(aux_name),
            &molecule,
        )
        .expect("aux basis");
        let v = two_center_coulomb(&aux, &molecule).expect("(P|Q)");
        let three = three_center(&basis, &aux, &molecule).expect("3-center");

        // V 的特征值
        let mut ours = symmetric_eigenvalues(&v).expect("eigenvalues");
        let theirs = expected
            .get("V_eigenvalue")
            .expect("V_eigenvalue reference");
        assert_eq!(ours.len(), theirs.len(), "V 维数不同");
        ours.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let scale = theirs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut worst = 0.0_f64;
        for (index, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
            if b.abs() < scale * 1.0e-10 {
                continue; // 近线性相关的方向：两边都可能落到噪声里
            }
            let relative = (a - b).abs() / b.abs();
            worst = worst.max(relative);
            // 大特征值按 1e-7 判 —— 这个门限由**几何精度**决定，不是积分精度：
            // K2 的 `BOHR_PER_ANGSTROM` 与 PySCF 用的常数差 1.8e-8（见
            // docs/choices.md 五十六），它给所有"依赖核间距"的量压了一个 ~2e-8 的相对地板。
            // 小到 1e-3 量级（条件数 1e5 的尾部）在双精度下更差，按 1e-5 判。
            let limit = if b.abs() > scale * 1.0e-2 {
                1.0e-7
            } else {
                1.0e-5
            };
            assert!(
                relative < limit,
                "{system}/{aux_name}: V 的第 {index} 个特征值 {a:.12e} vs PySCF {b:.12e}（相对差 {relative:.2e}，上限 {limit:.0e}）"
            );
        }
        println!("{system}/{basis_name}+{aux_name}: V 特征值最大相对差 {worst:.2e}");

        // 3 中心的奇异值（用 (n_orb² × n_aux) 矩阵的 BᵀB 特征值开方）
        let n = three.n_orbital;
        let n_aux = three.n_auxiliary;
        let mut gram = qwb_linalg::Matrix::zeros(n_aux, n_aux);
        for p in 0..n_aux {
            for q in p..n_aux {
                let mut sum = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        sum += three.get(i, j, p) * three.get(i, j, q);
                    }
                }
                gram[(p, q)] = sum;
                gram[(q, p)] = sum;
            }
        }
        let mut singular: Vec<f64> = symmetric_eigenvalues(&gram)
            .expect("gram eigenvalues")
            .into_iter()
            .map(|value| if value > 0.0 { value.sqrt() } else { 0.0 })
            .collect();
        singular.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let theirs = expected
            .get("three_center_singular_value")
            .expect("singular value reference");
        assert_eq!(singular.len(), theirs.len(), "3 中心维数不同");
        let biggest = theirs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut worst = 0.0_f64;
        for (index, (a, b)) in singular.iter().zip(theirs.iter()).enumerate() {
            if b.abs() < biggest * 1.0e-6 {
                continue; // 拟合基的"尾部"：平方后精度不够，跳过并如实说明
            }
            let relative = (a - b).abs() / b.abs();
            worst = worst.max(relative);
            let limit = if b.abs() > biggest * 1.0e-2 {
                1.0e-7
            } else {
                1.0e-5
            };
            assert!(
                relative < limit,
                "{system}/{aux_name}: 3 中心第 {index} 个奇异值 {a:.12e} vs PySCF {b:.12e}（相对差 {relative:.2e}，上限 {limit:.0e}）"
            );
        }
        println!("{system}/{basis_name}+{aux_name}: 3 中心奇异值最大相对差 {worst:.2e}");
    }
}

/// RI 恒等式的**残差**就是拟合误差：辅助基越好残差越小，且必须非零。
#[test]
fn ri_identity_residual_is_the_fit_error() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp").expect("def2-svp"),
        &molecule,
    )
    .expect("orbital basis");
    let exact = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");

    let mut residuals = Vec::new();
    for aux_name in ["sto-3g", "def2-svp", "def2-svp-ri"] {
        let aux = qwb_basis::InternalBasis::build(
            registry.get_basis(aux_name).expect(aux_name),
            &molecule,
        )
        .expect("aux basis");
        let three = three_center(&basis, &aux, &molecule).expect("3-center");
        let v = two_center_coulomb(&aux, &molecule).expect("(P|Q)");
        let w = inverse_sqrt(&v, 1.0e-10).expect("V^{-1/2}");
        let n = three.n_orbital;
        let n_aux = three.n_auxiliary;
        let mut a = vec![0.0_f64; n_aux * n * n];
        for p in 0..n_aux {
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for q in 0..n_aux {
                        sum += w[(q, p)] * three.get(i, j, q);
                    }
                    a[(p * n + i) * n + j] = sum;
                }
            }
        }
        // 全部 24⁴ 太慢：取前 6 个 AO（O 的 5 个 s + 第 1 个 p）上的 6⁴ 块。
        let mut worst = 0.0_f64;
        for i in 0..n.min(6) {
            for j in 0..n.min(6) {
                for k in 0..n.min(6) {
                    for l in 0..n.min(6) {
                        let mut ri = 0.0;
                        for p in 0..n_aux {
                            ri += a[(p * n + i) * n + j] * a[(p * n + k) * n + l];
                        }
                        worst = worst.max((ri - exact.get(i, j, k, l)).abs());
                    }
                }
            }
        }
        println!("aux = {aux_name:>12}: n_aux = {n_aux:3}, max|RI−exact| = {worst:.3e}");
        residuals.push((aux_name, worst));
    }

    let sto3g = residuals[0].1;
    let ri = residuals[2].1;
    assert!(
        ri < sto3g * 1.0e-2,
        "def2-SVP-RI 的残差 {ri:.3e} 应比 STO-3G 的 {sto3g:.3e} 小两个量级以上"
    );
    assert!(
        residuals[1].1 < sto3g,
        "def2-SVP 当辅助基时残差应小于 STO-3G"
    );
    assert!(ri > 0.0, "RI 残差恰好为 0：辅助基不完备，说明比较写错了");
}
