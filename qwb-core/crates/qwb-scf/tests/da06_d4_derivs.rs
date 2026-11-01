//! DA-06 的 D4 **导数层**验收：CN / EEQ 电荷 / C6 三个中间量的解析梯度对照中心差分。
//!
//! # 差分参考
//!
//! 三步的差分都由 `d4::properties` 现算：它正是
//! `coordination_numbers` → `eeq_charges` → `reference_weights` → `c6_matrix`
//! 的复合，扰动几何后重跑一次就能拿到 `CN(±h)`、`q(±h)`、`C6(±h)`。
//!
//! # 为什么差分有两种
//!
//! 裸的中心差分 `[f(+h) − f(−h)]/(2h)` 的误差是 `O(h²) = (h²/6)·f'''`。这些量对
//! 几何的三阶导不小（`kcn/rc ≈ 7.5/2.4 ≈ 3 Bohr⁻¹`，三次方就到 ~30 Bohr⁻³），
//! 所以 `h = 1e-3` 时裸差分的**截断**余差是 CN ~ 5e-7、电荷 ~ 8e-8、C6 ~ 2e-6 量级
//! —— 这是 stencil 的性质，不是导数的误差。实测在 `h = 8e-3 … 1.25e-4` 上
//! `|analytic − FD| / h²` 恒定到 4 位有效数字（见
//! `central_difference_error_scales_as_h_squared`），即解析导数本身是精确的。
//!
//! 因此本文件对同一批中心差分做两件事：
//!
//! * `h = 1e-3` 的**裸**值：照规范打印最坏偏差，并按实测的截断地板判据；
//! * 同一 stencil 在 `h` 与 `2h` 上的 **Richardson 外推**
//!   `(4·D(h) − D(2h))/3 = f + O(h⁴)`：这才是判据 **1e-9 / 1e-8 / 1e-8**
//!   所对照的量。实测 3e-12 / 3e-12 / 8e-11，比判据还宽三个数量级。
//!
//! 判据**没有**被放宽：1e-9（CN）、1e-8（电荷）、1e-8（C6）原样保留，只是让它
//! 对照一个没有截断污染的中心差分。
//!
//! # C6 的"冻结权重"导数
//!
//! `C6_ij` 对几何**没有**直接依赖，只经由参考权重 `w_r(i)(CN_i, q_i)`。解析梯度
//! 在**未扰动**几何上取权重作为线性化点（`c6_gradient` 的 `weights` 参数），链式
//! 乘到 `∂CN/∂R` 与 `∂q/∂R` 上；差分则扰动几何、重算 `CN`/`q`、重算权重、再算
//! `C6`。两条通路都要对上，才说明这条链没有漏项。
#![deny(unsafe_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_registry::d4::D4Dispersion;
use qwb_scf::d4;

/// K2 的 Å→Bohr（与 `da06_d4_properties.rs`、参考脚本一致）。
const BOHR_PER_ANGSTROM: f64 = 1.889_726_089_838_572;

/// 规范要求的中心差分步长（Bohr）。
const STEP: f64 = 1.0e-3;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn read_csv(path: &std::path::Path) -> (Vec<String>, Vec<BTreeMap<String, String>>) {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .expect("header")
        .split(',')
        .map(str::trim)
        .map(String::from)
        .collect();
    let rows = lines
        .map(|line| {
            line.split(',')
                .map(str::trim)
                .zip(header.iter())
                .map(|(value, key)| (key.clone(), value.to_string()))
                .collect()
        })
        .collect();
    (header, rows)
}

fn field<'a>(row: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    row.get(key)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("missing column `{key}`"))
}

/// `d3_dimers.csv` 里的一个二聚体（坐标 Bohr）。
fn dimer(rows: &[BTreeMap<String, String>], name: &str) -> (Vec<String>, Vec<[f64; 3]>) {
    let mut symbols = Vec::new();
    let mut positions = Vec::new();
    for row in rows.iter().filter(|row| field(row, "dimer") == name) {
        symbols.push(field(row, "element").to_string());
        positions.push([
            field(row, "x_angstrom").parse::<f64>().expect("x") * BOHR_PER_ANGSTROM,
            field(row, "y_angstrom").parse::<f64>().expect("y") * BOHR_PER_ANGSTROM,
            field(row, "z_angstrom").parse::<f64>().expect("z") * BOHR_PER_ANGSTROM,
        ]);
    }
    assert!(!symbols.is_empty(), "{name}: no rows in d3_dimers.csv");
    (symbols, positions)
}

fn element_indices(data: &D4Dispersion, symbols: &[String]) -> Vec<usize> {
    symbols
        .iter()
        .map(|symbol| {
            data.elements
                .iter()
                .position(|candidate| candidate == symbol)
                .unwrap_or_else(|| panic!("`{symbol}` outside the d4 data range"))
        })
        .collect()
}

fn displaced(positions: &[[f64; 3]], atom: usize, axis: usize, delta: f64) -> Vec<[f64; 3]> {
    let mut shifted = positions.to_vec();
    shifted[atom][axis] += delta;
    shifted
}

/// 中心差分 `[f(+h) − f(−h)]/(2h)`。
fn central<F>(positions: &[[f64; 3]], atom: usize, axis: usize, step: f64, evaluate: &F) -> f64
where
    F: Fn(&[[f64; 3]]) -> f64,
{
    let forward = evaluate(&displaced(positions, atom, axis, step));
    let backward = evaluate(&displaced(positions, atom, axis, -step));
    (forward - backward) / (2.0 * step)
}

/// Richardson 外推的中心差分：`(4·D(h) − D(2h))/3 = f + O(h⁴)`。
fn richardson<F>(positions: &[[f64; 3]], atom: usize, axis: usize, evaluate: &F) -> f64
where
    F: Fn(&[[f64; 3]]) -> f64,
{
    let fine = central(positions, atom, axis, STEP, evaluate);
    let coarse = central(positions, atom, axis, 2.0 * STEP, evaluate);
    (4.0 * fine - coarse) / 3.0
}

/// 解析梯度的三块（CN、电荷、C6）与权重。
struct Analytic {
    cn: Vec<[f64; 3]>,
    charge: Vec<[f64; 3]>,
    c6: Vec<[f64; 3]>,
    weights: Vec<Vec<f64>>,
}

fn analytic(
    data: &D4Dispersion,
    symbols: &[String],
    positions: &[[f64; 3]],
    total_charge: f64,
) -> (Vec<usize>, Analytic) {
    let names: Vec<&str> = symbols.iter().map(String::as_str).collect();
    let indices = element_indices(data, symbols);
    let properties = d4::properties(data, &names, positions, total_charge).expect("d4 properties");
    // 未扰动几何上的权重 = 线性化点（C6 的"冻结权重"导数）。
    let weights = d4::reference_weights(
        data,
        &indices,
        &properties.charges,
        &properties.coordination_numbers,
    );
    let cn = d4::cn_gradient(data, &indices, positions);
    let charge = d4::charge_gradient(data, &indices, positions, total_charge)
        .expect("analytic charge gradient");
    let c6 = d4::c6_gradient(data, &indices, positions, total_charge, &weights)
        .expect("analytic c6 gradient");
    (
        indices,
        Analytic {
            cn,
            charge,
            c6,
            weights,
        },
    )
}

#[test]
fn d4_derivatives_match_central_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let d4_data = registry.get_d4_dispersion("d4").expect("d4 data");
    let (_, rows) = read_csv(&reference_dir().join("d3_dimers.csv"));

    let mut worst_raw = [0.0_f64; 3];
    let mut worst_richardson = [0.0_f64; 3];
    let mut worst_translation = 0.0_f64;

    for name in ["water_dimer", "methane_dimer"] {
        let (symbols, positions) = dimer(&rows, name);
        let names: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let (indices, gradient) = analytic(d4_data, &symbols, &positions, 0.0);
        let n = positions.len();
        assert_eq!(indices.len(), n);
        assert_eq!(gradient.cn.len(), n * n, "{name}: CN gradient length");
        assert_eq!(
            gradient.charge.len(),
            n * n,
            "{name}: charge gradient length"
        );
        assert_eq!(gradient.c6.len(), n * n * n, "{name}: C6 gradient length");
        assert!(!gradient.weights.is_empty());

        // 差分只经由公开的 `properties`：它内部就是这四个函数的前后串联。
        let evaluate = |shifted: &[[f64; 3]]| {
            d4::properties(d4_data, &names, shifted, 0.0).expect("d4 properties")
        };

        let mut raw = [0.0_f64; 3];
        let mut rich = [0.0_f64; 3];
        for atom in 0..n {
            for axis in 0..3 {
                // ① CN
                for i in 0..n {
                    let cn_of = |shifted: &[[f64; 3]]| evaluate(shifted).coordination_numbers[i];
                    raw[0] = raw[0].max(
                        (gradient.cn[i * n + atom][axis]
                            - central(&positions, atom, axis, STEP, &cn_of))
                        .abs(),
                    );
                    rich[0] = rich[0].max(
                        (gradient.cn[i * n + atom][axis]
                            - richardson(&positions, atom, axis, &cn_of))
                        .abs(),
                    );
                }
                // ② EEQ 电荷
                for i in 0..n {
                    let q_of = |shifted: &[[f64; 3]]| evaluate(shifted).charges[i];
                    raw[1] = raw[1].max(
                        (gradient.charge[i * n + atom][axis]
                            - central(&positions, atom, axis, STEP, &q_of))
                        .abs(),
                    );
                    rich[1] = rich[1].max(
                        (gradient.charge[i * n + atom][axis]
                            - richardson(&positions, atom, axis, &q_of))
                        .abs(),
                    );
                }
                // ③ C6（差分重算 CN/电荷/权重，解析用未扰动权重链式）
                for i in 0..n {
                    for j in 0..n {
                        let c6_of = |shifted: &[[f64; 3]]| evaluate(shifted).c6[i][j];
                        raw[2] = raw[2].max(
                            (gradient.c6[(i * n + j) * n + atom][axis]
                                - central(&positions, atom, axis, STEP, &c6_of))
                            .abs(),
                        );
                        rich[2] = rich[2].max(
                            (gradient.c6[(i * n + j) * n + atom][axis]
                                - richardson(&positions, atom, axis, &c6_of))
                            .abs(),
                        );
                    }
                }
            }
            // 平移不变性：Σ_a ∂CN_i/∂R_a = 0。
            for i in 0..n {
                let mut total = [0.0_f64; 3];
                for a in 0..n {
                    for (axis, slot) in total.iter_mut().enumerate() {
                        *slot += gradient.cn[i * n + a][axis];
                    }
                }
                for value in total {
                    worst_translation = worst_translation.max(value.abs());
                }
            }
        }
        println!(
            "{name:14} atoms = {n:2}  raw h=1e-3: |ΔCN| = {:.3e}  |Δq| = {:.3e}  |ΔC6| = {:.3e}",
            raw[0], raw[1], raw[2]
        );
        println!(
            "{name:14} atoms = {n:2}  richardson: |ΔCN| = {:.3e}  |Δq| = {:.3e}  |ΔC6| = {:.3e}",
            rich[0], rich[1], rich[2]
        );
        for quantity in 0..3 {
            worst_raw[quantity] = worst_raw[quantity].max(raw[quantity]);
            worst_richardson[quantity] = worst_richardson[quantity].max(rich[quantity]);
        }
    }

    println!(
        "worst raw h=1e-3      : CN = {:.3e}, charge = {:.3e}, C6 = {:.3e}",
        worst_raw[0], worst_raw[1], worst_raw[2]
    );
    println!(
        "worst richardson      : CN = {:.3e}, charge = {:.3e}, C6 = {:.3e}",
        worst_richardson[0], worst_richardson[1], worst_richardson[2]
    );
    println!("worst |Σ_a ∂CN_i/∂R_a| = {:.3e}", worst_translation);

    // 判据（不放宽）：CN 1e-9、电荷 1e-8、C6 1e-8。对照物是中心差分的 Richardson
    // 外推 —— 裸 h=1e-3 的余差是 O(h²) 截断，见文件头。
    assert!(
        worst_richardson[0] < 1.0e-9,
        "CN gradient differs from central differences by {:.3e}",
        worst_richardson[0]
    );
    assert!(
        worst_richardson[1] < 1.0e-8,
        "charge gradient differs from central differences by {:.3e}",
        worst_richardson[1]
    );
    assert!(
        worst_richardson[2] < 1.0e-8,
        "C6 gradient differs from central differences by {:.3e}",
        worst_richardson[2]
    );
    // 裸 h=1e-3 的判据 = 实测截断地板（CN 5.4e-7、电荷 7.7e-8、C6 2.3e-6）留一个量级余量。
    assert!(
        worst_raw[0] < 1.0e-5,
        "raw h=1e-3 CN gradient differs by {:.3e}",
        worst_raw[0]
    );
    assert!(
        worst_raw[1] < 1.0e-6,
        "raw h=1e-3 charge gradient differs by {:.3e}",
        worst_raw[1]
    );
    assert!(
        worst_raw[2] < 1.0e-5,
        "raw h=1e-3 C6 gradient differs by {:.3e}",
        worst_raw[2]
    );
    assert!(
        worst_translation < 1.0e-12,
        "Σ_a ∂CN_i/∂R_a must vanish (translation invariance), got {worst_translation:.3e}"
    );
}

/// 裸中心差分的余差按 `h²` 缩放 ⇒ 那点余差是 stencil 的截断，不是导数误差。
#[test]
fn central_difference_error_scales_as_h_squared() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let d4_data = registry.get_d4_dispersion("d4").expect("d4 data");
    let (_, rows) = read_csv(&reference_dir().join("d3_dimers.csv"));
    let (symbols, positions) = dimer(&rows, "water_dimer");
    let names: Vec<&str> = symbols.iter().map(String::as_str).collect();
    let (indices, gradient) = analytic(d4_data, &symbols, &positions, 0.0);
    let n = positions.len();
    assert!(!indices.is_empty());

    let mut ratio = [0.0_f64; 2];
    for (slot, step) in [STEP, 0.5 * STEP].into_iter().enumerate() {
        let mut worst = 0.0_f64;
        for atom in 0..n {
            for axis in 0..3 {
                for i in 0..n {
                    let cn_of = |shifted: &[[f64; 3]]| {
                        d4::properties(d4_data, &names, shifted, 0.0)
                            .expect("properties")
                            .coordination_numbers[i]
                    };
                    let numeric = central(&positions, atom, axis, step, &cn_of);
                    worst = worst.max((gradient.cn[i * n + atom][axis] - numeric).abs());
                }
            }
        }
        ratio[slot] = worst / (step * step);
        println!("CN: h = {step:.3e}  worst/h² = {:.6e}", ratio[slot]);
    }
    let relative = (ratio[0] - ratio[1]).abs() / ratio[0];
    assert!(
        relative < 1.0e-2,
        "h² scaling must hold if the residual is truncation, got {:.3e} vs {:.3e}",
        ratio[0],
        ratio[1]
    );
}

/// 约束行的两个推论：`Σ_i ∂q_i/∂R_a = 0`（逐方向）与平面几何下 `∂q/∂z = 0`。
#[test]
fn charge_gradient_respects_the_constraint() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let d4_data = registry.get_d4_dispersion("d4").expect("d4 data");
    let (_, rows) = read_csv(&reference_dir().join("d3_dimers.csv"));
    let (symbols, positions) = dimer(&rows, "water_dimer");
    let (indices, gradient) = analytic(d4_data, &symbols, &positions, 0.0);
    let n = positions.len();
    assert_eq!(indices.len(), n);

    // Σq = Q 固定 ⇒ 移动任何一个原子都不能改变电荷总和：A 的最后一行给出
    // Σ_i dq_i = 0，这只有在约束行右端取 0 时才成立。
    let mut worst_total = 0.0_f64;
    // 水二聚体所有原子的 z 都是 0 ⇒ 所有 R̂ 的 z 分量为 0 ⇒ ∂q/∂z 严格为 0。
    let mut worst_z = 0.0_f64;
    for a in 0..n {
        for axis in 0..3 {
            let mut total = 0.0_f64;
            for i in 0..n {
                total += gradient.charge[i * n + a][axis];
            }
            worst_total = worst_total.max(total.abs());
            if axis == 2 {
                for i in 0..n {
                    worst_z = worst_z.max(gradient.charge[i * n + a][2].abs());
                }
            }
        }
    }
    println!(
        "water_dimer: worst |Σ_i ∂q_i/∂R_a| = {worst_total:.3e}, worst |∂q/∂z| = {worst_z:.3e}"
    );
    assert!(
        worst_total < 1.0e-12,
        "Σ_i ∂q_i/∂R_a must vanish, got {worst_total:.3e}"
    );
    assert!(
        worst_z < 1.0e-16,
        "the planar water dimer must have no charge response in z, got {worst_z:.3e}"
    );
}
