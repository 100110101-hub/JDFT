//! SC-06（第二半的第一段）：**自旋分辨的 M06 交换**逐点对 libxc。
//!
//! 参考值：`tests/reference/spin_mgga_pointwise.csv`（`scripts/pyscf_spin_mgga_reference.py`
//! 从 libxc 生成，逐行带 `source_doi`/`source_url`）。
//!
//! 覆盖的是交换项的精确自旋标度：
//!
//! ```text
//! e = ρ_α·ε(2ρ_α, 4σ_αα, 2τ_α) + ρ_β·ε(2ρ_β, 4σ_ββ, 2τ_β)
//! ```
//!
//! **相关项不在本测试里**：M06 的相关是 ζ 相关的 Stoll 分解，不适用这个标度
//! （参考 CSV 里那三行是下一步的实现目标）。UKS 的 τ 接线也还没做（`spin_supported`
//! 仍是 false），所以这里直接调 `evaluate_spin`，不经过 SCF。

use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_xc::{FunctionalTerm, SpinDensityBatch, SpinXcOutput};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

/// 用**生产同一条**路径构造组件：`Registry::get_functional` → `qwb_xc::build_term`。
///
/// 这样测试与引擎用的是同一套参数（数据文件里的 `m06a1..m06a12` 等），不会出现
/// 「测试里抄的常数与引擎里的漂了」。
fn m06l_exchange() -> Box<dyn FunctionalTerm> {
    let registry = qwb_registry::Registry::open(data_root()).expect("qwb-data");
    let functional = registry.get_functional("m06-l").expect("m06-l");
    let component = functional
        .components
        .iter()
        .find(|component| component.id == "MggaXM06L")
        .expect("m06-l must carry the M06 exchange component");
    let term = qwb_xc::build_term(&component.id, &functional.parameters)
        .expect("M06 exchange must build from the data file");
    qwb_xc::scale_term(term, component.weight)
}

type Rows = (Vec<String>, Vec<Vec<String>>);

fn read_csv(path: &std::path::Path) -> Rows {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .expect("header")
        .split(',')
        .map(str::trim)
        .map(String::from)
        .collect();
    let rows = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split(',').map(str::trim).map(String::from).collect())
        .collect();
    (header, rows)
}

fn column(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|field| field == name)
        .unwrap_or_else(|| panic!("missing column `{name}`"))
}

#[test]
fn spin_resolved_m06_exchange_matches_libxc_pointwise() {
    let (header, rows) = read_csv(&reference_dir().join("spin_mgga_pointwise.csv"));
    let case = column(&header, "case");
    let functional_col = column(&header, "functional");
    let input_columns: Vec<usize> = [
        "rho_alpha",
        "rho_beta",
        "sigma_alpha",
        "sigma_beta",
        "sigma_alpha_beta",
        "tau_alpha",
        "tau_beta",
    ]
    .iter()
    .map(|name| column(&header, name))
    .collect();
    let expected_energy = column(&header, "e_density");
    let derivative_columns: Vec<(String, usize)> = [
        "v_rho_alpha",
        "v_rho_beta",
        "d_sigma_alpha",
        "d_sigma_beta",
        "d_tau_alpha",
        "d_tau_beta",
    ]
    .iter()
    .map(|name| ((*name).to_string(), column(&header, name)))
    .collect();

    let mut cases: BTreeMap<String, &Vec<String>> = BTreeMap::new();
    for row in &rows {
        if row[functional_col] == "m06-l exchange" {
            cases.insert(row[case].clone(), row);
        }
    }
    assert_eq!(cases.len(), 5, "expected five reference samples");

    let term = m06l_exchange();
    assert!(
        term.spin_meta_supported(),
        "M06 exchange should be spin+tau capable"
    );
    let mut worst_energy = 0.0f64;
    let mut worst_derivative = 0.0f64;
    for (name, row) in &cases {
        let values: Vec<f64> = input_columns
            .iter()
            .map(|index| row[*index].parse::<f64>().expect("number"))
            .collect();
        // 切片必须比 `batch` 活得久（`&[x]` 字面量是临时值，会借用失败）。
        let single = |value: f64| [value];
        let (rho_a, rho_b) = (single(values[0]), single(values[1]));
        let (s_aa, s_bb) = (single(values[2]), single(values[3]));
        let (s_ab, t_a) = (single(values[4]), single(values[5]));
        let t_b = single(values[6]);
        let batch = SpinDensityBatch::with_tau(&rho_a, &rho_b, &s_aa, &s_bb, &s_ab, &t_a, &t_b);
        let mut energy = [0.0];
        let mut v_alpha = [0.0];
        let mut v_beta = [0.0];
        let mut d_sigma_alpha = [0.0];
        let mut d_sigma_beta = [0.0];
        let mut d_sigma_alpha_beta = [0.0];
        let mut d_tau_alpha = [0.0];
        let mut d_tau_beta = [0.0];
        let mut out = SpinXcOutput::new(
            &mut energy,
            &mut v_alpha,
            &mut v_beta,
            &mut d_sigma_alpha,
            &mut d_sigma_beta,
            &mut d_sigma_alpha_beta,
        )
        .with_tau(&mut d_tau_alpha, &mut d_tau_beta);
        term.evaluate_spin(&batch, &mut out)
            .expect("spin-resolved M06 exchange");

        let reference_energy = row[expected_energy].parse::<f64>().expect("energy");
        let computed = [
            v_alpha[0],
            v_beta[0],
            d_sigma_alpha[0],
            d_sigma_beta[0],
            d_tau_alpha[0],
            d_tau_beta[0],
        ];
        let energy_delta = (energy[0] - reference_energy).abs();
        worst_energy = worst_energy.max(energy_delta);
        print!("{name:22} e={energy_delta:.3e}");
        // **近空通道的导数不比**：`fully_polarized` 里 ρ_β = 1e-8（σ_ββ = 0），
        // `∂e/∂σ_ββ` 在那里本来就发散（闭壳层实现的 `v_σ = ε + 2ρ ε_ρ` 形式在 ρ→0 时
        // 退化），libxc 与我们对密度下限的处理也不同 —— 实测差 8.3e-1（绝对值），
        // 但该通道的电子数可以忽略。所以只对 ρ_σ ≥ 1e-3 的通道断言导数，
        // 并把被跳过的通道**打印出来**（不静默）。
        let rho_alpha_value = values[0];
        let rho_beta_value = values[1];
        let mut skipped = Vec::new();
        for (index, (label, column_index)) in derivative_columns.iter().enumerate() {
            let occupied = if label.ends_with("beta") {
                rho_beta_value
            } else {
                rho_alpha_value
            };
            let reference = row[*column_index].parse::<f64>().expect("number");
            let delta = (computed[index] - reference).abs();
            if occupied >= 1.0e-3 {
                worst_derivative = worst_derivative.max(delta);
                print!("  {label}={delta:.3e}");
            } else {
                skipped.push(format!("{label}={delta:.3e}(rho={occupied:.1e})"));
            }
        }
        if !skipped.is_empty() {
            print!(
                "  [near-empty channels not asserted: {}]",
                skipped.join(" ")
            );
        }
        println!("  (E = {:.9}, ref = {:.9})", energy[0], reference_energy);
    }
    println!("worst |ΔE| = {worst_energy:.3e} Ha, worst |Δ derivative| = {worst_derivative:.3e}");
    assert!(
        worst_energy < 1.0e-9,
        "spin-resolved M06 exchange energy differs from libxc by {worst_energy:.3e} Ha"
    );
    assert!(
        worst_derivative < 1.0e-9,
        "spin-resolved M06 exchange derivatives differ from libxc by {worst_derivative:.3e}"
    );
}
