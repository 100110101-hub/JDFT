//! RS-04 最后一刀的 CLI 端到端验收：`options.nmr.gauge: giao` 一路穿透到 `result.json`。
//!
//! 判据（三条，都是 `result.json` 上能直接看到的东西）：
//!
//! 1. `nmr_shieldings.method == "giao"` —— 方法名写实，别把共原规范的结果当 GIAO；
//! 2. 逐核各向同性屏蔽与 `qwb-response/tests/rs04_giao_nmr.rs` 的验收值（水/STO-3G 的
//!    O 365.672798、H 33.682096 ppm）差 < 0.1 ppm；
//! 3. `length_gauge_spread_ppm < 1` —— 这是 GIAO 的**定义性**性质（phase-2 §4.2）；
//!    非 GIAO 的两条路径在同一份几何上是 7.1 ppm（`t20`/`rs04_nmr_engine` 里都有实测）。
//!
//! 输入 YAML 写在临时目录里（仓库里没有 nmr 的示例，也不该为了测试往里加一个）；
//! `--out` 同样指到临时目录，避免往仓库里写运行产物。
use std::path::{Path, PathBuf};
use std::process::Command;

fn qwb_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_qwb"))
}

/// `qwb-core/examples`：CLI 的 `find_data_root()` 从这里往上能摸到 `../../qwb-data`。
fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwb-rs04-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

/// 水 / STO-3G / HF 的 `input.yaml`，几何与 `scripts/pyscf_giao_nmr_reference.py` 的
/// `SYSTEMS["water"]`（也是 `rs04_giao_nmr.rs::water()`）逐字相同 —— 验收值是在这份几何上
/// 钉的，换几何会差好几 ppm。
const INPUT_YAML: &str = "\
molecule:
  atoms:
    - symbol: O
      position_angstrom: [0.0, 0.0, 0.11779]
    - symbol: H
      position_angstrom: [0.0, 0.75545, -0.47116]
    - symbol: H
      position_angstrom: [0.0, -0.75545, -0.47116]
  charge: 0
  multiplicity: 1
basis_name: sto-3g
method_name: hf
grid_level: coarse
solvent: vacuum
jobtype: sp
options:
  scf_max_iter: 200
  scf_energy_threshold_hartree: 1.0e-11
  scf_density_threshold: 1.0e-9
  diis_history: 8
  threads: 0
  max_opt_steps: 100
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4
  freq_step_angstrom: 0.01
  ecp_name: null
  nmr:
    gauge: giao
    report_gauge_spread: true
";

#[test]
fn the_cli_reports_giao_shieldings_with_a_sub_ppm_gauge_spread() {
    let work = scratch("giao-input");
    let input_path = work.join("h2o_sto3g_nmr_giao.yaml");
    std::fs::write(&input_path, INPUT_YAML).expect("write input.yaml");

    let out = scratch("giao-out");
    let out_text = out.to_string_lossy().into_owned();
    let input_text = input_path.to_string_lossy().into_owned();
    let output = Command::new(qwb_bin())
        .args(["run", &input_text, "--out", &out_text])
        .current_dir(examples())
        .output()
        .expect("spawn qwb");
    let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "`qwb run` 失败：\n{log}");

    let task_dir = out.join("h2o_sto3g_nmr_giao");
    // CLI 的 `Reporter::log` 行落在 `<task>/log.txt` 里（stdout 只有 energy/task 两行）。
    let task_log = std::fs::read_to_string(task_dir.join("log.txt")).expect("log.txt");
    assert!(
        task_log.contains("gauge = giao"),
        "日志里应报出 GIAO 与它的规范变化：\n{task_log}"
    );
    let result_text = std::fs::read_to_string(task_dir.join("result.json")).expect("result.json");
    let result: serde_json::Value = serde_json::from_str(&result_text).expect("parse result.json");

    let block = result
        .get("nmr_shieldings")
        .expect("result.json 缺 nmr_shieldings 块（RS-04）");
    let method = block
        .get("method")
        .and_then(serde_json::Value::as_str)
        .expect("method");
    assert_eq!(method, "giao", "CLI 落盘的方法名必须是 giao");

    let nuclei = block
        .get("nuclei")
        .and_then(serde_json::Value::as_array)
        .expect("nuclei");
    assert_eq!(nuclei.len(), 3);
    let isotropic = |index: usize| -> f64 {
        nuclei[index]
            .get("isotropic_ppm")
            .and_then(serde_json::Value::as_f64)
            .expect("isotropic_ppm")
    };
    let spread = block
        .get("length_gauge_spread_ppm")
        .and_then(serde_json::Value::as_f64)
        .expect("length_gauge_spread_ppm");
    println!(
        "CLI（水/STO-3G，GIAO）：O = {:.6} ppm、H = {:.6}/{:.6} ppm、规范变化 = {spread:.3e} ppm",
        isotropic(0),
        isotropic(1),
        isotropic(2)
    );
    for (label, computed, expected) in [
        ("O", isotropic(0), 365.672_798_f64),
        ("H#1", isotropic(1), 33.682_096_f64),
        ("H#2", isotropic(2), 33.682_096_f64),
    ] {
        assert!(
            (computed - expected).abs() < 0.1,
            "{label} 的 GIAO 屏蔽 {computed:.6} ppm 不在 {expected:.6}±0.1 内"
        );
    }
    assert!(spread < 1.0, "GIAO 的规范变化 {spread:.3e} ppm 不小于 1");

    let _ = std::fs::remove_dir_all(&work);
    let _ = std::fs::remove_dir_all(&out);
}
