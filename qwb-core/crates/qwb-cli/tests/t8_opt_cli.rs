//! T8：`qwb run` 跑一个 **opt** 作业，产物必须是真 K3 结果目录 + 优化产物。
//!
//! 判据（T8 第一条："选 PBE/def2-SVP/opt → 提交 → Job Monitor → 3D 结果"）在这里
//! 只盖命令行这一半（GUI 那一半在 `qwb-studio/tests/test_gui.py`）：
//!
//! * 退出码 0（K5）；
//! * `<out>/<task_id>/` 通过 `qwb_io::layout::ResultDir` 的 K3 检查（含 `env.json`）；
//! * `result.json` 带终点梯度（`gradient_hartree_per_bohr`，形状 3N×1）与轨道；
//! * `artifacts/optimized.xyz` 存在，且它写的几何**不是**输入几何（真的动了）；
//! * `artifacts/optimization.json` 里 `used_analytic_gradient = true`、`converged = true`，
//!   轨迹至少两帧。
//!
//! 用 STO-3G（2 个基函数）跑，整条命令在本机约 5 秒；def2-SVP 那一档的实测数字记在
//! `docs/status.yaml` 的 T8 里（分钟级，不进 CI）。
use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn data_root() -> PathBuf {
    manifest_dir().join("../../../qwb-data")
}

fn example_input() -> PathBuf {
    manifest_dir().join("../../examples/h2_sto3g_pbe_opt.yaml")
}

fn temp_dir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("qwb_opt_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("temp dir");
    path
}

fn read_xyz(path: &Path) -> (usize, Vec<(String, [f64; 3])>) {
    let text = std::fs::read_to_string(path).expect("optimized.xyz");
    let mut lines = text.lines();
    let count: usize = lines
        .next()
        .expect("natoms line")
        .trim()
        .parse()
        .expect("natoms");
    let _comment = lines.next().expect("comment line");
    let mut atoms = Vec::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 4, "xyz 行格式：{line}");
        atoms.push((
            fields[0].to_string(),
            [
                fields[1].parse().expect("x"),
                fields[2].parse().expect("y"),
                fields[3].parse().expect("z"),
            ],
        ));
    }
    (count, atoms)
}

#[test]
fn opt_job_writes_a_k3_result_directory_with_optimisation_artifacts() {
    let out_root = temp_dir("cli");
    let output = Command::new(env!("CARGO_BIN_EXE_qwb"))
        .env("QWB_DATA_ROOT", data_root())
        .arg("run")
        .arg(example_input())
        .arg("--out")
        .arg(&out_root)
        .output()
        .expect("spawn qwb run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "opt 作业应成功\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("opt     "),
        "stdout 要报告优化步数/最大力，实际：\n{stdout}"
    );

    // task id = 输入文件的 stem（K4 约定）
    let task_dir = out_root.join("h2_sto3g_pbe_opt");
    let directory = qwb_io::layout::ResultDir::new(&task_dir);
    directory
        .check()
        .unwrap_or_else(|err| panic!("{} 不是完整的 K3 结果目录：{err}", task_dir.display()));
    let env = directory.read_env().expect("env.json");
    assert!(!env.rustc_version.trim().is_empty());
    assert!(env.threads >= 1);

    let result: qwb_engine_api::JobResult = serde_json::from_str(
        &std::fs::read_to_string(task_dir.join("result.json")).expect("result.json"),
    )
    .expect("result.json 必须是 K2 的 JobResult");
    assert!(result.scf_converged);
    let gradient = result
        .gradient_hartree_per_bohr
        .as_ref()
        .expect("opt 结果必须带终点梯度");
    assert_eq!(gradient.nrows(), 6, "2 个原子 ⇒ 3N = 6 行");
    assert_eq!(gradient.ncols(), 1);
    let max_force = gradient
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        max_force < 4.5e-4,
        "终点最大力 {max_force:.3e} Ha/Bohr 超过 T3 判据"
    );
    assert!(result.mo_coeffs.is_some(), "opt 结果必须带轨道系数");

    // 优化产物：几何真的从 0.80 Å 压到了平衡附近
    let (count, atoms) = read_xyz(&task_dir.join("artifacts").join("optimized.xyz"));
    assert_eq!(count, 2);
    assert_eq!(atoms.len(), 2);
    assert_eq!(atoms[0].0, "H");
    let distance = (atoms[1].1[2] - atoms[0].1[2]).abs();
    assert!(
        (0.70..0.78).contains(&distance),
        "优化后的 H–H 键长 {distance:.4} Å 不在 0.70–0.78 之间"
    );
    assert!(
        distance < 0.79,
        "输入几何是 0.80 Å，优化后 {distance:.4} Å —— 看起来根本没动"
    );

    let payload: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(task_dir.join("artifacts").join("optimization.json"))
            .expect("optimization.json"),
    )
    .expect("optimization.json");
    assert_eq!(payload["converged"], serde_json::json!(true));
    assert_eq!(
        payload["used_analytic_gradient"],
        serde_json::json!(true),
        "KS 解析梯度本该可用：{:?}",
        payload["analytic_fallback"]
    );
    let trajectory = payload["trajectory"].as_array().expect("trajectory 是数组");
    assert!(
        trajectory.len() >= 2,
        "轨迹至少要两帧（初值 + 一步），实际 {}",
        trajectory.len()
    );
}
