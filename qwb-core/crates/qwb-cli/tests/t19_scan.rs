//! RP-03 的 CLI 验收：`qwb scan` 真的跑完并把曲线写到 `artifacts/pes.json`。
//!
//! 判据里的"`pes.json` 可被 GUI 读取"在这一层验：文件存在、是合法 JSON、点集与
//! `result.json` 的 `pes_scan` 块逐字一致（同一个数据两条通道，不能各写一套）。
use std::path::{Path, PathBuf};
use std::process::Command;

fn qwb_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_qwb"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwb-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

#[test]
fn scan_runs_and_writes_pes_json() {
    let out = scratch("t19-scan");
    let output = Command::new(qwb_bin())
        .args([
            "scan",
            "h2o_scan.yaml",
            "--out",
            out.to_string_lossy().as_ref(),
        ])
        .current_dir(examples())
        .env("QWB_DATA_ROOT", repo_root().join("qwb-data"))
        .env("QWB_AUTO_FETCH", "0")
        .output()
        .expect("spawn qwb scan");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    assert_eq!(
        output.status.code().unwrap_or(-1),
        0,
        "`qwb scan` 应成功：\n{text}"
    );
    assert!(
        text.contains("scan     distance"),
        "输出应有扫描摘要：\n{text}"
    );
    assert!(text.contains("minimum"), "输出应报极小点：\n{text}");

    let task = out.join("h2o_scan");
    let pes_path = task.join("artifacts").join("pes.json");
    assert!(pes_path.is_file(), "应写出 {}", pes_path.display());
    let pes = std::fs::read_to_string(&pes_path).expect("read pes.json");
    let value: serde_json::Value = serde_json::from_str(&pes).expect("pes.json 必须是合法 JSON");
    let points = value["points"].as_array().expect("points 数组");
    assert_eq!(points.len(), 9, "示例是 9 点扫描");
    assert_eq!(value["kind"], "distance");
    assert_eq!(value["unit"], "angstrom");
    assert_eq!(value["relaxed"], false);
    assert!(value["max_adjacent_delta_hartree"].as_f64().unwrap_or(0.0) < 0.05);

    // 同一份数据两条通道：pes.json 与 result.json 的 pes_scan 必须逐字一致。
    let result = std::fs::read_to_string(task.join("result.json")).expect("result.json");
    let result_value: serde_json::Value = serde_json::from_str(&result).expect("result.json");
    assert_eq!(
        result_value["pes_scan"], value,
        "pes.json 与 result.json 的块不一致"
    );

    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn scan_without_options_scan_exits_1_and_names_the_field() {
    let out = scratch("t19-noopt");
    // h2o_cosmo.yaml 没有 options.scan。
    let output = Command::new(qwb_bin())
        .args([
            "scan",
            "h2o_cosmo.yaml",
            "--out",
            out.to_string_lossy().as_ref(),
        ])
        .current_dir(examples())
        .env("QWB_DATA_ROOT", repo_root().join("qwb-data"))
        .env("QWB_AUTO_FETCH", "0")
        .output()
        .expect("spawn qwb scan");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    assert_eq!(
        output.status.code().unwrap_or(-1),
        1,
        "缺字段应是输入错误 1：\n{text}"
    );
    assert!(text.contains("options.scan"), "提示里应点名字段：\n{text}");
    let _ = std::fs::remove_dir_all(&out);
}
