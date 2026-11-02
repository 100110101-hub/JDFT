//! RP-02 的 CLI 验收：`qwb irc` 跑完并把路径写到 `artifacts/irc.json`。
//!
//! 为了不把门禁拖成十分钟级，这个测试用示例的**廉价副本**（松 SCF 阈值 + 大平台阈值），
//! 判据（端点对已知极小 < 0.02 Å、单调、irc.json）由引擎测试 `rp02_irc.rs` 在正式阈值下验。
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

fn run(dir: &Path, args: &[&str]) -> (i32, String) {
    let output = Command::new(qwb_bin())
        .args(args)
        .current_dir(dir)
        .env("QWB_DATA_ROOT", repo_root().join("qwb-data"))
        .env("QWB_AUTO_FETCH", "0")
        .output()
        .expect("spawn qwb");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.code().unwrap_or(-1), text)
}

#[test]
fn irc_runs_and_writes_irc_json() {
    let dir = scratch("t21-irc");
    // 廉价副本：示例的几何 + 更松的 SCF 阈值 + 更大的平台阈值。
    let source = std::fs::read_to_string(examples().join("nh3_irc.yaml")).expect("read example");
    let cheap = source
        .replace(
            "scf_energy_threshold_hartree: 1.0e-9",
            "scf_energy_threshold_hartree: 1.0e-7",
        )
        .replace(
            "scf_density_threshold: 1.0e-7",
            "scf_density_threshold: 1.0e-5",
        )
        .replace("plateau_hartree: 1.0e-5", "plateau_hartree: 1.0e-4")
        .replace("max_steps: 60", "max_steps: 30");
    std::fs::write(dir.join("nh3_irc_cheap.yaml"), cheap).expect("write");

    let (code, text) = run(&dir, &["irc", "nh3_irc_cheap.yaml", "--out", "out"]);
    assert_eq!(code, 0, "`qwb irc` 应成功：\n{text}");
    assert!(text.contains("irc      imaginary"), "{text}");
    assert!(text.contains("monotone = true"), "{text}");

    let task = dir.join("out").join("nh3_irc_cheap");
    let irc_path = task.join("artifacts").join("irc.json");
    assert!(irc_path.is_file(), "应写出 {}", irc_path.display());
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&irc_path).expect("read")).expect("json");
    for side in ["forward", "reverse"] {
        let end = &value[side];
        assert!(end["monotone"].as_bool().unwrap_or(false), "{side} 应单调");
        assert!(end["points"].as_array().map_or(0, |points| points.len()) >= 3);
        assert!(end["minimum_energy_hartree"].as_f64().unwrap_or(f64::NAN) < -55.4);
    }
    // 同一份数据两条通道：irc.json 与 result.json 的块一致。
    let result = std::fs::read_to_string(task.join("result.json")).expect("result.json");
    let result_value: serde_json::Value = serde_json::from_str(&result).expect("json");
    assert_eq!(result_value["irc_result"], value);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_irc_input_exits_1_not_4() {
    let dir = scratch("t21-irc-missing");
    let (code, text) = run(&dir, &["irc", "definitely-not-here.yaml"]);
    assert_eq!(code, 1, "文件不存在是输入错误（1）：\n{text}");
    assert!(text.contains("no such file"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}
