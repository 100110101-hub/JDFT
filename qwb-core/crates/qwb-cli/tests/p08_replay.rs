//! P-08：`qwb replay` 必须真重算，并与记录的能量一致（`project.md` §7）。
//!
//! 两个方向都要测：
//!   1. 未篡改的结果目录 → replay 成功、能量差 < 1e-10 Ha；
//!   2. 篡改 `result.json` 的能量 → replay 必须**拒绝**（否则“复现门禁”是摆设）。

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn data_root() -> PathBuf {
    manifest_dir().join("../../../qwb-data")
}

fn example_input() -> PathBuf {
    manifest_dir().join("../../examples/h2_sto3g_svwn5.yaml")
}

fn temp_dir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("qwb_replay_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn qwb() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qwb"));
    command.env("QWB_DATA_ROOT", data_root());
    command
}

fn run_job(out_root: &Path) -> PathBuf {
    let output = qwb()
        .arg("run")
        .arg(example_input())
        .arg("--out")
        .arg(out_root)
        .output()
        .expect("spawn qwb run");
    assert!(
        output.status.success(),
        "qwb run 失败: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    out_root.join("h2_sto3g_svwn5")
}

#[test]
fn replay_reproduces_a_recorded_result() {
    let out_root = temp_dir("clean");
    let task_dir = run_job(&out_root);
    assert!(task_dir.join("result.json").is_file());

    let output = qwb().arg("replay").arg(&task_dir).output().expect("spawn");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "qwb replay 失败: {}\n{stdout}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("reproduced"),
        "replay 输出里没有 reproduced 标记:\n{stdout}"
    );

    // 解析 delta 行，确认真的在 1e-10 以内（不能只看退出码）。
    let mut worst: Option<f64> = None;
    for line in stdout.lines() {
        if let Some(rest) = line.trim().strip_prefix("delta")
            && let Some(number) = rest.split_whitespace().next()
        {
            worst = Some(number.parse().expect("delta 数值"));
        }
    }
    let delta = worst.expect("replay 输出里没有 delta 行");
    assert!(delta < 1.0e-10, "replay 能量差 {delta:e} Ha 超过 1e-10");

    let _ = std::fs::remove_dir_all(&out_root);
}

#[test]
fn replay_rejects_a_tampered_result() {
    let out_root = temp_dir("tampered");
    let task_dir = run_job(&out_root);

    // 把记录的能量改掉 1 µHa——远大于 1e-10 的阈值。
    let result_path = task_dir.join("result.json");
    let text = std::fs::read_to_string(&result_path).expect("read result.json");
    let original = "-1.157000499131";
    assert!(
        text.contains(original),
        "result.json 里的能量写法变了，请更新本测试"
    );
    std::fs::write(&result_path, text.replace(original, "-1.157001499131"))
        .expect("write result.json");

    let output = qwb().arg("replay").arg(&task_dir).output().expect("spawn");
    assert!(
        !output.status.success(),
        "篡改后的结果目录居然 replay 成功——复现门禁失效"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("不可复现"), "拒绝理由不对: {stderr}");

    let _ = std::fs::remove_dir_all(&out_root);
}
