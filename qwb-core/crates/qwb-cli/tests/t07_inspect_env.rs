//! T7 验收（第一条可执行判据）：**结果目录缺 `env.json` 时 `qwb inspect` 必须退出码 3**。
//!
//! K5 把"缺环境记录"归到 exit code 3（数据/复现门禁类错误）。这条测试故意把 `env.json`
//! 删掉再 inspect —— 只看退出码，不看输出文字，这样即使将来文案改了测试仍然有效。
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
    let path = std::env::temp_dir().join(format!("qwb_inspect_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("temp dir");
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
fn inspect_exits_three_when_env_json_is_missing() {
    let out_root = temp_dir("clean");
    let task_dir = run_job(&out_root);
    assert!(task_dir.join("env.json").is_file(), "run 应写出 env.json");

    // ① 完整的结果目录：inspect 成功
    let ok = qwb().arg("inspect").arg(&task_dir).output().expect("spawn");
    assert!(
        ok.status.success(),
        "完整结果目录的 inspect 应成功: {}",
        String::from_utf8_lossy(&ok.stderr)
    );

    // ② 删掉 env.json：inspect 必须以 3 退出（K5 的"数据/复现门禁"类错误）
    std::fs::remove_file(task_dir.join("env.json")).expect("remove env.json");
    let missing = qwb().arg("inspect").arg(&task_dir).output().expect("spawn");
    assert_eq!(
        missing.status.code(),
        Some(3),
        "缺 env.json 的 inspect 退出码应为 3，实际 {:?}\nstderr: {}",
        missing.status.code(),
        String::from_utf8_lossy(&missing.stderr)
    );

    let _ = std::fs::remove_dir_all(&out_root);
}
