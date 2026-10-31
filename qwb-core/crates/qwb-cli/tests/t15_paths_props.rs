//! PR-01 / K5 验收：`props` 走真实现，`ts`/`irc`/`scan` 按规矩拒绝（退出码 4 + 归哪个 WP）。
//!
//! 这条测试钉的是**退出码契约**而不是输出格式：
//! - `props` 在带 `options.excited_states` 的输入上真的跑出激发态（退出码 0）；
//! - `ts`/`irc`/`scan` 必须退出 **4** 并点名 RP-01/RP-02/RP-03 —— 未实装的功能不能假装成功；
//! - 输入文件不存在时退出 **1**（文件错与「功能没实现」必须能区分）。
use std::path::{Path, PathBuf};
use std::process::Command;

fn qwb_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_qwb"))
}

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn run(args: &[&str]) -> (i32, String) {
    let output = Command::new(qwb_bin())
        .args(args)
        .current_dir(examples())
        .output()
        .expect("spawn qwb");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.code().unwrap_or(-1), text)
}

#[test]
fn props_runs_the_excited_state_path() {
    let (code, text) = run(&["props", "h2o_tddft.yaml"]);
    assert_eq!(code, 0, "`qwb props` 应成功，实际输出：\n{text}");
    assert!(
        text.contains("state 1") && text.contains("Ha"),
        "输出里应有激发态能量与振子强度：\n{text}"
    );
}

#[test]
fn props_points_at_the_required_input_field_when_missing() {
    // h2_sto3g_svwn5.yaml 没有 options.excited_states
    let (code, text) = run(&["props", "h2_sto3g_svwn5.yaml"]);
    assert_eq!(
        code, 1,
        "缺 excited_states 时应是普通输入错误，实际：\n{text}"
    );
    assert!(
        text.contains("excited_states"),
        "提示里应点名缺少的字段：\n{text}"
    );
}

#[test]
fn unimplemented_path_commands_exit_4_and_name_the_owner() {
    for (command, owner) in [("ts", "RP-01"), ("irc", "RP-02"), ("scan", "RP-03")] {
        let (code, text) = run(&[command, "h2_sto3g_svwn5.yaml"]);
        assert_eq!(
            code, 4,
            "`qwb {command}` 未实装时必须退出 4，实际 {code}：\n{text}"
        );
        assert!(
            text.contains(owner),
            "`qwb {command}` 的输出必须点名 {owner}：\n{text}"
        );
    }
}

#[test]
fn missing_input_file_exits_1_not_4() {
    let (code, text) = run(&["ts", "definitely-not-here.yaml"]);
    assert_eq!(
        code, 1,
        "文件不存在应是输入错误(1)，而不是未实装(4)：\n{text}"
    );
    assert!(text.contains("no such file"), "应说明找不到文件：\n{text}");
}
