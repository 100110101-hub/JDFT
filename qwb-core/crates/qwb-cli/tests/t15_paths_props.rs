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

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwb-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
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
    // **必须给 `--out` 临时目录**：默认 out root 是相对 CWD 的 `results/`，而这里的
    // CWD 是 `examples/` —— 不带 `--out` 就会把**仓库里提交的**示例结果
    // `examples/results/h2o_tddft/` 整份重写（曾经真的发生过：门禁每跑一次，工作区
    // 就多一份"只有最后几位不同"的脏 diff）。下面 before/after 那条断言是防复发的。
    let fixture = examples()
        .join("results")
        .join("h2o_tddft")
        .join("result.json");
    let before = std::fs::read(&fixture).expect("仓库里应有提交的示例结果 result.json");

    let out = scratch("t15-props");
    let out_text = out.to_string_lossy().into_owned();
    let (code, text) = run(&["props", "h2o_tddft.yaml", "--out", &out_text]);
    assert_eq!(code, 0, "`qwb props` 应成功，实际输出：\n{text}");
    assert!(
        text.contains("state 1") && text.contains("Ha"),
        "输出里应有激发态能量与振子强度：\n{text}"
    );
    assert!(
        out.join("h2o_tddft").join("result.json").is_file(),
        "`--out` 目录里应有结果：{out_text}"
    );

    let after = std::fs::read(&fixture).expect("仓库里应有提交的示例结果 result.json");
    assert_eq!(
        before, after,
        "`qwb props` 重写了仓库里提交的示例结果目录（测试要用 `--out` 临时目录）"
    );
    let _ = std::fs::remove_dir_all(&out);
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
