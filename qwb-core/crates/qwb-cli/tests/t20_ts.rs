//! RP-01 的 CLI 验收：`qwb ts` 走真实现 —— 退出码、`ts_result` 块、以及"没收敛就退出 4"。
//!
//! 完整的成功路径（NH3 伞形鞍点）在 `qwb-engine/tests/rp01_ts.rs` 里跑一次就够；这里用
//! 小 `max_steps` 钉退出码契约，不让门禁再跑一遍五分钟级的搜索。
use std::path::{Path, PathBuf};
use std::process::Command;

fn qwb_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_qwb"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
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

/// 把示例的 `max_steps` 改成 1：一步不可能收敛，用来钉"没收敛 → 退出码 4"。
fn write_short_budget(dir: &Path) -> PathBuf {
    let source = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/nh3_ts.yaml"),
    )
    .expect("read nh3_ts.yaml");
    let patched = source.replace("max_steps: 40", "max_steps: 1");
    let path = dir.join("nh3_short.yaml");
    std::fs::write(&path, patched).expect("write");
    path
}

#[test]
fn an_unconverged_ts_search_exits_4_and_says_so() {
    let dir = scratch("t20-ts-short");
    write_short_budget(&dir);
    let (code, text) = run(&dir, &["ts", "nh3_short.yaml", "--out", "out"]);
    assert_eq!(
        code, 4,
        "没收敛的 TS 搜索必须是 Unsupported（退出码 4）：\n{text}"
    );
    assert!(text.contains("did not converge"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_ts_input_exits_1_not_4() {
    let dir = scratch("t20-ts-missing");
    let (code, text) = run(&dir, &["ts", "definitely-not-here.yaml"]);
    assert_eq!(code, 1, "文件不存在是输入错误（1）：\n{text}");
    assert!(text.contains("no such file"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}
