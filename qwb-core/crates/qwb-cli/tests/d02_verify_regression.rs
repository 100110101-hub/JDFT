//! D-02 验收（project.md §6.2）：`qwb data verify` 的回归集 —— **6 坏 / 6 好**。
//!
//! 坏文件必须退出码 `3`，且报告里出现 §6.2 指定的**那一个**错误 code；
//! 好文件（`qwb-data/examples/` 里六类各一个）必须退出码 `0`。
//!
//! 用真进程跑 `qwb`（`CARGO_BIN_EXE_qwb`），因此覆盖的是 CLI 的退出码契约（K5），
//! 而不是只测库函数。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn qwb_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_qwb"))
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    manifest_dir()
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir().join("../../.."))
}

fn bad_fixture(name: &str) -> PathBuf {
    manifest_dir()
        .join("tests/fixtures/d02/bad/basis")
        .join(name)
}

fn run_verify(paths: &[PathBuf]) -> Output {
    let mut command = Command::new(qwb_bin());
    command.arg("data").arg("verify");
    for path in paths {
        command.arg(path);
    }
    command.output().expect("qwb must run")
}

fn error_codes(stdout: &str) -> Vec<String> {
    // 报告行形如: `       [g2.citation] citation must contain at least one entry (G2)`
    stdout
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let rest = trimmed.strip_prefix('[')?;
            let (code, _) = rest.split_once(']')?;
            Some(code.to_string())
        })
        .collect()
}

// ─────────────────────────── 6 个坏文件 ───────────────────────────

#[test]
fn six_bad_files_exit_3_with_the_specified_code() {
    // (文件名, §6.2 指定的 code)
    let cases = [
        ("missing_citation.yaml", "schema.deserialize"),
        ("empty_citation.yaml", "g2.citation"),
        ("missing_license.yaml", "schema.deserialize"),
        ("sha_mismatch.yaml", "sha256.mismatch"),
        ("typo_field.yaml", "schema.deserialize"),
        ("unsorted_exponents.yaml", "semantic"),
    ];

    for (name, expected_code) in cases {
        let path = bad_fixture(name);
        assert!(path.is_file(), "缺夹具 {}", path.display());

        let output = run_verify(std::slice::from_ref(&path));
        assert_eq!(
            output.status.code(),
            Some(3),
            "{name} 必须以退出码 3 结束（K5 数据校验失败），stderr = {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let codes = error_codes(&stdout);
        assert!(
            codes.iter().any(|code| code == expected_code),
            "{name} 应报 code `{expected_code}`，实际 codes = {codes:?}\n{stdout}"
        );
        assert_eq!(
            codes.len(),
            1,
            "{name} 应只报一条错误（一处坏就一处），实际 = {codes:?}\n{stdout}"
        );
    }
}

// ─────────────────────────── 6 个好文件 ───────────────────────────

#[test]
fn six_good_files_exit_0() {
    let examples = repo_root().join("qwb-data/examples");
    let good = [
        examples.join("basis/example-sto-3g-H.yaml"),
        examples.join("ecp/example-ecp-cu.yaml"),
        examples.join("functional/example-svwn5.yaml"),
        examples.join("grid/example-grid-normal.yaml"),
        examples.join("solvent/example-water.yaml"),
        examples.join("molecule/example-h2o.yaml"),
    ];
    for path in &good {
        assert!(path.is_file(), "缺数据样例 {}", path.display());
    }

    let output = run_verify(&good);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(
        output.status.code(),
        Some(0),
        "6 个好文件必须退出 0\nstdout = {stdout}\nstderr = {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(error_codes(&stdout).is_empty(), "好文件不该有错误 code");
    assert!(
        stdout.contains("6 file(s) checked, 6 passed, 0 failed"),
        "汇总行不对: {stdout}"
    );
}

#[test]
fn verify_reports_all_files_in_one_run() {
    // 好 + 坏混跑：整体退出码 3，但好文件仍报 ok、坏文件报 FAIL。
    let mut paths = vec![bad_fixture("empty_citation.yaml")];
    paths.push(repo_root().join("qwb-data/examples/grid/example-grid-normal.yaml"));

    let output = run_verify(&paths);
    assert_eq!(output.status.code(), Some(3));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FAIL basis"), "{stdout}");
    assert!(stdout.contains("ok   grid"), "{stdout}");
    assert!(
        stdout.contains("2 file(s) checked, 1 passed, 1 failed"),
        "{stdout}"
    );
}
