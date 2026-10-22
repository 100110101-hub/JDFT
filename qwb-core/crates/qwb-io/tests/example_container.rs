//! C-04 验收：提交进仓库的 `qwb-contracts/studio/examples/example.qwb`
//! **能 open、meta 可读、zip 内部结构可校验**。

use std::path::{Path, PathBuf};

use qwb_io::container::{DATA_SNAPSHOT_FILE, META_FILE, QwbContainer};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn example_path() -> PathBuf {
    repo_root().join("qwb-contracts/studio/examples/example.qwb")
}

#[test]
fn committed_example_container_opens_and_validates() {
    let path = example_path();
    assert!(
        path.is_file(),
        "missing committed example {}: run `cargo run -p qwb-io --example make_example_qwb`",
        path.display()
    );

    let container = QwbContainer::open(&path).expect("example.qwb must open");

    // meta 可读
    let meta = container.meta();
    assert_eq!(meta.name, "example-project");
    assert_eq!(meta.task_count, 1);
    assert!(!meta.app_version.is_empty());
    assert!(!meta.created.is_empty());

    // 内部结构：meta.json 在第一位，且 K4 规定的重点条目都在
    assert_eq!(container.entry_names()[0], META_FILE);
    assert!(
        container
            .entry_names()
            .iter()
            .any(|n| n == DATA_SNAPSHOT_FILE)
    );
    assert!(
        container
            .entry_names()
            .iter()
            .any(|n| n == "tasks/demo-0001/input.yaml")
    );

    // 整容器结构校验（含 task_count 与 tasks/ 目录数一致）
    container
        .validate()
        .expect("example.qwb must be structurally valid");

    // meta.json 真的能再从容器里读出来
    let meta_bytes = container.read_file(META_FILE).expect("read meta.json");
    let parsed: serde_json::Value = serde_json::from_slice(&meta_bytes).expect("meta.json is JSON");
    assert_eq!(parsed["name"], "example-project");

    // input.yaml 能还原成合法的 JobInput（K4 输入格式闭环）
    let input_bytes = container
        .read_file("tasks/demo-0001/input.yaml")
        .expect("read input.yaml");
    let input: qwb_engine_api::JobInput =
        serde_yaml::from_slice(&input_bytes).expect("input.yaml parses");
    input.validate().expect("input.yaml is semantically valid");
    assert_eq!(input.molecule.n_atoms(), 2);
    assert_eq!(input.basis_name, "sto-3g");
}

#[test]
fn example_container_rejects_tampering() {
    // 复制一份、改一个字节，必须打不开或校验失败（容器是结构，不是信任的 blob）。
    let source = example_path();
    let dir = std::env::temp_dir().join(format!("qwb-c04-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let broken = dir.join("broken.qwb");

    let mut bytes = std::fs::read(&source).expect("read example");
    // 破坏 zip 的结尾（central directory）比破坏中间字节更必然失败。
    let len = bytes.len();
    bytes.truncate(len / 2);
    std::fs::write(&broken, &bytes).unwrap();
    assert!(
        QwbContainer::open(&broken).is_err(),
        "a truncated container must not open"
    );

    std::fs::remove_dir_all(&dir).ok();
}
