//! P-02 验收：**Python 写的 `.qwb` 必须能被 Rust 读**。
//!
//! 夹具 `qwb-studio/tests/fixtures/python_written.qwb` 由
//! `qwb_studio.container.QwbContainer`（Python `zipfile`）生成，内容见
//! `scripts/gen_p02_fixture.py`。这里做三件事：
//!
//! 1. Rust 的 `QwbContainer::open` 能打开它，`validate()` 结构自检通过；
//! 2. 两个 task 的 `input.yaml` 都能反序列化成 K2 的 `JobInput` 并通过语义校验
//!    （即 Python 写的字节是**真正的 K4 输入格式**，不是"看起来像 YAML"）；
//! 3. Rust 规范化重写（`write_files(vec![])`）之后内容逐条目不变 —— 不比较压缩后的
//!    字节：Python 的 zlib 与 Rust 的 Deflate 实现不保证产出相同字节流，K4 只要求
//!    **同一实现**重复导出稳定。跨语言承诺的是语义互操作。
//!
//! 夹具缺失或用例失败时，报错信息会直接给出重新生成的命令。

use std::path::{Path, PathBuf};

use qwb_engine_api::JobInput;
use qwb_io::container::{
    ContainerError, DATA_SNAPSHOT_FILE, META_FILE, QwbContainer, task_entry_path,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn fixture_path() -> PathBuf {
    repo_root().join("qwb-studio/tests/fixtures/python_written.qwb")
}

fn open_fixture() -> QwbContainer {
    let path = fixture_path();
    assert!(
        path.is_file(),
        "missing Python-written fixture {}: run `.venv/Scripts/python.exe scripts/gen_p02_fixture.py`",
        path.display()
    );
    QwbContainer::open(&path).expect("Rust must open a container written by Python's zipfile")
}

#[test]
fn python_written_container_opens_and_validates() {
    let container = open_fixture();

    // meta.json 是第一条目，且字段被 Rust 的 ProjectMeta 接受。
    assert_eq!(container.entry_names()[0], META_FILE);
    assert_eq!(container.meta().name, "python-fixture");
    assert_eq!(container.meta().task_count, 2);
    assert_eq!(
        container.meta().schema_version,
        qwb_io::container::K4_SCHEMA_VERSION
    );

    // task_count 与 tasks/ 目录数一致（K4 §3 结构性约束 4）。
    container
        .validate()
        .expect("Python-written container must be structurally valid");

    // 条目集合与夹具生成器一致（顺序 = K4 既定顺序：meta 第一，其余名字典序）。
    let expected: Vec<String> = vec![
        META_FILE.to_string(),
        "artifacts/mo-energies.txt".to_string(),
        DATA_SNAPSHOT_FILE.to_string(),
        task_entry_path("t1", "input.yaml").unwrap(),
        task_entry_path("t1", "log.txt").unwrap(),
        task_entry_path("t2", "input.yaml").unwrap(),
    ];
    assert_eq!(container.entry_names(), expected.as_slice());
}

#[test]
fn python_written_inputs_are_real_job_inputs() {
    let container = open_fixture();
    for (task_id, expected_atoms) in [("t1", 2), ("t2", 3)] {
        let entry = task_entry_path(task_id, "input.yaml").expect("entry path");
        let bytes = container
            .read_file(&entry)
            .unwrap_or_else(|e| panic!("read {entry}: {e}"));
        let input: JobInput = serde_yaml::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("{entry} must deserialize as JobInput: {e}"));
        input
            .validate()
            .unwrap_or_else(|e| panic!("{entry} must be semantically valid: {e}"));
        assert_eq!(input.molecule.n_atoms(), expected_atoms, "{entry}");
        assert_eq!(input.basis_name, "sto-3g", "{entry}");
        assert_eq!(input.method_name, "svwn5", "{entry}");
    }
}

#[test]
fn python_written_data_snapshot_is_valid() {
    let container = open_fixture();
    let bytes = container
        .read_file(DATA_SNAPSHOT_FILE)
        .expect("read data_snapshot.json");
    let snapshot: qwb_engine_api::DataSnapshot = serde_json::from_slice(&bytes)
        .expect("data_snapshot.json must deserialize as DataSnapshot");
    assert_eq!(snapshot.len(), 3, "fixture references three data files");
    snapshot
        .validate()
        .expect("every digest must be a 64-char lowercase sha256 hex");
}

#[test]
fn rust_canonical_rewrite_preserves_python_content() {
    let source = fixture_path();
    let dir = std::env::temp_dir().join(format!("qwb-p02-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let copy = dir.join("rewritten.qwb");
    std::fs::copy(&source, &copy).expect("copy fixture");

    let before = {
        let container = QwbContainer::open(&copy).expect("open copy");
        container
            .entry_names()
            .iter()
            .map(|name| (name.clone(), container.read_file(name).unwrap()))
            .collect::<Vec<_>>()
    };

    // 规范化重写：Rust 侧只重排/重压，条目内容必须原样保留。
    let mut container = QwbContainer::open(&copy).expect("reopen copy");
    container
        .write_files(Vec::new())
        .expect("canonical rewrite");
    drop(container);

    let after = {
        let container = QwbContainer::open(&copy).expect("reopen rewritten");
        container
            .entry_names()
            .iter()
            .map(|name| (name.clone(), container.read_file(name).unwrap()))
            .collect::<Vec<_>>()
    };

    assert_eq!(before.len(), after.len());
    for ((name_a, bytes_a), (name_b, bytes_b)) in before.iter().zip(after.iter()) {
        assert_eq!(name_a, name_b, "entry order must survive the rewrite");
        assert_eq!(
            bytes_a, bytes_b,
            "entry content {name_a} must survive the rewrite"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn python_written_container_rejects_tampering() {
    let source = fixture_path();
    let dir = std::env::temp_dir().join(format!("qwb-p02-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // 截断：必须打不开（不能"尽力打开"）。
    let truncated = dir.join("truncated.qwb");
    let mut bytes = std::fs::read(&source).expect("read fixture");
    let len = bytes.len();
    bytes.truncate(len / 2);
    std::fs::write(&truncated, &bytes).unwrap();
    assert!(
        QwbContainer::open(&truncated).is_err(),
        "a truncated Python-written container must not open"
    );

    // meta.json 不是合法 JSON：必须报 MetaInvalid，K5 退出码 3。
    // （Rust 的 write_files 只写自己序列化的 ProjectMeta，所以这里手工造 zip，
    //  模拟"别的写入器写坏了 meta.json"的情形。）
    let bad_meta = dir.join("bad-meta.qwb");
    {
        use std::io::Write as _;
        let file = std::fs::File::create(&bad_meta).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file(META_FILE, options).unwrap();
        writer.write_all(b"{ not json }").unwrap();
        writer.start_file("tasks/t1/input.yaml", options).unwrap();
        writer.write_all(b"jobtype: sp\n").unwrap();
        writer.finish().unwrap();
    }
    let err = QwbContainer::open(&bad_meta).expect_err("must reject invalid meta.json");
    assert!(matches!(err, ContainerError::MetaInvalid(_)));
    assert_eq!(err.exit_code(), 3);

    std::fs::remove_dir_all(&dir).ok();
}
