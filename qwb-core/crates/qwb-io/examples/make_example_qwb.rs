//! 生成 `qwb-contracts/studio/examples/example.qwb`（K4 示例容器）。
//!
//! ```text
//! cargo run -p qwb-io --example make_example_qwb
//! ```
//!
//! 示例项目只含**输入**，不含任何伪造的结果：里面有一个 `draft` 状态的 task，
//! 它的 `input.yaml` 是一份真实可跑的 H2/STO-3G/SVWN5 输入。

use std::path::{Path, PathBuf};

use qwb_engine_api::{Atom, GridLevel, JobInput, JobType, Molecule, TaskOptions};
use qwb_io::container::{META_FILE, ProjectMeta, QwbContainer, task_entry_path};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = repo_root();
    let data_root = repo.join("qwb-data");
    let out_dir = repo.join("qwb-contracts/studio/examples");
    let out_path = out_dir.join("example.qwb");

    let input = JobInput::new(
        Molecule::new(
            vec![
                Atom::new("H", [0.0, 0.0, -0.37]),
                Atom::new("H", [0.0, 0.0, 0.37]),
            ],
            0,
            1,
        ),
        "sto-3g",
        "svwn5",
        GridLevel::Normal,
        "vacuum",
        JobType::Sp,
        TaskOptions::default(),
    );

    let input_yaml = serde_yaml::to_string(&input)?;

    // data_snapshot 里放**数据文件自己声明的** canonical sha256：
    // 与 registry.yaml 的条目、文件内 sha256 字段三者一致，可交叉校验。
    let mut snapshot: Vec<(String, String)> = Vec::new();
    for relative in [
        "examples/basis/example-sto-3g-H.yaml",
        "examples/functional/example-svwn5.yaml",
    ] {
        let path = data_root.join(relative);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            format!(
                "missing data fixture {}: {e}; run `qwb-stamp-data` first",
                path.display()
            )
        })?;
        let value: serde_yaml::Value = serde_yaml::from_str(&text)?;
        let sha = value
            .get("sha256")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{} has no sha256 field", path.display()))?
            .to_string();
        snapshot.push((relative.to_string(), sha));
    }
    let snapshot_json = {
        let map: std::collections::BTreeMap<&str, &str> = snapshot
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        serde_json::to_string_pretty(&map)?
    };

    std::fs::create_dir_all(&out_dir)?;
    let meta = ProjectMeta::new(
        qwb_io::container::K4_SCHEMA_VERSION,
        "example-project",
        "2026-10-22T22:53:00Z",
        env!("CARGO_PKG_VERSION"),
        1,
    );
    let mut container = QwbContainer::create(&out_path, meta)?;
    container.write_files(vec![
        (
            task_entry_path("demo-0001", "input.yaml")?,
            input_yaml.into_bytes(),
        ),
        ("data_snapshot.json".to_string(), snapshot_json.into_bytes()),
    ])?;
    container.validate()?;

    println!("wrote {}", out_path.display());
    println!("entries:");
    for name in container.entry_names() {
        println!("  {name}");
    }
    println!("meta file is {META_FILE}");
    Ok(())
}
