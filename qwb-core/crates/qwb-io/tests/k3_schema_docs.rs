//! C-03 验收（第二半）：`qwb-contracts/studio/*.schema.yaml` 与 Rust 类型**逐字段一致**。
//!
//! K3 的 schema 是手写文档（不是 schemars 生成，因为 `engine-api` 要保持零 schema 依赖），
//! 所以用测试把“文档”和“代码”钉在一起：字段集、required 集、状态机转移，
//! 任何一边单方面改动都会让 C-03 失败。

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use qwb_engine_api::{DataSnapshot, EnvInfo, JobInput, JobResult};
use qwb_io::{TaskRecord, TaskState};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn load(path: &Path) -> serde_yaml::Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{} is not YAML: {e}", path.display()))
}

fn keys(value: &serde_yaml::Value) -> BTreeSet<String> {
    value
        .as_mapping()
        .unwrap_or_else(|| panic!("expected a mapping, got {value:?}"))
        .keys()
        .map(|k| k.as_str().expect("string key").to_string())
        .collect()
}

fn json_keys(value: &serde_json::Value) -> BTreeSet<String> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("expected a JSON object"))
        .keys()
        .cloned()
        .collect()
}

fn input() -> JobInput {
    qwb_engine_api::JobInput::new(
        qwb_engine_api::Molecule::new(
            vec![
                qwb_engine_api::Atom::new("H", [0.0, 0.0, -0.37]),
                qwb_engine_api::Atom::new("H", [0.0, 0.0, 0.37]),
            ],
            0,
            1,
        ),
        "sto-3g",
        "svwn5",
        qwb_engine_api::GridLevel::Normal,
        "vacuum",
        qwb_engine_api::JobType::Sp,
        qwb_engine_api::TaskOptions::default(),
    )
}

fn env() -> EnvInfo {
    EnvInfo::new(
        "0.1.0",
        "rustc 1.97.1",
        "a".repeat(64),
        DataSnapshot::new(),
        4,
        "windows 11 x86_64",
        "2025-01-10T08:00:00Z",
    )
}

#[test]
fn task_schema_matches_task_record() {
    let doc = load(&repo_root().join("qwb-contracts/studio/task.schema.yaml"));
    let documented = keys(&doc["properties"]);

    let record = TaskRecord::draft("t1", "H2 sp", "2025-01-10T08:00:00Z", input());
    let actual = json_keys(&serde_json::to_value(&record).expect("serialize"));

    assert_eq!(
        documented, actual,
        "task.schema.yaml 与 TaskRecord 的字段集不一致"
    );

    let required: BTreeSet<String> = doc["required"]
        .as_sequence()
        .expect("required list")
        .iter()
        .map(|v| v.as_str().expect("string").to_string())
        .collect();
    let expected_required: BTreeSet<String> =
        ["id", "name", "state", "created", "updated", "input"]
            .into_iter()
            .map(ToString::to_string)
            .collect();
    assert_eq!(required, expected_required);

    // 状态枚举也一致。
    let states: BTreeSet<String> = doc["$defs"]["TaskState"]["enum"]
        .as_sequence()
        .expect("enum")
        .iter()
        .map(|v| v.as_str().expect("string").to_string())
        .collect();
    let rust_states: BTreeSet<String> = TaskState::ALL
        .into_iter()
        .map(|s| s.as_str().to_string())
        .collect();
    assert_eq!(states, rust_states, "TaskState 取值不一致");
}

#[test]
fn state_machine_doc_matches_rust() {
    let doc = load(&repo_root().join("qwb-contracts/studio/task.schema.yaml"));
    let machine = &doc["x-state-machine"];

    let documented_transitions: BTreeSet<(String, String)> = machine["transitions"]
        .as_sequence()
        .expect("transitions")
        .iter()
        .map(|t| {
            (
                t["from"].as_str().expect("from").to_string(),
                t["to"].as_str().expect("to").to_string(),
            )
        })
        .collect();
    let rust_transitions: BTreeSet<(String, String)> = TaskState::LEGAL
        .iter()
        .map(|(from, to)| (from.as_str().to_string(), to.as_str().to_string()))
        .collect();
    assert_eq!(
        documented_transitions, rust_transitions,
        "K3 状态机文档与 TaskState::LEGAL 不一致"
    );

    let documented_terminal: BTreeSet<String> = machine["terminal"]
        .as_sequence()
        .expect("terminal")
        .iter()
        .map(|v| v.as_str().expect("string").to_string())
        .collect();
    let rust_terminal: BTreeSet<String> = TaskState::ALL
        .into_iter()
        .filter(|s| s.is_terminal())
        .map(|s| s.as_str().to_string())
        .collect();
    assert_eq!(documented_terminal, rust_terminal, "终态集合不一致");
}

#[test]
fn result_schema_matches_job_result() {
    let doc = load(&repo_root().join("qwb-contracts/studio/result.schema.yaml"));
    let documented = keys(&doc["$defs"]["JobResult"]["properties"]);

    let result = JobResult::new(1.5, true, DataSnapshot::new(), env())
        .with_gradient(nalgebra::DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 0.0]))
        .with_hessian(nalgebra::DMatrix::from_row_slice(3, 3, &[0.0; 9]))
        .with_dipole(nalgebra::DVector::from_vec(vec![0.0, 0.0, 0.0]))
        .with_orbitals(
            nalgebra::DVector::from_vec(vec![-0.5]),
            nalgebra::DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        // 阶段二的可选块（K7/K8/K9）全部挂上：这条断言同时覆盖新旧字段，
        // 任何一个"文档里有、Rust 里没有"（或反过来）都会红。
        .with_dispersion_energy(-0.001)
        .with_spin(qwb_engine_api::SpinResult::new(5, 5, 3, 2.0))
        .with_correlation(qwb_engine_api::CorrelationResult::new(
            qwb_engine_api::Mp2Kind::Mp2,
            -0.3,
            -76.3,
        ))
        .with_solvation(
            qwb_engine_api::SolvationResult::new(
                qwb_engine_api::SolventModel::Cosmo,
                "water",
                -76.4,
                -0.02,
                qwb_engine_api::solvation::CavityGeometry::new(42.0, 18.0, 1024),
                7,
            )
            .with_converged(true),
        )
        .with_excited_states(qwb_engine_api::ExcitedStates::new(
            qwb_engine_api::ExcitedStateMethod::Tda,
            1,
            vec![qwb_engine_api::Transition::new(1, 0.3, 412.0, 0.05)],
            qwb_engine_api::SumRules::new(7.4, 10.0),
        ))
        .with_nmr_shieldings(qwb_engine_api::NmrShieldings::new(
            [0.0, 0.0, 0.0],
            vec![qwb_engine_api::NucleusShielding::new(
                "H",
                0,
                31.0,
                [[31.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 32.0]],
            )],
        ))
        .with_polarizability(
            qwb_engine_api::Polarizability::from_static_tensor([
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 20.0],
            ])
            .expect("tensor"),
        )
        .with_multipoles(qwb_engine_api::Multipoles::new([0.0, 0.0, 0.0]))
        .with_numerics_report(
            qwb_engine_api::NumericsReport::new(
                qwb_engine_api::NumericsPath::Analytic,
                3,
                1.5,
                0.5,
            )
            .with_grid(qwb_engine_api::GridSummary::new(75, 302, 20.0)),
        );
    let actual = json_keys(&serde_json::to_value(&result).expect("serialize"));

    assert_eq!(
        documented, actual,
        "result.schema.yaml 的 JobResult 字段集与 Rust 不一致"
    );

    let required: BTreeSet<String> = doc["$defs"]["JobResult"]["required"]
        .as_sequence()
        .expect("required")
        .iter()
        .map(|v| v.as_str().expect("string").to_string())
        .collect();
    let expected_required: BTreeSet<String> = [
        "energy_hartree",
        "scf_converged",
        "data_snapshot",
        "log",
        "env",
    ]
    .into_iter()
    .map(ToString::to_string)
    .collect();
    assert_eq!(
        required, expected_required,
        "JobResult 的 required 集不一致"
    );
}

#[test]
fn result_schema_matches_env_info() {
    let doc = load(&repo_root().join("qwb-contracts/studio/result.schema.yaml"));
    let documented = keys(&doc["$defs"]["EnvInfo"]["properties"]);
    let actual = json_keys(&serde_json::to_value(env()).expect("serialize"));
    assert_eq!(
        documented, actual,
        "result.schema.yaml 的 EnvInfo 字段集与 Rust 不一致"
    );

    let required: BTreeSet<String> = doc["$defs"]["EnvInfo"]["required"]
        .as_sequence()
        .expect("required")
        .iter()
        .map(|v| v.as_str().expect("string").to_string())
        .collect();
    for field in [
        "engine_version",
        "rustc_version",
        "cargo_lock_sha256",
        "data_snapshot",
        "threads",
        "os",
        "created_at_utc",
    ] {
        assert!(
            required.contains(field),
            "EnvInfo.{field} 必须是 required（G5）"
        );
    }
}
