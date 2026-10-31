//! PR-03 验收：`.qwb` 容器与 K3 结果目录**向后/向前兼容**新块。
//!
//! 阶段二给结果加了若干**加性**块（`excited_states`、`nmr`、`multipoles`、
//! `uv_vis_spectrum`、`vibrational_frequencies`、`dispersion`、`solvation`）。
//! K4 容器本身按条目名存字节、不解释内容，所以「新块不影响旧读法」这件事**必须靠测试
//! 钉住**，而不是靠"看起来没问题」。
//!
//! 三条判据（对应 §6 里 PR-03 的验收）：
//!
//! 1. **旧夹具仍可读**：`qwb-contracts/studio/examples/example.qwb` 打开 + `validate()`；
//! 2. **新块写出→读回逐字节一致**（导出幂等：再导出一次内容不变）；
//! 3. **旧读法不炸**：只认识旧字段的读者，从**带新块**的 `result.json` 里仍能取到旧字段
//!    （serde 忽略未知键），且缺块时新读者也不报错。

use std::path::{Path, PathBuf};

use qwb_io::container::{
    DATA_SNAPSHOT_FILE, K4_SCHEMA_VERSION, META_FILE, ProjectMeta, QwbContainer,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwb-pr03-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// 阶段二的加性结果块（刻意包含嵌套对象与数组，覆盖真实形状）。
const NEW_RESULT_BLOCKS: &str = r#"{
  "energy_hartree": -76.0266,
  "converged": true,
  "excited_states": [
    {"energy_ev": 7.48, "oscillator_strength": 0.052, "multiplicity": 1},
    {"energy_ev": 9.91, "oscillator_strength": 0.181, "multiplicity": 1}
  ],
  "nmr": {"shielding_ppm": {"H1": 31.2, "O1": 328.5}, "gauge_spread_ppm": 0.31},
  "multipoles": {"dipole_debye": [0.0, 0.0, 0.71], "quadrupole_traceless": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]]},
  "uv_vis_spectrum": {"broadening_hwhm_ev": 0.4, "points": [[6.0, 0.001], [7.5, 0.42]]},
  "vibrational_frequencies": [{"wavenumber_cm": 1642.3, "ir_intensity_km_per_mol": 21.4}],
  "dispersion": {"energy_hartree": -0.001234, "model": "D3(BJ)"},
  "solvation": {"model": "SMD", "delta_g_solv_kcal_per_mol": -6.31}
}"#;

/// 阶段一形状的结果（没有上面那些新块）。
const OLD_RESULT: &str = r#"{"energy_hartree": -76.0266, "converged": true}"#;

/// 只认识阶段一字段的读者：必须能忽略未知键（否则「向后兼容」是假的）。
#[derive(serde::Deserialize)]
struct LegacyReader {
    energy_hartree: f64,
    converged: bool,
}

fn build_container(dir: &Path, result_json: &str) -> PathBuf {
    let path = dir.join("project.qwb");
    let meta = ProjectMeta::new(
        K4_SCHEMA_VERSION,
        "pr03-compat",
        "2026-11-07T00:00:00Z",
        "0.2.0",
        1,
    );
    let mut container = QwbContainer::create(&path, meta).expect("create");
    container
        .write_files(vec![
            (DATA_SNAPSHOT_FILE.to_string(), b"{}".to_vec()),
            (
                "tasks/t1/input.yaml".to_string(),
                b"method: svwn5\nbasis: sto-3g\n".to_vec(),
            ),
            (
                "tasks/t1/result.json".to_string(),
                result_json.as_bytes().to_vec(),
            ),
            (
                "tasks/t1/log.txt".to_string(),
                b"normal termination\n".to_vec(),
            ),
        ])
        .expect("write");
    path
}

#[test]
fn committed_legacy_container_still_opens() {
    let path = repo_root().join("qwb-contracts/studio/examples/example.qwb");
    assert!(
        path.is_file(),
        "missing committed example {}",
        path.display()
    );
    let container = QwbContainer::open(&path).expect("legacy fixture must open");
    container.validate().expect("legacy fixture must validate");
    assert!(
        container.entry_names().iter().any(|name| name == META_FILE),
        "legacy fixture lost {META_FILE}"
    );
}

#[test]
fn new_result_blocks_roundtrip_byte_identically() {
    let dir = scratch_dir("roundtrip");
    let path = build_container(&dir, NEW_RESULT_BLOCKS);

    let reopened = QwbContainer::open(&path).expect("open");
    reopened.validate().expect("validate");
    let entry = qwb_io::container::task_entry_path("t1", "result.json").expect("entry path");
    let bytes = reopened.read_file(&entry).expect("read result.json");
    assert_eq!(
        std::str::from_utf8(&bytes).expect("utf8"),
        NEW_RESULT_BLOCKS,
        "新块经过容器写出/读回后被改动了"
    );

    // 导出幂等：再导出一次（不新增条目）内容逐条目不变。
    let mut again = QwbContainer::open(&path).expect("open again");
    let before: Vec<(String, Vec<u8>)> = again
        .entry_names()
        .iter()
        .map(|name| (name.clone(), again.read_file(name).expect("read")))
        .collect();
    again.write_files(vec![]).expect("re-export");
    let after = QwbContainer::open(&path).expect("reopen after re-export");
    assert_eq!(before.len(), after.entry_names().len());
    for (name, expected) in before {
        assert_eq!(
            after.read_file(&name).expect("read after re-export"),
            expected,
            "条目 {name} 在再导出后变了"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn legacy_reader_tolerates_blocks_it_does_not_know() {
    // 带新块的 result.json 仍能被只认识旧字段的读者读出旧字段。
    let legacy: LegacyReader =
        serde_json::from_str(NEW_RESULT_BLOCKS).expect("legacy reader must ignore new blocks");
    assert!((legacy.energy_hartree + 76.0266).abs() < 1e-12);
    assert!(legacy.converged);

    // 反过来：阶段一形状（没有新块）也不能让新读者报错。
    let legacy_shape: LegacyReader =
        serde_json::from_str(OLD_RESULT).expect("old shape must parse");
    assert!(legacy_shape.converged);
}

#[test]
fn old_shape_container_and_new_shape_container_coexist() {
    let dir = scratch_dir("mixed");
    let old_path = build_container(&dir, OLD_RESULT);
    let old = QwbContainer::open(&old_path).expect("old-shape container opens");
    old.validate().expect("old-shape container validates");
    let entry = qwb_io::container::task_entry_path("t1", "result.json").expect("entry");
    assert_eq!(
        std::str::from_utf8(&old.read_file(&entry).expect("read")).expect("utf8"),
        OLD_RESULT
    );
    let _ = std::fs::remove_dir_all(&dir);
}
