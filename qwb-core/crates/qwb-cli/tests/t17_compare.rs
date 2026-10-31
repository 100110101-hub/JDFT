//! PR-07 端到端验收（CLI 一半）：`qwb compare` 的对比表必须**与单看一个结果目录一致**。
//!
//! 判据（§6 的 PR-07 第一项拆开写）：
//!
//! 1. **值一致（1e-10）**：表里每个任务的能量，与它自己 `result.json` 的
//!    `energy_hartree` 之差不超过 1e-10 —— 对比表只做读取与格式化，不重算、不二次舍入；
//!    谱峰同理（与 `result.json` 点集的局部极大值一致）；
//! 2. **ΔE 自洽**：CSV 的 `delta_e_kcal_per_mol` 等于「该任务能量 − 首个任务能量」乘
//!    627.5094740634593 Ha→kcal/mol，容差按 8 位小数的打印精度给 1e-8；
//! 3. **CSV 结构**：表头逐字等于 `COMPARISON_COLUMNS`；行数 = 任务数；无 CR；
//!    每行 7 列，数值列能被 `f64::from_str` 读回（这是 pandas/csv 能读的前提）；
//! 4. **只读**：`compare` 不改结果目录 —— 跑完两个目录的文件列表不变。
//!
//! 三个任务刻意选成「同输入跑两遍 + 换泛函」：前者钉住可复现性（能量逐位相同），
//! 后者让 ΔE 列出现非零值，顺便覆盖"没有谱"的分支（`h2o_sto3g_svwn5` 没有激发态）。

use std::path::{Path, PathBuf};
use std::process::Command;

fn qwb_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_qwb"))
}

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwb-pr07-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

const HARTREE_TO_KCAL_PER_MOL: f64 = 627.509_474_063_459_3;

/// CSV 列序（与 `qwb-cli` 的 `COMPARISON_COLUMNS` 冻结一致；Python 侧同名常量）。
const COLUMNS: &str =
    "task,method,basis,energy_hartree,delta_e_kcal_per_mol,converged,spectrum_peaks_ev";

fn run_example(input: &str, out_root: &Path) {
    let out_text = out_root.to_string_lossy().into_owned();
    let output = Command::new(qwb_bin())
        .args(["run", input, "--out", &out_text])
        .current_dir(examples())
        .output()
        .expect("spawn qwb run");
    assert_eq!(
        output.status.code(),
        Some(0),
        "`qwb run {input}` 失败：\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn result_json(task_dir: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(task_dir.join("result.json")).expect("result.json");
    serde_json::from_str(&text).expect("parse result.json")
}

fn energy_of(task_dir: &Path) -> f64 {
    result_json(task_dir)
        .get("energy_hartree")
        .and_then(serde_json::Value::as_f64)
        .expect("energy_hartree")
}

fn peaks_of(task_dir: &Path) -> Vec<f64> {
    let value = result_json(task_dir);
    let Some(points) = value
        .get("uv_vis_spectrum")
        .and_then(|block| block.get("points"))
        .and_then(serde_json::Value::as_array)
    else {
        return Vec::new();
    };
    let samples: Vec<(f64, f64)> = points
        .iter()
        .filter_map(|point| {
            let pair = point.as_array()?;
            Some((pair.first()?.as_f64()?, pair.get(1)?.as_f64()?))
        })
        .collect();
    let mut peaks = Vec::new();
    for window in samples.windows(3) {
        let (_, left) = window[0];
        let (energy, middle) = window[1];
        let (_, right) = window[2];
        if middle > left && middle >= right {
            peaks.push(energy);
        }
    }
    peaks
}

fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read_dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn comparison_table_matches_reading_one_result_dir() {
    let scratch_root = scratch("table");
    let (root_a, root_b, root_c) = (
        scratch_root.join("a"),
        scratch_root.join("b"),
        scratch_root.join("c"),
    );
    run_example("h2o_tddft.yaml", &root_a);
    run_example("h2o_tddft.yaml", &root_b);
    run_example("h2o_sto3g_svwn5.yaml", &root_c);

    // `qwb run --out DIR` 按输入文件主干建子目录（K3 布局）。
    let dirs = [
        root_a.join("h2o_tddft"),
        root_b.join("h2o_tddft"),
        root_c.join("h2o_sto3g_svwn5"),
    ];
    for dir in &dirs {
        assert!(dir.join("result.json").is_file(), "缺 {}", dir.display());
    }
    let before: Vec<Vec<String>> = dirs.iter().map(|dir| listing(dir)).collect();

    let csv_path = scratch_root.join("comparison.csv");
    let output = Command::new(qwb_bin())
        .arg("compare")
        .args(&dirs)
        .arg("--csv")
        .arg(&csv_path)
        .current_dir(examples())
        .output()
        .expect("spawn qwb compare");
    let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "`qwb compare` 失败：\n{log}");

    // ── 4. 只读：跑完目录内容不变 ─────────────────────────────────────────
    let after: Vec<Vec<String>> = dirs.iter().map(|dir| listing(dir)).collect();
    assert_eq!(before, after, "`qwb compare` 不得改动结果目录");

    // ── 3. CSV 结构 ───────────────────────────────────────────────────────
    let csv = std::fs::read_to_string(&csv_path).expect("comparison.csv");
    assert!(
        !csv.contains('\r'),
        "CSV 不能有 CR（Windows 尾巴会让别的工具多一列）"
    );
    let mut rows = csv.lines();
    assert_eq!(rows.next(), Some(COLUMNS), "CSV 表头必须逐字等于冻结列序");
    let rows: Vec<Vec<&str>> = rows.map(|row| row.split(',').collect()).collect();
    assert_eq!(rows.len(), dirs.len(), "CSV 行数应等于任务数");
    for row in &rows {
        assert_eq!(row.len(), 7, "每行 7 列：{row:?}");
    }
    assert_eq!(rows[0][0], "h2o_tddft", "行序应等于命令行给的目录顺序");
    assert_eq!(rows[2][0], "h2o_sto3g_svwn5");

    // ── 1. 值一致（1e-10）+ 2. ΔE 自洽 ────────────────────────────────────
    let reference = energy_of(&dirs[0]);
    for (row, dir) in rows.iter().zip(dirs.iter()) {
        let expected = energy_of(dir);
        let printed: f64 = row[3].parse().expect("energy_hartree 列可被 f64 读回");
        assert!(
            (printed - expected).abs() < 1e-10,
            "{} 的对比能量 {printed} 与单看结果 {expected} 相差超过 1e-10",
            row[0]
        );
        let delta: f64 = row[4].parse().expect("ΔE 列可被 f64 读回");
        let want = (expected - reference) * HARTREE_TO_KCAL_PER_MOL;
        assert!(
            (delta - want).abs() < 1e-8,
            "{} 的 ΔE {delta} 与由能量算出的 {want} 相差超过 1e-8",
            row[0]
        );
        // 方法/基组来自 input.yaml，不是占位符。
        assert_ne!(row[1], "?", "{} 的方法名没读出来", row[0]);
        assert_ne!(row[2], "?", "{} 的基组名没读出来", row[0]);
        // 谱峰列：与点集的局部极大值一致（没有谱就是空列）。
        let expected_peaks = peaks_of(dir);
        let printed_peaks: Vec<f64> = if row[6].is_empty() {
            Vec::new()
        } else {
            row[6]
                .split(';')
                .map(|peak| peak.parse().expect("峰列可被 f64 读回"))
                .collect()
        };
        assert_eq!(
            printed_peaks.len(),
            expected_peaks.len(),
            "{} 的峰数不一致：{printed_peaks:?} vs {expected_peaks:?}",
            row[0]
        );
        for (printed, expected) in printed_peaks.iter().zip(expected_peaks.iter()) {
            assert!(
                (printed - expected).abs() < 1e-4,
                "{} 的峰位 {printed} 与点集 {expected} 不一致",
                row[0]
            );
        }
    }
    assert!(
        !rows[0][6].is_empty(),
        "h2o_tddft 应该有展宽谱（PR-04），峰列不该是空的"
    );
    assert!(
        rows[2][6].is_empty(),
        "h2o_sto3g_svwn5 没有激发态，峰列应为空"
    );

    // 同一输入跑两遍：能量必须逐位相同（否则 1e-10 判据就是靠运气）。
    assert_eq!(
        rows[0][3], rows[1][3],
        "同一输入的两次运行能量应逐位相同：{} vs {}",
        rows[0][3], rows[1][3]
    );
    // 换泛函后 ΔE 必须非零（否则这条判据什么都没测到）。
    let delta_c: f64 = rows[2][4].parse().expect("ΔE");
    assert!(
        delta_c.abs() > 1e-3,
        "换泛函的 ΔE 不该接近 0（实得 {delta_c} kcal/mol）"
    );

    // 表格输出里也能找到同样的 10 位小数能量（人看的那份不能比 CSV 少信息）。
    for dir in &dirs {
        let text = format!("{:.10}", energy_of(dir));
        assert!(log.contains(&text), "stdout 表格里找不到 {text}：\n{log}");
    }

    let _ = std::fs::remove_dir_all(&scratch_root);
}

#[test]
fn compare_needs_at_least_two_dirs() {
    let out = scratch("one");
    let output = Command::new(qwb_bin())
        .arg("compare")
        .arg(&out)
        .current_dir(examples())
        .output()
        .expect("spawn qwb compare");
    assert_eq!(
        output.status.code(),
        Some(1),
        "只给一个目录应报输入错误（退出码 1）：\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn compare_reports_missing_result_json_as_input_error() {
    let out = scratch("missing");
    let a = out.join("a");
    let b = out.join("b");
    std::fs::create_dir_all(&a).expect("a");
    std::fs::create_dir_all(&b).expect("b");
    std::fs::write(
        b.join("result.json"),
        br#"{"energy_hartree": -1.0, "scf_converged": true}"#,
    )
    .expect("write");
    let output = Command::new(qwb_bin())
        .arg("compare")
        .arg(&a)
        .arg(&b)
        .current_dir(examples())
        .output()
        .expect("spawn qwb compare");
    assert_eq!(output.status.code(), Some(1), "缺 result.json 应是输入错误");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("result.json"),
        "错误信息要点出缺哪个文件：{stderr}"
    );
    let _ = std::fs::remove_dir_all(&out);
}
