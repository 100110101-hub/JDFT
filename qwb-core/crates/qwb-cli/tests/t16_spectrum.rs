//! PR-04 端到端验收：`qwb props` 跑出的 UV/Vis 谱必须**落在 RS-02 的激发能上**。
//!
//! 判据（§6 的 PR-04 第三项拆开写）：
//!
//! 1. `result.json` 里出现 `uv_vis_spectrum` 块，且 HWHM/步长**等于数据里的值**
//!    （0.4 / 0.05 eV）—— 展宽参数来自数据，不是代码常量；
//! 2. 结果目录里有 `artifacts/spectrum.csv`，两列、能量递增、点数与块一致；
//! 3. **谱峰位置与激发能一致**：CSV 里每个局部极大值，都能在 `excited_states.transitions`
//!    的跃迁能（eV）里找到一个距离不超过 1.5 个网格步长的；反过来，每个 f > 0.05 的跃迁
//!    也都必须有一个峰对应（太弱的跃迁可能被强邻居淹没，所以用 f 门槛，不用"全部"）。
//!
//! 用临时 `--out` 目录，避免往仓库里写运行产物。

use std::path::{Path, PathBuf};
use std::process::Command;

fn qwb_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_qwb"))
}

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwb-pr04-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn ev_per_hartree() -> f64 {
    27.211_386_245_988
}

#[test]
fn uvviz_spectrum_peaks_sit_on_the_rs02_excitation_energies() {
    for (input, task) in [
        ("h2o_tddft.yaml", "h2o_tddft"),
        ("benzene_tddft.yaml", "benzene_tddft"),
    ] {
        check_spectrum(input, task);
    }
}

fn check_spectrum(input: &str, task: &str) {
    let out = scratch(task);
    let out_text = out.to_string_lossy().into_owned();
    let output = Command::new(qwb_bin())
        .args(["props", input, "--out", &out_text])
        .current_dir(examples())
        .output()
        .expect("spawn qwb");
    let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "`qwb props` 失败：\n{log}");

    let task_dir = out.join(task);
    let result_text = std::fs::read_to_string(task_dir.join("result.json")).expect("result.json");
    let result: serde_json::Value = serde_json::from_str(&result_text).expect("parse result.json");

    // ── 1. 块存在，且展宽参数来自数据 ──────────────────────────────────────
    let spectrum = result
        .get("uv_vis_spectrum")
        .expect("result.json 缺 uv_vis_spectrum 块（PR-04）");
    let hwhm = spectrum
        .get("gaussian_hwhm_ev")
        .and_then(serde_json::Value::as_f64)
        .expect("hwhm");
    let step = spectrum
        .get("grid_step_ev")
        .and_then(serde_json::Value::as_f64)
        .expect("step");
    assert!(
        (hwhm - 0.4).abs() < 1e-12,
        "HWHM 应来自 qwb-data（0.4 eV），实得 {hwhm}"
    );
    assert!(
        (step - 0.05).abs() < 1e-12,
        "步长应来自 qwb-data（0.05 eV），实得 {step}"
    );

    // ── 2. CSV 存在且与块一致 ─────────────────────────────────────────────
    let csv = std::fs::read_to_string(task_dir.join("artifacts").join("spectrum.csv"))
        .expect("artifacts/spectrum.csv");
    let mut rows = csv.lines();
    assert_eq!(rows.next(), Some("energy_ev,intensity"));
    let mut csv_points = Vec::new();
    let mut previous = f64::NEG_INFINITY;
    for row in rows {
        let (energy, intensity) = row.split_once(',').expect("两列");
        let energy: f64 = energy.parse().expect("energy");
        let intensity: f64 = intensity.parse().expect("intensity");
        assert!(energy > previous, "CSV 能量必须递增");
        assert!(intensity.is_finite() && intensity >= 0.0);
        previous = energy;
        csv_points.push((energy, intensity));
    }
    let block_points = spectrum
        .get("points")
        .and_then(serde_json::Value::as_array)
        .expect("points");
    assert_eq!(
        csv_points.len(),
        block_points.len(),
        "CSV 点数与 result.json 的块不一致"
    );

    // ── 3. 峰位 vs RS-02 的激发能 ────────────────────────────────────────
    let transitions = result
        .get("excited_states")
        .and_then(|states| states.get("transitions"))
        .and_then(serde_json::Value::as_array)
        .expect("excited_states.transitions");
    let catalogue: Vec<(f64, f64)> = transitions
        .iter()
        .map(|t| {
            let energy = t
                .get("energy_hartree")
                .and_then(serde_json::Value::as_f64)
                .expect("energy")
                * ev_per_hartree();
            let strength = t
                .get("oscillator_strength")
                .and_then(serde_json::Value::as_f64)
                .expect("f");
            (energy, strength)
        })
        .collect();

    let mut peaks = Vec::new();
    for window in csv_points.windows(3) {
        let left = window[0].1;
        let (energy, middle) = window[1];
        let right = window[2].1;
        if middle > left && middle >= right {
            peaks.push(energy);
        }
    }
    assert!(!peaks.is_empty(), "展宽谱里没有局部极大值");

    let tolerance = 1.5 * step;
    for peak in &peaks {
        assert!(
            catalogue
                .iter()
                .any(|(energy, _)| (energy - peak).abs() <= tolerance),
            "峰 {peak:.4} eV 在激发能表里找不到对应（容差 {tolerance} eV）：{catalogue:?}"
        );
    }
    let strong: Vec<f64> = catalogue
        .iter()
        .filter(|(_, strength)| *strength > 0.05)
        .map(|(energy, _)| *energy)
        .collect();
    assert!(
        !strong.is_empty(),
        "H2O/STO-3G 应至少有一个 f > 0.05 的跃迁"
    );
    for energy in &strong {
        assert!(
            peaks.iter().any(|peak| (peak - energy).abs() <= tolerance),
            "强跃迁 {energy:.4} eV 没有对应的峰（峰：{peaks:?}）"
        );
    }

    let _ = std::fs::remove_dir_all(&out);
}
