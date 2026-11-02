//! 第 387 轮：`multiplicity > 1` 但没写 `scf_type` 时**必须拒绝**，不能静默算成闭壳层。
//!
//! 这是 USAGE §9.1 里挂了很久的那条 ⚠ 的收口。旧行为：O2 三重态（16 电子，偶数）不写
//! `scf_type` → 走闭壳层 RHF → **−147.0158990576 Ha、退出码 0、没有 spin 块**，而同一个
//! 体系显式写 `scf_type: rohf` 是 **−147.6321910352 Ha**（同一几何 R = 1.208 Å）。差 0.616 Ha
//! 的"看起来成功的结果"比报错危险得多（project-phase2.md §10.3 禁止静默降级），所以改成
//! 拒绝并点名字段。
//!
//! 三条判据：① 不写 `scf_type` 的三重态退出 **1**、消息里同时有 `scf_type` 与
//! `multiplicity = 3`、且**不留下** `result.json`（拒绝发生在任何数值之前）；
//! ② 同一份输入补上 `scf_type: rohf` 就真的算出 ROHF 的能量（退出 0）；
//! ③ 单重态不写 `scf_type` 的老路径**不受影响**（否则等于把旧输入全拒了）。
use std::path::{Path, PathBuf};
use std::process::Command;

fn qwb_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_qwb"))
}

/// `qwb-core/examples`
fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

/// 数据仓库根在仓库根（`qwb-cli` → `crates` → `qwb-core` → 仓库根）。
fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwb-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

/// O2 三重态：直接拿仓库里那份 `o2_uhf_sto3g.yaml`，只把 `scf_type` 那一行去掉。
///
/// 不手写整份 YAML 是因为 CLI 的读取器要求 `options` 块**字段齐**（缺一个就报
/// `missing field scf_max_iter`）—— 硬编码那份清单会在选项演进时悄悄过期。
fn o2_triplet(scf_type: Option<&str>) -> String {
    let source = std::fs::read_to_string(examples().join("o2_uhf_sto3g.yaml")).expect("读示例");
    let mut text = String::new();
    for line in source.lines() {
        if line.trim_start().starts_with("scf_type:") {
            continue;
        }
        text.push_str(line);
        text.push('\n');
    }
    if let Some(scf_type) = scf_type {
        text.push_str(&format!("  scf_type: {scf_type}\n"));
    }
    text
}

fn run(cwd: &Path, args: &[&str]) -> (i32, String) {
    let output = Command::new(qwb_bin())
        .args(args)
        .current_dir(cwd)
        // 数据根固定成仓库里的 `qwb-data`：临时目录里没有它，而这里也不需要联网取数据。
        .env("QWB_DATA_ROOT", data_root())
        .env("QWB_AUTO_FETCH", "0")
        .output()
        .expect("spawn qwb");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.code().unwrap_or(-1), text)
}

/// 从 stdout 的 `energy <值> Ha` 行里取出能量（CLI 的稳定输出格式，10 位小数）。
fn energy_hartree(text: &str) -> Option<f64> {
    text.lines()
        .find_map(|line| line.trim().strip_prefix("energy "))
        .and_then(|rest| rest.trim().strip_suffix(" Ha"))
        .and_then(|value| value.trim().parse().ok())
}

#[test]
fn multiplicity_without_scf_type_is_refused_and_leaves_no_result() {
    let dir = scratch("t18-guard");
    let input = dir.join("o2_no_scf_type.yaml");
    std::fs::write(&input, o2_triplet(None)).expect("写临时输入");

    let (code, text) = run(&dir, &["run", "o2_no_scf_type.yaml", "--out", "out"]);
    assert_eq!(
        code, 1,
        "三重态不写 scf_type 必须按输入错误退出 1（旧行为是算成闭壳层、退出 0）：\n{text}"
    );
    assert!(
        text.contains("scf_type") && text.contains("multiplicity = 3"),
        "提示里必须同时点名字段与多重度：\n{text}"
    );
    assert!(
        !dir.join("out")
            .join("o2_no_scf_type")
            .join("result.json")
            .is_file(),
        "被拒绝的输入不该留下 result.json（拒绝必须发生在任何数值之前）：{}",
        dir.join("out").join("o2_no_scf_type").display()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_same_input_runs_once_scf_type_is_given() {
    let dir = scratch("t18-rohf");
    std::fs::write(dir.join("o2_rohf.yaml"), o2_triplet(Some("rohf"))).expect("写临时输入");

    let (code, text) = run(&dir, &["run", "o2_rohf.yaml", "--out", "out"]);
    assert_eq!(code, 0, "显式 scf_type: rohf 应跑通：\n{text}");
    // 这个值由**示例自己的几何**决定：示例（与 `pyscf_open_shell_reference.py` 一致）用
    // R = 1.208 Å，得 −147.6321910352 Ha；`sc01c_rohf_engine` 用 1.2075 Å，得
    // −147.6321670245 Ha（PySCF −147.6321669907）。两者不是同一个体系，差 2.4e-5 全部
    // 来自那 0.0005 Å —— 所以这里断言的是本几何的值，不是 SC-01c 的参考值。
    let energy = energy_hartree(&text).expect("stdout 里应有 `energy ... Ha` 行");
    assert!(
        (energy + 147.632_191_035_2_f64).abs() < 1.0e-6,
        "ROHF 能量应落在 −147.6321910 Ha 附近，实际 {energy:.10}：\n{text}"
    );
    assert!(
        dir.join("out")
            .join("o2_rohf")
            .join("result.json")
            .is_file(),
        "跑通的作业应留下 result.json"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_closed_shell_default_still_works_for_singlets() {
    let dir = scratch("t18-singlet");
    let (code, text) = run(
        &examples(),
        &[
            "run",
            "h2_sto3g_svwn5.yaml",
            "--out",
            dir.to_string_lossy().as_ref(),
        ],
    );
    assert_eq!(
        code, 0,
        "单重态不写 scf_type 的老路径不能被新守卫误伤：\n{text}"
    );
    assert!(
        dir.join("h2_sto3g_svwn5").join("result.json").is_file(),
        "应留下 result.json"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
