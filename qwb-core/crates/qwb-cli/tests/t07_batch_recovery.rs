//! T7 第四条判据：`qwb batch` 的**持久化队列 + 崩溃恢复**。
//!
//! 语义（与冻结的 K3 表一致）：
//! * 首次运行把每个输入登记成 `queued` 并写 `<out>/queue.json`，逐个走 `queued → running →
//!   converged/failed`；
//! * 进程被杀之后再跑同一条命令：**`queued` 的接着做**；停在 `running` 的**不能**回到
//!   `queued`（K3 没有那条边），必须报告出来并用新 task id 重提，命令以非零退出码结束。
//!
//! 这个测试不靠 `kill -9` 的时序，而是**直接构造一个"被杀的现场"**（队列里一条 running、
//! 一条 queued，结果目录都不存在），再跑同一命令，验证上面两条语义。
use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn data_root() -> PathBuf {
    manifest_dir().join("../../../qwb-data")
}

fn example_input() -> PathBuf {
    manifest_dir().join("../../examples/h2_sto3g_svwn5.yaml")
}

fn temp_dir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("qwb_batch_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("temp dir");
    path
}

fn qwb() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qwb"));
    command.env("QWB_DATA_ROOT", data_root());
    command
}

/// 在 `dir` 里放 `count` 份输入（内容相同，文件名不同 ⇒ task id 不同）。
fn write_inputs(dir: &Path, count: usize) -> Vec<PathBuf> {
    let source = std::fs::read(example_input()).expect("example input");
    let mut out = Vec::new();
    for index in 0..count {
        let path = dir.join(format!("job{index}.yaml"));
        std::fs::write(&path, &source).expect("write input");
        out.push(path);
    }
    out
}

#[test]
fn batch_runs_every_input_and_records_the_queue() {
    let out_root = temp_dir("fresh");
    let inputs_dir = out_root.join("jobs");
    std::fs::create_dir_all(&inputs_dir).expect("jobs dir");
    write_inputs(&inputs_dir, 2);

    let output = qwb()
        .arg("batch")
        .arg(format!("{}/*.yaml", inputs_dir.display()))
        .arg("--parallel")
        .arg("2")
        .arg("--out")
        .arg(&out_root)
        .output()
        .expect("spawn qwb batch");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "batch 应全部成功: {}\n{stdout}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("2 converged"), "summary 不对:\n{stdout}");

    let queue = std::fs::read_to_string(out_root.join("queue.json")).expect("queue.json");
    assert_eq!(queue.matches("\"converged\"").count(), 2, "队列:\n{queue}");
    for index in 0..2 {
        let task = out_root.join(format!("job{index}"));
        assert!(task.join("result.json").is_file(), "缺 {}", task.display());
        assert!(task.join("env.json").is_file());
    }
    let _ = std::fs::remove_dir_all(&out_root);
}

#[test]
fn batch_resumes_queued_tasks_and_reports_interrupted_ones() {
    let out_root = temp_dir("crash");
    let inputs_dir = out_root.join("jobs");
    std::fs::create_dir_all(&inputs_dir).expect("jobs dir");
    let inputs = write_inputs(&inputs_dir, 2);

    // 构造"被 kill -9 的现场"：job0 卡在 running（结果目录没写出来），job1 还 queued。
    let queue = serde_json::json!([
        {
            "input": inputs[0].display().to_string(),
            "task_id": "job0",
            "state": "running"
        },
        {
            "input": inputs[1].display().to_string(),
            "task_id": "job1",
            "state": "queued"
        }
    ]);
    std::fs::write(
        out_root.join("queue.json"),
        serde_json::to_vec_pretty(&queue).expect("queue json"),
    )
    .expect("write queue");

    let output = qwb()
        .arg("batch")
        .arg(format!("{}/*.yaml", inputs_dir.display()))
        .arg("--out")
        .arg(&out_root)
        .output()
        .expect("spawn qwb batch");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // ① queued 的 job1 必须被做掉
    assert!(
        out_root.join("job1").join("result.json").is_file(),
        "queued 的任务没有被恢复执行:\n{stdout}\n{stderr}"
    );
    // ② running 的 job0 不能被"假装恢复"（K3 没有 running→queued 边）
    // 真的被 kill -9 时，rtifacts/ 目录可能已经建出来了，但 result.json 不会有；
    // 这里按"没有结果"判定，与真实崩溃后的现场一致。
    assert!(
        !out_root.join("job0").join("result.json").is_file(),
        "running 的任务不应被静默重跑"
    );
    assert!(
        stderr.contains("job0") && stderr.contains("running"),
        "应报告被中断的任务:\n{stderr}"
    );
    // ③ 有中断 ⇒ 命令必须以非零退出码结束
    assert!(
        !output.status.success(),
        "有中断任务时 batch 不应返回成功:\n{stdout}"
    );

    // ④ 把 job0 重新登记成 queued（模拟"用新 task id 重提"），再跑一次应当全部成功
    let queue = serde_json::json!([
        {
            "input": inputs[0].display().to_string(),
            "task_id": "job0",
            "state": "queued"
        },
        {
            "input": inputs[1].display().to_string(),
            "task_id": "job1",
            "state": "converged"
        }
    ]);
    std::fs::write(
        out_root.join("queue.json"),
        serde_json::to_vec_pretty(&queue).expect("queue json"),
    )
    .expect("write queue");
    let again = qwb()
        .arg("batch")
        .arg(format!("{}/*.yaml", inputs_dir.display()))
        .arg("--out")
        .arg(&out_root)
        .output()
        .expect("spawn qwb batch");
    assert!(
        again.status.success(),
        "重提之后应全部成功:\n{}\n{}",
        String::from_utf8_lossy(&again.stdout),
        String::from_utf8_lossy(&again.stderr)
    );
    assert!(out_root.join("job0").join("result.json").is_file());

    let _ = std::fs::remove_dir_all(&out_root);
}
