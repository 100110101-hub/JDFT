//! `qwb` — QuantumWorkbench 命令行（**K5 冻结签名**）。
//!
//! ```text
//! qwb run    input.yaml [--out DIR] [--threads N]
//! qwb batch  DIR/*.yaml [--parallel N]
//! qwb inspect DIR
//! qwb replay DIR [--dry-run]
//! qwb data list|show|verify
//! ```
//!
//! 退出码（K5 冻结）：`0` 成功 / `1` 输入错误 / `2` 收敛失败 / `3` 数据校验失败 / `4` 内部错误。
//!
//! # C 轨的边界（诚实说明）
//!
//! 本 crate 在 C 轨只负责**契约与结构**。真正的数值执行（积分/XC/SCF/优化）来自
//! I/X/S/G 轨与顶层 `qwb-engine`，`P-07` 才把 `run` 接到引擎上。因此：
//!
//! - `run` / `batch`：**完整做输入解析与校验**（这一步是真的），然后在“没有引擎后端”
//!   处返回退出码 `4`。不会假装算过。
//! - `inspect` / `replay --dry-run`：完全实装（结构性检查 + G5 复现门禁）。
//! - `replay`（不带 `--dry-run`）：需要引擎重算，属 P-08，返回 `4`。
//! - `data list|show|verify`：完全实装。
#![deny(unsafe_code)]
#![deny(missing_docs)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use qwb_engine_api::{EngineError, JobInput};
use qwb_io::{QwbContainer, ResultDir};
use qwb_registry::registry::RegistryManifest;
use qwb_registry::verify::{self, Report};

/// K5 冻结的命令行。
#[derive(Debug, Parser)]
#[command(
    name = "qwb",
    version,
    about = "QuantumWorkbench 分子 DFT 命令行",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// 执行一个 input.yaml
    Run(RunArgs),
    /// 批量执行一个目录下的 *.yaml
    Batch(BatchArgs),
    /// 检查一个结果目录或 .qwb 项目
    Inspect(InspectArgs),
    /// 用记录的环境重算并比对（复现）
    Replay(ReplayArgs),
    /// 数据仓库操作
    Data(DataArgs),
}

#[derive(Debug, Args)]
struct RunArgs {
    /// 输入文件（K4 input.yaml）
    input_yaml: PathBuf,
    /// 结果输出目录
    #[arg(long)]
    out: Option<PathBuf>,
    /// 线程数（覆盖 input.yaml 的 options.threads）
    #[arg(long)]
    threads: Option<u32>,
}

#[derive(Debug, Args)]
struct BatchArgs {
    /// `DIR/*.yaml` 形式的模式或目录
    pattern: String,
    /// 并行度
    #[arg(long)]
    parallel: Option<u32>,
}

#[derive(Debug, Args)]
struct InspectArgs {
    /// 结果目录，或 `.qwb` 容器
    path: PathBuf,
}

#[derive(Debug, Args)]
struct ReplayArgs {
    /// 结果目录
    path: PathBuf,
    /// 只检查可复现性，不重算
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct DataArgs {
    #[command(subcommand)]
    command: DataCommand,
}

#[derive(Debug, Subcommand)]
enum DataCommand {
    /// 列出注册表条目
    List,
    /// 显示一个条目
    Show {
        /// 数据键名
        name: String,
    },
    /// 校验数据文件（无参数时校验整个注册表）
    Verify {
        /// 要校验的文件；省略则校验 `registry.yaml` 及其全部条目
        paths: Vec<PathBuf>,
    },
}

/// CLI 层错误：带 K5 退出码。
#[derive(Debug)]
struct CliError {
    code: i32,
    message: String,
}

impl CliError {
    fn input(message: impl Into<String>) -> Self {
        Self {
            code: 1,
            message: message.into(),
        }
    }
    fn data(message: impl Into<String>) -> Self {
        Self {
            code: 3,
            message: message.into(),
        }
    }
    fn internal(message: impl Into<String>) -> Self {
        Self {
            code: 4,
            message: message.into(),
        }
    }
    fn from_engine(err: &EngineError) -> Self {
        Self {
            code: err.exit_code(),
            message: err.to_string(),
        }
    }
}

type CliResult<T> = Result<T, CliError>;

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("qwb: {}", err.message);
            ExitCode::from(u8::try_from(err.code).unwrap_or(4))
        }
    }
}

fn dispatch(cli: &Cli) -> CliResult<()> {
    match &cli.command {
        Command::Run(args) => cmd_run(args),
        Command::Batch(args) => cmd_batch(args),
        Command::Inspect(args) => cmd_inspect(args),
        Command::Replay(args) => cmd_replay(args),
        Command::Data(args) => match &args.command {
            DataCommand::List => cmd_data_list(),
            DataCommand::Show { name } => cmd_data_show(name),
            DataCommand::Verify { paths } => cmd_data_verify(paths),
        },
    }
}

// ---------------------------------------------------------------- run / batch

/// 收集日志的最小 `Reporter` 实现（CLI 不做进度动画）。
struct CliReporter {
    // `Reporter: Send + Sync`（K2）：必须用线程安全的容器。
    lines: std::sync::Mutex<Vec<String>>,
}

impl CliReporter {
    fn new() -> Self {
        Self {
            lines: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl qwb_engine_api::Reporter for CliReporter {
    fn progress(&self, stage: &str, pct: f64) {
        if let Ok(mut lines) = self.lines.lock() {
            lines.push(format!("[{pct:5.1}%] {stage}"));
        }
    }

    fn log(&self, line: &str) {
        if let Ok(mut lines) = self.lines.lock() {
            lines.push(line.to_string());
        }
    }

    fn is_cancelled(&self) -> bool {
        false
    }
}

impl CliReporter {
    fn into_lines(self) -> Vec<String> {
        self.lines.into_inner().unwrap_or_default()
    }
}

fn cmd_run(args: &RunArgs) -> CliResult<()> {
    let input = load_input(&args.input_yaml)?;
    if let Some(threads) = args.threads {
        eprintln!(
            "qwb: note: --threads {threads} is recorded but this build is single-threaded (I-04)"
        );
    }

    let root = find_data_root()?;
    let registry = qwb_registry::Registry::open(&root).map_err(|e| CliError::from_engine(&e))?;
    let handle = registry.engine_handle();
    let backend = Box::new(qwb_engine::RksBackend::new(registry));
    let engine = qwb_engine_api::Engine::with_backend(&handle, backend)
        .map_err(|e| CliError::from_engine(&e))?;

    let reporter = CliReporter::new();
    let result = engine
        .run(&input, &reporter)
        .map_err(|e| CliError::from_engine(&e))?;
    let lines = reporter.into_lines();

    // 结果目录（K3 布局）。任务 id：输入文件名（去扩展名）。
    let task_id = args
        .input_yaml
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("task")
        .to_string();
    let out_root = args.out.clone().unwrap_or_else(|| PathBuf::from("results"));
    let task_dir = out_root.join(&task_id);
    std::fs::create_dir_all(task_dir.join("artifacts"))
        .map_err(|e| CliError::internal(format!("cannot create {}: {e}", task_dir.display())))?;

    let write = |name: &str, bytes: Vec<u8>| -> CliResult<()> {
        std::fs::write(task_dir.join(name), bytes)
            .map_err(|e| CliError::internal(format!("cannot write {name}: {e}")))
    };
    let input_yaml = serde_yaml::to_string(&input)
        .map_err(|e| CliError::internal(format!("cannot serialise input: {e}")))?;
    write("input.yaml", input_yaml.into_bytes())?;
    write(
        "result.json",
        serde_json::to_vec_pretty(&result)
            .map_err(|e| CliError::internal(format!("cannot serialise result: {e}")))?,
    )?;
    write(
        "env.json",
        serde_json::to_vec_pretty(&result.env)
            .map_err(|e| CliError::internal(format!("cannot serialise env: {e}")))?,
    )?;
    write(
        "data_snapshot.json",
        serde_json::to_vec_pretty(&result.data_snapshot)
            .map_err(|e| CliError::internal(format!("cannot serialise snapshot: {e}")))?,
    )?;
    let mut log = lines.join("\n");
    log.push('\n');
    log.push_str(&result.log);
    log.push('\n');
    write("log.txt", log.into_bytes())?;

    println!("energy  {:.10} Ha", result.energy_hartree);
    println!("scf     converged = {}", result.scf_converged);
    println!("task    {}", task_dir.display());
    Ok(())
}
fn cmd_batch(args: &BatchArgs) -> CliResult<()> {
    let files = expand_pattern(&args.pattern)?;
    if files.is_empty() {
        return Err(CliError::input(format!(
            "no *.yaml matched {}",
            args.pattern
        )));
    }
    let mut ok = 0usize;
    for file in &files {
        load_input(file)?;
        ok += 1;
    }
    println!(
        "{ok} input(s) parsed and validated, parallel = {}",
        args.parallel.unwrap_or(0)
    );
    Err(CliError::internal(
        "batch execution needs the engine backend (P-07); inputs were only validated",
    ))
}

fn load_input(path: &Path) -> CliResult<JobInput> {
    let bytes = std::fs::read(path)
        .map_err(|e| CliError::input(format!("cannot read {}: {e}", path.display())))?;
    let input: JobInput = serde_yaml::from_slice(&bytes).map_err(|e| {
        CliError::input(format!("{} is not a valid input.yaml: {e}", path.display()))
    })?;
    input.validate().map_err(|e| CliError::from_engine(&e))?;
    Ok(input)
}

/// 展开 `DIR/*.yaml` 或目录。
fn expand_pattern(pattern: &str) -> CliResult<Vec<PathBuf>> {
    let (dir, filter) = match pattern.rsplit_once(['/', '\\']) {
        Some((dir, tail)) => (PathBuf::from(dir), tail.trim()),
        None => (PathBuf::from("."), pattern.trim()),
    };
    let suffix = filter.strip_prefix('*').unwrap_or(filter);
    let suffix = suffix.trim_start_matches('.');
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CliError::input(format!("cannot read directory {}: {e}", dir.display())))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => suffix.is_empty() || ext == suffix,
            None => false,
        })
        .collect();
    files.sort();
    Ok(files)
}

// -------------------------------------------------------------------- inspect

fn cmd_inspect(args: &InspectArgs) -> CliResult<()> {
    if args.path.is_dir() {
        let layout = ResultDir::new(&args.path);
        println!("kind    result directory");
        println!("path    {}", args.path.display());
        for name in qwb_io::layout::REQUIRED_FILES {
            println!("file    {name:<20} {}", mark(layout.has(name)));
        }
        println!(
            "file    {:<20} {}",
            qwb_io::layout::ENV_FILE,
            mark(layout.has_env_json())
        );
        match layout.check() {
            Ok(()) => {
                let env = layout
                    .read_env()
                    .map_err(|e| CliError::data(e.to_string()))?;
                println!(
                    "env     rustc {}, threads {}, os {}",
                    env.rustc_version, env.threads, env.os
                );
                println!("env     data snapshot: {} file(s)", env.data_snapshot.len());
                println!("status  ok");
                Ok(())
            }
            Err(err) => Err(CliError {
                code: err.exit_code(),
                message: err.to_string(),
            }),
        }
    } else if args.path.is_file() {
        let container =
            QwbContainer::open(&args.path).map_err(|e| CliError::data(e.to_string()))?;
        println!("kind    .qwb container (K4)");
        println!("path    {}", args.path.display());
        println!("meta    name = {}", container.meta().name);
        println!(
            "meta    created = {}, app = {}",
            container.meta().created,
            container.meta().app_version
        );
        println!("meta    tasks = {}", container.meta().task_count);
        println!("entries {}", container.entry_names().len());
        for name in container.entry_names() {
            println!("entry   {name}");
        }
        container
            .validate()
            .map_err(|e| CliError::data(e.to_string()))?;
        println!("status  ok");
        Ok(())
    } else {
        Err(CliError::data(format!(
            "{} is neither a result directory nor a .qwb file",
            args.path.display()
        )))
    }
}

fn mark(present: bool) -> &'static str {
    if present { "ok" } else { "MISSING" }
}

// --------------------------------------------------------------------- replay

fn cmd_replay(args: &ReplayArgs) -> CliResult<()> {
    let layout = ResultDir::new(&args.path);
    let env = layout.read_env().map_err(|err| CliError {
        code: err.exit_code(),
        message: err.to_string(),
    })?;
    println!("replay  {}", args.path.display());
    println!("rustc   {}", env.rustc_version);
    if let Some(channel) = &env.toolchain_channel {
        println!("channel {channel}");
    }
    println!("lock    sha256 {}", env.cargo_lock_sha256);
    println!("threads {}", env.threads);
    println!("snapshot {} file(s)", env.data_snapshot.len());
    for (path, sha) in env.data_snapshot.iter() {
        println!("data    {path} {sha}");
    }
    if args.dry_run {
        println!("status  reproducible environment recorded (dry-run, nothing recomputed)");
        return Ok(());
    }
    Err(CliError::internal(
        "recompute-on-replay needs the engine backend (P-08); use --dry-run to check the record",
    ))
}

// ----------------------------------------------------------------- data verbs

fn find_data_root() -> CliResult<PathBuf> {
    if let Ok(value) = std::env::var("QWB_DATA_ROOT") {
        let path = PathBuf::from(value);
        if path.is_dir() {
            return Ok(path);
        }
        return Err(CliError::data(format!(
            "QWB_DATA_ROOT={} is not a directory",
            path.display()
        )));
    }
    for candidate in ["qwb-data", "../qwb-data", "../../qwb-data"] {
        let path = PathBuf::from(candidate);
        if path.is_dir() {
            return Ok(path);
        }
    }
    Err(CliError::data(
        "cannot find qwb-data; set QWB_DATA_ROOT to the data repository root",
    ))
}

fn load_manifest(root: &Path) -> CliResult<RegistryManifest> {
    let path = root.join("registry.yaml");
    let bytes = std::fs::read(&path)
        .map_err(|e| CliError::data(format!("cannot read {}: {e}", path.display())))?;
    serde_yaml::from_slice(&bytes).map_err(|e| {
        CliError::data(format!(
            "{} is not a valid registry.yaml: {e}",
            path.display()
        ))
    })
}

fn cmd_data_list() -> CliResult<()> {
    let root = find_data_root()?;
    let manifest = load_manifest(&root)?;
    let issues = manifest.check();
    if !issues.is_empty() {
        for issue in &issues {
            eprintln!("registry: {issue}");
        }
        return Err(CliError::data("registry.yaml failed validation"));
    }
    println!(
        "registry {} v{} ({})",
        manifest.name,
        manifest.version,
        root.display()
    );
    for (domain, entry) in manifest.all_entries() {
        println!("{domain:<11} {:<24} {}", entry.name, entry.path);
    }
    Ok(())
}

fn cmd_data_show(name: &str) -> CliResult<()> {
    let root = find_data_root()?;
    let manifest = load_manifest(&root)?;
    let found = manifest
        .all_entries()
        .find(|(_, entry)| entry.name == name)
        .map(|(domain, entry)| (domain, entry.clone()));
    let Some((domain, entry)) = found else {
        return Err(CliError::input(format!("no data entry named `{name}`")));
    };
    println!("domain  {domain}");
    println!("name    {}", entry.name);
    println!("path    {}", root.join(&entry.path).display());
    println!("sha256  {}", entry.sha256);
    let report =
        verify::verify_file(&root.join(&entry.path)).map_err(|e| CliError::data(e.to_string()))?;
    println!(
        "verified {} ({} error(s), {} warning(s))",
        report.is_ok(),
        report.n_errors(),
        report.n_warnings()
    );
    if !report.is_ok() {
        print_report(&report);
        return Err(CliError::data(format!("{name} failed verification")));
    }
    Ok(())
}

fn cmd_data_verify(paths: &[PathBuf]) -> CliResult<()> {
    let reports = if paths.is_empty() {
        let root = find_data_root()?;
        verify::verify_registry(&root.join("registry.yaml"))
            .map_err(|e| CliError::data(e.to_string()))?
    } else {
        let mut reports = Vec::with_capacity(paths.len());
        for path in paths {
            reports.push(verify::verify_file(path).map_err(|e| CliError::data(e.to_string()))?);
        }
        reports
    };

    let mut failed = 0usize;
    for report in &reports {
        print_report(report);
        if !report.is_ok() {
            failed += 1;
        }
    }
    println!(
        "{} file(s) checked, {} passed, {} failed",
        reports.len(),
        reports.len() - failed,
        failed
    );
    if failed > 0 {
        return Err(CliError::data(format!(
            "{failed} file(s) failed verification"
        )));
    }
    Ok(())
}

fn print_report(report: &Report) {
    let status = if report.is_ok() { "ok  " } else { "FAIL" };
    println!(
        "{status} {:<11} {} ({} error(s), {} warning(s))",
        report.kind.dir_name(),
        report.path,
        report.n_errors(),
        report.n_warnings()
    );
    for issue in &report.issues {
        println!("       [{}] {}", issue.code, issue.message);
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_pattern, load_input};
    use std::path::PathBuf;

    fn fixture_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("qwb-cli-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const H2: &str = "molecule:
  atoms:
    - symbol: H
      position_angstrom: [0.0, 0.0, -0.37]
    - symbol: H
      position_angstrom: [0.0, 0.0, 0.37]
  charge: 0
  multiplicity: 1
basis_name: sto-3g
method_name: svwn5
grid_level: normal
solvent: vacuum
jobtype: sp
options:
  scf_max_iter: 128
  scf_energy_threshold_hartree: 1.0e-8
  scf_density_threshold: 1.0e-6
  diis_history: 8
  threads: 0
  max_opt_steps: 100
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4
  freq_step_angstrom: 0.01
  ecp_name: null
";

    #[test]
    fn valid_input_yaml_loads() {
        let dir = fixture_dir("good");
        let path = dir.join("h2.yaml");
        std::fs::write(&path, H2).unwrap();
        let input = load_input(&path).expect("valid");
        assert_eq!(input.molecule.n_atoms(), 2);
        assert_eq!(input.basis_name, "sto-3g");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn broken_input_yaml_is_input_error() {
        let dir = fixture_dir("bad");
        let path = dir.join("bad.yaml");
        std::fs::write(&path, "molecule: {}\n").unwrap();
        let err = load_input(&path).expect_err("invalid");
        assert_eq!(err.code, 1, "K5: 输入错误 → 退出码 1");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pattern_expansion_finds_yaml_only() {
        let dir = fixture_dir("glob");
        std::fs::write(dir.join("a.yaml"), H2).unwrap();
        std::fs::write(dir.join("b.yaml"), H2).unwrap();
        std::fs::write(dir.join("c.txt"), "x").unwrap();
        let pattern = format!("{}/*.yaml", dir.display());
        let files = expand_pattern(&pattern).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|p| p.extension().unwrap() == "yaml"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
