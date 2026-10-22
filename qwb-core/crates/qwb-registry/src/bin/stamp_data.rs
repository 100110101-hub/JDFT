//! `qwb-stamp-data`：给数据文件打 `sha256`（G2 的“工具生成”）。
//!
//! ```text
//! qwb-stamp-data <file>...           # 就地打戳
//! qwb-stamp-data <file>... --check   # 只校验，不写
//! ```
//!
//! ⚠ 写回会**丢弃 YAML 注释**（数据文件是工具产物；见 `qwb_registry::sha256` 模块文档）。

use std::path::Path;
use std::process::ExitCode;

use qwb_registry::sha256::stamp_bytes;
use qwb_registry::verify::{DataKind, infer_kind, verify_bytes};

fn main() -> ExitCode {
    let mut paths: Vec<String> = Vec::new();
    let mut check_only = false;
    for arg in std::env::args().skip(1) {
        if arg == "--check" {
            check_only = true;
        } else {
            paths.push(arg);
        }
    }
    if paths.is_empty() {
        eprintln!("usage: qwb-stamp-data <file>... [--check]");
        return ExitCode::from(1);
    }

    let mut failures = 0usize;
    for path in &paths {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) => {
                eprintln!("{path}: cannot read: {err}");
                failures += 1;
                continue;
            }
        };
        let kind = infer_kind(Path::new(path)).unwrap_or(DataKind::Basis);
        if check_only {
            let report = verify_bytes(kind, path, &bytes);
            if report.is_ok() {
                println!("ok      {path}");
            } else {
                failures += 1;
                for issue in report.errors() {
                    eprintln!("{}: [{}] {}", path, issue.code, issue.message);
                }
            }
            continue;
        }
        match stamp_bytes(&bytes) {
            Ok((digest, stamped)) => match std::fs::write(path, &stamped) {
                Ok(()) => println!("{digest}  {path}"),
                Err(err) => {
                    eprintln!("{path}: cannot write: {err}");
                    failures += 1;
                }
            },
            Err(err) => {
                eprintln!("{path}: {err}");
                failures += 1;
            }
        }
    }

    if failures > 0 {
        eprintln!("{failures} file(s) failed");
        return ExitCode::from(3);
    }
    ExitCode::SUCCESS
}
