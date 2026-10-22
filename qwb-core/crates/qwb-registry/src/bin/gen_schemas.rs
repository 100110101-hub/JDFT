//! `qwb-gen-schemas`：把 K1 schema 从 Rust 类型导出到 `qwb-contracts/schema/`。
//!
//! ```text
//! qwb-gen-schemas <out_dir>            # 生成/覆盖
//! qwb-gen-schemas <out_dir> --check    # 只校验，漂移则退出码 1（CI 用）
//! ```
//!
//! 同时产出 `<kind>.schema.json`（工具读）与 `<kind>.schema.yaml`（人读 + K1 规定的文件名）。

use std::path::PathBuf;
use std::process::ExitCode;

use qwb_registry::schema;
use qwb_registry::verify::DataKind;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let out_dir = match args.next() {
        Some(dir) => PathBuf::from(dir),
        None => {
            eprintln!("usage: qwb-gen-schemas <out_dir> [--check]");
            return ExitCode::from(1);
        }
    };
    let check_only = args.any(|a| a == "--check");

    if !check_only && let Err(err) = std::fs::create_dir_all(&out_dir) {
        eprintln!("cannot create {}: {err}", out_dir.display());
        return ExitCode::from(1);
    }

    let mut drift = 0usize;
    for kind in DataKind::ALL {
        let schema = schema::schema_for_kind(kind);
        let generated = match render(&schema) {
            Ok(files) => files,
            Err(err) => {
                eprintln!("{}: cannot render schema: {err}", kind.dir_name());
                return ExitCode::from(4);
            }
        };
        for (extension, content) in generated {
            let path = out_dir.join(format!("{}.schema.{extension}", kind.dir_name()));
            if check_only {
                match std::fs::read_to_string(&path) {
                    Ok(existing) if existing == content => {}
                    Ok(_) => {
                        eprintln!(
                            "drift: {} differs from the Rust types (run qwb-gen-schemas)",
                            path.display()
                        );
                        drift += 1;
                    }
                    Err(err) => {
                        eprintln!("missing: {} ({err})", path.display());
                        drift += 1;
                    }
                }
            } else if let Err(err) = std::fs::write(&path, &content) {
                eprintln!("cannot write {}: {err}", path.display());
                return ExitCode::from(4);
            } else {
                println!("wrote {}", path.display());
            }
        }
    }

    if drift > 0 {
        eprintln!("{drift} schema file(s) out of date");
        return ExitCode::from(1);
    }
    if check_only {
        println!("all K1 schemas match the Rust types");
    }
    ExitCode::SUCCESS
}

fn render(schema: &schemars::Schema) -> Result<[(String, String); 2], String> {
    let json = schema::to_json(schema).map_err(|e| e.to_string())?;
    let yaml = schema::to_yaml(schema)?;
    Ok([("json".to_string(), json), ("yaml".to_string(), yaml)])
}
