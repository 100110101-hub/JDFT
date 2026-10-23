//! 复现环境（K3 `env.json` / G5）：`EnvInfo` 的真实填充。
//!
//! 字段来源全部是**实际读取**，不是占位：
//!
//! - `rustc_version`：运行 `rustc --version`；
//! - `cargo_lock_sha256`：编译期嵌入的 `Cargo.lock` 字节的 sha256；
//! - `toolchain_channel`：仓库根 `rust-toolchain.toml` 的 channel（编译期嵌入）；
//! - `data_snapshot`：注册表清单的 `path -> sha256`；
//! - `threads`：目前无并行，记 1（有 rayon 后改成实际线程数）；
//! - `created_at_utc`：手写历法换算的 ISO 8601（不引入时间库）。

use qwb_engine_api::{DataSnapshot, EnvInfo};

/// 工作区 `Cargo.lock` 内容（编译期嵌入）。
pub const CARGO_LOCK: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"));

/// `rust-toolchain.toml` 内容（编译期嵌入，用于取 channel）。
pub const RUST_TOOLCHAIN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../rust-toolchain.toml"
));

/// 引擎版本。
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// 从 `rust-toolchain.toml` 里取 `channel = "..."`。
#[must_use]
pub fn toolchain_channel() -> Option<String> {
    for line in RUST_TOOLCHAIN.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("channel") {
            let rest = rest.trim_start().strip_prefix('=')?.trim();
            return Some(rest.trim_matches('"').to_string());
        }
    }
    None
}

/// 运行一个命令并取第一行输出（失败时返回 `unknown`）。
fn command_version(program: &str, argument: &str) -> String {
    std::process::Command::new(program)
        .arg(argument)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| format!("{program} (unavailable)"))
}

/// 构造 `EnvInfo`。
#[must_use]
pub fn build(registry: &qwb_registry::Registry) -> EnvInfo {
    let snapshot: DataSnapshot = registry.snapshot();
    let env = EnvInfo::new(
        ENGINE_VERSION,
        command_version("rustc", "--version"),
        qwb_registry::sha256::sha256_hex(CARGO_LOCK),
        snapshot,
        1,
        format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        iso8601_utc_now(),
    );
    match toolchain_channel() {
        Some(channel) => env.with_toolchain_channel(channel),
        None => env,
    }
}

/// 当前 UTC 时间，ISO 8601。
#[must_use]
pub fn iso8601_utc_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    iso8601_from_unix(seconds)
}

/// Unix 秒 → ISO 8601 UTC（Howard Hinnant 的 civil_from_days）。
#[must_use]
pub fn iso8601_from_unix(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let time_of_day = seconds % 86_400;
    let hour = time_of_day / 3600;
    let minute = (time_of_day % 3600) / 60;
    let second = time_of_day % 60;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year_shifted = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 {
        year_shifted + 1
    } else {
        year_shifted
    };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::{iso8601_from_unix, toolchain_channel};

    #[test]
    fn iso8601_conversion_is_correct() {
        assert_eq!(iso8601_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_from_unix(1792710970), "2026-10-22T23:16:10Z");
        // 闰日
        assert_eq!(iso8601_from_unix(1709208000), "2024-02-29T12:00:00Z");
    }

    #[test]
    fn toolchain_channel_is_read_from_the_repo() {
        let channel = toolchain_channel().expect("rust-toolchain.toml has a channel");
        assert!(!channel.is_empty());
        assert!(channel.starts_with('1'), "channel = {channel}");
    }
}
