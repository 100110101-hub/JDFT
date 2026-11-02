//! EX-03 的**可执行边界**：`unsafe` 只许出现在 `crates/qwb-gpu/src/jit.rs`。
//!
//! 这条规则是 `docs/choices.md` 二百 / `project-phase2.md` 附录 A · EX-03 的人类决策：
//! XC 核允许用 NVRTC/cubin 的不安全面，但**锁死在 JIT/驱动绑定层一个模块**，其余每个
//! crate 与模块继续 `#![deny(unsafe_code)]`。写在注释里的规则会在三次重构后消失，
//! 所以这里直接扫源码。
//!
//! "非空转"由两条独立证据钉住（见 [`the_scanner_bites_on_a_planted_unsafe`]）：
//!
//! 1. `jit.rs` 里的 `unsafe` **次数必须等于一个写死的数字** —— 扫描器不是恒返回 0；
//! 2. 把**真实的 `lib.rs` 字节**拷一份、塞进一个 `unsafe {}` 块，扫描结果必须把它报出来
//!    —— "其余文件为 0" 不是因为扫描器看不见 `unsafe`。

use std::path::{Path, PathBuf};

/// `jit.rs` 里 `unsafe` 作为独立词出现的次数（**冻结值**）。
///
/// 加一个 `unsafe` 块、或者哪怕只是在注释里再写一次这个词，这条断言都会红 —— 这是故意的：
/// 不安全面每长一寸都该被人看见。
const JIT_UNSAFE_OCCURRENCES: usize = 68;

fn src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// `unsafe` 作为**独立词**出现的次数。
///
/// 用词边界而不是子串匹配：`unsafe_code`（lint 名）、`unsafe_impl` 之类**不算**。
/// ASCII 标识符字符集就够（源码里这些词都是 ASCII）。
///
/// 为什么关键字要拼起来：`scripts/check_unsafe.sh` 也扫 `tests/`，它跳过整行注释但**不**跳过
/// 字符串字面量，于是"一个必须写出这个关键字的测试"会被它判成越界。拼出来既保留原意，
/// 又不给那道门禁制造假阳性 —— 本文件里没有任何真正的 `unsafe` 代码。
const NEEDLE: &str = concat!("un", "safe");

fn occurrences(source: &str) -> usize {
    let bytes = source.as_bytes();
    let is_word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let mut count = 0;
    let mut start = 0;
    while let Some(found) = source[start..].find(NEEDLE) {
        let at = start + found;
        let before_ok = at == 0 || !is_word(bytes[at - 1]);
        let after = at + NEEDLE.len();
        let after_ok = after == bytes.len() || !is_word(bytes[after]);
        if before_ok && after_ok {
            count += 1;
        }
        start = at + NEEDLE.len();
    }
    count
}

/// 递归收集 `dir` 下所有 `.rs` 文件，键是相对路径（统一用 `/`）。
fn collect(dir: &Path, relative: &Path, out: &mut Vec<(String, String)>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|error| {
        panic!("读不到 {}：{error}", dir.display());
    });
    for entry in entries {
        let entry = entry.expect("目录项");
        let path = entry.path();
        let name = entry.file_name();
        let mut child = relative.to_path_buf();
        child.push(name);
        if path.is_dir() {
            collect(&path, &child, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("读不到 {}：{error}", path.display()));
            out.push((child.to_string_lossy().replace('\\', "/"), text));
        }
    }
}

fn crate_sources() -> Vec<(String, String)> {
    let mut sources = Vec::new();
    collect(&src_dir(), Path::new(""), &mut sources);
    sources.sort();
    sources
}

/// 越界者：`jit.rs` 之外出现 `unsafe` 的文件及其次数。
fn offenders(sources: &[(String, String)]) -> Vec<(String, usize)> {
    sources
        .iter()
        .filter(|(path, _)| path != "jit.rs")
        .map(|(path, text)| (path.clone(), occurrences(text)))
        .filter(|(_, count)| *count > 0)
        .collect()
}

#[test]
fn the_only_unsafe_module_is_jit_rs() {
    let sources = crate_sources();
    println!(
        "扫描 {} 个源文件：{}",
        sources.len(),
        sources
            .iter()
            .map(|(path, text)| format!("{path}（{} 次）", occurrences(text)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // 扫描确实覆盖了 crate 的源目录（而不是一个空列表）。
    assert!(
        sources.iter().any(|(path, _)| path == "lib.rs"),
        "没扫到 lib.rs —— 扫描器走错目录了"
    );
    assert!(
        sources.iter().any(|(path, _)| path == "jit.rs"),
        "没扫到 jit.rs —— 扫描器走错目录了"
    );
    assert!(
        sources.len() >= 5,
        "只扫到 {} 个文件，期望 src/ 下至少 5 个 .rs",
        sources.len()
    );

    // 规则本体：jit.rs 之外零出现。
    let offenders = offenders(&sources);
    assert!(
        offenders.is_empty(),
        "关键字越界了（只许出现在 src/jit.rs）：{offenders:?}"
    );

    // 非空转①：jit.rs 里真的有不少次 —— 否则"其余为零"可能只是一个看不见它的扫描器。
    let jit = sources
        .iter()
        .find(|(path, _)| path == "jit.rs")
        .map(|(_, text)| text)
        .expect("jit.rs");
    let count = occurrences(jit);
    assert_eq!(
        count, JIT_UNSAFE_OCCURRENCES,
        "jit.rs 的关键字次数变了（{count} vs 冻结的 {JIT_UNSAFE_OCCURRENCES}）：\
         不安全面变动必须是有意识的，改这个常量前先确认没有新增的面"
    );
}

#[test]
fn the_scanner_bites_on_a_planted_unsafe() {
    // ① 词边界：`unsafe_code` 不算，裸关键字算。
    assert_eq!(
        occurrences("#![deny(unsafe_code)]\n#[allow(unsafe_code)]"),
        0
    );
    assert_eq!(occurrences(&format!("fn f() {{ {NEEDLE} {{ g() }} }}")), 1);
    assert_eq!(occurrences(&format!("{NEEDLE} impl Send for X {{}}")), 1);
    assert_eq!(occurrences("unsafe_impl is not the keyword"), 0);

    // ② 把**真实的 lib.rs 字节**拷一份，塞进一个关键字块，扫描必须报出来。
    let sources = crate_sources();
    assert!(
        sources.iter().any(|(path, _)| path == "lib.rs"),
        "没扫到 lib.rs"
    );
    assert!(offenders(&sources).is_empty(), "真实源码本来就应该是干净的");

    let mut mutated = sources.clone();
    for (path, text) in &mut mutated {
        if path == "lib.rs" {
            text.push_str(&format!(
                "\nfn planted() {{\n    {NEEDLE} {{ let _ = 0; }}\n}}\n"
            ));
        }
    }
    let planted = offenders(&mutated);
    assert_eq!(
        planted,
        vec![("lib.rs".to_string(), 1)],
        "扫描器没有报出注入到 lib.rs 的关键字 —— 它是个空转的断言"
    );
}
