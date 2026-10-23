//! I-06 验收（project.md §6.3）：积分缓存**开/关结果 identical**。
//!
//! 只在 `--features integral-cache` 下编译。三条断言：
//!
//! 1. 缓存返回的张量与直接算的**逐位相同**（`==`，不是近似）；
//! 2. 同一基组+分子第二次取用是**命中**，且返回同一个 `Arc`（没有重算、没有复制）；
//! 3. 内容不同（换分子 / 换基组细节）不会串味。

#![cfg(feature = "integral-cache")]

use std::path::{Path, PathBuf};

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::cache::{IntegralCache, fingerprint};
use qwb_integrals::two_electron::two_electron;
use qwb_registry::Registry;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn h2() -> (InternalBasis, Molecule) {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    );
    let basis = InternalBasis::build(basis_k1, &molecule).expect("build");
    (basis, molecule)
}

#[test]
fn cache_returns_bit_identical_tensors() {
    let (basis, molecule) = h2();
    let direct = two_electron(&basis, &molecule).expect("direct");

    let mut cache = IntegralCache::new();
    let cached = cache.two_electron(&basis, &molecule).expect("cached");
    assert_eq!(
        *cached, direct,
        "缓存结果必须与直接计算逐位相同（I-06 的验收就是 identical）"
    );
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 0);
}

#[test]
fn second_call_is_a_hit_and_does_not_copy() {
    let (basis, molecule) = h2();
    let mut cache = IntegralCache::new();
    let first = cache.two_electron(&basis, &molecule).expect("first");
    let second = cache.two_electron(&basis, &molecule).expect("second");

    assert_eq!(cache.hits(), 1, "第二次应命中");
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.len(), 1, "同内容只存一份");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "命中时应返回同一个 Arc（不复制张量）"
    );
}

#[test]
fn different_content_does_not_collide() {
    let (basis, molecule) = h2();
    let mut cache = IntegralCache::new();
    cache.two_electron(&basis, &molecule).expect("h2");

    // 同一分子、但几何不同 → 指纹不同 → 新条目。
    let shifted = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.40]),
            Atom::new("H", [0.0, 0.0, 0.40]),
        ],
        0,
        1,
    );
    assert_ne!(
        fingerprint(&basis, &molecule),
        fingerprint(&basis, &shifted),
        "几何不同必须指纹不同"
    );
    let shifted_basis = {
        let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");
        let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
        InternalBasis::build(basis_k1, &shifted).expect("build")
    };
    cache
        .two_electron(&shifted_basis, &shifted)
        .expect("shifted");
    assert_eq!(cache.len(), 2, "两个不同内容各存一份");
}

#[test]
fn fingerprint_is_stable_across_rebuilt_bases() {
    // 两个来源不同但内容相同的基组必须指纹相同（说明指纹用内容，不用地址）。
    let (basis_a, molecule) = h2();
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let basis_b = InternalBasis::build(basis_k1, &molecule).expect("rebuild");
    assert_eq!(
        fingerprint(&basis_a, &molecule),
        fingerprint(&basis_b, &molecule)
    );
}
