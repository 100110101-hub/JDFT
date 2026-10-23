//! X-04 验收（project.md §6.2）：网格管理（等级映射 + 缓存）。
//!
//! 判据原文：「切换等级释放旧网格（`Arc::strong_count` 断言）」。
//! 这里把它写成三条：
//!   ① 共享是 `Arc` 克隆，不复制点；
//!   ② 切换等级后，**旧网格的强引用必须掉回 1**（只剩测试里的那份），
//!      即缓存没有继续持有它——这就是「释放」的可观测判据；
//!   ③ 新等级构建失败时，当前缓存保持不变（不会出现半替换）。
use std::path::PathBuf;
use std::sync::Arc;

use qwb_engine_api::{Atom, Molecule};
use qwb_grid::manager::GridCache;
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    )
}

#[test]
fn switching_level_releases_the_old_grid() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let coarse = registry.get_grid("grid-coarse").expect("grid-coarse");
    let normal = registry.get_grid("grid-normal").expect("grid-normal");

    let mut cache = GridCache::build("grid-coarse", coarse, &molecule).expect("build");
    assert!(cache.is("grid-coarse"));
    assert_eq!(cache.shared_by(), 1, "刚建好时只该有缓存自己持有");

    // ① 共享是 Arc 克隆
    let old = cache.grid();
    assert_eq!(Arc::strong_count(&old), 2, "取一份共享后应是 2");
    assert_eq!(old.len(), cache.grid().len(), "共享的应是同一个网格");
    let old_points = old.len();

    // ② 切换等级：旧网格必须被释放
    cache
        .switch("grid-normal", normal, &molecule)
        .expect("switch");
    assert!(cache.is("grid-normal"));
    assert_eq!(
        Arc::strong_count(&old),
        1,
        "切换后缓存不该再持有旧网格（strong_count 应掉回 1）"
    );
    assert!(
        cache.grid().len() > old_points,
        "normal 等级的点数应多于 coarse（{} vs {})",
        cache.grid().len(),
        old_points
    );
    assert_eq!(cache.shared_by(), 1);
}

#[test]
fn a_failed_switch_keeps_the_current_grid() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let coarse = registry.get_grid("grid-coarse").expect("grid-coarse");

    let mut cache = GridCache::build("grid-coarse", coarse, &molecule).expect("build");
    let before = cache.grid();
    let before_points = before.len();

    // 造一个非法的等级定义：cutoff = 0
    let mut broken = coarse.clone();
    broken.cutoff = 0.0;
    let error = cache.switch("broken", &broken, &molecule);
    assert!(error.is_err(), "cutoff=0 的等级必须构建失败");

    // 缓存保持原样
    assert!(cache.is("grid-coarse"), "失败的切换不该改标签");
    assert_eq!(
        cache.grid().len(),
        before_points,
        "失败的切换不该丢掉当前网格"
    );
    assert_eq!(
        Arc::strong_count(&before),
        2,
        "当前网格仍是缓存 + 测试各一份"
    );
}

#[test]
fn repeated_switching_does_not_accumulate_grids() {
    // 来回切换多次：每次切换后旧的那份都应只剩测试持有的引用，
    // 也就是说缓存里同时只存在一个网格（不会越攒越多）。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let coarse = registry.get_grid("grid-coarse").expect("grid-coarse");
    let normal = registry.get_grid("grid-normal").expect("grid-normal");

    let mut cache = GridCache::build("grid-coarse", coarse, &molecule).expect("build");
    let mut previous = cache.grid();
    for step in 0..4 {
        let (label, level) = if step % 2 == 0 {
            ("grid-normal", normal)
        } else {
            ("grid-coarse", coarse)
        };
        cache.switch(label, level, &molecule).expect("switch");
        assert!(
            Arc::strong_count(&previous) <= 1,
            "第 {step} 次切换后旧网格仍被持有（strong_count = {}）",
            Arc::strong_count(&previous)
        );
        previous = cache.grid();
    }
    drop(previous);
    assert_eq!(cache.shared_by(), 1);
}
