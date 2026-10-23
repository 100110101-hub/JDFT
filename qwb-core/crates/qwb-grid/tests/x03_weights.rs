//! X-03 的一角：网格权重的**实际积分能力**（不依赖 Lebedev 表的部分）。
//!
//! 这里记一次踩坑，免得后人重犯：原子网格的 `Σw` **不等于** 4π。径向权重含
//! `r²` Jacobian，所以 `Σw ≈ ∫4πr²dr = (4π/3)·r_max³`（`r_max ≈ cutoff`）。
//! 换句话说「Σw = 4π」这种判据只有**角向**规则才成立。project.md 里 X-03 的
//! 「水 Σw 与 4π 差 < 1e-10」按字面照做会误判——已记进 docs/choices.md。
//!
//! 真正该测的是：这个网格能不能把**局域**被积函数积分准。用一个解析可积的
//! 高斯 `e^{-α r²}` 做标尺，解析值 `(π/α)^{3/2}`。

use std::path::PathBuf;

use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

#[test]
fn atomic_grid_integrates_gaussians_accurately() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let level = registry.get_grid("grid-coarse").expect("grid-coarse");
    let grid = qwb_grid::atomic_grid(level, [0.0, 0.0, 0.0]).expect("atomic grid");
    assert!(!grid.is_empty());

    for alpha in [0.5_f64, 1.0, 2.0] {
        let numerical: f64 = grid
            .points
            .iter()
            .map(|point| {
                let radius2: f64 = point.position_bohr.iter().map(|v| v * v).sum();
                point.weight * (-alpha * radius2).exp()
            })
            .sum();
        let analytic = (std::f64::consts::PI / alpha).powf(1.5);
        let relative = (numerical - analytic).abs() / analytic;
        println!(
            "alpha = {alpha}: numerical {numerical:.10}, analytic {analytic:.10}, rel {relative:.3e}"
        );
        assert!(
            relative < 1.0e-6,
            "alpha = {alpha}: 网格积分 {numerical:.10} 与解析 {analytic:.10} 相对差 {relative:.3e}"
        );
    }
}

#[test]
fn atomic_grid_weights_are_finite_and_non_negative() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let level = registry.get_grid("grid-coarse").expect("grid-coarse");
    let grid = qwb_grid::atomic_grid(level, [0.0, 0.0, 0.0]).expect("atomic grid");
    for point in &grid.points {
        assert!(
            point.weight.is_finite() && point.weight >= 0.0,
            "非有限或负权重: {}",
            point.weight
        );
        assert!(
            point.position_bohr.iter().all(|v| v.is_finite()),
            "非有限坐标: {:?}",
            point.position_bohr
        );
    }
    // 径向映射在 t→1 会发散；这里确认夹紧真的生效（坐标有限），而不是靠
    // inf×0 被悄悄吞掉。
    //
    // 注意 `cutoff` 的实际语义：它是径向映射 r = a·t²/(1−t²) 的**尺度**
    // （a = cutoff/8），不是硬边界。t 夹在 1−1e-3 处，外半径因此约为
    // 62×cutoff。这样节点在 [0, cutoff] 区间内的分辨率才够（高斯积分 1e-12）；
    // 若强行把最外节点压到 cutoff，全部节点会挤到 r < 1，网格反而变差。
    // 这一点已记进 docs/choices.md，等 D-13/X-03 收口时再决定是否换成
    // Mura–Knowles 一类的有界映射。
    let farthest = grid
        .points
        .iter()
        .map(|point| {
            point
                .position_bohr
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0_f64, f64::max);
    println!(
        "farthest point radius = {farthest:.6} bohr (cutoff = {})",
        level.cutoff
    );
    assert!(farthest.is_finite(), "最远点半径不是有限值");
    assert!(
        farthest < 200.0 * level.cutoff,
        "最远点 {farthest} 偏离 cutoff {} 超过两个量级，映射可能已经失控",
        level.cutoff
    );
}
