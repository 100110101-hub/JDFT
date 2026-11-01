//! DA-06 验收：D3(BJ) 的参数表能加载、**每个元素对都在**、取向与派生量都对得上。
//!
//! 判据（§6, DA-06）：「与官方参数表逐项 diff < 1e-12；**对每个元素对都有测试**；
//! 解析梯度有单测（FX-04）」。
//!
//! 本文件覆盖落盘侧：
//!
//! 1. 36 个元素（H–Kr）→ **666 个元素对**逐一断言：块存在、49 个值、全有限非负，
//!    且正反两种写法的 `pair_c6` 给出同一块（取向实现错会立刻红）；
//! 2. 钉住官方参考表里的具体数字（`reference.f90` 的 `c6ab_view` 展平表）：
//!    `C6(H,H)[0] = 3.0267`、`C6(H,H)[8] = 7.5916`、`C6(C,C)[0] = 49.113`、
//!    `C6(Kr,Kr)[0] = 130.4017`、`C6(O,O)[0] = 15.5059`、`C6(O,H)[0] = 6.718`、
//!    `C6(O,H)[7] = 6.0575`；
//! 3. `r4r2` 是**派生值** `sqrt(0.5·published·sqrt(Z))`（模型消费的就是它），
//!    对 36 个元素逐个核对；
//! 4. 只有 4 个泛函有 BJ 参数（b3lyp / pbe / pbe0 / hf）—— M06-L 与 M06-2X 上游只有
//!    零阻尼，**故意不在**这里（不是遗漏）；
//! 5. 破坏数据必须被发现：删一个元素对 → `Registry::open` 直接失败。
//!
//! 另外两条判据由别处兜底：「与官方参数表逐项 diff = 0」由生成器
//! `scripts/gen_d3_params.py --check`（门禁「DA-06: dispersion data matches upstream
//! sources」），「复现官方实现的能量与解析梯度」由 `scripts/validate_d3_data.py`
//! （门禁「DA-06: dispersion data reproduces s-dftd3」）—— 这里不重复，也不假装做了。

use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_registry::dispersion::{PAIR_C6_LEN, REFERENCE_POINTS};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// H–Kr 的 36 个元素符号（与数据文件的 `elements` 一致）。
const ELEMENTS: [&str; 36] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr",
];

#[test]
fn da06_covers_every_element_pair() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let names: Vec<&str> = registry.dispersion_names().collect();
    println!("dispersion data: {names:?}");
    assert_eq!(
        names,
        vec!["d3bj"],
        "DA-06 ships exactly one D3(BJ) data set"
    );

    let data = registry.get_dispersion("d3bj").expect("d3bj");
    assert!(data.check().is_empty(), "{:?}", data.check());
    assert_eq!(data.elements, ELEMENTS.to_vec());

    let mut pairs = 0usize;
    for (index, high) in ELEMENTS.iter().enumerate() {
        for low in &ELEMENTS[..=index] {
            let key = format!("{high}-{low}");
            let block = data
                .reference_c6
                .get(&key)
                .unwrap_or_else(|| panic!("reference_c6 is missing the pair `{key}`"));
            assert_eq!(
                block.len(),
                PAIR_C6_LEN,
                "{key}: {} values, expected {PAIR_C6_LEN}",
                block.len()
            );
            assert!(
                block.iter().all(|value| value.is_finite() && *value >= 0.0),
                "{key}: C6 values must be finite and non-negative"
            );
            let high_count = data.reference_count(high).expect("high-Z reference CN");
            let low_count = data.reference_count(low).expect("low-Z reference CN");
            assert!(
                (1..=REFERENCE_POINTS).contains(&high_count)
                    && (1..=REFERENCE_POINTS).contains(&low_count),
                "{key}: reference point counts {high_count}/{low_count} out of range"
            );
            assert_eq!(
                data.pair_c6(high, low).expect("forward pair"),
                data.pair_c6(low, high).expect("backward pair"),
                "{key}: pair_c6 must not depend on the argument order"
            );
            pairs += 1;
        }
    }
    println!("checked {pairs} element pairs");
    assert_eq!(pairs, 666, "36 elements must give 36*37/2 = 666 pairs");
}

#[test]
fn da06_pinned_reference_values_match_the_official_table() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_dispersion("d3bj").expect("d3bj");

    // `reference.f90` 的展平表按元素对分块、块内第一维（高 Z 元素）最快。
    let pinned = [
        ("H", "H", 0usize, 3.0267),
        ("H", "H", 8usize, 7.5916),
        ("O", "O", 0usize, 15.5059),
        ("C", "C", 0usize, 49.113),
        ("Ne", "Ne", 0usize, 6.2896),
        ("Kr", "Kr", 0usize, 130.4017),
        ("O", "H", 0usize, 6.718),
        ("O", "H", 7usize, 10.2371),
        ("O", "H", 1usize, 6.0575),
    ];
    for (high, low, index, expected) in pinned {
        let block = data.pair_c6(high, low).expect("pair");
        println!("C6({high},{low})[{index}] = {}", block[index]);
        assert!(
            (block[index] - expected).abs() < 1.0e-4,
            "C6({high},{low})[{index}] = {} but the official table has {expected}",
            block[index]
        );
    }

    // 取向的语义断言：官方实现反解出的 O-H 在 CN≈0 处是 10.0951，只有「块内第一维 =
    // 高 Z 元素（O）」的读法能给出它。转置存储时槽位 7/1 的值会互换（10.2371 与 6.0575
    // 对调），上面两个 pin 因此互为对照：任一取向改动都会让它们同时红。
    let oh = data.pair_c6("O", "H").expect("O-H");
    assert!(
        oh[7] > oh[1],
        "O-H block: slot 7 ({}) should exceed slot 1 ({}) for the high-Z-first layout",
        oh[7],
        oh[1]
    );
    assert_eq!(data.pair_c6("H", "O").expect("H-O"), oh);
}

#[test]
fn da06_derived_quantities_are_consistent() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_dispersion("d3bj").expect("d3bj");

    // r4r2 = sqrt(0.5·published·sqrt(Z))，对全部 36 个元素核对。
    for (index, symbol) in ELEMENTS.iter().enumerate() {
        let published = *data.r4r2_published.get(*symbol).expect("published r4r2");
        let derived = *data.r4r2.get(*symbol).expect("r4r2");
        let z = (index + 1) as f64;
        let expected = (0.5 * published * z.sqrt()).sqrt();
        assert!(
            (derived - expected).abs() < 1.0e-12,
            "{symbol}: r4r2 = {derived}, expected {expected} (published {published})"
        );
    }
    assert!((data.r4r2["H"] - 2.007_348_998_056_89).abs() < 1.0e-12);
    assert!((data.r4r2["Kr"] - 4.311_762_980_498_812).abs() < 1.0e-12);

    // CN 参数：上游 kcn = 16、scale = 4/3、半径是 Pyykko 的 Å 值。
    assert!((data.coordination_number.kcn - 16.0).abs() < 1.0e-12);
    assert!((data.coordination_number.scale - 4.0 / 3.0).abs() < 1.0e-15);
    assert!((data.coordination_number.radii_angstrom["H"] - 0.32).abs() < 1.0e-12);
    assert!((data.coordination_number.radii_angstrom["O"] - 0.63).abs() < 1.0e-12);
    let hydrogen = data.covalent_radius_bohr("H").expect("H radius");
    let expected = 0.32 * 4.0 / 3.0 * qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
    assert!((hydrogen - expected).abs() < 1.0e-12);

    assert!((data.reference.wf - 4.0).abs() < 1.0e-12);

    // 只有 4 个泛函有 BJ 参数；M06-L / M06-2X 上游只有零阻尼，故意不在这里。
    let mut functionals: Vec<&str> = data.damping_parameters.keys().map(String::as_str).collect();
    functionals.sort_unstable();
    assert_eq!(functionals, vec!["b3lyp", "hf", "pbe", "pbe0"]);
    assert!(data.damping("m06-l").is_none());
    assert!(data.damping("m06-2x").is_none());

    let b3lyp = data.damping("b3lyp").expect("b3lyp");
    assert!((b3lyp.s6 - 1.0).abs() < 1.0e-12);
    assert!((b3lyp.s8 - 1.9889).abs() < 1.0e-12);
    assert!((b3lyp.a1 - 0.3981).abs() < 1.0e-12);
    assert!((b3lyp.a2 - 4.4211).abs() < 1.0e-12);
    assert_eq!(b3lyp.doi, "10.1063/1.3382344");
    assert!((data.damping("hf").expect("hf").a2 - 2.883).abs() < 1.0e-12);
}

#[test]
fn da06_broken_dispersion_data_is_rejected() {
    // 只造一个**最小**数据根：一份色散文件 + 一份只索引它的 registry.yaml。
    // 这样测试不依赖 qwb-data 的其它目录（也不受第三方基组是否取回的影响）。
    let source = data_root();
    let temp = std::env::temp_dir().join(format!("qwb-da06-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let dispersion_dir = temp.join("dispersion");
    std::fs::create_dir_all(&dispersion_dir).expect("mkdir dispersion");

    let text = std::fs::read_to_string(source.join("dispersion/d3bj.yaml")).expect("read d3bj");
    std::fs::write(dispersion_dir.join("d3bj.yaml"), &text).expect("write d3bj");
    let sha = text
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("sha256:"))
        .expect("d3bj.yaml must start with its sha256")
        .trim()
        .to_string();
    let manifest = format!(
        concat!(
            "schema_version: {}\n",
            "name: qwb-data-da06\n",
            "version: 0.1.0\n",
            "created: 2026-11-09T00:00:00Z\n",
            "features:\n  - \"dispersion:d3bj\"\n",
            "dispersions:\n  - name: d3bj\n    path: dispersion/d3bj.yaml\n    sha256: {}\n",
        ),
        qwb_registry::K1_VERSION,
        sha
    );
    std::fs::write(temp.join("registry.yaml"), manifest).expect("write manifest");

    // 基线：这份最小数据根必须能加载，否则下面的断言证明不了什么。
    Registry::open(&temp).expect("baseline registry must load");

    // 改掉一个元素对的键 → 结构检查必须拒绝（错误里点名缺的那个元素对）。
    // 打戳后的文件里 `reference_c6` 是块序列，键后面直接换行（不是 `[`）。
    let broken = text.replace("\n  Kr-Kr:\n", "\n  Kr-Kr-typo:\n");
    assert!(
        text.contains("\n  Kr-Kr:\n"),
        "the Kr-Kr key must exist to be broken"
    );
    std::fs::write(dispersion_dir.join("d3bj.yaml"), &broken).expect("write broken");
    let error = Registry::open(&temp).expect_err("a missing element pair must be rejected");
    let message = format!("{error}");
    let head: String = message.chars().take(300).collect();
    println!("broken data error (first 300 chars): {head}");
    assert!(
        message.contains("Kr-Kr"),
        "the error should name the broken pair, got: {head}"
    );

    let _ = std::fs::remove_dir_all(&temp);
}
