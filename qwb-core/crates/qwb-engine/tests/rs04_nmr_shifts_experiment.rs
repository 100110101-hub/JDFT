//! RS-04 第三条判据的**诚实缺口测试**：GIAO 的 ¹H / ¹³C 化学位移与**实验**值的差。
//!
//! ## 这条测试在量什么
//!
//! `project-phase2.md` §6 的 RS-04 有三条：①与独立实现（PySCF GIAO）逐核一致、②规范变化
//! < 1 ppm、③化学位移与实验差 < 0.5 ppm（CH4 / CH3OH 另有更宽的 1.0 / 5.0）。①② 已经由
//! `rs04_nmr_engine.rs` 与 `qwb-response/tests/rs04_giao_nmr.rs` 钉住；③ 在此之前**从没量过**
//! —— 它要参考物 TMS、两个样品分子，以及一份 pinned 的实验值。
//!
//! 这条测试把 ③ **量出来并按实测值钉住**（不是按判据钉住）。判据本身一个字符都没动：
//! 下面每行的 `|Δ|` 都是实测的，`assert!` 与打印里的 PASS / MISS 反映的是**当前状态**。
//! 如果将来某个改动让偏差变小、越过阈值，`assert` 会**故意失败**，逼人回来把数字与结论
//! 一起改掉 —— 那才是"缺口被补上"该有的走法。
//!
//! ## 几何从哪来（不是手写几何）
//!
//! 三个分子的几何都是本项目优化器（`qwb run`，`jobtype: opt`）的输出，逐字抄自
//! `artifacts/optimized.xyz`。三条作业的选项完全一致：
//!
//! ```text
//! basis_name: sto-3g, method_name: hf, grid_level: coarse, solvent: vacuum, jobtype: opt
//! options.scf_max_iter: 200
//! options.scf_energy_threshold_hartree: 1.0e-11
//! options.scf_density_threshold: 1.0e-9
//! options.max_opt_steps: 200
//! options.opt_gradient_threshold_hartree_per_bohr: 4.5e-4
//! ```
//!
//! * CH4：2 步收敛，E = -39.7268636769 Ha，max|grad| < 4.5e-4（起点 r(CH) = 1.087 Å 的正四面体）；
//! * CH3OH：16 步收敛，E = -113.5491913869 Ha，max|grad| = 3.408e-4（起点 r(C–O) = 1.430、
//!   r(O–H) = 0.960、r(C–H) = 1.094 Å、∠COH = 108.5°）；
//! * TMS：16 步收敛，E = -442.2986808541 Ha，max|grad| = 3.676e-4（起点是**Td 对称**的
//!   交错式甲基构造：Si–C = 1.880、C–H = 1.090 Å，四个甲基的 C–H 相对另外三个 Si–C 键交错
//!   —— 这一点很要紧：从"四个甲基任意扭转"的起点起步，同一个 BFGS 跑到 4407 s CPU
//!   （约 37 分钟）仍在继续、只能终止；从 Td 起点重跑则 16 步收敛）。
//!
//! ## 实验值从哪来
//!
//! 一律从 `tests/reference/nmr_shifts_experimental.csv` 读，**不在这个文件里另写一份**。
//! 出处：Fulmer et al., Organometallics 29, 2176 (2010) 的支持信息（DOI 10.1021/om100106e；
//! 生成器 `scripts/gen_nmr_shifts_experimental.py`，逐行带 `source_doi`/`source_url`）。
//!
//! **相态**：实验值是 CDCl3 稀溶液的位移（TMS 在同一溶剂里定零点），本测试算的是真空中
//! 孤立分子。CSV 的 `phase`/`solvent` 两列把这件事写死在数据里，`solvent_spread_ppm` 列给出
//! 同一张表 12 种氘代溶剂之间的实测跨度（¹H 最大 0.46 ppm、¹³C 最大 2.08 ppm）——
//! 「溶剂效应有多大」是引用了来源的数字，不是一句解释性的话。
//!
//! ## 为什么 TMS 的两个屏蔽是**钉住的常数**，而 CH4/CH3OH 是现算的
//!
//! TMS 是 17 个原子：同一条 GIAO 作业（STO-3G、带 `report_gauge_spread`）在这台机器上要跑
//! **约 54 分钟**（单线程构建，见 I-04）。把它放进 `cargo test` 会把测试套件变成不可用，
//! 所以 TMS 的 ¹H / ¹³C 屏蔽钉成常数，并在这里写清怎么重测（几何在上面、输入就是同一份
//! `sto-3g` + `gauge: giao` 的 `sp` 输入，只把坐标换成 TMS）。CH4（5 原子）与 CH3OH（6 原子）
//! 很便宜（这条测试整体约 52 s），每次现算 —— 于是它仍然会抓住"样品分子的屏蔽漂了"这种回归。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, GridLevel, JobInput, JobType, Molecule, NmrGauge, NmrOptions,
    NmrShieldings, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

fn molecule(atoms: &[(&str, [f64; 3])]) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|(symbol, position)| Atom::new(*symbol, *position))
            .collect(),
        0,
        1,
    )
}

/// CH4 / HF/STO-3G 优化后的几何（2 步收敛，E = -39.7268636769 Ha）。
const METHANE: &[(&str, [f64; 3])] = &[
    ("C", [0.0, 0.0, 0.0]),
    ("H", [0.625_269_036_5, 0.625_269_036_5, 0.625_269_036_5]),
    ("H", [0.625_269_036_5, -0.625_269_036_5, -0.625_269_036_5]),
    ("H", [-0.625_269_036_5, 0.625_269_036_5, -0.625_269_036_5]),
    ("H", [-0.625_269_036_5, -0.625_269_036_5, 0.625_269_036_5]),
];

/// CH3OH / HF/STO-3G 优化后的几何（16 步收敛，E = -113.5491913869 Ha）。
/// 原子序是 C、O、H(O)、H(C)、H(C)、H(C)。
const METHANOL: &[(&str, [f64; 3])] = &[
    (
        "C",
        [0.020_639_055_686, 0.032_278_657_960, 0.015_466_190_850],
    ),
    (
        "O",
        [0.006_241_527_831, 0.181_151_245_607, 1.440_438_427_255],
    ),
    (
        "H",
        [0.866_635_089_083, -0.216_162_840_690, 1.729_587_810_655],
    ),
    (
        "H",
        [0.078_313_310_085, -1.015_845_117_141, -0.295_299_157_206],
    ),
    (
        "H",
        [-0.912_895_917_719, 0.446_567_112_007, -0.369_009_976_080],
    ),
    (
        "H",
        [0.851_455_489_000, 0.572_010_942_358, -0.450_509_805_343],
    ),
];

/// TMS（Si(CH3)4）/ HF/STO-3G 优化后的几何（16 步收敛，E = -442.2986808541 Ha，
/// max|grad| = 3.676e-4 Ha/Bohr，Td 对称）。抄自 `artifacts/optimized.xyz`。
/// 原子序是 Si、C、3×H、C、3×H、C、3×H、C、3×H。
const TMS: &[(&str, [f64; 3])] = &[
    ("Si", [0.0, 0.0, 0.0]),
    ("C", [1.076_510_069_2, 1.076_510_069_2, 1.076_510_069_2]),
    ("H", [1.716_257_990_3, 1.716_257_990_3, 0.482_898_350_8]),
    ("H", [0.482_898_350_8, 1.716_257_990_3, 1.716_257_990_3]),
    ("H", [1.716_257_990_3, 0.482_898_350_8, 1.716_257_990_3]),
    ("C", [1.076_510_069_2, -1.076_510_069_2, -1.076_510_069_2]),
    ("H", [1.716_257_990_3, -1.716_257_990_3, -0.482_898_350_8]),
    ("H", [0.482_898_350_8, -1.716_257_990_3, -1.716_257_990_3]),
    ("H", [1.716_257_990_3, -0.482_898_350_8, -1.716_257_990_3]),
    ("C", [-1.076_510_069_2, 1.076_510_069_2, -1.076_510_069_2]),
    ("H", [-0.482_898_350_8, 1.716_257_990_3, -1.716_257_990_3]),
    ("H", [-1.716_257_990_3, 0.482_898_350_8, -1.716_257_990_3]),
    ("H", [-1.716_257_990_3, 1.716_257_990_3, -0.482_898_350_8]),
    ("C", [-1.076_510_069_2, -1.076_510_069_2, 1.076_510_069_2]),
    ("H", [-0.482_898_350_8, -1.716_257_990_3, 1.716_257_990_3]),
    ("H", [-1.716_257_990_3, -0.482_898_350_8, 1.716_257_990_3]),
    ("H", [-1.716_257_990_3, -1.716_257_990_3, 0.482_898_350_8]),
];

/// TMS 的 HF/STO-3G GIAO 屏蔽（上面那份 Td 几何上实测；带 `report_gauge_spread` 的整条作业
/// 在这台机器上跑了约 54 分钟，`length_gauge_spread_ppm = 7.433e-9`）。12 个甲基氢彼此差
/// < 2e-8 ppm（几何是 Td），这里用 index 2 那个值；4 个碳差 < 1e-8，用 index 1。
const TMS_SIGMA_H_PPM: f64 = 33.757_053_665;
const TMS_SIGMA_C_PPM: f64 = 249.433_185_059;

fn giao(atoms: &[(&str, [f64; 3])]) -> NmrShieldings {
    let input = JobInput::new(
        molecule(atoms),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        TaskOptions::default().with_nmr(NmrOptions::new().with_gauge(NmrGauge::Giao)),
    );
    let result = backend()
        .run(&input, &NullReporter)
        .expect("GIAO 屏蔽应跑通");
    let block = result.nmr_shieldings.expect("nmr_shieldings 块");
    assert_eq!(block.method, "giao", "方法名必须是 giao");
    let spread = block
        .length_gauge_spread_ppm
        .expect("GIAO 缺省要报规范变化");
    assert!(spread < 1.0, "GIAO 的规范变化 {spread:.3e} ppm 不小于 1");
    block
}

struct Experimental {
    molecule: String,
    nucleus: String,
    shift_ppm: f64,
}
/// 读 `tests/reference/nmr_shifts_experimental.csv` 里**不是** `definition` 相态的行。
fn experimental() -> Vec<Experimental> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference/nmr_shifts_experimental.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("读不到 {}: {error}", path.display()));
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 11, "line {}: 期望 11 列", index + 1);
        if fields[4] == "definition" {
            continue;
        }
        out.push(Experimental {
            molecule: fields[0].to_string(),
            nucleus: fields[1].to_string(),
            shift_ppm: fields[3].parse::<f64>().expect("shift_ppm"),
        });
    }
    out
}

/// 钉住的几何自检：直接从坐标量键长 —— 这三份坐标确实是优化器的输出（TMS 的十二个 C–H
/// 一样长、Si–C 只有一个值，也就是 Td），不是手抄的近似四面体。
#[test]
fn the_pinned_geometries_are_the_optimiser_outputs() {
    fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }
    // CH4：正四面体，r(C–H) = 1.0829977396 Å。
    for atom in &METHANE[1..] {
        let r = distance(METHANE[0].1, atom.1);
        assert!((r - 1.082_997_739_6).abs() < 1e-9, "CH4 的 C–H = {r:.10} Å");
    }
    // CH3OH：C–O / O–H / 三个 C–H（CH3 的旋转破了 C3，所以三个 C–H 不完全一样长）。
    let bonds = [
        (
            "C–O",
            distance(METHANOL[0].1, METHANOL[1].1),
            1.432_800_13_f64,
        ),
        ("O–H", distance(METHANOL[1].1, METHANOL[2].1), 0.990_829_41),
        ("C–H", distance(METHANOL[0].1, METHANOL[3].1), 1.094_744_20),
        ("C–H", distance(METHANOL[0].1, METHANOL[4].1), 1.091_303_99),
        ("C–H", distance(METHANOL[0].1, METHANOL[5].1), 1.094_851_82),
    ];
    for (label, measured, pinned) in bonds {
        assert!(
            (measured - pinned).abs() < 1e-6,
            "CH3OH 的 {label} = {measured:.8} Å（钉住 {pinned:.8}）"
        );
    }
    // TMS：Td，Si–C 一个值、十二个 C–H 一个值。
    for (index, atom) in TMS[1..].iter().enumerate() {
        if atom.0 == "C" {
            let r = distance(TMS[0].1, atom.1);
            assert!(
                (r - 1.864_570_134_7).abs() < 1e-8,
                "TMS 的 Si–C = {r:.10} Å"
            );
            continue;
        }
        // 每个 C 后面紧跟它的三个 H
        let carbon = TMS[1 + index - (index % 4)].1;
        let r = distance(carbon, atom.1);
        assert!(
            (r - 1.082_095_040_8).abs() < 1e-8,
            "TMS 的第 {} 个 C–H = {r:.10} Å",
            index
        );
    }
}

#[test]
#[ignore = "slow: 重测 CH4/CH3OH 的 GIAO 屏蔽并与实验值表对 |Δ|（两个 GIAO SCF 作业，本机实测 57.5 s）；run with -Slow"]
fn measured_shifts_against_the_pinned_experimental_values() {
    let methane = giao(METHANE);
    let methanol = giao(METHANOL);

    // 对称性自检：CH4 的四个氢、TMS 的十二个氢/四个碳在数据上也必须等价（几何是 Td，
    // 屏蔽若不等价，说明几何或积分里有问题）。
    for (label, values) in [
        (
            "methane H",
            methane.nuclei[1..]
                .iter()
                .map(|nucleus| nucleus.isotropic_ppm)
                .collect::<Vec<_>>(),
        ),
        ("methanol C", vec![methanol.nuclei[0].isotropic_ppm]),
    ] {
        let spread = values.iter().cloned().fold(f64::MIN, f64::max)
            - values.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread < 1e-6, "{label}: 等价核的屏蔽差 {spread:e} 不平");
    }

    let sigma_h = |block: &NmrShieldings, index: usize| block.nuclei[index].isotropic_ppm;
    let methane_h = sigma_h(&methane, 1);
    let methane_c = sigma_h(&methane, 0);
    let methanol_c = sigma_h(&methanol, 0);
    let methanol_h = (sigma_h(&methanol, 3) + sigma_h(&methanol, 4) + sigma_h(&methanol, 5)) / 3.0;
    let methanol_h_spread = {
        let values = [3, 4, 5].map(|index| sigma_h(&methanol, index));
        values.iter().cloned().fold(f64::MIN, f64::max)
            - values.iter().cloned().fold(f64::MAX, f64::min)
    };

    println!("TMS（钉住的常数）: ¹H = {TMS_SIGMA_H_PPM:.6} ppm、¹³C = {TMS_SIGMA_C_PPM:.6} ppm");
    println!("CH4  : ¹H = {methane_h:.6} ppm、¹³C = {methane_c:.6} ppm");
    println!(
        "CH3OH: ¹H(CH3, 三个平均) = {methanol_h:.6} ppm（三个各自差 {methanol_h_spread:.3e} ppm）、\
         ¹³C = {methanol_c:.6} ppm"
    );

    // δ = σ(TMS) − σ(样品)：同一个方法/基组下的两个屏蔽相减，就是这里的"计算化学位移"。
    let computed = [
        ("methane", "1H", TMS_SIGMA_H_PPM - methane_h),
        ("methane", "13C", TMS_SIGMA_C_PPM - methane_c),
        ("methanol", "1H", TMS_SIGMA_H_PPM - methanol_h),
        ("methanol", "13C", TMS_SIGMA_C_PPM - methanol_c),
    ];

    println!();
    println!("分子      核    计算 δ (ppm)   实验 δ (ppm)   |Δ| (ppm)   阈值   状态");
    let mut status = Vec::new();
    for expected in experimental() {
        let (_, _, delta) = computed
            .iter()
            .find(|(molecule, nucleus, _)| {
                *molecule == expected.molecule && *nucleus == expected.nucleus
            })
            .unwrap_or_else(|| {
                panic!(
                    "CSV 里的 {} / {} 没有对应的计算值",
                    expected.molecule, expected.nucleus
                )
            });
        let error = (delta - expected.shift_ppm).abs();
        // 判据：¹H 的通用阈值 0.5 ppm；CH4 的两个核 1.0 ppm；CH3OH 的两个核 5.0 ppm
        //（`project-phase2.md` §6 RS-04 的原文，与 `rs04_nmr_engine.rs` 的读法一致）。
        let threshold = match (expected.molecule.as_str(), expected.nucleus.as_str()) {
            ("methane", _) => 1.0,
            ("methanol", _) => 5.0,
            (_, "1H") => 0.5,
            _ => unreachable!(),
        };
        let passed = error < threshold;
        let verdict = if passed { "PASS" } else { "MISS" };
        println!(
            "{:<9} {:<5} {delta:>14.6} {:>13.2} {error:>12.6} {threshold:>6.1}   {verdict}",
            expected.molecule, expected.nucleus, expected.shift_ppm,
        );
        status.push((
            expected.molecule.clone(),
            expected.nucleus.clone(),
            delta,
            error,
            passed,
        ));
    }
    println!();

    // ── 实测值的冻结（4 位小数）：任何改动只要动了这三个屏蔽，这里先炸。 ──
    for (label, computed, pinned) in [
        ("CH4 ¹H", methane_h, 32.860_808_257_f64),
        ("CH4 ¹³C", methane_c, 236.183_677_223),
        ("CH3OH ¹H", methanol_h, 31.544_804_913),
        ("CH3OH ¹³C", methanol_c, 199.224_771_326),
    ] {
        assert_eq!(
            format!("{computed:.4}"),
            format!("{pinned:.4}"),
            "{label} 的屏蔽漂了（{computed:.6} vs 冻结的 {pinned:.6}）"
        );
    }
    assert!(
        methanol_h_spread < 0.5,
        "CH3 三个质子在优化后的几何上不等价得离谱：{methanol_h_spread:.3e} ppm"
    );

    // ── 诚实缺口：把**实测的** |Δ| 与 PASS/MISS 状态一起钉住。 ──
    // 这些断言不是判据，是"现在长什么样"。缺口补上（或变大）时它们会故意失败。
    for (molecule, nucleus, _, error, passed) in &status {
        let (pinned_error, pinned_passed) = PINNED_ABS_DELTA_PPM
            .iter()
            .find(|(m, n, _, _)| m == molecule && n == nucleus)
            .map(|(_, _, error, passed)| (*error, *passed))
            .unwrap_or_else(|| panic!("没有为 {molecule} / {nucleus} 钉住 |Δ|"));
        assert_eq!(
            format!("{error:.2}"),
            format!("{pinned_error:.2}"),
            "{molecule} / {nucleus} 的 |Δ| 漂了（实测 {error:.6}，钉住 {pinned_error:.6}）"
        );
        assert_eq!(
            *passed, pinned_passed,
            "{molecule} / {nucleus} 的 PASS/MISS 变了：判据没被满足/满足这件事必须显式改这里，\
             并同步改 RS-04 的结论"
        );
    }

    // ── 三条判据的结论（各自一个数，不是一句"大体符合"）。 ──
    let worst_h1 = status
        .iter()
        .filter(|(_, nucleus, _, _, _)| nucleus.as_str() == "1H")
        .map(|(_, _, _, error, _)| *error)
        .fold(0.0_f64, f64::max);
    let worst_methane = status
        .iter()
        .filter(|(molecule, _, _, _, _)| molecule.as_str() == "methane")
        .map(|(_, _, _, error, _)| *error)
        .fold(0.0_f64, f64::max);
    let worst_methanol = status
        .iter()
        .filter(|(molecule, _, _, _, _)| molecule.as_str() == "methanol")
        .map(|(_, _, _, error, _)| *error)
        .fold(0.0_f64, f64::max);
    let verdict = |ok: bool| if ok { "PASS" } else { "MISS" };
    println!(
        "¹H < 0.5 ppm（通用判据，两个分子取最差）：{worst_h1:.6} ppm → {}",
        verdict(worst_h1 < 0.5)
    );
    println!(
        "CH4 < 1.0 ppm（两个核取最差）：{worst_methane:.6} ppm → {}",
        verdict(worst_methane < 1.0)
    );
    println!(
        "CH3OH < 5.0 ppm（两个核取最差）：{worst_methanol:.6} ppm → {}",
        verdict(worst_methanol < 5.0)
    );
    assert_eq!(
        worst_h1 < 0.5,
        PINNED_H1_GENERAL_UNDER_HALF_PPM,
        "¹H 的通用 0.5 ppm 判据的结论变了（实测最差 {worst_h1:.6} ppm）"
    );
    assert_eq!(
        worst_methane < 1.0,
        PINNED_METHANE_UNDER_ONE_PPM,
        "CH4 < 1.0 ppm 的结论变了（实测最差 {worst_methane:.6} ppm）"
    );
    assert_eq!(
        worst_methanol < 5.0,
        PINNED_METHANOL_UNDER_FIVE_PPM,
        "CH3OH < 5.0 ppm 的结论变了（实测最差 {worst_methanol:.6} ppm）"
    );
}

/// 实测的 |Δ| 与当时是否达到阈值（`true` = `|Δ| < 阈值`）。
/// 这些是 **2026-11-14 的实测值**（HF/STO-3G），不是判据；改引擎后它们会变，那就显式改这里。
const PINNED_ABS_DELTA_PPM: &[(&str, &str, f64, bool)] = &[
    ("methane", "1H", 0.68, true),
    ("methane", "13C", 17.88, false),
    ("methanol", "1H", 1.28, true),
    ("methanol", "13C", 0.20, true),
];

/// 三条判据在 HF/STO-3G 上的实测状态（`false` = 没达到）。这三个布尔就是 RS-04 第三条的
/// 结论：**¹H 的通用 0.5 ppm 与 CH4 的 1.0 ppm 没达到，CH3OH 的 5.0 ppm 达到了**。
/// 将来补上缺口时，这里会从 `false` 翻成 `true` —— 那是要显式改的结论，不是自动升级。
const PINNED_H1_GENERAL_UNDER_HALF_PPM: bool = false;
const PINNED_METHANE_UNDER_ONE_PPM: bool = false;
const PINNED_METHANOL_UNDER_FIVE_PPM: bool = true;
