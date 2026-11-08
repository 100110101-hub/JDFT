//! RP-01 的**诚实缺口测试**：把这一轮**实测**到的势垒按实测值钉住，并把与文献的差算出来。
//!
//! `project-phase2.md` §6 的 RP-01 判据是「三个反应（HCN↔HNC、CH₃NC↔CH₃CN、H₂CO→H₂+CO）的
//! 势垒与**文献**差 < 1.5 kcal/mol（DFT 水平）；max|grad| < 4.5e-4；Hessian 恰有一个虚频」。
//!
//! # 这个文件里的是什么、不是什么
//!
//! * **文献**值**不在**这里：它们是 `tests/reference/rp01_literature_barriers.csv` 的行
//!   （生成器 `scripts/gen_rp01_literature.py`，逐行带 DOI/URL/理论级别/零点能口径）。
//!   这里只从那份 CSV 读数，绝不另抄一份 —— 两份拷贝迟早会不一致。
//! * **实测**值在这里是常数：能量逐字抄自 `qwb` 的 `result.json`，几何与命令行写在每个数的
//!   旁边。它们**不是**参考值（没有外部实现/实验能复现它们），是这一轮的测量记录。
//!
//! # 这一轮量到了什么（判据一个字符都没改）
//!
//! 测量驱动是 `scripts/rp01_measure.py`（`jobtype: opt` / `ts` / `irc`），层级的每一级都是
//! `grid_level: normal`：
//!
//! | 层级 | HCN↔HNC | CH₃NC↔CH₃CN | H₂CO→H₂+CO |
//! | --- | --- | --- | --- |
//! | HF/STO-3G | **69.25**，IRC 双向连到两个极小点 | 鞍点没找到（见下） | 7.26 的驻点 **IRC 走不动** → 不算 |
//! | PBE/def2-SVP | **没做完**（成本，见下） | 没做（成本） | 没做（成本） |
//!
//! HF/STO-3G 的数与文献（48.06 / 41.0 / 81.9）差 21.2 / — / 74.6 kcal/mol —— 这正是
//! 「层级不够」的量级，也说明为什么 RP-01 的判据必须落在 DFT 上。
//!
//! # 为什么 PBE/def2-SVP 没做完（实测成本，不是估计）
//!
//! * HCN / PBE / def2-SVP / grid-normal 单点：**28.5 s**（40 次 SCF 迭代，72900 个格点）；
//! * HCN 同层级的 `opt`（3 步收敛）：**457.8 s** ⇒ 一步解析梯度 ≈ **150 s**；
//! * 一次数值 Hessian = `6N` 次梯度（`ts` 每一步都要一次）：HCN 9 次 ⇒ ≈ **22 min**，
//!   H2CO 12 次 ⇒ ≈ 30 min，CH₃NC 18 次 ⇒ ≈ 45 min；
//! * `ts` 在 HCN 上从 PySCF 的 PBE 鞍点几何出发**第 183 秒就失败**：
//!   `scf not converged in 200 iterations`（鞍点区的 SCF 会震荡）；
//! * 一条双向 IRC 约 60–130 个点，每点 ≈ 2–3 次 SCF ⇒ **单反应 2–4 小时**。
//!
//! 所以这一轮**一个 DFT 势垒都没有交付**：判据未达成，原因是成本，不是算错了。这些数字是
//! 下一轮（或 EX-03 的 GPU 路径/并行化落地之后）重开这件事的起点。
//!
//! # 两条断言（刻意会「变红」）
//!
//! ① 实测值一旦漂移（优化器阈值、SCF 判据、格点），这里会红 —— 那是回归，要人来看；
//! ② **判据未达成**这件事也是断言的一部分：如果将来某个改动让 HF/STO-3G 的偏差真的进了
//!    1.5 kcal/mol，`assert!` 会故意失败，逼人回来把数字与结论一起改掉。
use std::path::PathBuf;

/// 1 Hartree = 627.5094740631 kcal/mol（CODATA 2018；与 `scripts/rp01_measure.py` 同一个常数）。
const KCAL_PER_HARTREE: f64 = 627.5094740631;

/// 本轮实测的能量（Hartree），逐字抄自 `result.json`。
///
/// * `HF_*`：HF/STO-3G、`grid_level: normal`、`options.max_opt_steps: 200`、
///   `opt_gradient_threshold_hartree_per_bohr: 4.5e-4`、`scf_max_iter: 200`、
///   能量/密度阈值 `1e-9` / `1e-7`。命令：`python scripts/rp01_measure.py --method hf
///   --basis sto-3g --grid normal --out .tmp-rp01-measure/dry-hf --stages minima,ts,irc`。
struct Measured {
    energy_hartree: f64,
}

const HF_HCN: Measured = Measured {
    // `r1_reactant`：2 步收敛。
    energy_hartree: -91.675_208_989_859_04,
};
const HF_HNC: Measured = Measured {
    // `r1_product`：4 步收敛。
    energy_hartree: -91.644_437_225_356_88,
};
const HF_HCN_TS: Measured = Measured {
    // `r1_ts`：2 步收敛，max|g| = 1.12e-4 Ha/Bohr，n_imaginary = 1，虚频 −1248.53 cm⁻¹。
    energy_hartree: -91.564_851_025_848_78,
};
const HF_H2CO: Measured = Measured {
    // `r3_reactant`：5 步收敛。
    energy_hartree: -112.354_347_089_541_82,
};
const HF_H2_TS: Measured = Measured {
    // `r3_ts`：27 步收敛，max|g| = 3.36e-4，n_imaginary = 1，虚频 −129.52 cm⁻¹ —— 但
    // **IRC 双向都走不动**（退出码 4：`irc: both probe directions increase the energy` 或
    // 「第一步没有降能」）。所以它是驻点，不是这条反应的鞍点，势垒不计入。
    energy_hartree: -112.342_769_708_455,
};

/// 文献值（从 CSV 读，不在这里写一份）。
struct Literature {
    value: f64,
}

fn literature(reaction: &str) -> Literature {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference/rp01_literature_barriers.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("读不到 {}: {error}", path.display()));
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 3 || fields[0] != reaction || fields[1] != "computed" {
            continue;
        }
        return Literature {
            value: fields[2].parse().expect("barrier_kcal_per_mol"),
        };
    }
    panic!("CSV 里没有 {reaction} 的 computed 行")
}

fn barrier_kcal(transition_state: f64, reactant: f64) -> f64 {
    (transition_state - reactant) * KCAL_PER_HARTREE
}

/// HF/STO-3G 的实测势垒，逐条钉住；并把与文献的差算出来。
///
/// **这个测试在量什么**：这一轮真正走通了 `opt → ts → irc` 全链的那个反应（HCN↔HNC）。它的
/// 势垒在 HF/STO-3G 上是 69.25 kcal/mol，与文献的 48.06 差 21.19 —— 差得离谱，而这正是
/// 「层级不够」的证据（STO-3G/HF 把 1,2-H 迁移的势垒算高了几十 kcal/mol）。
#[test]
fn the_hf_sto_3g_barriers_and_their_gap_to_the_literature_are_pinned() {
    // ── HCN ↔ HNC：唯一一个「ts + 双向 IRC」都验收过的反应 ────────────────────
    let hcn_barrier = barrier_kcal(HF_HCN_TS.energy_hartree, HF_HCN.energy_hartree);
    assert!(
        (hcn_barrier - 69.2507).abs() < 5.0e-3,
        "HCN↔HNC 的 HF/STO-3G 势垒 = {hcn_barrier:.4} kcal/mol（钉住的 69.2507）"
    );
    // 反应能（HNC − HCN）也必须对：它检验的是两个极小点，不是鞍点。
    let isomerisation = barrier_kcal(HF_HNC.energy_hartree, HF_HCN.energy_hartree);
    assert!(
        (isomerisation - 19.3096).abs() < 5.0e-3,
        "HNC − HCN = {isomerisation:.4} kcal/mol（钉住的 19.3096）"
    );
    // IRC 的双向端点（`irc_r1_ts`，退出码 0，两端 monotone）：forward 走到 HNC、
    // reverse 走到 HCN，与上面两个极小点的能量差 < 1e-6 Ha —— 鞍点**连对了**。
    let irc_forward = -91.644_437_224_800_09;
    let irc_reverse = -91.675_208_839_749_98;
    assert!(
        (irc_forward - HF_HNC.energy_hartree).abs() < 1.0e-6,
        "IRC forward 端 {irc_forward:.10} 不是 HNC"
    );
    assert!(
        (irc_reverse - HF_HCN.energy_hartree).abs() < 1.0e-6,
        "IRC reverse 端 {irc_reverse:.10} 不是 HCN"
    );

    let literature_hcn = literature("r1_hcn_hcnc").value;
    let gap_hcn = (hcn_barrier - literature_hcn).abs();
    assert!(
        (gap_hcn - 21.1902).abs() < 5.0e-3,
        "HCN↔HNC 的与文献差 = {gap_hcn:.4} kcal/mol（钉住的 21.1902）"
    );

    // ── H₂CO → H₂ + CO：驻点找到了，但 IRC 走不动 ─────────────────────────
    let h2co_barrier = barrier_kcal(HF_H2_TS.energy_hartree, HF_H2CO.energy_hartree);
    assert!(
        (h2co_barrier - 7.2649).abs() < 5.0e-3,
        "H₂CO 的 HF/STO-3G 驻点 = {h2co_barrier:.4} kcal/mol（钉住的 7.2649）"
    );
    // 这个数**不计入**：`irc` 对它的两个方向都报「能量不降」（退出码 4）。独立实现
    // （PySCF 2.14.0）在同一层级从 (d = 1.25 Å, H–H = 1.20 Å) 出发做同样的本征向量跟随，
    // 直接落到 H₂CO 的极小点（E = −112.3543471 Ha、无虚频）—— 也就是说 STO-3G 的
    // H₂CO 表面上**没有**这条反应的鞍点。两个实现指向同一个结论，所以这条不计入势垒。
    let literature_h2co = literature("r3_h2co_h2_co").value;
    println!(
        "H₂CO 的 HF/STO-3G 驻点 7.2649 vs 文献 {literature_h2co:.2}：差 {:.1}，且 IRC 没走通 → 不计入",
        (h2co_barrier - literature_h2co).abs()
    );

    // ── 判据（< 1.5 kcal/mol）一条都没达成：这是本轮的**实测结论** ──────────
    assert!(
        gap_hcn > 1.5,
        "HCN↔HNC 的 HF/STO-3G 偏差 {gap_hcn:.2} 竟然进了 1.5 kcal/mol —— 这个测试是给\
         「层级不够」钉的缺口，结论变了就要回来改它"
    );
    println!(
        "HF/STO-3G：HCN↔HNC 势垒 {hcn_barrier:.4} kcal/mol（文献 {literature_hcn:.4}，\
         差 {gap_hcn:.4}）；H₂CO 驻点 {h2co_barrier:.4}（IRC 未验收）；CH₃NC↔CH₃CN 鞍点未找到"
    );
}

/// **这一轮为什么没有 DFT 势垒**：把实测成本钉住，免得下一轮把它当"再跑一次就行"。
///
/// 这些数是墙钟（本机 16 逻辑核 / 32 GB，`threads: 4`），来自实际跑过的命令：
/// * HCN / PBE / def2-SVP / grid-normal `sp`：28.5 s（40 次迭代、72900 个格点）；
/// * HCN 同层级 `opt`（3 步）：457.8 s；
/// * HCN 的 `ts` 从 PySCF 的 PBE 鞍点几何出发：183 s 后失败于
///   `scf not converged in 200 iterations`。
///
/// 于是「一次数值 Hessian」与「一条双向 IRC」的量级是：HCN 22 min / 2–4 h，
/// H2CO 30 min / 3–5 h，CH3NC 45 min / 5–8 h。判据要的是三个反应**都**跑完 ts + IRC。
#[test]
fn the_pbe_def2svp_cost_that_blocked_the_measurement_is_pinned() {
    // 实测：单点 28.5 s；opt 的 3 步 457.8 s ⇒ 一步梯度 ≈ 152.6 s。
    let single_point_s = 28.5_f64;
    let opt_three_steps_s = 457.8_f64;
    let gradient_s = opt_three_steps_s / 3.0;
    assert!(
        gradient_s > 5.0 * single_point_s,
        "一步梯度（{gradient_s:.1} s）只有单点的 {} 倍 —— 这个成本模型变了，结论要重看",
        gradient_s / single_point_s
    );
    // 一次数值 Hessian = 6N 次梯度（`ts` 每一步一次）。
    let hessian_hcn_min = 9.0 * gradient_s / 60.0;
    let hessian_ch3nc_min = 18.0 * gradient_s / 60.0;
    assert!(
        (hessian_hcn_min - 22.9).abs() < 1.0,
        "HCN 的 Hessian ≈ {hessian_hcn_min:.1} min"
    );
    assert!(
        (hessian_ch3nc_min - 45.8).abs() < 2.0,
        "CH₃NC 的 Hessian ≈ {hessian_ch3nc_min:.1} min"
    );
    println!(
        "PBE/def2-SVP/grid-normal 实测：单点 {single_point_s:.1} s、一步梯度 {gradient_s:.1} s、\
         一次 Hessian（HCN 9 次梯度）≈ {hessian_hcn_min:.1} min、\
         （CH₃NC 18 次）≈ {hessian_ch3nc_min:.1} min；`ts` 在第 183 秒因 SCF 不收敛失败"
    );
}
