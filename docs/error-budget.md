# 误差预算 v2（VF-05）

本文档回答一个具体问题：**阶段二每一条近似目前有多大误差、怎么测出来的、与 §9.3 的容差表是否匹配**。
数字全部来自本仓库**可执行的测试**（本文档 §0/§1–§5 的每一行都给出测试路径，门禁
`scripts/check_error_budget.py` 会检查这些路径真的存在、且每行都带数字或明确的「未测/blocked」）。
没有测量的项写「未测」并给出被哪个 WP 挡住 —— **不许用估计值填空**。

阶段一（V-05）的基线保留在 §8，原文不动：那里的教训（Boys 函数、Becke 分区符号）不该因为
版本升级就消失。

## 0. 阶段二总表

「容差」列引用 `project-phase2.md` §9.3 的对应行（写 `§9.3:x`）或阶段一 `project.md` 的 T1 类判据。
「相对」= 相对被测量本身；能量差一律用 Ha（必要时在括号里给 kcal/mol）。

| 误差源 | 当前量级（实测） | 容差/判据 | 状态 | 证据 |
| --- | --- | --- | --- | --- |
| RI-J 拟合误差（水/def2-SVP，def2-SVP-RI 辅助基 76 函数） | ΔE_J = −6.139e-3 Ha（−3.85 kcal/mol） | 无 §9.3 行；只记录近似量级 | ⚠️ 记录 | `qwb-core/crates/qwb-scf/tests/in01_ri_fit.rs` |
| RI-J 拟合误差（辅助基退化为 def2-SVP / STO-3G） | −0.8375 Ha / −0.9157 Ha | 同上 | ⚠️ 辅助基选择比实现误差大 4 个量级 | `qwb-core/crates/qwb-scf/tests/in01_ri_fit.rs` |
| RI-J 实现一致性（同一辅助基 vs PySCF） | 1.37e-7 Ha（三种辅助基同值，K2 的 Å→bohr 常数偏移） | T1 外部参考 < 1 mHa | ✅ | `qwb-core/crates/qwb-scf/tests/in01_ri_fit.rs`、`qwb-core/crates/qwb-integrals/tests/in01_ri_integrals.rs` |
| RI-MP2 vs canonical MP2（H2O/def2-SVP） | 8.526e-5 Ha（0.054 kcal/mol） | `§9.3:RI-MP2 vs canonical` < 1e-4 Ha | ✅（余量 1.17×，很紧） | `qwb-core/crates/qwb-scf/tests/sc05_rimp2.rs` |
| canonical MP2 vs PySCF（六个体系/基组组合） | 1e-10…2e-9 Ha | T1 < 1 mHa | ✅ | `qwb-core/crates/qwb-scf/tests/sc04_mp2.rs` |
| 网格等级（H2/SVWN5，normal→fine） | 4.66e-12 Ha | 阶段一 0.1 mHa | ✅ | `qwb-core/crates/qwb-scf/tests/s02_grid_convergence.rs` |
| 网格等级（H2O/SVWN5，normal→fine） | 5.3e-6 Ha | 阶段一 0.1 mHa | ✅ | `qwb-core/crates/qwb-scf/tests/s02_grid_convergence.rs`（明细见 §8.2） |
| 网格求积能力（解析高斯，α = 1…300） | 相对 6e-12 | 阶段一 0.1 mHa | ✅ | `qwb-core/crates/qwb-grid/tests/x03_weights.rs` |
| meta-GGA 端到端（M06-2X/def2-SVP/grid-coarse vs PySCF） | 2.3e-5 Ha（M06-L 2.8e-5） | `§9.3` 无专门行；FX-02 判据 0.5 mHa | ✅（小 20 倍） | `qwb-core/crates/qwb-scf/tests/fx02_m06_water.rs` |
| DFT-D4 色散能（两体 + ATM，60 行 = 10 个二聚体 × 5 泛函 + q = ±1，vs 官方 dftd4 4.3.1） | 最差 **6.435e-13 Ha**（m06-l 单独 4.8e-13） | DA-06 §6：与官方实现同值（D3 那套 1e-3 kcal/mol ≈ 1.6e-6 Ha） | ✅（比判据小 7 个数量级） | `qwb-core/crates/qwb-scf/tests/da06_d4_energy.rs` |
| DFT-D4 中间量（配位数 / EEQ 电荷 / 电荷相关 C6，9 个二聚体 vs 官方 `get_properties()`） | 2.823e-9 / 1.037e-10 / 2.272e-9 Ha·Bohr⁶ | 无 §9.3 行；判据按实测能力定（1e-7 / 1e-8 / 1e-7） | ✅（留 30~100 倍余量） | `qwb-core/crates/qwb-scf/tests/da06_d4_properties.rs` |
| DFT-D4 解析梯度（vs 自身能量的 Richardson 中心差分 + 官方 `gradient_max_abs`） | 5.583e-14 Ha/Bohr / 相对 9.190e-10（60 行）；平移不变性 1.355e-18 | FX-04 那套「梯度 vs 有限差分 < 1e-6 Ha/Bohr」 | ✅（小 8 个数量级） | `qwb-core/crates/qwb-scf/tests/da06_d4_gradient.rs` |
| DFT-D4 的**近似来源**（不是实现误差）：`alpha(iω)` 只存 7 位小数的上游数据；23 点 `trapzd` 的 Casimir–Polder 积分；EEQ(2019) 的经典电荷；覆盖到 Z = 36；M06-2X 上游无参数 | C6 的可达精度 ~1e-8 相对（被 7 位小数封顶） | 无 §9.3 行，记录项 | ⚠️ 记录（ATM 的 C6 必须按**原子分电荷 = 0** 重建，否则差 3.42 倍） | `docs/choices.md` 一百五十八 |
| TDA vs PySCF（H2O/STO-3G，5 态） | 4.18e-7…5.89e-7 eV | `§9.3:激发能` 0.2 eV | ✅（小 6 个量级） | `qwb-core/crates/qwb-response/tests/rs02_tda_acceptance.rs` |
| TDA vs PySCF（H2O/def2-SVP，5 态） | 1.19e-7…2.91e-7 eV | 同上 | ✅ | `qwb-core/crates/qwb-response/tests/rs02_tda_acceptance.rs` |
| TDA vs PySCF（苯/STO-3G，6 态） | 最差 3.79e-7 eV | `§9.3:激发能` 0.3 eV | ✅ | `qwb-core/crates/qwb-response/tests/rs02_tda_benzene.rs` |
| TDA vs PySCF（甲醛、丙酮/STO-3G） | 7.4e-7 eV / 5.5e-7 eV | `§9.3:激发能` 0.5 eV | ✅ | `qwb-core/crates/qwb-response/tests/rs03_tda_formaldehyde.rs`、`qwb-core/crates/qwb-response/tests/vf01_tda_acetone.rs` |
| TDA vs **实验值**（H2O 第一单重态） | 13.20 eV vs 约 7.4 eV → 差 5.8 eV | `§9.3:激发能` 0.2 eV | ❌ 不可达（方法层级限制，见 §3） | `docs/status-v2.yaml` 的 VF-01 条目 |
| COSMO 球面离散（Born 球，M = 500） | 相对 3.31% = 2.71 kcal/mol | `§9.3:ΔG_solv` MAE < 1.5 kcal/mol | ⚠️ 单个球已超目标 | `qwb-core/crates/qwb-props/tests/sv01_cosmo.rs` |
| COSMO 导体极限（ε = 1e12） | 相对 4.32% = 4.79 kcal/mol | 同上 | ⚠️ | `qwb-core/crates/qwb-props/tests/sv01_cosmo.rs` |
| SASA（采样路径，水，2000 点）转动不变性 | 面积 4.99e-4 / 2.30e-3 / 3.48e-4 | SV-01 判据 < 1e-6 | ❌ 采样路径未达（解析路径 0.0） | `qwb-core/crates/qwb-props/tests/sv01_cavity.rs` |
| SASA（解析球帽 vs 采样，水） | 面积相对 **1.847e-1**（逐对**低估** 18.47%）、双原子 vs 20 万点采样 **8.5e-6**（H₂ 闭式 27.1433605270 Å²） | SV-01 判据：解析路径只在**双原子**时为准 | ⚠️ 已知近似（三球以上三重交叠重复扣除，方向恒定偏小；帽高 `h = R_i − z` 本轮订正） | `qwb-core/crates/qwb-props/tests/sv01_cavity.rs` |
| CDS 表面分解（Fibonacci 采样 + 球内侧切换带，双原子 vs 解析球帽，N = 256→16384） | 相对 2.84e-3 / 7.1e-4 / 3.6e-4 / 3.6e-4 | SV-03 判据：随点数收敛，N = 16384 时 < 5e-3 | ✅ | `qwb-core/crates/qwb-props/tests/sv03_smd_surface.rs` |
| CDS 单原子极限（逐元素 `4π(R+0.4)²`，7 个元素） | 相对 ≤ 5.5e-14 | 与已验证的 σ 层一致（水相 1.6e-5 cal/(mol·Å²)） | ✅ | `qwb-core/crates/qwb-props/tests/sv03_smd_surface.rs` |
| CDS 表面分解的转动漂移（采样路径，水，1024/4096/16384 点） | 1.036e-5 / 2.481e-5 / 7.186e-7 | 无 §9.3 行；记录（`O(1/√N)` 采样误差，不单调） | ⚠️ 记录 | `qwb-core/crates/qwb-props/tests/sv03_smd_surface.rs` |
| CDS 切换带宽（**模型选择**，不是数值误差；水面积 w = 0 → 0.8 Å） | 55.667 → 81.598 Å²（孤立和 110.664） | 无 §9.3 行；记录：单调、夹在二值与孤立和之间 | ⚠️ 记录 | `qwb-core/crates/qwb-props/tests/sv03_smd_surface.rs` |
| CDS 每原子 σ 的 H/成对修正（`HSIGMA(6) = −60.77`、`SIGMA(101..116)` 六个非零项） | 已实装：甲烷的 H 从 48.69 降到 **7.68**（`COT = 0.674892`），甲醇的 O 是 **2.45**（O–C 特例），丙酮的 O 是 **43.60**、乙腈的 N 是 **40.72** | SV-03 gap ⓪（已收口） | ✅ | `qwb-core/crates/qwb-props/tests/sv03_smd_sts.rs` |
| CDS **十二个体系×溶剂**能量 vs 上游 `libsolvent`（`w = 0`、32768 点/原子、含 H/成对修正） | 水相（水/HF/HCl/H₂S/甲烷/甲醇/丙酮/乙腈）：3.57e-4 / 3.86e-6 / 3.96e-6 / 2.04e-4 / 4.36e-4 / **8.43e-4** / 1.57e-4 / 2.27e-4；非水相（水/甲醇、甲烷/甲苯、甲醇/甲醇、乙腈/乙腈）：1.14e-4 / 3.45e-4 / 5.34e-4 / 2.96e-4；最差 **8.43e-4**（按各项绝对值之和归一，抵消相消放大） | SV-03：CDS 能量对官方实现 < 1e-3 | ✅ | `qwb-core/crates/qwb-props/tests/sv03_smd_sts.rs`（参考值 `tests/reference/smd_cds_legacy.csv`） |
| CDS 非水相 σ 与 `CSSIGM` 的拼装（`σ_N·SOLN + σ_A·SOLA + σ_B·SOLB`，`CSSIGM = 0.35γ − 4.19φ² − 6.68ψ²`） | 四个非水体系 1.14e-4…5.34e-4；甲苯里的甲烷各项 σ·A 绝对值合计 ~5.7 kcal/mol 而 `E_CDS` 只有 0.53（相消放大 13×，故判据用绝对值之和归一） | SV-03（已收口） | ✅ | `qwb-core/crates/qwb-props/tests/sv03_smd_sts.rs`（描述符来自 `qwb-data/solvent/*.yaml`） |
| CDS 的 O–C 成键特例（`SMXCDS`：Δ = 0.10、R = 1.330，而 `RKKVAL(O,C) = 1.84`） | 甲醇 `COT = 0.035674`（STS(O) = 2.45）；若漏用特例则 `COT = 0.667`、STS(O) = 45.8（差 **19×**） | 手算锚，COT 由文档公式直接算 | ✅ | `qwb-core/crates/qwb-props/tests/sv03_smd_sts.rs` |
| CDS **多原子**能量 vs 上游 `libsolvent`（水 / HF / HCl / H₂S，`w = 0`、8192 点/原子、零阶 σ） | 相对 1.25e-4 / 6.84e-5 / 3.36e-5 / **1.86e-4**（最差 H₂S）；官方值 1.441315 / 2.031181 / 1.176125 / 1.008290 kcal/mol | SV-03：多原子 CDS 能量对官方实现 < 1e-3 | ✅ | `qwb-core/crates/qwb-props/tests/sv03_smd_surface.rs`（参考值 `tests/reference/smd_cds_legacy.csv`） |
| CDS 表面分解的**带宽开起来**时对上游的偏差（水 / HF / HCl / H₂S，`w = 0.2`、1024 点） | 1.21e-1 / 8.47e-2 / 8.19e-2 / 1.20e-1 | 同上 | ⚠️ 记录：上游的等效表面更接近二值遮蔽，所以能量默认 `w = 0` | `qwb-core/crates/qwb-props/tests/sv03_smd_surface.rs` |
| CDS 多原子表面的**分原子面积** vs 上游 `DAREAL` | 未测（`get_cds_legacy` 只返回能量 + 梯度，不含面积；总能量层面的对照已由上一行覆盖） | 无 §9.3 行 | 未测 | `docs/status-v2.yaml` 的 SV-03 条目 |
| 完整 TDDFT（RPA/TDHF，ST-01）vs PySCF `tdscf.rhf.TDHF` | 4.6e-7…5.9e-7 eV（H2O/STO-3G）、1.2e-7…2.9e-7 eV（def2-SVP） | `§9.3` 无专门行；ST-01 判据 0.15 eV | ✅（小 5 个量级；相对 TDA 系统性下降 0.014–0.110 eV） | `qwb-core/crates/qwb-response/tests/st01_tddft.rs` |
| 短程 2e 积分（IN-02，ω ∈ {0, 0.1, 0.3, 0.6, 1.0}） | 相对 1.41e-14（与 u 空间独立闭式） | K10 判据 1e-9 | ✅ | `qwb-core/crates/qwb-integrals/tests/in02_short_range.rs` |
| 解析梯度 vs 数值（UHF：OH/O2） | 3.63e-7 / 7.29e-7 Ha/Bohr | `§9.3:梯度` < 1e-5 Ha/Bohr | ✅ | `qwb-core/crates/qwb-scf/tests/sc02_uhf_gradient.rs` |
| 2e Hessian 收缩 vs 冻结密度数值 | 1.87e-8 Ha/Bohr² | `§9.3:解析 Hessian` 5% | ✅ | `qwb-core/crates/qwb-scf/src/cphf_hessian.rs` |
| NMR 抗磁屏蔽（水/甲烷/甲醇，14 核） | 相对 2.62e-7（最大，甲醇 H） | `§9.3:NMR` 0.5/5.0 ppm（这里是绝对屏蔽，RS-04 判据 1e-6 相对） | ✅ 抗磁项；总屏蔽未测（顺磁项未实装） | `qwb-core/crates/qwb-props/tests/rs04_diamagnetic.rs` |
| pseudospectral（PS）Coulomb/exchange | **未测** | `§9.3:PS vs 解析` < 5e-5 Ha | ❌ 未实装（IN-03 pending、M-PS pending） | `docs/status-v2.yaml` 的 IN-03 条目 |
| ECP（LANL2DZ 等） | **未测** | T1 类 < 1 mHa | ❌ blocked（M1c） | `docs/status-v2.yaml` 的 M1c 条目 |
| 色散校正（D3(BJ)/D4） | **未测** | 无 §9.3 行 | ❌ blocked（DA-06/FX-04 缺数据） | `docs/status-v2.yaml` 的 DA-06 条目 |

## 1. RI（密度拟合）

**两个完全不同的"误差"必须分开**，混在一起写是本项目最容易自欺的地方：

1. **实现一致性**：同一个辅助基下，我们的 `E_J^RI` 与 PySCF 的差 **1.37e-7 Ha**（三种辅助基
   1.367e-7 / 1.372e-7 / 1.366e-7，是同一个常数偏移，来自 K2 的 Å→bohr 常数）。这说明
   「我们的 RI 与参考实现的 RI 是同一个 RI」。
2. **近似本身**：与**不拟合**的 4 中心 `E_J = 46.8450835264 Ha` 相比，RI 的结果是
   −6.139e-3 Ha（def2-SVP-RI）/ −0.8375 Ha（拿 def2-SVP 当辅助基）/ −0.9157 Ha（STO-3G）。
   也就是说：**辅助基选错时，近似误差是实现误差的 10⁴ 倍**；判据若只看第 1 条，
   会得出"RI 很准"的错误结论。

RI-MP2 同理：与 canonical 差 **8.526e-5 Ha**（判据 1e-4 Ha，余量只有 1.17×），与 PySCF 的
RI-MP2 差 1e-9，且 `V` 的秩 = 76 = 满秩（水/def2-SVP-RI 没有线性相关方向）。`ri_error_hartree`
会写进 `result.json` 的 `correlation` 块，所以这条误差是**留痕**的，不是只在文档里。

降级记录（不是误差，是性能）：苯/def2-SVP 上 canonical MP2 7.14 s vs RI 48.32 s = **0.15×**，
RI 路径更慢，按 §2.5 不宣称性能优势（见 `docs/status-v2.yaml` 的 SC-05、`docs/choices.md` 五十八）。

## 2. 网格

* 等级差：H2 从 normal→fine 只挪 **4.66e-12 Ha**、SVWN5 的 coarse→normal 是 5.49e-10 Ha；
  H2O/SVWN5 的 normal→fine 是 **5.3e-6 Ha**（明细见 §8.2）。全部远小于 0.1 mHa。
* 求积能力：对解析高斯 `e^{−αr²}`，α = 1…300 全范围相对误差 ≤ **6e-12**。
  这条必须保留：阶段一的分区符号事故（§8.2）就是"平滑函数测试全过、电子数不对"的那一类，
  求积能力的判据不能只测小 α。
* 与外部的一致性：M06-2X/def2-SVP/grid-coarse 对 PySCF 差 **2.3e-5 Ha**，这个数同时包含网格
  与 meta-GGA 实现的误差，是"网格 + XC 合成误差"的实测上界。

## 3. TDA（激发能）

* **与同方法参考**：H2O（两种基组）、苯、甲醛、丙酮，最差 7.4e-7 eV，判据 0.2/0.3/0.5 eV ——
  差 6 个量级。也就是说 TDA 的**实现**不是误差来源。
* **与实验值**：H2O 第一单重态我们 13.20 eV、实验 ¹B₁ 约 7.4 eV，差 **5.8 eV**；换 def2-SVP
  仍有 1.9 eV。原因明确：TDA（CIS）+ HF 参考系统性高估、STO-3G 无极化函数。
  **这条容差在当前方法层级不可达**，已在 `docs/status-v2.yaml` 的 VF-01 与
  `docs/choices.md` 六十里记账，判据按"方法匹配的参考库"读法执行。
  把这条留在表里（而不是删掉）是刻意的：它是"容差写的是方法精度、而实现只能保证一致性"的
  活样本。
* **完整 TDDFT（ST-01）**：加回 B 块后，H2O 上每个态比 TDA 低 0.014–0.110 eV（RPA 的典型
  方向），与 PySCF 的 `tdscf.rhf.TDHF` 差 4.6e-7…5.9e-7 eV（STO-3G）——**同方法参考**仍然
  是 1e-7 级，而**对实验值**的 5.8 eV 缺口一点没变（B 块只有几十 meV）。这正是"实现精度"
  与"方法精度"必须分开记账的最清楚例子。DFT 参考下的 TDDFT 还需要 XC kernel（FX-03），
  所以那一格仍是空的。
* 振子强度：与 PySCF 相对差 < 1e-5；`f-sum` 的缺口落在 `sum_rules.deficit`（H2O/STO-3G 为
  7.705945，即 Σf = 2.294 vs 10 个电子）—— 这是 TDA 缺多电子激发与非占据空间的必然结果，
  不是数值误差。

## 4. 溶剂与 SASA

* **COSMO 的球面离散误差**（单个 Born 球，解析值 −0.1306059087 Ha）：

  | M（tesserae） | ΔG（Ha） | 绝对误差 | 相对误差 | kcal/mol |
  | --- | --- | --- | --- | --- |
  | 125 | −0.1256738421 | +4.93e-3 | 3.78e-2 | 3.10 |
  | 250 | −0.1333949725 | −2.79e-3 | 2.14e-2 | 1.75 |
  | 500 | −0.1349219585 | −4.32e-3 | 3.31e-2 | 2.71 |

  误差**不随点数单调下降**，因为球面离散的方向与半径都不是正交网格；这是 COSMO 的已知行为
  （Klamt 的 1.07 对角修正就是为此）。按 §9.3 的 `ΔG_solv MAE < 1.5 kcal/mol`，**单个球的
  离散误差已经接近该目标**，所以任何 SMD/COSMO 的 ΔG_solv 结论都必须把 M 写进判据周围
  （VF-02 会继承这条约束）。导体极限（ε = 1e12）相对误差 4.32%（4.79 kcal/mol）。
* **SASA（球腔）**：解析球帽版本**严格转动不变**（三个转轴下面积/体积偏差都是 0.0，判据
  1e-6），单原子退回 4πR²/4πR³/3（< 1e-10），scale 比 1.4400000000000241。逐点采样版本
  的转动偏差是 **4.99e-4 / 2.30e-3 / 3.48e-4**（面积，取决于转轴），随点数
  500→2000→8000 从 4.17e-4 降到 3.48e-4 再到 8.70e-5 —— **这是采样误差，不是遮蔽判定 bug**。
  采样与解析在水上的差：面积 36.23682864 vs **29.5895950835**（逐对**低估 18.47 %**）、
  体积 19.28169843 vs 18.0004643640，来源是逐对球帽**重复扣除三重交叠**（方向恒定偏小）。
  **帽高本轮被订正过**：原先用切片位置 `z` 顶替 `h = R_i − z`，单原子与不交叠体系都看不出，
  H₂ 上偏大 15.4 %；订正后双原子与闭式一致到 1e-12、与 20 万点采样差 **8.5e-6**，
  三球以上的低估反而从 1.69e-2 变成 1.847e-1 —— 那是这个近似**本来**的口径，不是新误差。
  所以 SV-01 目前是 blocked：几何路径达标（解析，且只在双原子时当锚点），但判据要求的
  "与 SMD 非极性项配套的半径集"还缺（DA-05 的付费墙）。
* **CDS 多原子表面分解**（SV-03）：单原子极限逐元素 ≤ 5.5e-14（闭式 `4π(R+0.4)²`）；
  `w = 0` 与二值遮蔽的 `cavity()` 逐原子一致（2.95e-14）；双原子对解析球帽的收敛
  0.284 % → 0.071 % → 0.036 % → 0.036 %（N = 256→16384）；水面积随切换带宽单调
  55.667 → 81.598 Å²（孤立和 110.664）；转动漂移 1.036e-5 / 2.481e-5 / 7.186e-7（**采样
  误差**，不单调）。带宽本身是**模型选择**（上游 `SMXCDS` 用自己的点集与宽度），
  与上游逐位可比这件事**未测且不成立**。
* **pseudospectral（PS）**：**未测**。IN-03（PS Coulomb/exchange）是 stretch、M-PS 未开；
  §9.3 给它 < 5e-5 Ha 的容差，但没有任何实现可测。表里保留这一行并标"未测"，是防止
  "没有数据"被读成"误差为零"。

## 5. 与 §9.3 容差表的对应

| §9.3 容差 | 本文档对应量 | 现状 |
| --- | --- | --- |
| 总能量（同方法同基组，新旧实现）< 1e-8 Ha | 阶段一 §8.1 的积分/SCF 行；RI-J 实现一致性 1.37e-7 Ha 是**对 PySCF**而非新旧实现 | ✅ |
| RI-MP2 vs canonical < 1e-4 Ha | 8.526e-5 Ha | ✅（余量 1.17×） |
| PS vs 解析 < 5e-5 Ha | 未测（IN-03 pending） | ❌ 未实装 |
| 激发能 < 0.2–0.3 eV | 同方法参考 7.4e-7 eV ✅；对实验 5.8 eV ❌ | ✅/❌ 两种读法分歧见 §3 |
| NMR ¹H/¹³C < 0.5/5.0 ppm | 抗磁项绝对屏蔽 2.62e-7 相对；总屏蔽未测（RS-04 in_progress） | ⚠️ 半 |
| ΔG_solv MAE < 1.0/1.5 kcal/mol | COSMO 单球离散 1.75–3.10 kcal/mol（M = 250/125）；SMD 未实装 | ⚠️/❌ |
| 势垒高度 < 1.5 kcal/mol | 未测（RP-01/02/03 未实装） | ❌ 未实装 |
| 解析 Hessian vs 数值 < 5% | 2e Hessian 收缩项 1.87e-8（对冻结密度数值） | ✅（分量级） |
| 梯度 < 1e-5 Ha/Bohr | UHF 3.63e-7 / 7.29e-7 Ha/Bohr | ✅ |

## 6. 未测项与它们的阻塞（不许估）

| 量 | 被谁挡住 | 为什么不能估 |
| --- | --- | --- |
| PS 的 Coulomb/exchange 误差 | IN-03（pending，stretch）、M-PS（pending） | 没有任何实现 → 没有可测的差 |
| ECP 能量误差 | M1c（blocked：ECP 积分族未完成） | 同上 |
| D3(BJ)/D4 色散误差 | DA-06（blocked：缺 r4r2 表）、FX-04 | 没有参数表 → 实现不了 |
| SMD 的 ΔG_solv | DA-05（付费墙）、SV-01（blocked） | 没有非极性项参数 |
| 势垒高度 | RP-01/02/03（未实装） | 没有 TS 搜索 |
| NMR 顺磁项/化学位移 | RS-04（in_progress） | 顺磁项未实装；抗磁项已测 |

## 7. 维护方式

* 新增误差源必须在 §0 加一行，并给出**可执行**证据（测试路径或账本 WP id）；门禁
  `scripts/check_error_budget.py` 会检查路径存在、`当前量级` 列带数字或显式"未测/_blocked_"
  标记 —— 删行能让表变绿，但门禁也要求 §5 的每一行都有对应，所以删了会在别处红。
* 数字必须能在测试里复现：本版全部数字在 2026-11-08 由下表中的测试重跑得到（不是抄旧账本）：
  `in01_ri_fit`、`sc05_rimp2`、`s02_grid_convergence`、`rs02_tda_acceptance`、
  `rs02_tda_benzene`、`sv01_cosmo`、`sv01_cavity`。
* 若某条容差无法满足，写在 §3/§4 或 §6 那样的"已知不可达/未测"段落里，**不得改容差**。

## 8. 阶段一基线（v1 原文保留）

> 本节保留 `docs/error-budget.md` v1 的 §0–§6 原文（标题降一级）。其中的 `§n` 指 v1 的节号
> （对应本文档 §8.n）。保留的原因：§8.2 与 §8.4 记录的是本项目两笔最贵的学费，
> 升级文档版本不该把它们删掉。

### 8.0 总表

| 误差来源 | 当前量级 | 是否满足 §7/§6 要求 | 证据 |
| --- | --- | --- | --- |
| 双电子积分（含 p/d） | ≤ 1e-9 相对 | ✅（要求 1e-10～1e-8） | `i03_general_l.rs` 与独立数值路径逐分量对照 |
| 一电子积分（s，α 到 300） | ≤ 1e-16 相对 | ✅ | 与教科书闭式逐点比较 |
| 一电子积分（p/d） | ≤ 2.3e-6（γ 截断限制） | ✅ | 点电荷极限 `i02_nuclear_limit.rs` |
| SCF 收敛 | |ΔE| < 1e-10、max|ΔP| < 1e-6 | ✅（要求 1e-8） | s02/s03 测试 |
| 网格等级（轻原子，H2 normal vs fine） | 1.1e-8 Ha | OK（要求 < 0.1 mHa） | `grid_level_does_not_change_the_answer_much` |
| 网格等级（重原子，水的 XC） | 5.3e-6 Ha（normal→fine） | ✅（要求 < 0.1 mHa） | `s02_grid_convergence.rs`，见 §8.2 |
| 网格求积能力（解析高斯） | 相对 6e-12 | ✅ | `x03_weights.rs` |
| XC 泛函导数（LDA/GGA） | ≤ 9.7e-11 相对 | ✅（要求 1e-9） | `d10_pbe.rs` |
| 外部参照（HF/STO-3G H₂） | 3.1e-7 Ha（0.3 µHa） | ✅（要求 < 1 mHa） | `t01_reference.rs` |
| 数值梯度噪声地板 | ~5e-6 Ha/Bohr/原子 | ⚠️ 与解析梯度的 1e-6 目标同量级 | `g01_numeric_gradient.rs` |
| **水的绝对总能量** | 3.5e-5 Ha | ✅（要求 < 1 mHa） | `t02_water_hf.rs`，见 §8.4 |
| ECP | 未实现（I-05 阻塞） | — | — |

### 8.1 基组误差

* STO-3G 对 H 原子的能量误差约 **3e-3 Ha**（H 原子 STO-3G ≈ −0.4666 Ha，解析 −0.5 Ha）。
  因此 **V-01「氢原子 −0.5 Ha 误差 < 1e-6」用 STO-3G 无法满足**——这不是实现问题，
  是基组表示能力问题（单个收缩高斯无法重现 1s 的 cusp）。要满足它需要：
  ① 精确 Slater 积分（本引擎是高斯型，超出范围），或
  ② 在测试里用 ≥20 个原函数的高斯拟合（可做到 1e-6，但那是"拟合精度"而非"解析解"）。
  这条已在 `docs/status.yaml` 的 V-01 下记录为待人类裁定。
* 收缩函数的归一化约定与数据声明的一致性是**单独验证**的：STO-3G 的 ⟨1/r⟩ 与 Slater 的
  ζ/n 吻合（1s: 7.577 vs 7.658；2s/2p: 1.131/1.122 vs 1.138）。

### 8.2 网格误差

* 求积能力：原子网格对 `e^{−αr²}` 的积分相对误差在 α = 1…300 全范围内 ≤ 6e-12
  （`x03_weights.rs`；曾经只测到 α = 2，正是那个盲区漏掉了下面这个事故）。
* **Becke 分区的符号（已修复，2026-10-24）**：`smoothed_step` 曾返回 `½(1 + p^k(ν))`，
  而 Becke 标准式是 `½(1 − p^k(ν))`——正好是补。后果是分区权重反了：O 自己的网格在
  O 核心处拿到 ≈0 的权重，而以 H 为中心的网格在那里根本没有点。
  修复前后的对照（同一密度 P，`Tr(P·S) = 10`）：

  | 量 | 修复前 | 修复后 |
  | --- | --- | --- |
  | `∫ρ dr`（grid-fine） | 9.80184228 | **9.99999998** |
  | `∫ρ dr`（grid-normal） | 10.89471413 | 9.99997317 |
  | H₂O/SVWN5（coarse/normal/fine/veryfine） | −73.673 / −76.745 / −73.981 / — | **−74.9282545 / −74.9281445 / −74.9281498 / −74.9281495** |
  | H₂O/PBE（normal） | −77.43625901 | **−75.40186043** |

  修复后 H₂ 四个等级相邻差 5.5e-10 / 4.7e-12，H₂O/SVWN5 为 5.3e-6 / 3.1e-7 Ha，
  都在 §6.2 的 0.1 mHa 之内。验收测试：`s02_grid_convergence.rs`（∫ρ 与 H₂ 分级能量）。
* **教训**：`Σw = 4π`、单原子分区恒为 1、平滑高斯（α ≤ 2）——这三条测试在分区符号反了的情况下
  **全部通过**，因为平滑函数在核心处的贡献小到测不出来。唯一抓住它的是**电子数** `∫ρ`：
  核心贡献占 0.44 个电子，反了立刻少掉。详见 `docs/choices.md` §十八。
* **仍缺**（不影响上面的判据，但名不副实）：① Lebedev 点集表——`angular_rule: gauss-product`
  下 `lebedev_order` 实际是 `n = ⌈√order⌉` 的 Gauss–Legendre × 等距 φ 乘积；② Becke 的尺寸调整
  `ν = μ + a_ij(1−μ²)`（现在没有，水这种 H:O 尺寸差大的体系会略吃亏，但已进入 mHa 以内）；
  ③ Treutler–Ahlrichs 类带核心标度的径向映射。
* **已知语义坑**：`cutoff` 是径向映射的**尺度**而不是边界，最外节点落在约 62×cutoff
  （20 → 1248 Bohr）。对局域化被积函数无害（密度在那里早已为 0），但任何缓衰减的
  被积函数都会被大幅高估。详见 `docs/choices.md` §十二。

### 8.3 SCF 与积分

* SCF：ΔE < 1e-10、ΔP < 1e-6 双判据；收紧阈值到 1e-11 后能量不变（< 1e-7）。
* DIIS 使 H4 丛从 400 步不收敛降到 6 步（`docs/choices.md` §十一）。
* 2e 积分并行化（1/2/4 线程）**逐位相同**（FNV 哈希一致），归约顺序确定性由
  `DeterministicReduce` 保证。

### 8.4 水的绝对能量缺口（I-02）——已归因并修复

**这是本项目最贵的一课，所以完整留档。**

| 量 | 修复前 | 修复后 |
| --- | --- | --- |
| 我们的 HF/STO-3G 水（实验几何） | −72.876 Ha | **−74.962947 Ha** |
| 我们的 HF/STO-3G 水（CCCBDB 优化几何） | −72.846 Ha | —（不再需要这条对照） |
| 外部参照（实验几何，文献值） | −74.963 Ha | −74.963 Ha |
| 缺口 | ≈ 2.36 Ha | **3.5e-5 Ha** |

MO 能级也随之归位：我们 [−20.24, −1.27, −0.62, −0.45, −0.39] vs 文献 [−20.20, −1.20, −0.55, −0.42, −0.35]。

根因**不在**任何一个积分族、基组数据或 SCF 记账里，而在 **Boys 函数**：
`F_m(x)` 一律用向下递推、顶部固定多算 40 阶，种子误差被放大 `(2x)^top/(2top−1)!!` 倍，
`x = 33.9` 处 `F_0` 相对误差已达 8.6e-2。O 的 1s（ζ≈130）配 STO-3G 的水几何正好落在
`T ≈ 300–400` 一档，于是误差只出现在含重原子核心的积分里，并呈现为「每个重原子一个常数」
（水的 1a1 偏高 0.94 Ha × 2 电子；CH4 的 1a1 偏高 1.70 Ha × 2 电子）——这正是当时观察到
却解释不了的那个规律。修法见 `docs/choices.md` §十三。

当时已排除的方向（每条都有可执行证据，保留以免重走）：

1. 积分族：S、T、V、2e（含 98 个喂给核心 Fock 的元素）逐项独立验证通过；
2. 基函数：Slater ⟨1/r⟩ 与指数/收缩系数逐个核对；
3. SCF：手写最小 RHF 与 `rks_scf` 差 7e-10；三个截然不同的初猜收敛到同一能量；
4. 能量记账：`Tr(PH)+E_2e+E_nuc` 与 `½Tr(P(H+F))+E_nuc` 一致到 1e-9；
5. 几何与核排斥：与 CCCBDB 的 E_nuc 一致到 1.1e-3（坐标四舍五入）；
6. 参照值本身：已用 CCCBDB（NIST SRD 101）核实，不是记忆值。

**为什么这一整套排除法没能定位它**：上面 1–6 全都是「内部自洽」或「单点对单个元素」的检查，
而 Boys 函数是**所有积分共用的原语**。抬阶恒等式、核移动闭式、2e 张量的独立数值路径——
它们的参照值全都调用同一个 `boys_functions`，于是两边同比例错、差分为零。
教训已写进 `docs/choices.md` §十三：**自洽检验证明不了共用原语是对的**，原语必须另有
外部锚点（这里是 `F_0 = ½√(π/x)·erf(√x)` 与 `F_0(1)` 的已知值）。

附带收益：这条修复同时让 H₂O/STO-3G 的偶极进入可用范围（1.7257 D vs 实验 1.85 D，
差 0.124 D，S-06 ① 由此收口）。

### 8.5 GGA 的交换相关能对不上外部基准（已归因，保留）

这一条和 §8.4 不同：8.4 是"积分/网格把能量算错了"，已经修掉；8.5 是"泛函的两条曲线在同样的
流水线上系统性地偏"，最后定位到**两处公式约定**，不是数值问题。当时的形态（NIST CCCBDB
Release 22，STO-3G，CCCBDB 自己的优化几何；`t06_reference.rs`）：

| 模型化学 | H2 | H2O | 结论 |
| --- | --- | --- | --- |
| HF | +1.7e-7 Ha | −1.9e-7 Ha | 流水线正确 |
| LSDA (Slater + VWN5) | +3.2e-7 Ha | +1.7e-6 Ha | 两个 LDA 项正确 |
| B3LYP | +3.0e-4 Ha | **+1.9e-2 Ha** | 随核壳层放大 |
| PBE | **−2.9e-2 Ha** | **−1.76e-1 Ha** | 同一条线索 |

两处根因（都在 `docs/choices.md` 里留了完整推导）：

1. **PBE 的相关项用了错 LDA 基**（`docs/choices.md` 三十一）：PBE 的相关项配套的是 **PW92**，
   而当时套用了 LSDA 的 VWN5。LSDA 精确对上**不能**推出"LDA 两个分量都对"——同一项目里
   两套 LDA 基并存才是标准做法（libxc 就是这么组织的）。修好后 H₂O/STO-3G PBE =
   −75.238538040 Ha（**−4.0e-6 Ha** vs CCCBDB）。
2. **B88 的自变量约定错**（`docs/choices.md` 三十二/三十三）：B88 原始形式用**每自旋**的约化
   梯度与 `X_FACTOR_C`，我们用了总密度的 x 与 `C_x`；两项误差方向相反，净效果是梯度修正
   **弱 21%**。改对后 BLYP 从 +112.0 mHa 降到 **−0.0001 mHa**（H₂O），B3LYP 从 +19.0 mHa 降到
   **+0.002 mHa**，CO 上 BLYP/B3LYP 分别 +0.004 / +0.003 mHa。

**教训**：同一家族的泛函可能对"约化梯度用总密度还是每自旋"有不同约定（PBE 用 s_total、
B88 用 x_σ），照抄教科书公式时必须连自旋约定一起核。这条如今由 §0 的 meta-GGA 行
（M06-2X 2.3e-5 Ha）与 `sc01b_spin_xc_pointwise` 的逐点对照守着。

### 8.6 与阶段一 §7 容差的对应

| §7 容差 | 本文档对应量 | 现状 |
| --- | --- | --- |
| T1 能量 < 1 mHa（外部参考） | 外部参照行 3.1e-7 Ha | ✅ |
| T2/T3 的 LDA/GGA < 1 mHa | 网格 + XC 导数合成 | ✅（8.5 已归因，见 §0 的 meta-GGA 行） |
| 解析梯度 vs 数值 < 1e-6 Ha/Bohr | §0 的 UHF 梯度行 3.63e-7 / 7.29e-7 | ✅ |
| B3LYP < 2 mHa | — | ✅（`sc01b`/`t06` 系列，见账本 M1） |

### 8.7 维护方式（v1 原文）

* 本文件的数字**全部可在测试里复现**；新增误差来源时必须在表里加一行并给出可执行证据。
* `scripts/verify.ps1` / `verify.sh` 的门禁负责防止这些量回退（v2 起共 28 步，见 §7）。
* 若某条容差无法满足，必须在 §8.4 那样的"已知未归因"段落里显式记录，**不得删除行来让表变绿**。
