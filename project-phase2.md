# project-phase2.md · QuantumWorkbench v3.0（第二阶段）

> 本文件是**阶段二**的唯一开工依据。阶段一的规范是 [`project.md`](./project.md)（v2.1），
> 它**继续有效**：K1–K6 的语义、T1–T8 的验收、§14 的决策台账都不作废，阶段二的每一条
> 都建立在它们之上。两者冲突时以本文件为准（只允许本文件**放宽**阶段一的"非目标"，不允许
> 放宽任何验收）。
>
> 阶段一已于 2026-11-05 收口：T1–T8 全 `done`，门禁 26/26 全绿（`verify-r60.log`，
> Python 侧 182 例），`docs/status.yaml` 56 项 `done`。阶段二从这份基线出发。
>
> **v3.0（2026-11-06）**：首次成文。它刻意比阶段一**更宽**、也更允许失败 —— 见 §2.5。

---

## 0. 五分钟上手（新会话从这里开始）

目标：把 QWB 从"能算准的小分子闭壳层 DFT"推进到"**Jaguar 那一档的方法广度**"——
开壳层、meta-GGA 与 range-separated 泛函、色散校正、RI-MP2、TDDFT/CIS 激发态、
COSMO/SMD/PBF 溶剂化、过渡态与 IRC、解析 Hessian、NMR/极化率/VCD 谱学、以及（可选）
pseudospectral 数值路径。**仍然是产品面模仿 Jaguar、算法面全部自研。**

```powershell
cd D:\python_II\Jaguar

# 1. 看阶段二状态：哪些 WP done、下一个该做哪个
type docs\status-v2.yaml

# 2. 跑全部门禁（阶段一的门禁原样跑，一步都不许红）
powershell -NoProfile -File scripts\verify.ps1
bash scripts/verify.sh

# 3. 找下一个 WP：§6 中依赖全 done 且 ID 最小的那个；里程碑顺序见 §10.5
python scripts\check_status.py docs\status-v2.yaml
```

四条铁律（前三条继承阶段一，第四条是阶段二新增）：

1. **K1–K6 语义冻结**，只允许**加性**扩展（新可选字段、新变体、新 feature 词）。
   阶段二新增的 K7–K10 同样一经落地即冻结。
2. **每完成一个 WP 立刻写 `docs/status-v2.yaml`**（`state: done` + 验收命令输出摘要）。
3. **不得伪造结果、不得把未实装的能力标 `done`、不得用自算值冒充文献值。**
4. **阶段二允许"做不到"**：stretch 项达不到验收时，按 §2.5 的 kill criteria 记
   `state: blocked` 并交出**实测数字**（差多少、卡在哪、下一步要什么）。这不算失败；
   把"没做到"写成"做到了"才是失败。

---

## 1. 阅读顺序

| 你要做什么 | 看哪节 |
|---|---|
| 知道阶段二从什么基线上出发 | §2.0 |
| 理解"模仿 Jaguar"到哪一步、哪些明确不做 | §2.2 / §2.3 |
| 理解哪些条目允许失败、怎么记账 | §2.5 |
| 确认契约（K1–K6 冻结 + K7–K10 新增） | §4 |
| 我要做哪个包 | §6（WBS，每行都有可执行验收） |
| 阶段的"完成"刻度 | §7（里程碑 M1–M8） |
| 新增数据怎么来、许可怎么确认 | §8 |
| 验收怎么判（尤其是新的物理不变量） | §9 |
| agent 每步必须怎么做、卡住怎么办 | §10 |
| 已定决策（**别再重新讨论**） | §13 |

---

## 2. 目标与非目标

### 2.0 阶段一基线（已交付、已冻结）

阶段二的**起点**，任何一条回退了都算阶段二的 bug：

| 能力 | 状态 | 证据（阶段一） |
|---|---|---|
| 闭壳层 RHF / LDA / GGA / hybrid | ✅ | SVWN5、PBE、B88、LYP、PBE0、B3LYP（测试内构造）；CCCBDB 差 ≤ 4e-6 Ha |
| 基组 STO-3G / def2-SVP / def2-TZVP / 6-311G(d,p) / LANL2DZ（+ECP） | ✅ | 与 BSE 导出逐项 diff < 1e-12 |
| 单点 / 几何优化（BFGS + 解析梯度） | ✅ | H2O O–H 0.9623 Å（实验 0.9572） |
| 数值 Hessian + 简正模 + IR + 热化学 | ✅（慢） | PBE/6-311G(d,p)×0.991 最差 21.5 cm⁻¹；H2O/def2-SVP Hessian 155.6 s |
| 偶极 / Mulliken–Löwdin / 轨道能级 / cube / 布居 | ✅ | T2 断言 |
| 工作流：K3 状态机、`qwb batch` 队列与 `kill -9` 恢复、`qwb replay`（ΔE < 1e-10） | ✅ | T7 |
| GUI（PySide6 + VTK）+ Python API（pyo3 abi3）+ CLI（K5） | ✅ | T8 |
| 数据按需取回（`sources.json` 配方 + sha256 钉 + `qwb data install`） | ✅ | D-16 |
| 门禁 | ✅ | `scripts/verify.{ps1,sh}` 26/26（第 60 轮）；Python 182 例 |

**明确不在基线里**（阶段二要补或明确不做）：开壳层、meta-GGA、range-separated、色散校正、
MP2/任何相关方法、激发态、任何真实溶剂模型、过渡态/IRC/扫描、解析 Hessian、NMR/极化率、
VCD、pseudospectral、EPR、NBO。

**进入条件**：阶段二的任何 WP 开工前，先跑一次 `scripts/verify.ps1`（或 `verify.sh`）确认
26 项全绿、`docs/status.yaml` 的 T1–T8 仍 `done`。基线红就先修基线，不做新功能。

### 2.1 阶段二目标

1. **方法广度**：覆盖 Jaguar 常用方法集合的**可算子集**——开壳层（UHF/ROHF/UKS）、
   meta-GGA（M06-2X）、range-separated hybrid（ωB97X-D、CAM-B3LYP）、色散校正（D3(BJ)/D4）、
   RI-MP2、CIS/TDDFT（TDA）。
2. **物理化学能力**：几何优化之外的**反应路径**（过渡态搜索、IRC、刚性/弛豫扫描）、
   **隐式溶剂**（COSMO → SMD，PBF 为 stretch）、**谱学**（解析 Hessian、IR/Raman、NMR 屏蔽、
   静态/动态极化率、VCD 为 stretch）。
3. **数值路径**：至少新增两条**独立数值路径**（RI 密度拟合、pseudospectral Coulomb/exchange），
   并且它们必须与解析路径在约定容差内等价（K10）——**等价性是准入条件，不是加分项**。
4. **产品面**：新作业类型进 CLI/Python/GUI（含 PES 扫描与谱图可视化、结果对比表），
   K5 只做加性子命令，K3 状态机不变。
5. **可信度**：每个新能力都要有**外部参照**（文献值 / CCCBDB / 已发表基准集）或
   **独立数值路径**的交叉验证；`docs/error-budget.md` 每个误差源都有实测量级。

### 2.2 与 Jaguar 的对齐范围（第二阶段）

Jaguar 的功能面见 §12.3 的两个来源。列"✅ / ⚠ / ❌"时以**能不能通过 §9 的验收**为准，
不以"有接口"为准。

| Jaguar 的能力 | 阶段一 | 阶段二计划 | 说明 |
|---|---|---|---|
| RHF / LDA / GGA / hybrid DFT | ✅ | ✅ 保持 | — |
| **UHF / ROHF / UKS（开壳层）** | ❌ | ✅ `SC-01` | 双峰、自旋污染 ⟨S²⟩ 必须报出来 |
| **meta-GGA（M06-2X 一族）** | ❌（K1 拒 `requires.laplacian`） | ✅ `FX-01` / `FX-02` | 需要 τ 与 ∇²ρ；K1 的 `requires` 做加性扩展 |
| **range-separated hybrid（ωB97X-D、CAM-B3LYP）** | ❌ | ✅ `FX-03` / `IN-02` | erfc 分域库仑；ω 来自数据 |
| **色散校正（D3(BJ) / D4）** | ❌ | ✅ `FX-04` / `DA-06` | 参数表进数据，含解析梯度 |
| **LMP2 / RI-MP2** | ❌ | ⚠ 只做 **RI-MP2**（canonical） | LMP2 是 stretch（`ST-03`）；先用小体系对齐文献 |
| **GVB-PP / GVB-LMP2** | ❌ | ❌ 不做 | 与 K2/K3 的"单参考"语义冲突，收益/成本不划算（§2.3） |
| **CIS / TDDFT 激发态** | ❌ | ✅ TDA 形式 `RS-02` / `RS-03` | 完整 TDDFT（含 TDA 之外的耦合项）为 stretch |
| 几何优化 | ✅ | ✅ 保持 | — |
| **过渡态搜索（QST 类）** | ❌ | ✅ `RP-01` | 只做"给初猜 + Hessian 引导"的局部搜索，不做全局构象搜索 |
| **IRC** | ❌ | ✅ `RP-02` | 质量加权；两端必须落到正确极小 |
| **扫描 / PES** | ❌ | ✅ `RP-03` | 刚性 + 弛豫两种；进 GUI |
| 频率（IR） | ✅（数值） | ✅ 解析 Hessian `RS-05` | 数值路径保留为回归对照 |
| **NMR（GIAO 屏蔽）** | ❌ | ✅ `RS-04` | 需要 GIAO 积分 + CPHF；1H/13C 分开验收 |
| **极化率（静态/动态）** | ❌ | ✅ 静态 `RS-01`；动态 = TDDFT 的副产品 | 静态用有限场交叉验证 |
| **Raman / VCD** | ❌ | ⚠ Raman `RS-06`（stretch）；VCD `RS-07`（stretch） | 都依赖极化率导数；失败要按 §2.5 记账 |
| **UV/Vis 谱** | ❌ | ✅ `PR-04`（TDDFT 振子强度 + 展宽） | 数据侧只需光谱展宽参数 |
| **pKa 预测** | ❌ | ⚠ `ST-04`（stretch） | 需要溶剂化 + 热化学 + 经验校正，链条长 |
| 溶剂化：**PBF（Poisson–Boltzmann）** | ⚠ 只有字段 | ⚠ COSMO→**SMD** `SV-01..03`；PBF `SV-04`（stretch） | PBF 要有限元/边界元解，风险最高 |
| 分子表面（ESP/密度/MO 等值面） | ✅ 部分（cube + 3D） | ✅ 保持 + ESP 着色改进 `PR-05` | — |
| **multipole moments（> 偶极）** | ❌ | ✅ `RS-08`（四极/八极，含 traceless 形式） | 成本低、验收干净 |
| **pseudospectral（PS）积分** | ❌（明确排除） | ⚠ `IN-03`（**可选路径**，K10 门禁） | 阶段一排除它是因为它会搅浑验收；阶段二把它关进"等价性门禁"里再试 |
| EPR / hyperfine | ❌ | ❌ 不做（UKS 之后可低成本加，`ST-05`） | — |
| NBO | ❌ | ❌ 不做 | 要做就单独一个阶段（方法学自洽性要求高） |
| QM/MM（QSite）、显式溶剂 MD、周期体系 | ❌ | ❌ 不做 | — |
| MPP / GPU 并行 | ❌（仅 rayon 确定性归约） | ⚠ 只做**线程**并行扩展（`IN-05`/`SC-06`），不做 MPI/GPU | — |

**"⚠"的含义**：允许按 §2.5 收敛成 `blocked` 或降级成更弱的判据，但必须留下实测数字。

### 2.3 非目标（第二阶段仍然不做）

- 多参考（CASSCF/CASPT2/MRCI）、耦合簇（CCSD/CCSD(T)/EOM-CC）、GVB 家族。
- 相对论（4c/2c）、旋-轨耦合、磁性质（除 NMR 的屏蔽张量与 VCD 的磁偶极项）。
- 周期体系、QM/MM、显式溶剂与 MD、自由能微扰（FEP）。
- 全局构象搜索、反应网络自动搜索、机器学习势。
- MPI / 多节点 / GPU 内核。
- NBO、EPR、pKa（pKa 仅 stretch，见 §2.2）。
- **不改 K1–K6 的语义**（加性扩展见 §4.1）。

### 2.4 允许使用的第三方库（阶段二增补）

**原则：阶段二不新增引擎依赖。** 现有白名单（`project.md` §2.4）已覆盖：
`nalgebra`/`faer`、`rayon`、`ndarray`、`serde*`、`sha2`、`proptest`、`criterion`；
Python 侧 `numpy`、`pydantic`、`PySide6`、`VTK`、`pytest`、`hypothesis`、`mypy`。

若某 WP 确实需要新库（例如稀疏线性求解器、FFT、特殊函数），按 §10.4 走批准流程，
并在 `qwb-core/deny.toml` 白名单里登记**理由**。默认做法是**自己写**：
Bessel/erfc/球谐求值都在现有 `qwb-grid`/`qwb-integrals` 里，不引入 SciPy。

**外部程序（黑盒参照）**：延续 `project.md` §14 #12 —— PySCF（WSL 内）与文献值只用来
**产数字**，代码与数据一个字节都不进本仓库；调用脚本放 `scripts/`，产物进
`qwb-core/tests/reference/*.csv` 并带 `source_doi`/`source_url`。

### 2.5 Stretch 条款与 kill criteria（**阶段二特有**）

阶段一不允许"做不到"。阶段二把高风险条目单独标记，避免整个阶段被一条卡死：

| 级别 | 标记 | 规则 |
|---|---|---|
| 必做 | 无后缀 | 达不到就是该 WP `blocked`，且**阻塞依赖它的里程碑** |
| 可降级 | `⚠` | 允许把判据从"外部文献对照"降到"独立数值路径交叉验证"，但必须在 `note` 里写明降级理由与差距 |
| 可失败 | `stretch` | 允许以 `blocked` 收口：必须交出 ①实测数字 ②差多少 ③卡在哪一步 ④下一步需要什么（人/数据/算力/文献） |

**kill criteria（达到任一条即停手并记 `blocked`）**：

1. 该 WP 的**解析/独立路径对照**已经差到无法用"实现 bug"解释（> 1e-3 相对），且排查
   两轮后仍无收敛方向；
2. 需要新增**引擎依赖**而按 §2.4 未获批准；
3. 需要的**参考数据/文献**取不到，且没有可接受的独立路径（`project.md` §7.0 的 fallback
   也不适用）；
4. 单次验收运行超过 **8 小时**墙钟仍拿不到可判定的结果（性能不可接受 = 判据不可执行）。

**禁止**：把 stretch 项的"部分实现"写进"能力一览"；在 `docs/USAGE.md` 的诚实清单里
必须按实际状态标注（`✅ / ⚠ 部分 / ❌ 未做`），与 `docs/status-v2.yaml` 一致。

---

## 3. 组织原则

1. **契约先行**：K7–K10 先落到 `qwb-contracts/`（schema + Rust 类型 + 测试），再写实现。
   加性扩展 K1/K2/K3 时，一次只加一个字段组，并同步 `.pyi`、schema、`*.md`。
2. **数值路径必须成对**：任何近似（RI、PS、grid 阈值、TDA、冻结核）都要有一个**独立**
   参照路径，差值写进测试与 `docs/error-budget.md`。这是阶段二的核心纪律。
3. **物理不变量优先于"看起来对"**：f-sum rule（TRK）、gauge invariance（NMR 的
   length/gauge 一致性）、rotational invariance（极化率张量）、translational invariance
   （梯度）、variational upper bound（CIS/TDA 能量 ≥ 精确）、⟨S²⟩ 自洽性。
4. **每个工作包都可独立验收**，验收必须是可复制命令或可检查断言；没有"感觉差不多"。
5. **垂直切片推进**：先让一条端到端链路（M1–M8）通，再补精度；不做"层内串行"。
6. **回归优先**：阶段一的 25 项门禁与 T1–T8 断言**一步都不许红**。新能力的门禁**只加不减**。
7. **agent 不被模糊处卡死**：沿用 `project.md` §10.3 的分级规则；阶段二再加一条
   "选**代价更低**的解释"——当两种解释都能通过验收时，选实现更短、依赖更少的那个，
   并记录在 `note`。

**风险优先级**（阶段二调整）：正确性/可验证性 > 集成风险 > 性能 > 功能覆盖。
理由：阶段二的方法都带近似，**性能可以慢慢补，错的近似会污染整条结论链**。

---

## 4. 契约

### 4.1 K1–K6：冻结 + 加性扩展清单

**K1–K6 的语义一个都不改。** 阶段二需要的扩展全部是**可选字段 / 新变体 / 新 feature 词**，
旧输入文件、旧结果目录、旧 Python 代码必须继续工作（有回归测试）。

| 契约 | 加性扩展 | 兼容性 |
|---|---|---|
| **K1** `functional` | `requires` 增加 `laplacian` / `kinetic_density`；新增 `range_separation: {omega_bohr_inv, scheme}`（可选）；`components` 增加 `id: MetaGgaM06x` 等新组件名；新增数据类型 `dispersion`（见 §8） | 旧文件缺字段即默认 `false`/`null`；老泛函字节不变 |
| **K1** `solvent` | 新增 `model: vacuum\|cosmo\|smd\|pbf`（可选，缺省 `vacuum`）、`refractive_index`、`radii_set`、`surface_tension` 等 §8 列出的字段 | 旧 8 溶剂文件只需补 `model: vacuum` 或用缺省 |
| **K1** `basis` | 新增 `role: ri-aux`（辅助基）与 `fitting: {for: mp2\|jk}` 可选块 | 旧文件不受影响 |
| **K2** `TaskOptions` | `scf_type: rhf\|rohf\|uhf\|rks\|uks`、`numerics: {path, accuracy_level}`、`solvent_model`、`dispersion: {kind, params}`、`tddft: {n_states, ...}`、`mp2: {...}`（全部 `Option`） | 旧 `input.yaml` 逐字段仍可解析 |
| **K2** `SUPPORTED_FEATURES` | 新增词：`jobtype:ts`、`jobtype:irc`、`jobtype:scan`、`method:mp2`、`method:tddft`、`solvent:cosmo`、`solvent:smd`、`numerics:pseudospectral`、`numerics:ri` | 旧 feature 词不变；`features()` 只增 |
| **K2** `JobResult` | 新增**可选块**：`excited_states`、`nmr_shieldings`、`polarizability`、`multipoles`、`solvation`、`correlation`、`numerics_report`、`dispersion_energy_hartree`、`spin` | 旧结果目录仍通过 `open_result_dir` 自检（`serde(default)`） |
| **K3** | 新增 `artifacts/` 里的**可选**文件（`spectrum.csv`、`pes.json`、`irc.json`、`ts_search.json`）；状态机**仍 6 条转移** | 旧结果目录不受影响 |

> **K2/K3 的结果块不走 `schemars`**（阶段二实施时的澄清）：`engine-api` 保持**零 schema 依赖**
> 是 C-03 的既定约定，所以 K7–K9 的类型在 `qwb-contracts/studio/result.schema.yaml` 里是
> **手写文档**，由 `qwb-io/tests/k3_schema_docs.rs` 逐字段钉住；`schemars` 只用于 K1 的数据类型。
> 阶段二最初写的"K7 由 schemars 生成"对 K2 不成立 —— 按实际约定执行，已记入 `docs/choices.md`。
| **K5** | 新增子命令 `qwb props`、`qwb scan`、`qwb ts`、`qwb irc`（全部加性）；退出码表**不变** | 旧脚本继续工作 |
| **K6** | `.pyi` 增加新类/新方法/新字段（全部可选参数） | `mypy --strict` 保绿 |

> **K6 的扩展与 PR-02 同步做**（阶段二实施时的澄清）：`.pyi` 是 `qwb_engine` 绑定的**类型桩**，
> 桩里声明了而 binding 没实现，就等于桩在撒谎（`mypy` 会绿，运行时 `AttributeError`）。
> 所以 KV-05 只落 CLI；`TaskOptions`/`JobResult` 的新字段进桩必须与 pyo3 侧同一提交（§6.9 PR-02）。

### 4.2 K7 · 响应与性质契约（新增）

`qwb-props` 的第二套能力面，供 CPHF/TDDFT/NMR/极化率共用。**下面只是形态示意**：
字段名与类型以实际 Rust 类型为准，schema 由 `schemars` 生成（`project.md` §14 #1 的约定不变）。

```text
ExcitedStates {
  method: "cis" | "tda" | "tddft",
  n_states: usize,
  transitions: [{
    index, energy_hartree, wavelength_nm, oscillator_strength,
    rotatory_strength: Option<f64>,       // VCD/ECD 用
    dominant: [{from, to, weight}]        // 展成 MO 对
  }],
  sum_rules: { trk_oscillator: f64, n_electrons: f64 }   // f-sum rule 自检
}

NmRShieldings {
  method: "giao",
  gauge_origin: [f64; 3],
  nuclei: [{element, index, isotropic_ppm, anisotropy_ppm, tensor: [[f64;3];3]}],
  reference: Option<{compound, shift_ppm}>,   // 化学位移的参考物
  length_gauge_spread_ppm: f64               // 一致性自检
}

Polarizability {
  static: [[f64; 3]; 3],                     // a.u.
  frequencies_hartree: [f64],                // 动态（可选）
  dynamic: Option<[[[f64; 3]; 3]]>,
  rotational_invariants: {iso, aniso}        // 转动不变量
}

Multipoles { dipole_debye, quadrupole: {traceless: [[f64;3];3]}, octupole: Option<...> }
```

**K7 门禁**：① `sum_rules.trk_oscillator` 与电子数差 < 1e-3（CIS/TDA 完备基下成立，
有限基下记录差距）；② `length_gauge_spread_ppm < 1 ppm`（NMR，同基组、同原点附近）；
③ 极化率张量对转动协变（把分子转 30° 再算，主轴不变量不变 < 1e-6）；④ 静态极化率与
有限场（±0.001 a.u. 场下的能量二阶差分）差 < 1e-4 a.u.。

### 4.3 K8 · 溶剂化契约（新增）

```text
SolvationResult {
  model: "cosmo" | "smd" | "pbf",
  total_hartree,                    // 含溶剂的总能量
  electrostatic_hartree,            // 极化/库仑项
  cavitation_hartree: Option<f64>,  // SMD 的 3 项非极性贡献
  dispersion_hartree: Option<f64>,
  repulsion_hartree: Option<f64>,
  surface: { area_angstrom2, volume_angstrom3, n_tesserae },
  iterations: usize,
  converged: bool
}
```

**K8 门禁**：① 同分子气相/溶剂化能量差可复现（两次运行 byte-identical）；
② COSMO 在 ε→∞ 极限下与真空能量差的符号与量级合理（水分子 −8~−12 kcal/mol 量级）；
③ 表面面积与体积对转动不变（< 1e-6 相对）；④ 溶剂化梯度与有限差分一致
（< 1e-5 Ha/Bohr，仅 COSMO/SMD 的解析梯度可达，PBF 允许"无数值梯度"并显式拒绝 `opt`）。

### 4.4 K9 · 开壳层与相关方法契约（新增）

```text
SpinResult { n_alpha, n_beta, multiplicity, s_squared, spin_contamination, converged_broken_symmetry: bool }
CorrelationResult { method: "mp2" | "ri-mp2", scs: bool, e2_hartree, e_total_hartree,
                    aux_basis: Option<String>, ri_error_hartree: Option<f64> }
```

**K9 门禁**：① `s_squared` 与精确值差 < 1e-6（闭壳层）；开壳层按分子给期望值；
② `spin_contamination`（`⟨S²⟩ − S(S+1)`）必须报出来，不允许沉默；
③ 相关能对基组单调（cc-pVDZ→TZVP 之间方向正确）；
④ RI-MP2 与 canonical MP2 差 < 0.1 mHa（同基组，用小体系断言）。

### 4.5 K10 · 数值路径等价契约（新增）

**这是阶段二最重要的一条契约。** 任何"更快但更糙"的路径都必须满足：

```text
NumericsReport {
  path: "analytic" | "ri" | "pseudospectral",
  accuracy_level: u8,               // 1..5，语义见 §8.5（对应 Jaguar 的 iacc 思路）
  grid: Option<{radial, angular, cutoff_bohr}>,
  ri: Option<{aux_basis, fit_error_hartree}>,
  ps: Option<{grid, dealiasing, coulomb_error_hartree, exchange_error_hartree}>,
  energy_hartree, wall_time_s
}
```

**K10 门禁**（三条同时满足才允许该路径进入 `SUPPORTED_FEATURES`）：

1. **等价性**：同一体系同基组，与解析路径的**总能量**差 < `equiv_tol`，
   其中 `equiv_tol` 由数据侧声明（`numerics.yaml` 的 `equiv_tol_hartree`，默认 1e-6 Ha，
   PS 允许 5e-5 Ha 并在报告里显式写出）；
2. **收敛性**：`accuracy_level` 提高一级时误差**单调下降**（至少三点，回归测试钉住）；
3. **可复现/确定性**：同一 level 下 1 线程与 N 线程 byte-identical（继承 G7），
   且 `numerics_report` 必须进 `result.json`——**用了近似必须留痕**。

### 4.6 契约升级流程

沿用 `project.md` §10.4：先 `docs/choices.md` 登记理由与影响面 → 改
`qwb-contracts/CHANGELOG.md` → 同步 Rust 类型 / schema / `.md` / `.pyi` / 测试 →
人类评审 → 修订本文件 §4 并 bump 版本（§14）。

---

## 5. 依赖 DAG

```
阶段一基线（K1–K6 冻结 + T1–T8 + 25 项门禁）
   │
   ├──> KV（契约 K7–K10）── 必须先落，其它轨只依赖它
   │
   ├──> DA（数据：meta-GGA/RS/色散/辅助基/溶剂/参考集）
   │        │
   ├──> IN（erfc 积分、3 中心 RI、PS 路径、线程并行）
   │        │
   ├──> FX（meta-GGA、RS-hybrid、色散）──┐
   ├──> SC（开壳层、MP2/RI-MP2）──────────┤
   │                                      ▼
   │                              M1 开壳层 → M2 metaGGA/D → M3 RI-MP2
   ├──> RS（CPHF → 极化率/Hessian → CIS/TDA → NMR → Raman/VCD → 多极）
   │                              ▼
   │                        M4 激发态 → M7 谱学套件
   ├──> SV（COSMO → SMD → PBF）──▶ M5 隐式溶剂
   └──> RP（TS → IRC → 扫描）────▶ M6 反应路径
                                      │
                              PR（产品面）──▶ M8
                                      │
                              VF（验证/基准集）── 贯穿
```

**关键规则**：

- `KV-*` 未 `done` 前，**不要**开始写对应能力的实现（避免"实现牵引契约"）。
- `RS-01`（CPHF）是 `RS-02/RS-04/RS-05/RS-06` 的公共前置；先把 CPHF 做成
  可复用的 `qwb-response` crate，再做各性质。
- `IN-02`（erfc 积分）必须先于 `FX-03`（RS-hybrid）。
- `IN-01`（3 中心 RI）必须先于 `SC-05`（RI-MP2）与 `IN-04`（RI-JK）。
- `IN-03`（PS）**不阻塞任何里程碑**（`stretch`）：它只吃 `K10`。

---

## 6. WBS 工作包

> 每个包：`ID · 依赖 · 产物 · 验收`。验收必须是**可复制执行的命令**或**可检查断言**。
> 状态记录在 `docs/status-v2.yaml`。`⚠`/`stretch` 的含义见 §2.5。

### 6.0 状态总览

| 轨道 | WP 数 | 状态 |
|---|---|---|
| KV（契约 K7–K10 + 加性扩展） | 5 | 全部 done |
| DA（数据） | 8 | 全部 pending |
| IN（积分/数值路径） | 5 | 全部 pending |
| FX（泛函） | 4 | 全部 pending |
| SC（SCF/相关/开壳层） | 8 | SC-01 done；SC-01b in_progress；SC-01c blocked；其余 pending |
| RS（响应与性质） | 8 | 全部 pending |
| SV（溶剂化） | 5 | 全部 pending |
| RP（反应路径） | 4 | 全部 pending |
| PR（产品面） | 8 | 全部 pending |
| VF（验证） | 6 | 全部 pending |
| ST（stretch 备选池） | 5 | 全部 pending（允许 `blocked`） |
| **合计** | **66** | — |

### 6.1 契约轨（KV）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| KV-01 | 基线 | K7 结果块：`ExcitedStates`/`NmrShieldings`/`Polarizability`/`Multipoles`（`engine-api` 的 Rust 类型 + `studio/result.schema.yaml` 的 `$defs` + `qwb_studio.models` 的 pydantic 模型） | `qwb-io/tests/k3_schema_docs.rs` 的 `result.schema.yaml` ↔ `JobResult` **逐字段一致**（手写 schema + 一致性测试 = C-03 的约定；`engine-api` 保持零 schema 依赖）；`test_models.py` 的 pydantic ↔ schema 一致；新增块全部可选，旧 `result.json` 仍解析（回归测试） |
| KV-02 | KV-01 | K8 溶剂化类型 + `Solvent` 加性字段（`refractive_index`/`radii_set`/`surface_tension`/`ionic_strength`）+ `JobInput.options.solvent_model` | `qwb-gen-schemas --check` 不漂移（K1 schema 重新生成）；旧 `solvent/*.yaml` 仍 `qwb data verify` 通过；SMD 缺 n 或半径集时 verify 明确报错；`n² ≤ ε` 的自洽检查有单测 |
| KV-03 | KV-01 | K9 开壳层/相关方法类型 + `TaskOptions.scf_type`/`mp2`/`excited_states` + `JobResult.spin`/`correlation` + 跨字段校验 | 旧 `input.yaml` 全部仍可解析（`phase1_options_json_still_parses_without_the_new_keys`）；`SUPPORTED_FEATURES` 只增不删；**引擎对未实现的选项显式拒绝**（`Unsupported`，退出码 4，消息带 WP 编号），与阶段一等价的写法（`rks`/`rhf`/`vacuum`）放过 |
| KV-04 | KV-01 | K10 `NumericsReport`（K2）+ `numerics` 数据域（K1 `NumericsDef`、`qwb-data/numerics/numerics.yaml`、`registry.yaml` 的 `numerics:` 段）+ `data verify` / `data list` 支持新域 + 加载器的跨文件引用校验 | `qwb data verify` 17/17 通过（新增 1 个数据文件）；`numerics.schema.{json,yaml}` 生成且 `--check` 绿；等级重复/越界、容差过松、PS 配点不单调、去别名回退各有单测；等级指向不存在的网格时 `Registry::open` 返回 `Data`；`NumericsReport` 的 `within_tolerance` 与 `validate`（无留痕不算通过）有单测 |
| KV-05 | KV-02..04 | K5 加性子命令（`props`/`ts`/`irc`/`scan`：签名 + `--help` + 明确的 `Unsupported` 退出码 4）+ `cli.md` 同步。**K6 `.pyi` 与 pyo3 binding 的扩展移到 PR-02**（理由见 §4.6 末尾） | 四个子命令 `--help` 退出 0；未实现时退出码 4 且消息指出归属 WP；输入文件不存在时退出码 1（不被 4 盖掉）；既有子命令的签名与退出码未变（K5 冻结表）；`cli.md` 与实际 `--help` 一致 |

### 6.2 数据轨（DA）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| DA-01 | KV-01 | `M06-2X` 泛函数据（τ 依赖、参数、citation）+ `requires.kinetic_density: true` | `qwb data verify` 通过；参数逐个带 citation；与文献给定值逐项 diff == 0（参数数量在 20 以内，全列进 PR） |
| DA-02 | KV-01 | `M06-L` / `M11` 中**至少一个**补充 meta-GGA（用于交叉验证，不是装饰） | 同上；且与 DA-01 共用同一 `MetaGga` 组件（不复制数学形式） |
| DA-03 | KV-01 | range-separated 数据：`ωB97X-D`、`CAM-B3LYP`（`omega_bohr_inv`、混合系数、色散参数） | schema 通过；ω 与原始论文一致（列出处）；`data verify` 通过 |
| DA-04 | KV-02 | 溶剂数据 v2：COSMO/SMD 所需的 ε、n²、Bondi/UFF 半径、SASA 网格参数，先做 8 个溶剂（水、甲醇、乙醇、乙腈、DMSO、氯仿、甲苯、四氢呋喃） | ε 与 n 与 NIST/原始文献一致（水 78.3553 @ 298.15 K）；`radii_set` 显式；缺 license 即失败（G2） |
| DA-05 | KV-02 | SMD 的 4 项非极性参数（cavitation/dispersion/repulsion 的原子参数表） | 与原始论文表逐项 diff == 0；参数表带 DOI；进 `DATA_LICENSES.md` |
| DA-06 | KV-01 | 色散校正数据：D3(BJ) 元素对参数 + D4 的电荷/配位依赖参数（先 H–Kr） | 与官方参数表逐项 diff < 1e-12；对每个元素对都有测试；解析梯度有单测（`FX-04`） |
| DA-07 | KV-01 | 辅助基（RI）：`def2-SVP-RI`/`def2-TZVP-RI`/`def2-universal-JKFIT`（经 `sources.json` 按需取回） | `qwb data install --check` 全 ok；与 BSE 逐项 diff < 1e-12；与主基组配对测试通过 |
| DA-08 | KV-01 | 参考集扩充：激发能（H2O/苯/甲醛）、NMR 屏蔽/化学位移（H2O/CH4/CH3OH）、溶剂化自由能（8 溶剂 × 3 分子）、势垒高度（3 个反应）——全部进 `tests/reference/*.csv` | 每行有 `source_doi`/`source_url`；脚本可重建（`scripts/gen_*.py --check` 绿）；缺来源即门禁失败 |

### 6.3 积分 / 数值路径轨（IN）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| IN-01 | KV-04 | 3 中心与 4 中心 **RI 积分**（Obara-Saika，密度拟合），含辅助基加载 | `(ij|P)` 与直接 4 中心对比：拟合误差对 H2O/def2-SVP < 1e-6 Ha（能量）；单测：小体系 RI 恒等式（Σ_P (ij|P)(P|kl) 与 (ij|kl) 的差 = 拟合残差）；积分张量对称性 < 1e-12 |
| IN-02 | KV-04 | **erfc 分域**积分（短程/长程库仑），供 RS-hybrid 用 | ω→0 极限退回全库仑（差 < 1e-10）；ω→∞ 极限短程项 → 0；与数值积分对照 < 1e-9；`(ss\|ss)` 的解析闭式在有限 ω 下可对（写进测试） |
| IN-03 | KV-04, `stretch` | **pseudospectral** Coulomb/exchange 路径（网格 + 去别名 + K10 报告） | 与解析路径总能量差 < 5e-5 Ha（H2O/def2-SVP）；accuracy level 提高时误差单调下降（3 点）；1 vs N 线程 byte-identical；**失败按 §2.5 记账，不阻塞里程碑** |
| IN-04 | IN-01 | RI-J/RI-JK 的 Fock 构建路径（K10 门禁） | 与解析 4 中心 Fock 差 < 1e-6 Ha（H2O/def2-SVP）；大体系的墙钟优势有实测数字（≥ 2×，否则记录"无收益"并降级） |
| IN-05 | 基线 | 线程并行扩展到 2e/Fock 构建（确定性归约），继承 G7 | 1 vs 4 线程 byte-identical；H2O/def2-TZVP 的 Fock 构建 ≥ 2.5× 加速（否则记录实测并降级） |

### 6.4 泛函轨（FX）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| FX-01 | KV-01 | meta-GGA 基础设施：`FunctionalTerm` 扩展（τ、∇²ρ）、网格求值管线、`Requires` 门禁 | 有限差分导数（对 ρ、σ、τ）< 1e-9；缺 τ 的数据被 `Engine` 明确拒绝（`Unsupported`，不是 panic） |
| FX-02 | FX-01, DA-01 | `M06-2X` 实现 + 在 DA-08 参考集上验收 | 与文献给定原子化能/反应能差 < 2 kcal/mol（或与 PySCF 黑盒同值 < 0.5 mHa）；`t06_reference` 的 7 体系不回归 |
| FX-03 | KV-01, IN-02, DA-03 | `ωB97X-D`、`CAM-B3LYP`（短程交换混合 + ω 来自数据） | ω→0 时与 `B3LYP` 的差 < 1e-6 Ha（实现正确的强约束）；激发能用例进 M4；`data verify` 通过 |
| FX-04 | KV-01, DA-06 | 色散校正（D3(BJ)）作为可加项，含**解析梯度** | D3 梯度 vs 有限差分 < 1e-6 Ha/Bohr；与官方参考实现（黑盒脚本）在 10 个二聚体上差 < 1e-3 kcal/mol；`dispersion_energy_hartree` 进 `result.json` |

### 6.5 SCF / 相关轨（SC）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| SC-01 | KV-03 | **UHF**（开壳层 HF）：双套密度、`F_σ = H + J(P) − K(P_σ)`、`⟨S²⟩` 与 `spin_contamination`、**自旋平均的原子密度初猜**（SAD）；引擎接线 `scf_type: uhf` + `method_name: hf` + `sp`，结果带 K9 的 `spin` 块 | ① 闭壳层退化：H2/STO-3G 的 UHF == RHF（< 1e-9 Ha）、`⟨S²⟩ < 1e-10`、α/β 密度逐元素相同；② O2 三重态（STO-3G）能量与 `uhf_open_shell.csv` 的 PySCF 值差 < 1e-6，`n_α=9`/`n_β=7`；③ 自旋污染**必须报出来**：O2 `⟨S²⟩ = 2.0034`、OH `0.7533`（与 PySCF 差 < 1e-6），且断言 `⟨S²⟩ > 2.0`（UHF 的本性）；④ OH 双峰同值；⑤ `Tr[P_σ S] = n_σ`；⑥ 两次运行逐位相同；⑦ 引擎侧 `spin` 块落盘、`uhf`+DFT 与 `uhf`+`opt` 显式拒绝 |
| SC-01b | SC-01 | **UKS**（自旋分辨 XC）：`SpinDensityBatch`、交换的 `½[E_x(2ρ_α)+E_x(2ρ_β)]`、VWN5 的自旋插值与 GM/PBE-C/B88/LYP 的自旋形式、UKS 的 SCF 循环（每通道一套 Fock） | XC 侧：`ε_c(ρ/2,ρ/2)` 退回闭壳层（< 1e-12）、`∂ε/∂ρ_σ` 与中心差分 < 1e-9、ζ=1 等于铁磁参数集、缺参数时 `SpinUnsupported`。SCF 侧：O2/OH 的 UKS 与 PySCF 同值 < 1e-6、闭壳层体系 `uks` ≡ `rks` < 1e-8 |
| SC-01c | SC-01 | **ROHF**（纯自旋，`⟨S²⟩ = S(S+1)` 精确）—— **blocked**：Guest–Saunders 等效 Fock 的第一版实现收敛到错误的驻点（O2 高 2.05 Ha、OH 高 1.12 Ha；闭壳层情形 O2→H2 精确 ≡ RHF，说明闭壳层机制没问题、错在 C-O/O-V 混合块） | 解锁条件：正确的有效 Fock 推导，或改用"轨道旋转直接优化"（用真实能量梯度做线搜索）；验收仍是 O2 `⟨S²⟩ = 2.0` 且能量与 PySCF 的 ROHF 差 < 1e-6 |
| SC-02 | SC-01 | 开壳层解析梯度（UHF/UKS）与优化接线 | 与数值梯度 < 1e-6 Ha/Bohr（OH、O2）；`opt` 在开壳层跑通（符号与自旋不变） |
| SC-03 | SC-01 | 开壳层的 SCF 稳定性（level shift、DIIS 变体、破对称初猜） | 5 个开壳层体系全部收敛到 1e-8（列出墙钟）；不收敛时返回 `ScfNotConverged` |
| SC-04 | KV-03, IN-01 | canonical MP2（半正则化，RHF/RKS 参考）+ 相关能 | H2O/cc-pVDZ 的 MP2 相关能与文献差 < 0.1 mHa；基组单调性（DZ→TZ）方向正确；`correlation` 块进 `result.json` |
| SC-05 | SC-04, IN-01, DA-07 | RI-MP2（K10 报告） | 与 SC-04 差 < 0.1 mHa（H2O/def2-SVP）；墙钟优势 ≥ 3×（否则记录并降级）；`ri_error_hartree` 进报告 |
| SC-06 | SC-01, KV-03 | 开壳层 + 泛函交叉（UKS+meta-GGA、UKS+D3）的最小可用组合 | 至少 1 个开壳层体系在 M06-2X/D3 下跑通并给出梯度；`--threads` 归约仍确定性 |

### 6.6 响应与性质轨（RS）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| RS-01 | KV-01, SC-01 | `qwb-response` crate：**CPHF/UCPHF** 求解器（迭代，2e 驱动） | 静态极化率 vs 有限场（±0.001 a.u.）< 1e-4 a.u.（H2O/STO-3G 与 def2-SVP）；CPHF 迭代收敛到 1e-8；1 vs N 线程 identical |
| RS-02 | RS-01, DA-08 | **TDA/TDDFT** 激发能 + 振子强度（`ExcitedStates`） | H2O 前 3 个单重态跃迁能与文献/PySCF 差 < 0.2 eV；苯的 ¹B₂ᵤ/¹B₁ᵤ 差 < 0.3 eV；`f-sum rule` 差 < 1e-3；振子强度 > 0 且排序合理 |
| RS-03 | RS-02 | **CIS**（HF 参考的 TDA） | H2O/甲醛的跃迁能与文献差 < 0.5 eV；与 RS-02 的 HF 极限一致（用 `hf` 方法时两者差 < 1e-6 a.u.） |
| RS-04 | RS-01 | **NMR GIAO 屏蔽**（需 GIAO 积分 + CPHF 右端） | H2O 的 ¹H 屏蔽 vs 实验化学位移（换算后）差 < 0.5 ppm；CH4/CH3OH 的 ¹H 与 ¹³C 各自 < 1.0/5.0 ppm；`length_gauge_spread_ppm < 1` |
| RS-05 | RS-01 | **解析 Hessian**（CPHF 二阶导数；闭壳层 RHF/RKS） | H2O/def2-SVP 的解析 Hessian 与数值（现有实现）Frobenius 差 < 5%；频率差 < 5 cm⁻¹；墙钟比数值快 ≥ 20×（实测数字进 status） |
| RS-06 | RS-05, RS-01, `stretch` | **Raman 活性**（极化率导数 → 强度） | 与文献 Raman 峰**相对强度**排序一致（至少 H2O/CH4）；绝对强度的标度因子单列；失败按 §2.5 记账 |
| RS-07 | RS-05, RS-06, `stretch` | **VCD**（磁偶极 + 极化率导数 → 旋转强度） | CH3OH 的 VCD 符号图样与实验一致（至少 3 个主峰）；失败按 §2.5 记账 |
| RS-08 | 基线 | **多极展开**（四极 traceless、八极可选） | 原点多选一致性（原点平移后 traceless 四极不变 < 1e-6）；H2O 的四极矩与文献差 < 5% |

### 6.7 溶剂化轨（SV）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| SV-01 | KV-02, DA-04 | 空腔构建（SASA tesserae）+ COSMO 的介电屏蔽（迭代求解表面电荷） | 表面面积/体积对转动不变（< 1e-6）；水/甲醇的 ΔG_solv 与实验差 < 1.5 kcal/mol；`converged` 标志可靠 |
| SV-02 | SV-01 | COSMO 的**解析梯度** + `opt` 接线 | 溶剂化梯度 vs 有限差分 < 1e-5 Ha/Bohr；水中甲醇的优化几何与气相差合理（O–H 伸长约 0.01 Å 量级）；`opt` 跑通 |
| SV-03 | SV-01, DA-05 | **SMD**（COSMO + 4 项非极性项 + 参数化半径） | 8 溶剂 × 3 分子的 ΔG_solv 与实验 MAE < 1.0 kcal/mol（SMD 的公开精度目标）；`solvation` 块各项分解可查 |
| SV-04 | SV-01, `stretch` | **PBF**（Poisson–Boltzmann，边界元/有限元求解） | 与 COSMO 在 ε→∞ 极限的一致性 < 0.5 kcal/mol；含离子的情形（0.1 M NaCl）有解且量级合理；失败按 §2.5 记账 |
| SV-05 | SV-02, RS-05 | 溶剂中的频率/热化学（气相 → 溶液相的热化学修正） | H2O 在 COSMO 下的 ZPE 与气相差 < 0.5 kcal/mol；熵项符号正确 |

### 6.8 反应路径轨（RP）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| RP-01 | SC-02, RS-05 | **过渡态搜索**（给初猜 + Hessian 引导的局部 Newton–Raphson / QST2 类） | 3 个反应（HCN↔HNC、CH₃NC↔CH₃CN、甲醛→H₂+CO）的势垒与文献差 < 1.5 kcal/mol（DFT 水平）；`max|grad| < 4.5e-4`；Hessian 恰有**一个**虚频（且图形正确） |
| RP-02 | RP-01 | **IRC**（质量加权内禀反应坐标，双向） | 两端优化后几何与已知极小差 < 0.02 Å；能量单调下降（允许 ≤ 1 个平台步）；`irc.json` 落盘 |
| RP-03 | RP-01 | **扫描**：刚性坐标扫描 + 弛豫扫描（每步可选约束） | 扫描点能量连续（相邻差有界）；`pes.json` 可被 GUI 读取；`jobtype:scan` 进 `SUPPORTED_FEATURES` |
| RP-04 | RP-01, SC-02 | 反应路径的**梯度正确性套件**：TS/IRC 全程的梯度一致性 | 每个 RP-01/02/03 的测试里都有"解析 vs 数值梯度 < 1e-5"断言（不允许只有能量对） |

### 6.9 产品面轨（PR）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| PR-01 | KV-05 | CLI：`qwb props`（激发态/NMR/极化率/多极）、`qwb ts`、`qwb irc`、`qwb scan` | `--help` 全通；退出码表未变；`qwb inspect` 能显示新结果块（旧结果目录仍 ok） |
| PR-02 | PR-01 | Python API（pyo3）暴露新的 `JobInput` 选项与结果块 | `mypy --strict` 通过；与 CLI 能量逐位相同；GIL 仍释放（并发测试） |
| PR-03 | PR-01, KV-05 | `.qwb` 容器与 K3 结果目录**向后兼容**新块（缺块时旧读法不炸） | 旧夹具 `example.qwb` 仍可读；新结果写出再读回 roundtrip；`gen_p02_fixture.py --check` 绿 |
| PR-04 | RS-02, RS-03 | GUI/CLI 的 **UV/Vis 谱**（振子强度 + 高斯展宽 → `spectrum.csv` + 图） | 展宽参数来自数据；H2O/苯的谱峰位置与 RS-02 一致；GUI 出图不阻塞 |
| PR-05 | RS-08, SV-03 | GUI 的 **ESP/溶剂化表面**着色（多极 + 表面电荷） | 与 Mulliken/ESP 电荷的趋势一致（同分子不同原子排序正确）；3D 面板渲染 < 1 s |
| PR-06 | RP-03 | GUI 的 **PES 扫描可视化**（能量曲线 + 结构联动） | 打开 `pes.json` 出图；点选点显示对应几何；100 点扫描不卡 |
| PR-07 | PR-01, VF-01 | **结果对比表**（多任务：能量/ΔE/谱峰；导出 CSV） | 两任务对比值与单看一致（1e-10）；导出 CSV 可被 `pandas` 读（仅测试用 pandas，不进产品依赖） |
| PR-08 | 全部 PR | 使用文档更新（`docs/USAGE.md` 新章节 + 诚实清单同步） | 文档里每个 ✅ 都能在 `status-v2.yaml` 找到对应 WP；`⚠/❌` 与代码状态一致（人工核对 + 检查脚本） |

### 6.10 验证轨（VF）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| VF-01 | RS-02 | 激发态基准集（H2O/苯/甲醛/丙酮；外部来源） | CSV 带来源；与库的差在 §6.6 的容差内；CI 跑 |
| VF-02 | SV-03 | 溶剂化基准集（8 溶剂 × 3 分子的 ΔG_solv） | 同上；MAE 报进 status |
| VF-03 | RP-01 | 势垒高度基准集（3 个反应 × 2 泛函） | 同上；虚频数 + 图形断言 |
| VF-04 | RS-04 | NMR 基准集（屏蔽/位移 + 参考物） | 同上；gauge 一致性断言 |
| VF-05 | 全部 | `docs/error-budget.md` v2：RI/PS/网格/TDA/溶剂/SASA 各自的误差量级实测 | 每个误差源一行实测数字 + 来源；与 §9 容差表一致 |
| VF-06 | KV-04, IN-03 | **K10 等价性套件**：所有近似路径 × 所有参考体系的自动比对 | 一个脚本跑完（`scripts/check_numerics_equivalence.py`），输出差异表；任一超差即门禁红 |

### 6.11 Stretch 备选池（ST）— 允许失败，不影响里程碑

这五条**不进入任何里程碑的依赖**。做成了是加分，做不成按 §2.5 记 `blocked` + 实测数字。
它们的共同点：要么需要一套新的基础设施（LMP2 的局域化、VCD 的磁导数），要么依赖
一条长链条（pKa 的"溶剂化 + 热化学 + 经验校正"）。

| ID | 依赖 | 产物 | 验收（达成时的判据） |
|---|---|---|---|
| ST-01 | RS-02 | **完整 TDDFT**（含 TDA 之外的耦合项 B 矩阵） | 与 TDA 相比激发能系统性下降；H2O 前 3 态与文献差 < 0.15 eV；`method: tddft` 与 `tda` 的报告分列 |
| ST-02 | RS-01 | **动态极化率与一阶超极化率 β**（多频率） | 与文献（静态/动态）差 < 10%；转动不变量协变 < 1e-6 |
| ST-03 | SC-04 | **LMP2**（局域 MP2，需局域化轨道与配对域） | 与小体系 canonical MP2 差 < 1 mHa；大体系墙钟线性化（有实测）；失败即 `blocked` |
| ST-04 | SV-03, RS-05 | **pKa 工作流**（去质子化自由能 + 参考酸的经验校正） | 5 个羧酸/酚的 pKa MAE < 1.0（用公开校正参数）；链条任一环失败即 `blocked` |
| ST-05 | SC-01 | **EPR / 超精细耦合**（UKS 的自旋密度 + 各向同性 A 张量） | CH₃ 自由基的 ¹H 耦合常数与实验差 < 20%；`epr` 块进结果 |

---

## 7. 垂直切片里程碑 M1–M8

> 每个 M 是"项目真正向前一步"的刻度：只有全部前置 WP 通过、且该 M 的端到端命令通过才算完成。
> `⚠` 项失败不阻塞 M 的完成，但必须在 `docs/status-v2.yaml` 里以 `blocked` + 实测数字登记。
>
> 本节出现的 `examples/*.yaml` **不在仓库里**（阶段一的六个示例之外）——它们是各 M 前置 WP
> 的产物之一：新作业类型必须带示例输入 + 参考 CSV，否则"能做"这句话没有可执行依据。

### M1 · 开壳层第一步：UHF（自由基能算了）

**前置 WP**：KV-03、SC-01

**验收命令**：

```powershell
qwb run examples/o2_uhf_sto3g.yaml          # O2 三重态（UHF）
```

**断言**（**修正版**；原版写"O2 的 ⟨S²⟩ 与 2.0 差 < 1e-6"，对 UHF/UKS 物理上不可能 ——
见 `docs/choices.md` 四十七）：

- 闭壳层退化：H2/STO-3G 的 `scf_type: uhf` 与 `rhf` 能量差 < 1e-9，`⟨S²⟩ < 1e-10`；
- O2 三重态（STO-3G）：`E = −147.6339696 Ha`（PySCF 同值，差 < 1e-6），`n_α=9`/`n_β=7`；
- **污染必须报出来**：O2 `⟨S²⟩ = 2.0034`、OH `0.7533`，与 PySCF 差 < 1e-6，
  且 `spin_contamination` 落进 `result.json` 的 `spin` 块（不是只写日志）；
- OH 双峰（STO-3G）同值；
- `scripts/verify.ps1` 全绿；旧 T1–T8 无回归。

### M1b · 开壳层第二步：UKS（自由基也能用 DFT 算了）

**前置 WP**：M1 + SC-01b

**断言**：

- O2/OH 的 **UKS**（SVWN5、PBE）与 PySCF 同值（< 1e-6），`spin` 块落盘；
- 闭壳层体系 `uks` ≡ `rks`（< 1e-8）—— 退化一致性；
- 交换项的自旋标度与相关项的自旋插值各有独立断言（见 SC-01b 的 XC 侧判据）。

### M1c · 开壳层第三步：ROHF（纯自旋）—— **blocked**

**前置 WP**：M1 + SC-01c

ROHF 的验收是"O2 的 `⟨S²⟩ = 2.0` 精确、能量与 PySCF 的 ROHF 同值"，目前**未达成**：
第一版 Guest–Saunders 等效 Fock 在开壳层情形收敛到错误的驻点（差 2.05 Ha / 1.12 Ha）。
阻塞记录见 `docs/status-v2.yaml` 的 SC-01c 与 `docs/choices.md` 四十九。

### M2 · meta-GGA 与色散

**前置 WP**：KV-01、DA-01、DA-02、DA-06、FX-01、FX-02、FX-04

**验收命令**：

```powershell
qwb run examples/h2o_m062x_def2svp.yaml
qwb run examples/benzene_dimer_d3.yaml      # 色散校正 + 相互作用能
```

**断言**：

- M06-2X 的有限差分导数（ρ/σ/τ）< 1e-9；
- 与 PySCF 黑盒同值 < 0.5 mHa（H2O/def2-SVP），或与文献原子化能差 < 2 kcal/mol；
- D3(BJ) 的解析梯度 vs 有限差分 < 1e-6 Ha/Bohr；
- 苯二聚体的相互作用能（M06-2X-D3/def2-SVP）与文献差 < 1 kcal/mol（BSSE 未校正，记在 note）；
- ω→0 时 RS 泛函退回 B3LYP（`FX-03` 若已 done，作为附加断言）。

### M3 · RI-MP2：第一次有相关能

**前置 WP**：KV-03、DA-07、IN-01、SC-04、SC-05

**验收命令**：

```powershell
qwb run examples/h2o_mp2_ccpvdz.yaml        # canonical MP2
qwb run examples/h2o_ri_mp2_def2svp.yaml    # RI-MP2
```

**断言**：

- H2O/cc-pVDZ 的 MP2 总能量与文献差 < 0.5 mHa；
- RI-MP2 与 canonical MP2 差 < 0.1 mHa（同基组）；
- 基组单调性：def2-SVP → def2-TZVP 的 E2 方向正确（更负）；
- `result.json` 的 `correlation` 块含 `method`/`aux_basis`/`ri_error_hartree`；
- 墙钟优势 ≥ 3×（否则在 status 里记实测并降级判据）。

### M4 · 激发态：CIS 与 TDA/TDDFT

**前置 WP**：KV-01、RS-01、RS-02、RS-03、VF-01

**验收命令**：

```powershell
qwb props examples/h2o_tddft.yaml
qwb props examples/benzene_tddft.yaml
```

**断言**：

- H2O 前 3 个单重态跃迁能与文献/PySCF 差 < 0.2 eV；
- 苯的 ¹B₂ᵤ / ¹B₁ᵤ 差 < 0.3 eV（TDA 的系统性偏差记在 note）；
- `sum_rules.trk_oscillator` 与电子数差 < 1e-3；
- CIS 与 TDA 在 `hf` 参考下能量一致（< 1e-6 a.u.）；
- 振子强度全部 ≥ 0，最强峰排序与文献一致。

### M5 · 隐式溶剂：COSMO → SMD

**前置 WP**：KV-02、DA-04、DA-05、SV-01..SV-03、VF-02

**验收命令**：

```powershell
qwb run examples/h2o_cosmo_pbe.yaml --out results
qwb run examples/meoh_smd_opt.yaml
```

**断言**：

- 8 溶剂 × 3 分子的 ΔG_solv 与实验 MAE < 1.0 kcal/mol（SMD）/ < 1.5 kcal/mol（COSMO）；
- 表面面积/体积对转动不变（< 1e-6）；
- 水中甲醇的 `opt` 收敛，键长变化与文献量级一致；
- `solvation` 块的能量分解求和 == `total_hartree`（< 1e-10）；
- `solvent: vacuum` 的结果与阶段一**逐位相同**（回归）。

### M6 · 反应路径：TS 与 IRC

**前置 WP**：KV-05、SC-02、RS-05、RP-01..RP-04、VF-03

**验收命令**：

```powershell
qwb ts examples/hcn_ts.yaml
qwb irc examples/hcn_ts.yaml --out results
qwb scan examples/hcn_scan.yaml
```

**断言**：

- 3 个反应的势垒与文献差 < 1.5 kcal/mol（DFT 水平）；
- TS 的 `max|grad| < 4.5e-4` Ha/Bohr，Hessian **恰好一个**虚频，且虚频对应的位移方向正确（打印）；
- IRC 两端优化后与已知极小差 < 0.02 Å，能量单调下降（≤ 1 个平台步）；
- 每个 RP 测试都含"解析 vs 数值梯度 < 1e-5"断言；
- `ts_search.json` / `irc.json` / `pes.json` 落盘且能被 GUI 读。

### M7 · 谱学套件：解析 Hessian + NMR + 极化率

**前置 WP**：RS-01、RS-04、RS-05、RS-08、VF-04

**验收命令**：

```powershell
qwb props examples/h2o_hessian_nmr.yaml
qwb run examples/h2o_freq_analytic.yaml
```

**断言**：

- 解析 Hessian 与数值 Hessian Frobenius 差 < 5%，频率差 < 5 cm⁻¹，墙钟 ≥ 20× 加速（实测进 status）；
- H2O ¹H 屏蔽（换算位移）差 < 0.5 ppm；CH4/CH3OH ¹H < 1.0 ppm、¹³C < 5.0 ppm；
- `length_gauge_spread_ppm < 1`；
- 静态极化率 vs 有限场 < 1e-4 a.u.，转动不变量协变（< 1e-6）；
- 四极矩原点平移不变性（< 1e-6）。

### M8 · 产品面收口：新能力真的能用

**前置 WP**：PR-01..PR-08、VF-05、VF-06

**验收命令**：

```powershell
python -m qwb_studio.gui --out studio-out --workers 2   # 建分子 → 选 M06-2X/TDDFT/溶剂 → 提交 → 谱图
qwb batch jobs\*.yaml --parallel 4
qwb inspect results\<task>
```

**断言**：

- CLI/Python/GUI 三条路对同一输入给出**逐位相同**的能量与谱峰（1e-10）；
- 旧结果目录（阶段一产物）在新版本下 `inspect`/`open_result_dir` 仍 ok（向后兼容回归）；
- GUI：TDDFT 谱图、PES 曲线、溶剂化表面三处均可交互且不阻塞（QTimer 计数断言）；
- 结果对比表导出 CSV，两任务差值与单看一致（1e-10）；
- `docs/USAGE.md` 的诚实清单与 `status-v2.yaml` 一致（每条 ✅ 有 WP 编号）。

### M-PS（可选里程碑）· pseudospectral 路径

**前置 WP**：KV-04、IN-03、VF-06

不阻塞 M1–M8。达标条件：K10 三条件全满足 + 大体系墙钟优势 ≥ 3×。
达不到就 `blocked`，并在 `docs/choices.md` 记"为什么这条路径在我们的验收体系下不划算"。

---

## 8. 数据层新增要求

数据纪律沿用 `project.md` §8.6（citation/license/sha256 三处一致、按需取回、`DATA_LICENSES.md` 自动生成）。
阶段二新增四类数据：

### 8.1 泛函参数（meta-GGA / RS / 色散）

- **一处分数学形式，一处放参数**：`LdaSvwn5`、`GgaPbeX`… 的约定不变；新组件
  `MetaGgaM06x`、`MetaGgaM06c`、`GgaB97X`、`HfExchangeShortRange` 等**只加不复制**。
- **`requires` 是硬门禁**：数据声明 `laplacian: true` 而网格/管线不提供 ∇²ρ 时，
  `Engine` 必须返回 `Unsupported`（不是 panic、不是静默降级）。
- **色散**是独立数据类型 `dispersion`（`kind: d3bj|d4`、元素对参数、`s6/s8/a1/a2`），
  与泛函解耦（同一个 D3 参数通用于任何泛函）。
- 每条参数表逐项 diff 官方表 < 1e-12，并**把参数条数写进测试断言**（防"少抄一行"）。

### 8.2 溶剂数据 v2

| 字段组 | 内容 | 陷阱 |
|---|---|---|
| 介电/光学 | `epsilon`、`refractive_index`（SMD 用 n²） | 温度必须写明；水 78.3553 @ 298.15 K |
| 半径 | `radii_set: bondi\|uff\|smd` + 每元素半径 | **不能混用**不同半径集；SMD 用自己的表 |
| 表面 | tesserae 数、面积/体积算法、`surface_tension` | 面积对转动不变是验收项 |
| 非极性项 | SMD 的 cavitation/dispersion/repulsion 原子参数 | 4 项分解必须能逐项对账 |
| PBF | 离子强度、离子半径、介电剖面 | `stretch` 项，先只留字段 |

### 8.3 辅助基（RI）

- 走 `sources.json` 按需取回（BSE 导出，同阶段一），`role: ri-aux` + `fitting.for`。
- 必须与主基组**配对校验**（`Registry::load_pair` 的姊妹函数 `load_aux_pair`）：
  辅助基覆盖的元素 ⊇ 主基组覆盖的元素，否则 `Data` 错误。

### 8.4 参考集（`tests/reference/*.csv`）

沿用 `project.md` §7.0 的政策，新增四张表：

| 文件 | 内容 | 来源要求 |
|---|---|---|
| `excited_states.csv` | 体系/基组/方法/态/能量(eV)/振子强度 | 带 `source_doi`/`source_url`（CCCBDB、原始文献） |
| `nmr_shifts.csv` | 体系/核/屏蔽(ppm)/化学位移(ppm)/参考物 | 实验值与理论值**分列**，不得混用 |
| `solvation_free_energy.csv` | 溶剂/溶质/ΔG_solv(kcal/mol)/温度 | 实验值，注明实验方法（汽液平衡/溶解度） |
| `barrier_heights.csv` | 反应/方法/基组/势垒(kcal/mol)/虚频数 | 文献理论值或实验活化能，注明是哪一个 |

**规则**：拿不到外部值的，允许用"内部独立路径"（如 PS vs 解析、RI vs canonical），
但必须在 `note` 写明"内部独立路径，非文献值"，并进 `docs/error-budget.md`。

### 8.5 `numerics/numerics.yaml`（K10 的配置数据）

> **实施时的修正**：本节原本写的是数据根下的 `numerics.yaml`，但 K1 的每个数据域都在
> 自己的子目录里（`basis/`、`grid/`、…），而且 `qwb-data/DATA_LICENSES.md` 的生成器
> 按子目录扫生产数据 —— 放在根目录会让它**静默漏掉**这份数据的许可。所以实际落在
> `qwb-data/numerics/numerics.yaml`，并在 `registry.yaml` 的 `numerics:` 段登记。

```yaml
schema_version: 1.0.0
levels:                       # accuracy_level 1..5 的语义（1 最快）
  - level: 1
    grid: {radial: 50, angular: 194, cutoff_bohr: 20}
    ps: {grid: 40, dealiasing: false}
  - level: 3
    grid: {radial: 75, angular: 302, cutoff_bohr: 20}
    ps: {grid: 60, dealiasing: true}
equiv_tol_hartree:            # K10 门禁的容差（按路径分）
  ri: 1.0e-6
  pseudospectral: 5.0e-5
```

- 等级语义与 `grid/*.yaml` **不重复**：`levels` 只是把"粒度"与"精度等级"绑定，
  网格参数仍然只有 `grid/*.yaml` 一份。
- 新增等级必须同时给"更高一级误差更小"的回归测试（K10 第 2 条）。

---

## 9. 验证体系

阶段一的三层（单元 / 物理极限 / 基准集）保留，阶段二再加**第四层：数值路径等价性**
（K10）与**第五层：物理不变量**。

### 9.1 五层验证

| 层 | 内容 | 例子 |
|---|---|---|
| 单元 | 每个 WP 自带 | RI 恒等式、erfc 极限、CPHF 收敛、D3 梯度 |
| 物理极限 | 有严格解的极限 | ω→0 退回全库仑；ε→∞ 的 COSMO；`uks` 在闭壳层退化为 `rks`；线性分子 3N-5 |
| **数值路径等价** | 近似 vs 解析 | RI-MP2 vs canonical MP2；PS vs 解析；RI-J vs 4 中心（K10） |
| **物理不变量** | 与实现无关的恒等式 | f-sum rule、gauge invariance、转动/平动不变性、`⟨S²⟩`、Hessian 对称性、变分上界 |
| 基准集 | 外部参照 | 激发能、ΔG_solv、势垒、NMR 位移 |

### 9.2 门禁 G（阶段二增补）

- **G8 等价性门禁**：任何新近似路径必须过 K10 三条件，否则不得进 `SUPPORTED_FEATURES`。
- **G9 不变量门禁**：每个新性质必须有至少一条物理不变量的自动测试（`f-sum`/gauge/
  转动协变/原点平移）。缺了视为该 WP 未完成。
- **G10 留痕门禁**：`result.json` 里必须能看到"用了什么近似"（`numerics_report`、
  `solvation.model`、`correlation.method`、`dispersion.kind`）。看不到 = 不可复现 = 不合格。
- 阶段一的 G1–G7 **原样保留**。

### 9.3 容差表（阶段二）

| 量 | 容差 | 依据 |
|---|---|---|
| 总能量（同方法同基组，新旧实现） | < 1e-8 Ha | 回归 |
| RI-MP2 vs canonical MP2 | < 1e-4 Ha | 文献量级 |
| PS vs 解析总能量 | < 5e-5 Ha | Jaguar 量级，记录实测 |
| 激发能（vs 文献/PySCF） | < 0.2–0.3 eV | TDA 的系统性偏差 |
| NMR ¹H / ¹³C 位移 | < 0.5 / 5.0 ppm | GIAO 常态 |
| ΔG_solv（SMD / COSMO） | MAE < 1.0 / 1.5 kcal/mol | SMD 公开精度目标 |
| 势垒高度 | < 1.5 kcal/mol | DFT 常态 |
| 解析 Hessian vs 数值 | < 5%（Frobenius），频率 < 5 cm⁻¹ | 阶段一数值路径作尺子 |
| 梯度（解析 vs 数值） | < 1e-5 Ha/Bohr | 比阶段一（1e-6）略松：新性质含更多近似项，**必须写明理由** |

---

## 10. Agent 执行协议

### 10.1 开工前（每个 WP）

1. 读 `docs/status-v2.yaml`，确认依赖全 `done`；读 `docs/status.yaml` 确认阶段一基线未退化。
2. 读对应契约（K7–K10 在 `qwb-contracts/`，加性扩展示意在 §4.1）。
3. 检查产物是否已存在且验收通过 —— 通过就标 `done`，不重复做。
4. 跑一遍门禁确认基线绿（红就先修基线，这是唯一允许扩大改动范围的情形）。
5. 若该 WP 标了 `stretch`：先写下**你打算怎么判定失败**（kill criteria 对应哪条），
   再动手。写不出来说明这条 WP 的定义不清楚，先补 §6。

### 10.2 执行中

1. 只改本 WP 涉及的代码/数据/契约片段。
2. 不得改 K1–K6 语义；K7–K10 一旦落地即冻结（只允许加性扩展）。
3. 不得删除或修改其它 WP 的产物；跨轨改动在 PR body 里列出。
4. 每次提交附：`cargo fmt --check`、`cargo clippy -- -D warnings`、该 WP 验收命令输出摘要。
5. **`stretch` 项的中途产出**（例如只有能量对、没有梯度）不得写进 `USAGE.md` 的能力表，
   但可以在 `status-note` 里写"已到哪一步"。
6. 新增依赖走 §2.4 的批准流程；默认自己实现。

### 10.3 卡住时怎么办

沿用 `project.md` §10.3 的分级表，并补充：

| 情形 | 动作 |
|---|---|
| 验收容差定得太紧（实测差 1.2×判据） | 先查实现；两轮后仍差的，**记录实测并把判据降级**（`⚠`），在 `note` 写清差距与新判据依据 |
| 需要外部文献但拿不到 | 走 `project.md` §7.0 的内部独立路径 fallback；再不行 → §2.5 kill criteria 3 |
| 某近似路径在合法参数下**发散**（PS 去别名失败、CPHF 不收敛） | 先试更保守参数（更高 level）；仍失败记 `blocked` + 实测，不做"调到能过"的参数微调（那是拟合验收，不是实现） |
| 两个 WP 的容差互相矛盾 | 停，记 `docs/choices.md`，转做其它可并行 WP |
| 里程碑被一条 `stretch` 卡住 | 把该 `stretch` 标 `blocked` 后**继续下一个里程碑**（阶段二允许里程碑内出现 blocked 的 stretch 项） |

### 10.4 契约升级 / 依赖批准

沿用 `project.md` §10.4。阶段二额外要求：K7–K10 的任何变更都要在
`qwb-contracts/CHANGELOG.md` 里写**为什么现在必须改**，并列出受影响 WP（用 ID）。

### 10.5 推荐工作顺序

```
（阶段一基线：门禁 26/26 绿，T1–T8 done）
 1. KV-01..KV-05                              ← 契约先立
 2. DA-01, DA-06, FX-01, FX-04                ← meta-GGA 基础设施 + 色散（最快见效）
 3. SC-01 → M1（UHF）；SC-01b → M1b（UKS）；SC-01c（ROHF）blocked 待重做 ← 开壳层
 4. FX-02 → M2（meta-GGA+D3）
 5. DA-07, IN-01, SC-04, SC-05 → M3            ← RI-MP2
 6. RS-01（CPHF）→ RS-02, RS-03 → M4            ← 激发态
 7. IN-02, DA-03, FX-03                        ← RS-hybrid（与 M4 并行）
 8. DA-04, DA-05, SV-01..SV-03 → M5            ← 溶剂化（COSMO→SMD）
 9. RS-05（解析 Hessian）── 是 M6/M7 的前置
10. SC-02, RS-05, RP-01..RP-04 → M6            ← 反应路径
11. RS-04, RS-08 → M7（谱学套件）
12. IN-03（PS，stretch）── 独立，随时可停
13. PR-01..PR-08 → M8                          ← 产品面收口
14. VF-05, VF-06 → 收尾（误差预算 + 等价性套件）
```

### 10.6 进度记账

每完成一个 WP：`docs/status-v2.yaml` 写
`{state: done, ts: <ISO8601 UTC>, artifact: <一句话>, verification: <命令/断言 + 实测数字>}`。
被卡：`{state: blocked, reason: ..., log: ..., gap: <差多少>}` **且** 在 `docs/choices.md` 登记。

`python scripts/check_status.py docs/status-v2.yaml` 必须绿（它检查键名格式、state 取值、
YAML 可解析、以及 `done` 的 WP 引用的文件确实存在）。

---

## 11. 风险与门禁对照

阶段一的 R-01..R-20 继续有效，阶段二新增：

| ID | 风险 | 缓解 | 门禁 |
|---|---|---|---|
| S-01 | 响应理论实现错但"看起来像对的"（CPHF 右端漏项） | 三重对照：有限场、sum-over-states、变分/振荡强度符号；不变量测试 | G9 |
| S-02 | TDDFT 用了完整耦合项但只验 TDA 级别的量 | 明确 `method: tda` 与 `tddft` 分开；报告里写清 | G10 |
| S-03 | meta-GGA 的 ∇²ρ 在低密度区数值爆炸 | 数据侧 `stability` 截断规则 + 与 PySCF 同值对照 | G3 |
| S-04 | RI 拟合误差被"看起来对"的能量掩盖 | `ri_error_hartree` 必须落盘；K10 套件对每个体系比对 | G8 |
| S-05 | PS 去别名参数被当成调参旋钮 | K10 第 2 条（单调收敛）+ 禁止"调到能过" | G8 |
| S-06 | COSMO/SMD 的表面电荷不收敛却报 converged | `converged` 独立判定（残差阈值）+ 与 ε→∞ 极限一致 | K8 门禁 |
| S-07 | 溶剂化参数的许可/出处不明（SMD 表来自论文） | 参数表逐项带 DOI；进 `DATA_LICENSES.md`；拿不到就 `stretch` 化 | G2 |
| S-08 | 过渡态"收敛到一个虚频"但图形不对 | 虚频位移方向必须打印 + 与 IRC 两端一致 | G9 |
| S-09 | 解析 Hessian 与数值 Hessian 互相对照变成"同错" | 再加一条独立对照：频率 vs 实验（H2O 的 3 个模式） | G9 |
| S-10 | 阶段二改动悄悄破坏阶段一（回归漏网） | 阶段一 25 项门禁 + T1–T8 断言**每次都要跑**；旧结果目录向后兼容测试 | G1/G5 |
| S-11 | 阶段膨胀导致长期不验收 | 里程碑先于精度：M1–M8 的端到端命令优先跑通；`stretch` 一律不阻塞 | §2.5 |
| S-12 | 新数据（辅助基/溶剂）又踩再分发条款 | 沿用 D-16：`sources.json` 配方 + 按需取回；`redistribute_ok` 逐条确认 | G2 |

---

## 12. 附录

### 12.1 目录结构（阶段二新增部分）

```text
qwb-contracts/
  schema/            # + excited_states / nmr_shieldings / polarizability /
                     #   multipoles / solvation / correlation / numerics_report
                     #   / dispersion / numerics 的 schema（由 Rust 类型生成）
  engine-api/        # K2 加性字段
  studio/            # K5 新子命令说明、K6 .pyi 更新

qwb-data/
  functional/        # + m06-2x.yaml / m06-l.yaml / wb97x-d.yaml / cam-b3lyp.yaml
  dispersion/        # 新：d3bj.yaml / d4.yaml
  solvent/           # + cosmo/smd 参数与半径集（8 溶剂）
  basis/             # + RI 辅助基（按需取回）
  numerics/          # 新：accuracy level 与等价容差（K10；registry.yaml 的 numerics 段）

qwb-core/crates/
  qwb-response/      # 新：CPHF/CPKS、TDDFT/CIS、GIAO、极化率、Raman/VCD
  qwb-solvation/     # 新：空腔、COSMO/SMD（+PBF stretch）
  qwb-reaction/      # 新：TS 搜索、IRC、扫描
  qwb-integrals/     # + RI 3 中心 / erfc 分域 / pseudospectral（feature gate：ri / pseudospectral）
  qwb-basis/         # + 辅助基加载、配对校验
  qwb-xc/            # + meta-GGA 组件、RS-hybrid、色散项
  qwb-scf/           # + UHF/ROHF/UKS、MP2/RI-MP2
  qwb-props/         # + 多极、谱（消费 qwb-response 的产物）
  qwb-engine/        # 新作业类型接线（ts/irc/scan/props）

qwb-studio/qwb_studio/
  gui/               # + 谱图面板、PES 面板、溶剂化表面着色、结果对比表
  models/            # + 新结果块的 pydantic 模型（与 K7–K9 一致）

scripts/
  check_numerics_equivalence.py   # K10 套件
  gen_reference_*.py              # 参考集生成（--check 幂等）
docs/
  status-v2.yaml  error-budget.md（v2）  choices.md（继续追加）
```

### 12.2 命名与单位

沿用 `project.md` §12.2（原子单位 + 显式后缀：`_hartree`、`_angstrom`、
`_hartree_per_bohr`、`_hartree_per_bohr2`、`_debye`、`_ppm`、`_ev`、`_kcal_per_mol`）。
新增两类单位后缀：`_ppm`（NMR/化学位移）、`_ev`（激发能/波长换算前）。

### 12.3 阶段二对齐的外部来源

写 §2.2 的"Jaguar 有什么"时只用了这两个来源（不是实现依据，只是能力清单）：

- Jaguar 产品页：<https://www.schrodinger.com/platform/products/jaguar/>
- Wikipedia 的 Jaguar 条目（能力清单有引用标注）：
  <https://en.wikipedia.org/wiki/Jaguar_(software)>

**明确不做的事**：不引用其算法细节作为验收依据；Jaguar 的数值不进入我们的参考集
（参考值只用公开文献/CCCBDB/我们自己跑的黑盒外部程序）。

### 12.4 环境事实

沿用 `project.md` §13（Windows 11 + Rust 1.97.1 + Python 3.12 venv + WSL bash）。
阶段二新增两条：

- **PySCF 只在 WSL 里**（`python3 -m pip install --user pyscf`），仅作黑盒尺子；
- **算力预期**：M5/M6/M7 的验收体系比阶段一大一个量级（苯的 TDDFT、水的解析 Hessian、
  溶剂化的迭代求解），建议每个 WP 的验收命令都**自带小体系版本**（H2O/STO-3G）以便
  快速迭代，大体系只在提交前跑一次。

---

## 13. 决策台账（阶段二，**已定，别再重新讨论**）

阶段一 §14 的 22 条继续有效。阶段二新增：

| # | 事项 | 结论 | 类别 |
|---|---|---|---|
| 23 | 开壳层的实现顺序 | **UKS 先于 UHF/ROHF**（DFT 是主战场；UHF 用于 CIS 参考） | 已定 |
| 24 | meta-GGA 的 τ/∇²ρ 来源 | 从 MO 系数直接求（不引入"密度梯度拟合"） | 已定 |
| 25 | RS-hybrid 的分域函数 | **erfc 短程**（Yukawa 型），与 ω→0 退回全库仑的极限测试 | 已定 |
| 26 | 相关方法 | **只做 canonical MP2 与 RI-MP2**；不做 SCS-MP2/LMP2/CC（LMP2 是 `ST-03` stretch） | 已定 |
| 27 | 激发态 | **先 TDA，后完整 TDDFT**；`method` 字段区分，报告写清楚 | 已定 |
| 28 | 溶剂化模型顺序 | **COSMO → SMD → PBF**；PBF 是 stretch | 已定 |
| 29 | 解析 Hessian 的位置 | 放 `qwb-response`（与 CPHF 共用），不新建 crate | 已定 |
| 30 | NMR 的规范 | **GIAO**（不用 CSGT/IGLO）；`length_gauge_spread_ppm < 1` 作为门禁 | 已定 |
| 31 | pseudospectral | **可选路径**，只认 K10 三条件；不进默认路径，不影响里程碑 | 已定（`stretch`） |
| 32 | 并行扩展 | 只做**线程**（rayon 确定性归约）；MPI/GPU 不做 | 已定 |
| 33 | 新依赖 | 默认不加；要加走批准 + `deny.toml` 白名单 + 理由 | 已定 |
| 34 | 阶段二的"完成" | 见 §14.2；`stretch` 允许 `blocked`，非 stretch 不允许 | 已定 |
| 35 | 账本文件 | **独立 `docs/status-v2.yaml`**（阶段一的 `status.yaml` 不再追加，只读作基线） | 已定 |

---

## 14. 变更记录与完成定义

### 14.1 变更记录

- **v3.0（2026-11-06）**：首次成文。定义 K7–K10、10 条轨道 59 个 WP、8 个里程碑、
  五层验证、G8–G10 门禁、`stretch` 与 kill criteria。**刻意不保证全部可实现**——
  阶段二的目标是"把 Jaguar 的方法广度拆成可验收的台阶"，不是"每条都成功"。

### 14.2 阶段二完成定义（Definition of Done）

同时满足以下五条才算阶段二完成：

1. **M1–M8 全部通过**（M-PS 可选）；每个 M 的端到端命令在干净 clone 上可复现；
2. **门禁全绿**：阶段一 25 项 + 阶段二新增项（KV/DA/IN/FX/SC/RS/SV/RP/PR/VF 各自的
   检查已进 `scripts/verify.{ps1,sh}`，两个脚本**同步**）；
3. **`docs/status-v2.yaml` 全部 `done` 或** `blocked` **且带实测数字 + gap 说明**；
   允许存在的 `blocked` 只来自 §2.5 标了 `stretch` 的 WP；
4. **诚实清单一致**：`docs/USAGE.md` 的"能算什么/不能算什么"与账本、代码三方一致
   （有检查脚本，缺一项即红）；
5. **误差账本**：`docs/error-budget.md` v2 对每条近似给出实测量级，且与 §9.3 的容差表不矛盾。

---

# 附录 A — v3.1 重规划：并行与 GPU 优先、数据先行（2026-11-05）

本附录**修改**上文而不删除它：WP 编号、§6 验收判据、KV 契约、K10 三条件、台账纪律一律不变；
变的是**执行顺序与执行方式**。冲突时以本附录为准；本附录没说的，仍按原文。

## A.1 已核对的环境事实（本轮实测，不是假设）

| 项 | 值 | 用途 |
| --- | --- | --- |
| CPU | 8 物理核 / 16 逻辑线程 | 门禁分片、测试并行的上限是 **8**，不是 16 |
| GPU | NVIDIA RTX 3070 Laptop, 8 GiB, 驱动 581.80 | 可做 CUDA；显存 8 GiB 决定单次 batch 的规模 |
| 其它 GPU | AMD Radeon iGPU（共享内存） | **不用**：算力与生态都不值得维护第二条路径 |
| Rust | cargo 1.97.1 | 无 `cargo-nextest`（需安装，见 EX-01） |
| 现状门禁 | `verify.ps1` 26 步，墙钟 ~45–70 min | 本附录的直接靶子 |

## A.2 新的执行优先级（覆盖 §10.5 的排队，只影响先后）

1. **D0 数据先行**（基组 / ECP / 溶剂 / 参考值）：任何实现开始前，先把该实现要用的数据取回、钉
   sha256、写 `DATA_LICENSES.md`。理由：数据缺口会在 WP 中途变成「无法验证」，本 phase 已经在
   SMD 参数、def2 辅助基上各吃过一次。
2. **EX 执行基础设施**（并行门禁 + GPU 后端）：先把验证速度降下来，后面每个 WP 的往返成本才降。
   这一步的收益是**乘性**的（剩下的 40 多个 WP 都要跑门禁）。
3. 然后 M1–M8 按原顺序推进，但**同一里程碑内允许并行**：相互无依赖的 WP 可同时开工，各自独立
   验收、独立记账；里程碑之间仍串行（M3 依赖 M1 的 SCF 契约，等等）。
4. M-PS 与 ST-01..ST-05 仍是 stretch，允许 blocked，按 §2.5 交实测差距。

## A.3 EX 阶段：把门禁从 ~60 min 压到 ≤ 15 min

**EX-01 并行测试编排**（目标：测试墙钟 ≤ 8 min）
- 装 `cargo-nextest`（进程级隔离 + `--test-threads` 与 `-j` 解耦）；装不上就退回
  「`cargo test` 按 crate 分片 + PowerShell 并行作业」，两种方案都要实测记录选型。
- 现状慢的主因不是 CPU 不够，是**重测试串行**：`uks_matches_pyscf_for_o2_and_oh`、
  `sc02_uhf_gradient_*`、`cccbdb_sto3g_hf_lsda_agree`、数值 Hessian 各自独占总进程 60–600 s。
- 验收：同一份 26 步门禁，测试段墙钟 ≤ 8 min；**且**测试集合、断言、容差一字不改。

**EX-02 测试分层与缓存**（目标：总墙钟 ≤ 15 min）
- 把「已知的慢档」标 `#[ignore]` 归入 `slow` lane（现在 cccbdb 慢档已经是这个模式），门禁跑
  `--fast`，`slow` lane 由**同一份脚本**的 `-Slow` 开关跑；`verify.ps1`/`verify.sh` 必须同步。
- 参考数据（PySCF CSV）一律**入库缓存**，门禁不再重算；需要用脚本重建时显式 `-Regenerate`。
- 编译缓存：`sccache` 或共享 `target/`，实测增量门禁不重编。

**EX-03 GPU 后端**（feature-gated，CPU 永远是可用的基准路径）
- 先做两个真正 gemm 密集、收益可测的核：
  (a) **J 构建 + AO→MO 变换**（4 索引张量收缩，`cudarc`/`cust` + cuBLAS）；
  (b) **DFT 格点 XC**（批量求值，`array`/自定义核）。
- 明确的非目标：2e Hermite 递推（分支密集，GPU 上不划算）、小矩阵对角化（PCIe 往返吃掉收益）。
- 验收：①`cargo test --features gpu` 与 CPU 路径在**规定容差**内一致（J/K/能量 1e-10 Ha，
  XC 能量 1e-10）；②单核 CPU vs GPU 的加速比**实测**并记账；③`--features gpu` 不可用时门禁
  自动退回 CPU（不得因无 GPU 而红）；④CPU 路径的位模式不得改变（`127a91e` 起的位级回归）。
- 诚实边界：8 GiB 显存装不下大基组 4 索引张量，EX-03 要有**分块**与「显存放不下就退回 CPU」的判定。

**EX-04 门禁预算与看板**
- `verify.ps1` 末尾打印每步墙钟；预算超了就是**红**（否则墙钟会慢慢爬回来）。
- 记录在 `docs/status-v2.yaml` 的实测数字里，作为 EX-* 的验收依据。

## A.4 D0 阶段：数据先行（放在所有实现之前）

| WP | 内容 | 钉法 |
| --- | --- | --- |
| D0-01 | 基组/ECP 全量补齐（sto-3g、def2-SVP/TZVP、6-311G**、LANL2DZ + RI 辅助基） | BSE 取回 + `qwb data sign` 的 sha256 + `registry.yaml` version 递增 |
| D0-02 | 溶剂/SMD：非极性项（γ、φ²、ψ²）与 Abraham 参数 | 只收**有 DOI 的表**；取不回的显式写「缺，阻塞 SV-02/03」 |
| D0-03 | 参考值缓存（PySCF / CCCBDB）：SCF、梯度、频率、NMR、溶剂化 | 生成脚本入库 + CSV 入库 + 生成命令写进注释 |
| D0-04 | 数据许可证与再分发审计 | `DATA_LICENSES.md` 每文件一行，`qwb data verify` 全绿 |

规则：**没有钉住的数据，不写实现**。数据缺口的 WP 直接 blocked 并写清缺哪个数（不是「做不了」）。

## A.5 并行与收尾纪律（回答「怎么并行」）

- 一族互不依赖的 WP 可交给**子代理并行**，每个子代理只碰自己的 crate 与测试文件；
  合并前由主会话统一 `fmt`/`clippy`/门禁——子代理不自己改台账，避免台账竞争。
- 台账只在主会话写：`docs/status-v2.yaml` 一个文件一个写者。
- GPU/并行改动属于「修改上一 phase 的内容」，**允许**，但必须在 `docs/choices.md` 记：改了什么、
  为什么、CPU 基准有没有动、位级回归怎么保证。
