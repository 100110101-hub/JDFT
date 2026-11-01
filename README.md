# QuantumWorkbench (QWB) · JDFT

分子 DFT 软件。**产品面模仿 Jaguar，算法面全部自研**：不调用 PySCF / ORCA / Gaussian /
Q-Chem / Psi4 / libxc / ase 的代码或数据。PySCF 只在 `scripts/pyscf_*.py` 里当**外部尺子**
产参考数字（产物是 `tests/reference/*.csv`，每行带 DOI/URL），代码与数据一个字节都不进仓库。

同一棵 git 树里三个"仓"，靠目录隔离：`qwb-contracts/`（K1–K10 契约）→ `qwb-data/`（数据仓）
→ `qwb-core/`（Rust 引擎）+ `qwb-studio/`（Python 产品层）。
`qwb-core` 以 path 依赖引用 `qwb-contracts/engine-api`（理由见 `project.md` §14 #2 / #21）。

| 想知道 | 去哪看 |
|---|---|
| 怎么装、`input.yaml` 每个字段什么意思、CLI/Python/GUI 怎么用、结果怎么读 | [`docs/USAGE.md`](./docs/USAGE.md) |
| 契约、WBS、验收判据（阶段一） | [`project.md`](./project.md) |
| 契约 K7–K10、里程碑 M1–M8、GPU/性能条款（阶段二 v3.1） | [`project-phase2.md`](./project-phase2.md) |
| 每个 WP 的 `state`、artifact 与**实测数字** | [`docs/status.yaml`](./docs/status.yaml)、[`docs/status-v2.yaml`](./docs/status-v2.yaml) |
| 近似的误差预算（每条都有实测出处） | [`docs/error-budget.md`](./docs/error-budget.md) |
| 为什么这么做、踩过什么坑 | [`docs/choices.md`](./docs/choices.md)（按时间编号一百七十九条） |

## 30 秒验证它真的能算

```powershell
cd qwb-core
cargo run --release -q -p qwb-cli --bin qwb -- run examples/h2_sto3g_svwn5.yaml --out results
```

```text
scf     converged = true
energy  -1.1570005110 Ha
task    results\h2_sto3g_svwn5
```

`results/h2_sto3g_svwn5/` 是完整的 K3 结果目录（`input.yaml` / `result.json` / `env.json` /
`data_snapshot.json` / `log.txt` / `artifacts/`），可以用 `qwb inspect` 检查、用 `qwb replay`
重算比对。

**第一次跑会先按需取回这次要用的基组/ECP**（约 1 秒，需要网络）。仓库**故意不含**这 10 个
从 Basis Set Exchange 导出的第三方数据文件（再分发条款见 §数据政策），只含配方与 sha256 钉；
想一次备齐：

```powershell
qwb data install                    # 或 python scripts\fetch_data.py --all
```

取回一次之后就是纯本地；`$env:QWB_AUTO_FETCH='0'` 可关掉自动取回。细节见
[`docs/USAGE.md`](./docs/USAGE.md) §8.4。

## 现在能算什么（实测，不是宣称）

| 能力 | 证据（数字都在账本里） |
|---|---|
| 闭壳层 RHF / LDA(SVWN5) / GGA(PBE, B88) / hybrid(PBE0) | STO-3G 上 7 体系 × 5 模型化学与 CCCBDB 差 ≤ 4e-6 Ha；苯（36 基函数）B3LYP grid-normal −1.16 mHa、PBE −1.33 mHa（判据 2 mHa） |
| meta-GGA（`m06-l` / `m06-2x`） | 交换/相关逐点对 libxc < 1e-9；H2O/def2-SVP 端到端差 < 3e-5 Ha。**格点成本是瓶颈**，见"没做" |
| 开壳层 UHF / UKS / ROHF | UHF 的 O2/OH 与 PySCF 差 3e-8；UKS/PBE 的 O2 差 5.8e-7、OH 1.1e-6；ROHF 5 体系最差 3.6e-8、`⟨S²⟩` 精确等于 `S(S+1)`（NH2 的 90 mHa 之谜查出来是初猜，改 SAD 后 1.7e-8） |
| ECP（LANL2DZ，Cu/Zn） | 矩阵元与解析解 3e-14；CuH/ZnO 单点跑通（ZnO 需要 SCF 退火退路） |
| 基组 | STO-3G / def2-SVP / def2-TZVP / 6-311G(d,p) / LANL2DZ（+ 三个 RI 辅助基）；与 BSE 导出逐项 diff < 1e-12，文件按 sha256 钉取回 |
| 几何优化（BFGS + 解析梯度） | H2/STO-3G 三步到 3.4e-5 Ha/Bohr；H2O/def2-SVP PBE 到 2.67e-4 Ha/Bohr；UHF 用解析梯度、ROHF 目前走数值梯度；UKS 还不能 opt |
| 数值频率 / 热化学 / IR | 代码与测试在（数值 Hessian + 标度因子 + IR 强度），**但没接进引擎**：`jobtype: freq` 以退出码 4 拒绝 |
| MP2 / RI-MP2 | canonical 与 PySCF 差 1e-9；RI-MP2 的拟合误差落盘（H2O/def2-SVP **8.526e-5 Ha**，`ri_error_hartree`）。RI 现在**不更快**（IN-04 实测 0.49×），如实记在账本 |
| TDA / CIS / TDHF + UV/Vis 谱 | H2O/STO-3G 前三态 13.1998 / 15.1244 / 16.8053 eV（与 PySCF 同值），TDHF 13.1580 / 15.1099 / 16.6954 eV；展宽曲线写 `artifacts/spectrum.csv` |
| COSMO / C-PCM | H2O/STO-3G 的 `E_solv − E_vac = −7.274 kcal/mol`（实验 −6.32，判据 1.5 内）；空腔 36.458 Å² / 19.388 Å³ / 410 碎片 |
| 色散 D3(BJ) / D3(0) / D4 | 与官方实现（s-dftd3 / dftd4）逐项对上（能量与解析梯度，1e-6 kcal/mol 量级）；水/PBE0/STO-3G 实测 `E_disp` = −2.7686e-4 Ha（D3BJ）、−1.5626e-4 Ha（D4） |
| 幽灵原子 / counterpoise BSSE | 水二聚体 STO-3G 与 PySCF 的 `ghost:` 差 5e-8；BSSE(A) = **2.318e-5 Ha**（带变分符号断言） |
| EPR / 超精细耦合 | O2/STO-3G UHF：`a_iso(O) = −81.48 MHz`，Fermi 接触项 + 锚点定标，常数与公式各留一条独立测试 |
| 多极矩 | 每次闭壳层 `sp` 落 `multipoles`（无迹四极矩，原点 = 核电荷中心，随块落盘） |
| 工作流 | K3 状态机 + 结果目录、`qwb batch` 持久队列与 `kill -9` 恢复、`qwb replay` 复现（delta 0.000e0）、`qwb inspect`、`qwb compare`（表格 + CSV） |
| GUI（PySide6 + VTK） | Build → 提交 → Job Monitor → 3D 全链路；102.8 MB 日志不卡；崩溃后重启恢复未完成视图 |
| Python API（pyo3，abi3-py39） | 与 CLI 能量逐位相同；GIL 真释放（单次 1.000 s vs 两个并发 1.078 s） |

## 阶段二进度（账本口径，82 个 WP）

`docs/status-v2.yaml` 现在是 **47 done / 4 in_progress / 9 blocked / 22 pending**。
里程碑：**M1 / M1b / M1c / M3 / M4 已交付**，M2 半交付，M5–M8 未开始（按 §10.5 的依赖顺序推进）。

| WP | 状态 | 差在哪（账本原文的要点） |
|---|---|---|
| SV-03 | in_progress | SMD 的 CDS 项能量层已对上上游（12 体系 × 溶剂最差 8.43e-4 Ha），缺 `E_CD`/`E_S`/`E_C` 三项非极性参数（付费墙，无本地 oracle）与引擎接线（`solvent_model: smd` 仍 exit 4） |
| RS-04 | in_progress | 抗磁 + 顺磁的积分层已落，顺磁项本体（1/r³ 加权 + CPHF）与 GIAO 规范化未做，所以还没有 `nmr_shieldings` 块 |
| VF-06 | in_progress | K10 的 `equivalence_tolerance.ri` 声明 1e-6，而 RI-MP2 与 canonical 的实测差是 **8.526e-5 Ha**（是辅助基不完备，不是实现错，两者差值逐位相同）→ 等价性套件目前是报告模式，等容差语义重新界定 |
| M2 | in_progress | 能力（meta-GGA + D3/D4 + S22 几何 + ghost 原子）都在；第 ④ 条"与文献差 < 1 kcal/mol"缺数：苯的 M06-2X/def2-SVP 单点 1077 s 未收敛完（格点成本），BSSE 校正也还没有协议 |
| DA-03 | blocked | ωB97X-D / CAM-B3LYP 的 DFT 半边不在范围内（K1 拒绝 `requires.laplacian`，LYP 无数据） |
| DA-08 | blocked | 溶剂化自由能与势垒高度的基准集没有可对齐的实现（依赖 SV-03 / RP-01..03），不造假数据 |
| IN-04 | blocked | RI-J / RI-JK 实测更慢（水/def2-SVP：4 指标 0.172 s vs RI-J 0.349 s），判为当前不值得做 |
| FX-03 | blocked | 长程校正泛函需要 DA-03 的 B97/LYP 与 B3LYP 对照 |
| RS-05 | blocked | 解析 Hessian 的 20× 目标：要 CPHF 3N 次响应 + 2e 导数核 ≥ 7.7×，两条都不可少；实测证明单独做任一条都不够 |
| RS-06 | blocked | Raman 需要 `∂α/∂R` 与简正模式，而简正模式依赖 RS-05 的频率 |
| SV-01 | blocked | 解析 SASA：点采样在任意转动下只有 O(1/√N) 收敛（500→8000 点：4.2e-4→8.7e-5），判据 1e-6 要换解析球冠公式 |
| ST-05 | blocked | EPR 的各向异性项与实验基准未完成（Fermi 接触项已交付并有 PySCF 对照） |
| EX-03 | blocked | GPU 的格点 XC 需要 NVRTC / cubin，属**另一类** unsafe 面，等人类批准；J 构建与 T1 变换已完成（T1 实测 60.8×） |

## 明确没做（不要当成能用）

- `jobtype: freq` **未接到 `EngineBackend`**（退出码 4）；带 ECP 的 `opt`/`freq`；UKS 的 `opt`
  （开壳层 UKS 的 SCF 收敛是 SC-02 的剩余一半）；UKS 的 B88/LYP 自旋实现（BLYP/B3LYP 的开壳层被显式拒绝）。
- 隐式溶剂只到 COSMO/C-PCM 的单点：`solvent_model: smd`（SV-03）与 `pbf`（SV-04）以退出码 4 拒绝；
  CDS 能量层已在 crate 里并有验收，但**没有接线**。
- `options.numerics` 一律退出码 4：近似路径（RI-J、pseudospectral）还没过 K10 的三条门禁。
- 解析 Hessian（RS-05）、GIAO NMR（RS-04）、Raman（RS-06）、VCD（RS-07）、TS / IRC / 扫描（RP-01..04）：
  `qwb ts|irc|scan` 只落签名，退出码 4。
- GPU：只有 `gemm` 原语、显存预算与自动回退、AO→MO 变换；格点 XC 的 GPU 核仍 blocked（EX-03）。
- 并行只到"Fock 构建按行并行（确定性归约）"：4 指标积分张量仍是单线程，`--threads`/`options.threads`
  只写进 `env.json`（记 1），不驱动并行度。
- 开壳层结果不写 `artifacts/populations.json` 与 cube：K2 的 `mo_coeffs` 只有 α 一套，
  拿它算布居会算出一个错的数 —— 宁可不写。
- **开壳层必须显式写 `options.scf_type`**：省略时多重度不参与决策，引擎会按闭壳层算出一个
  能量高 0.6 Ha、也没有 `spin` 块的"成功"结果（已知缺陷，输入校验还没补）。细节见
  [`docs/USAGE.md`](./docs/USAGE.md) §3.4。
- `.qwb` 容器：K4 读写与互操作有测试，但 CLI 没有"打包"子命令（Python 侧 `QwbContainer` 可以）。

完整的"能算什么/不能算什么"、判据数字与出处见 [`docs/USAGE.md`](./docs/USAGE.md) §9、
[`docs/error-budget.md`](./docs/error-budget.md) 与两份账本。

## 一条命令跑完所有门禁（33 项）

```powershell
# Windows（三个 bash 门禁会通过 WSL 跑）
pwsh -NoProfile -File scripts\verify.ps1
# Linux / CI / WSL
bash scripts/verify.sh
```

需要仓库根的 `.venv`（Python 侧检查、pytest 与 pyo3 扩展构建都用它）：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install mypy numpy pydantic pytest maturin PySide6 vtk
```

门禁覆盖：3 个 bash 策略门禁（依赖集中 / unsafe 白名单 / rayon 确定性）→ 第三方数据就位
（缺则取回）→ K2 契约 crate 的 fmt/clippy/doc/test → `qwb-core` 的 fmt/clippy、**编译与计时分离**
的 test（两条 `cargo-nextest` 泳道 + doctest）、I-06 积分缓存、I-00 faer 后端、`cargo deny`、
K1 schema 不漂移、G2 数据校验 → D-15 数据许可、DA-08 参考集来源、VF-05 误差预算、DA-06 色散数据
三条、VF-06 等价性差异表 → 两份账本的合法性 → K6 mypy ×2、pyo3 fmt/clippy/构建安装、pytest 两条泳道、
`.qwb` 夹具新鲜度。

实测墙钟（8 物理核 / 16 逻辑核，Rust 1.97.1）：

| 情形 | 总墙钟 | 测试段 |
|---|---|---|
| 暖路径（源码没动，只有增量链接） | **279–322 s（4.7–5.4 min）** | **155–161 s** |
| 冷路径（改了共享类型，测试二进制全要重编） | 976 s（16.3 min） | 161 s（编译 518 s 单独一步、不计入测试段） |

两个硬指标是**测试段 ≤ 8 min、总墙钟 ≤ 15 min**：前两条都达成，冷路径的 16.3 min 如实记在
`docs/choices.md` 一百七十一，不靠把某一步挪出计时来凑。

## 目录

| 目录 | 内容 | 契约 |
|---|---|---|
| `qwb-contracts/` | K1 schema（`schemars` 从 Rust 类型生成，勿手改）、K2 `qwb-engine-api`、K3 task/result schema、K4 `.qwb`、K5 CLI 约定、K6 Python 桩；阶段二增补 K7–K10 | K1–K10 |
| `qwb-data/` | 数据仓：`registry.yaml`（v0.19.0，34 个生产文件）、`sources.json`（10 个第三方文件的配方）、`basis/ ecp/ functional/ grid/ solvent/ dispersion/ numerics/`、`examples/`（K1 样例，非交付物） | K1 |
| `qwb-core/` | Cargo workspace：`qwb-registry` / `qwb-basis` / `qwb-integrals` / `qwb-xc` / `qwb-grid` / `qwb-linalg` / `qwb-scf` / `qwb-props` / `qwb-response` / `qwb-gpu` / `qwb-engine` / `qwb-io` / `qwb-cli` | K2/K4/K5 |
| `qwb-studio/` | Python 产品层：`qwb_studio`（pydantic 模型 / `.qwb` / 调度器 / 结果目录 / CLI 薄壳 / GUI / 数据取回）+ `rust/qwb-py`（pyo3 `qwb_engine`） | K3/K4/K6 |
| `docs/` | `USAGE.md`、`status.yaml`（阶段一账本）、`status-v2.yaml`（阶段二账本）、`choices.md`（决策日志）、`error-budget.md`、`HANDOVER.md`、`unblock.md` | — |
| `scripts/` | `verify.ps1` / `verify.sh`（全部门禁）+ 三个 bash 策略门禁 + 数据/夹具/参考生成脚本 + 账本与来源检查 | G6/G7 |

## 数据政策

- **34 个生产数据文件**（33 个数据文件 + `registry.yaml` 本身：8 基组 + 2 ECP + 6 泛函 +
  4 网格 + 8 溶剂 + 1 元素级 SMD 数据 + 2 套 D3 + 1 套 D4 + 1 份 numerics）全部带 sha256 钉，
  `qwb data verify` 实测 **34/34 通过**。
- 其中 **10 个第三方文件不进仓库**（BSE 导出的 5 个轨道基组、3 个 RI 辅助基、2 个 LANL2DZ ECP）：
  LANL2DZ 的 ECP 原始论文没有明确再分发条款，所以标成 `commercial_ok: false` /
  `redistribute_ok: false`，仓库只带配方（`qwb-data/sources.json`）与摘要钉，由使用者按
  `qwb data install` 取回。
- 参考集的**原始快照**（`mnsol.F`、`radii.py`、`s22.py`，LGPL/学术许可）放在 gitignored 的
  `_*_staging/raw/` 里，派生出的 CSV/JSON 随仓库提交并逐行带 DOI/URL；`scripts/check_reference_sources.py`
  在门禁里核这件事。
- 自研/自转录数据是 `Apache-2.0`，逐文件登记在 [`DATA_LICENSES.md`](./DATA_LICENSES.md)（34 个文件）。

## 许可

代码 `Apache-2.0`（见 [`LICENSE`](./LICENSE)）；数据许可见 [`DATA_LICENSES.md`](./DATA_LICENSES.md)。
第三方数据的再分发条款逐条登记，`redistribute_ok: false` 的文件**不进仓库**。
