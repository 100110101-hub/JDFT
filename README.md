# QuantumWorkbench (QWB) v2.1 · JDFT

分子 DFT 软件（**产品面模仿 Jaguar**，算法面全部自研：不调用 PySCF/ORCA/Gaussian/QE/Psi4/libxc/ase）。

- **用法请看 [`docs/USAGE.md`](./docs/USAGE.md)** —— 安装、`input.yaml` 逐字段说明、CLI、Python API、
  GUI、结果目录、数据仓、故障排查，都在那一份里。
- **阶段一规范与验收判据看 [`project.md`](./project.md)** —— 契约 K1–K6、WBS、垂直切片 T1–T8、§7.0 参考值政策。
- **阶段二规划看 [`project-phase2.md`](./project-phase2.md)** —— v3.0：开壳层、meta-GGA/RS 泛函、色散、
  RI-MP2、TDDFT/CIS、COSMO/SMD（PBF 为 stretch）、TS/IRC/扫描、解析 Hessian、NMR/极化率/VCD、
  pseudospectral（可选路径）。契约 K7–K10、里程碑 M1–M8、含"允许失败"的 stretch 条款。
- **进度与实测数字看 [`docs/status.yaml`](./docs/status.yaml)**（阶段一）与
  [`docs/status-v2.yaml`](./docs/status-v2.yaml)（阶段二）—— 每个 WP 的 `state`、
  artifact、verification（含实测值与来源）。
- **为什么这么做看 [`docs/choices.md`](./docs/choices.md)** —— 按时间编号的决策与踩坑日志。

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

`results/h2_sto3g_svwn5/` 是一个完整的 K3 结果目录（`result.json` / `env.json` /
`data_snapshot.json` / `log.txt` / `input.yaml` / `artifacts/`），可以用
`qwb inspect` 检查、用 `qwb replay` 重算比对。

**第一次跑会先自动取回这次要用的基组/ECP**（约 1 秒，需要网络）。仓库**故意不含**
这 7 个从 Basis Set Exchange 导出的第三方数据文件（再分发条款），只含配方与 sha256 钉；
想一次备齐：

```powershell
qwb data install                    # 或 python scripts\fetch_data.py --all
```

取回一次之后就是纯本地。细节见 [`docs/USAGE.md`](./docs/USAGE.md) §8.4。

## 一条命令跑完所有门禁

```powershell
# Windows（三个 bash 门禁会通过 WSL 跑）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1
# Linux / CI / WSL
bash scripts/verify.sh
```

需要仓库根的 `.venv`（Python 侧检查 + pyo3 扩展构建都用它）：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install mypy numpy pydantic pytest maturin PySide6 vtk
```

当前状态：**26/26 全绿**（第 67 轮，Python 侧 182 个测试；第 67 轮把 UKS（M1b）接进门禁）。
逐项与最近轮次记在 [`docs/status.yaml`](./docs/status.yaml) 的 C-06。

## 目录

| 目录 | 内容 | 契约 |
|---|---|---|
| `qwb-contracts/` | K1 schema（`schemars` 从 Rust 类型生成，勿手改）、K2 `qwb-engine-api`、K3 task/result schema、K4 `.qwb` 定义、K5 CLI 约定、K6 Python 桩（阶段二增补 K7–K10） | K1–K10 |
| `qwb-data/` | 数据仓：`registry.yaml`、`sources.json`（第三方数据配方）、`functional/ grid/ solvent/`、`examples/`（K1 样例，非交付物）；`basis/ ecp/` 里的第三方文件按需取回 | K1 |
| `qwb-core/` | Cargo workspace：`qwb-registry` / `qwb-basis` / `qwb-integrals` / `qwb-xc` / `qwb-grid` / `qwb-linalg` / `qwb-scf` / `qwb-props` / `qwb-engine` / `qwb-io` / `qwb-cli` | K2/K4/K5 |
| `qwb-studio/` | Python 产品层：`qwb_studio`（pydantic 模型 / `.qwb` / 调度器 / `env.json` / 结果目录 / CLI 薄壳 / GUI）+ `rust/qwb-py`（pyo3 `qwb_engine`） | K3/K4/K6 |
| `docs/` | `USAGE.md`（使用手册）、`status.yaml`（阶段一账本）、`status-v2.yaml`（阶段二账本）、`choices.md`（决策日志）、`HANDOVER.md`、`unblock.md`、`error-budget.md` | — |
| `scripts/` | `verify.ps1` / `verify.sh`（全部门禁一条命令）+ 三个 G 门禁脚本 + 数据/夹具/外部基准生成脚本 | G6/G7 |

三个"仓"（contracts / data / core）在**同一个 git 工作树**里，靠目录隔离；
`qwb-core` 以 path 依赖引用 `qwb-contracts/engine-api`（理由见 `project.md` §14 #2 / #21）。

## 能力一览（实测，不是宣称）

**已实装并用外部基准验证**

| 能力 | 证据 |
|---|---|
| 闭壳层 RHF / LDA(SVWN5) / GGA(PBE, B88) / hybrid(B3LYP², PBE0) | CCCBDB Release 22，STO-3G，7 体系 × 5 模型化学，差 ≤ 4e-6 Ha |
| 苯的 B3LYP/PBE（36 个基函数） | 距 CCCBDB grid-coarse −0.01/−0.04 mHa，grid-normal −1.16/−1.33 mHa（判据 2 mHa） |
| 几何优化（BFGS + 解析梯度） | PBE0/def2-SVP 的 H2O O–H 0.9623 Å（实验 0.9572）；PBE/def2-TZVP 4 步到 2.67e-4 Ha/Bohr |
| 频率（数值 Hessian + 标度因子 + 热化学 + IR） | PBE/6-311G(d,p)×0.991 最差 21.5 cm⁻¹（判据 30）；PBE0 60.5 cm⁻¹ 作为不通过对照组记录 |
| 开壳层 UHF（`scf_type: uhf`） | O2 三重态/STO-3G 与 PySCF 差 3.4e-8、⟨S²⟩ 2.003411（污染如实报出）；OH 差 3e-8 |
| 开壳层 UKS（`scf_type: uks`，`svwn5`/`pbe`） | O2 的 UKS/PBE 与 PySCF 差 5.8e-7、UKS/svwn5 差 2.1e-7、OH/PBE 差 1.1e-6（容差 5e-6，网格）；闭壳层 `uks` ≡ `rks` 差 6.2e-14 |
| ECP（LANL2DZ，Cu/Zn） | 矩阵元与解析解 3e-14；CuH/ZnO 单点跑通（ZnO 需 SCF 退火） |
| 基组 | STO-3G / def2-SVP / def2-TZVP / 6-311G(d,p) / LANL2DZ（+LANL2DZ 价基）；数据逐项对 BSE 导出 diff < 1e-12；文件按需从 BSE 取回并核对 sha256 钉 |
| 工作流 | K3 状态机 + 结果目录、`qwb batch` 持久化队列与 `kill -9` 恢复、`qwb replay` 复现（delta 0.000e0） |
| GUI（PySide6 + VTK） | Build→提交→Job Monitor→3D 全链路；102.8 MB 日志不卡；崩溃后重启恢复未完成视图 |
| Python API（pyo3，abi3-py39） | 与 CLI 能量逐位相同；GIL 真释放（单次 1.000 s vs 两个并发 1.078 s） |

² B3LYP 只能在测试里程序化构造：K1 的 phase-1 语义门禁拒绝 `requires.laplacian = true`（冻结契约）。

**没做（不要当成能用）**

- `JobType::Freq` 未接到 `EngineBackend`（会以退出码 4 拒绝）；频率代码由测试直接调用。
- 带 ECP 的 `opt`/`freq`（ECP 对核坐标的导数还没实现）。
- 隐式溶剂（只有 `vacuum`）、meta-GGA、ROHF（SC-01c：blocked）、TDDFT、NBO、TS/IRC/scan。
- 开壳层的 `opt`/`freq`（SC-02）、开壳层的 Mulliken 布居/cube（K2 只有 α 轨道集）。
- UKS 的 B88/LYP 自旋实现（所以 BLYP/B3LYP 的开壳层会被显式拒绝）。
- 积分层并行（`--threads` 只记账）；ECP 的解析权重导数（Hessian 因此比理想情况慢）。

完整的"能算什么/不能算什么"、判据数字与出处见 [`docs/USAGE.md`](./docs/USAGE.md) §9
与 [`docs/status.yaml`](./docs/status.yaml)。

## 许可

代码 `Apache-2.0`（见 [`LICENSE`](./LICENSE)）。数据许可见 [`DATA_LICENSES.md`](./DATA_LICENSES.md)：
自研/自转录数据为 `Apache-2.0`；**外部来源**（BSE 导出的 STO-3G/def2/6-311G(d,p)、
LANL2DZ 的基组与 ECP）逐条登记，其中 LANL2DZ 的 ECP 两项因为原始论文没有明确再分发条款，
标成 `commercial_ok: false` / `redistribute_ok: false` —— 所以这 7 个文件**不进仓库**，
由使用者从上游取回（`qwb data install`），仓库只带配方与摘要钉。
