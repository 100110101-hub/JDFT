# QuantumWorkbench (QWB) v2.1

分子 DFT 软件（**产品面模仿 Jaguar**，算法面全部自研）。

**唯一开工依据是 [`project.md`](./project.md)** —— 本 README 只说明"目录长什么样"和
"一条命令怎么验"，不构成规范。新会话请从 `project.md` §0 开始。

## 目录

| 目录 | 内容 | 契约 |
|---|---|---|
| `qwb-contracts/` | K1 schema（schemars 生成）、K2 `qwb-engine-api` crate、K3 task/result schema、K4 格式定义、K5 CLI 约定、K6 Python 桩 | K1–K6 |
| `qwb-data/` | 数据仓：`registry.yaml`、`basis/ ecp/ functional/ grid/ solvent/`、`examples/`（K1 样例，非交付物） | K1 |
| `qwb-core/` | Cargo workspace：`qwb-registry`（K1 + 校验）、`qwb-io`（K3 状态机 + K4 容器）、`qwb-cli`（K5） | K2/K4/K5 |
| `qwb-studio/` | Python 产品（P 轨）：`qwb_studio`（pydantic 模型 / `.qwb` / 调度器 / `env.json` / 结果目录 / CLI 壳 / 端到端 runner）+ `rust/qwb-py`（pyo3 `qwb_engine` 扩展） | K3/K4/K6 |
| `docs/` | `status.yaml`（WBS 状态）、`choices.md`（决策日志）、`unblock.md`、`error-budget.md` | — |
| `scripts/` | `verify.ps1` / `verify.sh`（一条命令跑完全部门禁）+ 三个 G 门禁脚本 + 数据/夹具生成脚本 | G6/G7 |

> 三个"仓"（contracts / data / core）当前在**同一个 git 工作树**里，靠目录隔离；
> `qwb-core` 以 path 依赖引用 `qwb-contracts/engine-api`。原因与拆仓触发条件见
> `project.md` §14 #2 / #21。

## 一条命令验收

```powershell
# Windows 本地
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1

# Linux / CI / WSL
bash scripts/verify.sh
```

门禁需要仓库根的 `.venv`（Python 侧四类检查 + pyo3 扩展构建都跑它）：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install mypy numpy pydantic pytest maturin
```

当前状态：**24 项全绿**（2026-10-25 第 44 轮；fmt / clippy `-D warnings` / test / `cargo deny` /
K1 schema 不漂移 / G2 数据校验 / 三个门禁脚本 / `mypy --strict` ×2 / pyo3 扩展的
fmt-clippy-构建安装 / Python 侧 129 个单测 / `.qwb` 夹具新鲜度；两个 Rust workspace
加 Python 侧都覆盖）。各项与最近一次通过的轮次记在 `docs/status.yaml` 的 C-06。

退出码契约见 K5：`0` 成功 / `1` 输入错误 / `2` 收敛失败 / `3` 数据校验失败 / `4` 内部错误。

## 现在能做什么、还不能做什么（诚实说明）

**已实装**

* **契约层**：K1 数据模型 + schema 生成 + 数据校验（`qwb data list|show|verify|sign`）、
  K3 状态机与结果目录布局、K4 `.qwb` 容器（Rust 与 Python 双向互操作）、
  K5 CLI（9 个子命令）、K6 Python 类型桩 + 真正的 pyo3 绑定。
* **数值内核**：闭壳层 RHF / LDA / GGA(PBE, BLYP) / hybrid(B3LYP, PBE0) 的 SCF，
  解析梯度（H2O/CH4 等小体系）、几何优化、数值 Hessian + 热化学 + IR，
  STO-3G / def2-SVP / def2-TZVP / 6-311G(d,p) 基组。与 CCCBDB 的 STO-3G 外部基准
  对得上（H2/H2O/CO/CH4/NH3/N2 五个模型化学都在 1e-4 Ha 内，详见 `docs/status.yaml` T6/V-03）。
* **产品层（P 轨）**：Python 侧从模型层到端到端全通 —— 建模、`.qwb` 读写、本地调度器
  （并发 / 取消 / `kill -9` 恢复）、`env.json` 生成、结果目录 G5 校验、`qwb` 薄壳、
  pyo3 绑定（与 CLI 能量逐位相同、GIL 真释放）、Python 端到端跑通 T1
  （写出的结果目录 Rust `qwb replay` 能重算，delta `0.000e0 Ha`）。

**还没实装（不要当成能用）**

* **作业类型**：`EngineBackend` 目前只接 `sp`；`opt` / `freq` 走 CLI 或 pyo3 会被拒
  （退出码 4）。优化/Hessian 代码存在并被测试覆盖，但还没接到 `run` 的作业分发上。
* **ECP**（T5）：数据缺 D-07/D-08，且 K1 的 ECP 语义要人类拍板，故仍 `blocked`。
* **频率判据**（T4）：谐振频率已能算，但要压到「与实验差 < 30 cm⁻¹」还差 2e 导数预筛；
  目前 STO-3G 标度后仍差 +377/+107/+236 cm⁻¹，如实记在 `docs/status.yaml`。
* **KS 解析梯度的基组运动项**（T6）：B3LYP 解析梯度还没到 1e-6，已知缺口写在 G-01。
* **GUI**（P-09 / P-10）：还没开始（项目树 / Task Panel / Job Monitor / VTK 3D）。
* **溶剂**：只有 `vacuum` 能算，隐式溶剂是接口占位。

