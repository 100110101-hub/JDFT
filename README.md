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
| `qwb-studio/` | Python 产品（P 轨，尚未创建） | K3/K6 |
| `docs/` | `status.yaml`（WBS 状态）、`choices.md`（决策日志） | — |
| `scripts/` | `verify.ps1` / `verify.sh`（一条命令跑完全部门禁）+ 三个 G 门禁脚本 | G6/G7 |

> 三个"仓"（contracts / data / core）当前在**同一个 git 工作树**里，靠目录隔离；
> `qwb-core` 以 path 依赖引用 `qwb-contracts/engine-api`。原因与拆仓触发条件见
> `project.md` §14 #2 / #21。

## 一条命令验收

```powershell
# Windows 本地（含 mypy）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1

# Linux / CI / WSL
bash scripts/verify.sh
```

当前状态：**通过 15 项，失败 0 项**（fmt / clippy `-D warnings` / test / `cargo deny` /
K1 schema 不漂移 / G2 数据校验 / `mypy --strict` ×2 / 三个门禁脚本；两个 workspace 都覆盖）。

退出码契约见 K5：`0` 成功 / `1` 输入错误 / `2` 收敛失败 / `3` 数据校验失败 / `4` 内部错误。

## 现在能做什么、还不能做什么（诚实说明）

**已实装**：K1 数据模型 + schema 生成 + 数据校验（`qwb data verify|list|show`）、
K3 状态机与结果目录布局、K4 `.qwb` 容器读写与结构校验、K5 CLI（`inspect` /
`replay --dry-run` 完全可用）、K6 类型桩。

**还没实装**：任何数值内核。`qwb run` / `qwb batch` 会**完整解析并校验输入**，
然后返回退出码 `4`（"本构建未链接引擎后端"）—— 不会伪造结果。
数值内核按 `project.md` §10.5 的顺序推进（D → I → X → S → T1 …）。
