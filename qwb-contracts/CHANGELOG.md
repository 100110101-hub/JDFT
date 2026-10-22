# qwb-contracts CHANGELOG

语义化版本。**任何破坏性变更必须在此登记并经人类评审**（§4 契约前置要求）。

## 1.1.0 — 2026-10-22（D 轨增量：K1 新增一个可选字段）

### K1 · `Functional.parameter_sources`（**新增可选字段，非破坏**）
- 类型：`BTreeMap<String, String>`（参数名 → 出处），`#[serde(default)]`，缺省即空。
- 动机：§6.2 D-09 的验收要求“参数逐个带 citation”，而 K1 的 `citation` 是文件级的，
  无法定位到单个参数。§10.3 明确允许“新增可选字段/新变体/新方法”直接做。
- 校验（`Functional::check_structure`）：`parameter_sources` 的键必须是 `parameters`
  里存在的键（禁止悬空来源）；不强制覆盖全部参数（部分参数可由文件级 citation 覆盖）。
- 兼容性：旧数据文件（无此字段）行为不变；`functional.schema.json/yaml` 已重新生成。

## 0.1.1 — 2026-10-22（随 project.md v2.1）

### K3 · 状态机新增一条转移（**新增，非破坏**）
- `queued → cancelled`：取消一个尚未开始的任务。P-03 的"取消即停"需要它，
  T7 的"kill -9 后恢复 queued 任务"也需要它。
- 同步改动：`qwb-core/crates/qwb-io/src/task.rs`（`TaskState::LEGAL` 5 → 6）、
  `studio/task.schema.yaml`（`x-state-machine.transitions`）、单测与 proptest。
- **明确不改**：`failed → queued`（重试）**不**加入。重试语义 = 用**新 task id** 重建任务，
  保证结果目录不可变、可追溯。这条现在是冻结决策（project.md §4 K3 / §14 #15）。

### 文档
- `project.md` v2.1 把 C 轨实际落地的 K1–K6 形态全部写入 §4，并清掉 v2.0 的 6 处内部矛盾
  （详见 project.md §15）。本 CHANGELOG 与 project.md §4 应始终一致。

## 0.1.0 — 2026-10-22（C 轨：K1–K6 首次冻结）

### K1 · 数据 schema（`schema/`）
- 新增 7 份 schema，由 `qwb-core/crates/qwb-registry` 的 Rust 类型经 `schemars` 生成：
  `basis, ecp, functional, grid, solvent, molecule, registry`。
  每份同时产出 `.schema.json`（工具读）与 `.schema.yaml`（人读，K1 规定的文件名）。
- 生成器 `qwb-gen-schemas <dir> [--check]`；CI 用 `--check` 防漂移。
- 共同字段：`name, version, citation[], license{spdx,commercial_ok,redistribute_ok}, created, sha256?, notes?`。
- 规范化 sha256 规则（与排版无关）见 `docs/choices.md` §2.5。

### K2 · `qwb-engine-api` 0.1.0（`engine-api/`）
- `JobInput / JobResult / Molecule / Atom / TaskOptions / GridLevel / JobType /
  DataSnapshot / EnvInfo / Registry / EngineError / Reporter / Engine`。
- 形式化补充：`Registry` 定义为“数据根 + feature 集合”句柄；新增 `EngineBackend` +
  `Engine::with_backend`；新增 `EngineError::InvalidInput`（→ K5 退出码 1）。
- 矩阵 JSON 表示定案为 **list-of-list**（K4 二选一）。
- 公共 enum/struct 全部 `#[non_exhaustive]`；`#![deny(unsafe_code)]`、`#![deny(missing_docs)]`。

### K3 · Task / Result（`studio/task.schema.yaml`, `studio/result.schema.yaml`）
- Task 状态机：`draft→queued→running→{converged|failed|cancelled}`，**只有这 5 条**。
- `result.json` = `JobResult`；`env.json` = `EnvInfo`；缺 `env.json` 视为不完整结果（G5）。
- 文档与 Rust 类型的一致性由 `qwb-core/crates/qwb-io/tests/k3_*.rs` 强制。

### K4 · 文件格式（`studio/qwb-format.md`）
- 输入 `input.yaml` ≡ `JobInput`；结果目录 5 文件 + `artifacts/`；
  项目容器 `.qwb` = Zip（`meta.json` 第一），**不用 HDF5**。
- 容器结构性约束（路径穿越/绝对路径/重复条目/缺 meta 一律拒绝）与**字节可复现**保证。

### K5 · CLI（`qwb-core/crates/qwb-cli`, `studio/cli.md`）
- `run / batch / inspect / replay / data {list,show,verify}`；退出码 0–4 冻结，
  映射由 `EngineError::exit_code()` 单测钉死。

### K6 · Python（`studio/python/qwb_engine.pyi`）
- 冻结桩：`Engine(registry_path=..., *, features=...)`、`run(job, reporter=None)`、
  `request_cancel()`；错误类型带 `exit_code`；矩阵为 `NDArray[float64] | None`。
- 验收：`mypy --strict` 对桩文件与一份 typed consumer 均 0 issue。

### 已知缺口
- ~~K3 缺 `queued→cancelled`~~ → **0.1.1 已补**。
- `failed→queued`（重试）**故意不加**：重试 = 新 task id（冻结决策）。
- K5 的 4 处增量用法已写进 project.md §4 K5（不再是"隐含增量"）。
