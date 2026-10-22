# qwb-contracts CHANGELOG

语义化版本。**任何破坏性变更必须在此登记并经人类评审**（§4 契约前置要求）。

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

### 已知缺口（等人类决策，未擅自扩展）
- K3 缺 `queued→cancelled`、`failed→queued`（重试）。
- K5 的 4 处增量用法见 `docs/choices.md` §5。
