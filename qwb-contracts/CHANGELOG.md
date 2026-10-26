# qwb-contracts CHANGELOG

语义化版本。**任何破坏性变更必须在此登记并经人类评审**（§4 契约前置要求）。

## 2.0.0 — 阶段二契约轨道 KV-01..KV-03（2026-11-06，`project-phase2.md` §4）

**全部是加性变更**：旧 `input.yaml` / 旧 `result.json` / 旧 Python 代码继续工作（有回归测试）。

### K1 · `Solvent`（**新增 4 个可选字段 + 枚举**，非破坏）

- 新增 `RadiiSet { bondi, uff, smd }`；新增可选字段 `refractive_index`（n，SMD 用 n²）、
  `radii_set`（半径集必须声明，混用会静默错数）、
  `surface_tension_cal_per_mol_angstrom2`（空化项）、`ionic_strength_molar`（PBF）。
- 校验（`Solvent::check_structure`）：`n² ≤ ε_static`（物理约束）；`cavity_model = smd`
  必须同时给 `refractive_index` 与 `radii_set = smd`；张力/离子强度非负。
- 兼容性：阶段一的溶剂文件（没有这些键）逐字节仍通过；`solvent.schema.{json,yaml}` 已重新生成。

### K2 · 阶段二的结果块与选项（**新增可选字段/新变体**，非破坏）

- 新模块 `properties.rs`（K7）：`ExcitedStates` / `Transition` / `TransitionContribution` /
  `SumRules` / `NmrShieldings` / `NucleusShielding` / `NmrReference` / `Polarizability` /
  `Multipoles` / `ExcitedStateMethod`。
- 新模块 `correlation.rs`（K9）：`ScfType { rhf, rohf, uhf, rks, uks }`、`Mp2Kind { mp2, ri-mp2 }`、
  `Mp2Options`、`SpinResult`、`CorrelationResult`。
- 新模块 `solvation.rs`（K8）：`SolventModel { vacuum, cosmo, smd, pbf }`、`SolvationResult`。
- `JobResult` 新增 8 个**可选块**：`dispersion_energy_hartree` / `spin` / `correlation` /
  `solvation` / `excited_states` / `nmr_shieldings` / `polarizability` / `multipoles`，
  全部 `serde(default) + skip_serializing_if`：不设置就不出现在 `result.json` 里。
- `TaskOptions` 新增 4 个可选选项：`scf_type` / `mp2` / `excited_states` / `solvent_model`，
  并配套 builder（`with_scf_type` 等 —— `non_exhaustive` 结构体不允许下游写 struct literal）。
- `JobInput::validate` 新增**跨字段**校验（只在显式给新选项时生效）：SCF 类型 ↔ 方法族
  （HF 变体要求 `hf`）、开壳层 ↔ 多重度、MP2 只支持闭壳层参考、`solvent_model` ↔ 溶剂名。
- `SUPPORTED_FEATURES` 新增词汇：`method:mp2` / `method:tddft` / `solvent:cosmo|smd|pbf` /
  `numerics:ri|pseudospectral` / `property:excited-states|nmr|polarizability|multipoles`。
  **声明支持 ≠ 已实现**：后端仍然逐个判（见下条）。
- **引擎侧守卫**（`qwb-engine`）：`scf_type` 的开壳层变体、`mp2`、`excited_states`、
  非真空 `solvent_model` 一律 `Unsupported`（K5 退出码 4）并指向负责的 WP；
  与阶段一行为等价的写法（`rks`/`rhf`/`vacuum`）放过。
- **明确不改**：`JobType` 仍只有 `sp`/`opt`/`freq`（`ts`/`irc`/`scan` 要等 KV-05 与 RP 轨）。

### K3 · `result.schema.yaml`（**新增 `$defs` + JobResult 可选属性**，非破坏）

- 新增 `$defs`：`SpinResult` / `CorrelationResult` / `SolvationResult` / `ExcitedStates` /
  `Transition` / `SumRules` / `NmrShieldings` / `Polarizability` / `Multipoles`。
- `JobResult` 的 8 个新属性全部可选（不进 `required`）。
- **K2 结果块不走 schemars**：沿用 C-03 的约定（`engine-api` 保持零 schema 依赖，
  `result.schema.yaml` 是手写文档，由 `qwb-io/tests/k3_schema_docs.rs` 逐字段钉住）。
  阶段二原计划里写的"由 schemars 生成"对 K2 不成立，已按实际约定执行并记录在
  `docs/choices.md`。

## 1.2.0 — 2026-10-23（D 轨增量：K1 新增一个可选字段）

### K1 · `GridLevelDef.angular_rule`（**新增可选字段，非破坏**）
- 类型：枚举 `AngularRule { Lebedev, GaussProduct }`，序列化为 `lebedev` / `gauss-product`；
  `#[serde(default)]`，缺省 = `lebedev`。
- 动机：X-03 的验收要求 Lebedev 网格，但 Lebedev 点集**没有闭式生成法**，必须逐阶数制表，
  而仓库里还没有表。把"用哪条角向规则"写进数据（而不是在代码里替换），既保持契约诚实，
  又让 `lebedev` 可以在表到位后原样接入。
- 当前数据：四个等级都声明 `gauss-product`（Gauss–Legendre(cosθ) × 均匀 φ，
  `Σw = 4π` 精确成立）；`lebedev_order` 保留其含义（gauss-product 下作为角向点数预算）。
- 兼容性：旧数据（无此字段）默认走 `lebedev`，行为对已有文件不变；schema 已重新生成。
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
