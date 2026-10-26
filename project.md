# project.md · QuantumWorkbench v2.1

> 本文件是 **唯一开工依据**。agent 与人类开发者都按本文件执行。
> 不在本文件中的事项视为未授权，需人类决策后修订本文件再执行。
>
> **v2.1（2026-10-22）**：按 C 轨（C-01..C-06）实际落地的契约修订，并**清掉所有会让 agent
> 卡住的缺口**。变更清单见 §15。凡 v2.0 与本版冲突，以本版为准。

---

## 0. 五分钟上手（新会话从这里开始）

目标：做一个**模仿 Jaguar 的分子 DFT 软件**——产品面（项目/任务/结果/调度/复现 +
GUI/Python/CLI 同一 Task 模型）对齐 Jaguar，算法面**全部自研**。

> **阶段二已开工**：本文件描述的第一阶段（T1–T8）已于 2026-11-05 收口；下一阶段的规范是
> [`project-phase2.md`](./project-phase2.md)（v3.0，契约 K7–K10、里程碑 M1–M8、
> 含 stretch 条款）。本文件的 K1–K6 语义、§14 决策与全部验收**继续有效**，阶段二不得放宽。

```powershell
cd D:\python_II\Jaguar

# 1. 看状态：哪些 WP 已 done、下一个该做哪个
type docs\status.yaml

# 2. 跑全部门禁（一条命令）
powershell -NoProfile -File scripts\verify.ps1   # Windows 本地（含 mypy）
bash scripts/verify.sh                           # Linux / CI / WSL

# 3. 找下一个 WP：§6 中依赖全 done 且状态最小的那个
```

三条铁律：

1. **契约（§4 K1–K6）已冻结**，语义不得改；只允许**新增**（新字段可选、新变体、新方法）。
   真要改语义 → 走 §10.4 的升级流程，不要自己改。
2. **每完成一个 WP 立刻写 `docs/status.yaml`**（`state: done` + 验收命令输出摘要）。
3. **本文件与代码冲突时**：以**代码 + 本文件的 §4** 为准，并把差异在提交里同步修本文件
   （本版就是这么来的）。不要静默偏离。

---

## 1. 阅读顺序

| 你要做什么 | 看哪节 |
|---|---|
| 理解项目边界与"模仿 Jaguar"到什么程度 | §2 |
| 理解为什么这样排顺序 | §3 |
| 确认契约（已冻结，不要改） | §4（K1–K6） |
| 我要做哪个包 | §6（WBS，每行都有可执行验收） |
| 项目的"完成"刻度 | §7（垂直切片 T1–T8） |
| 数据层怎么做（本项目中心） | §8 |
| 验收怎么判 | §9 |
| agent 每步必须怎么做 | §10 |
| Rust 特有的坑 | §11 |
| 目录结构 / 命名 / 提交规范 | §12 |
| 本机环境事实与命令 | §13 |
| 已定决策台账（**别再重新讨论**） | §14 |
| v2.0 → v2.1 改了什么 | §15 |

---

## 2. 目标与非目标

### 2.1 目标

从零实现一个**分子 DFT 软件**：

- **引擎自研**：不调用 PySCF/ORCA/Gaussian/QE/Psi4/libxc/ase 等电子结构引擎或泛函库。
- **数据是中心**：基组、ECP、泛函、网格、溶剂参数的**获取、本地实现、验证、版本化**是核心工作。
- **工作流完整**：项目 / 分子 / 任务 / 结果 / 调度 / 复现。
- **界面完整**：GUI（Python + PySide6）+ Python API（pyo3）+ CLI（clap），共用同一套 Task 模型。
- **可复现是底线**：每个结果都能回溯到输入 + 数据快照 + 环境（`env.json` 必须含 Rust
  toolchain、`Cargo.lock` 的 sha256、crate 版本清单）。

### 2.2 与 Jaguar 的对齐范围（**v2.1 新增：回答"模仿到什么程度"**）

第一阶段**对齐 Jaguar 的产品面与方法集合**，**不复制它的算法内核**。

| Jaguar 的能力 | QWB 第一阶段 | 说明 |
|---|---|---|
| HF / LDA / GGA / 杂化 DFT | ✅ 做 | SVWN5、PBE、B88、LYP、B3LYP（§8.3） |
| ECP（LACVP 系） | ✅ 做（子集） | LANL2DZ + def2-ECP，元素 H–Kr |
| 单点 / 几何优化 / 频率 | ✅ 做 | T1/T2/T3/T4 |
| 偶极 / 布居电荷 / 轨道能级 | ✅ 做 | S-06（Jaguar 也报这些） |
| 溶剂（PBF/PCM） | ⚠ 只做接口 + 数据 | 引擎返回 `Unsupported`（§8.5） |
| 基组（6-31G*、cc-pVTZ、LACVP） | ✅ 做等价集合 | STO-3G、def2-SVP/TZVP、cc-pVDZ + ECP |
| 项目 / 任务 / 结果 / 调度 / 复现 | ✅ 做 | T7 |
| GUI 工作流（Maestro 那一层） | ✅ 做简化版 | T8，PySide6 + VTK |
| **pseudospectral（PS）积分** | ❌ 不做 | 见下 |
| **PBF / RISM 溶剂** | ❌ 不做 | 只留字段 |
| MP2 / CC / GVB / TDDFT / meta-GGA / range-separated | ❌ 不做 | §2.3 |
| 过渡态 / IRC / 势能面扫描 / NBO / EPR | ❌ 不做 | 第一阶段外 |

**为什么不模仿 PS 积分**：Jaguar 的 pseudospectral 方法把 Coulomb/exchange 放到网格上算，
是**另一套积分与误差分析体系**。本项目契约要求"每个数都能回溯 + 与解析/文献对照"，
而 PS 的误差来源（网格密度、去别名）会把 T1 首次端到端推远，并把 §9 的三层验证搅浑。
所以：**解析积分（Obara-Saika）+ 网格只用于 XC**，与 §6 的 I/X 轨一致。

### 2.3 非目标（第一阶段明确不做）

固体/周期、TS/IRC/scan、TDDFT、多参考、耦合簇、MP2、相对论（ECP 除外）、GPU、MPI 大规模、
meta-GGA、双杂化、range-separated、显式溶剂、QM/MM、SMD、pseudospectral 积分、PBF/RISM、
NBO/EPR。

### 2.4 允许使用的第三方库（冻结，v2.1 修订）

**引擎（Rust）**：

`serde`, `serde_json`, `serde_yaml`, `schemars`（**v2.1 批准**，C-01 定案）,
`nalgebra`, `faer`（可选后端）, `rayon`, `ndarray`（数据交换）, `tracing`,
`tracing-subscriber`, `anyhow`, `thiserror`, `clap`, `zip`, `sha2`, `hex`, `bytemuck`,
`wide`（portable SIMD）, `approx`（测试）, `proptest`, `criterion`, `pyo3`, `numpy`。

**GUI / 产品（Python）**：

`PySide6`, `VTK`, `rdkit`（**仅**结构读写）, `numpy`, `pydantic`（v2，P-01 需要）,
`pytest`, `hypothesis`, `mypy`（类型门禁）, `maturin`。

**明确不引入**：`h5py` / 系统 libhdf5（K4 定案）、`ndarray-npy`（R-14 定案不引入）、
`cargo-audit`（`cargo deny check advisories` 已覆盖）、任何电子结构引擎或泛函库。

**任何不在此列表且进入产品依赖树的库，需人类批准，并进 `qwb-core/deny.toml` 白名单。**

---

## 3. 组织原则

1. **契约先行，并行不返工**。K1–K6 已冻结（C 轨完成），数据/引擎/产品三轨可以并行。
2. **数据层随垂直切片增量补**。第一次只需要 H 与 STO-3G。
3. **引擎按垂直切片推进，不做层内串行**。第一次就跑通一条能量，把集成风险前移。
4. **产品轨先于引擎成熟开工**。Product 轨前期用 mock 数据。
5. **每个工作包都可独立验收**。没有"感觉差不多了"这一步；§6 每行都有可复制命令。
6. **agent 不自由发挥**；但**也不允许被模糊处卡死** —— 见 §10.3 的决策规则。
7. **Rust 特有约束**：
   - `#![deny(unsafe_code)]` 默认；只允许 3 类 crate 例外：FFI（BLAS）、SIMD intrinsics
     （`wide` 已封装）、获批的热点优化。例外需 `// SAFETY:` 注释 + 人类批准 + 进
     `scripts/unsafe_allowlist.txt`。
   - **`Cargo.lock` 入 git**，禁止无 PR 记录 `cargo update`。
   - **`cargo fmt --check` / `cargo clippy -- -D warnings` 必须全绿**才能合入。
   - **workspace 层级不得引入循环依赖**。
   - **三方依赖只能出现在 `qwb-core/Cargo.toml` 的 `[workspace.dependencies]`**
     （`scripts/check_deps_centralized.sh` 强制）。

**风险优先级**：集成风险 > 数学正确性风险 > 数据正确性风险 > 性能。

---

## 4. 契约 K1–K6（**已冻结**；v2.1 = 实际落地形态）

> 契约 crate / schema 文件已存在于仓库；本节签名与字段**与代码逐项一致**（有测试强制）。
> 变更只允许新增，语义不得改。任何变更需 `CHANGELOG.md` + 人类评审 + 修订本节。

### K1 · 数据 Schema

**路径**：`qwb-contracts/schema/`（生成物）、`qwb-core/crates/qwb-registry/src/*.rs`（真相源）

**生成方式（已定案）**：`schemars` 从 Rust 类型生成，`BasisSet` 等类型即唯一真相源。
生成器：`cargo run -p qwb-registry --bin qwb-gen-schemas -- <dir> [--check]`。
每类产出 `<kind>.schema.json`（工具读）+ `<kind>.schema.yaml`（人读，K1 规定文件名）。

**7 类**：`basis`、`ecp`、`functional`、`grid`、`solvent`、`molecule`、`registry`
（枚举 `qwb_registry::verify::DataKind`，顺序即生成顺序）。

**共同字段**（在 5 个数据集类型里内联声明；`schema.rs` 的测试强制 5 处定义逐字节一致）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `String` | 数据键名（数据目录下的标识） |
| `version` | `String` | 独立 semver；变更即 bump |
| `citation` | `Vec<Citation>` | **required**，≥1 条；每条 `title` + `doi`/`url` |
| `license` | `License` | **required**；`spdx`、`commercial_ok`、`redistribute_ok` |
| `created` | `String` | ISO 8601 UTC |
| `sha256` | `Option<String>` | 工具生成（规则见下） |
| `notes` | `Option<String>` | 备注/来源说明 |

**规范化 sha256 规则**（唯一实现 `qwb_registry::sha256`）：解析 YAML → 删顶层 `sha256`
→ 转 JSON（键字典序）→ 对紧凑 JSON 取 SHA-256（64 位小写 hex）。
**与排版/注释/键序无关**，改一个数字必换摘要。`qwb-stamp-data` 打戳（会丢 YAML 注释，
所以说明写进 `notes`）。

**各类型特有字段**（与代码一致）：

- `BasisSet`：`elements: Vec<String>`、`angular: spherical|cartesian`、
  `normalization: l2|none`、`role: orbital|ri-j|ri-jk|ecp-fit`、
  **`shells: Vec<ShellBlock>`**（`ShellBlock { element, shells: Vec<Shell> }`，
  每元素一块，必须与 `elements` 一一对应）。
  原因：一个基组文件可覆盖多元素（D-03 的 STO-3G 覆盖 H/C/N/O/F），平坦 `shells[]` 无法区分元素。
- `Shell`：`l: u32`、`exponents: Vec<f64>`（Bohr⁻²，大→小）、
  **`coefficients: Vec<Vec<f64>>`**（每条内层 = 一个收缩函数，长度 == `exponents.len()`）、
  `contraction_type: segmented|general`。
  原因：def2-TZVP 的通用收缩（D-05）无法用一维系数表达。
- `Ecp`：`element`、`n_core_electrons`、`z_eff`（必须 == Z − n_core）、
  `r_power_convention: "r^n"|"r^(n-2)"`、`local_part{id,terms[{power,exponent,coefficient}]}`、
  `nonlocal_parts[{l,id,terms[]}]`。
- `Functional`：`components[{id,weight}]`、`requires{rho,sigma,laplacian}`、
  `parameters: BTreeMap<String,f64>`、`stability{density_floor,sigma_ceiling}`。
- `GridLevelDef`：`level: coarse|normal|fine|veryfine`、`radial_points`、`lebedev_order`、
  `becke_sharpness`、`radii_scale?`、`cutoff`（Bohr）。
- `Solvent`：`epsilon`、`temperature_K`、`atomic_radii: BTreeMap<符号,f64>`（Å）、
  `cavity_model: none|pcm|smd`、`scale`。
- `MoleculeFile`（用户输入，**不受 G2 许可门禁**）：`name`、`notes?`、
  `atoms[{symbol,position_angstrom}]`、`charge`、`multiplicity`、`citation?`、
  `created`、`sha256?`。校验含电子数奇偶与多重度一致、坐标重合检测。
- `RegistryManifest`：`schema_version`（必须 == `K1_VERSION`）、`name`、`version`、
  `created`、`features: Vec<String>`、
  5 个条目表 `bases/ecps/functionals/grids/solvents`，每条 `{name, path, sha256}`。

**校验代码（机器可读）**：`yaml.parse`、`schema.deserialize`、`sha256.compute`、
`sha256.mismatch`、`g2.citation`、`g2.license`、`g2.sha256`、`semantic`、`registry.*`。
`verify` 与 `load` 的语义分工见 §8.6。

**Rust 实现约定**：所有类型 `#[serde(deny_unknown_fields)]` + `#[non_exhaustive]`；
反序列化拼错字段必须报错。

### K2 · 引擎 Rust API（`qwb-contracts/engine-api`，crate `qwb-engine-api` 0.1.0）

```rust
//! qwb-engine-api: 冻结的引擎接口。任何破坏性变更需经 CHANGELOG。
#![deny(unsafe_code)] #![deny(missing_docs)]

pub const SUPPORTED_FEATURES: &[&str];   // "<域>:<值>" 词汇表，见下

pub struct Atom { pub symbol: String, pub position_angstrom: [f64; 3] }        // non_exhaustive
pub struct Molecule { pub atoms: Vec<Atom>, pub charge: i32, pub multiplicity: u32 }
pub enum GridLevel { Coarse, Normal, Fine, VeryFine }   // Default = Normal；as_str()
pub enum JobType { Sp, Opt, Freq }                       // as_str()
pub struct TaskOptions {
    pub scf_max_iter: u32,                       // 128
    pub scf_energy_threshold_hartree: f64,       // 1e-8
    pub scf_density_threshold: f64,              // 1e-6
    pub diis_history: u32,                       // 8
    pub threads: u32,                            // 0 = auto
    pub max_opt_steps: u32,                      // 100
    pub opt_gradient_threshold_hartree_per_bohr: f64,  // 4.5e-4（T3 门禁值）
    pub freq_step_angstrom: f64,                 // 0.01
    pub ecp_name: Option<String>,
}
pub struct JobInput { pub molecule: Molecule, pub basis_name: String,
    pub method_name: String, pub grid_level: GridLevel, pub solvent: String,
    pub jobtype: JobType, pub options: TaskOptions }
pub struct JobResult {
    pub energy_hartree: f64,
    pub gradient_hartree_per_bohr: Option<DMatrix<f64>>,   // 3N×1
    pub hessian_hartree_per_bohr2: Option<DMatrix<f64>>,  // 3N×3N
    pub dipole_debye: Option<DVector<f64>>,
    pub mo_energies_hartree: Option<DVector<f64>>,
    pub mo_coeffs: Option<DMatrix<f64>>,
    pub scf_converged: bool,
    pub data_snapshot: DataSnapshot,   // path -> sha256
    pub log: String,
    pub env: EnvInfo,                  // rustc, Cargo.lock sha256, 数据版本, threads, os
}
pub struct DataSnapshot(BTreeMap<String, String>);   // serde(transparent)
pub struct EnvInfo { pub engine_version, rustc_version, cargo_lock_sha256: String,
    pub data_snapshot: DataSnapshot, pub threads: u32, pub os, created_at_utc: String,
    pub toolchain_channel: Option<String>, pub crate_versions: BTreeMap<String, String> }
pub struct Registry { root: PathBuf, declared_features: BTreeSet<String> }  // 数据根 + 能力声明

pub enum EngineError {
    InvalidInput(String),                              // K5 退出码 1
    Data(String),                                      // 3
    ScfNotConverged { iters: u32, max: u32 },          // 2
    Unsupported(String), Io(..), Internal(String),      // 4
}
impl EngineError { pub fn exit_code(&self) -> i32 }    // 上表，单测钉死

pub trait Reporter: Send + Sync { fn progress(&self, stage: &str, pct: f64);
    fn log(&self, line: &str); fn is_cancelled(&self) -> bool }
pub struct NullReporter;                               // 什么都不做

pub trait EngineBackend: Send + Sync {
    fn run(&self, input: &JobInput, reporter: &dyn Reporter) -> Result<JobResult, EngineError>;
    fn request_cancel(&self);
}
pub struct Engine { /* opaque */ }
impl Engine {
    pub fn new(registry: &Registry) -> Result<Self, EngineError>;  // 契约 crate 无后端 ⇒ 恒 Unsupported
    pub fn with_backend(registry: &Registry, backend: Box<dyn EngineBackend>)
        -> Result<Self, EngineError>;
    pub fn registry_root(&self) -> &Path;
    pub fn run(&self, input: &JobInput, reporter: &dyn Reporter) -> Result<JobResult, EngineError>;
    pub fn request_cancel(&self);
}
```

**冻结约定**：

- 公共 struct/enum 一律 `#[non_exhaustive]`；因为 `non_exhaustive` 禁止跨 crate 的
  struct literal，**所有类型都提供 `new(..)` 与 `with_*` 建造方法**（API 的一部分，不是可选项）。
- 公共字段后缀 `_hartree`/`_bohr`/`_debye`/`_angstrom`，禁止裸 `energy`。
- 错误 enum 只允许新增变体（`#[non_exhaustive]`），禁止删除；新变体必须进 `exit_code()`。
- 所有跨线程类型 `Send + Sync`（有静态断言测试）。
- `Engine::run` 接受 `&dyn Reporter`，**引擎不得直接写 stdout**。
- **`Registry` 在契约 crate 里只是"数据根 + 本次计算声明的 feature 集合"的句柄**，
  不解析数据文件（加载属 D-01）。`Engine::new`/`with_backend` 校验：
  数据根不存在 → `Data`；声明了 `SUPPORTED_FEATURES` 之外的 feature → `Unsupported`。
- **矩阵 JSON 表示 = list-of-list**（K4 二选一定案）；由 engine-api 显式 serde 适配，
  不依赖 nalgebra 私有布局；`None` 字段**省略**而非写 `null`。
- `SUPPORTED_FEATURES`（第一阶段词汇表）：
  `jobtype:sp|opt|freq`、`basis:orbital|ri-j|ri-jk|ecp-fit`、
  `functional:lda|gga|hybrid`、`grid`、`solvent:vacuum|implicit`、`ecp`。
  表外 feature（如 `functional:meta-gga`、`solvent:smd`）→ `Unsupported`。
  **"声明支持 ≠ 本构建实现"**：后端还会再判一次。

### K3 · Task / Result Schema（产品面）

**路径**：`qwb-contracts/studio/task.schema.yaml`、`result.schema.yaml`
（手写文档；与 Rust 类型的一致性由 `qwb-core/crates/qwb-io/tests/k3_*.rs` 强制）

**Task 状态机（v2.1：6 条转移，冻结）**：

```
draft → queued → running → converged
           │        ├→ failed
           └→ cancelled
                    └→ cancelled
```

| from | to |
|---|---|
| `draft` | `queued` |
| `queued` | `running` |
| `queued` | **`cancelled`**（v2.1 新增：取消尚未开始的任务，P-03 需要） |
| `running` | `converged` |
| `running` | `failed` |
| `running` | `cancelled` |

- 终态：`converged` / `failed` / `cancelled`，**没有出边**。
- **重试语义（冻结）**：`failed` **不回到** `queued`；重试 = 用**新 task id** 重建任务
  （保证结果目录不可变、可追溯）。不要再加 `failed→queued`。
- `TaskState::LEGAL`（Rust）是该表的唯一真相源；`task.schema.yaml` 的 `x-state-machine`
  与 proptest 都对齐它。

**结果目录布局（冻结）**：

```
<task_id>/
  input.yaml          # JobInput（K2 类型 ≡ K4 格式）
  env.json            # rustc, cargo.lock sha256, data_snapshot, threads, OS, toolchain, crate 版本
  log.txt
  result.json         # JobResult 序列化（矩阵 list-of-list）
  data_snapshot.json  # 与 env/result 里的 data_snapshot 同内容
  artifacts/
```

**没有 `env.json` 视为不完整结果**，`qwb inspect` 退出码 **3**（G5）。

### K4 · 文件格式

**详细定义**：`qwb-contracts/studio/qwb-format.md`（含全部结构约束）。

- **输入**：YAML，形态 ≡ `qwb_engine_api::JobInput`（不允许第二套字段）。
- **项目容器 `.qwb`**：**Zip**（`zip` crate，**只启用 `deflate`**；不用 HDF5）。
  内部：`meta.json`（第一位）+ `tasks/<id>/{input.yaml,result.json,env.json,log.txt}`
  + `data_snapshot.json` + `artifacts/**`。
- **结构性约束（必须拒绝，不是"约定"）**：绝对路径/盘符/反斜杠、`..`、空路径段、
  目录条目、重复条目、缺 `meta.json`、`meta.task_count` ≠ `tasks/` 目录数、截断的 zip。
- **字节可复现**：条目时间戳固定 1980-01-01T00:00:00、Deflate、条目顺序固定 ⇒
  相同内容导出**逐字节相同**（有测试）。这支撑 G5 的 replay 比对。
- **矩阵**：list-of-list（见 K2）。大矩阵二进制快通道（`.npy`/`ndarray-npy`）
  **不引入**（R-14 已定案）。
- **分子**：`.xyz`/`.mol`/`.sdf`，Python 侧 rdkit 只做结构读写；引擎不解析。

### K5 · CLI 契约（`qwb-core/crates/qwb-cli`，二进制 `qwb`）

```
qwb run    input.yaml [--out DIR] [--threads N]
qwb batch  PATTERN|DIR [--parallel N]          # PATTERN 形如 DIR/*.yaml
qwb inspect PATH                                # 结果目录 或 .qwb 容器
qwb replay PATH [--dry-run]
qwb data list
qwb data show NAME
qwb data verify [PATH...]                       # 省略则校验整个注册表
```

**退出码（冻结）**：`0` 成功 / `1` 输入错误 / `2` 收敛失败 / `3` 数据校验失败 / `4` 内部错误。
映射唯一实现在 `EngineError::exit_code()`；容器/布局错误算 `3`，`Io` 算 `4`。

**数据根定位**：`QWB_DATA_ROOT` 环境变量，或从当前目录向上依次找 `qwb-data`。
**不加新命令行参数。**

**各子命令在"引擎未接入"阶段的真实行为**见 `qwb-contracts/studio/cli.md`：C 轨
`run`/`batch` 完整解析校验输入后返回 `4`（**不伪造结果**）；`inspect`/`replay --dry-run`/
`data *` 完全实装。

### K6 · Python API 契约

**路径**：`qwb-contracts/studio/python/qwb_engine.pyi`（已冻结，`mypy --strict` 0 issue）

**构建**：`pyo3` + `maturin`；ABI **`abi3-py39`**（R-12 已定案）。

```python
eng = qwb.Engine(registry_path="qwb-data/", features=["jobtype:sp"])
result = eng.run(job_input, reporter=None)
result.energy_hartree                # float
result.gradient_hartree_per_bohr     # np.ndarray | None
result.data_snapshot.items()         # list[tuple[str, str]]
result.env.rustc_version             # str
qwb.EngineError / InvalidInputError / ScfNotConvergedError / DataError / UnsupportedError
    .exit_code                       # 与 K5 一致
```

**硬约定**：

- **GIL 释放**：`Engine.run` 内部必须 `py.allow_threads`；CI 用 concurrency 测试强制。
- **矩阵零拷贝**：初期全部拷贝（< 500×500 可接受），接口不变（R-18）。
- **`Reporter` 回调**：Rust 侧批量推送，禁止逐行跨 GIL 回调。
- 桩文件里的 `__init__` 签名就是 P-06 必须实现的签名（`registry_path` 为关键字）。

---

## 5. 依赖 DAG

```
K1..K6 (契约) ✅ 已冻结
   │
   ├──> D-tracks (数据)            ─┐
   ├──> I/X/S-plumbing (引擎骨架)   ─┤
   └──> P-scaffold (产品骨架)       ─┤
                                      ▼
                                T1 集成（H2/STO-3G/SVWN5 单点能）
                                      ▼
                                T2（H2O/def2-SVP/PBE + 性质）
                                      ▼  T3（梯度+优化）→ T4（频率）
                                      ▼  T5（ECP）   → T6（B3LYP）
                                      ▼
                                T7（工作流+调度+复现）→ T8（GUI）
```

**关键规则**：

- K1–K6 已冻结（C 轨 `done`）；D/I/X/S/P 可并行开工。
- `qwb-linalg`（I-00）必须在 I-01 前冻结接口，让 `nalgebra`/`faer` 后端可切换。

---

## 6. WBS 工作包

> 每个包：`ID · 依赖 · 产物 · 验收`。验收必须是**可复制执行的命令**或**可检查断言**。
> 状态记录在 `docs/status.yaml`。**C 轨已全部 done，保留其验收以便回归。**

### 6.0 状态总览（v2.1）

| 轨道 | 状态 |
|---|---|
| C（契约） | **C-01..C-06 全 done** |
| D（数据） | 全部 pending（数据仓只有 `example-` 样例，**非交付物**） |
| I / X / S / G / F | 全部 pending |
| P（产品） | 全部 pending |
| V（验证） | 全部 pending |

### 6.1 契约轨（C 轨）— **done**

| ID | 产物（实际） | 验收（可回归） |
|---|---|---|
| C-01 ✅ | `qwb-registry` 的 K1 类型 + schemars 生成的 7×2 schema；`qwb-data/examples/` 6 个样例 | `cargo run -p qwb-gen-schemas -- ../qwb-contracts/schema --check` 无漂移；`qwb data verify` 通过；缺 citation 的文件 → 退出码 3 且 code `g2.citation`（v2.0 的"`data verify` 属 D-02"矛盾已消除：C-01 只要最小 verify，D-02 扩成回归集） |
| C-02 ✅ | `qwb-contracts/engine-api`（`qwb-engine-api` 0.1.0，23 tests） | `cargo clippy --all-targets -- -D warnings`；`cargo doc --no-deps`（`RUSTDOCFLAGS=-D warnings`）；`Engine::new` 返回 `Unsupported`（单测） |
| C-03 ✅ | `task.schema.yaml` + `result.schema.yaml` + `qwb-io::task` 状态机（6 转移） | proptest 覆盖 6×6 转移关系与 `LEGAL` 精确相等；schema 文档与 Rust 类型逐字段一致 |
| C-04 ✅ | `qwb-format.md` + `qwb-io::container` + `studio/examples/example.qwb` | `cargo test -p qwb-io --test example_container`；容器导出字节可复现（单测） |
| C-05 ✅ | `qwb-cli`（K5）+ `studio/cli.md` + `python/qwb_engine.pyi` | 全部子命令 `--help` 退出 0；`mypy --strict` 桩 + typed consumer 均 0 issue |
| C-06 ✅ | `qwb-core` workspace + 根 `rust-toolchain.toml`（1.97.1）+ `deny.toml` + `.github/workflows/ci.yml` + `scripts/*.sh` | `cargo fmt --all --check`；`cargo clippy --workspace --all-targets -- -D warnings`；`cargo test --workspace`；`cargo deny check`（四类全 ok）；三个门禁脚本退出 0 |

### 6.2 数据轨（D 轨）— 项目中心（详细要求见 §8）

| ID | 依赖 | 产物 | 验收（可执行） |
|---|---|---|---|
| D-01 | C-01, C-06 | `qwb-registry` 加载器：`Registry::open(root)`、`get_basis/get_functional/get_ecp/get_grid/get_solvent(name)`、`load_pair(basis, ecp)`、`features()` | ① **空注册表必须能 `open`**（`Ok`，0 条目 —— 否则工具链冷启动不可用；"0 条目"只是 `data verify` 的错，不是 load 的错）；② 3 个畸形文件各返回**字段级** `Data` 错误；③ `load_pair` 元素不匹配 → `Data` |
| D-02 | D-01 | `qwb data verify` 回归集（6 坏 / 6 好） | 6 个坏文件分别对应：缺 `citation`、`citation: []`、缺 `license`、`sha256` 不符、拼错字段、`exponents` 非降序 → 各退出码 3，code 分别为 `schema.deserialize` / `g2.citation` / `schema.deserialize` / `sha256.mismatch` / `schema.deserialize` / `semantic`；6 个好文件退出码 0 |
| D-03 | D-01 | `STO-3G`（H,C,N,O,F）正式数据 | ① 与 BSE 导出逐项 diff < 1e-12（相对）；② 每元素基函数数 H=1、C/N/O/F=5；③ 球谐；④ 至少 3 元素的指数/系数贴进 PR body 供人核对 |
| D-04 | D-01 | `def2-SVP`（H,C,N,O） | 与 BSE 逐项 diff < 1e-12；**每元素基函数数写进测试断言**（以 BSE 导出为准，不靠记忆） |
| D-05 | D-01 | `def2-TZVP`（H,C,N,O） | 同上；**必须含通用收缩**（一个 shell 多收缩列） |
| D-06 | D-01 | `cc-pVDZ`（H,C,N,O） | 同上 |
| D-07 | D-01 | `LANL2DZ` + ECP（先 Cu、Zn，再 Na–Kr） | `Registry::load_pair("lanl2dz","lanl2dz")` 通过；`n_core_electrons` 与基组价电子一致；`z_eff == Z − n_core`（已有校验器） |
| D-08 | D-01 | `def2-ECP`（先 K–Kr） | 与 D-04/05/06 配对校验通过；`r_power_convention` 显式 |
| D-09 | D-01 | `SVWN5`（含 `dε/dρ`、`dε/dσ`） | X-02 有限差分导数 < 1e-9；参数逐个带 citation |
| D-10 | D-01 | `PBE`（交换+相关，κ=0.804、μ=0.21951） | 同 D-09 |
| D-11 | D-01 | `B88`、`LYP` | 同 D-09 |
| D-12 | D-01 | `B3LYP` 复合定义（0.8 B88 + 0.2 HF + 0.81 LYP + 0.19 VWN5） | 与 V-03 基准比对 < 2 mHa（水/苯/CO） |
| D-13 | D-01 | 网格等级 `coarse/normal/fine/veryfine` | 初始参数 `(radial,lebedev,sharpness)`：`(50,194,0.6)`、`(75,302,0.6)`、`(99,590,0.6)`、`(140,974,0.8)`，`cutoff=20 Bohr`；相邻等级能量差 < 0.1 mHa（T2 分子上断言） |
| D-14 | D-01 | 溶剂数据（8 种） | schema 通过；ε 与 NIST 一致（水 78.3553 @ 298.15 K）；半径单列并注明来源；`cavity_model` 显式 |
| D-15 | D-02..D-14 | `scripts/gen_data_licenses.py` + `DATA_LICENSES.md` + 每数据 sha256 | 脚本可重跑且输出幂等；所有数据带 license/citation/sha256 |

### 6.3 积分轨（I 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| I-00 | C-06 | `qwb-linalg` 抽象层（`nalgebra` 默认，`faer` 可选 feature） | 两后端都编译；同一接口 roundtrip identical；`cargo bench` 有 baseline（V-06） |
| I-01 | C-02, D-01 | 基组加载器 → 内部 AO（`BasisSet`） | H2/STO-3G AO 数 = 2；C/STO-3G AO 数 = 5；`angular` 只从数据读 |
| I-02 | I-01 | 1e 积分（重叠、动能、核吸引） | H2/STO-3G 的 S、T、V 与**解析闭式**（写进测试）逐项 < 1e-10 |
| I-03 | I-01 | 2e 积分（**Obara-Saika 为主**，R-03 已定案） | H2/STO-3G `(ss|ss)` 与解析值 < 1e-10；**另实现一条 Rys/数值路径交叉验证**（同值 < 1e-10），保证数值独立性 |
| I-04 | I-03 | 2e 并行化（`rayon`，确定性归约） | 1 线程 vs 4 线程结果 **byte-identical**（对积分张量取 sha256 比对） |
| I-05 | I-02 | ECP 矩阵元（局域 + 非局域） | 与数值积分对照 < 1e-8；Cu 的 `l=2` 非局域项单独断言 |
| I-06 | I-03 | 积分缓存（可选 feature） | 开/关结果 identical |

### 6.4 XC 轨（X 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| X-01 | C-02, D-01 | `trait FunctionalTerm: Send + Sync`（`energy_density`、`potential_derivs`、`spin_scaling`、`requires`），**批量切片输入** | 单测 + doc test；批量接口在 1e6 点上无逐点分配（`criterion` 粗测） |
| X-02 | X-01 | 有限差分导数校验工具 | D-09/10/11 的 `dε/dρ`、`dε/dσ` 与数值差 < 1e-9（CI 必跑） |
| X-03 | X-01, D-13 | Becke 分区 + Lebedev 网格（`rayon` 并行生成） | 均匀电子气极限通过；水 Σw 与 4π 差 < 1e-10；**单线程与并行 identical** |
| X-04 | X-03 | 网格管理（等级映射 + 缓存） | 切换等级释放旧网格（`Arc::strong_count` 断言） |
| X-05 | X-02, X-04 | 泛函求值集成到 SCF | T1 单点能路径打通 |

### 6.5 SCF 轨（S 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| S-01 | I-02, I-03 | 密度矩阵 → Fock 构建（无 XC） | 与直接矩阵乘对照 < 1e-12 |
| S-02 | S-01, X-05 | RKS SCF（Fock + XC） | H2/STO-3G/SVWN5 收敛到 1e-8 Ha（T1 前置） |
| S-03 | S-02 | DIIS + 收敛判据 | H2O/PBE 收敛到 1e-8；不收敛时返回 `ScfNotConverged{iters,max}` |
| S-04 | S-02 | UKS（开壳层） | O2 三重态跑通；⟨S²⟩ 与理论值差 < 1e-6 |
| S-05 | S-03 | SCF 检查点/重启（serde） | kill 后从 checkpoint 续，结果 identical |
| **S-06** | S-03 | **性质：偶极矩、Mulliken/Löwdin 布居、轨道能级**（`qwb-props`；v2.1 新增，补上 §12.1 里无主的 crate） | ① H2O/STO-3G/SVWN5 偶极与实验 1.85 D 差 < 0.15 D；② Mulliken 与 Löwdin 电荷各自求和 == 总电荷（1e-10）；③ 写进 `JobResult.dipole_debye` / `mo_energies_hartree`；④ 两次运行 byte-identical |

### 6.6 梯度 / 优化轨（G 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| G-01 | S-03 | 解析梯度（Hellmann-Feynman + 基组响应 + XC 网格项） | 与数值梯度最大分量差 < 1e-6 Ha/Bohr（H2O/def2-SVP/PBE 与 H2/STO-3G/SVWN5） |
| G-02 | G-01 | BFGS + 冗余内坐标 | H2O 优化收敛：最大力 < 4.5e-4 Ha/Bohr；键长误差 < 0.01 Å（实验 0.9572 Å） |
| G-03 | G-02 | 多步轨迹 + 收敛报告 | H2O、CH4、NH3 三个分子优化全部收敛 |

### 6.7 频率轨（F 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| F-01 | G-01 | 数值 Hessian（中心差分，`rayon` 并行位移） | H2O/def2-SVP/PBE Hessian 与文献/更高精度差 < 5%（相对 Frobenius 范数） |
| F-02 | F-01 | 简正模 + IR 强度 + 热化学 | H2O IR 峰位与实验差 < 30 cm⁻¹（PBE 尺度）；ZPE、S、G 为正且量级正确；3N-6 全实（线性 3N-5） |

### 6.8 产品轨（P 轨）— Python + maturin

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| P-01 | C-03 | `qwb-studio/qwb_studio/models`：`Project/Molecule/Task/Result`（pydantic v2） | 单测通过；YAML/JSON roundtrip；非法状态转移抛 `ValueError`；与 `task.schema.yaml` 字段一一对应（有测试比对） |
| P-02 | P-01, C-04 | `.qwb` 读写（Python `zipfile`） | 与 Rust `QwbContainer` **互操作**：Rust 写→Python 读、Python 写→Rust 读；损坏 zip 明确报错 |
| P-03 | P-01, C-05 | 本地调度器（进程池 + 队列 + 状态机 + 取消 + kill -9 恢复） | 10 个 mock 任务并发；取消 `queued` 与 `running` 都生效（对应 K3 两条 cancelled 边）；进程被杀后重启能恢复 `queued` |
| P-04 | P-01, C-02 | `env.json` 生成器（`rustc --version`、`Cargo.lock` sha256、数据快照、threads、os、crate 版本） | 缺 `env.json` 时 `Result` 构造失败；生成物能被 `EnvInfo::validate` 接受 |
| P-05 | P-01 | Python 侧 `qwb` 入口（shell 转发到 Rust CLI） | 全部子命令 `--help` 通过；退出码与 K5 一致 |
| P-06 | C-02, P-04 | `qwb_engine` pyo3（maturin，`abi3-py39`） | 与 `qwb_engine.pyi` 一致：`mypy --strict` 通过；并发测试证明 GIL 被释放 |
| P-07 | P-05, P-06 | `qwb run` 端到端 | 跑通 T1 |
| P-08 | P-03, P-04 | `qwb replay` | 已跑结果 replay 后能量差 < 1e-10（`--dry-run` 已可用） |
| P-09 | P-07, P-08 | GUI 骨架（PySide6：项目树 + Task Panel + Job Monitor；显示能量/偶极/电荷） | 创建/提交/查看 mock 任务不阻塞 UI；大日志（>100 MB）滚动不卡 |
| P-10 | P-09, G-02, F-02 | GUI 3D（VTK：分子、轨道、ESP、振动） | 打开 H2O 结果可视化正确；ESP 用 S-06 的电荷着色 |

### 6.9 验证轨（V 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| V-01 | S-02 | 氢原子解析解测试 | 能量 −0.5 Ha 误差 < 1e-6；SCF 收敛到 1e-8 |
| V-02 | G-01 | 有限差分梯度一致性测试 | 基准分子全自动跑，全部 < 1e-6 Ha/Bohr |
| V-03 | S-02..G-02 | 基准集 + `tests/reference/*.csv`（**外部引用**，见 §7.0） | H2、H2O、CH4、NH3、CO、N2、苯 的 CSV；每行带 `source_doi` 或 `source_url`；CI 断言与库值的差 |
| V-04 | 全部 | CI 分层（PR / 每日 / 每周） | `scripts/verify.sh`（或 `verify.ps1`）全绿；PR 跑单元 + 小基准；每日 T3–T6；每周完整基准 + criterion 归档 |
| V-05 | 全部 | `docs/error-budget.md` | 每类误差来源（基组、网格、SCF、积分、ECP）与量级；与 §7 容差一致 |
| V-06 | I-03, X-03, S-03 | `criterion` 基准套 `benches/` | 每次发布前跑；回归超 20% 报警（软门禁） |

---

## 7. 垂直切片里程碑 T1–T8

> 每个 T 是"项目真正向前一步"的刻度。只有全部前置 WP 通过、且该 T 端到端命令通过才算完成。

### 7.0 参考值政策（v2.1 新增，**消除"文献值查不到就卡死"**）

所有"与文献对照"的断言都从仓库内的 `qwb-core/tests/reference/*.csv` 读
（下文简写为 `tests/reference/`），格式：

```csv
system,basis,method,jobtype,energy_hartree,source_doi,source_url,note
H2,sto-3g,svwn5,sp,-1.137,10.1000/xxxx,https://cccbdb.nist.gov/,CCCBDB
```

规则：

1. **优先外部引用**：NIST CCCBDB、原始文献 DOI。CSV 必须有 `source_doi` 或 `source_url`。
2. **允许的 fallback（仅当外部值确实取不到）**：用**数值独立路径**交叉验证
   （如 I-03 的 Rys 路径 vs Obara-Saika、`faer` vs `nalgebra`、单线程 vs 多线程），
   在 CSV `note` 写"内部独立路径，非文献值"，并在 `docs/error-budget.md` 登记。
   这是**允许推进**的路径，不算降级验收。
3. 禁止：拿自己算的值回填当"文献值"且不注明。

### T1 · 最小端到端：H2 / STO-3G / SVWN5 单点能

**前置 WP**：C-01..C-06 ✅、D-01..D-03、D-09、D-13、I-00..I-04、X-01..X-05、S-01..S-03、P-01..P-07、V-01

**验收命令**：`qwb run examples/h2_sto3g_svwn5.yaml`（`qwb-core/examples/` 下）

**断言**：
- SCF 收敛到 1e-8 Ha；`JobResult.scf_converged == true`
- 能量与 `tests/reference/h2_sto3g_svwn5.csv` 差 < 1 mHa
- `env.json`、`data_snapshot.json` 存在且通过 `EnvInfo::validate`
- `qwb replay <task_id> --dry-run` 通过；完整 replay 后能量差 < 1e-10
- `scripts/verify.sh` 全绿

### T2 · 主流计算：H2O / def2-SVP / PBE 单点能 + 性质 + 网格收敛

**前置 WP**：T1 + D-04、D-10、D-13、S-06

**断言**：
- 能量与 `tests/reference/h2o_def2svp_pbe.csv` 差 < 1 mHa
- 相邻网格等级能量差 < 0.1 mHa
- 单线程与 4 线程结果 **byte-identical**（G7）
- 偶极与实验 1.85 D 差 < 0.15 D；Mulliken/Löwdin 电荷和 == 总电荷

### T3 · 几何优化

**前置 WP**：T2 + G-01..G-03

**断言**：最大力 < 4.5e-4 Ha/Bohr；H2O 键长 0.9572 Å 误差 < 0.01 Å；解析梯度 vs 数值梯度 < 1e-6 Ha/Bohr。

### T4 · 频率

**前置 WP**：T3 + F-01、F-02

**断言**：3N-6 全实（线性分子 3N-5）；H2O 的 IR 峰位与实验差 < 30 cm⁻¹；ZPE > 0、S > 0、G 量级正确。

### T5 · ECP + 重元素

**前置 WP**：T2 + D-07、D-08、I-05 + K2 已支持 `TaskOptions.ecp_name`

**被测体系（冻结，避免歧义）**：CuH / LANL2DZ 价基 + LANL2DZ ECP；ZnO / def2-SVP + def2-ECP。

**断言**：
- Cu `n_core_electrons == 28`、`z_eff == 1.0`，与基组价电子一致
- 两个体系各跑通单点能；ZnO 与 `tests/reference/zno_def2svp_pbe_ecp.csv` 差 < 5 mHa
- 非局域 ECP 矩阵元与数值对照 < 1e-8

### T6 · 杂化泛函 B3LYP

**前置 WP**：T2 + D-11、D-12 + K 矩阵（S-02 扩充）

**断言**：水/苯/CO 总能量与基准差 < 2 mHa；HF 交换混合系数 0.20 与定义一致；B3LYP 梯度有限差分一致（< 1e-6）。

### T7 · 工作流 + 调度 + 复现

**前置 WP**：T1–T6 + P-01..P-08

**验收命令**：`qwb batch jobs/*.yaml --parallel 4` && `qwb replay <task_id>`

**断言**：
- K3 状态机 6 条转移全部由 proptest 覆盖（含两条 `cancelled` 边）
- `kill -9` 后重启恢复 `queued` 任务（P-03）
- 结果目录无 `env.json` 时 `qwb inspect` 退出码 3
- replay 后能量差 < 1e-10

### T8 · GUI 端到端

**前置 WP**：T7 + P-09、P-10

**断言**：
- 打开项目 → 建分子 → 选 PBE/def2-SVP/opt → 提交 → Job Monitor → 3D 结果，全程不崩溃不阻塞
- 大日志（> 100 MB）UI 不卡
- 崩溃后重启恢复未完成视图

---

## 8. 数据层详细要求（项目中心）

### 8.1 基组

**来源**：BSE 导出 + 一手文献核对。每个基组追溯原始 DOI，并核对 **H、C、N、O 至少 3 个元素**
的指数/系数。**不得**从其他项目代码仓库复制文件（BSE 网站导出可以）。

**本地实现**：
- 内部统一 schema（YAML → `serde_yaml` → `BasisSet`）。
- 加载器：`qwb-registry` → `Registry` → `get_basis(name) -> BasisSet`。
- **枚举/球谐切换只有一处**：`BasisSet.angular`；`Engine` 显式读取，禁止隐式默认（R-06）。
- **归一化只有一处**：加载时一次性处理，运行时不转换。
- 多元素文件用 `ShellBlock`（§4 K1），**每个元素必须恰好一个块**。

**验证（CI）**：
1. schema 校验（`deny_unknown_fields`）。
2. 指数/系数与 BSE 逐项 diff（< 1e-12 相对）。
3. 归一化：H2/STO-3G 重叠积分与解析闭式一致（I-02）。
4. 球谐 vs 笛卡尔：d 壳层函数数 5 vs 6（计数断言），能量差异存在且有限。
5. ECP-pairing：有 ECP 元素必须能找到匹配基组，否则 `Registry::load_pair` 返回 `EngineError::Data`。

### 8.2 ECP

- **来源**：Stuttgart / def2-ECP 官方 + BSE 导出（**再分发条款必须先确认**，见 §14 #20）。
- **陷阱显式声明**：`r^n` vs `r^(n-2)` 强制 `r_power_convention`，加载器统一转内部约定。
- **配对校验**：`Registry::load_pair(basis, ecp)` 是唯一入口。

### 8.3 泛函（最重，本地实现）

**设计**：每个泛函拆成 `FunctionalTerm`（X-01）；数据侧只放参数与组合，代码侧放数学形式。
不允许参数散落；一个泛函不能在多处重写。

| 组件 | 数学形式 | 依赖 |
|---|---|---|
| `LdaSvwn5` | VWN5 + 自旋插值 | ρ |
| `GgaPbeX` | PBE 交换（κ=0.804, μ=0.21951） | ρ, σ |
| `GgaPbeC` | PBE 相关 | ρ, σ |
| `GgaB88` | Becke 88 | ρ, σ |
| `GgaLyp` | LYP 相关 | ρ, σ |
| `HfExchange` | K 矩阵混合 | 来自 S-02 |

**验证三道门禁**：① 有限差分导数 < 1e-9（X-02，CI 必跑）；② 单点值 vs `tests/reference`
（LDA/GGA < 1 mHa、B3LYP < 2 mHa）；③ 数值稳定性：低密度/超大梯度截断规则写进
`functional.yaml` 的 `stability` 字段，有单元测试。

### 8.4 XC 网格

- 参数数据化：`grid/*.yaml` 存等级 → 参数；**代码不含硬编码点数**。
- 收敛测试：T2 分子相邻等级能量差 < 0.1 mHa，CI 断言。
- 陷阱显式：Becke 锐度、原子半径缩放全在数据里（`becke_sharpness`、`radii_scale`）。

### 8.5 溶剂

- 第一阶段只做接口与数据，不做完整 PCM。
- 字段：`epsilon`、`temperature_K`、`atomic_radii`、`cavity_model`、`scale`。
- `cavity_model` 必须显式；loader 不因不认识而报错，但 `Engine::run` 返回 `Unsupported`。
- 验证：ε 与 NIST 一致（水 78.3553 @ 298.15 K）。

### 8.6 数据治理（v2.1 补充 load / verify 分工）

| 规则 | 机制 |
|---|---|
| 数据与代码分仓（逻辑上） | `qwb-data/` 独立目录 + 独立 semver（物理拆仓见 §14 #21） |
| 每条数据带 sha256 | 规范化内容摘要；文件内 / `registry.yaml` / `data_snapshot` **三处一致** |
| 每条数据带 citation | schema required，缺失即拒（`g2.citation`） |
| 每条数据带 license | `commercial_ok` / `redistribute_ok` 显式 |
| 变更必须 bump version | CI 检查 `CHANGELOG.md` |
| 结果快照 | 每次计算写 `data_snapshot.json` |
| 发布清单 | `scripts/gen_data_licenses.py` 生成 `DATA_LICENSES.md` |
| cargo-deny 联动 | `deny.toml` 的 `[licenses]` 映射依赖许可；数据许可由 `DATA_LICENSES.md` 人审 |

**load ≠ verify**（v2.1 明确，消除 v2.0 的 D-01 矛盾）：

- **load**（`Registry::open`）：只要求文件语法/结构可解析。**空注册表必须 `Ok`**，
  否则 `data list`/`show`/工具链在冷启动时不可用。
- **verify**（`qwb data verify`）：G2 门禁。0 条目、缺 citation/license/sha256 一律退出码 3。

---

## 9. 验证体系

三层，缺一不可：

1. **单元层**：每 WP 自带（积分对照、泛函导数、网格精度、ECP 矩阵元、状态机转移）。
2. **物理极限层**：氢原子解析解（V-01）、均匀电子气、HF 一致性（V-02）、网格 Σw 收敛。
3. **基准集层**：H2、H2O、CH4、NH3、CO、N2、苯（V-03）；对照几何/频率/高精度文献能量，
   **全部经 `tests/reference/*.csv` 且带来源**（§7.0）。

**本地/CI 门禁（一条命令）**：Windows 用 `scripts/verify.ps1`，Linux/CI/WSL 用
`scripts/verify.sh`（两者覆盖同一清单，见 §13）。它等价于：

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`（两个 workspace）
- `cargo test --workspace`（含 proptest）
- `cargo deny check`（advisories/bans/licenses/sources）
- `qwb-gen-schemas --check`（K1 不漂移）
- `qwb data verify`（G2）
- `mypy --strict`（K6 桩 + typed consumer）
- `scripts/check_deps_centralized.sh`、`check_unsafe.sh`、`check_rayon_determinism.sh`

**CI 分层**：PR 跑上面全部；每日跑 T3–T6；每周跑完整基准 + `cargo criterion` 归档 + diff 报警。

**门禁 G**：

- **G1 契约门禁**：K1–K6 已冻结；改动只允许新增；破坏性改动需人类评审。
- **G2 数据门禁**：D 轨每次合入必须有 citation + license + sha256；`cargo-deny` 许可映射一致。
- **G3 物理门禁**：新增泛函必须过 X-02 有限差分导数。
- **G4 集成门禁**：T 之前 WP 的验收必须全绿。
- **G5 复现门禁**：UI 可见结果目录必须含 `env.json`；CI 强制（`inspect` 退出码 3）。
- **G6 Rust 门禁**：`#![deny(unsafe_code)]`（例外需 SAFETY + 人类批准 + 白名单）；
  `Cargo.lock` 变更必须带 PR 说明；`cargo clippy -- -D warnings` 全绿。
- **G7 并行门禁**：所有 `rayon` 归约必须确定性（按索引分块，禁止顺序依赖 `reduce`）；
  1 线程 vs N 线程 byte-identical。

---

## 10. Agent 执行协议

### 10.1 开工前（每个 WP）

1. 读 `docs/status.yaml`，确认该 WP 依赖全部 `done`。
2. 读 `qwb-contracts/` 中相关 K 文件与 `qwb-contracts/studio/*.md`。
3. 检查产物是否已存在；若存在且验收通过，标 `done` 不重复做。
4. 跑全套门禁（`scripts/verify.ps1` 或 `scripts/verify.sh`）；若基线红，先修基线
   （唯一允许扩大改动范围的情形）。
5. 确认没有其它 WP 正在改同一文件（`docs/status.yaml` 的 `owner` 字段）。

### 10.2 执行中

1. 只改本 WP 涉及的代码/数据。
2. 不得改 K1–K6 的语义（见 §10.4）。
3. 不得删除或修改其他 WP 的产物。
4. 每次提交附件：`cargo fmt --check` + `cargo clippy -- -D warnings` + 该 WP 验收命令输出摘要。
5. **公共 API 变更**（`pub` 项）必须在 PR body 里显式列出，触发 G1 复核。
6. **新增 crate 依赖**只能加到 `[workspace.dependencies]`，并在 PR body 说明理由（人类批准）。

### 10.3 卡住时怎么办（**v2.1：默认前进，不默认停下**）

v2.0 的"任何不明确就停下上报"会把 agent 卡死。v2.1 分级如下：

| 情形 | 动作 |
|---|---|
| 验收命令的**细节**不明确（哪个分子、哪个容差） | **选 §6/§7 里更严格的解释**，在 `docs/status.yaml` 的 `note` 写明，继续做 |
| 需要新增**可选字段/新变体/新方法**（不破坏既有语义） | 直接做，进 `CHANGELOG.md`，继续 |
| 参考值查不到 | 走 §7.0 的**内部独立路径 fallback**，在 CSV `note` 与 `docs/error-budget.md` 登记 |
| 需要改**已冻结语义**（删字段、改状态机、改退出码含义） | **停**，写进 `docs/choices.md` 的"待人类裁决"，转做**其它可并行的 WP** |
| 两个 WP 的验收互相冲突 | 停，写进 `docs/choices.md`，转做其它可并行的 WP |
| 基线编译/测试红且非本 WP 引入 | 停并报告；这是唯一允许"不动手"的硬阻塞 |

**禁止**：伪造结果、把未实装的功能标 `done`、用自算值冒充文献值、静默跳过失败的门禁。

### 10.4 契约升级流程（真正需要动 K 时）

1. 在 `docs/choices.md` 登记"为什么必须改" + 影响面（哪些 WP 已依赖它）。
2. 改 `qwb-contracts/CHANGELOG.md`（语义变更必须 bump）。
3. 同步改：Rust 类型、生成的 schema、`*.md`、`*.pyi`、测试。
4. 人类评审通过后，**同时修订本文件 §4**，并把 v2.x 号 +1（§15）。

### 10.5 推荐工作顺序

```
（C 轨已 done）
1. D-01, D-02, D-03, D-09, D-13          ← 数据能撑起 T1 即可
2. I-00, I-01..I-04
3. X-01..X-05
4. S-01..S-03（S-06 可与 T2 并行）
5. T1 集成通过
6. D-04, D-10 → T2（含 S-06 性质）
7. G-01..G-03 → T3
8. F-01..F-02 → T4
9. D-07, D-08, I-05 → T5
10. D-11, D-12 + S-02 扩充 K 矩阵 → T6
11. P-01..P-08 → T7
12. P-09..P-10 → T8
13. V-04, V-05, V-06 → 收尾
```

### 10.6 进度记账

每完成一个 WP：`docs/status.yaml` 写
`{state: done, ts: <ISO8601 UTC>, artifact: <一句话>, verification: <命令/断言>}`。
若被卡：`{state: blocked, reason: ..., log: ...}` **且** 在 `docs/choices.md` 登记。

---

## 11. 风险与门禁对照

| ID | 风险 | 缓解 | 门禁 |
|---|---|---|---|
| R-01 | 契约缺陷导致返工 | 契约已冻结 + `#[non_exhaustive]` + schema 与类型同源 | G1 |
| R-02 | 数据转录错误 | 逐项 diff BSE + 双人核对 + CI 断言 | G2 |
| R-03 | Rys vs Obara-Saika | **已定案：Obara-Saika 为主，Rys 作数值独立交叉验证**（同值 < 1e-10） | — |
| R-04 | 泛函参数笔误 | 单点基准 + 导数有限差分双门禁 | G3 |
| R-05 | 网格精度不足 | 等级映射 + 收敛测试 + 数据化锐度参数 | — |
| R-06 | 球谐/笛卡尔隐性切换 | `angular` 单入口 + 单元测试 | — |
| R-07 | `rayon` 竞争 | TSan（nightly）+ 单/多线程 identical | G7 |
| R-08 | 结果不可复现 | `env.json` 强制 + `replay` | G5, G6 |
| R-09 | 数据许可冲突 | `DATA_LICENSES.md` + `cargo-deny` 映射 | G2 |
| R-10 | 参考实现污染 | 只允许 CI 黑盒对照，代码不得复制；`grep` 门禁 | CI |
| R-11 | faer/nalgebra 性能不足 | `qwb-linalg` 抽象层 + 基准守门；必要时接系统 BLAS（需批准） | V-06 |
| R-12 | pyo3 abi3 版本错配 | **冻结 `abi3-py39`** | G1 |
| R-13 | `rayon` 非确定归约 | 统一 `DeterministicReduce` 辅助函数 | G7 |
| R-14 | HDF5 跨平台痛 | Zip + JSON 已定案；**不引入 `ndarray-npy`** | — |
| R-15 | 编译时间过长 | workspace 分 crate；feature gate（`faer`、`python`）；CI `sccache` | — |
| R-16 | `unsafe` 扩散 | `deny(unsafe_code)` + 白名单脚本 | G6 |
| R-17 | crate 生态 supply chain | `cargo deny`（四类）+ 依赖集中 + 新 crate 需批准 | G6 |
| R-18 | 矩阵与 numpy 拷贝开销 | 初期全部拷贝（≤500×500 可接受），接口不变 | — |
| R-19 | llvm/toolchain 偏移影响数值 | 根 `rust-toolchain.toml` 冻结 `1.97.1`；对比 CI 用同镜像 | G6 |
| **R-20** | **`serde_yaml` 已 unmaintained** | §2.4 冻结它；`deny.toml` 只 ignore `RUSTSEC-2024-0449`；迁移为后续可选（需人类批） | G6 |

---

## 12. 附录

### 12.1 目录结构（v2.1 = 实际布局；**单 git 工作树**，见 §14 #2）

```
project.md                 # 本文件
README.md  rust-toolchain.toml  .gitattributes  .gitignore
.github/workflows/ci.yml

qwb-contracts/             # 契约（C 轨）
  schema/                  # K1：7 类 × {json,yaml}，由 schemars 生成
  engine-api/              # K2：crate qwb-engine-api（独立 workspace，自带 Cargo.lock）
  studio/
    task.schema.yaml  result.schema.yaml      # K3
    qwb-format.md                             # K4
    cli.md                                    # K5
    examples/example.qwb                      # K4 示例容器
    python/qwb_engine.pyi                     # K6
    python/tests/consumer_example.py          # K6 类型门禁
  CHANGELOG.md

qwb-data/                  # 数据（D 轨）
  registry.yaml  DATA_LICENSES.md  CHANGELOG.md
  basis/ ecp/ functional/ grid/ solvent/      # 生产数据（D 轨交付）
  examples/                                   # K1 样例（example- 前缀，非交付物）

qwb-core/                  # Cargo workspace（引擎）
  Cargo.toml  Cargo.lock  deny.toml
  crates/
    qwb-registry/          # K1 类型 + 校验（C-01）+ 加载器（D-01）
    qwb-io/                # K3 状态机/布局 + K4 容器（C-03/C-04）
    qwb-cli/               # K5 CLI（C-05）
    qwb-linalg/  qwb-basis/  qwb-integrals/  qwb-xc/  qwb-grid/
    qwb-scf/  qwb-gradient/  qwb-hessian/  qwb-opt/  qwb-freq/
    qwb-props/             # S-06
    qwb-engine/  qwb-python/
  benches/  tests/  examples/

qwb-studio/                # Python 产品（P 轨）
  qwb_studio/{models,scheduler,gui}
  examples/  tests/  pyproject.toml

docs/  status.yaml  choices.md  error-budget.md
scripts/  verify.sh  check_deps_centralized.sh  check_unsafe.sh
          check_rayon_determinism.sh  unsafe_allowlist.txt
          gen_data_licenses.py
```

> `qwb-core` 以 **path 依赖**引用 `qwb-contracts/engine-api`（单仓，§14 #2）。

### 12.2 命名与单位

- 引擎内部**原子单位**；边界（CLI/API/UI）用 Å/eV 且显式字段后缀。
- 能量 `_hartree`；长度 `_angstrom`；梯度 `_hartree_per_bohr`；Hessian `_hartree_per_bohr2`；
  偶极 `_debye`；温度 `_K`；数据文件里的基组指数/ECP 指数/网格 cutoff 用 **Bohr**。
- 时间戳 ISO 8601 UTC。
- Rust crate 名统一 `qwb-*`；类型 PascalCase；函数 snake_case。
- Python 绑定包 `qwb_engine`；产品包 `qwb_studio`。

### 12.3 提交规范

`<轨道>-<ID>: <一句话>`，例如 `I-02: add one-electron overlap integrals`。

PR 必须含：WBS ID、`cargo fmt --check`、`cargo clippy --workspace -- -D warnings`、
该 WP 验收命令输出、新增 crate 的理由 + `deny.toml` 差异、公共 API 变更清单。

### 12.4 状态文件格式

```yaml
C-01: {state: done, ts: "2026-10-22T22:53:00Z", artifact: "...", verification: "..."}
D-03: {state: in_progress, owner: agent-a}
D-04: {state: blocked, reason: "BSE export format changed", log: "logs/D-04.log"}
```

### 12.5 人类决策清单

**已全部收敛 → §14**。仍未定的只有"具体数据来源的许可确认"与"代码许可最终选择"，
两者都给了**默认值且不阻塞**。

---

## 13. 环境事实（v2.1 新增，避免每个会话重新探测）

开发机：Windows 11（当前工作目录 `D:\python_II\Jaguar`，**路径含空格需引号**）。

| 工具 | 版本/位置 | 备注 |
|---|---|---|
| rustc / cargo | 1.97.1（`C:\Users\JYLin\.cargo\bin`） | 由根 `rust-toolchain.toml` 钉住；`rustup toolchain list` 有 `1.97.1` |
| cargo-deny | 0.20.2（已装） | `cargo deny check` 可用 |
| cargo clippy / fmt | 随 toolchain | — |
| Python | 系统 3.9.9；仓库 `.venv` = 3.12.10 | **用 `.venv`**：`.\.venv\Scripts\python.exe` |
| mypy / numpy | `.venv` 内（mypy 2.4.0、numpy 2.5.4） | `mypy --strict` 需要 numpy 桩 |
| bash | `C:\WINDOWS\system32\bash.exe`（WSL） | 仓库路径 = `/mnt/d/python_II/Jaguar` |
| git | 2.53 | 已 init；行尾由 `.gitattributes` 统一 LF |
| 网络 | crates.io 可达 | `cargo add` / `cargo install` 可用 |

**标准命令**：

```powershell
cd D:\python_II\Jaguar

# 全门禁（Windows 原生，含 mypy）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1

# 全门禁（WSL bash；CI 用同一个）
bash scripts/verify.sh

# 契约仓单独
cd qwb-contracts\engine-api
cargo test --all-targets
cd ..\..

# K1 schema 重新生成 / 校验
cd qwb-core
cargo run -q -p qwb-registry --bin qwb-gen-schemas -- ..\qwb-contracts\schema
cargo run -q -p qwb-registry --bin qwb-gen-schemas -- ..\qwb-contracts\schema --check

# 数据打戳 / 校验
cargo run -q -p qwb-registry --bin qwb-stamp-data -- ..\qwb-data\basis\<file>.yaml
cargo run -q -p qwb-cli --bin qwb -- data verify

# K6 类型门禁
cd ..
$env:MYPYPATH = "D:\python_II\Jaguar\qwb-contracts\studio\python"
.\.venv\Scripts\python.exe -m mypy --strict qwb-contracts\studio\python\qwb_engine.pyi
.\.venv\Scripts\python.exe -m mypy --strict qwb-contracts\studio\python\tests\consumer_example.py
```

**已知环境坑**：`rustup` 自更新曾失败（不影响构建）；WSL 里 Windows 的 `cargo` 是
`cargo.exe`（bash 不自动补 `.exe`），`scripts/verify.sh` 已做兜底；`verify.ps1`
**必须保持 ASCII-only**（Windows PowerShell 5.1 用 ANSI 解码无 BOM 的脚本，中文会炸）；
`data verify` 的类别推断依赖父目录名（`basis/`、`ecp/`…），样例放 `examples/basis/` 也可以。

---

## 14. 决策台账（**已定，别再重新讨论**）

| # | 事项 | 结论 | 类别 |
|---|---|---|---|
| 1 | K1 生成方式 | **`schemars` 从 Rust 类型生成**；`schemars` 进 §2.4 白名单 | 已定 |
| 2 | 代码仓库拓扑 | **单 git 工作树**：`qwb-contracts/`、`qwb-data/`、`qwb-core/` 三顶层目录；`qwb-core` 用 path 依赖引用 `engine-api` | 已定（可后续拆仓，见 #21） |
| 3 | `rust-toolchain.toml` 位置 | **仓库根**（同时覆盖两个 workspace） | 已定 |
| 4 | MSRV / toolchain | **1.97.1**（本机已装） | 已定 |
| 5 | `qwb-linalg` 默认后端 | **`nalgebra`**，`faer` 为可选 feature | 已定 |
| 6 | R-03 数值方案 | **Obara-Saika 为主 + Rys 数值交叉验证** | 已定 |
| 7 | `ndarray-npy` | **不引入** | 已定 |
| 8 | pyo3 ABI | **`abi3-py39`** | 已定 |
| 9 | 首版泛函清单 | **SVWN5、PBE、B88、LYP、B3LYP、HF**（只这些） | 已定 |
| 10 | 溶剂 | 第一阶段**只做接口 + 数据**，引擎返回 `Unsupported` | 已定 |
| 11 | ECP 元素范围 | **H–Kr**（先 Cu、Zn，再 Na–Kr；def2-ECP 先 K–Kr） | 已定 |
| 12 | 参考实现（libxc 等）边界 | **只允许 CI 黑盒对照，不引入任何代码/数据**；R-10 的 grep 门禁 | 已定 |
| 13 | 是否允许热点 `unsafe` | **默认不允许**；需人类批准 + `// SAFETY:` + 白名单 | 已定 |
| 14 | `serde_yaml` | 保留（§2.4 冻结）；`deny.toml` 只 ignore `RUSTSEC-2024-0449` | 已定（可迁移，需批准） |
| 15 | K3 状态机 | **6 条转移**（含 `queued→cancelled`）；重试 = 新 task id | 已定（v2.1） |
| 16 | `data_snapshot` 摘要 | 用**规范化内容 sha256**（与文件内 `sha256`、registry 条目三处一致） | 已定 |
| 17 | CLI 数据根 | `QWB_DATA_ROOT` 环境变量或向上搜索，**不加命令行参数** | 已定 |
| 18 | `zip` feature | **只 `deflate`**（关闭 bzip2/lzma/aes/zstd） | 已定 |
| 19 | 代码许可 | **`Apache-2.0`**（所有 crate 的 `license` 字段）。人类可一次性改写；**不阻塞** | 默认已采用 |
| 20 | 数据许可 | 自研数据用 `Apache-2.0`；**外部来源（Stuttgart/def2-ECP、BSE）逐条确认后再写 `redistribute_ok: true`** —— 这是 D-07/D-08 的前置检查，不是全项目阻塞 | 默认 + 逐条确认 |
| 21 | 物理拆三仓 | 暂不拆。触发条件：数据仓需要独立对外发布 | 观察项 |
| 22 | `cargo-audit` | **不引入**，`cargo deny check advisories` 覆盖 | 已定 |

---

## 15. 变更记录

### v2.0 → v2.1（2026-10-22）

**目标澄清**

- 新增 §2.2：明确"模仿 Jaguar"= **对齐产品面 + 第一阶段方法集合**，并**排除 PS 积分、
  PBF/RISM**；把 TS/IRC/scan/NBO/EPR 加入非目标。
- §2.4 允许库补充 `schemars`（已批准）、`numpy`、`pydantic`、`mypy`；移除 `h5py` 残留
  与 `cargo-audit`；明确 `ndarray-npy` 不引入。
- 新增 **S-06（性质：偶极/布居电荷/轨道能级）**：补上 §12.1 里存在但无 WP 的 `qwb-props`，
  并让 T2 断言"Jaguar 也会报的东西"。

**契约与实现对齐**（§4）

- K1：写实际形态 —— `shells: Vec<ShellBlock>`、`coefficients: Vec<Vec<f64>>`、
  `notes` 字段、规范化 sha256 规则、`MoleculeFile` 不受 G2 门禁、`RegistryManifest` 形状、
  7 类 schema 文件名与生成方式、机器可读错误 code 列表。
- K2：写出**真实 API**（`Registry` 句柄语义、`EngineBackend` + `with_backend`、
  `InvalidInput` 变体、`SUPPORTED_FEATURES` 词汇表、`non_exhaustive` 的构造方法约定、
  矩阵 list-of-list、`exit_code` 映射）。
- K3：状态机从 5 条扩到 **6 条**（新增 `queued→cancelled`）；明确**重试 = 新 task id**，
  不再需要 `failed→queued`。同步改了 `qwb-io::task`、`task.schema.yaml` 与测试。
- K4：写明"只 deflate"、字节可复现、结构性拒绝清单、`data_snapshot` 摘要语义。
- K5：把 C 轨实装时的增量**写进契约**（`batch PATTERN|DIR`、`data show NAME`、
  `data verify [PATH...]`、`QWB_DATA_ROOT`）。
- K6：固化 `.pyi` 实际签名（`registry_path` 关键字、错误类带 `exit_code`、`abi3-py39`）。

**消除 v2.0 内部矛盾**

- C-01 的验收曾要求 `qwb data verify`（D-02 的产物）→ 改为"C-01 只要最小 verify，
  D-02 扩成 6 坏/6 好回归集"。
- D-01 的"空注册表可加载"与 G2 的"必须有条目"冲突 → §8.6 明确 **load ≠ verify**。
- T1 的"与 P-04 参考值比"是错引用（P-04 是 `env.json` 生成器）→ 改为 §7.0 的
  `tests/reference/*.csv`。
- T5 的"ZnO"没定基组/方法 → 冻结为 CuH/LANL2DZ 与 ZnO/def2-SVP+def2-ECP。
- §9 的 `cargo audit`、`cargo-nextest`（本机未装）→ 统一为 `scripts/verify.sh`。
- §12.1 目录与实际不符（缺 `qwb-io`、`rust-toolchain.toml` 在根、单仓）→ 按实际重写。

**让 agent 不被卡住**

- §10.3 新增分级决策规则（默认选更严格解释并前进；只有改冻结语义才停）。
- §7.0 新增参考值政策（外部引用优先 + 允许"内部独立路径" fallback 并登记）。
- §13 新增**环境事实**（toolchain、cargo-deny、mypy、WSL bash、标准命令）。
- 新增 **`scripts/verify.ps1`**（Windows 原生）/ **`scripts/verify.sh`**（Linux/CI）：
  一条命令跑完全部门禁，**agent 不需要记住 12 条命令**。
- §14 新增**决策台账**：把 v2.0 §12.5 的 13 项人类待定事项全部收敛为定论或"默认 + 不阻塞"。
