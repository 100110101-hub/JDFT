# project.md · QuantumWorkbench v2.0

> 本文件是 **唯一开工依据**。agent 与人类开发者都按本文件执行。
> 不在本文件中的事项视为未授权，需人类决策后修订本文件再执行。

---

## 1. 阅读顺序

| 你要做什么 | 看哪节 |
|---|---|
| 理解项目边界 | §2 |
| 理解为什么这样排顺序 | §3 |
| 开工前必须冻结什么 | §4（契约 K1–K6） |
| 我要做哪个包 | §6（WBS） |
| 项目的"完成"刻度 | §7（垂直切片 T1–T8） |
| 数据层怎么做（本项目中心） | §8 |
| 验收怎么判 | §9 |
| agent 每步必须怎么做 | §10 |
| Rust 特有的坑 | §11 |
| 目录/Cargo workspace 结构 | §12 |

---

## 2. 目标与非目标

### 2.1 目标

从零实现一个**分子 DFT 软件**：

- **引擎自研**：不调用 PySCF/ORCA/Gaussian/QE/Psi4/libxc/ase 等电子结构引擎或泛函库。
- **数据是中心**：基组、ECP、泛函、网格、溶剂参数的**获取、本地实现、验证、版本化**是项目核心工作。
- **工作流完整**：项目 / 分子 / 任务 / 结果 / 调度 / 复现。
- **界面完整**：GUI（Python）+ Python API（pyo3）+ CLI（clap），共用同一套 Task 模型。
- **可复现是底线**：每个结果都能回溯到输入 + 数据快照 + 环境（`env.json` 必须含 Rust toolchain、`Cargo.lock` 的 sha256、crate 版本清单）。

### 2.2 非目标（第一阶段明确不做）

固体/周期、TDDFT、多参考、耦合簇、相对论（ECP 除外）、GPU、MPI 大规模、meta-GGA、双杂化、range-separated、显式溶剂、QM/MM、SMD。

### 2.3 允许使用的第三方库（冻结）

**引擎（Rust）**：

`serde`, `serde_yaml`, `serde_json`, `nalgebra`, `faer`（可选后端）, `rayon`, `ndarray`（数据交换）, `nalgebra`+`numpy`(pyo3 bridging), `tracing`, `tracing-subscriber`, `anyhow`, `thiserror`, `clap`, `zip`, `sha2`, `hex`, `bytemuck`, `wide`（portable SIMD）, `approx`（测试）, `proptest`, `criterion`, `pyo3`, `numpy`.

**GUI / 产品（Python）**：

`PySide6`, `VTK`, `h5py` **不再使用**（见 K4）, `rdkit`（仅结构读写）, `pytest`, `hypothesis`, `maturin`.

**外部系统级**：

BLAS/LAPACK（`faer` 纯 Rust 不需要，若选 `ndarray-linalg` 后端需系统 BLAS）、系统 libhdf5（**不引入**，见 K4）。

**任何不在此列表且进入产品依赖树的库，需人类批准，并进 `cargo-deny.toml` 白名单。**

---

## 3. 组织原则

1. **契约先行，并行不返工**。先冻结 K1–K6，数据/引擎/产品三轨才能并行。
2. **数据层随垂直切片增量补**。第一次只需要 H 与 STO-3G。
3. **引擎按垂直切片推进，不做层内串行**。第一次就跑通一条能量，把集成风险前移。
4. **产品轨先于引擎成熟开工**。Product 轨前期用 mock 数据。
5. **每个工作包都可独立验收**。没有"感觉差不多了"这一步。
6. **agent 不自由发挥**。执行协议写明每一步检查什么。
7. **Rust 特有约束**：
   - **`#![deny(unsafe_code)]` 默认**；只允许 3 类 crate 例外：FFI（BLAS）、SIMD intrinsics（`wide` 已封装）、获批的热点优化。例外需 `// SAFETY:` 注释 + 人类批准。
   - **`Cargo.lock` 入 git**，禁止 `cargo update` 在无 PR 记录情况下修改。
   - **`cargo fmt` / `cargo clippy -- -D warnings` 必须全绿**才能合入。
   - **workspace 层级不得引入循环依赖**（cargo 会检测，但也要人有意识）。

**风险优先级**：集成风险 > 数学正确性风险 > 数据正确性风险 > 性能。

---

## 4. 契约 K1–K6（冻结）

> 所有契约文件**必须先于任何代码**提交到 `qwb-contracts/` 仓库。任何变更需 `CHANGELOG.md` + 人类评审通过。

### K1 · 数据 Schema

**路径**：`qwb-contracts/schema/`

**文件**：`basis.schema.yaml`、`ecp.schema.yaml`、`functional.schema.yaml`、`grid.schema.yaml`、`solvent.schema.yaml`、`molecule.schema.yaml`、`registry.schema.yaml`。

**共同字段（所有数据）**：`name`, `version`, `citation[]`（每项 `title` + `doi`/`url`）, `license`（`spdx`, `commercial_ok`, `redistribute_ok`）, `created`, `sha256`（工具生成）。

**基组特有**：`elements[]`, `angular`（`spherical`/`cartesian`）, `normalization`, `role`（`orbital`/`ri-j`/`ri-jk`/`ecp-fit`）, `shells[]`（`l`, `exponents[]`, `coefficients[]`, `contraction_type`）。

**ECP 特有**：`element`, `n_core_electrons`, `z_eff`, `r_power_convention`, `local_part`, `nonlocal_parts[]`。

**泛函特有**：`components[]`（`id`, `weight`）, `requires`（`rho`/`sigma`/`laplacian`）, `parameters`。

**网格特有**：`level`, `radial_points`, `lebedev_order`, `becke_sharpness`, `cutoff`。

**溶剂特有**：`epsilon`, `temperature_K`, `atomic_radii`, `cavity_model`, `scale`。

**Rust 实现约定**：schema 用 `schemars` 从 Rust 类型生成（保证 K1 与代码 100% 对齐），或手写成 `serde_yaml` 反序列化 struct 并加 `#[serde(deny_unknown_fields)]`。**二选一，须在 C-01 内定案。**

### K2 · 引擎 Rust API（重写）

**路径**：`qwb-contracts/engine-api/src/lib.rs`

```rust
//! qwb-engine-api: 冻结的引擎接口。任何破坏性变更需经 CHANGELOG。

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JobInput {
    pub molecule: Molecule,
    pub basis_name: String,
    pub method_name: String,
    pub grid_level: GridLevel,
    pub solvent: String,      // "vacuum" 或溶剂键
    pub jobtype: JobType,     // Sp | Opt | Freq
    pub options: TaskOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JobResult {
    pub energy_hartree: f64,
    pub gradient_hartree_per_bohr: Option<DMatrix<f64>>, // 3N×1
    pub hessian_hartree_per_bohr2: Option<DMatrix<f64>>, // 3N×3N
    pub dipole_debye: Option<DVector<f64>>,
    pub mo_energies_hartree: Option<DVector<f64>>,
    pub mo_coeffs: Option<DMatrix<f64>>,
    pub scf_converged: bool,
    pub data_snapshot: DataSnapshot,   // path -> sha256
    pub log: String,
    pub env: EnvInfo,                  // rustc, cargo.lock sha256, data versions
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("data load failed: {0}")]
    Data(String),
    #[error("scf not converged in {iters} iterations (max {max})")]
    ScfNotConverged { iters: u32, max: u32 },
    #[error("unsupported feature: {0}")]
    Unsupported(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("internal error: {0}")]
    Internal(String),
}

pub trait Reporter: Send + Sync {
    fn progress(&self, stage: &str, pct: f64);
    fn log(&self, line: &str);
    fn is_cancelled(&self) -> bool;
}

pub struct Engine { /* opaque */ }

impl Engine {
    pub fn new(registry: &Registry) -> Result<Self, EngineError>;
    pub fn run(&self, input: &JobInput, reporter: &dyn Reporter) -> Result<JobResult, EngineError>;
    pub fn request_cancel(&self);
}
```

**冻结约定**：

- 公共 struct/enum 一律 `#[non_exhaustive]`，防止下游依赖字段全集。
- 公共字段后缀 `_hartree`/`_bohr`/`_debye`，禁止裸 `energy`。
- 错误 enum 只允许新增变体（`#[non_exhaustive]`），禁止删除。
- 所有跨线程类型 `Send + Sync`。
- `Engine::run` 接受 `&dyn Reporter`，引擎不得直接写 stdout。

### K3 · Task / Result Schema（产品面）

**路径**：`qwb-contracts/studio/task.schema.yaml`、`result.schema.yaml`

**Task 状态机（冻结）**：

```
draft → queued → running → converged
                   │
                   ├→ failed
                   └→ cancelled
```

**结果目录布局（冻结）**：

```
<task_id>/
  input.yaml
  env.json            # rustc, cargo.lock sha256, data_snapshot, threads, OS
  log.txt
  result.json         # JobResult 序列化（矩阵用 base64 f64 LE 或 list-of-list，二选一，见 K4）
  data_snapshot.json
  artifacts/
```

**没有 `env.json` 视为不完整结果，UI 应标红。**

### K4 · 文件格式（重写）

**输入**：YAML，`input.schema.yaml` 定义。

**项目容器 `.qwb`**：**Zip 容器**（`zip` crate）。内部布局：

```
meta.json            # Project 元数据
tasks/<id>/input.yaml
tasks/<id>/result.json
tasks/<id>/env.json
tasks/<id>/log.txt
data_snapshot.json
artifacts/**         # 图、表、导出文件
```

**为什么不用 HDF5**：`hdf5` crate 依赖系统 libhdf5，Windows 分发痛苦；`hdf5-metno` 分支可用但 API 不稳定。Zip + JSON 是纯 Rust、跨平台、人类可探查、可差异对比。**二进制矩阵数组**：大矩阵用 `ndarray-npy` 序列化为 `.npy` 放进 zip（若引入，需人类批准）。

**分子**：`.xyz`、`.mol`、`.sdf`（rdkit 只做结构读写，Python 侧）。

**缓存**：`.npz`（`ndarray-npy` 或自写）。非契约必须。

### K5 · CLI 契约

用 `clap` 生成，签名冻结：

```
qwb run    input.yaml [--out DIR] [--threads N]
qwb batch  DIR/*.yaml [--parallel N]
qwb inspect DIR
qwb replay DIR [--dry-run]
qwb data list|show|verify
```

退出码：`0` 成功、`1` 输入错误、`2` 收敛失败、`3` 数据校验失败、`4` 内部错误（对应 `EngineError` 变体映射）。

### K6 · Python API 契约（重写）

**路径**：`qwb-contracts/studio/python/qwb_engine.pyi`

**构建**：`pyo3` + `maturin`。ABI：`abi3-py39`（或升级到 py310，见 §11 R-12）。

**契约签名（冻结）**：

```python
import qwb_engine as qwb

eng = qwb.Engine(registry_path="qwb-data/")
result = eng.run(job_input)   # JobInput 与 Rust 同名同字段
result.energy_hartree          # float
result.gradient_hartree_per_bohr  # np.ndarray | None
result.data_snapshot          # dict[str, str]
result.env.rustc_version
```

**硬约定**：

- **GIL 释放**：`Engine::run` 内部必须 `py.allow_threads(|| ...)`，否则 Python 端并发废掉。CI 用 concurrency 测试强制。
- **矩阵零拷贝**：小矩阵（< 500×500）拷贝可接受；大矩阵走 `numpy` 的 `PyArray` 视图或 shared memory。**初期全部拷贝，标 TODO**，但接口不变。
- **`Report` 回调**：Rust 侧 `Reporter` 通过 `pyo3` 桥到 Python callback 时，必须释放 GIL 后批量推送，禁止逐行回调。

---

## 5. 依赖 DAG

```
K1..K6 (契约)
   │
   ├──> D-tracks (数据)            ─┐
   ├──> I/X/S-plumbing (引擎骨架)   ─┤
   └──> P-scaffold (产品骨架)       ─┤
                                      ▼
                                T1 集成（H2/STO-3G/SVWN5 单点能）
                                      ▼
                                T2（H2O/def2-SVP/PBE）
                                      ▼
                                T3（解析梯度 + 优化）
                                      ▼
                                T4（频率）
                                      ▼
                                T5（ECP + 重原子）
                                      ▼
                                T6（B3LYP）
                                      ▼
                                T7（工作流 + 调度 + 复现）
                                      ▼
                                T8（GUI 端到端）
```

**关键规则**：

- K1–K6 冻结前，D/I/X/S/P 任何一轨**只能写脚手架**。
- `qwb-linalg`（线性代数抽象层）必须在 I-01 前冻结，让 `nalgebra`/`faer` 后端可切换。

---

## 6. WBS 工作包

> 每个包：`ID · 依赖 · 产物 · 验收`。验收必须有可执行命令或可检查断言。
> 状态记录在 `docs/status.yaml`。

### 6.1 契约轨（C 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| C-01 | — | K1 schema（`schemars` 生成或手写 serde struct）+ 5 个示例 | `qwb data verify` 拒载缺 citation；好文件通过 |
| C-02 | C-01 | K2 `qwb-engine-api` crate（`#![deny(unsafe_code)]`, `#![deny(missing_docs)]`） | crate 编译；`cargo doc` 无 warning；`Engine::new` 可返回 `Unsupported` 错 |
| C-03 | C-01 | K3 task/result schema + 状态机图 | schema 通过；状态机 proptest 覆盖合法/非法转移 |
| C-04 | C-02, C-03 | K4 文件格式定义 + `.qwb` 示例 | `.qwb` 示例能 open、meta 可读、zip 内部校验 |
| C-05 | C-02 | K5/K6 CLI + Python `.pyi` 契约 | `qwb --help` 与 `qwb_engine.pyi` 通过 `mypy` |
| C-06 | C-01..C-05 | 工作区骨架 + `rust-toolchain.toml` + `cargo-deny.toml` + CI 模板 | workspace 编译；`cargo-deny check` 通过；CI 绿灯 |

### 6.2 数据轨（D 轨） — 项目中心

> **详细要求见 §8。**

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| D-01 | C-01, C-06 | `qwb-data` + `qwb-registry` crate（serde 反序列化）+ `registry.yaml` | 空注册表可加载；非法文件报具体字段 |
| D-02 | D-01 | `qwb data verify` 子命令 | 3 个坏文件被拒；3 个好文件通过 |
| D-03 | D-01 | `STO-3G`（H,C,N,O,F） | 指数/系数与文献逐项一致；H2 重叠积分与解析对照 |
| D-04 | D-01 | `def2-SVP`（H,C,N,O） | 与 BSE 导出 diff < 1e-12 相对 |
| D-05 | D-01 | `def2-TZVP`（同范围） | 同上 |
| D-06 | D-01 | `cc-pVDZ`（H,C,N,O） | 同上 |
| D-07 | D-01 | `LANL2DZ` + ECP（Na–Kr，先 Cu,Zn） | 配对校验通过 |
| D-08 | D-01 | `def2-ECP`（K–Kr 先） | 与 D-04/05/06 配对校验 |
| D-09 | D-01 | `SVWN5`（含 `dε/dρ`, `dε/dσ`） | X-02 有限差分导数 < 1e-9 |
| D-10 | D-01 | `PBE`（交换+相关） | 同上 |
| D-11 | D-01 | `B88`、`LYP` | 同上 |
| D-12 | D-01 | `B3LYP` 复合定义 | 与文献基准能量差 < 2 mHa（水/苯/CO） |
| D-13 | D-01 | 网格等级 `coarse/normal/fine/veryfine` | 相邻等级能量差 < 0.1 mHa（T2 分子上验证） |
| D-14 | D-01 | 溶剂数据（8 种） | schema 通过；ε 与文献一致；`cavity_model` 字段齐全 |
| D-15 | D-02..14 | `DATA_LICENSES.md` + 每数据 sha256 + `cargo-deny` 数据许可映射 | 所有数据带 license；`DATA_LICENSES.md` 生成 |

### 6.3 积分轨（I 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| I-00 | C-06 | `qwb-linalg` 抽象层（`nalgebra` 默认，`faer` 可选 feature） | 两边后端编译通过；同一接口 roundtrip identical |
| I-01 | C-02, D-01 | 基组加载器 → 内部 AOs（`BasisSet`） | H2/STO-3G AO 数、对称性正确 |
| I-02 | I-01 | 1e 积分（重叠、动能、核吸引） | 数值对照 < 1e-10（每基函数） |
| I-03 | I-01 | 2e 积分（Obara-Saika 或 Rys，见 R-03） | 小体系与解析对照 < 1e-8 |
| I-04 | I-03 | 2e 积分并行化（`rayon`，**确定性归约**） | 1 线程 vs 4 线程结果 byte-identical |
| I-05 | I-02 | ECP 矩阵元（局域 + 非局域） | 与数值对照 < 1e-8 |
| I-06 | I-03 | 积分缓存（可选 feature） | 开关结果 identical |

### 6.4 XC 轨（X 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| X-01 | C-02, D-01 | `trait FunctionalTerm: Send + Sync`（`energy_density`, `potential_derivs`, `spin_scaling`, `requires`），**批量切片输入**（非标量，避免函数调用开销） | 单元测试通过；doc test 通过 |
| X-02 | X-01 | 有限差分导数校验工具 | D-09/10/11 全通过 |
| X-03 | X-01, D-13 | Becke 分区 + Lebedev 网格构造（`rayon` 并行 generate） | 均匀电子气极限通过；水 Σw 与 4π 差 < 1e-10；**并行与单线程 identical** |
| X-04 | X-03 | 网格管理（等级映射 + 缓存） | 切换等级释放旧网格（`Arc` 引用计数） |
| X-05 | X-02, X-04 | 泛函求值集成到 SCF | T1 单点能路径打通 |

### 6.5 SCF 轨（S 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| S-01 | I-02, I-03 | 密度矩阵 → Fock 构建（无 XC） | 与 `nalgebra`/`faer` 直接乘对照 < 1e-12 |
| S-02 | S-01, X-05 | RKS SCF（Fock + XC） | H2/STO-3G/SVWN5 能量收敛到 1e-8 Ha |
| S-03 | S-02 | DIIS + 收敛判据 | H2O/PBE 收敛到 1e-8 |
| S-04 | S-02 | UKS（开壳层） | O2 三重态跑通 |
| S-05 | S-03 | SCF 检查点 / 重启（`serde` 序列化轨道） | kill 后从 checkpoint 续，结果 identical |

### 6.6 梯度 / 优化轨（G 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| G-01 | S-03 | 解析梯度（Helmann-Feynman + 基组响应 + XC 网格项） | 与数值梯度最大分量差 < 1e-6 Ha/Bohr（T2 分子集） |
| G-02 | G-01 | BFGS + 冗余内坐标 | H2O 优化到收敛，键长误差 < 0.01 Å（T3） |
| G-03 | G-02 | 多步轨迹 + 收敛报告 | 3 个测试分子优化全部收敛 |

### 6.7 频率轨（F 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| F-01 | G-01 | 数值 Hessian（中心差分，`rayon` 并行位移） | H2O Hessian 与文献差 < 5% |
| F-02 | F-01 | 简正模 + IR 强度 + 热化学 | H2O IR 峰位与文献差 < 30 cm⁻¹（PBE 尺度） |

### 6.8 产品轨（P 轨）

> 产品骨架在 T1 前可用 mock；T7/T8 才真正端到端。**P 轨是 Python + maturin**。

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| P-01 | C-03 | Python `Project`/`Molecule`/`Task`/`Result` 数据模型（pydantic v2） | 单元测试通过；YAML/JSON roundtrip |
| P-02 | P-01, C-04 | `.qwb` zip 容器读写（Python `zipfile`） | roundtrip 保结构；损坏 zip 明确报错 |
| P-03 | P-01, C-05 | 本地调度器（`concurrent.futures` 进程池 + 队列 + 状态机 + 取消） | 10 个 mock 任务并发；取消即停 |
| P-04 | P-01, C-02 | `env.json` 生成器（含 `rustc --version`、`Cargo.lock` sha256、数据快照） | 缺 `env.json` 时 `Result` 构造失败 |
| P-05 | P-01 | `qwb` CLI（**Rust `clap` 实现**，Python 只做 shell 转发） | 所有子命令 `--help` 通过；退出码与 K5 一致 |
| P-06 | C-02, P-04 | `qwb_engine` pyo3 绑定（`maturin` 构建） | 与 K6 `.pyi` 一致；`mypy` 通过 |
| P-07 | P-05, P-06 | `qwb run` 端到端 | 跑通 T1 |
| P-08 | P-03, P-04 | 复现机制 `qwb replay` | 已跑结果 replay 后能量差 < 1e-10 |
| P-09 | P-07, P-08 | GUI 骨架（PySide6：导航 + Task Panel + Job Monitor） | 创建/提交/查看 mock 任务 |
| P-10 | P-09, G-02, F-02 | GUI 3D（VTK：分子、轨道、ESP、振动） | 打开 H2O 结果可视化正确 |

### 6.9 验证轨（V 轨）

| ID | 依赖 | 产物 | 验收 |
|---|---|---|---|
| V-01 | S-02 | 氢原子解析解测试 | 能量 -0.5 Ha 误差 < 1e-6 |
| V-02 | G-01 | 有限差分梯度一致性测试 | 所有基准分子自动跑 |
| V-03 | S-02..G-02 | 基准集：H2、H2O、CH4、NH3、CO、N2、苯 | CSV 对照表 |
| V-04 | 全部 | CI 流水线（`cargo nextest` + `pytest` + `cargo clippy -- -D warnings` + `cargo-deny`） | PR 跑单元+小基准；每日完整；结果归档 |
| V-05 | 全部 | `docs/error-budget.md` | 每类误差来源与量级 |
| V-06 | I-03, X-03, S-03 | `criterion` 基准套 `benches/` | 每次发布前跑；回归超 20% 报警（软门禁） |

---

## 7. 垂直切片里程碑 T1–T8

> **每个 T 是"项目真正向前一步"的刻度。** 只有所有前置 WP 通过，且该 T 端到端命令通过，才算完成。

### T1 · 最小端到端：H2 / STO-3G / SVWN5 单点能

**前置 WP**：C-01..C-06、D-01..D-03、D-09、D-13、I-00..I-04、X-01..X-05、S-01..S-03、P-01..P-07、V-01

**验收命令**：`qwb run examples/h2_sto3g_svwn5.yaml`

**断言**：

- 能量收敛到 1e-8 Ha
- 能量与 P-04 参考值差 < 1 mHa
- `env.json`、`data_snapshot.json` 存在
- `qwb replay` 后能量 identical
- `cargo clippy --workspace -- -D warnings` 通过

### T2 · 主流计算：H2O / def2-SVP / PBE 单点能 + 网格收敛

**前置 WP**：T1 + D-04、D-10、D-13

**断言**：

- 能量与文献差 < 1 mHa
- 相邻网格等级能量差 < 0.1 mHa
- 单线程与 4 线程结果 byte-identical（`rayon` 确定性归约门禁）

### T3 · 几何优化

**前置 WP**：T2 + G-01..G-03

**断言**：

- 最大力 < 4.5e-4 Ha/Bohr
- 键长误差 < 0.01 Å
- 解析梯度 vs 数值梯度差 < 1e-6 Ha/Bohr

### T4 · 频率

**前置 WP**：T3 + F-01、F-02

**断言**：

- 3N-6 全实（线性分子 3N-5）
- H2O IR 峰位与文献差 < 30 cm⁻¹
- 热化学量合理（正数、量级正确）

### T5 · ECP + 重元素

**前置 WP**：T2 + D-07、D-08、I-05 + K2 支持 ECP 字段

**断言**：

- Cu `n_core_electrons` 与基组价电子一致
- ZnO 单点能跑通，与文献差 < 5 mHa
- 非局域 ECP 项与数值对照 < 1e-8

### T6 · 杂化泛函 B3LYP

**前置 WP**：T2 + D-11、D-12 + K 矩阵（S-02 扩充）

**断言**：

- 总能量与文献差 < 2 mHa
- HF 交换混合系数与定义一致
- B3LYP 梯度有限差分一致

### T7 · 工作流 + 调度 + 复现

**前置 WP**：T1–T6 + P-01..P-08

**验收命令**：`qwb batch jobs/*.yaml --parallel 4` && `qwb replay <task_id>`

**断言**：

- 状态机所有合法转移被 proptest 覆盖
- kill -9 后重启恢复 queued 任务
- 结果目录无 `env.json` 时 `qwb inspect` 退出码 3

### T8 · GUI 端到端

**前置 WP**：T7 + P-09、P-10

**断言**：

- 从打开项目 → 建分子 → 选 PBE/def2-SVP/opt → 提交 → Job Monitor → 3D 结果，全程不崩溃不阻塞
- 大日志（> 100 MB）UI 不卡
- 崩溃后重启恢复未完成视图

---

## 8. 数据层详细要求（项目中心）

> **这一节是"为什么本项目不是玩具"的答案。** 数据层质量直接决定引擎可信度。

### 8.1 基组

**来源**：BSE 导出 + 一手文献核对。每个基组追溯原始 DOI 并核对 **H、C、O 至少 3 个元素**的指数/系数。**不得**从其他项目代码仓库复制文件。

**本地实现**：

- 内部统一 schema（YAML → `serde_yaml` → `BasisSet` struct）。
- 加载器：`qwb-registry` crate → `Registry` → `get_basis(name) -> BasisSet`。
- **枚举/球谐切换只有一处**：`BasisSet.angular` 字段；`Engine` 显式读取；禁止隐式默认。
- **归一化只有一处**：加载时一次性处理，运行时不转换。

**验证（CI）**：

1. schema 校验
2. 指数/系数与 BSE 逐项 diff（< 1e-12 相对）
3. 归一化：H2/STO-3G 重叠积分与文献一致
4. 球谐 vs 笛卡尔：d 壳层差异按理论预期
5. ECP-pairing：有 ECP 元素必须找到匹配基组，否则 `Registry::load_pair` 返回 `EngineError::Data`

### 8.2 ECP

- **来源**：Stuttgart / def2-ECP 官方 + BSE 导出。
- **陷阱显式声明**：`r^n` vs `r^(n-2)` 在 schema 中强制 `r_power_convention`，加载器统一转为内部约定。
- **配对校验**：`Registry::load_pair(basis, ecp)` 唯一入口。

### 8.3 泛函（最重，本地实现）

**设计**：

- 每个泛函拆成 `FunctionalTerm`（Rust trait，X-01 交付）。
- 数据侧只放参数与组合；代码侧放数学形式。
- 不允许参数散落；一个泛函不能在多处重写。

**实现清单**：

| 组件 | 数学形式 | 依赖 |
|---|---|---|
| `LdaSvwn5` | VWN5 + 自旋插值 | ρ |
| `GgaPbeX` | PBE 交换（κ=0.804, μ=0.21951） | ρ, σ |
| `GgaPbeC` | PBE 相关 | ρ, σ |
| `GgaB88` | Becke 88 | ρ, σ |
| `GgaLyp` | LYP 相关 | ρ, σ |
| `HfExchange` | K 矩阵混合 | 来自 S-02 |

**验证三道门禁**：

1. 有限差分导数：`dε/dρ`, `dε/dσ` 与数值差 < 1e-9（X-02 工具，CI 必跑）。
2. 单点值：水/苯/CO 固定基组下的总能量与文献对照 < 1 mHa（LDA/GGA）/ 2 mHa（B3LYP）。
3. 数值稳定性：低密度、超大梯度截断规则写进 `functional.yaml` 的 `stability` 字段，有单元测试。

### 8.4 XC 网格

- 参数数据化：`grid.yaml` 存等级 → 参数；代码不含硬编码点数。
- 收敛测试：每个 T2 分子相邻等级能量差 < 0.1 mHa，CI 断言。
- 陷阱显式：Becke 锐度、原子半径缩放全在数据里。

### 8.5 溶剂

- 第一阶段只做接口与数据，不做完整 PCM。
- 字段：`epsilon`、`temperature_K`、`atomic_radii`、`cavity_model`、`scale`。
- `cavity_model` 必须显式声明；引擎不实现时 loader 不报错，但 `Engine::run` 返回 `EngineError::Unsupported`。
- 验证：ε 与 NIST / 文献一致（水 78.3553 @ 298.15 K）。

### 8.6 数据治理

| 规则 | 机制 |
|---|---|
| 数据与代码分仓 | `qwb-data/` 独立仓库，独立 semver |
| 每条数据带 sha256 | `qwb-registry` 计算并比对 |
| 每条数据带 citation | schema required，缺失即拒 |
| 每条数据带 license | `commercial_ok` / `redistribute_ok` 显式字段 |
| 变更必须 bump version | CI 检查 `CHANGELOG.md` |
| 结果快照 | 每次计算写 `data_snapshot.json` |
| 发布清单 | `scripts/gen_data_licenses.py` 生成 `DATA_LICENSES.md` |
| **cargo-deny 联动** | `deny.toml` 的 `[licenses]` 映射数据许可；CI 拒未许可数据 |

---

## 9. 验证体系

三层，缺一不可：

1. **单元层**：每 WP 自带（积分对照、泛函导数、网格精度、ECP 矩阵元、状态机转移）。
2. **物理极限层**：氢原子解析解（V-01）、均匀电子气、HF 一致性（V-02）、网格 Σw 收敛。
3. **基准集层**：H2、H2O、CH4、NH3、CO、N2、苯（V-03）；对照实验几何/频率、高精度文献能量。

**CI 分层**：

- PR：`cargo fmt --check`、`cargo clippy --workspace -- -D warnings`、`cargo test --workspace`、`cargo deny check`、`cargo audit`、T1+T2、`pytest`（Python 侧）。
- 每日：T3–T6。
- 每周：完整基准集 + `cargo criterion` 归档 + diff 报警。

**门禁 G**：

- **G1 契约门禁**：K1–K6 冻结前，D/I/X/S/P 轨只允许脚手架。
- **G2 数据门禁**：D 轨每次合入必须有 citation + license + sha256；`cargo-deny` 许可映射一致。
- **G3 物理门禁**：新增泛函必须过 X-02 有限差分导数。
- **G4 集成门禁**：T 之前 WP 的验收必须全绿。
- **G5 复现门禁**：UI 可见结果目录必须含 `env.json`；CI 强制。
- **G6 Rust 门禁**：`#![deny(unsafe_code)]`（例外需 SAFETY 注释 + 人类批准）；`Cargo.lock` 变更必须带 PR 说明；不允许 `cargo update` 无记录变更。
- **G7 并行门禁**：所有 `rayon` 使用的归约必须**确定性**（按索引分块，不许 `reduce` 顺序依赖）；1 线程 vs N 线程 byte-identical。

---

## 10. Agent 执行协议

> 每个 agent 执行工作包前**必须**执行以下步骤。

### 10.1 开工前

1. 读 `docs/status.yaml`，确认该 WP 依赖全部 `done`。
2. 读 `qwb-contracts/` 中相关 K 文件（编号见 §6 每行）。
3. 检查产物是否已存在；若存在且验收通过，标 `done` 不重复。
4. 任一依赖未 done、契约不明确、或需改契约 → **停止并上报人类**。
5. 跑 `cargo check --workspace`；若基线编译失败，**停止并上报**（不许在红基线上叠代码）。

### 10.2 执行中

1. 只改本 WP 涉及的代码/数据。
2. 不得修改 K1–K6（必须停并上报）。
3. 不得删除或修改其他 WP 的产物。
4. 每次提交附件：`cargo fmt --check` + `cargo clippy --workspace -- -D warnings` + 该 WP 的验收命令输出摘要。
5. **公共 API 变更**（`pub` 项）必须在 PR body 里显式列出，触发 G1 复核。
6. **新增 crate 依赖**必须在 PR body 里说明理由，并进 `deny.toml` 白名单（人类批准）。

### 10.3 验收

1. 逐条执行 §6 中该 WP 的验收命令。
2. 全过 → `status.yaml` 写 `state: done` + `pr` + `ts`。
3. 任一失败 → `state: blocked` + 原因 + 日志路径。

### 10.4 禁止事项

- 不得在未冻结 K 时写实质逻辑。
- 不得静默修改数据文件。
- 不得在验收未过时推进下一 WP。
- 不得复制 libxc/pyscf 等参考实现代码进产品（CI 黑盒对照见 §11）。
- 不得引入 §2.3 未允许的第三方库进产品依赖树。
- 不得 `cargo add` 未审批的 crate（含 dev-dependency，因 supply chain 风险）。
- 不得用 `unsafe` 除非 SAFETY 注释 + 人类批准。
- 不得用 `rayon::reduce` 等不确定性归约进入热路径。

### 10.5 推荐的工作顺序（对 agent 极重要）

```
1. C 轨全部（C-01..C-06）
2. D-01, D-02, D-03, D-09, D-13
3. I-00, I-01..I-04
4. X-01..X-05
5. S-01..S-03
6. T1 集成通过
7. D-04, D-10 → T2
8. G-01..G-03 → T3
9. F-01..F-02 → T4
10. D-07, D-08, I-05 → T5
11. D-11, D-12 + S 扩充 → T6
12. P-01..P-08 → T7
13. P-09..P-10 → T8
14. V-04, V-05, V-06 → 收尾
```

---

## 11. 风险与门禁对照

| ID | 风险 | 缓解 | 门禁 |
|---|---|---|---|
| R-01 | 契约缺陷导致返工 | 契约先冻结 + `#[non_exhaustive]` 保护 | G1 |
| R-02 | 数据转录错误 | 逐项 diff BSE + 双人核对 + CI 断言 | G2 |
| R-03 | Rys vs Obara-Saika 选择 | I-03 开工前 1 天做 1D 基准定性判断，写进 `docs/choices.md` | — |
| R-04 | 泛函参数笔误 | 单点基准 + 导数有限差分双门禁 | G3 |
| R-05 | 网格精度不足 | 等级映射 + 收敛测试 + 数据化锐度参数 | — |
| R-06 | 球谐/笛卡尔隐性切换 | `angular` 字段唯一入口 + 单元测试 | — |
| R-07 | `rayon` 竞争 | TSan（`-Zsanitizer=thread` nightly）+ 单/多线程 identical | G7 |
| R-08 | 结果不可复现 | `env.json` 强制（含 toolchain + Cargo.lock sha）+ `replay` | G5, G6 |
| R-09 | 数据许可冲突 | `DATA_LICENSES.md` + `cargo-deny` 许可映射 | G2 |
| R-10 | 参考实现污染 | 只允许 CI 黑盒对照，代码不得复制；`grep` 门禁 | CI |
| **R-11** | **faer/nalgebra 性能不足** | `qwb-linalg` 抽象层 + `criterion` 基准守门；必要时接系统 BLAS（需人类批准） | V-06 |
| **R-12** | **pyo3 abi3 版本错配** | 冻结 `abi3-py39`；若下游要求升级到 py310，迁移成本已知 | G1 |
| **R-13** | **`rayon` 非确定归约** | 全工程统一 `DeterministicReduce` 辅助函数；1 vs N 线程 byte-identical | G7 |
| **R-14** | **HDF5 跨平台痛** | K4 换 zip + JSON（已定案）；若引入 `ndarray-npy` 需人类批准 | — |
| **R-15** | **编译时间过长** | workspace 分 crate（≤10）；feature gate 拆分（`faer`、`hdf5`、`python`）；CI 缓存 `sccache` | — |
| **R-16** | **`unsafe` 扩散** | `#![deny(unsafe_code)]` 默认；例外 crate 需 `// SAFETY:` + 人类批准 + 门禁脚本 | G6 |
| **R-17** | **crate 生态 supply chain** | `cargo-deny`（duplicates/advisories/licenses/sources）+ `cargo-audit`；新 crate 需人类批准 | G6 |
| **R-18** | **矩阵与 numpy 拷贝开销** | 初期全部拷贝（500×500 内可接受）；大矩阵留 TODO；接口不变 | — |
| **R-19** | **llvm 版本偏移影响数值** | `rust-toolchain.toml` 冻结；对比 CI 用同一 toolchain 镜像 | G6 |

---

## 12. 附录

### 12.1 目录结构（三仓）

```
qwb-contracts/
  schema/                 # K1
  engine-api/             # K2 (Rust crate: qwb-engine-api)
    Cargo.toml
    src/lib.rs
  studio/                 # K3/K5/K6 (schemas, python .pyi)
  CHANGELOG.md

qwb-data/
  basis/  ecp/  functional/  grid/  solvent/
  registry.yaml
  DATA_LICENSES.md
  CHANGELOG.md

qwb-core/                 # Cargo workspace
  Cargo.toml
  rust-toolchain.toml
  deny.toml
  crates/
    qwb-engine-api/       # 冻结接口 (K2)
    qwb-linalg/           # nalgebra/faer 抽象
    qwb-registry/         # 数据加载 (D-01)
    qwb-basis/
    qwb-integrals/
    qwb-xc/
    qwb-grid/
    qwb-scf/
    qwb-gradient/
    qwb-hessian/
    qwb-opt/
    qwb-freq/
    qwb-props/
    qwb-io/               # .qwb + log
    qwb-engine/           # 顶层实现
    qwb-cli/              # clap (K5)
    qwb-python/           # pyo3 (K6)
  benches/
  tests/                  # 集成测试 (T 级端到端)
  examples/

qwb-studio/               # Python 产品
  qwb_studio/{models,scheduler,gui}
  examples/
  tests/
  pyproject.toml

docs/
  status.yaml
  choices.md
  error-budget.md
scripts/
  gen_data_licenses.py
  check_unsafe.sh          # 白名单 unsafe 检查
  check_rayon_determinism.sh
```

### 12.2 命名

- 引擎内部**原子单位**；边界（CLI/API/UI）用 Å/eV 且显式字段后缀。
- 能量后缀 `_hartree`；长度 `_angstrom`；梯度 `_hartree_per_bohr`；Hessian `_hartree_per_bohr2`。
- 时间戳 ISO 8601 UTC。
- Rust crate 名统一 `qwb-*`；Rust 类型 PascalCase；函数 snake_case；公共 trait 以动词或名词明示用途。
- Python 绑定包名 `qwb_engine`；产品包名 `qwb_studio`。

### 12.3 提交规范

`<轨道>-<ID>: <一句话>`，例如 `I-02: add one-electron overlap integrals`.

PR 必须含：

- WBS ID
- `cargo fmt --check` 结果
- `cargo clippy --workspace -- -D warnings` 结果
- 该 WP 验收命令输出
- 若新增 crate：理由 + `deny.toml` 差异
- 若改公共 API：显式列出

### 12.4 状态文件格式

```yaml
C-01: {state: done, pr: "#12", ts: "2025-01-10T08:00:00Z"}
D-03: {state: in_progress, owner: agent-a}
D-04: {state: blocked, reason: "BSE export format changed", log: "logs/D-04.log"}
```

### 12.5 人类决策清单（agent 不得自行决定）

- Rust 生态版本策略：MSRV 冻结值、是否允许 `-Z` nightly feature
- `qwb-linalg` 最终后端：`nalgebra` vs `faer` 为默认（当前默认 `nalgebra`）
- K1 schema 生成方式：`schemars` 生成 vs 手写 serde struct
- pyo3 abi3 版本（`abi3-py39` vs `abi3-py310`）
- 首版泛函清单增删
- 溶剂是否第一阶段实现
- ECP 元素范围
- 数据仓库开源许可选择
- 参考实现（libxc 等）在 CI 中的使用边界
- R-03 的数值方案选择（默认 Obara-Saika）
- 是否引入 `ndarray-npy`（R-14）
- 是否允许为热点引入经批准的 `unsafe`（R-16）

---

**本文件即项目总纲。子文档 `docs/choices.md`、`docs/error-budget.md`、`DATA_LICENSES.md` 由相关 WP 产出并挂在本文件之下。**
