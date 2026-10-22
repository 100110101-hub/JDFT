# K4 · 文件格式（冻结）

> 一句话：**输入 YAML、项目是 Zip、结果目录是 5 个文件，没有 HDF5。**

## 1. 输入 `input.yaml`

就是 `qwb_engine_api::JobInput` 的 YAML 序列化（K2 类型 ≡ K4 格式，不允许两套字段）。

```yaml
molecule:
  atoms:
    - symbol: H
      position_angstrom: [0.0, 0.0, -0.37]
    - symbol: H
      position_angstrom: [0.0, 0.0, 0.37]
  charge: 0
  multiplicity: 1
basis_name: sto-3g
method_name: svwn5
grid_level: normal          # coarse | normal | fine | veryfine
solvent: vacuum
jobtype: sp                 # sp | opt | freq（K2 的 Sp/Opt/Freq 拼写也接受）
options:
  scf_max_iter: 128
  scf_energy_threshold_hartree: 1.0e-8
  scf_density_threshold: 1.0e-6
  diis_history: 8
  threads: 0                # 0 = 自动
  max_opt_steps: 100
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4
  freq_step_angstrom: 0.01
  ecp_name: null
```

约定：单位后缀强制（`_hartree` / `_angstrom` / `_bohr`）；未知字段**必须报错**
（`deny_unknown_fields`）；边界单位是 Å/Hartree，引擎内部 Bohr。

## 2. 结果目录（K3 冻结布局）

```
<task_id>/
  input.yaml
  env.json            # 缺它 = 不完整结果，UI 标红（G5）
  log.txt
  result.json         # JobResult（K3 result.schema.yaml）
  data_snapshot.json  # 与 result.json/env.json 里的 data_snapshot 同内容
  artifacts/
```

`qwb inspect <dir>` 的判据：目录存在 + `env.json` 存在且内容完整 + 其余 4 个必需
文件存在。任一不满足 → 退出码 `3`（K5）。

## 3. 项目容器 `.qwb`（Zip）

```
meta.json            # ProjectMeta：schema_version, name, created, app_version, task_count, notes?
tasks/<id>/input.yaml
tasks/<id>/result.json
tasks/<id>/env.json
tasks/<id>/log.txt
data_snapshot.json
artifacts/**
```

### 为什么不是 HDF5（R-14 已定案）
`hdf5` crate 依赖系统 libhdf5，Windows 分发痛苦；`hdf5-metno` API 不稳定。
Zip + JSON 是纯 Rust、跨平台、人类可探查、可 diff。**不再讨论 HDF5。**

### 结构性约束（加载器必须拒绝，不是“约定”）
1. 条目名必须是相对 POSIX 路径：禁止绝对路径、盘符、反斜杠、`..`、空段、目录条目。
2. 条目名不得重复。
3. `meta.json` 必须存在且可解析；它是容器里的第一个条目。
4. `meta.task_count` 必须等于 `tasks/` 下的目录数。
5. 不是合法 zip、或 zip 被截断 → 必须是错误，不能“尽力打开”。

### 字节可复现
相同内容必须导出**逐字节相同**的容器：条目时间戳固定 1980-01-01T00:00:00、
压缩方法固定 Deflate、条目顺序固定（`meta.json` 第一，其余按既定顺序）。
这支撑 `qwb replay` 的“结果 identical”门禁（G5）。

### 矩阵表示（K4 二选一：**list-of-list**，已定案）
`result.json` 里的矩阵/向量是嵌套 JSON 数组，**不使用 base64、不使用 nalgebra 的
私有 serde 布局**。理由：可 diff、可 grep、无外部解码步骤；代价是体积与精度往返，
由 `f64` 的 round-trip 保证（Rust 的 `f64` serde 是精确十进制往返）。

大矩阵的二进制快通道（`.npy` 或 `ndarray-npy`）**未引入**：需人类批准（R-14）。

## 4. 分子 `.xyz` / `.mol` / `.sdf`
结构读写由 Python 侧 `rdkit` 负责（§2.3 只允许它做结构读写）；引擎不解析这些格式。

## 5. 缓存 `.npz`
非契约必须；不得成为结果的唯一载体（结果必须自包含在结果目录里）。
