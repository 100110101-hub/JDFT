# QuantumWorkbench 使用手册

> 这份文档写给**要用它算东西的人**：怎么装、怎么跑、每个字段什么意思、结果文件怎么看、出问题怎么查。
> 规范本身（契约、验收判据）在 [`../project.md`](../project.md)（阶段一）与
> [`../project-phase2.md`](../project-phase2.md)（阶段二）；每个 WP 的**实测数字**在
> [`status.yaml`](./status.yaml) / [`status-v2.yaml`](./status-v2.yaml)；为什么这么做在
> [`choices.md`](./choices.md)；误差预算在 [`error-budget.md`](./error-budget.md)。
>
> 文中所有命令都在本机（Windows 11 + Rust 1.97.1 + Python 3.12、8 物理核 / 16 逻辑核）实跑过，
> 输出片段是真实输出。标「实测」的数字是本轮（第 386 轮）量的；其余数字都给出了它在账本里的出处。

---

## 目录

1. [五分钟上手](#1-五分钟上手)
2. [安装与构建](#2-安装与构建)
3. [输入文件：`input.yaml` 逐字段说明](#3-输入文件inputyaml-逐字段说明)
4. [命令行 `qwb`](#4-命令行-qwb)
5. [Python API（pyo3）](#5-python-apipyo3)
6. [图形界面 qwb-studio](#6-图形界面-qwb-studio)
7. [结果目录怎么看](#7-结果目录怎么看)
8. [数据仓 `qwb-data` 与校验](#8-数据仓-qwb-data-与校验)
9. [能算什么、不能算什么（诚实清单）](#9-能算什么不能算什么诚实清单)
10. [验收与门禁](#10-验收与门禁)
11. [开发与扩展](#11-开发与扩展)
12. [故障排查](#12-故障排查)
13. [常见问题](#13-常见问题)

---

## 1. 五分钟上手

```powershell
# ① 进仓库、建 Python 环境（只有 Python 侧与 pyo3 扩展需要它）
cd <仓库根>
python -m venv .venv
.venv\Scripts\python.exe -m pip install mypy numpy pydantic pytest maturin PySide6 vtk

# ② 跑第一个作业：H2 / STO-3G / SVWN5 单点能（实测 0.7 s）
cd qwb-core
cargo run --release -q -p qwb-cli --bin qwb -- run examples/h2_sto3g_svwn5.yaml --out results
```

```text
scf     converged = true
energy  -1.1570005110 Ha
task    results\h2_sto3g_svwn5
```

第一次跑会看到取数提示（只下这份输入要用的那几个文件，约 1 秒）：

```text
qwb: note: 本地缺 10 个第三方数据文件（basis/sto-3g.yaml, …, ecp/lanl2dz-zn.yaml），正在按需取回…
fetch   basis/sto-3g.yaml（missing）
saved   basis/sto-3g.yaml  sha256=fce3af14b8b3ea121844756117a2489b5f7859be452be02b54509dc20b8ff249
```

**为什么不是开箱即算**：`qwb-data/basis/` 与 `qwb-data/ecp/` 里有 **10 个文件是从
[Basis Set Exchange](https://www.basissetexchange.org/) 导出的第三方数据**，仓库不提交它们
（LANL2DZ 的 ECP 原始论文没有明确再分发条款），只提交配方与 sha256 钉。`run` / `batch` 会自动取回
**这份输入用到的**文件，取回时核对摘要钉，对不上就报错、不落盘。想提前一次备齐：

```powershell
qwb data install                                        # 10 个文件
python scripts\fetch_data.py --all                      # 等价（不需要 PYTHONPATH）
```

取回之后就是纯本地。细节见 §8.4。

结果在 `qwb-core/results/h2_sto3g_svwn5/`，K3 冻结布局：

```text
h2_sto3g_svwn5/
  input.yaml            # 这次算的输入（原样回写）
  result.json           # K2 的 JobResult：能量/轨道/偶极/日志/可选块
  env.json              # 复现环境（rustc、cargo.lock 摘要、数据快照摘要、OS…）
  data_snapshot.json    # 注册表里每个数据文件的 sha256（33 个）
  log.txt               # Reporter 输出 + 引擎日志
  artifacts/            # 自由形态产物：populations.json、homo.cube、lumo.cube、optimized.xyz…
```

再跑一个**几何优化**（H2 / STO-3G / PBE，解析梯度，实测 10.2 s）：

```powershell
cargo run --release -q -p qwb-cli --bin qwb -- run examples/h2_sto3g_pbe_opt.yaml --out results
```

```text
opt     3 steps, max|grad| = 3.436e-5 Ha/Bohr
geometry results\h2_sto3g_pbe_opt\artifacts\optimized.xyz
scf     converged = true
energy  -1.1520984592 Ha
task    results\h2_sto3g_pbe_opt
```

`artifacts/optimized.xyz` 是优化后的几何（标准 XYZ，Å）；`artifacts/optimization.json` 是轨迹
`[step, E, max|grad|]`。

---

## 2. 安装与构建

### 2.1 需要什么

| 组件 | 版本 | 用途 | 必需？ |
|---|---|---|---|
| Rust | **1.97.1**（仓库根 `rust-toolchain.toml` 固定） | 引擎与 CLI | 必需 |
| C 工具链 | MSVC Build Tools（Windows）/ gcc（Linux） | 链接 | 必需 |
| 网络 | 任意 | **首次**取回第三方基组/ECP（§8.4）；之后纯离线 | 首次必需 |
| Python | 3.10–3.12 | Python API、GUI、门禁里的 Python 侧 | 用 Python 面才必需 |
| WSL 的 `bash` | 任意 | 三个 bash 策略门禁（`scripts/verify.ps1` 会调） | 跑门禁才必需 |
| PySide6 / vtk | 见 §2.3 | GUI 与 3D | 用 GUI 才必需 |
| NVIDIA 驱动 + `libcublas` | CUDA 12/13 任一 | `qwb-gpu` 的 GEMM 快路 | 可选 |

> **不需要**：PySCF、libxc、ORCA、Gaussian、Psi4、ase 等任何电子结构库；也不需要 CUDA 工具链
> 才能**构建**。`qwb-gpu` 用 `libloading` 在**运行时**找 cuBLAS，没有 GPU/运行库时返回
> `Unavailable` 并退回 CPU（附录 A.3 的"CPU 永远是基准路径"）。
> PySCF 只在 `scripts/pyscf_*.py` 里当**外部尺子**产参考数字（§10.3）。

### 2.2 只算东西（不跑门禁）

```powershell
cd qwb-core
cargo build --release            # 约 3–5 分钟；产物 target/release/qwb.exe
cargo test --release -- --ignored  # 可选：把慢档（频率/优化/Hessian/长基准）也跑一遍
```

调试构建（`cargo build`）慢 10–50 倍，**算真实体系请一律 `--release`**。本手册里的耗时都是 release。

数据不用手动准备：第一次 `run` 会把这份输入需要的第三方文件取回来（§8.4）。

### 2.3 Python 侧怎么装

仓库根的 `.venv` 被三处共用：门禁里的 Python 检查、`qwb_studio` 的测试、pyo3 扩展的安装位置。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install mypy numpy pydantic pytest maturin PySide6 vtk
```

**pyo3 扩展 `qwb_engine`** 不是从 PyPI 装的，要现场构建（`abi3-py39`，与 K6 的 `.pyi` 对齐）：

```powershell
cd qwb-studio\rust\qwb-py
..\..\..\.venv\Scripts\python.exe -m maturin build --release -o dist
..\..\..\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
```

改过 Rust 侧（尤其是 `qwb-engine` / `qwb-scf` / `qwb-contracts`）之后**必须重建一次 wheel**，
否则 Python 侧跑的仍是旧引擎 —— 这是最常见的"改了没生效"。门禁里的
`P-06: build + install qwb_engine` 每一步都重装，所以门禁跑过的树不会踩这个坑。

### 2.4 让 Python 找得到 `qwb_studio`

没有 `pip install -e`，靠 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = '<仓库根>\qwb-studio'
```

否则 `python -m qwb_studio …` 报 `No module named qwb_studio`。

---

## 3. 输入文件：`input.yaml` 逐字段说明

一个作业 = 一个 YAML（K4 的 `input.yaml`）。字段名与 K2 的 `JobInput` **一一对应**，
未知字段会被 `deny_unknown_fields` 直接拒绝（不静默忽略，退出码 1）。

```yaml
molecule:
  atoms:
    - symbol: O                  # 元素符号（区分大小写；必须是该基组覆盖的元素）
      position_angstrom: [0.0, 0.0, 0.0]
    - symbol: H
      position_angstrom: [0.0, 0.757, 0.586]
    - symbol: H
      position_angstrom: [0.0, -0.757, 0.586]
  charge: 0                      # 分子总电荷（整数）
  multiplicity: 1                # 自旋多重度 2S+1

basis_name: def2-svp             # 必须是 qwb-data/registry.yaml 里注册过的名字（§8.1）
method_name: pbe                 # hf / svwn5 / pbe / pbe0 / b88 / m06-l / m06-2x（§9.1）
grid_level: normal               # coarse | normal | fine | veryfine
solvent: vacuum                  # vacuum 或 qwb-data/solvent/ 里的溶剂名（§3.7）
jobtype: sp                      # sp | opt（freq 未接线 → 退出码 4）

options:
  # ── 阶段一的 9 个数值控制字段：**全部必填**（没有隐藏默认值，漏一个就报缺字段）──
  scf_max_iter: 128                        # SCF 最大步数（退路也不会超过它）
  scf_energy_threshold_hartree: 1.0e-8     # |ΔE| 收敛判据
  scf_density_threshold: 1.0e-6            # max|ΔP| 收敛判据
  diis_history: 8                          # DIIS 子空间长度（真的接到了 DIIS 上）
  threads: 0                               # 0 = 自动；本构建只记账，见 §9.4
  max_opt_steps: 100                       # jobtype: opt 的最大步数
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4   # opt 的力判据（最大分量）
  freq_step_angstrom: 0.01                 # 预留给 freq（尚未接线）
  ecp_name: null                           # null = 全电子；有 ECP 时写名字，见 §3.3

  # ── 阶段二加性字段：可省；省略时行为与阶段一逐位相同 ──
  scf_type: null                           # null / rhf / rks / uhf / uks / rohf，见 §3.4
  mp2: null                                # {kind: mp2 | ri-mp2, aux_basis_name: ...}，见 §3.5
  excited_states: null                     # {n_states: 5, method: cis | tda | tddft}，见 §3.6
  solvent_model: null                      # vacuum | cosmo | smd（pbf → 退出码 4），见 §3.7
  dispersion: null                         # {data: d3bj | d3zero | d4}，见 §3.8
  epr: false                               # 开壳层的超精细耦合，见 §3.9
  numerics: null                           # 精度等级/近似路径：**一律退出码 4**，见 §3.10
```

两条容易踩的：

* `options` 里**阶段一的 9 个字段都是必填**（K2 的 `TaskOptions` 没有 `serde(default)`），
  只写 `scf_max_iter` 会得到 `options: missing field 'scf_energy_threshold_hartree'`（退出码 1）；
* 阶段二的字段是**可选**的，而且"写了就一定生效或一定拒绝"，没有"写了但被忽略"这条路
  （§3.10 的 `numerics` 就是"拒绝"那一类）。

### 3.1 几何与单位

* 坐标单位是 **Å**（`position_angstrom`），引擎内部统一转 Bohr（转换只发生在入口一次）。
* 原子顺序 = 输出里轨道/梯度/超精细耦合的顺序；换顺序能量不变，但矩阵形状随之变化。
* 元素必须被所选的基组覆盖，否则报 `data load failed`（退出码 3）。实测覆盖范围（`qwb data list`）：

| 基组 | 元素 | 备注 |
|---|---|---|
| `sto-3g` | H C N O F | 第三方（BSE），按需取回 |
| `def2-svp` | H C N O | 第三方，按需取回 |
| `def2-tzvp` | H C N O F | 第三方，按需取回 |
| `6-311g(d,p)` | H C N O | 第三方，按需取回 |
| `lanl2dz` | H O Cu Zn | 第三方；配 ECP 时是价基（§3.3） |
| `def2-svp-ri` | H C N O | **RI 辅助基**（`role: ri-j`），不能当轨道基组 |
| `def2-tzvp-ri` | H C N O F | RI 辅助基，含 g（l=4），球谐只到 l=3 → 不能当辅助基 |
| `def2-universal-jkfit` | H C N O F | `role: ri-jk`，同上含 g |

三个 RI 辅助基不是轨道基组：把 `basis_name` 写成它们会得到"元素不覆盖"以外的怪结果，别这么用。
当前**只有 `def2-svp-ri` 能被真正用作 RI-MP2 的辅助基**（另两个含 l=4，建辅助基时以
`Unsupported` 明确报错，不是静默跳过）。

### 3.2 幽灵原子（ghost / counterpoise）

给原子加 `ghost: true`：**只提供基函数，不提供核电荷与电子**。这是 counterpoise（BSSE）校正的
标准做法：把伙伴分子放在原位置、换成 ghost，单体的能量里就只剩"伙伴的基函数"这一项基组不完备效应。

```yaml
molecule:
  atoms:
    - symbol: O
      position_angstrom: [0.0, 0.0, 0.0]
    - symbol: H
      position_angstrom: [0.0, 0.757, 0.586]
    - symbol: H
      position_angstrom: [0.0, -0.757, 0.586]
    - symbol: O                  # 伙伴分子的 O，只带基函数
      position_angstrom: [0.0, 0.0, 3.0]
      ghost: true
```

实测（上面这个输入，H2O/STO-3G/HF）：`energy -74.9630370286 Ha`，而同一分子不带 ghost 是
−74.9631464092 Ha —— ghost 的基函数让能量**变低**（变分原理），这正是 BSSE 的符号。

规则（都由测试钉住）：

* 默认 `false`；`false` 时**不写进** JSON/YAML（旧输出逐字节不变）；
* 只放行 `jobtype: sp`：`opt`/`freq`、色散、ECP、溶剂一律**显式拒绝**（退出码 4，消息里带
  `counterpoise` 与 `sp`）—— 那些路径要逐个审过核相关量才能带 ghost 跑，没审过就放过会得到
  "看起来正常、其实把幽灵当真实原子"的数；
* 全是 ghost 的分子被 `Molecule::validate` 拒绝（没有电子）；
* 布居与 cube 对 ghost 自动跳过（Z=0），偶极的核项也跳过；
* Python API（pyo3）**还没有** ghost 参数，这条只能走 `input.yaml` / Engine API。

### 3.3 ECP（有效核心势）

`ecp_name` 指向 `qwb-data/ecp/` 下的一个条目：`lanl2dz-cu`（Cu，10 芯电子，`z_eff = 19`）或
`lanl2dz-zn`（Zn，18 芯，`z_eff = 12`）。要点：

* 基组必须是**同一族**的 `lanl2dz`（`Registry::load_pair` 是唯一配对入口，配对按元素校验）；
* 点电荷吸引用 `z_eff`，超出部分作为矩阵元加进 `H_core`；
* **带 ECP 的作业只支持 `sp`**：`opt`/`freq` 以退出码 4 明确拒绝（ECP 对核坐标的导数还没实现）；
* MP2 也不能配 ECP（占据空间要按 `z_eff` 数，M3 不在范围内）。

最小例子（`qwb-core/examples/cuh_lanl2dz_pbe.yaml`）：

```yaml
molecule:
  atoms:
    - symbol: Cu
      position_angstrom: [0.0, 0.0, 0.0]
    - symbol: H
      position_angstrom: [0.0, 0.0, 1.46]
  charge: 0
  multiplicity: 1
basis_name: lanl2dz
method_name: pbe
grid_level: normal
solvent: vacuum
jobtype: sp
options:
  scf_max_iter: 200
  # …其余 8 个必填字段…
  ecp_name: lanl2dz-cu
```

### 3.4 `scf_type`：自旋处理与组合规则

`scf_type` 是**可选**字段；省略 = 阶段一语义（闭壳层 RHF/RKS）。能用的是五个值：

| 值 | 含义 | 与 `method_name` | 多重度 | `jobtype` |
|---|---|---|---|---|
| `rhf` | 闭壳层 RHF | 必须是 `hf` | 必须 1 | sp / opt |
| `rks` | 闭壳层 RKS（DFT） | 不能是 `hf` | 必须 1 | sp / opt |
| `uhf` | 非限制性 UHF | 必须是 `hf` | **> 1** | sp / opt |
| `rohf` | 限制性开壳层 ROHF（纯自旋） | 必须是 `hf` | **> 1** | sp / opt |
| `uks` | 非限制性 UKS（自旋极化 DFT） | 不能是 `hf` | **> 1** | **sp only** |

**显式写了 `scf_type` 时，多重度必须与它一致**（退出码 1）：

* `uhf`/`rohf`/`uks` 配 `multiplicity: 1` → 拒绝；`rhf`/`rks` 配 `multiplicity > 1` → 拒绝；
* `uhf`/`rohf` 配 DFT 泛函、`uks` 配 `hf` → 拒绝（HF 变体 ↔ `hf`，KS 变体 ↔ 泛函）；
* **省略** `scf_type` 且 `multiplicity > 1` → 拒绝（缺省是闭壳层，它代表不了自旋态，见下）。

> **省略 `scf_type` 且 `multiplicity > 1` → 直接拒绝（退出码 1）**。缺省路径是闭壳层，而
> 闭壳层在物理上代表不了自旋态（它只检查"偶数、正的电子数"）。第 387 轮之前这里是个**静默
> 降级**：O2 三重态（16 个电子，偶数）会被算成**能量高 0.616 Ha、且没有 `spin` 块**的闭壳层
> 态、退出码还是 0（实测 −147.0158990576 Ha；同一几何写 `scf_type: rohf` 是 −147.6321910352
> Ha）。现在同一份输入得到退出码 1 和一句点名 `options.scf_type` 的提示 —— 引擎**不替用户猜**
> `uhf`/`rohf`/`uks`（UHF 有自旋污染、ROHF 是另一套占据约定，选哪个是用户的决定）。
> **开壳层仍然必须显式写 `scf_type`。** 单重态的老路径完全不受影响。

开壳层的最小例子：

```yaml
# O2 三重态 / STO-3G / ROHF（examples/ 里暂时没有，下面这份可直接跑）
molecule:
  atoms:
    - symbol: O
      position_angstrom: [0.0, 0.0, 0.0]
    - symbol: O
      position_angstrom: [0.0, 0.0, 1.208]
  charge: 0
  multiplicity: 3
basis_name: sto-3g
method_name: hf
grid_level: coarse          # HF 不碰 XC 网格；字段只是为了完整
solvent: vacuum
jobtype: sp
options:
  scf_max_iter: 200
  scf_energy_threshold_hartree: 1.0e-8
  scf_density_threshold: 1.0e-6
  diis_history: 8
  threads: 0
  max_opt_steps: 100
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4
  freq_step_angstrom: 0.01
  ecp_name: null
  scf_type: rohf
```

实测（第 386 轮）：ROHF 给 `−147.6321910352 Ha`、`spin` 块 `s_squared = 2.0`（精确）、
`spin_contamination = 0.0`；同一个体系 `scf_type: uhf` 是 `−147.6339696424 Ha`、
`s_squared = 2.003411`（UHF 有自旋污染）。ROHF 的能量高于 UHF 是**正常**的（UHF 变分更松）。

三条要知道的事实：

1. **UHF 不是纯自旋态**：O2 的 `⟨S²⟩ = 2.0034`（不是 2.0）、OH 是 `0.7533`（不是 0.75）。
   `spin_contamination` 就是差值；纯自旋请用 `rohf`（那时 `⟨S²⟩` 精确等于 `S(S+1)`，
   因为它描述的是 CSF 而不是单个行列式）。
2. **ROHF 的 `mo_energies_hartree` 是等效 Fock 的本征值**，不是 Koopmans 轨道能；
   只有一套空间轨道（`mo_coeffs` 也是那一套）。
3. **初猜决定解**：O2 的 UHF 有两个 SCF 解（差 0.255 Ha）。默认用**自旋平均的原子密度**
   初猜（SAD），落在变分极小；`initial_guess = hcore` 会落到较高的那个（PySCF 用
   `init_guess='hcore'` 也是同一个数）。NH2 的 90 mHa 之谜最后就是这条（见 choices 一百七十八）。
4. 开壳层结果**不写** `artifacts/populations.json` 与 `homo.cube`/`lumo.cube`，并在 stdout
   说明原因：K2 的 `mo_coeffs` 只有 α 轨道集，拿它当"双占据"会算出一个错的 Mulliken 电荷 ——
   宁可不写，也不静默写错。`mo_energies_hartree`/`mo_coeffs` 对 UHF/UHS 是 **α** 那一套，
   β 的轨道能写进 `log.txt`。

### 3.5 相关能：`mp2` / `ri-mp2`

在闭壳层参考（`rhf`/`rks`，缺省就是它）之上叠相关能，在 `options.mp2` 里写 `kind`：

```yaml
options:
  mp2:
    kind: mp2                 # canonical：4 中心积分，无拟合误差
```

```console
$ qwb run examples/h2o_mp2_def2svp.yaml
$ python -c "import json;print(json.load(open('results/h2o_mp2_def2svp/result.json'))['correlation'])"
{'method': 'mp2', 'e2_hartree': -0.203643462358, 'e_total_hartree': -76.164613453426}
```

RI-MP2（密度拟合）换成 `kind: ri-mp2` + `aux_basis_name`（当前只有 `def2-svp-ri` 可用）：

```yaml
options:
  mp2:
    kind: ri-mp2
    aux_basis_name: def2-svp-ri
```

```console
$ qwb run examples/h2o_ri_mp2_def2svp.yaml
$ python -c "import json;print(json.load(open('results/h2o_ri_mp2_def2svp/result.json'))['correlation'])"
{'method': 'ri-mp2', 'e2_hartree': -0.203558201565, 'e_total_hartree': -76.164528192633,
 'aux_basis_name': 'def2-svp-ri', 'ri_error_hartree': 8.526079280285e-05}
```

约定：

* **相关能与 SCF 用同一个基组、同一份 2e 张量**，没有"MP2/cc-pVTZ 但 SCF 用 def2-SVP"这种混搭；
  `correlation.e_total_hartree = energy_hartree + e2_hartree`（`energy_hartree` 仍是 SCF 能量）。
* **无冻结核**（所有占据轨道都进相关计算）；参考态必须是闭壳层，`jobtype` 只能是 `sp`
  （MP2 梯度没实现）；带 ECP 时拒绝。
* **拟合误差必须落盘**：`ri_error_hartree = |E2(RI) − E2(canonical)|`，引擎会用同一份 4 中心张量
  再算一遍 canonical 来填它（K10 第 ③ 条：近似必须留痕）。所以 RI 作业的墙钟**比纯 RI 慢**。
* **精度**：H2O/def2-SVP + def2-SVP-RI 的误差是 **0.085 mHa**；苯上同一个辅助基是 0.47 mHa
  （PySCF 的 RI-MP2 同值）—— 这是辅助基的性质，不是实现的错。
* **没有性能优势（实测后降级）**：水/def2-SVP 上 4 指标 J 比 RI-J 更快（IN-04：0.172 s vs
  0.349 s），苯上 canonical MP2 7.14 s vs RI-MP2 48.32 s。所以 RI-MP2 可用但**不宣称更快**。

canonical MP2 的精度（`scripts/pyscf_mp2_reference.py` 的 PySCF 2.14.0 参考，同几何、同基组）：

| 体系/基组 | 本引擎 E_corr | PySCF E_corr | 差 |
|---|---|---|---|
| H2O/STO-3G | −0.0356083055 | −0.0356083061 | 6e-10 |
| H2O/def2-SVP | −0.2036434624 | −0.2036434634 | 1e-9 |
| H2O/def2-TZVP | −0.2726346686 | −0.2726346708 | 2e-9 |
| H2/STO-3G | −0.0131380733 | −0.0131380736 | 3e-10 |
| H2/def2-SVP | −0.0262678128 | −0.0262678129 | 1e-10 |
| H2/def2-TZVP | −0.0277524219 | −0.0277524220 | 1e-10 |

### 3.6 激发态：`cis` / `tda` / `tddft` + UV/Vis 谱

`options.excited_states` 一开，SCF 之后就算激发态，并把跃迁展宽成 UV/Vis 谱：

```yaml
options:
  excited_states:
    n_states: 5
    method: tda            # cis | tda | tddft
```

```console
$ qwb props examples/h2o_tddft.yaml
spectrum …\h2o_tddft\artifacts\spectrum.csv (274 points, HWHM = 0.400 eV)
scf     converged = true
energy  -74.9631464092 Ha
state 1 0.485084 Ha  13.1998 eV  f = 0.003519
state 2 0.555811 Ha  15.1244 eV  f = 0.000000
state 3 0.617584 Ha  16.8053 eV  f = 0.077880
state 4 0.705085 Ha  19.1863 eV  f = 0.059186
state 5 0.810171 Ha  22.0459 eV  f = 1.162083
task    …\h2o_tddft
```

* `method: cis` 与 `method: tda` 在 HF 参考下是**同一个东西**（A 矩阵只有
  `2(ia|jb) − (ij|ab)`，没有 XC kernel）；`method: tddft` 是含 B 块的完整 RPA/TDHF。
  H2O/STO-3G 的 TDHF 前三态是 13.1580 / 15.1099 / 16.6954 eV（`examples/h2o_tddft_tdhf.yaml`）。
* **目前只支持 `hf` 参考**。DFT 参考要 XC kernel `f_xc`（FX-03，blocked），写了会以退出码 4
  拒绝并点名 FX-03 —— 不允许把 CIS 的数标成 `tda` 交出去。
* 只支持闭壳层 + `jobtype: sp`。
* 结果：`excited_states` 块（`method` / `n_states` / `transitions` / `sum_rules`）+
  `uv_vis_spectrum` 块（`gaussian_hwhm_ev` = 0.4、`grid_step_ev` = 0.05、`points`）+
  `artifacts/spectrum.csv`。展宽参数来自数据（`qwb-data/numerics/numerics.yaml` 的
  `spectrum_defaults`），不写死在代码里。
* `sum_rules` 里报 `trk_oscillator` 与 `n_electrons`：TRK 和规则在截断态数下**不可能**闭合，
  判据是"它必须被报出来"，不是"它必须是 0"。

### 3.7 隐式溶剂：COSMO / C-PCM

写 `solvent: <溶剂名>` 就进溶液相；`solvent_model` 可省（缺省用溶剂数据里的 `cavity_model`，
当前 8 个溶剂都是 `pcm`，即 C-PCM）：

```yaml
solvent: water
jobtype: sp
options:
  solvent_model: cosmo     # 可省；写了会与溶剂数据交叉校验
```

实测（`examples/h2o_cosmo.yaml`，H2O/STO-3G/HF，1.6 s）：`energy -74.9747389362 Ha`
（气相 −74.9631464092），`result.json` 的 `solvation` 块是 K8 的：

```json
{"model": "cosmo", "solvent_name": "water", "total_hartree": -0.011592526967646677,
 "electrostatic_hartree": -0.011592526967646677,
 "surface_area_angstrom2": 36.4581576121684, "surface_volume_angstrom3": 19.388098029035742,
 "n_tesserae": 410, "iterations": 33, "converged": true}
```

`E_solv − E_vac = −0.0115925 Ha = −7.274 kcal/mol`（实验 −6.32，判据 1.5 kcal/mol 内；
差的那 0.95 就是缺的非极性项，SV-03）。空腔碎片数随原子数自动降采样（上限 1500），
降采样会写进日志，不静默改变精度。

四条现状规则（都由测试钉住）：

* **只支持 `jobtype: sp`**：溶剂化的梯度/优化属 SV-02，`opt` 以退出码 4 拒绝；
* `solvent_model: smd` **能算**（`sp`，第 387 轮接线）：静电用同一套 C-PCM 表面方程、
  半径换成 SMD 的参数化半径（eq. 16），能量里加**一项**非极性 `G_CDS`（eq. 4.77）——
  它可分成 `Σ σ_j A_j` 与 `σ^[M] Σ A_j` 两半，不是"空化/色散/排斥三项"
  （K8 的那个说法已订正，见 §7 的 `solvation` 块与 `docs/choices.md` 一百八十二）；
  `solvent_model: pbf` → 退出码 4（SV-04）；
* `solvent: vacuum` 配 `solvent_model: cosmo` 是自相矛盾，解析阶段就报错（退出码 1）；
  反过来 `solvent: water` 配 `solvent_model: vacuum` 也一样；
* 半径有两个来源，**不许混**：`solvent_model: smd` 一律用 SMD 的参数化半径（eq. 16，缩放 1.0），
  与数据里的 `radii_set` 无关（那是 COSMO 的键）；其余情况按数据的 `radii_set`
  （`radii_set: smd` 时也用参数化半径，由 `sv03_smd_radii_cosmo.rs` 直接测）。
  **当前没有溶剂文件声明 `radii_set: smd`**，所以那条数据分支只有测试在走；
  `solvent_model: smd` 这条由 `sv03_smd_engine.rs` 的 `smd_switches_the_cavity_to_the_parameterised_radii`
  钉住（甲苯里的 C：Bondi 1.70 vs SMD 1.85，面积 47.753 → 49.545 Å²）。

### 3.8 色散校正：`d3bj` / `d3zero` / `d4`

色散是**可加项**：`result.energy_hartree = SCF 能量 + dispersion_energy_hartree`，
两部分都落盘，所以"加了哪些项"可核对。`data` 缺省是 `d3bj`。

```yaml
options:
  dispersion:
    data: d3bj              # d3bj | d3zero | d4
    functional: null        # 省 = 用作业自己的 method_name；阻尼参数按泛函拟合，必要时可覆盖
```

实测（H2O/STO-3G/PBE0，同一几何）：

| 数据 | `E_scf` / Ha | `E_dispersion` / Ha | `energy_hartree` / Ha |
|---|---|---|---|
| `d3bj` | −75.245574238503 | −2.768645e-4 | −75.2458511030 |
| `d4` | −75.245574238503 | −1.562629e-4 | −75.2457305014 |

* 三套数据在门禁里逐项对上游源码重建（`gen_d3_params.py --check`）并对官方实现
  （s-dftd3 / dftd4）的参考值（`validate_d3_data.py`，能量与解析梯度）。
* `d3zero` 的 `R0_ij` 是**查表值**（成对 vdW 半径），不是两原子半径相加；M06 家族在上游
  只有这一套参数，所以 M06-2X-D3 的断言走它。
* `d4` 需要 EEQ 电荷（每个参考体系一对 `alpha(iω)` 在加载时现算），数据形状与 D3 不同，
  有独立的 K1 类型（`d4dispersion.schema.*`）。M06-2X 在上游**没有** D4 参数，文件里写的是
  `unsupported_functionals`，不编造数值。
* **只支持 `jobtype: sp`**：优化器不认色散力，`opt` + 色散以退出码 4 拒绝（不允许报一个
  优化器从未最小化过的总数）。

### 3.9 EPR / 超精细耦合

`options.epr: true` 在**开壳层**单点上报自旋密度与各向同性超精细耦合（Fermi 接触项）：

```yaml
molecule: { atoms: [...O2...], charge: 0, multiplicity: 3 }
method_name: hf
options:
  scf_type: uhf
  epr: true
```

O2/STO-3G/UHF 实测：`epr` 块每个核一个 `{nucleus_index, symbol, spin_density_au,
a_iso_mhz, a_iso_gauss}`，两核 `a_iso(O) = −81.4773 MHz = −29.0732 G`。UHF/UKS/ROHF 都支持
（ROHF 的自旋密度 = `D_o`，是精确的）。

* 闭壳层体系、非 `sp` 作业 → 退出码 1（自旋密度恒为零，报出来全是 0 等于静默失败）；
* 各向异性（偶极–偶极）项与实验基准还没做完 → ST-05 仍 blocked；
* 常数用 21 cm 线锚点定标，直算值作为独立测试保留（≤ 0.5%），两者不一致就说明常数表或公式形状写错。

### 3.10 `numerics` 与其他预留字段（**契约已就位，引擎显式拒绝**）

| 字段 | 现状 | 退出码 | 归属 |
|---|---|---|---|
| `options.numerics` | 精度等级 1..5 + 路径（`analytic`/`ri`/`pseudospectral`）能解析，但**没有任何代码在读** | 4 | IN-03 / IN-04（K10 门禁） |
| `options.solvent_model: pbf` | 同上 | 4 | SV-04 |
| `jobtype: freq` | `hessian::numerical_hessian` 存在且有测试，但没接进作业分发 | 4 | M7 / F-01 |
| `qwb ts` / `irc` / `scan` | K5 只落签名 | 4 | M6 / RP-01..03 |

"显式拒绝"是这个项目的硬规矩：任何"写进输入但被静默忽略"的字段都算缺陷。相应地，结果目录里
**已经会出现**的块是：`spin`、`correlation`、`excited_states`、`uv_vis_spectrum`、`solvation`、
`dispersion_energy_hartree`、`epr`、`multipoles`（见 §7.2）；**还不会出现**的是
`nmr_shieldings`（RS-04）、`polarizability`（RS-04/RS-08 的其余部分）、`numerics_report`（K10）。

### 3.11 从哪抄模板

`qwb-core/examples/` 下 19 个文件覆盖了常见组合（耗时是本轮实测或账本记录，标"慢档"的是分钟以上）：

| 文件 | 体系 | 作业 / 特点 |
|---|---|---|
| `h2_sto3g_svwn5.yaml` | H2 / STO-3G / SVWN5 | sp（最快，实测 0.7 s） |
| `h2o_sto3g_svwn5.yaml` | H2O / STO-3G / SVWN5 | sp（O 上有 p 壳层） |
| `h2_sto3g_pbe_opt.yaml` | H2 / STO-3G / PBE | opt（实测 10.2 s，3 步） |
| `h2o_def2svp_pbe_opt.yaml` | H2O / def2-SVP / PBE | opt（慢档） |
| `cuh_lanl2dz_pbe.yaml` | CuH / LANL2DZ + ECP / PBE | sp（ECP） |
| `zno_lanl2dz_pbe.yaml` | ZnO / LANL2DZ + ECP / PBE | sp（SCF 需要退火退路，慢档） |
| `o2_uhf_sto3g.yaml` | O2 三重态 / STO-3G / UHF | sp（实测 0.5 s） |
| `o2_uks_pbe_sto3g.yaml` | O2 三重态 / STO-3G / UKS-PBE | sp（grid-fine，实测 11.8 s） |
| `oh_uks_pbe_sto3g.yaml` | OH 双峰态 / STO-3G / UKS-PBE | sp（grid-fine） |
| `h2o_cosmo.yaml` | H2O / STO-3G / HF + COSMO | sp（实测 1.6 s） |
| `h2o_mp2_def2svp.yaml` | H2O / def2-SVP / canonical MP2 | sp（实测 0.7 s） |
| `h2o_ri_mp2_def2svp.yaml` | H2O / def2-SVP / RI-MP2 | sp（实测 1.2 s，含 canonical 对账） |
| `h2o_tddft.yaml` | H2O / STO-3G / TDA | `qwb props`（5 态 + 谱） |
| `h2o_tddft_tdhf.yaml` | H2O / STO-3G / TDHF | `qwb props`（含 B 块） |
| `benzene_tddft.yaml` | 苯 / STO-3G / TDA | `qwb props`（实测 8.6 s） |
| `h2o_pbe0_d3bj_sto3g.yaml` | H2O / STO-3G / PBE0 + D3(BJ) | sp（色散可加项） |
| `h2o_m062x_def2svp.yaml` | H2O / def2-SVP / M06-2X | sp（meta-GGA；**grid-fine 不收敛，见下**） |
| `benzene_dimer_d3.yaml` | 苯二聚体 / def2-SVP / M06-2X + D3(0) | sp（**小时级**，不是快速验证） |
| `benzene_monomer_d3.yaml` | 苯单体（二聚体的前 12 个原子） | sp（同上） |

`h2o_m062x_def2svp.yaml` 的现状要特别说明：它写的是 `grid_level: fine`，而**判据是在
coarse/normal 上标定的**。实测（第 386 轮，release）：normal 格点 **12 步 / 9.5 s** 收敛到
`−76.3231969 Ha`（与账本记的 PySCF 值差 5e-6 Ha 内）；fine 格点每步只贵约 1.2×，但同一个初猜下
**跑满 20 步仍未到 1e-8**（第 12 步 `dE` 仍是 5.6e-5，normal 在 12 步已到 1.9e-7）——
fine 档在本体系上收敛明显更慢，整跑是分钟以上到十几分钟量级。要复现判据数字请把 `grid_level`
临时改成 `normal`；把 meta-GGA 的格点收缩真正做快是 EX-03，仍 blocked（CPU 侧要动位级回归、
GPU 侧要新的 unsafe 面，都需要人批准）。

---

## 4. 命令行 `qwb`

```text
qwb run      执行一个 input.yaml
qwb batch    批量执行一个目录下的 *.yaml
qwb inspect  检查一个结果目录或 .qwb 项目
qwb replay   用记录的环境重算并比对（复现）
qwb data     数据仓库操作
qwb props    性质与响应（激发态）；`--states` 覆盖态数
qwb compare  多任务对比表（能量 / ΔE / 谱峰 + CSV 导出）
qwb ts|irc|scan  过渡态 / IRC / 扫描（K5 只落签名，退出码 4）
```

### 4.1 退出码（K5 冻结）

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 输入错误（YAML 语法、字段缺失/越界、多重度与 `scf_type` 矛盾…） |
| 2 | 收敛失败（SCF 或几何优化没到判据） |
| 3 | 数据校验失败（基组/泛函名不存在、sha256 不符、缺 `env.json`） |
| 4 | 内部错误 / 不支持的 feature（`freq`、ECP 的 opt、`smd`、`numerics`…） |

脚本里判成功请只看退出码，别 grep stdout。

### 4.2 `qwb run` —— 跑一个作业

```powershell
qwb run <INPUT_YAML> [--out <OUT>] [--threads <N>]
```

* `--out` 默认 `results`；结果目录名 = 输入文件的 **stem**（`h2.yaml` → `results/h2/`）。
* `--threads` 覆盖 `options.threads`，但本构建只记账（§9.4）。
* 成功时 stdout 是给人看的摘要，**真相在结果目录里**：

```text
scf     converged = true
energy  -1.1570005110 Ha
task    results\h2_sto3g_svwn5
```

作业带 `excited_states` 时还会多一行 `spectrum …`（谱曲线路径），但**不打印**每条跃迁 ——
跃迁要看 `result.json` 或用 `qwb props`。

### 4.3 `qwb batch` —— 批量

```powershell
qwb batch "jobs/*.yaml" --parallel 4 --out results
```

* 队列写在 `<out>/queue.json`（JSON 数组，每项 `{input, task_id, state}`，原子写）。
* `kill -9` 之后重跑同一条命令：**`queued` 的接着做**；停在 `running` 的**不会**回到 `queued`
  （K3 没有那条边），会报告出来并以非零退出码结束 —— 用新 task id 重提。
* `--parallel` 是**进程级**并发（每个作业一个子进程），不是线程。

### 4.4 `qwb inspect` —— 看结果目录 / `.qwb`

```powershell
qwb inspect results\h2_sto3g_svwn5
qwb inspect projects\test.qwb
```

检查 K3 必需文件与 `env.json` 完整性（G5）；缺 `env.json` 时**退出码 3**。实测输出：

```text
kind    result directory
path    …\h2o_tddft
file    input.yaml           ok
file    result.json          ok
file    log.txt              ok
file    data_snapshot.json   ok
file    env.json             ok
env     rustc rustc 1.97.1 (8bab26f4f 2026-07-14), threads 1, os windows x86_64
env     data snapshot: 33 file(s)
status  ok
```

### 4.5 `qwb replay` —— 复现

```powershell
qwb replay results\h2_sto3g_svwn5 --dry-run   # 只查能不能复现（数据快照是否还一致）
qwb replay results\h2_sto3g_svwn5             # 真重算并比对能量
```

判据是**重算能量与记录值差 < 1e-10 Ha**（实测 `0.000e0`）。数据文件被改过（sha256 不符）时
`--dry-run` 就会报错，这是它的主要用途。

### 4.6 `qwb data` —— 数据仓操作

```powershell
qwb data list                     # 列出注册表（basis / ecp / functional / grid / solvent / dispersion / d4 / smd-atomic / numerics）
qwb data show pbe                 # 打印一个条目的完整 YAML
qwb data verify                   # 校验整个注册表（G2：sha256 + schema + 语义）
qwb data verify path\file.yaml    # 校验单个文件
qwb data sign path\file.yaml      # 按规范化摘要重新签名（改了数据之后）
qwb data install                  # 取回缺的第三方数据（§8.4）
qwb data install --check          # 只体检：缺/坏 → 退出码 3（K5 的数据失败码）
qwb data install --need basis\lanl2dz.yaml ecp\lanl2dz-cu.yaml
qwb data install --for-input qwb-core\examples\cuh_lanl2dz_pbe.yaml
qwb data install --all --force    # 全部重下（摘要已正确也重下）
```

`qwb data verify` 实测结尾：`34 file(s) checked, 34 passed, 0 failed`。
没取回第三方数据时，那 10 个条目会被**单独**记为失败（不是"整个注册表坏了"），
消息里直接给取数命令。

**数据根**怎么找：`QWB_DATA_ROOT` 环境变量 > 从当前目录逐级向上找 `qwb-data/`。
CLI 没有 `--data-root` 参数（§14 #17 的决定）；Python 侧的数据入口有 `--data-root`。

### 4.7 `qwb props` —— 激发态

```powershell
qwb props examples\h2o_tddft.yaml --out results
qwb props examples\h2o_tddft.yaml --states 3      # 覆盖 options.excited_states.n_states
```

输出 5 行 `state N …`（能量/Ha、eV、振子强度）+ 一行 `spectrum …`（曲线路径），
同时写完整的结果目录。边界（都有单测）：

* 输入里**没有** `options.excited_states` → 退出码 1 并提示抄 `examples/h2o_tddft.yaml`
  （不会替你猜一个态数）；
* 输入文件不存在 → 退出码 1；`--help` 永远可用。

### 4.8 `qwb compare` —— 多任务对比表

```powershell
qwb compare results\h2o_tddft results\h2o_tddft_tdhf --csv cmp.csv
```

```text
task                     method      basis          energy/Ha dE/(kcal/mol)   conv  peaks(eV)
h2o_tddft                    hf     sto-3g     -74.9631464092       0.0000   true  13.20, 16.80
h2o_tddft_tdhf               hf     sto-3g     -74.9631464092       0.0000   true  13.16, 16.71, 19.11, 21.96
csv     …\cmp.csv
```

* 至少两个目录；顺序就是 CSV 的行序；`dE` 相对**第一行**。
* `peaks(eV)` 取的是 `result.json` 里 `uv_vis_spectrum` 的局部极大值（没有谱就留空）。
* CSV 的列是冻结的（`COMPARISON_COLUMNS`），Python 侧 `qwb_studio.comparison` 用同一套。

### 4.9 `qwb ts` / `irc` / `scan` —— 只落签名

```text
$ qwb ts examples/h2_sto3g_svwn5.yaml
qwb: `qwb ts` is declared in K5 (phase-2 KV-05) but not implemented yet: transition-state search lands with M6 (RP-01)
$ echo $LASTEXITCODE
4
```

三个都是退出码 4 + 指向负责的 WP；`--help` 可用、输入文件不存在时报退出码 1（输入错误），
不会被"功能没实现"的 4 盖掉。

---

## 5. Python API（pyo3）

`import qwb_engine as qe`。接口与冻结桩 `qwb-contracts/studio/python/qwb_engine.pyi` 一致
（`mypy --strict` 在门禁里检查）。最小例子：

```python
import qwb_engine as qe

molecule = qe.Molecule(
    [qe.Atom("O", (0.0, 0.0, 0.0)),
     qe.Atom("H", (0.0, 0.757, 0.586)),
     qe.Atom("H", (0.0, -0.757, 0.586))],
    charge=0, multiplicity=1,
)
job = qe.JobInput(molecule, "def2-svp", "pbe", grid_level="normal", jobtype="sp")

result = qe.run_job(job, "<仓库根>/qwb-data")     # 一次性
print(result.energy_hartree, result.scf_converged)

engine = qe.Engine("<仓库根>/qwb-data")           # 复用注册表
result = engine.run(job)                          # 可反复调用
print(result.mo_energies_hartree[:5], result.dipole_debye)
```

要点：

* `qe.JobInput(molecule, basis, method, grid_level=…, solvent=…, jobtype=…, options=…)`
  的字段可以之后赋值（`job.jobtype = "opt"`），**未知字段没有**（桩里没有就真的没有）。
  阶段二的选项挂在 `qe.TaskOptions` 上（`scf_type` / `mp2` / `excited_states` /
  `solvent_model` / `numerics` / `dispersion`），`TaskOptions()` 无参构造、字段赋值。
* 结果里阶段二的块是**普通 dict**（`spin` / `correlation` / `excited_states` /
  `uv_vis_spectrum` / `solvation` / `multipoles` / `epr`），缺块就是 `None`，
  键名与 `result.json` 逐字一致。
* 异常层次：`EngineError` 基类，`InvalidInputError` / `ScfNotConvergedError` /
  `DataError` / `UnsupportedError` 四个子类，每个都带 `exit_code`（与 K5 一致）。
* **GIL 真的释放**：计算期间别的 Python 线程照跑（门禁里的 `test_engine_binding.py` 钉这一点）。
* 没有撤销/中断 API；要取消请用子进程 + kill（GUI 的 `engine` runner 就是这么做的），
  或者用 `Engine.request_cancel()` 的协作取消标志。
* **ghost 原子还没有 Python 入口**（K2 的 `Atom::new_ghost` 只在 Rust 侧）：要算 counterpoise，
  用 `input.yaml` 走 CLI。

---

## 6. 图形界面 qwb-studio

```powershell
$env:PYTHONPATH = '<仓库根>\qwb-studio'
$env:QWB_ENGINE_BIN = '<仓库根>\qwb-core\target\release\qwb.exe'   # 可选，见下
.venv\Scripts\python.exe -m qwb_studio.gui --out studio-out --workers 2
```

窗口结构（图标栏 + 左侧项目树 + 右侧七个标签页）：

| 标签 | 用途 |
|---|---|
| **Build** | 建分子（元素 + x/y/z + 电荷/多重度）→ 选基组/方法/网格/作业类型 → 生成 `input.yaml` → 提交真作业 |
| **Task** | 任务 id、mock 时长、作业输入路径、提交/运行队列/取消按钮 |
| **Job Monitor** | 表格：task / 状态 / 原因（终态带颜色） |
| **Result** | 能量、偶极、SCF 状态、`env.json` 摘要 |
| **Log** | 流式日志（块数上限 5000，100 MB 日志也不会把内存吃满） |
| **3D** | VTK 离屏渲染：球棍分子、轨道等值面、ESP 电荷着色、振动帧 ± |
| **Data** | 第三方基组/ECP 的状态表 + 「刷新状态 / 下载缺失 / 全部重下」（菜单「数据」同款） |

要点：

* **Build 页的下拉框来自 `qwb-data/registry.yaml`**：界面里能选的就是数据里有的（元素按所选
  基组过滤）。`b3lyp` 不在注册表里，所以不出现。
* **GUI 目前只覆盖阶段一的产品面**：作业类型只给 `sp` / `opt`，没有 `scf_type` / 色散 /
  激发态 / 溶剂模型的控件（那些走 `input.yaml` 或 Python API）。
* **Data 页**是手动入口（下载在工作线程里跑，界面不卡）：「缺失 / 摘要不符」会标黄。
  提交作业时不用先来这里 —— worker 是 `qwb run`，它会自动取回缺的文件。
* **取消**：`queued` 立即落 `cancelled`；`running` 先写协作取消标志，超过宽限期再 kill。
* **崩溃恢复**：重启后从 `<out>/queue.json` 重建 Task Panel；停在 `running` 的任务落 `failed`
  （reason 带 `interrupted`），`queued` 的保留可以直接接着跑。
* **`QWB_ENGINE_BIN` 一定要指向 release 二进制**：默认会优先用
  `qwb-core/target/debug/qwb.exe`，debug 构建慢 10–50 倍。
* 提交真作业时 GUI 会把 `input.yaml` 暂存到 `<out>/.staging/<task_id>.yaml`，**不污染**
  K3 结果目录；`qwb run` 用文件 stem 当任务 id，所以文件名必须等于 task id。

无显示器环境（服务器/CI）加 `$env:QT_QPA_PLATFORM='offscreen'`，门禁里的 GUI 测试就是这么跑的。

---

## 7. 结果目录怎么看

### 7.1 文件清单（K3 冻结）

| 文件 | 内容 | 缺了会怎样 |
|---|---|---|
| `input.yaml` | 这次作业的输入（原样回写） | `qwb inspect` 报缺失 |
| `result.json` | K2 `JobResult`（见 §7.2） | 同上 |
| `env.json` | 复现环境（G5：`engine_version`/`rustc_version`/`cargo_lock_sha256`/`data_snapshot`/`threads`/`os`/`created_at_utc`） | **退出码 3**，结果视为不完整 |
| `data_snapshot.json` | 注册表里每个数据文件的 sha256（实测 33 个） | `qwb inspect` 报缺失 |
| `log.txt` | `Reporter` 输出 + 引擎日志（SCF 每步、优化每步、ECP 行…） | `qwb inspect` 报缺失 |
| `artifacts/` | 自由形态产物（§7.3） | 允许为空 |

### 7.2 `result.json` 的块：什么时候出现

| 块 | 什么时候出现 | 关键字段 |
|---|---|---|
| `energy_hartree` / `scf_converged` / `mo_energies_hartree` / `mo_coeffs` / `dipole_debye` / `log` / `env` / `data_snapshot` | 每次 | `mo_coeffs` 是**系数矩阵**（列 = MO） |
| `spin` | `scf_type: uhf \| uks \| rohf` | `n_alpha` / `n_beta` / `multiplicity` / `s_squared` / `spin_contamination` / `broken_symmetry` |
| `correlation` | `options.mp2` | `method` / `e2_hartree` / `e_total_hartree` / `aux_basis_name` / `ri_error_hartree` |
| `excited_states` | `options.excited_states` | `method` / `n_states` / `transitions` / `sum_rules` |
| `uv_vis_spectrum` | 同上（且 `numerics.yaml` 有 `spectrum_defaults`） | `gaussian_hwhm_ev` / `grid_step_ev` / `points` |
| `vibrational_frequencies` | `jobtype: freq` | `method` / `modes[]`（`wavenumber_cm`、`ir_intensity_km_per_mol`、`imaginary`）/ `zero_point_cm` / `vibrational_modes` |
| `thermochemistry` | `jobtype: freq` | `temperature_k` / `pressure_pa` / `zero_point_hartree` / `internal_energy_hartree` / `enthalpy_hartree` / `entropy_j_per_mol_k` / `gibbs_hartree` / `rotational_constants_cm` / `symmetry_number` |
| `solvation` | `solvent: <非 vacuum>` | `model` / `solvent_name` / `electrostatic_hartree` / `cds_hartree`（SMD 才有，eq. 4.77）/ `cds_atomic_hartree` + `cds_molecular_hartree`（它的两半）/ 空腔几何 |
| `dispersion_energy_hartree` | `options.dispersion` | 可加项；`energy_hartree = SCF + 它 |
| `epr` | `options.epr: true`（开壳层 sp） | `method` / `nuclei[]`（`spin_density_au`、`a_iso_mhz`、`a_iso_gauss`） |
| `multipoles` | 每次闭壳层 `sp` | `quadrupole_traceless_debye_angstrom` / `origin_angstrom` |
| `gradient_hartree_per_bohr` / `hessian…` | `opt` 的梯度 / 未接线的 freq | 形状 `3N×1`（按原子再按 x/y/z 摊平） |
| `nmr_shieldings` / `polarizability` / `numerics_report` | **还不出现** | RS-04 / K10 未交付（别指望） |

### 7.3 `artifacts/`

| 文件 | 什么时候 | 内容 |
|---|---|---|
| `populations.json` | 闭壳层 `run` | Mulliken / Löwdin 布居与电荷（3D 的 ESP 着色用它） |
| `homo.cube` / `lumo.cube` | 闭壳层 `run` | Gaussian cube 格式的轨道等值面输入（0.35 Bohr 间距、4 Bohr 外扩、2e6 点上限） |
| `optimized.xyz` | `jobtype: opt` | 优化后的几何（标准 XYZ，Å）；3D 面板优先读它 |
| `optimization.json` | `jobtype: opt` | 轨迹 `[step, E, max|grad|]`、步数、是否解析梯度、退化原因 |
| `spectrum.csv` | `options.excited_states` | 两列（eV、强度）的展宽曲线 |

开壳层作业**不写**前两项（原因见 §3.4 第 4 点）。

### 7.4 读 `result.json` 的两个坑

1. `mo_coeffs` 是**系数矩阵**（列 = MO，满足 `CᵀSC = 1`），不是密度矩阵。
2. 矩阵按 K2 的约定是 **list-of-list**（行 = 第一个索引），`gradient_hartree_per_bohr` 的形状是
   `3N×1`（按原子再按 x/y/z 摊平）。

### 7.5 Python 读结果

```python
from qwb_studio import open_result_dir
directory = open_result_dir("qwb-core/results/h2o_sto3g_svwn5")
print(directory.result.energy_hartree)
print(directory.env.rustc_version, directory.env.threads)
```

`open_result_dir` 会把 K3 必需文件与 `env.json` 完整性都查一遍，缺东西抛 `ResultDirError`。

---

## 8. 数据仓 `qwb-data` 与校验

### 8.1 清单（实测 `qwb data list`）

```text
basis       sto-3g / def2-svp / def2-tzvp / 6-311g(d,p) / lanl2dz
            def2-svp-ri / def2-tzvp-ri / def2-universal-jkfit      （三个 RI 辅助基）
ecp         lanl2dz-cu（Cu 10 芯 / z_eff 19）、lanl2dz-zn（Zn 18 芯 / z_eff 12）
functional  pbe / svwn5 / pbe0 / b88 / m06-l / m06-2x
grid        grid-coarse / grid-normal / grid-fine / grid-veryfine
solvent     water / methanol / ethanol / acetonitrile / dmso / chloroform / toluene / thf
dispersion  d3bj / d3zero
d4dispersion d4
smdatomic   smd-atomic（SMD 的元素级 vdW 半径与逐元素 σ 表，SV-03）
numerics    numerics（K10 精度等级 1..5 + 等价容差）
```

共 **34 个生产数据文件**（含 `registry.yaml` 本身算 1 个），全部登记在
`DATA_LICENSES.md`，`qwb data verify` 实测 **34/34 通过**。

四个网格等级的实际参数（`radial_points` × `lebedev_order`）：coarse 50×194、normal 75×302、
fine 99×590、veryfine 140×（见 `qwb-data/grid/*.yaml`）。`numerics.yaml` 的 5 个等级**引用**这些
网格名，不复制参数。

### 8.1.1 `numerics` 域（阶段二 K10）

`qwb-data/numerics/numerics.yaml` 是**精度等级与等价容差**的数据（`project-phase2.md` §8.5）：

* `levels[].grid_level` 只是**引用** `grid/*.yaml` 的 `name` —— 网格参数仍然只有一份；
* `levels[].ps` 是 pseudospectral 路径的配点配置；等级越高配点必须越多，去别名一旦打开不许关回去；
* `equivalence_tolerance` 是**验收判据**（`ri: 1e-6`、`pseudospectral: 5e-5`），不是文献值；
* `spectrum_defaults` 是 UV/Vis 展宽参数（HWHM 0.4 eV、步长 0.05 eV、padding 6 HWHM）。

**一处已知的不一致（VF-06，in_progress）**：`ri` 的 1e-6 想约束的是"两种算法实现同一数学对象"，
而 RI-MP2 与 canonical MP2 的实测差是 **8.526e-5 Ha** —— 差的是**辅助基不完备**，不是实现
（同一个差值就是 SC-05 记的 `ri_error_hartree`，逐位相同）。所以等价性套件目前是**报告模式**
（`scripts/check_numerics_equivalence.py --report`），要把它改成判红得先把容差语义重新界定。
现在写 `options.numerics` 会得到退出码 4（§3.10）。

### 8.2 一份数据文件长什么样

```yaml
name: pbe
version: 1.0.0
citation:                       # 每项要有 title + (doi 或 url)，缺了 verify 不过
  - title: "Generalized Gradient Approximation Made Simple"
    doi: "10.1103/PhysRevLett.77.3865"
license:                        # spdx 非空；两个布尔必须显式写
  spdx: Apache-2.0
  commercial_ok: true
  redistribute_ok: true
created: 2026-10-22T23:00:00Z
components:                     # 泛函 = 组件 + 权重 + 参数，数学形式在代码里
  - id: GgaPbeX
    weight: 1.0
parameters:
  pbe_kappa: 0.804
sha256: <64 位小写十六进制，由工具戳>
```

### 8.3 改数据的纪律

1. 改完 → `qwb data sign <file>`（规范化摘要，"文件内 sha256 + registry 条目 + 结果目录快照"三处一致）；
2. → `qwb data verify`（schema + 语义 + sha256 交叉校验）；
3. → `python scripts/gen_data_licenses.py`（重生成 `DATA_LICENSES.md`，门禁 D-15 会比对）；
4. → 重跑相关测试（基组换了跑 D-03/D-04/D-07，泛函换了跑 `t06_reference` 与对应的 `fx0x_*`）；
5. 数据版本要 bump（`registry.yaml` 的 `version`），并在 `qwb-data/CHANGELOG.md` 记一行。

**外部数据**（BSE 导出、LANL2DZ ECP）的许可逐条确认后才允许写 `redistribute_ok: true`；
`lanl2dz` 的两个 ECP 文件目前是 `noassertion` + 两个 `false`。

### 8.4 第三方数据不在仓库里（按需取回）

`qwb-data/` 下有 **10 个文件不进仓库**：

| 文件 | 来源 |
|---|---|
| `basis/sto-3g.yaml`、`def2-svp.yaml`、`def2-tzvp.yaml`、`6-311g-dp.yaml`、`lanl2dz.yaml` | Basis Set Exchange 导出 |
| `basis/def2-svp-ri.yaml`、`def2-tzvp-ri.yaml`、`def2-universal-jkfit.yaml` | BSE 导出的 RI 辅助基 |
| `ecp/lanl2dz-cu.yaml`、`ecp/lanl2dz-zn.yaml` | BSE 导出的 LANL2DZ ECP |

原因是**再分发**：LANL2DZ 的 ECP 原始论文（Hay–Wadt）没有给出明确的再分发条款，
把文件放进公开仓库就等于再分发（project.md §14 #20）。仓库提交的是**配方**
`qwb-data/sources.json`：

| 字段 | 含义 |
|---|---|
| `path` | 相对 `qwb-data/` 的路径（与 `registry.yaml` 条目一致） |
| `kind` | `basis` / `ecp` |
| `fetch` | BSE 的 `basis` 名 + 元素（原子序数）；下载走 `api/basis/<basis>/format/json/` |
| `transcribe` | 转录元数据：`name`/`version`/`notes`/`citations`/`license`/`created`/`angular`/`normalization`/`role`（ECP 用 `element`） |
| `sha256` | 内容摘要钉 —— 与 `registry.yaml` 里同一条**必须相等** |

三条取回入口（**同一个实现** `qwb-studio/qwb_studio/datafetch.py`）：

```powershell
qwb data install                                     # Rust CLI（推荐）
python scripts\fetch_data.py --all                   # 脚本（不需要 PYTHONPATH）
python -m qwb_studio.datafetch --all                 # 模块入口（需要 PYTHONPATH）
```

GUI：**Data 页 → 「下载缺失」**（或菜单「数据 → 下载缺失」）。常用开关：

```powershell
qwb data install --check                               # 只体检：10/10 就绪 / 缺哪个
python scripts\fetch_data.py --list                     # 表格：ok / MISSING / STALE
python scripts\fetch_data.py --for-input <INPUT_YAML>   # 只取这份输入要用的
python scripts\fetch_data.py --need basis\lanl2dz.yaml  # 只取一条
python scripts\fetch_data.py --all --force              # 全部重下
```

**流程与保证**：下载 → 转录 → 用 `qwb data sign` 打 K1 摘要 → 核对清单里的钉 →
（一致才）原子替换目标文件。所以：

* 摘要对不上就**报错且不落盘**（上游修订号变了、或转录规则漂移了，都得人看一眼）；
* 取回的文件与"当初提交的文件"内容等价，`registry.yaml` 的钉、结果目录的
  `data_snapshot.json`、历史验收数字全都仍然有效；
* 中断不会留下半个文件（先写临时文件，最后一步才是 `move`）。

**自动取回**：`qwb run` / `qwb batch` 开始前会检查一遍，缺就按这份输入需要的取回，只下缺的那几个；
取数失败只 warning，不阻断不需要那份数据的作业。关闭自动取回：`$env:QWB_AUTO_FETCH = '0'`
（此时缺数据的作业以退出码 3 报错，消息里给出取数命令）。

**完全离线**：在有网机器上跑一次 `qwb data install`，把整个 `qwb-data/` 目录拷过去即可
（`sources.json` 的摘要钉保证两边一致）。

**自研数据不吃这一套**：`functional/`、`grid/`、`solvent/`、`dispersion/`、`numerics/` 是随仓库
提交的。它们缺失是**硬错误**（数据根被裁剪过），`Registry::open` 不会把它当成"还没下载"而静默跳过。

**原始快照不进仓库**：`mnsol.F`（SMD 参数）、`radii.py`（SMD 半径）、`s22.py`（S22 几何）放在
gitignored 的 `qwb-data/reference/_*_staging/raw/` 里，派生物（CSV/JSON）随仓库提交，
每个家族都有带 `--check` 的生成脚本，`scripts/check_reference_sources.py` 在门禁里核
"逐行带 DOI/URL"这件事（当前 21 个家族 / 1917 行）。

---

## 9. 能算什么、不能算什么（诚实清单）

### 9.1 方法

| `method_name` | 类型 | 状态 |
|---|---|---|
| `hf` | 闭壳层 RHF | 能算 |
| `hf` + `scf_type: uhf` | 非限制性 UHF | 能算（与 PySCF 差 3e-8，`spin` 块落盘） |
| `hf` + `scf_type: rohf` | ROHF（纯自旋） | 能算（5 体系最差 3.6e-8，`⟨S²⟩` 精确 `S(S+1)`） |
| `svwn5` | LDA（参数集是 VWN **RPA**，libxc 的 `LDA_C_VWN_RPA`） | 能算（闭壳层与 `uks`） |
| `pbe` | GGA | 能算（闭壳层与 `uks`） |
| `b88` | GGA 交换（单独使用） | 能算，一般作为组件 |
| `pbe0` | hybrid（α_hf = 0.25） | 能算（闭壳层与 `uks`） |
| `m06-l` | meta-GGA（τ 依赖） | 能算（FX-01 的 τ 管线 + FX-02；逐点对 libxc < 1e-9） |
| `m06-2x` | hybrid meta-GGA（54% 精确交换） | 能算，**格点成本高**（§3.11） |
| `b3lyp` | hybrid | **数据层不注册**（K1 拒绝 `requires.laplacian`，冻结契约）：写 `method_name: b3lyp` 得到退出码 3；只能在测试里程序化构造 |
| `svwn5`/`pbe` + `scf_type: uks` | 自旋极化 KS | 能算，**sp only**；`b88`/`blyp` 的自旋路径没实现（被显式拒绝） |

参考态之上还能叠相关能（`options.mp2`，§3.5）与激发态（`options.excited_states`，§3.6）。

### 9.2 作业类型 × `scf_type` 矩阵

| | `jobtype: sp` | `jobtype: opt` | `jobtype: freq` |
|---|---|---|---|
| 闭壳层（`rhf`/`rks`/缺省） | ✓ | ✓（解析梯度） | ✗ 退出码 4 |
| `uhf` | ✓ | ✓（SC-02 解析梯度） | ✗ |
| `rohf` | ✓ | ✓（**数值梯度**：解析梯度只接到 UHF，`rohf` 每一步用位移重跑 SCF；`optimization.json` 的 `used_analytic_gradient` 会是 false） | ✗ |
| `uks` | ✓ | ✗ 退出码 4 | ✗ |
| 带 ECP | ✓ | ✗ 退出码 4 | ✗ |
| 带溶剂（COSMO） | ✓ | ✗ 退出码 4 | ✗ |
| 带色散 | ✓ | ✗ 退出码 4 | ✗ |
| 带 ghost 原子 | ✓ | ✗ 退出码 4 | ✗ |
| 带 `mp2` | ✓ | ✗ 退出码 4 | ✗ |
| 带 `excited_states` | ✓ | ✗ 退出码 4 | ✗ |

### 9.3 精度现状（都有实测记录）

* **能量**：STO-3G 上 7 个体系（H2/H2O/CO/CH4/NH3/N2/苯）× 5 个模型化学与 CCCBDB 差
  ≤ 4e-6 Ha；苯（36 个基函数）B3LYP grid-normal 差 −1.16 mHa、PBE 差 −1.33 mHa（判据 2 mHa）。
* **几何**：PBE0/def2-SVP 的 H2O O–H = 0.9623 Å（实验 0.9572，差 0.0051）；PBE 一族系统性
  偏长 ~0.011 Å（泛函偏差，不是实现问题）。
* **频率**：PBE/6-311G(d,p) × 0.991 标度后最差 21.5 cm⁻¹（判据 30）；PBE0 一族 60.5 cm⁻¹
  （不通过，如实记作对照组）。注意频率**没接进引擎**（§9.4）。
* **ECP**：矩阵元与解析解差 3e-14；CuH 与 ZnO 单点跑通。
* **ZnO/LANL2DZ 的 SCF 需要退路**：单初猜 DIIS 不收敛，引擎会自动退到"多初猜 → 费米展宽退火"
  （23 步收敛）。这是正常路径，不是错误。
* **网格敏感度**：B3LYP/PBE 在苯上 grid-coarse 反而比 grid-normal 更接近基准（−0.01 vs −1.16 mHa），
  量级 ~1 mHa —— 需要 1 mHa 级能量时要注意。
* **开壳层**：UHF/UKS 的自旋污染如实报出（O2 的 `⟨S²⟩`：UHF 2.0034、UKS 2.00039、ROHF 2.0 精确）；
  UKS/PBE 的 O2 与 PySCF 差 5.8e-7、OH 差 1.1e-6（容差 5e-6，网格）。
* **meta-GGA**：M06-2X/M06-L 的逐点导数对 libxc < 1e-9；H2O/def2-SVP 端到端（coarse）
  差 2.3e-5 / 2.8e-5 Ha。
* **COSMO**：H2O −7.274 kcal/mol（实验 −6.32，判据 1.5 内）；缺非极性项是已记账的偏差。
* **色散**：D3(BJ)/D3(0)/D4 与官方实现逐项对上（门禁里三条数据检查）。

### 9.4 明确没做

* **`jobtype: freq` 已接**（第 388 轮）：数值 Hessian（`numerical-hessian`）→ 简正模 →
  谐振频率 + IR 强度 + 理想气体热化学（298.15 K / 1 atm），结果在 `result.json` 的
  `vibrational_frequencies` 与 `thermochemistry` 两块。**只做闭壳层极小点**：开壳层、ECP、
  溶剂、色散、带 ghost 的组合一律显式拒绝；几何若有**真虚频**（< −100 cm⁻¹，零模噪声实测
  到 −48.7）也拒绝 —— 3N−6 的振动集合是按波数选的，鞍点要先做子空间投影（RP-01）。
  **转动对称数 σ** 要显式给 `options.symmetry_number`（水 2、氨 3、甲烷 12）；缺省 1 会让
  S 偏大 R ln σ（水 +5.76 J/(mol·K)）。
* **ECP 的导数**：带 ECP 的 `opt`/`freq` 不支持。
* **开壳层的 opt 只到 HF**（`uhf`/`rohf`）；`uks` 只能 `sp`（开壳层 UKS 的 SCF 收敛是 SC-02 的剩余一半）。
* **开壳层的布居/cube**：只有 α 套轨道，宁可不写（§3.4）。
* **溶剂**只到单点，且静电都是 C-PCM 一族：`opt`/`freq` 退出码 4（SV-02），PBF 退出码 4
  （SV-04）。**SMD 的 `sp` 已接**（静电 + `G_CDS`），但 SV-03 的判据（8 溶剂 × 3 分子的
  ΔG_solv 对实验 MAE < 1.0 kcal/mol）还没走完：**缺实验参考集与参数化层级**（分子内的热化学项
  在 1 M ↔ 1 M 的转移能里互相抵消，上一版这条写错了），而且
  SMD 原文的静电是 IEF-PCM(SWIG) 而不是我们这套 C-PCM 表面 —— 两处都记在
  `docs/error-budget.md`。**旧账本的"三项非极性项在付费墙后"是错的**：SMD 的非极性项
  只有一项 `G_CDS`，参数来自公开的 MNSOL 描述符表，已随数据钉住（见 choices 一百八十二）。
* **近似路径**：`options.numerics` 一律退出码 4（IN-03 pseudospectral、IN-04 RI-J 都没过 K10）。
* **解析 Hessian**（RS-05，blocked：20× 的两条前提都不成立）、**GIAO NMR**（RS-04，in_progress）、
  **Raman**（RS-06，blocked：要简正模式 → 要 RS-05）、**VCD**（RS-07 pending）。
* **TS / IRC / 扫描**（RP-01..04 pending）：CLI 只落签名。
* **GPU**：只有 `gemm` 原语、显存预算 + 自动回退、AO→MO 变换（实测 60.8×）；格点 XC 的 GPU 核
  需要 NVRTC/cubin（另一类 unsafe 面），EX-03 仍 blocked。
* **并行**：4 指标积分张量单线程；Fock 构建按行并行（确定性归约，I-05）；`--threads` 与
  `options.threads` 只写进 `env.json`（记 1），不驱动并行度。
* **`.qwb` 容器**：K4 读写与双向互操作有测试，但 CLI 没有"打包"子命令（Python 侧 `QwbContainer` 可以）。

### 9.5 逐项对应的 WP（机器核对）

§9.1–§9.4 是给人读的；这张表是**同一批事实的机读版本**，由门禁
`scripts/check_usage_honest_list.py` 对着 `docs/status-v2.yaml` 的 `state` 核对：

* `✅` 要求列出的 WP **全部** `done`（说"能算"必须有已验收的工作包撑着）；
* `⚠` 不许出现 `pending`（`pending` 意味着代码还没落地，那就不是"能算但有缺口"）；
* `❌` 不许出现 `done`（也不能拿已完成的工作包当"不能算"的挡箭牌）；
* 文档里任何位置出现 `✅/⚠/❌` 却没进这张表 → 门禁红（"顺手在正文里夸一句能算"写不进去）。

`⚠` 的意思是「能算，但有记账的缺口」，缺口写在第 4 列。阶段一就有的能力（闭壳层单点、
ECP、频率库函数）在 `docs/status.yaml`；这里只列阶段二台账里能对上号的条目。

<!-- honest-list:begin -->
| 条目 | 声称 | WP | 缺口 / 依据 |
|---|---|---|---|
| 闭壳层单点：`hf`/`svwn5`/`pbe`/`b88`/`pbe0`/`m06-l` | ✅ | VF-01, FX-01, FX-02 | CCCBDB 对账 + τ 管线 |
| 开壳层 UHF 单点（`scf_type: uhf`） | ✅ | SC-01 | 与 PySCF 差 3e-8 |
| 开壳层 UHF 解析梯度 + `opt` | ✅ | SC-02 | 解析 vs 数值 7.3e-7 Ha/Bohr |
| 开壳层 UKS 单点（`svwn5`/`pbe`） | ✅ | SC-01b, M1b | O2/OH 与 PySCF 5.8e-7…1.1e-6 |
| 开壳层 UKS × meta-GGA（τ 通道） | ✅ | SC-06 | M06-L/M06-2X 端到端 3.3e-5…4.5e-5 Ha |
| ROHF（纯自旋）单点与数值梯度 `opt` | ✅ | SC-01c, M1c | 5 体系最差 3.6e-8，⟨S²⟩ 精确 |
| canonical MP2 | ✅ | SC-04 | 与参考差 < 0.1 mHa |
| RI-MP2 | ⚠ | SC-05 | 正确性达标，但苯/def2-SVP 墙钟 48.3 s > canonical 7.1 s（记账，不改判据） |
| 激发态 CIS/TDA/TDDFT（**HF 参考**） | ✅ | RS-02, RS-03, ST-01 | 与 PySCF 5e-7 eV |
| DFT 参考的 TDDFT（要 `f_xc` 进响应矩阵） | ❌ | FX-03 | 非 `hf` 参考显式拒绝并点名 FX-03 |
| 动态极化率 α(ω) 与一阶超极化率 β | ⚠ | ST-02 | 库函数 + 测试有，尚未进 `result.json` |
| 多极展开（traceless 四极 / 八极） | ✅ | RS-08 | — |
| 色散 D3(BJ)/D3(0)（含开壳层与线程确定性） | ✅ | FX-04, DA-06 | 对官方 s-dftd3 逐项 |
| 幽灵原子 / counterpoise（BSSE）单点 | ⚠ | M2 | 只支持 `sp`；色散/ECP/溶剂未审，显式拒绝 |
| COSMO / C-PCM 单点 | ⚠ | SV-01 | 面积判据仍 blocked（解析球帽已部分落地）；缺非极性项 |
| SMD 单点（静电 + `G_CDS`） | ⚠ | SV-03 | 判据（8 溶剂 × 3 分子对实验 MAE < 1.0）缺**实验参考集**与参数化层级；与 PySCF 的 SMD 比：`G_CDS` 差 3.9e-4、**静电差 1.2961** kcal/mol（表面表示不同）；标准态项 `RT ln(RT/p°) = +1.8943` 按对方约定加 |
| 溶剂化解析梯度 / 优化 | ❌ | SV-02 | 依赖 SV-01 |
| 溶液相频率 / 热化学 | ❌ | SV-05 | 依赖 SV-01/SV-02 |
| `jobtype: freq`（**数值** Hessian：频率/IR/热化学） | ⚠ | RS-05 | 已接线（第 388 轮）：频率对 PySCF 解析 Hessian 差 ≤ 0.19 cm⁻¹；**解析** Hessian 仍 blocked（20× 的前提不成立），freq 因此是 3N+1 次 SCF 级的成本 |
| GIAO NMR 屏蔽 | ❌ | RS-04 | 缺 London 复基组层（common-origin 下 O 差 1.50 ppm） |
| Raman 活性 | ❌ | RS-06 | 要简正模式 → 要 RS-05 |
| VCD | ❌ | RS-07 | pending |
| TS / IRC / 扫描（PES） | ❌ | RP-01, RP-02, RP-03 | CLI 只落签名，退出码 4 |
| 近似路径：pseudospectral / RI-J 的 Fock 构建 | ❌ | IN-03, IN-04 | 都受 K10 等价性门禁约束，`options.numerics` 退出码 4 |
| ωB97X-D / CAM-B3LYP / B3LYP | ❌ | DA-03 | LYP 要 `∇²ρ`，K1 冻结契约拒绝 `requires.laplacian` |
| EPR 超精细耦合 | ❌ | ST-05 | 未过 K10（anchor 定标仍差 0.169% 量级之外的判据） |
| LMP2 | ❌ | ST-03 | pending |
| pKa 工作流 | ❌ | ST-04 | pending |
| 溶剂化自由能 / 势垒 / NMR 基准集 | ❌ | VF-02, VF-03, VF-04 | 都没有参考集（DA-08 卡在 SV-01/SV-03） |
| GPU：`gemm`、显存预算 + 自动回退、AO→MO（60.8×） | ⚠ | EX-01, EX-02, EX-03 | 格点 XC 的 GPU 核要 NVRTC/cubin（另一类 unsafe 面），EX-03 blocked |
| `--threads` / `options.threads` 驱动并行度 | ⚠ | IN-05 | Fock 构建按行并行（确定性归约），但线程数只写进 `env.json`，4 指标张量仍单线程 |
| 开壳层的布居 / cube | ⚠ | SC-01b | 只有 α 套轨道，宁可不写（§3.4） |
| `.qwb` 容器的 CLI 打包子命令 | ⚠ | PR-03 | 容器读写与双向互操作有测试，CLI 没有"打包"子命令 |
| Python 绑定（pyo3）与 `.pyi` 类型面 | ✅ | PR-02, KV-05 | 与 CLI 能量逐位相同；桩里声明就得实现 |
| `qwb compare` 对比表 | ✅ | PR-07 | 只读，与各任务 `result.json` 差 < 1e-10 |
| UV-Vis 展宽谱块（`uv_vis_spectrum`） | ✅ | PR-04 | 展宽参数版本化 + sha256 钉住 |
<!-- honest-list:end -->

---

## 10. 验收与门禁

### 10.1 一条命令

```powershell
# Windows（三个 bash 门禁会通过 WSL 跑）
pwsh -NoProfile -File scripts\verify.ps1

# Linux / CI / WSL
bash scripts/verify.sh
```

**35 个步骤**：

1. 3 个 bash 策略门禁：依赖集中（`check_deps_centralized.sh`）、unsafe 白名单
   （`check_unsafe.sh`）、rayon 确定性（`check_rayon_determinism.sh`）；
2. 第三方数据就位（`fetch_data.py --check`，缺则 `--all`）；
3. K2 契约 crate 的 fmt / clippy `-D warnings` / `cargo doc`（`RUSTDOCFLAGS=-D warnings`）/ test；
4. `qwb-core` 的 fmt / clippy；**编译与计时分离**的 test（`cargo-nextest` 两条 `--partition`
   泳道各 4 线程 + 单独的 `cargo test --workspace --doc`，因为 nextest 不跑 doctest）；
   I-06 积分缓存（feature on）、I-00 faer 后端（另一套产物）、`cargo deny`、K1 schema 不漂移、
   G2 数据校验（`qwb data verify`）；
5. 数据与参考：D-15 数据许可一致、DA-08 参考集逐行带来源、VF-05 误差预算逐行可核、
   PR-08 诚实清单与账本一致（+ 检查器自测）、DA-06 三条（色散数据对上游、对 s-dftd3 的能量与
   梯度、零阻尼单独一条）、VF-06 等价性差异表（**报告模式**）；
6. 账本合法性：`docs/status.yaml` 与 `docs/status-v2.yaml`（两份都要能解析）；
7. Python 侧：K6 `mypy --strict` ×2、pyo3 crate 的 fmt/clippy、maturin 构建并安装 wheel、
   `qwb-studio` pytest（**两条泳道**：一个 42–45 s 的 GUI 压力测试独占一条，其余并发；
   `--deselect` 必须恰好命中 1 个、A 泳道必须恰好 1 passed，否则红）、`.qwb` 夹具新鲜度。

实测墙钟（8 物理核 / 16 逻辑核）：

| 情形 | 总墙钟 | 测试段 |
|---|---|---|
| 暖路径（源码没动，只有增量链接） | **279–322 s（4.7–5.4 min）** | **155–175 s** |
| 暖路径（第 387 轮末次实测：K2 校验 + 文档门禁，其余增量） | **303.3 s（5.1 min）** | **170.5 s** |
| 冷路径（改了共享类型） | 976 s（16.3 min） | 161 s（编译 518 s 单独一步） |

目标里的两个硬指标是**测试段 ≤ 8 min、总墙钟 ≤ 15 min**：前两条达成；冷路径 16.3 min 如实记在
`choices.md` 一百七十一，没有把某一步挪出计时去凑数。

> 门禁第 2 步会 `python scripts/fetch_data.py --check`，缺就 `--all`。所以**新 clone 上的第一次
> `verify` 需要网络**（10 个小文件）；取回之后就是纯本地的。

### 10.2 单独跑某一类

| 想跑什么 | 命令 |
|---|---|
| 全部 Rust 测试（含慢档） | `cd qwb-core; cargo test --release -- --ignored` |
| 单个 crate | `cargo test -p qwb-scf` |
| 单个测试文件 | `cargo test -p qwb-engine --test kv_phase2_options` |
| Python 侧 | `cd qwb-studio; ..\.venv\Scripts\python.exe -m pytest -q`（需 `PYTHONPATH`） |
| 数据校验 | `cargo run -q -p qwb-cli --bin qwb -- data verify` |
| 第三方数据体检 / 取回 | `python scripts\fetch_data.py --check`（缺则 `--all`） |
| 账本合法性 | `python scripts\check_status.py`（阶段二账本加参数 `docs/status-v2.yaml`） |
| 诚实清单与账本对账 | `python scripts\check_usage_honest_list.py`（`--self-test` 跑检查器自测：三种漂移必须被抓到） |
| 数据许可一致性 | `python scripts\gen_data_licenses.py --check` |
| K10 等价性差异表 | `python scripts\check_numerics_equivalence.py --report` |

### 10.3 外部尺子

* **CCCBDB**（NIST SRD 101，Release 22）：`scripts/fetch_cccbdb_reference.py` +
  `scripts/gen_cccbdb_reference.py` → `tests/reference/cccbdb_*.csv`（每行带原始 `energy3x.asp` URL）。
* **PySCF 2.14.0**（只在 WSL 里装，仅当黑盒对照）。各能力的参考生成器（都只产数字，
  产物是 `tests/reference/*.csv`）：`pyscf_reference.py`、`pyscf_open_shell_reference.py`、
  `pyscf_uks_reference.py`、`pyscf_spin_xc_reference.py`、`pyscf_spin_mgga_reference.py`、
  `pyscf_uks_mgga_reference.py`、`pyscf_m06_exchange_reference.py`、`pyscf_mp2_reference.py`、
  `pyscf_rimp2_reference.py`、`pyscf_ri_invariants.py`、`pyscf_rij_reference.py`、
  `pyscf_tda_reference.py`（+ benzene/formaldehyde/acetone）、`pyscf_tdh_reference.py`、
  `pyscf_hyperfine_reference.py`、`pyscf_epr_reference.py`、`pyscf_angmom_reference.py`、
  `pyscf_polarizability.py`、`pyscf_dynamic_polarizability_reference.py`、
  `pyscf_quadrupole_reference.py`、`pyscf_nmr_shieldings_reference.py`、`pyscf_beta_ref.py`。
* **官方色散实现**：`dftd3_dispersion_reference.py` / `dftd4_dispersion_reference.py`（s-dftd3 / dftd4，
  产能量与解析梯度参考），门禁里 `validate_d3_data.py` 逐项比对。
* **SMD 上游实现**：`smd_cds_legacy_reference.py`（CDS 的遗留实现）、`smd_radii_reference.py`
  （参数化半径），产物在 `qwb-props/tests/reference/`。
* 边界（§14 #12）：这些只**产数字**，代码与数据一个字节都不进本仓库；本项目的实现不依赖它们，
  基组数据仍来自 BSE 导出（逐项 diff < 1e-12）。

### 10.4 慢档与 CI 分层

慢档（几十秒到几十分钟）一律 `#[ignore]`，需要显式 `--ignored` 才跑，**数字写在测试的文档注释
与账本里**，不删除。默认门禁只跑快档，所以一次 `verify` 是 5 分钟量级（§10.1）而不是小时级。
`scripts/check_reference_sources.py` 保证每个参考 CSV 家族都有带 `--check` 的生成脚本，
`scripts/check_error_budget.py` 保证误差预算的每一行都能解析到存在的路径或账本 WP。

---

## 11. 开发与扩展

### 11.1 目录与各层职责

```text
qwb-contracts/          契约层（改这里要慎重：K1–K6 冻结，阶段二只允许加性扩展 K7–K10）
  schema/               K1：由 Rust 类型用 schemars 生成，DO NOT EDIT
  engine-api/           K2：JobInput/JobResult/EngineBackend/EngineError
  studio/               K3 task/result schema、K4 `.qwb`、K5 说明、K6 `.pyi` 桩
qwb-data/               数据仓（K1）：registry.yaml + basis/ecp/functional/grid/solvent/
                        dispersion/numerics（第三方文件不入库，配方在 sources.json，见 §8.4）
qwb-core/               Rust workspace
  crates/qwb-registry   数据加载与校验（K1 的唯一实现）
  crates/qwb-basis      基组 → 内部表示（球谐/直角切换只在这里）
  crates/qwb-integrals  1e/2e 积分 + 解析导数 + RI 三中心 + 角动量/超精细积分
  crates/qwb-xc         泛函组件（LDA/GGA/hybrid/meta-GGA，含自旋与 τ 通道）
  crates/qwb-grid       Becke 分区 + 径向/角向求积
  crates/qwb-linalg     nalgebra 后端（faer 可选）
  crates/qwb-scf        RHF/ROHF/UHF/RKS/UKS 迭代、DIIS、ECP、解析梯度、Hessian、色散
  crates/qwb-props      偶极、布居、多极、COSMO/SMD（CDS）、EPR、UV/Vis 展宽
  crates/qwb-response   CPHF、极化率、TDA/CIS、TDHF、动态极化率与超极化率
  crates/qwb-gpu        cuBLAS 快路（运行时加载，CPU 永远可回退）
  crates/qwb-engine     K2 后端实现：数据 → SCF → JobResult
  crates/qwb-io         K3 状态机与布局、K4 `.qwb` 容器
  crates/qwb-cli        K5 命令行
qwb-studio/             Python 产品层
  qwb_studio/           pydantic 模型、容器、调度器、结果目录、CLI 薄壳、GUI、
                        数据取回（bse_data.py 转录 / datafetch.py 下载与校验）
  rust/qwb-py/          pyo3 扩展 qwb_engine（K6 的实现）
scripts/                门禁、数据生成/抓取、参考集生成、账本与来源检查
docs/                   status.yaml / status-v2.yaml（账本）、choices.md、USAGE.md、
                        error-budget.md、HANDOVER.md、unblock.md
```

### 11.2 加一项数据（以新基组为例）

**第三方数据（BSE 导出）走 `sources.json` 配方**，文件本身不入库：

1. 在 `qwb-data/sources.json` 的 `entries` 里加一条：`path` / `kind` / `fetch`
   （BSE 的 basis 名 + 元素）/ `transcribe`（名字/版本/引用/许可…）/ `sha256` 先填 64 个 0；
2. `python scripts\fetch_data.py --need basis\<name>.yaml` —— 它会下载、转录、打戳，
   并因为摘要与钉不符而**拒绝落盘**，同时打印真实摘要；
3. 把打印的摘要抄进 `sources.json` 那条，再跑一次（这次应当 `saved`）；
4. 在 `qwb-data/registry.yaml` 的 `bases:` 里登记 `name/path/sha256`（两处钉必须相等）；
5. `qwb data verify` + `python scripts\gen_data_licenses.py`；
6. 加一个 `d0x_*.rs` 测试：与导出逐项 diff（相对差 < 1e-12）+ 基函数数从导出算出来。

**离线手上有导出 JSON 时**用薄封装脚本（共用同一份转录实现）：

```powershell
python scripts\bse_to_qwb_data.py --input <json> --output qwb-data\basis\<name>.yaml --name <name> ...
qwb data sign qwb-data\basis\<name>.yaml        # 打 sha256（K1 规则只有这一份实现）
```

**自研数据**（泛函/网格/溶剂/色散/数值等级）直接提交文件并在 `registry.yaml` 登记 ——
它们不在 `sources.json` 里，所以"缺失"就是硬错误，不会被当成还没下载。

### 11.3 加一个泛函

1. 在 `qwb-core/crates/qwb-xc` 里实现 `FunctionalTerm`
   （能量密度、`v_ρ`、`v_σ`、可选 `v_{∇²ρ}`；meta-GGA 还要 `v_τ` 与 `meta_gga()` 声明）；
2. 写 `qwb-data/functional/<name>.yaml`：`components` + `parameters` + `requires` + 引用/许可；
3. 有限差分测试（点值导数 < 1e-9），然后跑 `t06_reference` 看外部基准是否仍对上；
4. hybrid 的 HF 交换写成 `- id: HfExchange, weight: 0.25`（数据侧声明，代码侧读成 α_hf）；
5. 需要自旋通道的泛函要把 `evaluate_spin` 也实现，否则开壳层会被显式拒绝（`uks` 只放行
   `svwn5`/`pbe`/`pbe0` 这类已有自旋实现的项）。

### 11.4 加一个作业类型

1. 在 `qwb-engine` 里实现后端路径（参考 `run_optimisation`），并在 `reject_unimplemented_options`
   里**显式拒绝**所有你还没审过的选项组合；
2. 结果里放不进 `JobResult` 字段的东西走 `artifacts/`（K3 允许自由形态）或旁路交接
   （参考 `OptSink`）；
3. `JobInput.options` 里已有的旋钮别新造（`max_opt_steps`/`opt_gradient_threshold_…` 等）；
4. 加测试，并在 `docs/status-v2.yaml` 里记"哪些体系实测过、数字是多少"。

### 11.5 提交前自检

```powershell
cd qwb-core; cargo fmt --all; cargo clippy --workspace --all-targets -- -D warnings
cd ..; python scripts\check_status.py; python scripts\check_status.py docs\status-v2.yaml
python scripts\gen_data_licenses.py --check; python scripts\check_reference_sources.py
python scripts\check_error_budget.py
pwsh -NoProfile -File scripts\verify.ps1
```

改了 `scripts/verify.ps1` 就必须同步改 `scripts/verify.sh`（两边覆盖同一份清单，门禁项数必须相等）。

---

## 12. 故障排查

### 12.1 报错 → 原因 → 怎么办

| 现象 | 退出码 | 常见原因 | 处理 |
|---|---|---|---|
| `data load failed: unknown basis 'xxx'` | 3 | 基组名不在注册表（大小写/拼写） | `qwb data list` 看真名 |
| `data load failed: unknown functional 'b3lyp'` | 3 | `b3lyp` 不在数据层（K1 拒绝 `requires.laplacian`） | 用 `pbe0`；B3LYP 只在测试里程序化构造 |
| 同上，但消息里有 `hint: 'xxx' 是第三方数据` | 3 | 文件还没取回（且自动取回被关掉/网络不通） | `qwb data install`（§8.4） |
| `data load failed: basis ... covers [...]` | 3 | 分子里有基组不覆盖的元素 | 换基组（§3.1 的表） |
| `invalid input: closed-shell RKS needs an even, positive electron count` | 1 | 奇数电子体系按闭壳层算 | 开壳层显式写 `scf_type` |
| 开壳层体系退出码 0、能量偏高、**没有 `spin` 块** | 0 | 没写 `scf_type`（已知缺陷：缺省路径按闭壳层算，见 §3.4） | 显式写 `uhf` / `rohf` / `uks` |
| `invalid input: options.scf_type = 'uks' is open-shell but multiplicity = 1` | 1 | 单重态配了开壳层 | 改多重度或改 `scf_type` |
| `options: missing field 'scf_max_iter'`（或别的阶段一字段） | 1 | `options` 的 9 个阶段一字段是必填 | 抄 `examples/` 的完整 `options` 块 |
| `scf not converged in N iterations` | 2 | 近简并/坏初猜/格点太细 | 加大 `scf_max_iter`；引擎会自动试多初猜与退火；meta-GGA 见 §3.11 |
| `unsupported feature: jobtype 'freq' is not wired …` | 4 | 用了 `freq` | 用 `sp`/`opt`；频率见 F-01/F-02 的测试 |
| `unsupported feature: geometry optimisation with an ECP …` | 4 | 带 ECP 的 opt | ECP 只支持 `sp` |
| `unsupported feature: options.scf_type = 'uks' currently supports 'sp' only` | 4 | 开壳层 UKS 优化 | 用 `sp`；或用 `uhf`/`rohf` 做 HF 优化 |
| `unsupported feature: options.solvent_model = 'pbf' is not implemented yet` | 4 | 想要 PBF | PBF 见 SV-04；SMD 用 `smd`（已接） |
| `unsupported feature: solvent 'water' currently supports 'sp' only` | 4 | 溶剂体系做 opt | 溶剂只支持单点 |
| `unsupported feature: ghost atoms … implemented for 'sp' only` | 4 | ghost + opt/freq/色散/ECP/溶剂 | 只做 `sp` |
| `unsupported feature: options.numerics is not wired yet` | 4 | 想用近似路径 | 去掉 `options.numerics`（IN-03/IN-04 未过 K10） |
| `env.json is missing: ... (result is not reproducible)` | 3 | 结果目录被删了文件 | 重跑；不要手工搬文件 |
| `cargo` 报 `link.exe` 1104 / 打不开 `.exe` | — | 上一次的测试进程还在跑（Windows 文件锁） | `Get-Process qwb | Stop-Process -Force` 后重试 |
| Python `No module named qwb_engine` | — | wheel 没装 | 按 §2.3 构建并安装 |
| Python 侧结果与 Rust 不一致 | — | 装了旧 wheel | 重建 wheel（§2.3），这是最常见的"改了没生效" |
| `No module named qwb_studio` | — | 没设 `PYTHONPATH` | `$env:PYTHONPATH='<仓库根>\qwb-studio'` |
| GUI 起不来（headless） | — | 没有显示 | `$env:QT_QPA_PLATFORM='offscreen'` |
| GUI 里作业很慢 | — | 用的是 debug 版 `qwb.exe` | 设 `QWB_ENGINE_BIN` 指向 release |
| `check_deps_centralized.sh` 报空行 | 1 | 仓库开了 `core.autocrlf=true`，文件是 CRLF | 脚本已处理；自己加脚本请剥 `\r` |

### 12.2 怎么知道"算得对不对"

按可信度从高到低：

1. **有外部基准的体系**（STO-3G 的 H2/H2O/CO/CH4/NH3/N2、苯的 B3LYP/PBE、开壳层的 O2/OH/NO/CH3/NH2、
   TDDFT 的 H2O/苯/甲醛、色散的二聚体、SMD 的 CDS 能量）→ 直接比 `tests/reference/*.csv`
   （每行带 DOI/URL）；
2. **没有外部值的体系**（def2-SVP 的能量、ECP 体系）→ 用与参照无关的不变量：
   `Tr(P·S) = 电子数`、偶极与实验比（同量级）、网格等级加密后能量差（< 0.1 mHa）、
   RI 的 `ri_error_hartree`、色散的解析梯度 vs 有限差分；
3. 再不行就跑 `qwb replay` 确认可复现 —— 但这只能证明可复现，**不能证明正确**。

### 12.3 SCF 不收敛怎么办

引擎已经在 `sp` 路径上按顺序试三级退路（日志里能看到）：

```text
scf: single guess failed (...), retrying with multistart
scf: retry multistart converged in 40 iterations: E = ...
# 或者
scf: retrying with annealed smearing (Fermi broadening, then integer occupations)
scf: annealing converged in 23 iterations: E = ...
```

仍然不收敛时你能调的：`scf_max_iter`（上限，退路不会超过它）、`diis_history`（近简并体系调到
16–24 有用）、`grid_level`（**meta-GGA 在 fine 上可能不收敛，见 §3.11**）。
**注意**：退火"收敛"不等于"算对"—— 苯/STO-3G/PBE 的退火会停在比真值高 19 Ha 的伪态上
（已知记录，引擎因此会在退火之后再做一个有界多初猜交叉验证取更低者）。

---

## 13. 常见问题

**Q：为什么算能量要 `--release`？**
A：debug 构建慢 10–50 倍。H2O/def2-SVP/PBE 的单点 release 实测 7 秒（debug 构建要十几分钟）。

**Q：`threads` 设了没用？**
A：4 指标积分张量单线程；真正的并行在 Fock 构建（按行并行、确定性归约）、Hessian 的位移与
`qwb batch` 的进程级并发。`--threads` 与 `options.threads` 目前只写进 `env.json`（记 1）。

**Q：为什么 `b3lyp` 不能写在 `input.yaml` 里？**
A：K1 的语义门禁拒绝 `requires.laplacian = true`（"no meta-GGA"，阶段一冻结范围），所以数据层
没有 `b3lyp.yaml`，写它会得到退出码 3 + "available: [b88, m06-2x, m06-l, pbe, pbe0, svwn5]"。
它在测试里程序化构造；数据层可用的 hybrid 是 `pbe0`（以及 54% 交换的 `m06-2x`）。

**Q：`solvent: water` 不写 `solvent_model` 会怎样？**
A：会算溶液相。`solvent_model` 缺省用溶剂数据里的 `cavity_model`（8 个溶剂都是 `pcm`），
所以 `solvent: water` + `sp` 直接给 COSMO/C-PCM 的结果（§3.7）。

**Q：开壳层为什么必须写 `scf_type`？**
A：因为缺省路径是闭壳层，而"偶数电子"检查挡不住 O2 这种体系（16 个电子）—— 实测它会得到一个
能量高 0.616 Ha、也没有 `spin` 块的"成功"结果（退出码 0），而同体系写 `scf_type: rohf` 就正确。
这条校验（缺省 + 多重度 > 1 时拒绝）目前**还没有实装**，所以引擎不会替你挡，务必自己写。

**Q：ZnO 为什么算得很慢？**
A：ZnO/LANL2DZ 的单初猜 DIIS 不收敛，引擎要依次试多初猜与退火才能收敛（release 下整条命令
约 4 分钟）。这不是错误路径，而是这个体系本身的已知难点（闭壳层受限的 O 原子是伪解）。

**Q：M06-2X 为什么慢？**
A：meta-GGA 的格点 XC 是 `O(n_points·n_ao²)`，常系数又比 LDA/GGA 大（τ 通道）。水/def2-SVP 在
normal 上 9.5 s，苯单体在 normal 上 906 s CPU 还没收敛完（`xc_cost_breakdown` 与 M2 的账本条目）。
CPU 侧能拿的 9% 已经拿了（改求和顺序会动位级回归，需要人批准）；GPU 侧要 NVRTC 自定义核（EX-03，blocked）。

**Q：结果里的能量能直接和实验比吗？**
A：不能。总能量是模型化学的量，和实验没有可比性；能比的是**能量差**（相互作用能、原子化能、
反应能）与**几何/频率**。本项目的频率判据用的是"标度后的谐振频率 vs 实验基频"。

**Q：怎么把结果发给别人复现？**
A：把整个结果目录（或 `.qwb` 容器）发过去。对方跑 `qwb replay <dir>` 即可；`env.json` 里记了
rustc 版本、`cargo.lock` 摘要与数据快照摘要（33 个文件的 sha256），数据被动过会直接报错。

**Q：clone 下来 `qwb-data/basis/` 是空的？**
A：不是坏了，是**故意的**：那 10 个基组/ECP 文件是 BSE 导出的第三方数据，仓库只提交配方
（`qwb-data/sources.json`）与 sha256 钉，避免再分发（§8.4）。跑第一个作业会自动取回，
也可以 `qwb data install` 一次备齐。

**Q：能完全离线用吗？**
A：能。取回过一次之后所有计算都不需要网络；离线机器也可以在有网机器上跑一次
`qwb data install`，把整个 `qwb-data/` 拷过去（摘要钉会校验两边一致）。
`$env:QWB_AUTO_FETCH='0'` 关掉自动取回。

**Q：数据取回失败/摘要不符怎么办？**
A：摘要不符意味着"上游内容或转录规则变了"，脚本会拒绝落盘并打印两个摘要。先看
`qwb-data/CHANGELOG.md` 与 `qwb-data/sources.json` 的 `transcribe` 段是不是有意改过；
确认要更新就改两处钉（`sources.json` 与 `registry.yaml`），再 `qwb data verify` +
`python scripts/gen_data_licenses.py`。

**Q：文档在哪？**
A：[`../project.md`](../project.md)（阶段一规范与验收判据）、[`../project-phase2.md`](../project-phase2.md)
（阶段二 v3.1）、[`status.yaml`](./status.yaml) / [`status-v2.yaml`](./status-v2.yaml)（账本，
每条判据都有实测数字）、[`choices.md`](./choices.md)（决策与踩坑日志）、
[`error-budget.md`](./error-budget.md)（误差预算）、[`HANDOVER.md`](./HANDOVER.md)（交接要点）、本文。
