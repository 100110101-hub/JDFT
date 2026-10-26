# QuantumWorkbench 使用手册

> 这份文档写给**要用它算东西的人**：怎么装、怎么跑、每个参数什么意思、结果文件怎么看、
> 出问题怎么查。规范本身（契约、验收判据、决策台账）在 [`../project.md`](../project.md)；
> 当前进度与每条判据的实测数字在 [`status.yaml`](./status.yaml)；
> 为什么这么做在 [`choices.md`](./choices.md)。
>
> 文中所有命令都在本机（Windows 11 + Rust 1.97.1 + Python 3.12）实际跑过，
> 输出片段是真实输出；命令后面标注「（实测）」的地方，数字来自本轮实测。

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

# ② 跑第一个作业：H2 / STO-3G / SVWN5 单点能（约 1 秒）
cd qwb-core
cargo run --release -q -p qwb-cli --bin qwb -- run examples/h2_sto3g_svwn5.yaml --out results
```

第一次跑会先看到两行取数提示（实测）：

```text
qwb: note: 本地缺 7 个第三方数据文件（basis/sto-3g.yaml, …, ecp/lanl2dz-zn.yaml），正在按需取回…
数据缺失/过期：basis/sto-3g.yaml，正在从 BSE 取回…
fetch   basis/sto-3g.yaml（missing）
saved   basis/sto-3g.yaml  sha256=fce3af14b8b3ea121844756117a2489b5f7859be452be02b54509dc20b8ff249
已取回：basis/sto-3g.yaml
qwb: note: 已取回 1 个第三方数据文件
```

**为什么不是开箱即算**：`qwb-data/basis/` 与 `qwb-data/ecp/` 里的基组/ECP 是从
[Basis Set Exchange](https://www.basissetexchange.org/) 导出的第三方数据，仓库**不提交**
它们（LANL2DZ 的 ECP 原始论文没有明确再分发条款），只提交配方与 sha256 钉。
`run` / `batch` 会自动把**这份输入要用到的**文件取回来（只下缺的那几个），取回时会核对
摘要钉，对不上就报错、不落盘。详见 §8.4。想提前一次备齐：

```powershell
qwb data install                                        # 全部 7 个文件
python scripts\fetch_data.py --all                      # 等价（不需要 PYTHONPATH）
```

取回之后就是纯本地：同一台机器上再跑任何作业都不需要网络。

真实输出（实测）：

```text
scf     converged = true
energy  -1.1570005110 Ha
task    results\h2_sto3g_svwn5
```

结果在 `qwb-core/results/h2_sto3g_svwn5/`，里面是 K3 冻结布局：

```text
h2_sto3g_svwn5/
  input.yaml            # 这次算的输入（原样）
  result.json           # K2 的 JobResult：能量/轨道/偶极/日志/env
  env.json              # 复现环境（rustc、cargo.lock 摘要、数据快照摘要、线程数…）
  data_snapshot.json    # 用到的每个数据文件的 sha256
  log.txt               # Reporter 输出 + 引擎日志
  artifacts/            # 自由形态产物：populations.json、homo.cube、lumo.cube、optimized.xyz…
```

再跑一个**几何优化**（H2 / STO-3G / PBE，解析梯度，约 13 秒）：

```powershell
cargo run --release -q -p qwb-cli --bin qwb -- run examples/h2_sto3g_pbe_opt.yaml --out results
```

```text
opt     3 steps, max|grad| = 3.436e-5 Ha/Bohr
geometry results\h2_sto3g_pbe_opt\artifacts\optimized.xyz
scf     converged = true
energy  -1.1520971657 Ha
task    results\h2_sto3g_pbe_opt
```

`artifacts/optimized.xyz` 就是优化后的几何（标准 XYZ，单位 Å）：

```text
2
qwb opt: E = -1.1520971657 Ha, steps = 3, converged = true
H      -0.000000000000     0.000000000000    -0.367256321978
H      -0.000000000000     0.000000000000     0.367256321978
```

---

## 2. 安装与构建

### 2.1 需要什么

| 组件 | 版本 | 用途 | 必需？ |
|---|---|---|---|
| Rust | **1.97.1**（仓库根 `rust-toolchain.toml` 固定） | 引擎与 CLI | 必需 |
| C 工具链 | MSVC Build Tools（Windows）/ gcc（Linux） | 链接 | 必需 |
| 网络 | 任意 | **首次**取回第三方基组/ECP（§8.4）；之后纯离线 | 首次必需 |
| Python | 3.10–3.12 | Python API、GUI、门禁里的 Python 侧 | 用 Python 面才必需 |
| WSL 的 `bash` | 任意 | 三个 bash 门禁脚本（`scripts/verify.ps1` 会调） | 跑门禁才必需 |
| PySide6 / vtk | 见 §2.3 | GUI 与 3D | 用 GUI 才必需 |

> **不需要**：PySCF、libxc、ORCA、Gaussian、Psi4、ase 等任何电子结构库。
> 引擎全部自研；PySCF 只在 `scripts/pyscf_reference.py` 里当**外部尺子**用过一次
> （只产数字，不参与实现，见 §10.3）。

### 2.2 只算东西（不跑门禁）

```powershell
cd qwb-core
cargo build --release            # 约 3–5 分钟；产物 target/release/qwb.exe
cargo test --release -- --ignored # 可选：把慢档（频率/优化/Hessian）也跑一遍
```

调试构建（`cargo build`）慢 10–50 倍，**算真实体系请一律 `--release`**。
本手册里的耗时数字都是 release。

数据不用手动准备：第一次 `run` 会把这份输入需要的第三方文件取回来（§8.4）。
想一次备齐全部 7 个：`qwb data install`。

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

改过 Rust 侧（尤其是 `qwb-engine` / `qwb-scf`）之后**必须重建一次 wheel**，
否则 Python 侧跑的仍是旧引擎 —— 这是最常见的"改了没生效"。

### 2.4 让 Python 找得到 `qwb_studio`

没有 `pip install -e`，靠 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = '<仓库根>\qwb-studio'
```

否则 `python -m qwb_studio …` 报 `No module named qwb_studio`。

---

## 3. 输入文件：`input.yaml` 逐字段说明

一个作业 = 一个 YAML（K4 的 `input.yaml`）。字段名与 K2 的 `JobInput` **一一对应**，
未知字段会被 `deny_unknown_fields` 直接拒绝（不静默忽略）。

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
  multiplicity: 1                # 自旋多重度；闭壳层只接受 1

basis_name: def2-svp             # 必须是 qwb-data/registry.yaml 里注册过的名字
method_name: pbe                 # hf / svwn5 / pbe / pbe0 / b88（见 §9.1）
grid_level: normal               # coarse | normal | fine | veryfine
solvent: vacuum                  # 目前只支持 vacuum（隐式溶剂是接口占位）
jobtype: sp                      # sp | opt（freq 还没接线，会以退出码 4 拒绝）

options:
  scf_max_iter: 128                        # SCF 最大步数（引擎的退路不会超过这个数）
  scf_energy_threshold_hartree: 1.0e-8     # |ΔE| 收敛判据
  scf_density_threshold: 1.0e-6            # max|ΔP| 收敛判据
  diis_history: 8                          # DIIS 子空间长度
  threads: 0                               # 0 = 自动；本构建的积分层是单线程，见 §9.4
  max_opt_steps: 100                       # jobtype: opt 的最大步数
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4   # opt 的力判据（最大分量）
  freq_step_angstrom: 0.01                 # 预留给 freq（尚未接线）
  ecp_name: null                           # null = 全电子；有 ECP 时写名字，见 §3.2
```

### 3.1 几何与单位

* 坐标单位是 **Å**（`position_angstrom`），内部统一转 Bohr。
* 原子顺序 = 输出里轨道/梯度的顺序，换顺序会换结果文件的形状（能量不变）。
* 元素必须被所选的基组覆盖，否则报 `data load failed`（退出码 3）。覆盖范围：

| 基组 | 元素 |
|---|---|
| `sto-3g` | H C N O F |
| `def2-svp` | H C N O |
| `def2-tzvp` | H C N O F |
| `6-311g(d,p)` | H C N O |
| `lanl2dz` | H O Cu Zn |

### 3.2 ECP（有效核心势）

`ecp_name` 指向 `qwb-data/ecp/` 下的一个条目：`lanl2dz-cu`（Cu，10 芯电子，`z_eff=19`）或
`lanl2dz-zn`（Zn，18 芯，`z_eff=12`）。要点：

* 基组必须是**同一族**的 `lanl2dz`（`Registry::load_pair` 是唯一配对入口，配对按元素校验）；
* 点电荷吸引用 `z_eff`、超出部分作为矩阵元加进 `H_core`；
* **带 ECP 的作业只支持 `sp`**：`opt`/`freq` 会以退出码 4 明确拒绝（ECP 对核坐标的导数还没实现）。

最小例子（`qwb-core/examples/cuh_lanl2dz_pbe.yaml`，实测 −213.9067554378 Ha @ grid-coarse）：

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
  ecp_name: lanl2dz-cu
```

### 3.3 从哪抄模板

`qwb-core/examples/` 下的六个文件覆盖了常见组合：

| 文件 | 体系 | 作业 |
|---|---|---|
| `h2_sto3g_svwn5.yaml` | H2 / STO-3G / SVWN5 | sp（最快，约 1 秒） |
| `h2o_sto3g_svwn5.yaml` | H2O / STO-3G / SVWN5 | sp |
| `h2_sto3g_pbe_opt.yaml` | H2 / STO-3G / PBE | opt（约 13 秒） |
| `h2o_def2svp_pbe_opt.yaml` | H2O / def2-SVP / PBE | opt（慢档，分钟级） |
| `cuh_lanl2dz_pbe.yaml` | CuH / LANL2DZ+ECP / PBE | sp |
| `zno_lanl2dz_pbe.yaml` | ZnO / LANL2DZ+ECP / PBE | sp（SCF 需要退火退路） |

---

## 4. 命令行 `qwb`

### 4.1 退出码（K5 冻结）

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 输入错误（YAML 语法、字段越界、多重度 0…） |
| 2 | 收敛失败（SCF 或几何优化没到判据） |
| 3 | 数据校验失败（基组/泛函名不存在、sha256 不符、缺 `env.json`） |
| 4 | 内部错误 / 不支持的 feature（ECP 的 opt、非 vacuum 溶剂、`freq`…） |

脚本里判成功请只看退出码，别 grep stdout。

### 4.2 `qwb run` —— 跑一个作业

```powershell
qwb run <INPUT_YAML> [--out <OUT>] [--threads <N>]
```

* `--out` 默认 `results`；结果目录名 = 输入文件的 **stem**（`h2.yaml` → `results/h2/`）。
* `--threads` 会覆盖 `options.threads`，但本构建的积分层是单线程，只在日志里记账（§9.4）。
* 成功时 stdout 是给人看的三行摘要，**真相在结果目录里**：

```text
scf     converged = true
energy  -1.1570005110 Ha
task    results\h2_sto3g_svwn5
```

### 4.3 `qwb batch` —— 批量

```powershell
qwb batch "jobs/*.yaml" --parallel 4 --out results
```

* 队列写在 `<out>/queue.json`（JSON 数组，每项 `{input, task_id, state}`，原子写）。
* `kill -9` 之后重跑同一条命令：**`queued` 的接着做**；停在 `running` 的**不会**回到
  `queued`（K3 没有那条边），会报告出来并以非零退出码结束 —— 用新 task id 重提。
* `--parallel` 是**进程级**并发（每个作业一个子进程），不是线程。

### 4.4 `qwb inspect` —— 看结果目录 / `.qwb`

```powershell
qwb inspect results\h2_sto3g_svwn5
qwb inspect projects\test.qwb
```

检查 K3 必需文件、`env.json` 完整性（G5）；缺 `env.json` 时**退出码 3**。

真实输出（实测）：

```text
kind    result directory
path    results\h2_sto3g_svwn5
file    input.yaml           ok
file    result.json          ok
file    log.txt              ok
file    data_snapshot.json   ok
file    env.json             ok
env     rustc rustc 1.97.1 (8bab26f4f 2026-07-14), threads 1, os windows x86_64
env     data snapshot: 15 file(s)
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
qwb data list                     # 列出注册表（bases / ecps / functionals / grids）
qwb data show pbe                 # 打印一个条目的完整 YAML
qwb data verify                   # 校验整个注册表（G2：sha256 + schema + 语义）
qwb data verify path\file.yaml    # 校验单个文件
qwb data sign path\file.yaml      # 按规范化摘要重新签名（改了数据之后）
qwb data install                  # 取回缺的第三方数据（见 §8.4）
qwb data install --check          # 只检查：缺/坏 → 退出码 3（K5 的数据失败码）
qwb data install --need basis\lanl2dz.yaml ecp\lanl2dz-cu.yaml
qwb data install --for-input qwb-core\examples\cuh_lanl2dz_pbe.yaml
qwb data install --all --force    # 全部重下（摘要已正确也重下）
```

`qwb data verify` 真实输出（实测）：

```text
ok   basis       ../qwb-data/basis/sto-3g.yaml (0 error(s), 0 warning(s))
...
16 file(s) checked, 16 passed, 0 failed
```

没取回第三方数据时，那 7 个条目会被**单独**记为失败（不是"整个注册表坏了"），
消息里直接给取数命令：

```text
FAIL basis       qwb-data\basis/sto-3g.yaml (1 error(s), 0 warning(s))
       [registry.path] cannot read qwb-data\basis/sto-3g.yaml: ...（第三方数据，本地未取回：跑 `qwb data install`）
16 file(s) checked, 9 passed, 7 failed
```

**数据根**怎么找：`QWB_DATA_ROOT` 环境变量 > 从当前目录逐级向上找 `qwb-data/`。
CLI 没有 `--data-root` 参数（§14 #17 的决定）；Python 侧的数据入口有 `--data-root`。

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

result = qe.run_job(job, "<仓库根>/qwb-data")   # 一次性
print(result.energy_hartree, result.scf_converged)

engine = qe.Engine("<仓库根>/qwb-data")          # 复用注册表
result = engine.run(job)                                    # 可反复调用
print(result.mo_energies_hartree[:5], result.dipole_debye)
```

要点：

* `qe.JobInput(molecule, basis, method, grid_level=…, solvent=…, jobtype=…, options=…)`
  的字段可以之后赋值（`job.jobtype = "opt"`），**未知字段没有**（桩里没有就真的没有）。
* 异常层次：`EngineError` 基类，`InvalidInputError`/`ScfNotConvergedError`/`DataError`/
  `UnsupportedError` 四个子类，每个都带 `exit_code`（与 K5 一致）。
* **GIL 真的释放**：计算期间别的 Python 线程照跑；两个作业并发墙钟远小于串行和
  （门禁里的 `test_engine_binding.py` 钉这一点）。
* 没有撤销/中断 API；要取消请用子进程 + kill（GUI 的 `engine` runner 就是这么做的）。

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

* **Build 页的下拉框来自 `qwb-data/registry.yaml`**：界面里能选的就是数据里有的
  （元素按所选基组过滤）。`b3lyp` 不在注册表里（K1 拒绝 `requires.laplacian`），所以不出现。
* **Data 页**是手动入口（下载在工作线程里跑，界面不卡）：表格每行是 `sources.json` 的
  一条清单，「缺失 / 摘要不符」会标黄，下载日志显示在下方文本框。
  提交作业时不用先来这里 —— worker 是 `qwb run`，它会自动取回缺的文件。
  离线机器可以在这里一次备齐，或直接拷整个 `qwb-data/`。
* 作业类型只给 `sp` / `opt`：`freq` 还没接到引擎上，给一个必然失败的选项是坑用户。
* **取消**：`queued` 立即落 `cancelled`；`running` 先写协作取消标志，超过宽限期再 kill。
* **崩溃恢复**：重启后从 `<out>/queue.json` 重建 Task Panel；停在 `running` 的任务落
  `failed`（reason 带 `interrupted`），`queued` 的保留可以直接接着跑。
* **`QWB_ENGINE_BIN` 一定要指向 release 二进制**：默认会优先用
  `qwb-core/target/debug/qwb.exe`，debug 构建慢 10–50 倍，def2-SVP 的 opt 会从几分钟变成几小时。
* 提交真作业时 GUI 会把 `input.yaml` 暂存到 `<out>/.staging/<task_id>.yaml`，**不污染**
  K3 结果目录；`qwb run` 用文件 stem 当任务 id，所以文件名必须等于 task id。

无显示器环境（服务器/CI）加 `QT_QPA_PLATFORM=offscreen`，门禁里的 GUI 测试就是这么跑的。

---

## 7. 结果目录怎么看

### 7.1 文件清单（K3 冻结）

| 文件 | 内容 | 缺了会怎样 |
|---|---|---|
| `input.yaml` | 这次作业的输入（原样回写） | `qwb inspect` 报缺失 |
| `result.json` | K2 `JobResult`：`energy_hartree`/`gradient_hartree_per_bohr`/`hessian…`/`dipole_debye`/`mo_energies_hartree`/`mo_coeffs`/`scf_converged`/`data_snapshot`/`log`/`env` | 同上 |
| `env.json` | 复现环境（G5：`engine_version`/`rustc_version`/`cargo_lock_sha256`/`data_snapshot`/`threads`/`os`/`created_at_utc`） | **退出码 3**，结果视为不完整 |
| `data_snapshot.json` | 用到的每个数据文件的 sha256（`path -> digest`） | `qwb inspect` 报缺失 |
| `log.txt` | `Reporter` 输出 + 引擎日志（SCF 每步、优化每步、ECP 行…） | `qwb inspect` 报缺失 |
| `artifacts/` | 自由形态产物，见下 | 允许为空 |

`artifacts/` 里目前会写：

| 文件 | 什么时候 | 内容 |
|---|---|---|
| `populations.json` | 每次 `run` | Mulliken / Löwdin 布居与电荷（3D 的 ESP 着色用它） |
| `homo.cube` / `lumo.cube` | 每次 `run` | Gaussian cube 格式的轨道等值面输入（0.35 Bohr 间距、4 Bohr 外扩、2e6 点上限） |
| `optimized.xyz` | `jobtype: opt` | 优化后的几何（标准 XYZ，Å）；3D 面板优先读它 |
| `optimization.json` | `jobtype: opt` | 轨迹 `[step, E, max|grad|]`、步数、是否解析梯度、退化原因 |

### 7.2 读 `result.json` 的两个坑

1. `mo_coeffs` 是**系数矩阵**（列 = MO，满足 `CᵀSC = 1`），不是密度矩阵。
   以前的版本把密度塞在这里，字段名与内容不符，已修（见 `choices.md`）。
2. 矩阵按 K2 的约定是 **list-of-list**（行 = 第一个索引），
   `gradient_hartree_per_bohr` 的形状是 `3N×1`（按原子再按 x/y/z 摊平）。

### 7.3 Python 读结果

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
basis       sto-3g        basis/sto-3g.yaml        H C N O F
basis       def2-svp      basis/def2-svp.yaml      H C N O
basis       def2-tzvp     basis/def2-tzvp.yaml     H C N O F
basis       6-311g(d,p)   basis/6-311g-dp.yaml     H C N O
basis       lanl2dz      basis/lanl2dz.yaml       H O Cu Zn
ecp         lanl2dz-cu    ecp/lanl2dz-cu.yaml     Cu 10 芯 / z_eff 19
ecp         lanl2dz-zn    ecp/lanl2dz-zn.yaml     Zn 18 芯 / z_eff 12
functional  pbe svwn5 pbe0 b88
grid        grid-coarse / grid-normal / grid-fine / grid-veryfine
```

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

1. 改完 → `qwb data sign <file>`（规范化摘要，与"文件内 sha256 + registry 条目"三处一致）；
2. → `qwb data verify`（schema + 语义 + sha256 交叉校验）；
3. → `python scripts/gen_data_licenses.py`（重生成 `DATA_LICENSES.md`，门禁 D-15 会比对）；
4. → 重跑与之相关的测试（基组换了就跑 D-03/D-04/D-07，泛函换了就跑 `t06_reference`）。

**外部数据**（BSE 导出、Stuttgart/def2-ECP）的许可逐条确认后才允许写 `redistribute_ok: true`；
`lanl2dz` 的两个 ECP 文件目前是 `noassertion` + 两个 `false`（原始论文没写明确的再分发条款）。

### 8.4 第三方数据不在仓库里（按需取回）

`qwb-data/` 下有 7 个文件**不进仓库**：

| 文件 | 来源 |
|---|---|
| `basis/sto-3g.yaml`、`basis/def2-svp.yaml`、`basis/def2-tzvp.yaml`、`basis/6-311g-dp.yaml`、`basis/lanl2dz.yaml` | Basis Set Exchange 导出 |
| `ecp/lanl2dz-cu.yaml`、`ecp/lanl2dz-zn.yaml` | Basis Set Exchange 导出的 LANL2DZ ECP |

原因是**再分发**：LANL2DZ 的 ECP 原始论文（Hay–Wadt）没有给出明确的再分发条款，
把文件放进公开仓库就等于再分发（project.md §14 #20）。所以仓库提交的是**配方**
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

GUI：**Data 页 → 「下载缺失」**（或菜单「数据 → 下载缺失」）。

常用开关：

```powershell
qwb data install --check                               # 只体检：7/7 就绪 / 缺哪个
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

**自动取回**：`qwb run` / `qwb batch` 开始前会检查一遍，缺就按这份输入需要的取回
（`--for-input` 的语义），只下缺的那几个；取数失败只 warning，不阻断不需要那份数据的作业。
关闭自动取回：`$env:QWB_AUTO_FETCH = '0'`（此时缺数据的作业以退出码 3 报错，
消息里给出取数命令）。

**完全离线**：在有网机器上跑一次 `qwb data install`，把整个 `qwb-data/` 目录拷过去即可
（`sources.json` 的摘要钉保证两边一致）。

**自研数据不吃这一套**：`functional/`、`grid/`、`solvent/` 是随仓库提交的。它们缺失是
**硬错误**（数据根被裁剪过），`Registry::open` 不会把它当成"还没下载"而静默跳过 ——
错误消息也不一样（`（该文件不是第三方数据：数据根被裁剪过，请从仓库恢复）`）。

---

## 9. 能算什么、不能算什么（诚实清单）

### 9.1 方法

| `method_name` | 类型 | 状态 |
|---|---|---|
| `hf` | 闭壳层 RHF | 能算（引擎内部翻译成"空 terms + α_hf = 1"） |
| `svwn5` | LDA（VWN5） | 能算 |
| `pbe` | GGA | 能算 |
| `b88` | GGA 交换（单独的 B88） | 能算，一般作为组件 |
| `pbe0` | hybrid（α_hf = 0.25） | 能算 |
| `b3lyp` | hybrid（α_hf = 0.20） | **数据层不注册**：K1 的 phase-1 语义门禁拒绝 `requires.laplacian = true`（冻结契约）。只能在测试里程序化构造 |

### 9.2 作业类型

| `jobtype` | 状态 |
|---|---|
| `sp` | 能算（含 ECP 体系） |
| `opt` | 能算（BFGS + Armijo 线搜索；**ECP 体系拒绝**） |
| `freq` | **没接到 `EngineBackend`**，会以退出码 4 拒绝。`hessian::numerical_hessian` 存在且有测试（F-01/F-02 直接调它），但没接进作业分发 |

### 9.3 精度现状（都有实测记录）

* **能量**：STO-3G 上六个体系 × 五个模型化学与 CCCBDB 差 ≤ 4e-6 Ha；
  苯（36 个基函数）B3LYP grid-normal 差 −1.16 mHa、PBE 差 −1.33 mHa（判据 2 mHa）。
* **几何**：PBE0/def2-SVP 的 H2O O–H = 0.9623 Å（实验 0.9572，差 0.0051）；
  PBE 一族系统性偏长 ~0.011 Å（泛函偏差，不是实现问题）。
* **频率**：PBE/6-311G(d,p) × 0.991 标度后最差 21.5 cm⁻¹（判据 30）；
  PBE0 一族 60.5 cm⁻¹（不通过，如实记作对照组）。
* **ECP**：矩阵元与解析解差 3e-14；CuH 与 ZnO 单点跑通。
* **ZnO/LANL2DZ 的 SCF 需要退路**：单初猜 DIIS 不收敛，引擎会自动退到"多初猜 →
  费米展宽退火"（23 步收敛）。这是正常路径，不是错误。
* **网格敏感度**：B3LYP/PBE 在苯上 grid-coarse 反而比 grid-normal 更接近基准
  （−0.01 vs −1.16 mHa），量级 ~1 mHa —— 需要 1 mHa 级能量时要注意。

### 9.4 明确没做

* **溶剂**：只有 `vacuum`；隐式溶剂是接口与数据占位。
* **ECP 的导数**：带 ECP 的几何优化/频率不支持。
* **并行**：积分层单线程；并行只出现在 Hessian 的位移与 `qwb batch` 的进程级并发。
  `--threads` 目前只记账。
* **meta-GGA、ROHF/UKS、TDDFT、NBO、TS/IRC/scan**：不在第一阶段范围。
* **`.qwb` 容器**：K4 读写与双向互操作已通过测试，但 CLI 还没有"打包"子命令
  （Python 侧 `QwbContainer` 可以）。

---

## 10. 验收与门禁

### 10.1 一条命令

```powershell
# Windows（会通过 WSL 跑三个 bash 门禁）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1

# Linux / CI / WSL
bash scripts/verify.sh
```

25 个步骤：1 个数据取回步骤（第三方基组/ECP 缺则取回，§8.4）+ 3 个 bash 门禁
（依赖集中/unsafe 策略/rayon 确定性）+ K2 契约 crate 的 fmt/clippy/doc/test +
`qwb-core` 的 fmt/clippy/test + I-06/I-00/deny + K1 schema 不漂移 + G2 数据校验 +
D-15 数据许可 + 10.6 账本合法性 + K6 mypy ×2 + pyo3 fmt/clippy/构建安装 +
Python 侧 182 个测试 + `.qwb` 夹具新鲜度。

> 门禁第一步会 `python scripts/fetch_data.py --check`，缺就 `--all`。所以**新 clone 上的
> 第一次 `verify` 需要网络**（7 个小文件）；取回之后就是纯本地的。

最后一次实测：**25/25 全绿**（`verify-r59.log`，Python 侧 182 passed）。

### 10.2 单独跑某一类

| 想跑什么 | 命令 |
|---|---|
| 全部 Rust 测试（含慢档） | `cd qwb-core; cargo test --release -- --ignored` |
| 单个 crate | `cargo test -p qwb-scf` |
| 单个测试文件 | `cargo test -p qwb-engine --test t5_ecp` |
| Python 侧 | `cd qwb-studio; ..\.venv\Scripts\python.exe -m pytest -q`（需 `PYTHONPATH`） |
| 数据校验 | `cargo run -q -p qwb-cli --bin qwb -- data verify` |
| 第三方数据体检 / 取回 | `python scripts\fetch_data.py --check`（缺则 `--all`） |
| 账本合法性 | `python scripts\check_status.py` |
| 数据许可一致性 | `python scripts\gen_data_licenses.py --check` |

### 10.3 外部尺子

* **CCCBDB**（NIST SRD 101，Release 22）：`scripts/fetch_cccbdb_reference.py` +
  `scripts/gen_cccbdb_reference.py` → `tests/reference/cccbdb_*.csv`（每行带原始 `energy3x.asp` URL）。
* **PySCF**（只在 WSL 里装，仅当黑盒对照）：
  `python3 scripts/pyscf_reference.py --grid-level 3` → H2O/def2-SVP/PBE 的独立值
  −76.271970822413 Ha（本项目差 2.87e-5 Ha）。
* 边界（§14 #12）：这些只**产数字**，代码与数据一个字节都不进本仓库；
  本项目的实现不依赖它们，基组数据仍来自 BSE 导出（逐项 diff < 1e-12）。

### 10.4 CI 分层（V-04）

慢档（几十秒到几十分钟）一律 `#[ignore]`，需要显式 `--ignored` 才跑，**数字写在测试的
文档注释与 `status.yaml` 里**，不删除。默认门禁只跑快档，所以一次 `verify` 约 40 分钟
（其中 `t06_reference` 的 7 体系 × 5 泛函占大头）。

---

## 11. 开发与扩展

### 11.1 目录与各层职责

```text
qwb-contracts/          契约层（改这里要慎重：K1–K6 冻结）
  schema/               K1：由 Rust 类型用 schemars 生成，DO NOT EDIT
  engine-api/           K2：JobInput/JobResult/EngineBackend/EngineError
  studio/               K3 task/result schema、K4 定义、K5 说明、K6 .pyi 桩
qwb-data/               数据仓（K1）：registry.yaml + basis/ecp/functional/grid/solvent
                        （basis/ecp 里的 7 个第三方文件不入库，配方在 sources.json，见 §8.4）
qwb-core/               Rust workspace
  crates/qwb-registry   数据加载与校验（K1 的唯一实现）
  crates/qwb-basis      基组 → 内部表示（球谐/直角切换只在这里）
  crates/qwb-integrals  1e/2e 积分 + 解析导数（Obara-Saika / McMurchie-Davidson）
  crates/qwb-xc         泛函组件（LDA/GGA/hybrid）
  crates/qwb-grid       Becke 分区 + 径向/角向求积
  crates/qwb-linalg     nalgebra 后端（faer 可选）
  crates/qwb-scf        RHF/RKS 迭代、DIIS、ECP、解析梯度、Hessian
  crates/qwb-props      偶极、Mulliken/Löwdin 布居
  crates/qwb-engine     K2 后端实现：数据 → SCF → JobResult
  crates/qwb-io         K3 状态机与布局、K4 `.qwb` 容器
  crates/qwb-cli        K5 命令行
qwb-studio/             Python 产品层
  qwb_studio/           pydantic 模型、容器、调度器、结果目录、CLI 薄壳、GUI、
                        数据取回（bse_data.py 转录 / datafetch.py 下载与校验）
  rust/qwb-py/          pyo3 扩展 qwb_engine（K6 的实现）
scripts/                门禁、数据生成/抓取、账本校验
docs/                   status.yaml（进度账本）、choices.md（决策日志）、
                        USAGE.md（本文）、unblock.md、error-budget.md、HANDOVER.md
```

### 11.2 加一项数据（以新基组为例）

**第三方数据（BSE 导出）走 `sources.json` 配方**，文件本身不入库：

1. 在 `qwb-data/sources.json` 的 `entries` 里加一条：`path` / `kind` / `fetch`
   （BSE 的 basis 名 + 元素）/ `transcribe`（名字/版本/引用/许可…）/ `sha256` 先填 64 个 0；
2. `python scripts\fetch_data.py --need basis\<name>.yaml` —— 它会下载、转录、打戳，
   并因为摘要与钉不符而**拒绝落盘**，同时打印真实摘要；
3. 把打印的摘要抄进 `sources.json` 那条，再 `python scripts\fetch_data.py --need basis\<name>.yaml`
   （这次应当 `saved`）；
4. 在 `qwb-data/registry.yaml` 的 `bases:` 里登记 `name/path/sha256`（两处钉必须相等）；
5. `qwb data verify` + `python scripts\gen_data_licenses.py`；
6. 加一个 `d0x_*.rs` 测试：与导出逐项 diff（相对差 < 1e-12）+ 基函数数从导出算出来。

**离线手上有导出 JSON 时**（不经过网络）用薄封装脚本，两者共用同一份转录实现：

```powershell
python scripts\bse_to_qwb_data.py --input <json> --output qwb-data\basis\<name>.yaml --name <name> ...
qwb data sign qwb-data\basis\<name>.yaml        # 打 sha256（K1 规则只有这一份实现）
```

**自研数据**（泛函/网格/溶剂）直接提交文件，并在 `registry.yaml` 登记 —— 它们不在
`sources.json` 里，所以"缺失"就是硬错误，不会被当成还没下载。

### 11.3 加一个泛函

1. 在 `qwb-core/crates/qwb-xc` 里实现 `FunctionalTerm`
   （能量密度、`v_ρ`、`v_σ`、可选 `v_{∇²ρ}`）；
2. 写 `qwb-data/functional/<name>.yaml`：`components` + `parameters` + 引用/许可；
3. 有限差分测试（点值导数 < 1e-9），然后跑 `t06_reference` 看外部基准是否仍对上；
4. hybrid 的 HF 交换写成 `- id: HfExchange, weight: 0.25`（数据侧声明，代码侧读成 α_hf）。

### 11.4 加一个作业类型

1. 在 `qwb-engine` 里实现后端路径（参考 `run_optimisation`），
2. 结果里放不进 `JobResult` 字段的东西走 `artifacts/`（K3 允许自由形态）或旁路交接
   （参考 `OptSink`）；
3. `JobInput.options` 里已有的旋钮别新造（`max_opt_steps`/`opt_gradient_threshold_…` 等）；
4. 加测试，并且在 `docs/status.yaml` 里记"哪些体系实测过、数字是多少"。

### 11.5 提交前自检

```powershell
cd qwb-core; cargo fmt --all; cargo clippy --workspace --all-targets -- -D warnings
cd ..; python scripts\check_status.py; python scripts\gen_data_licenses.py --check
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1
```

---

## 12. 故障排查

### 12.1 报错 → 原因 → 怎么办

| 现象 | 退出码 | 常见原因 | 处理 |
|---|---|---|---|
| `data load failed: unknown basis 'xxx'` | 3 | 基组名不在注册表（大小写/拼写） | `qwb data list` 看真名 |
| 同上，但消息里有 `hint: 'xxx' 是第三方数据` | 3 | 文件还没取回（且自动取回被关掉/网络不通） | `qwb data install`（§8.4） |
| `data load failed: basis ... covers [...]` | 3 | 分子里有基组不覆盖的元素 | 换基组（§3.1 的表） |
| `invalid input: closed-shell RKS needs an even, positive electron count` | 1 | 开了壳层体系（如 O 原子、OH 自由基） | 本构建只有闭壳层；换体系 |
| `scf not converged in N iterations` | 2 | 近简并/坏初猜 | 加大 `scf_max_iter`；引擎会自动试多初猜与退火，仍失败才报错 |
| `unsupported feature: jobtype 'freq' is not wired …` | 4 | 用了 `freq` | 用 `sp`/`opt`，频率见 F-01/F-02 的测试 |
| `unsupported feature: geometry optimisation with an ECP …` | 4 | 带 ECP 的 opt | ECP 只支持 sp |
| `unsupported feature: solvent '...' is not implemented` | 4 | `solvent` 不是 `vacuum` | 改成 `vacuum` |
| `env.json is missing: ... (result is not reproducible)` | 3 | 结果目录被删了文件 | 重跑；不要手工搬文件 |
| `cargo` 报 `link.exe` 1104 / 打不开 `.exe` | — | 上一次的测试进程还在跑（Windows 文件锁） | `Get-Process | ? { $_.ProcessName -match 't5_ecp|t06_' } | Stop-Process -Force` |
| Python `No module named qwb_engine` | — | wheel 没装 | 按 §2.3 构建并安装 |
| Python 侧结果与 Rust 不一致 | — | 装了旧 wheel | 重建 wheel（§2.3），这是最常见的"改了没生效" |
| `No module named qwb_studio` | — | 没设 `PYTHONPATH` | `$env:PYTHONPATH='<仓库根>\qwb-studio'` |
| GUI 起不来（headless） | — | 没有显示 | `$env:QT_QPA_PLATFORM='offscreen'` |
| GUI 里作业很慢 | — | 用的是 debug 版 `qwb.exe` | 设 `QWB_ENGINE_BIN` 指向 release |
| `check_deps_centralized.sh` 报空行 | 1 | 仓库开了 `core.autocrlf=true`，文件是 CRLF | 已在脚本里处理；若自己加脚本请剥 `\r` |

### 12.2 怎么知道"算得对不对"

按可信度从高到低：

1. **有外部基准的体系**（STO-3G 的 H2/H2O/CO/CH4/NH3/N2、苯的 B3LYP/PBE）→ 直接比
   `tests/reference/cccbdb_*.csv`；
2. **没有外部值的体系**（def2-SVP 的能量、ECP 体系）→ 用三条与参照无关的不变量：
   `Tr(P·S) = 电子数`、偶极与实验比（同量级）、网格等级加密后能量差（< 0.1 mHa）；
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

仍然不收敛时你能调的东西（`input.yaml` 里）：`scf_max_iter`（这是上限，退路不会超过它）、
`diis_history`（近简并体系调到 16–24 有用）。**注意**：退火"收敛"不等于"算对"——
苯/STO-3G/PBE 的退火会停在比真值高 19 Ha 的伪态上（这是已知记录，引擎因此会在退火之后
再做一次有界多初猜交叉验证取更低者）。

---

## 13. 常见问题

**Q：为什么算能量要 `--release`？**
A：debug 构建慢 10–50 倍。H2O/def2-SVP/PBE 的单点 release 实测 **7 秒**（debug 构建要十几分钟）。

**Q：`threads` 设了没用？**
A：本构建的积分层是单线程，`--threads` 与 `options.threads` 只记账。
真正的并行在 Hessian 的位移（rayon + 确定性归约）与 `qwb batch` 的进程级并发。

**Q：为什么 `b3lyp` 不能写在 `input.yaml` 里？**
A：K1 的 phase-1 语义门禁拒绝 `requires.laplacian = true`（"no meta-GGA"，冻结范围），
所以数据层没有 `b3lyp.yaml`。它在测试里程序化构造（`t06_benzene.rs`），
用 `pbe0` 才是数据层可用的 hybrid。

**Q：ZnO 为什么算得很慢？**
A：ZnO/LANL2DZ 的单初猜 DIIS 不收敛，引擎要依次试多初猜与退火才能收敛
（release 下整条命令约 4 分钟）。这不是错误路径，而是这个体系本身的已知难点
（闭壳层受限的 O 原子是伪解）。

**Q：结果里的能量能直接和实验比吗？**
A：不能。总能量是模型化学的量，和实验没有可比性；能比的是**能量差**（原子化能、
反应能）与**几何/频率**。本项目的频率判据用的是"标度后的谐振频率 vs 实验基频"。

**Q：怎么把结果发给别人复现？**
A：把整个结果目录（或 `.qwb` 容器）发过去。对方跑 `qwb replay <dir>` 即可；
`env.json` 里记了 rustc 版本、`cargo.lock` 摘要与数据快照摘要，数据被动过会直接报错。

**Q：clone 下来 `qwb-data/basis/` 是空的？**
A：不是坏了，是**故意的**：那 7 个基组/ECP 文件是 BSE 导出的第三方数据，仓库只提交配方
（`qwb-data/sources.json`）与 sha256 钉，避免再分发（§8.4）。跑第一个作业会自动取回，
也可以 `qwb data install` 一次备齐。

**Q：能完全离线用吗？**
A：能。取回过一次之后所有计算都不需要网络；离线机器也可以在有网机器上跑一次
`qwb data install`，把整个 `qwb-data/` 拷过去（摘要钉会校验两边一致）。
`$env:QWB_AUTO_FETCH='0'` 关掉自动取回。

**Q：数据取回失败/摘要不符怎么办？**
A：摘要不符意味着"上游内容或转录规则变了"，脚本会拒绝落盘并打印两个摘要。
先看 `qwb-data/CHANGELOG.md` 与 `qwb-data/sources.json` 的 `transcribe` 段是不是有意改过；
确认要更新就改 `sources.json` 与 `registry.yaml` 两处钉（`qwb data install` 打印的就是新钉）。

**Q：文档在哪？**
A：[`../project.md`](../project.md)（规范与验收判据）、[`status.yaml`](./status.yaml)（进度账本，
每条判据都有实测数字）、[`choices.md`](./choices.md)（决策与踩坑日志，按时间编号）、
[`HANDOVER.md`](./HANDOVER.md)（交接要点）、本文。
