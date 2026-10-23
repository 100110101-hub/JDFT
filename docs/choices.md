# docs/choices.md — 决策日志

> **分工（v2.1 起）**：
> - **冻结的结论**（"这件事按哪个方案做"）→ `project.md` §14 决策台账。
> - **本文件** → (1) 指向已并入 project.md 的历史决策；(2) 仍需人类拍板但**不阻塞**的事项；
>   (3) agent 在每个 WP 里的就地判断（追加式日志）。
>
> 旧的"13 项待人类裁决"清单已全部收敛，见 `project.md` §14；`project.md` §15 记录了
> v2.0 → v2.1 的逐条变更。

---

## 一、已并入 project.md 的决策（不要在这里重复讨论）

C 轨期间做出的契约形式化判断，已全部写入 `project.md`，本文件只保留索引：

| 主题 | 落在哪 | 一句话 |
|---|---|---|
| K1 用 schemars 生成 | §4 K1、§2.4 | Rust 类型是唯一真相源；`schemars` 进白名单 |
| `shells: Vec<ShellBlock>` | §4 K1 | 一个基组可覆盖多元素，平坦 `shells[]` 表达不了 |
| `coefficients: Vec<Vec<f64>>` | §4 K1 | def2-TZVP 的通用收缩需要二维 |
| `notes` 字段 | §4 K1 | `sha256` 是自我引用字段，注释会被 stamp 丢掉，说明得能持久化 |
| 规范化 sha256 规则 | §4 K1、§8.6 | 与排版/注释/键序无关；文件 / registry / snapshot 三处一致 |
| 分子不受 G2 许可门禁 | §4 K1 | 分子是用户输入，没有 DOI/许可 |
| `Registry` = 数据根 + feature 句柄 | §4 K2 | 契约 crate 不解析数据文件 |
| `EngineBackend` + `with_backend` | §4 K2 | 契约 crate 无后端 ⇒ `Engine::new` 恒 `Unsupported` |
| `EngineError::InvalidInput` → 退出码 1 | §4 K2 | 补上 K5 的"输入错误"变体 |
| 矩阵 = list-of-list | §4 K2/K4 | 不依赖 nalgebra 私有 serde 布局 |
| K3 = 6 条转移（含 `queued→cancelled`） | §4 K3、§14 #15 | 取消排队中的任务；重试 = 新 task id |
| `.qwb` 只用 deflate + 字节可复现 | §4 K4 | 内容相同 ⇒ 容器字节相同（支撑 replay 比对） |
| K5 的 4 处增量 | §4 K5 | `batch PATTERN\|DIR`、`data show NAME`、`data verify [PATH...]`、`QWB_DATA_ROOT` |
| 单 git 工作树（三顶层目录） | §14 #2、§12.1 | path 依赖 `engine-api`，不拆仓 |
| toolchain 钉 1.97.1（放仓库根） | §14 #3/#4、§13 | 覆盖两个 workspace |
| `zip` 只开 deflate | §14 #18 | 甩掉 bzip2/lzma/aes 依赖树 |
| `serde_yaml` 保留 + 一条 advisory ignore | §14 #14、R-20 | `unsafe-libyaml` 归档，迁移需人类批 |
| `wildcards = warn`（cargo-deny） | §3、§9 | workspace 继承依赖被误报成 `*`；改用 `check_deps_centralized.sh` 硬门禁 |

---

## 二、仍需人类拍板，但**不阻塞**（有默认值，agent 照默认推进）

| # | 事项 | 当前默认 | 不阻塞的原因 |
|---|---|---|---|
| 1 | 代码许可（所有 crate 的 `license`） | `Apache-2.0` | 已在各 manifest 里；改成别的是一次批量替换 |
| 2 | 数据仓许可（自研数据） | `Apache-2.0` | 同上；样例已按此填写 |
| 3 | 外部数据再分发条款（Stuttgart/def2-ECP、BSE） | **逐条确认后才写 `redistribute_ok: true`** | 只挡 D-07/D-08 的真实数据落盘，不挡引擎/产品轨 |
| 4 | 是否物理拆三仓 | 暂不拆 | 触发条件：数据仓要独立对外发布 |
| 5 | `serde_yaml` 是否迁移到 saphyr/serde_yml | 保留 | 已用 `deny.toml` 显式 ignore 并登记 R-20 |
| 6 | 是否引入经批准的 `unsafe`（热点） | 不允许 | 需要基准数据证明收益后才谈；`scripts/unsafe_allowlist.txt` 为空 |

---

## 三、WP 就地判断日志（agent 追加）

> 规则见 `project.md` §10.3：细节不明确时**选更严格的解释 + 记在这里 + 继续**。
> 只有"改已冻结语义"或"两个 WP 验收冲突"才停。

| 日期 | WP | 判断 | 依据 |
|---|---|---|---|
| 2026-10-22 | C-01 | 样例数据全部加 `example-` 前缀并写 `notes` | 防止被误当 D-03/D-09/D-14 交付物；生产查找不会命中 |
| 2026-10-22 | C-01 | `MoleculeFile` 也带 `sha256`（但不要求） | 让 `qwb-stamp-data` 对所有文件统一工作 |
| 2026-10-22 | C-03 | 当时严格 5 条转移并把缺口上报 | 当时 §10.2 禁止自行扩展；v2.1 已由人类批准扩到 6 条 |
| 2026-10-22 | C-05 | `data list/show` 实装，`run/batch/replay` 返回 4 | 前两者是纯 manifest 读取；后两者需要引擎后端，不能伪造 |
| 2026-10-22 | C-06 | `verify.ps1` 保持 ASCII-only | Windows PowerShell 5.1 用 ANSI 解码无 BOM 脚本，中文会炸（实测） |
| 2026-10-22 | C-06 | mypy 在桩文件目录里跑，不用 MYPYPATH | MYPYPATH 在 Windows/WSL 间是路径转义雷区（实测 WSL 下 wslpath 不可靠） |

---

## 四、怎么用本文件

- **开工前**：只看 `project.md` §14；本文件不需要读，除非你怀疑遇到了模糊处。
- **遇到 §10.3 表里"停"的那两类**：在 §三表格追加一行，写清"为什么必须改"与影响面，
  然后转去做其它可并行的 WP。
- **不要**在这里重复记录 §14 已经定下的事 —— 那会让两个文件打架（v2.0 → v2.1 的教训）。

---

## 九、X-03 的前置：Lebedev 角向网格需要表（待人类确认来源）

X-03 要求 "Becke 分区 + Lebedev 网格（order 194/302/590/974）"。Lebedev 点集**没有闭式
生成法**，是每个阶数一张表（302 阶 = 302 个点 × 3 坐标 + 权重）。目前 D-13 的数据里
只有 `lebedev_order` 这个整数，仓库里没有点数表。

可行的三条路（需人类选一条，它同时是**数据许可**问题，性质同 D-07）：

| 方案 | 说明 | 代价 |
|---|---|---|
| A. 引入 Lebedev 表作为 **qwb-data** 的新数据类型（`grid/lebedev-<order>.yaml`） | 与本项目"数据为中心"一致；需要 k1 schema 新增一类 + 来源与许可（多数表源自 Lebedev 1976 论文或公开表） | K1 增量（新增可选文件类型，非破坏）+ 一次性数据录入 |
| B. 用 **Gauss–Legendre(cosθ) × 均匀 φ** 乘积规则替代 | 数学上合法、Σw = 4π 同样精确、无需外部表；但 `lebedev_order` 字段会变成"角向点数的预算"，与字段名不符 | 与 D-13/K1 的语义有偏差，需要在 §8.4 说明 |
| C. 把 `lebedev_order` 解释为"角向精度目标"，内部选最接近的乘积规则阶数 | 最省事，但把契约字段的解释权交给实现，不建议 | 契约模糊化 |

**当前状态**：X-03 **未开工**（不是失败），等选定方案。其余 X 轨项（X-04/X-05）依赖 X-03，
所以 T1 的 XC 网格这一段因此停在 X-02。

已完成的 X 轨：X-01（`FunctionalTerm` + Slater/VWN5）、X-02（有限差分导数门禁，同时收尾 D-09）。

### 附：R-06 球谐转换的当前进度（2026-10-23）

- **基组层已修**：`InternalShell.angular` 来自数据，`n_functions()` 按球谐 `2l+1` 计数
  （def2-SVP 的 C 现在是 14 个 AO，此前是直角坐标的 15）。
- **积分层未修**：`one_electron` / `two_electron` 的核是直角坐标实现（d = 6），
  所以对 `l >= 2` 显式返回 `Unsupported`。需要在 SCF 之前补
  `qwb-basis` 的 cartesian→spherical 变换矩阵 `T`（每个壳层一块，块对角），
  并在积分产出后做 `M_sph = Tᵀ M_cart T`。T2（def2-SVP）被它阻塞；T1（STO-3G，全 s/p）不受影响。

### R-06 进展（2026-10-23，第二轮）

- **变换矩阵已实现并验证**：`qwb-basis::spherical::{Transform, spherical_transform}`。
  `l<=1` 是单位阵（球谐与直角坐标一一对应）；`l=2` 是 6×5 的实球谐变换，
  测试断言 ① 五列标准正交（`TᵀT = I₅`，1e-15）、② 每列与 `s` 方向 `r²/√3` 正交
  （"球谐 d 里没有 s 成分"）、③ 形状 6→5、④ `l>=3` 返回 `None`（系数不猜）。
- **还差最后一步**：把 `T` 接进积分层（`M_sph = Tᵀ M_cart T`）。`one_electron`
  目前对 `l>=2` 显式 `Unsupported`，所以不会出错值；接上后 T2 的 def2-SVP 才可算。
