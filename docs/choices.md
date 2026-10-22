# docs/choices.md — 已做的选择 + 待人类裁决项

> 本文件记录 **C 轨（C-01..C-06）** 期间做出的判断。凡属 project.md §12.5 列举的
> “agent 不得自行决定”的事项，或对 K1–K6 做了**形式化补充**的地方，都在这里登记。
> §10.1.4：契约不明确时 agent 必须停下并上报 —— 这些条目就是上报内容。

---

## 一、本次已冻结的契约（C 轨产物）

| 契约 | 产物 | 与 project.md 的关系 |
|---|---|---|
| K1 | `qwb-contracts/schema/*.schema.{json,yaml}`（由 Rust 类型生成）+ `qwb-registry` 类型 | §4 K1 + 下方 §2 的形式化 |
| K2 | `qwb-contracts/engine-api`（`qwb-engine-api` 0.1.0） | §4 K2 + 下方 §3 的形式化 |
| K3 | `qwb-contracts/studio/{task,result}.schema.yaml` + `qwb-io::task` | §4 K3，状态机严格 5 条转移 |
| K4 | `qwb-contracts/studio/qwb-format.md` + `qwb-io::container` | §4 K4（Zip，二选一取 list-of-list） |
| K5 | `qwb-core/crates/qwb-cli` + `qwb-contracts/studio/cli.md` | §4 K5 + 下方 §5 的**增量** |
| K6 | `qwb-contracts/studio/python/qwb_engine.pyi` | §4 K6 |

---

## 二、K1 的形式化补充（需要评审）

1. **`schemars` 生成 —— 人类已批准（本轮）**。§2.3 的允许库清单里没有 `schemars`，
   本轮已获批准，代价是把它加进白名单（仍在 `qwb-registry` 的常规依赖里，不是可选 feature）。
   §4 K1 的“二选一”因此定案为 **schemars 生成**，手写 serde struct 方案作废。

2. **`BasisSet.shells` 的元素类型是 `ShellBlock { element, shells }`**，不是平坦的
   `shells[]`。原因：一个基组文件可以覆盖多元素（D-03 的 STO-3G 覆盖 H/C/N/O/F），
   平坦列表无法区分元素。字段名保留 `shells`。

3. **`Shell.coefficients` 是二维 `Vec<Vec<f64>>`（每个内层列表 = 一个收缩函数）**。
   原因：def2-TZVP 的通用收缩（D-05）无法用一维 `coefficients[]` 表达。
   校验强制每个内层列表长度 == `exponents.len()`。

4. **新增可选字段 `notes`**（所有数据类型 + 分子）。用途：记录来源/核对说明。
   动机：`sha256` 是工具生成的自我引用字段，注释会在 `stamp` 时丢失，需要一个
   能持久保存的“这数据是什么状态”的位置（`qwb-data` 的样例就靠它标注“非交付物”）。

5. **`sha256` 的规范化规则**（`qwb-registry::sha256`）：解析 YAML → 删除顶层
   `sha256` → 转 JSON（键按字典序）→ 对紧凑 JSON 取 SHA-256。
   即**与排版/注释/键序无关**，改一个数字必换摘要。
   代价：`stamp` 写回会丢弃 YAML 注释（数据文件按“工具产物”对待）。

6. **分子文件不受 G2 许可门禁约束**。§4 K1 说“共同字段（所有数据）”，但分子是
   **用户输入**（用户自己画的分子没有 DOI/许可）。因此 `MoleculeFile` 只有
   `name/notes/atoms/charge/multiplicity/citation?/created/sha256?`，没有 `license`。
   `qwb-data/examples/molecule/` 是样例；生产分子不入数据仓。

7. **注册表清单（`registry.yaml`）的形状**：`schema_version/name/version/created/features`
   + 5 个按域分组的条目表（`bases/ecps/functionals/grids/solvents`），每条
   `{name, path, sha256}`。**两处 sha256 必须一致**（文件内 vs 清单），
   `data verify` 做交叉校验 —— 任一处被改都会被发现（G2）。

---

## 三、K2 的形式化补充（需要评审）

§4 K2 的签名引用了 `Registry`、`EnvInfo`、`DataSnapshot`、`Molecule`、`GridLevel`、
`JobType`、`TaskOptions`，但没有定义它们。本次定义如下：

1. **`Registry` = “数据根 + 本次计算声明的能力集合”的句柄**，不是数据加载器。
   加载器属 D-01（`qwb-registry`）。理由：契约 crate 不解析文件格式、不引入 IO/格式依赖。
   K6 的 `Engine(registry_path=...)` 与之一致。
2. **新增 `EngineBackend` trait + `Engine::with_backend`**。K2 把 `Engine` 放在契约 crate，
   但数值实现在 `qwb-core`。因此 `Engine` 是门面：`Engine::new` 只校验注册表并
   **必然返回 `Unsupported`**（“本构建未链接引擎后端”），真正的入口是
   `Engine::with_backend(registry, backend)`（由顶层 `qwb-engine` 使用）。
   这正好满足 C-02 的验收“`Engine::new` 可返回 `Unsupported` 错”，且不伪造能力。
3. **新增错误变体 `EngineError::InvalidInput(String)` → 退出码 1**。
   K5 的退出码 1（输入错误）在 K2 里没有对应变体。K2 冻结规则允许“只新增变体”，
   故新增该变体并在 `EngineError::exit_code()` 里钉死映射（单测覆盖）。
4. **`SUPPORTED_FEATURES` 词汇表**：`"<域>:<值>"` 字符串（如 `jobtype:opt`、
   `functional:hybrid`、`ecp`）。注册表声明了表外 feature（如 `functional:meta-gga`、
   `solvent:smd`）→ `Engine::new` 返回 `Unsupported`。这是 §8.5 要求的
   “loader 不报错、`Engine::run` 报 Unsupported” 的落地点。
   注意：“声明支持 ≠ 已实现”，后端会再判一次。
5. **矩阵 JSON 表示 = list-of-list**（K4 的二选一定案）。由 `qwb-engine-api` 的
   `opt_matrix_json` / `opt_vector_json` 显式实现，**不依赖 nalgebra 的私有 serde 布局**。
   `None` 字段不写（省略而非 `null`）。
6. `JobResult`/`JobInput`/`EnvInfo` 等 `#[non_exhaustive]` 结构体对外不可 struct literal，
   因此提供 `new(...)` 构造函数与 `with_*` 建造方法（否则后端无法构造结果）。

---

## 四、K3 / K4 的说明

1. **状态机严格实现 K3 图上的 5 条转移**，`TaskState::LEGAL` 是唯一真相源，
   `task.schema.yaml` 的 `x-state-machine` 与之由测试强制一致。
2. **已知缺口（需要人类裁决，agent 未擅自扩展）**：
   - `queued → cancelled`：取消一个尚未开始的任务。P-03 的“取消即停”很可能需要它；
   - `failed → queued`：重试语义。
   这两条都不在 §7 的冻结图上，因此**当前实现拒绝**，等人类决策后修 K3 + CHANGELOG。
3. **`.qwb` 容器字节可复现**：条目时间戳固定 1980-01-01、Deflate、条目顺序固定、
   `meta.json` 第一位。相同内容导出逐字节相同（单测断言），直接支撑 G5 的 replay 比对。
4. **`zip` 只用 `deflate` feature**（关闭 bzip2/lzma/aes/zstd 默认特性）。
   理由：K4 只需要 Deflate；默认特性会引入 `libbz2-rs-sys`、`lzma-rust2`、以及
   第二份 `sha2`/`digest`（重复版本 + 新许可）。裁剪后 `cargo deny` 四类全绿。
5. **`data_snapshot.json` 用“规范化内容 sha256”**，与文件内 `sha256` 字段、
   `registry.yaml` 条目三者一致（可交叉校验）。没有采用“原始文件字节 sha256”，
   因为那会与另两处的摘要不一致，制造两套 hash 语义。

---

## 五、K5 的增量（不破坏既有用法，待确认）

K5 只写了子命令名。实装时的增量：

1. `qwb batch` 位置参数同时接受字面模式 `DIR/*.yaml` 与裸目录 `DIR`
   （Windows shell 不展开 glob）。
2. `qwb data show` 需要一个名字参数。
3. `qwb data verify` 接受可选文件列表（D-02 的“3 坏/3 好”验收需要）。
4. 数据根通过 `QWB_DATA_ROOT` 环境变量或向上搜索 `qwb-data/` 定位，**不新增命令行参数**。

C 轨各子命令的真实能力（哪些故意失败、为什么）见 `qwb-contracts/studio/cli.md`。

---

## 六、仓库与工具链

1. **单 git 工作树，三个顶层目录（`qwb-contracts/ qwb-data/ qwb-core/`）**，
   而不是 §12.1 的“三仓”。理由：C 轨需要 `qwb-core` 以 **path 依赖**引用
   `qwb-contracts/engine-api`；物理拆仓会让 CI 需要额外 checkout/publish 流程。
   拆不拆是**人类裁决项**（拆的话 K2 需要发布或 git 依赖策略）。
2. **`rust-toolchain.toml` 放在仓库根**（不是 §12.1 的 `qwb-core/`），
   因为它要同时覆盖 `qwb-core` 与 `qwb-contracts/engine-api` 两个 workspace。
3. **toolchain 钉 `1.97.1`**：本机实际安装版本。§12.5 的“MSRV 冻结值”仍需人类确认。
4. **代码许可填 `Apache-2.0`**（所有 crate 的 `license` 字段）。
   这是占位值：代码许可不在 §12.5 的清单里，但必须写点东西；**请人类确认或改**。
5. **`serde_yaml` 接受 RUSTSEC-2024-0449**（其实现 `unsafe-libyaml` 已归档）。
   §2.3 冻结了 `serde_yaml`，迁移（`serde_yml` / `saphyr` / 自带解析）是**人类裁决项**。
   `deny.toml` 里只有这一条 ignore，其余 advisory 一律拒绝。
6. **`cargo-deny` 的 `wildcards` 只能是 `warn`**：workspace 继承依赖（`foo.workspace = true`）
   会被误报成 `*`。因此“不许随手加依赖”改成可检查的流程约束：
   三方依赖只允许出现在 `[workspace.dependencies]`，由 `scripts/check_deps_centralized.sh` 校验。

---

## 七、`qwb-data` 现状（重要，避免误用）

`qwb-data/examples/` 下 6 个文件是 **K1 schema 样例（SPECIMEN）**，不是生产数据：

| 文件 | 状态 |
|---|---|
| `basis/example-sto-3g-H.yaml` | 只有 H；D-03 要求 H/C/N/O/F + BSE 逐项 diff |
| `ecp/example-ecp-cu.yaml` | **系数是合成值**，只为让 schema 有东西可校验；真实数据属 D-07 |
| `functional/example-svwn5.yaml` | `parameters` 故意留空；VWN5 常数属 D-09 |
| `grid/example-grid-normal.yaml` | 点数/锐度为常见取值；收敛门禁属 D-13 |
| `solvent/example-water.yaml` | ε = 78.3553 @ 298.15 K 是 §8.5 的参考值；半径是占位值 |
| `molecule/example-h2o.yaml` | 几何样例（用户输入类，不受 G2 门禁） |

保护措施：文件名与 `name` 字段都带 **`example-` 前缀**，所以生产查找
（`basis_name: sto-3g`）**不会**命中样例；每个文件的 `notes` 写明自己是样例。
D-03..D-14 交付真实数据时用正式名字（`sto-3g` 等），不会与样例冲突。

---

## 八、待人类裁决清单（汇总）

| # | 事项 | 当前状态 | 影响 |
|---|---|---|---|
| 1 | 代码仓库许可（`Apache-2.0` 是占位） | 待定 | 分发前必须确定 |
| 2 | 三仓 vs 单仓（§12.1） | 单仓 | 拆仓需定 K2 的发布/依赖策略 |
| 3 | MSRV / toolchain 冻结值 | 1.97.1 | R-19 |
| 4 | `schemars` 进 §2.3 白名单 | 本轮已批准，需写进文档 | C-01 前提 |
| 5 | `serde_yaml` 是否迁移（unmaintained） | 保持，ignore 一条 advisory | G6/R-17 |
| 6 | K3 补 `queued→cancelled`、`failed→queued` | **未擅自扩展** | 阻塞 P-03 的取消语义 |
| 7 | `pyo3` abi3 版本（py39 还是 py310） | 桩按 `abi3-py39` 写 | R-12 |
| 8 | `qwb-linalg` 默认后端 | 未开工（I-00） | R-11 |
| 9 | R-03 数值方案（Obara-Saika vs Rys） | 未开工（I-03） | R-03 |
| 10 | 是否引入 `ndarray-npy` | 未引入 | R-14 |
| 11 | 首版泛函清单 / 溶剂是否第一阶段实现 / ECP 范围 | 未开工 | §12.5 |
| 12 | 数据仓库开源许可选择 | 样例用 `Apache-2.0` 占位 | D-15 |
| 13 | 参考实现（libxc 等）在 CI 的边界 | 未使用 | R-10 |
