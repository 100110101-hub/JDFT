# K5 · CLI 契约（冻结）

实现：`qwb-core/crates/qwb-cli`（`clap` derive），二进制名 `qwb`。

## 签名

```
qwb run    input.yaml [--out DIR] [--threads N]
qwb batch  DIR/*.yaml [--parallel N]
qwb inspect DIR
qwb replay DIR [--dry-run]
qwb data list|show|verify|sign|install
qwb props  input.yaml [--out DIR] [--states N]     # 阶段二加性（KV-05）
qwb ts     input.yaml [--out DIR]                 # 阶段二加性（KV-05）
qwb irc    input.yaml [--out DIR]                 # 阶段二加性（KV-05）
qwb scan   input.yaml [--out DIR]                 # 阶段二加性（KV-05）
```

## 退出码（冻结）

| 码 | 含义 | 来源 |
|---|---|---|
| `0` | 成功 | — |
| `1` | 输入错误 | 参数/`input.yaml` 解析或语义非法（`EngineError::InvalidInput`） |
| `2` | 收敛失败 | `EngineError::ScfNotConverged` |
| `3` | 数据校验失败 | `EngineError::Data`、容器/布局错误、`env.json` 缺失 |
| `4` | 内部错误 | `EngineError::Unsupported` / `Internal` / `Io`，以及“本构建不含引擎后端” |

映射的实现是 `qwb_engine_api::EngineError::exit_code()`（单元测试钉死）。

## C 轨（契约阶段）各子命令的真实能力

C 轨只做契约与结构，数值执行属 I/X/S/G 轨 + `P-07`。因此下面这些**故意不成功**，
并且会说明原因，不会伪造结果：

| 子命令 | C 轨行为 |
|---|---|
| `run` | 完整解析并校验 `input.yaml`（含打印摘要），然后退出 `4`：无引擎后端 |
| `batch` | 展开并逐个校验输入，然后退出 `4`：无引擎后端；任一输入非法先退出 `1` |
| `inspect <dir>` | **完全实装**：必需文件 + `env.json` 完整性与 G5 检查 |
| `inspect <.qwb>` | **完全实装**：K4 容器结构校验 |
| `replay --dry-run` | **完全实装**：读 `env.json` 打印复现记录 |
| `replay`（无 dry-run） | 退出 `4`：重算属 `P-08` |
| `data list` | 完全实装（读 `registry.yaml`） |
| `data show <name>` | 完全实装（含条目文件校验） |
| `data verify [PATH...]` | 完全实装（无参数时校验整个注册表） |

## 阶段二加性子命令（KV-05，2026-11-06）

`props` / `ts` / `irc` / `scan` 是 `project-phase2.md` §4.1 里 K5 的**加性**扩展：
签名先落（含 `--help`），实现随里程碑跟上。当前行为**明确且一致**：

| 子命令 | 现状 | 归属 |
|---|---|---|
| `props` | 退出码 `4`：`declared in K5 ... not implemented yet` | M4（RS-02）起，M7 收口 |
| `ts` | 同上 | M6（RP-01） |
| `irc` | 同上 | M6（RP-02） |
| `scan` | 同上 | M6（RP-03） |

两条边界（都有单测）：

1. **输入文件不存在 → 退出码 `1`**（输入错误），不会被"功能没实现"盖掉 ——
   否则用户会以为功能存在、只是坏了；
2. 既有子命令的签名与退出码**一个都没变**（K5 冻结表的 0/1/2/3/4 映射不变）。

## 对冻结签名的**增量**（不破坏既有用法，待人类确认）

1. `qwb batch` 的位置参数在 Windows 上不会被 shell 展开，因此实现同时接受字面模式
   `DIR/*.yaml` 与裸目录 `DIR`。
2. `qwb data show` 需要一个名字参数（K5 只写了 `show`）。
3. `qwb data verify` 接受可选文件列表（D-02 的“3 个坏文件/3 个好文件”验收需要）。
4. 数据根目录通过 `QWB_DATA_ROOT` 或向上搜索 `qwb-data/` 定位，**不新增命令行参数**。
