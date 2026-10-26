# `qwb-data/` —— 数据仓库（K1）

这里是引擎唯一的**数据真相源**：基组、ECP、泛函、网格、溶剂。索引是 `registry.yaml`
（K1 `registry.schema.yaml`），每条数据带 `name` / `version` / `citation` / `license` /
内容摘要 `sha256`。`qwb data verify` 会把索引、文件内声明、内容摘要三处对齐检查。

```
qwb-data/
├── registry.yaml      # 清单：条目名、路径、sha256（K1）
├── sources.json       # 第三方数据的**取数配方**（下载/转录/摘要钉），见下
├── CHANGELOG.md       # 数据版本历史
├── DATA_LICENSES.md   # 仓库根的许可汇总（由 scripts/gen_data_licenses.py 生成）
├── basis/             # 基组（部分文件不在仓库里，按需取回）
├── ecp/               # 赝势（部分文件不在仓库里，按需取回）
├── functional/        # 泛函定义（自研，随仓库）
├── grid/              # 网格等级定义（自研，随仓库）
├── solvent/           # 溶剂定义（随仓库）
└── examples/          # SPECIMEN：形状示例，不在 manifest 里
```

## 第三方数据不在仓库里（按需取回）

`basis/sto-3g.yaml`、`basis/def2-svp.yaml`、`basis/def2-tzvp.yaml`、`basis/6-311g-dp.yaml`、
`basis/lanl2dz.yaml`、`ecp/lanl2dz-cu.yaml`、`ecp/lanl2dz-zn.yaml` 是从
[Basis Set Exchange](https://www.basissetexchange.org/) 导出的第三方数据，
**仓库不提交它们**：LANL2DZ 的 ECP 原始论文（Hay–Wadt）没有给出明确的再分发条款
（`redistribute_ok: false`，project.md §14 #20），把文件放进公开仓库等于再分发。

仓库提交的是**配方** `sources.json`：

| 字段 | 含义 |
| --- | --- |
| `path` | 相对本目录的路径（与 `registry.yaml` 的条目一致） |
| `kind` | `basis` / `ecp` |
| `fetch` | BSE 的 `basis` 名与元素（原子序数）；下载走 `api/basis/<basis>/format/json/` |
| `transcribe` | 转录元数据：`name` / `version` / `notes` / `citations` / `license` / `created` / `angular` / `normalization` / `role`（ECP 用 `element`） |
| `sha256` | 内容摘要钉 —— 与 `registry.yaml` 里同一条**必须相等** |

取回的三条路（语义完全相同，都是同一个实现 `qwb-studio/qwb_studio/datafetch.py`）：

```powershell
qwb data install                      # Rust CLI（run/batch 会自动先试一次）
python scripts/fetch_data.py --all    # 独立脚本，不需要 PYTHONPATH
python -m qwb_studio.datafetch --all  # 模块入口
```

GUI 也有入口：**Data 页 → 「下载缺失」**（或菜单「数据 → 下载缺失」）。

流程是「取回 → 转录 → 打戳 → 核对钉 → 原子替换」，摘要对不上就报错且**不落盘**。
因此"取回的文件"与"当初提交的文件"内容等价，`registry.yaml` 的钉、结果目录里的
`data_snapshot.json`、以及历史验收数据全都仍然有效。

## 离线 / 无网络

* 已经取回过一次之后，`~/.cache` 之外的任何操作都不需要网络（文件就在本地）。
* 完全离线的机器：在有网络的机器上跑一次 `qwb data install`，把整个 `qwb-data/`
  目录拷过去即可（`sources.json` 的摘要钉保证两边一致）。
* `QWB_AUTO_FETCH=0` 关闭 `run/batch` 的自动取回；此时缺数据的作业会以 K5 退出码
  `3`（数据校验失败）报错，并在消息里给出取数命令。

## 维护：新增一条第三方数据

1. 在 `sources.json` 里加一条（`fetch` + `transcribe`，`sha256` 先随便填 64 个 0）；
2. 跑 `python scripts/fetch_data.py --need basis/<新文件>.yaml`；
   它会打印实际的摘要，并因为与钉不符而拒绝落盘 —— 把打印的摘要抄进 `sources.json`；
3. 在 `registry.yaml` 的对应域里加条目（`name` / `path` / `sha256`），两处钉必须一致；
4. 重跑 `qwb data verify` 与 `python scripts/gen_data_licenses.py`；
5. 提交：`sources.json` + `registry.yaml` + `DATA_LICENSES.md`（**不要**提交数据文件本身）。

自研数据（泛函/网格/溶剂）不是第三方数据：直接提交文件，`sources.json` 里没有它们，
所以缺失就是**硬错误**（数据根被裁剪过），不会被当成"还没下载"。

## 单位与约定（§12.2）

* 长度一律 **Bohr**（基组/ECP 指数、网格 cutoff）；分子坐标例外，用 Å（字段名带 `_angstrom`）。
* ECP 的 `r_power_convention` 必须显式声明：BSE/EMSL 的 `r_exponents` 是 `r^n`，
  Stuttgart 原始表是 `r^(n-2)`，差两个幂次。
* ECP 的局域通道 = 导出里**最大的 `l`**，其余 `l` 是投影算子。
* `sha256` 只由 Rust 打戳（`qwb data sign` = `qwb-stamp-data`）：规则是
  「YAML 去掉顶层 `sha256` 键 → JSON（键排序）→ SHA-256」，Python 侧不复刻。
