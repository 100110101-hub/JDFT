# DATA_LICENSES

> 正式版本由 `scripts/gen_data_licenses.py` 生成（D-15）。在此之前，本文件是
> **手写的最小版本**，只覆盖 `qwb-data/examples/` 下的 K1 样例。

数据许可门禁规则（§8.6 / G2）：

1. 每条数据必须带 `license.spdx`、`commercial_ok`、`redistribute_ok`。
2. 每条数据必须带 `citation`（`title` + `doi`/`url`）。
3. 每条数据必须带内容 `sha256`（工具生成，见 `qwb-stamp-data`）。
4. `qwb-core/deny.toml` 的 `[licenses].allow` 是依赖许可白名单；
   数据许可由本文件的映射人工评审。
5. 缺任一项 → `qwb data verify` 退出码 3 → CI 红。

## 生产数据（qwb-data 0.2.0）

| 数据 | 文件 | SPDX | 商用 | 再分发 | 来源与许可依据 |
|---|---|---|---|---|---|
| sto-3g（H,C,N,O,F） | `basis/sto-3g.yaml` | Apache-2.0 | yes | yes | 数值是 1969 年论文发表**的科学常数**（Hehre/Stewart/Pople, doi:10.1063/1.1672392），不构成受版权保护的表达；本文件是我们自己的 K1 结构 + 机器转录，故按 Apache-2.0 授权。BSE 只作导出通道（§8.1 允许），ground truth 存于 `qwb-core/crates/qwb-registry/tests/fixtures/bse/sto-3g.json`。 |

| svwn5 | `functional/svwn5.yaml` | Apache-2.0 | yes | yes | 参数是 1980 年发表的**科学常数**（Vosko/Wilk/Nusair, doi:10.1139/p80-159）与精确常数 2/3、−1/(6π)；本文件是我们自己的 K1 结构，按 Apache-2.0 授权。 |
| grid-coarse / normal / fine / veryfine | `grid/grid-*.yaml` | Apache-2.0 | yes | yes | 网格参数取自 project.md §6.2 冻结值与 Becke 1988 / Lebedev 的公开方法（doi:10.1063/1.454033）；本文件是我们的数据，按 Apache-2.0 授权。 |

## 当前样例（`examples/`，**非生产数据**，不进 registry.yaml）

| 数据 | 文件 | SPDX | 商用 | 再分发 | 备注 |
|---|---|---|---|---|---|
| example-sto-3g-H | `examples/basis/example-sto-3g-H.yaml` | Apache-2.0 | yes | yes | 仅 H；D-03 交付真实 STO-3G |
| example-ecp-cu | `examples/ecp/example-ecp-cu.yaml` | Apache-2.0 | yes | yes | **系数合成**；D-07 交付真实 LANL2DZ ECP |
| example-svwn5 | `examples/functional/example-svwn5.yaml` | Apache-2.0 | yes | yes | 参数留空；D-09 交付 |
| example-grid-normal | `examples/grid/example-grid-normal.yaml` | Apache-2.0 | yes | yes | D-13 交付真实等级表 |
| example-water | `examples/solvent/example-water.yaml` | Apache-2.0 | yes | yes | ε 来自 NIST；半径占位，D-14 交付 |
| example-h2o | `examples/molecule/example-h2o.yaml` | —（用户输入） | — | — | 分子不受 G2 门禁 |

## 待人类决策

- **数据仓的开源许可选择**（§12.5）。目前所有样例填 `Apache-2.0` 占位。
- 真实数据（尤其 Stuttgart/def2-ECP 表格）的再分发条款需要逐项确认后
  才能把 `redistribute_ok: true` 写进去 —— **这是 D-07/D-08 的前置条件**。
