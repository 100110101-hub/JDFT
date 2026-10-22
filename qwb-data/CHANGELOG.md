# qwb-data CHANGELOG

数据仓独立 semver（§8.6）。**任何数据改动都必须 bump version 并在此登记**，
否则 CI 拒绝（§8.6「变更必须 bump version」）。

## 0.2.0 — 2026-10-22

### 新增：生产数据 `basis/sto-3g.yaml`（D-03）
- H、C、N、O、F 五个元素的 STO-3G，**由 BSE 导出经
  `scripts/bse_to_qwb_data.py` 机器转录**（不手抄）；BSE 原始导出存为
  `qwb-core/crates/qwb-registry/tests/fixtures/bse/sto-3g.json` 作为 ground truth。
- 校验：`cargo test -p qwb-registry --test d03_sto3g_bse` —— 逐元素逐系数与 BSE
  diff < 1e-12；每元素基函数数 H=1、C/N/O/F=5；`angular: spherical`；
  生产名 `sto-3g` 可由 `Registry::get_basis` 取到。
- 引用：Hehre, Stewart & Pople, *J. Chem. Phys.* **51**, 2657 (1969),
  doi:10.1063/1.1672392。
- `registry.yaml` 版本 bump 0.1.0 → 0.2.0（§8.6：数据变更必须 bump）。

## 0.1.0 — 2026-10-22

### 新增：K1 schema 样例（`examples/`，**非生产数据**）
- `examples/basis/example-sto-3g-H.yaml` — 仅 H 的 STO-3G 样例
- `examples/ecp/example-ecp-cu.yaml` — Cu ECP 样例（**系数为合成值**）
- `examples/functional/example-svwn5.yaml` — SVWN5 样例（`parameters` 为空）
- `examples/grid/example-grid-normal.yaml` — normal 级网格样例
- `examples/solvent/example-water.yaml` — 水样例（ε=78.3553 @ 298.15 K）
- `examples/molecule/example-h2o.yaml` — 分子文件样例（不受许可门禁）

所有样例的 `name` 都带 `example-` 前缀，因此**不会**被生产的
`basis_name: sto-3g` / `method_name: svwn5` 之类查找命中。

### 新增
- `registry.yaml`：索引上述 5 个数据样例（`schema_version: 1.0.0`）。
- `DATA_LICENSES.md`：当前样例的许可映射（正式生成器属 D-15）。

### 未开始（勿当真数据使用）
D-03..D-14 的真实基组/ECP/泛函/网格/溶剂数据**尚未存在**。参考 `docs/status.yaml`。
