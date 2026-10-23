# qwb-data CHANGELOG

数据仓独立 semver（§8.6）。**任何数据改动都必须 bump version 并在此登记**，
否则 CI 拒绝（§8.6「变更必须 bump version」）。

## 0.4.0 — 2026-10-23

### 新增：`basis/def2-svp.yaml`（D-04，H/C/N/O）
- 由 `scripts/bse_to_qwb_data.py` 从 BSE 导出机器转录（revision "Data from Turbomole 7.3"），
  原始导出存为 `qwb-core/crates/qwb-registry/tests/fixtures/bse/def2-svp.json`。
- **`normalization: none`**：def2 的收缩系数没有重新归一化（实测 C 1s 的 ⟨f|f⟩ = 0.978，
  归一化原函数约定下），这是 Turbomole 的惯例；加载器按 §8.1 在加载时归一一次。
  STO-3G 仍是 `l2`（⟨f|f⟩ = 1 ± 1e-10）——两者不同是数据事实，不是笔误。
- 基函数数（球谐）：H = 5（2s1p），C/N/O = 14（3s2p1d）。
- 引用：Weigend & Ahlrichs, *Phys. Chem. Chem. Phys.* **7**, 3297 (2005), doi:10.1039/b508541a。
- `registry.yaml` 0.3.0 → 0.4.0。
## 0.3.0 — 2026-10-22

### 新增：生产数据 `functional/svwn5.yaml`（D-09）
- SVWN5 = Slater 交换（`LdaSlaterX`，α = 2/3）+ VWN5 相关（`LdaSvwn5`）。
- **parameterization V 是数值确认的**，不是凭记忆：VWN5 常数在 rs=1..20 复现
  Ceperley-Alder 顺磁相关能到 ~1e-3 Ha；而常被误标为 "VWN5" 的 VWN3/RPA 常数
  （x0=-0.10498, b=3.72744, c=12.9352）偏差 ~1.7e-2 Ha。负对照写进测试。
- 13 个参数逐个带 `parameter_sources`（K1 新增可选字段，见 qwb-contracts/CHANGELOG）。
- 有限差分导数门禁（X-02，< 1e-9）属于 X 轨，D-09 的 status 里注明未覆盖。

### 新增：网格等级 `grid/{coarse,normal,fine,veryfine}.yaml`（D-13）
- 冻结初始参数 `(50,194,0.6)`、`(75,302,0.6)`、`(99,590,0.6)`、`(140,974,0.8)`，
  `cutoff = 20 Bohr`；`radii_scale` 故意留空（原子半径表在代码侧，X-03 需要缩放时再数据化）。
- "相邻等级能量差 < 0.1 mHa" 属 T2 断言（需引擎），未在 D-13 内伪造。

### 变更：`registry.yaml` 只索引生产数据
- `examples/` 下的 K1 样例**移出**清单：它们保留为 D-02 的回归夹具（按路径校验），
  但不再进引擎加载路径 —— 否则同一个 grid 等级会有两个定义，`get_grid_by_level`
  出现歧义（现在它也会把重复等级报成 `Data` 错误）。
- 版本 bump 0.2.0 → 0.3.0。

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
