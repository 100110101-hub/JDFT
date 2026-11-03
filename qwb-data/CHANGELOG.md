# qwb-data CHANGELOG

数据仓独立 semver（§8.6）。**任何数据改动都必须 bump version 并在此登记**，
否则 CI 拒绝（§8.6「变更必须 bump version」）。

## 0.20.0 — 2026-11-14

### 扩展：`basis/sto-3g.yaml` 与 `basis/def2-svp.yaml` 加 **Si**（RS-04 的 TMS 参考物）

RS-04 §6 的第三条判据（¹H/¹³C 化学位移 vs 实验）要用 TMS（Si(CH3)4, 13 个原子）作参考物，
而在此之前 `qwb-data/basis/` 里**没有任何含 Si 的基组**（sto-3g = H/C/N/O/F、
def2-svp = H/C/N/O、6-311G(d,p) = H/C/N/O、def2-TZVP = H/C/N/O/F、LANL2DZ = H/O/Cu/Zn）——
也就是说这条判据在数据层根本量不了。这一版把 Si 补进两个最常用的基组：

- `sto-3g`：`fetch.elements` 加 14，Si = 1s/2s/3s + 2p/3p，**9 个基函数**（总数 21 → 30）；
- `def2-svp`：`fetch.elements` 加 14，Si = [4s3p1d]，**18 个基函数**（总数 47 → 65）；
- 两个文件的 H/C/N/O(/F) 字节**没有变**：从 BSE 重新导出、用同一份 `bse_data.transcribe`
  转录，逐行 diff 只有 `notes` 里那句元素清单和新增的 Si 块（`fixtures/bse/*.json` 同样
  只多出元素 14，既有元素的 JSON 逐位相同）。

`registry.yaml` 0.19.0 → **0.20.0**（两条 sha256 重新钉：sto-3g
`fcaa652c4b158b450e3d851686ad7ab8df4b3414b06c03d18a7bd99e745eb7d4`、def2-svp
`f7f27a38738bb1cf9e21660f2d2453c349295f1e7c66b214edb96686909287df`）；`sources.json`
的 `elements`/`notes`/`sha256` 同步；`DATA_LICENSES.md` 重新生成。

**验证**：`qwb data verify` 34/34 通过；`cargo test -p qwb-registry`（d01/d03/d04）绿 ——
d03/d04 仍然把整份数据与 BSE 导出逐项 diff 到 1e-12（现在包含 Si），并钉住 Si 的基函数数
（STO-3G 9 / def2-SVP 18）与壳层组成。

## 0.13.0 — 2026-11-09

### 新增：`dispersion/d3zero.yaml`（DA-06 的零阻尼那一半）

M06-L 与 M06-2X 在上游**没有 BJ 参数**，只有零阻尼（D3(0)）：`rs6 = 1.581 / 1.619`、
`s8 = 0`。这份数据把零阻尼补齐，元素范围与参考表与 `d3bj.yaml` 完全一致（同一个生成器
从同一个解析结果写出来，参考 C6/CN/r4r2 逐位相同）：

- **零阻尼参数**：6 个泛函的 `s6/s8/rs6/rs8/alp`（b3lyp / pbe / pbe0 / hf / m06-l / m06-2x）；
- **成对 vdW 半径表**（666 个元素对，Å）：上游的 `R0_ij` 是**查表值**，不是两原子半径相加
  —— `get_vdw_rad_pair_num` 直接读 `vdwrad(i + j(j-1)/2)`，只有 `get_vdw_rad_num(i)` 才是
  `0.5*vdwrad(i,i)`；
- **阻尼函数**（`damping/zero.f90`）：`t6 = (rs6·R0/r)^alp`、`t8 = (rs8·R0/r)^(alp+2)`、
  `f = 1/(1+6·t)`、`E = −(s6·C6·f6/r⁶ + s8·C8·f8/r⁸)`。

`registry.yaml` 0.12.0 → 0.13.0（新增 `d3zero` 条目）；`dispersion.schema.*` 增加
`zero_damping_parameters` 与 `vdw_radii_angstrom`（K1 加性扩展）；`DATA_LICENSES.md`
重新生成（31 → 32 个文件）。

**验证**：`cargo test -p qwb-scf --test fx04_dispersion_zero` 3 例绿 —— 40 个 (二聚体, 泛函)
组合与官方 `ZeroDampingParam` 的最大偏差 **1.302e-6 kcal/mol**、梯度 **2.755e-10 Ha/Bohr**；
解析梯度 vs 中心差分 **1.29e-13 Ha/Bohr**；引擎侧 `m06-2x + data: d3zero` 能算（用 `d3bj`
仍显式报错）。门禁新增 `DA-06: zero-damping data reproduces s-dftd3` 一步。

## 0.12.0 — 2026-11-09

### 新增：`dispersion/d3bj.yaml`（DA-06）

D3(BJ) 色散校正的整套参数，元素范围 H–Kr（36 个元素、666 个元素对）：

- **参考 C6 表**：每个元素对 7×7 = 49 个值（Ha·Bohr⁶），按 Fortran 展平顺序保存，
  块内**第一维是高 Z 元素**、第二维是低 Z 元素。这个取向是拿官方实现反解出来的
  （两原子 O-H 在 CN≈0 处 C6 = 10.0951；转置读法给 6.14，差 40%），
  同核元素对在转置下不变，所以只有异核对才能发现搞反了。
- **参考 CN**：每个元素 1–7 个参考点（H 2、O 3、Kr 1 …），与上表的参考点索引一一对应。
- **CN 参数**：`kcn = 16`、`scale = 4/3`（D3 用 Pyykko 共价半径的 4/3 倍）、半径存
  **Å 原值**（上游表把金属降过 10%），Å→Bohr 用 K2 的 `BOHR_PER_ANGSTROM`。
- **r4r2**：`r4r2` 是**派生值** `sqrt(0.5·published·sqrt(Z))`（模型消费的就是它），
  `r4r2_published` 是同一文件里公布的 `<r⁴>/<r²>`。用错哪一个都会让 C8/C6 差 2–16 倍。
- **阻尼参数**：`b3lyp` / `pbe` / `pbe0` / `hf` 四组 `s6/s8/a1/a2` + DOI。
  M06-L 与 M06-2X **没有** BJ 参数（上游 `get_rational_damping` 无此分支，只有零阻尼），
  所以不在这份数据里 —— 不是遗漏。
- **溯源**：`provenance.upstream` 逐个列出上游文件（URL + sha256），
  生成器是 `scripts/gen_d3_params.py`（`--check` 逐项比较，容差 0）。

`registry.yaml` 0.11.1 → 0.12.0（新增 `dispersions:` 段与 `dispersion:d3bj` feature）；
`qwb-contracts/schema/dispersion.schema.{json,yaml}` 新增（K1 加性扩展，由
`qwb-gen-schemas` 从 Rust 类型生成）；`DATA_LICENSES.md` 重新生成（30 → 31 个文件，
许可 LGPL-3.0-or-later —— 数据转写自 s-dftd3 的源码表）。

**验证**：`qwb data verify` 31 passed；`cargo test -p qwb-registry --test da06_dispersion`
4 例绿（666 个元素对逐一断言）；`scripts/validate_d3_data.py` 复现官方 s-dftd3 的
能量与解析梯度，最大偏差 8.559e-10 Ha = 5.371e-7 kcal/mol。

## 0.8.0 — 2026-10-27

### 新增：三个 RI 辅助基（DA-07 / IN-01 / SC-05）
- `basis/def2-svp-ri.yaml`（H C N O，54 个壳层，max l = 3）、
  `basis/def2-tzvp-ri.yaml`（H C N O F，95 个壳层，max l = 4）、
  `basis/def2-universal-jkfit.yaml`（H C N O F，106 个壳层，max l = 4）。
  三者分别对应 BSE 的 `def2-svp-rifit` / `def2-tzvp-rifit` / `def2-universal-jkfit`
  （revision 'Data from Turbomole 7.3'），`role` 是 `ri-j` / `ri-j` / `ri-jk`
  —— **不是轨道基组**，不能写进 `basis_name`。
- `normalization: none`（与其余 def2 数据一致：Turbomole 的收缩系数不重新归一化，
  由 loader 在加载时归一化一次）。
- **能力边界（如实记账）**：`def2-tzvp-ri` 与 `def2-universal-jkfit` 含 g 函数（l = 4），
  而球谐变换目前只到 l = 3，所以它们能通过 `data verify`、但被用作辅助基时会以
  `Unsupported` 明确报错。SC-05 只用 `def2-svp-ri`。
- `registry.yaml` 0.7.0 → 0.8.0；`DATA_LICENSES.md` 重新生成（19 → 22 个文件）；
  `sources.json` 增加三条 `bse` 条目（含 sha256 钉）。

## 0.7.0 — 2026-11-07

### 新增：`functional/m06-l.yaml` 与 `functional/m06-2x.yaml`（DA-01 / DA-02 / FX-02）
- M06-L（45 个参数）与 M06-2X（38 个参数）：交换的动能级数 `a1..a12`（M06-L 另有
  VS98 的 `d1..d6` 与 `alpha`）与相关的 `gamma_ss/gamma_ab/alpha_ss/alpha_ab` +
  `c_ss/c_ab/d_ss/d_ab`。数值逐项对照 libxc 的 `par_m06l`/`par_m06_2x` 与
  `m06l_values`/`m062x_values`。
- 两个文件都声明 `requires.kinetic_density: true`（FX-01 的 K1 加性字段），
  所以引擎会真的算 τ；数据与实现的一致性由 `check_meta_gga_declaration` 双向核对。
- M06-2X 用 `HfExchange` 分量表达 54% 精确交换（`MggaXM062x` 的 `a1 = 0.46` 已折进
  半局域部分，两者互补：`0.46 + 0.54 = 1`）。
- `registry.yaml` 0.6.0 → 0.7.0；`DATA_LICENSES.md` 重新生成（17 → 19 个文件）。

## 0.6.0 — 2026-11-06

### 新增：PBE 的 PW92 三组参数 + 把 `pbe_mu` 改成精确值（阶段二 SC-01b）
- `functional/pbe.yaml` 与 `functional/pbe0.yaml` 增加 18 个参数：`pw92_p_*`（顺磁）、
  `pw92_f_*`（铁磁）、`pw92_a_*`（自旋刚度 α_c），每组是 `A`/`alpha`/`b1..b4`。
  自旋分辨的 PBE 相关（UKS）需要 ε_c^P、ε_c^F 与 α_c 三套；闭壳层只需要第一套，
  而以前那三个常数是**写死在 `qwb-xc/src/gga.rs` 里**的。现在三套都进数据、走同一个
  `Pw92Set::energy_and_derivative`，`σ→0` 时与旧实现逐位相同（回归钉住）。
- **`pbe_mu` 从 `0.21951` 改成 `0.21951497276451704`**（= `beta*pi^2/3`，由同一文件里
  17 位的 `pbe_beta` 算出，不再手抄）。这是**修一个数据不一致**：PBE 原文里 μ 与 β
  本来就是同一组常数推出的，不是两个独立参数。旧值只有 6 位有效数字，把 PBE 总能量
  顶在 2.6e-5 Ha 的偏差上（`docs/choices.md` 五十一）。
- 影响面（全部重新钉过，见各测试的注释）：PBE 总能量整体移动 ~1e-5 Ha 量级
  （CuH/LANL2DZ 从 −213.9067554378 → −213.9067719351）；`GGA_X_PBE` 的逐点
  σ 导数与 libxc 差从 2.3e-5 掉到 1e-12。LDA、基组、ECP 数据一律未动。
- `parameter_sources` 里给新增参数写了来源（PW92, PRB 45, 13244）。
- `registry.yaml` 0.5.0 → 0.6.0（两个 sha256 重新盖章）。生产数据文件数不变（17 个）。

## 0.5.0 — 2026-11-06

### 新增：`numerics/numerics.yaml`（阶段二 KV-04 / K10，`project-phase2.md` §8.5）
- 新的数据**域** `numerics`（`registry.yaml` 新增 `numerics:` 段；K1 新增 `NumericsDef`：
  `levels[]` + `equivalence_tolerance`）。
- 五级精度：`1..5`，每级**只引用** `grid/*.yaml` 的 `name`（`grid-coarse`/`grid-normal`/
  `grid-fine`/`grid-veryfine`），网格参数仍然只有一份；加载器会校验引用存在。
- pseudospectral 配点：30 → 60 → 90 → 120 → 180（每原子），去别名从第 3 级起打开且**不许关回去**
  （K10 条件 ② 的数据侧一半，`NumericsDef::check_monotone` 强制）。
- **等价容差是我们自己的判据**，不是文献值：RI 1e-6 Ha、pseudospectral 5e-5 Ha
  （比 RI 松，见 phase-2 §9.3）。它们进数据是因为**改判据要留痕**：改这个 = 改验收标准，
  必须 bump 本文件版本 + 在 PR body 里说明。
- 许可：自研，`Apache-2.0`（可商用、可再分发）。
- `registry.yaml` 0.4.0 → 0.5.0；`DATA_LICENSES.md` 自动重新生成（16 → 17 个生产数据文件）。

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
