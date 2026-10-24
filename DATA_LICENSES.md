# 数据许可与出处（自动生成，请勿手改）

本文件由 `scripts/gen_data_licenses.py` 生成。重跑必须得到**逐字节相同**的结果
（排序固定 + 不含时间戳），所以它可以直接进 CI 比对。

共 12 个生产数据文件。`qwb-data/examples/` 下的 SPECIMEN 不计入
（它们是形状示例，不在 manifest 里）。

| 数据 | 版本 | 许可 | 商用 | 可再分发 | sha256（前 16 位） |
| --- | --- | --- | --- | --- | --- |
| `qwb-data/basis/6-311g-dp.yaml` | 1.0.0 | Apache-2.0 | true | true | `cfda92e209fe8705` |
| `qwb-data/basis/def2-svp.yaml` | 1.0.0 | Apache-2.0 | true | true | `177675559ef883e0` |
| `qwb-data/basis/def2-tzvp.yaml` | 1.0.0 | Apache-2.0 | true | true | `263b684f8c936200` |
| `qwb-data/basis/sto-3g.yaml` | 1.0.0 | Apache-2.0 | true | true | `fce3af14b8b3ea12` |
| `qwb-data/functional/b88.yaml` | 1.0.0 | Apache-2.0 | true | true | `efa0fa7b9f8aa98d` |
| `qwb-data/functional/pbe.yaml` | 1.0.0 | Apache-2.0 | true | true | `6f9c7a390d0d813e` |
| `qwb-data/functional/svwn5.yaml` | 1.0.0 | Apache-2.0 | true | true | `e62bd0014d27c73d` |
| `qwb-data/grid/grid-coarse.yaml` | 1.0.0 | Apache-2.0 | true | true | `48eaeaa2d73acd45` |
| `qwb-data/grid/grid-fine.yaml` | 1.0.0 | Apache-2.0 | true | true | `25080b62b80416b1` |
| `qwb-data/grid/grid-normal.yaml` | 1.0.0 | Apache-2.0 | true | true | `a792b362b1fdb278` |
| `qwb-data/grid/grid-veryfine.yaml` | 1.0.0 | Apache-2.0 | true | true | `5c630f5a1ec0688d` |
| `qwb-data/registry.yaml` | 0.4.0 | ? | ? | ? | `—（manifest，条目各自带` |

## 引用

### `qwb-data/basis/6-311g-dp.yaml`（6-311g(d,p) 1.0.0）

- Self-consistent molecular orbital methods. XX. A basis set for correlated wave functions (Krishnan, Binkley, Seeger, Pople, JCP 72, 650, 1980)

### `qwb-data/basis/def2-svp.yaml`（def2-svp 1.0.0）

- Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H-Rn (Weigend & Ahlrichs, PCCP 7, 3297, 2005)

### `qwb-data/basis/def2-tzvp.yaml`（def2-tzvp 1.0.0）

- Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H-Rn (Weigend & Ahlrichs, PCCP 7, 3297, 2005)

### `qwb-data/basis/sto-3g.yaml`（sto-3g 1.0.0）

- Self-Consistent Molecular-Orbital Methods. I. Use of Gaussian Expansions of Slater-Type Atomic Orbitals

### `qwb-data/functional/b88.yaml`（b88 1.0.0）

- Density-functional exchange-energy approximation with correct asymptotic behavior (Becke, Phys. Rev. A 38, 3098)
- Electron correlations by the Slater X-alpha method (Dirac-Slater LDA exchange, alpha = 2/3) —— 供 ε_x^LDA

### `qwb-data/functional/pbe.yaml`（pbe 1.0.0）

- Generalized Gradient Approximation Made Simple (Perdew, Burke & Ernzerhof, Phys. Rev. Lett. 77, 3865)
- Accurate spin-dependent electron liquid correlation energies for local spin density calculations: a critical analysis (Vosko, Wilk & Nusair, Can. J. Phys. 58, 1200) —— 供相关项里的 ε_c^LDA

### `qwb-data/functional/svwn5.yaml`（svwn5 1.0.0）

- Accurate spin-dependent electron liquid correlation energies for local spin density calculations: a critical analysis (Vosko, Wilk & Nusair, Can. J. Phys. 58, 1200)
- Electron correlations by the Slater X-alpha method (Dirac-Slater LDA exchange, alpha = 2/3)

### `qwb-data/grid/grid-coarse.yaml`（grid-coarse 1.0.0）

- A multicenter numerical integration scheme for polyatomic molecules (Becke partitioning)
- Quadratures on a sphere (Lebedev angular grids)

### `qwb-data/grid/grid-fine.yaml`（grid-fine 1.0.0）

- A multicenter numerical integration scheme for polyatomic molecules (Becke partitioning)
- Quadratures on a sphere (Lebedev angular grids)

### `qwb-data/grid/grid-normal.yaml`（grid-normal 1.0.0）

- A multicenter numerical integration scheme for polyatomic molecules (Becke partitioning)
- Quadratures on a sphere (Lebedev angular grids)

### `qwb-data/grid/grid-veryfine.yaml`（grid-veryfine 1.0.0）

- A multicenter numerical integration scheme for polyatomic molecules (Becke partitioning)
- Quadratures on a sphere (Lebedev angular grids)

### `qwb-data/registry.yaml`（qwb-data 0.4.0）


## 门禁状态

所有数据都带 license / citation / sha256，且 sha256 与内容一致。
