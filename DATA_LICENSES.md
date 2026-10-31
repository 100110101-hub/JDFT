# 数据许可与出处（自动生成，请勿手改）

本文件由 `scripts/gen_data_licenses.py` 生成。重跑必须得到**逐字节相同**的结果
（排序固定 + 不含时间戳），所以它可以直接进 CI 比对。

共 30 个生产数据文件。`qwb-data/examples/` 下的 SPECIMEN 不计入
（它们是形状示例，不在 manifest 里）。

其中 `qwb-data/basis/` 与 `qwb-data/ecp/` 的部分文件是从 Basis Set Exchange 导出的
第三方数据：仓库**不提交**这些文件（再分发条款，见 `qwb-data/sources.json` 的
`redistribute_ok`），按需用 `qwb data install`（或 `python -m qwb_studio.datafetch --all`）
取回并核对摘要钉。下表因此不区分「已取回」与「未取回」——许可、引用与摘要都由清单
钉住，与本机是否下载过无关。

| 数据 | 版本 | 许可 | 商用 | 可再分发 | sha256（前 16 位） |
| --- | --- | --- | --- | --- | --- |
| `qwb-data/basis/6-311g-dp.yaml` | 1.0.0 | Apache-2.0 | true | true | `cfda92e209fe8705` |
| `qwb-data/basis/def2-svp-ri.yaml` | 1.0.0 | Apache-2.0 | true | true | `dd7c72e050bc4240` |
| `qwb-data/basis/def2-svp.yaml` | 1.0.0 | Apache-2.0 | true | true | `177675559ef883e0` |
| `qwb-data/basis/def2-tzvp-ri.yaml` | 1.0.0 | Apache-2.0 | true | true | `ba72aea87b0198f3` |
| `qwb-data/basis/def2-tzvp.yaml` | 1.0.0 | Apache-2.0 | true | true | `263b684f8c936200` |
| `qwb-data/basis/def2-universal-jkfit.yaml` | 1.0.0 | Apache-2.0 | true | true | `5ae0422479300349` |
| `qwb-data/basis/lanl2dz.yaml` | 1.0.1 | NOASSERTION | false | false | `bc14a7e191563cb1` |
| `qwb-data/basis/sto-3g.yaml` | 1.0.0 | Apache-2.0 | true | true | `fce3af14b8b3ea12` |
| `qwb-data/ecp/lanl2dz-cu.yaml` | 1.0.0 | NOASSERTION | false | false | `a44df71fd56eea38` |
| `qwb-data/ecp/lanl2dz-zn.yaml` | 1.0.0 | NOASSERTION | false | false | `55b3391680aaad1d` |
| `qwb-data/functional/b88.yaml` | 1.0.0 | Apache-2.0 | true | true | `efa0fa7b9f8aa98d` |
| `qwb-data/functional/m06-2x.yaml` | 1.0.0 | Apache-2.0 | true | true | `e277f61ab597f848` |
| `qwb-data/functional/m06-l.yaml` | 1.0.0 | Apache-2.0 | true | true | `6894c5ebbee3b960` |
| `qwb-data/functional/pbe.yaml` | 1.0.0 | Apache-2.0 | true | true | `b8b322f96b0e22fe` |
| `qwb-data/functional/pbe0.yaml` | 1.0.0 | Apache-2.0 | true | true | `e741078195d85f78` |
| `qwb-data/functional/svwn5.yaml` | 1.0.0 | Apache-2.0 | true | true | `e62bd0014d27c73d` |
| `qwb-data/grid/grid-coarse.yaml` | 1.0.0 | Apache-2.0 | true | true | `48eaeaa2d73acd45` |
| `qwb-data/grid/grid-fine.yaml` | 1.0.0 | Apache-2.0 | true | true | `25080b62b80416b1` |
| `qwb-data/grid/grid-normal.yaml` | 1.0.0 | Apache-2.0 | true | true | `a792b362b1fdb278` |
| `qwb-data/grid/grid-veryfine.yaml` | 1.0.0 | Apache-2.0 | true | true | `5c630f5a1ec0688d` |
| `qwb-data/numerics/numerics.yaml` | 1.0.0 | Apache-2.0 | true | true | `fea96fd0bd356da0` |
| `qwb-data/registry.yaml` | 0.11.1 | ? | ? | ? | `—（manifest，条目各自带` |
| `qwb-data/solvent/acetonitrile.yaml` | 2.1.0 | Apache-2.0 | true | true | `641eb80227e75fb8` |
| `qwb-data/solvent/chloroform.yaml` | 2.1.0 | Apache-2.0 | true | true | `01ae69f7167f95c8` |
| `qwb-data/solvent/dmso.yaml` | 2.1.0 | Apache-2.0 | true | true | `00714c4363b488b6` |
| `qwb-data/solvent/ethanol.yaml` | 2.1.0 | Apache-2.0 | true | true | `02929d534e28a595` |
| `qwb-data/solvent/methanol.yaml` | 2.1.0 | Apache-2.0 | true | true | `b52e4908be7b167c` |
| `qwb-data/solvent/thf.yaml` | 2.1.0 | Apache-2.0 | true | true | `3845409494ef95d9` |
| `qwb-data/solvent/toluene.yaml` | 2.1.0 | Apache-2.0 | true | true | `820497a4ffcbca4b` |
| `qwb-data/solvent/water.yaml` | 2.1.0 | Apache-2.0 | true | true | `69b4d37ed71958f7` |

## 引用

### `qwb-data/basis/6-311g-dp.yaml`（6-311g(d,p) 1.0.0）

- Self-consistent molecular orbital methods. XX. A basis set for correlated wave functions (Krishnan, Binkley, Seeger, Pople, JCP 72, 650, 1980)

### `qwb-data/basis/def2-svp-ri.yaml`（def2-svp-ri 1.0.0）

- Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H-Rn (Weigend & Ahlrichs, PCCP 7, 3297, 2005)

### `qwb-data/basis/def2-svp.yaml`（def2-svp 1.0.0）

- Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H-Rn (Weigend & Ahlrichs, PCCP 7, 3297, 2005)

### `qwb-data/basis/def2-tzvp-ri.yaml`（def2-tzvp-ri 1.0.0）

- Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H-Rn (Weigend & Ahlrichs, PCCP 7, 3297, 2005)

### `qwb-data/basis/def2-tzvp.yaml`（def2-tzvp 1.0.0）

- Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H-Rn (Weigend & Ahlrichs, PCCP 7, 3297, 2005)

### `qwb-data/basis/def2-universal-jkfit.yaml`（def2-universal-jkfit 1.0.0）

- Universal basis sets of Gaussian-type functions for the fitting of the Coulomb and exchange contributions (Weigend, PCCP 8, 1057, 2006)

### `qwb-data/basis/lanl2dz.yaml`（lanl2dz 1.0.1）

- Ab initio effective core potentials for molecular calculations. Potentials for K to Au including the outermost core orbitals

### `qwb-data/basis/sto-3g.yaml`（sto-3g 1.0.0）

- Self-Consistent Molecular-Orbital Methods. I. Use of Gaussian Expansions of Slater-Type Atomic Orbitals

### `qwb-data/ecp/lanl2dz-cu.yaml`（lanl2dz-cu 1.0.0）

- Ab initio effective core potentials for molecular calculations. Potentials for K to Au including the outermost core orbitals

### `qwb-data/ecp/lanl2dz-zn.yaml`（lanl2dz-zn 1.0.0）

- Ab initio effective core potentials for molecular calculations. Potentials for the transition metal atoms Sc to Hg

### `qwb-data/functional/b88.yaml`（b88 1.0.0）

- Density-functional exchange-energy approximation with correct asymptotic behavior (Becke, Phys. Rev. A 38, 3098)
- Electron correlations by the Slater X-alpha method (Dirac-Slater LDA exchange, alpha = 2/3) —— 供 ε_x^LDA

### `qwb-data/functional/m06-2x.yaml`（m06-2x 1.0.0）

- The M06 suite of density functionals for main group thermochemistry, thermochemical kinetics, noncovalent interactions, excited states, and transition elements: two new functionals and systematic testing of four M06-class functionals and 12 other functionals (Zhao & Truhlar, Theor. Chem. Acc. 120, 215)
- Density functionals with broad applicability in chemistry (Zhao & Truhlar, Acc. Chem. Res. 41, 157)

### `qwb-data/functional/m06-l.yaml`（m06-l 1.0.0）

- The M06 suite of density functionals for main group thermochemistry, thermochemical kinetics, noncovalent interactions, excited states, and transition elements: two new functionals and systematic testing of four M06-class functionals and 12 other functionals (Zhao & Truhlar, Theor. Chem. Acc. 120, 215)
- A new local density functional for main-group thermochemistry, transition metal bonding, thermochemical kinetics, and noncovalent interactions (Zhao & Truhlar, J. Chem. Phys. 125, 194101)

### `qwb-data/functional/pbe.yaml`（pbe 1.0.0）

- Generalized Gradient Approximation Made Simple (Perdew, Burke & Ernzerhof, Phys. Rev. Lett. 77, 3865)
- Accurate spin-dependent electron liquid correlation energies for local spin density calculations: a critical analysis (Vosko, Wilk & Nusair, Can. J. Phys. 58, 1200) —— 供相关项里的 ε_c^LDA

### `qwb-data/functional/pbe0.yaml`（pbe0 1.0.0）

- Toward reliable density functional methods without adjustable parameters: The PBE0 model (Adamo & Barone, J. Chem. Phys. 110, 6158)
- Generalized Gradient Approximation Made Simple (Perdew, Burke & Ernzerhof, Phys. Rev. Lett. 77, 3865)
- Accurate spin-dependent electron liquid correlation energies for local spin density calculations: a critical analysis (Vosko, Wilk & Nusair, Can. J. Phys. 58, 1200)

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

### `qwb-data/numerics/numerics.yaml`（numerics 1.0.0）

- A multicenter numerical integration scheme for polyatomic molecules (Becke partitioning)
- New Methods For Electronic Structure Calculations on Large Molecules (pseudospectral approach)

### `qwb-data/registry.yaml`（qwb-data 0.11.1）


### `qwb-data/solvent/acetonitrile.yaml`（acetonitrile 2.1.0）

- Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions (SMD)
- van der Waals volumes and radii (Bondi)

### `qwb-data/solvent/chloroform.yaml`（chloroform 2.1.0）

- Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions (SMD)
- van der Waals volumes and radii (Bondi)

### `qwb-data/solvent/dmso.yaml`（dmso 2.1.0）

- Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions (SMD)
- van der Waals volumes and radii (Bondi)

### `qwb-data/solvent/ethanol.yaml`（ethanol 2.1.0）

- Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions (SMD)
- van der Waals volumes and radii (Bondi)

### `qwb-data/solvent/methanol.yaml`（methanol 2.1.0）

- Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions (SMD)
- van der Waals volumes and radii (Bondi)

### `qwb-data/solvent/thf.yaml`（thf 2.1.0）

- Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions (SMD)
- van der Waals volumes and radii (Bondi)

### `qwb-data/solvent/toluene.yaml`（toluene 2.1.0）

- Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions (SMD)
- van der Waals volumes and radii (Bondi)

### `qwb-data/solvent/water.yaml`（water 2.1.0）

- Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions (SMD)
- van der Waals volumes and radii (Bondi)

## 门禁状态

所有数据都带 license / citation / sha256，且 sha256 与内容一致。
