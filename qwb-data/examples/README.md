# qwb-data/examples — K1 schema 样例（SPECIMEN）

这里**不是生产数据**。它们存在的唯一目的是让 K1 的 schema、加载器路径、
`qwb data verify`、以及 `.qwb` 示例容器有真实文件可校验。

## 为什么可以放心放着

- 文件名和 `name` 字段都带 `example-` 前缀 → 生产查找（如 `basis_name: sto-3g`）
  不可能命中它们；
- 每个文件的 `notes` 字段写明自己是样例以及由哪个 WP 交付真实数据；
- `DATA_LICENSES.md` 与 `docs/status.yaml` 都标注了对应 D-xx 尚未完成。

## 已核对的数值 vs 占位值

| 文件 | 可信 | 占位/合成 |
|---|---|---|
| `basis/example-sto-3g-H.yaml` | H 的指数与收缩系数（文献常见值） | 只覆盖 H（D-03 要 5 个元素并做 BSE diff） |
| `ecp/example-ecp-cu.yaml` | `Z_eff = Z - n_core = 1`（Cu, 28 芯电子）自洽 | **全部高斯系数是合成的** |
| `functional/example-svwn5.yaml` | 组成项与所需密度量 | `parameters: {}`（VWN5 常数属 D-09） |
| `grid/example-grid-normal.yaml` | Lebedev 302 是合法阶数 | 径向点数/锐度为常见取值，非 D-13 定案 |
| `solvent/example-water.yaml` | ε = 78.3553 @ 298.15 K（§8.5 参考值） | 原子半径是占位值 |

## 维护方式

```powershell
# 改完数据后重新打 sha256（G2 要求）
cargo run -q -p qwb-registry --bin qwb-stamp-data -- ..\qwb-data\examples\basis\example-sto-3g-H.yaml
# 或一次全部
cargo run -q -p qwb-cli --bin qwb -- data verify
```

⚠ `stamp` 会把文件重写成规范形式，**YAML 注释会丢失** —— 想留下说明请写进 `notes`。
