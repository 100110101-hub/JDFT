# QuantumWorkbench (QWB) v2.0

分子 DFT 软件的开工仓库。**唯一开工依据是 [`project.md`](./project.md)**；本 README 只说明目录与本地验收命令，不构成规范。

## 目录（对应 project.md §12.1）

| 目录 | 内容 | 契约 |
|---|---|---|
| `qwb-contracts/` | K1–K6 契约：数据 schema、`qwb-engine-api` crate、产品面 schema、CLI/Python 契约 | K1–K6 |
| `qwb-data/` | 数据仓库：`basis/ ecp/ functional/ grid/ solvent/`、`registry.yaml`、`examples/` | K1 |
| `qwb-core/` | Cargo workspace：`qwb-registry`、`qwb-io`、`qwb-cli` … | K2/K4/K5 |
| `docs/` | `status.yaml`（WBS 状态）、`choices.md`（已做选择与待人类裁决项） | — |
| `scripts/` | 门禁脚本（unsafe 白名单、rayon 确定性） | G6/G7 |

> 约 3 仓（`qwb-contracts` / `qwb-data` / `qwb-core`）当前在**同一个 git 工作树**中，靠目录隔离；物理拆仓与否见 `docs/choices.md` 的待裁决项。

## 本地验收（C 轨）

```powershell
cd qwb-core
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check
cargo run -q -p qwb-cli -- data verify
cd ..\qwb-contracts\engine-api
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps
cd ..\..
.\.venv\Scripts\python.exe -m mypy --strict qwb-contracts\studio\python\qwb_engine.pyi
```

退出码契约见 K5（`0` 成功 / `1` 输入错误 / `2` 收敛失败 / `3` 数据校验失败 / `4` 内部错误）。
