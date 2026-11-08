#!/usr/bin/env python3
"""T5 的 ZnO 参考行：**PySCF 2.14.0 的 PBE/LANL2DZ 值**（外部标尺，不是本项目自己的输出）。

# 为什么换成 PySCF 的值

这一份原先的三行（`-103.7068571731` / `-103.7069534201` / `-103.7779303565`）是
**RP-02 缺陷的产物**，不是 ZnO 的能量。2026-11 的测量：

* 三行都等于 `H_core` 初猜密度的能量（同一个基组/网格上独立算：
  `-103.7068819011` / `-103.7069781304` / `-103.7779550527`，差 2.5e-5 Ha —— 残差是
  展宽结束后那一步的 DIIS 外推没有**恰好**回到初猜，HCN 上同一现象的残差是 3e-7）；
* 初猜能量**与展宽温度无关**，所以 CSV 原来那句「结果与展宽温度/步数无关，
  T = 0.1–0.5、步数 10–80 给同一个能量」正是缺陷的签名，而不是"收敛解唯一"的证据。

修掉 `fermi_occupations` 的方向之后，本项目的 SCF 在 ZnO/LANL2DZ 上**不收敛**
（限制性闭壳层的 O 原子 + 一个更强的既有问题：Zn 的 LANL2DZ ECP 的 l=2 通道被
弱化了约 8 倍 —— 见下面「现状」一节），所以这些数**不能**用本项目自己的输出重新钉。
（那个 ECP 缺陷已在 2026-11 修掉，见文件末尾的「现状」；这三行仍然是外部标尺。）

# 现在这三行是什么

PySCF 的复算结果，几何/基组/ECP/泛函与本项目的测试逐项对齐：

* 基组：BSE 的 LANL2DZ（H/O/Cu/Zn）。PySCF 的 `lanl2dz` 与 `qwb-data/basis/lanl2dz.yaml`
  逐壳层一致（指数与收缩系数逐位相同，`nao = 27`、`nelectron = 20`）；
* ECP：Zn 的 LANL2DZ ECP（18 芯，`z_eff = 12`），PySCF 的 `lanl2dz` ECP 与
  `qwb-data/ecp/lanl2dz-zn.yaml` 的每一项（含 `r^n` 幂次与通道 l = 0,1,2 + 局域 l = 3）
  逐项相同；O 与 H 是全电子（LANL2DZ 对第一排没有 ECP）；
* 泛函：`pbe` ↔ PySCF `PBE`；网格 `grids.level = 3`；`conv_tol = 1e-11`。

三条行对应三个口径，5 mHa 的容差是 T5 判据给的（不是这里定的）：

| 行 | 几何 | 说明 |
| --- | --- | --- |
| `zno_total_energy_r1p70_coarse` | r(Zn–O) = 1.70 Å | 门禁版（本项目 grid-coarse） |
| `zno_total_energy_r1p70` | r(Zn–O) = 1.70 Å | 慢档（本项目 grid-normal） |
| `zno_total_energy_equilibrium` | r(Zn–O) = 1.77 Å | oracle 位能曲线的最低点 |

`equilibrium` 是 **1.77 Å**，不是原来的 1.85 Å：oracle 在 1.60–1.95 Å 上扫描（含
1.74–1.81 Å 的细扫），最低点是 1.76 = -140.9656042058、**1.77 = -140.9656054282**、
1.78 = -140.9655225639。「1.85 Å 是平衡点」同样是伪解时代的结论 —— oracle 在 1.70 Å
的能量比 1.85 Å **更低**（-140.9635394365 vs -140.9629552417）。实验值 1.704 Å
（NIST），oracle 的 1.77 Å 与 LANL2DZ 对 ZnO 的已知系统偏差同量级。

# 现状（必须知道，否则会误读这三行）

**2026-11 更新：本项目的引擎现在到得了这三行。** 下面这段是修好 ECP 之前的实测记录，
留着是因为它标定了"这些数是外部标尺，不是本项目自己的回归钉"：

```text
本项目（RP-02 修复后，grid-coarse，r = 1.70，QWB_SCF_TRACE=1，max_iter = 200）
  rung 1（单初猜 DIIS）     200 步不收敛，能量在 -119…-121 Ha 之间振荡
  rung 2（多初猜 ×5）        每个 60 步，不收敛
  rung 3（费米展宽退火）     200 步不收敛（sum(f) = 20 全程正确，占据不再是零）
  rung 4（阻尼 Roothaan）    极限环：dE → 7e-9 Ha 而 max|dP| 卡在 4.445e-1
最终报 scf not converged in 200 iterations（max 200）
```

而 T5 的两个慢档测试（`#[ignore]`）就是按这三行做验收的。它们当时**是红的**，
红到下面这个既有缺陷被单独修掉为止（2026-11 已修，见文件末尾的"现状"一节）：

> **Zn 的 LANL2DZ ECP 的 l=2（d）非局域通道被弱化了约 8 倍。** 同一套参数、同一套基组：
> `Tr(S⁻¹ V_ecp)` 本项目 `+1.331`、PySCF `-21.701`；`S⁻¹V_ecp` 最负的 5 个本征值
> 本项目 `-0.78 … -0.14`、PySCF `-6.98 … -6.27`。3d¹⁰ 于是没被 ECP 好好束缚住，
> 总能量比 oracle 高约 20 Ha，SCF 停在伪态上（闭壳层受限的 O 是个伪解）。
> 这是 RP-02 之外的独立缺陷，**没有**跟 RP-02 一起修（修它要先动 ECP 的投影实现，
> 再按这里同样的口径重新验收 CuH 与 ZnO）。

也就是说：这三行是**答案**（外部标尺），不是"本项目自己的输出"。把行改成项目的输出
（原来那三行）才是真正的错误 —— 即使现在引擎已经落在答案的 1e-4 Ha 以内。

用法（WSL，PySCF 2.14.0）：

    python3 scripts/pyscf_zno_lanl2dz_reference.py            # 重新生成 CSV
    python3 scripts/pyscf_zno_lanl2dz_reference.py --check    # 只比对，不写盘
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import PYSCF_DOI, PYSCF_URL, rel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "zno_lanl2dz_pbe_ecp.csv"
)

HEADER = ["quantity", "value_hartree", "tolerance_hartree", "source", "source_doi", "source_url"]

#: T5 判据给的容差（不是本脚本选的）。
TOLERANCE = "0.005"
#: 这三行的来源标记：外部独立实现（PySCF）算出来的值，不是本项目自己的回归钉。
SOURCE = "pyscf-2.14.0-pbe-grids3"

COMMENTS = """\
# T5 参考值：ZnO / LANL2DZ（Zn 18 芯 ECP，z_eff = 12）/ PBE —— **PySCF 2.14.0 的复算值**
#
# 生成器：scripts/pyscf_zno_lanl2dz_reference.py（`--check` 用同一份代码重算并逐字比对）。
# 几何/基组/ECP/泛函的口径与对齐方式写在该脚本的文件头里；简言之：BSE 的 LANL2DZ 与
# PySCF 的 `lanl2dz` 逐壳层一致（nao = 27、nelectron = 20），Zn 的 ECP 逐项一致
# （18 芯，z_eff = 12，l = 0,1,2 非局域 + l = 3 局域），O/H 全电子，`grids.level = 3`。
#
# 行与几何（5 mHa 是 T5 判据给的容差）：
#   r1p70_coarse / r1p70  —— r(Zn–O) = 1.70 Å，门禁版（grid-coarse）与慢档（grid-normal）
#                            都对照**同一个** oracle 值：PySCF 的 grids.level = 1/2/3 之间
#                            只差 1.4e-5 Ha，本项目 coarse/normal/fine 之间约 1e-4 Ha，
#                            两者都远在 5 mHa 之内。
#   equilibrium           —— r(Zn–O) = 1.77 Å，oracle 曲线的最低点（离散扫描 1.60–1.95，
#                            细扫 1.74–1.81：1.76 = -140.9656042058、1.77 = -140.9656054282、
#                            1.78 = -140.9655225639）。**不是**原来的 1.85 Å。
#
# ── 旧的三行是 RP-02 缺陷的产物（实测，2026-11）───────────────────────────────
# 旧行：r1p70_coarse -103.7068571731、r1p70 -103.7069534201、equilibrium -103.7779303565。
# 它们都等于 `H_core` 初猜密度的能量，而不是 ZnO 的能量：同一个基组/网格上独立算
# E[P_guess] = Tr(PH) + ½Tr(PJ) + E_xc + E_nuc 得
#     grid-coarse r=1.70: -103.7068819011   （旧行 -103.7068571731，差 2.5e-5）
#     grid-normal r=1.70: -103.7069781304   （旧行 -103.7069534201，差 2.5e-5）
#     grid-normal r=1.85: -103.7779550527   （旧行 -103.7779303565，差 2.5e-5）
# 残差来自展宽结束后那一步的 DIIS 外推没有恰好回到初猜（HCN 上同一现象的残差是 3e-7）。
# 机制（见 qwb-scf/src/rks.rs 的 fermi_occupations 与收敛验收）：费米二分的两个分支写反，
# 占据数全塌成 0 → 密度归零 → 展宽结束后从零密度出发落回初猜 → 「能量不再变、密度不再动」
# 两条旧判据全过 → 把初猜能量当收敛解返回。初猜能量**与展宽温度无关**，所以旧 note 里
# 「与展宽温度/步数无关，T = 0.1–0.5、步数 10–80 给同一个能量」正是缺陷的签名。
#
# ── 现状（2026-11-2x 更新）：ECP 修好之后，本项目的引擎**已经到得了**这三行 ─────
# RP-02 修好之后、ECP 还没修时，ZnO/LANL2DZ/PBE（grid-coarse, r = 1.70, max_iter = 200）
# 的四级退路全都不收敛：单初猜 DIIS 200 步在 -119…-121 Ha 之间振荡；多初猜 5×60 步不行；
# 费米展宽退火 200 步不收敛；阻尼 Roothaan 进极限环（dE → 7e-9 Ha 而 max|dP| 卡在 4.445e-1）。
#
# 根因是一个**独立于 RP-02 的既有缺陷**，两层：
#   ① BSE 的 `r_exponents` 是 `r^(n-2)` 约定，数据里却声明成 `r^n`，加载器也没归一 ——
#      每个 ECP 项被整体抬 `r²`，局域通道的 `-18/r` 变成 `-18 r`；
#   ② 数据里的 `l < L` 块**已经是** `ΔV_l = V_l − V_L`（PySCF/libcint 的
#      `ECPscalar` = `V_local + Σ_l V_l P_l`，实测差 8.9e-16），实现里又减了一次 `V_L`。
# 两条加起来把 Zn 的 d 通道压成 1/18（`S⁻¹V_ecp` 的 d 子块对角元 -0.283 而 PySCF 是 -5.166），
# 3d¹⁰ 没被束缚住，总能量比 oracle 高约 20 Ha。
#
# 修好之后实测（2026-11）：单初猜 DIIS **28 步**收敛
#     grid-coarse r=1.70: -140.9636049221（本行 -140.9635394365，差 6.6e-5 Ha）
#     grid-normal r=1.70: -140.9635194044（差 2.0e-5 Ha）
#     grid-normal r=1.77: -140.9655952885（equilibrium 行，差 1.0e-5 Ha）
# T5 的两个慢档测试因此**转绿**（断言与容差一个字没改）。逐通道的对外核对钉在
# `qwb-core/crates/qwb-scf/tests/i05_ecp_oracle.rs`（1460 个矩阵元，最大差 4.6e-13）。
# 所以：这三行仍然是**答案**（外部标尺），而本项目的引擎现在落在答案的 1e-4 Ha 以内。
"""


def geometry(distance_angstrom: float):
    return [("Zn", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, distance_angstrom))]


def energy(distance_angstrom: float) -> float:
    """PySCF 2.14.0 的 PBE/LANL2DZ 总能量（Hartree）。"""
    from pyscf import dft, gto, scf

    mol = gto.M(
        atom=geometry(distance_angstrom),
        unit="Angstrom",
        basis="lanl2dz",
        # LANL2DZ 只对 Zn 有 ECP；PySCF 对 O 会打一行 "ECP lanl2dz not found for O" 的
        # 提示，那正是"O 是全电子"的正确行为。
        ecp="lanl2dz",
        charge=0,
        spin=0,
        verbose=0,
    )
    method = dft.RKS(mol)
    method.xc = "PBE"
    method.grids.level = 3
    method.conv_tol = 1.0e-11
    method.conv_tol_grad = 1.0e-7
    method.max_cycle = 500
    value = method.kernel()
    if not method.converged:
        # 只在 DIIS 停住时用二阶方法把**同一个解**收敛出来（这只影响怎么到解）。
        solver = scf.RHF(mol).newton()
        solver.conv_tol = 1.0e-11
        value = solver.kernel()
        method = solver
    if not method.converged:
        raise RuntimeError(f"PySCF did not converge at r = {distance_angstrom} Å")
    return float(value)


def rows():
    r1p70 = energy(1.70)
    equilibrium = energy(1.77)
    out = [
        {
            "quantity": "zno_total_energy_r1p70_coarse",
            "value_hartree": f"{r1p70:.10f}",
            "tolerance_hartree": TOLERANCE,
            "source": SOURCE,
        },
        {
            "quantity": "zno_total_energy_r1p70",
            "value_hartree": f"{r1p70:.10f}",
            "tolerance_hartree": TOLERANCE,
            "source": SOURCE,
        },
        {
            "quantity": "zno_total_energy_equilibrium",
            "value_hartree": f"{equilibrium:.10f}",
            "tolerance_hartree": TOLERANCE,
            "source": SOURCE,
        },
    ]
    for row in out:
        row["source_doi"] = PYSCF_DOI
        row["source_url"] = PYSCF_URL
    return out


def render(data) -> str:
    lines = COMMENTS.rstrip("\n").split("\n")
    lines.append(",".join(HEADER))
    for row in data:
        lines.append(",".join(row[column] for column in HEADER))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写盘")
    args = parser.parse_args()

    text = render(rows())
    if args.check:
        current = io.open(OUTPUT, encoding="utf-8").read() if os.path.exists(OUTPUT) else ""
        if current != text:
            print(f"FAIL {rel(OUTPUT)} does not match a fresh PySCF run")
            got = current.splitlines()
            want = text.splitlines()
            for index in range(max(len(got), len(want))):
                left = got[index] if index < len(got) else "<missing>"
                right = want[index] if index < len(want) else "<missing>"
                if left != right:
                    print(f"  line {index + 1}:\n    file: {left}\n    gen : {right}")
                    break
            return 1
        print(f"ok: {rel(OUTPUT)} matches PySCF ({len(text.splitlines())} lines)")
        return 0

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    io.open(OUTPUT, "w", encoding="utf-8", newline="\n").write(text)
    print(f"wrote {rel(OUTPUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
