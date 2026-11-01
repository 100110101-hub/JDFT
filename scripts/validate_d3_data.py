"""用 `qwb-data/dispersion/d3bj.yaml` 复算 D3(BJ) 的能量与梯度，与官方实现对照（DA-06）。

这个脚本**不是**引擎的一部分，它是数据的自检：

1. 只从数据文件里取值（C6 参考表、r4r2、rcov、kcn/scale、参考 CN 与权重宽度、阻尼参数）；
2. 按 D3 原文的公式算 CN → 权重 → C6 → 两点能量与解析梯度；
3. 与 `scripts/dftd3_dispersion_reference.py` 生成的官方参考（`d3_dispersion.csv`）逐原子比。

它证明的是「这份数据 + 这几个常量能复现官方实现」，也就是判据里「与官方参数表逐项
diff < 1e-12」的**端到端**版本；Rust 侧（FX-04）只负责把同一组公式搬过去。

用法：
    .venv/Scripts/python.exe -X utf8 scripts/validate_d3_data.py [--tolerance 1e-6]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "qwb-data", "dispersion", "d3bj.yaml")
GEOMETRY = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "d3_dimers.csv"
)
REFERENCE = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "d3_dispersion.csv"
)
#: 零阻尼（D3(0)）的两份：数据与官方参考。
DATA_ZERO = os.path.join(ROOT, "qwb-data", "dispersion", "d3zero.yaml")
REFERENCE_ZERO = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "d3zero_dispersion.csv"
)

#: 与 K2 的 `BOHR_PER_ANGSTROM` 一致（CODATA 2018）。
BOHR_PER_ANGSTROM = 1.889726089838572
HARTREE_TO_KCAL = 627.5094740631


def load_data(path):
    import yaml  # noqa: PLC0415

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class Model:
    """D3(BJ) 的最小实现（只依赖数据文件）。"""

    def __init__(self, data, functional):
        self.elements = data["elements"]
        self.index = {symbol: position for position, symbol in enumerate(self.elements)}
        cn = data["coordination_number"]
        self.kcn = float(cn["kcn"])
        self.scale = float(cn["scale"])
        self.radii = {
            symbol: float(value) * self.scale * BOHR_PER_ANGSTROM
            for symbol, value in cn["radii_angstrom"].items()
        }
        self.wf = float(data["reference"]["wf"])
        self.r4r2 = {symbol: float(value) for symbol, value in data["r4r2"].items()}
        self.reference_cn = {
            symbol: [float(value) for value in values]
            for symbol, values in data["reference_cn"].items()
        }
        self.reference_c6 = {
            key: [float(value) for value in values]
            for key, values in data["reference_c6"].items()
        }
        # BJ 那份有 `damping_parameters`；零阻尼的子类只借基类的 CN/权重/C6 部分，
        # 阻尼参数在子类里读自己的表。
        params = data.get("damping_parameters", {}).get(functional, {})
        self.s6 = float(params.get("s6", 1.0))
        self.s8 = float(params.get("s8", 0.0))
        self.a1 = float(params.get("a1", 0.0))
        self.a2 = float(params.get("a2", 0.0))

    # ---- CN ---------------------------------------------------------------
    def coordination_numbers(self, symbols, positions):
        n = len(symbols)
        radii = [self.radii[symbol] for symbol in symbols]
        cn = [0.0] * n
        for i in range(n):
            for j in range(i, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < 1e-12 or r2 > 40.0 * 40.0:
                    continue
                r = math.sqrt(r2)
                rc = radii[i] + radii[j]
                countf = 1.0 / (1.0 + math.exp(-self.kcn * (rc / r - 1.0)))
                cn[i] += countf
                if i != j:
                    cn[j] += countf
        return cn

    # ---- 参考体系权重 ------------------------------------------------------
    def weights(self, symbol, cn_value):
        reference = self.reference_cn[symbol]
        dcn = [value - cn_value for value in reference]
        dmin = min(value * value for value in dcn)
        gw = [math.exp(-self.wf * (value * value - dmin)) for value in dcn]
        norm = sum(gw)
        return [value / norm for value in gw]

    def weights_derivative(self, symbol, cn_value):
        reference = self.reference_cn[symbol]
        dcn = [value - cn_value for value in reference]
        dmin = min(value * value for value in dcn)
        gw = [math.exp(-self.wf * (value * value - dmin)) for value in dcn]
        norm = sum(gw)
        gw = [value / norm for value in gw]
        slope = [2.0 * self.wf * value for value in dcn]
        mean = sum(w * s for w, s in zip(gw, slope))
        return gw, [w * (s - mean) for w, s in zip(gw, slope)]

    # ---- C6 ---------------------------------------------------------------
    def reference_c6_pair(self, symbol_i, symbol_j):
        a, b = self.index[symbol_i], self.index[symbol_j]
        hi, lo = (a, b) if a >= b else (b, a)
        key = f"{self.elements[hi]}-{self.elements[lo]}"
        return self.reference_c6[key]

    def pair_c6(self, symbol_i, symbol_j, gw_i, gw_j, dgw_i, dgw_j):
        table = self.reference_c6_pair(symbol_i, symbol_j)
        # 参考表每个元素对固定是 7×7（Fortran 第一维最快），实际用到的只是
        # 「高 Z 元素的参考点个数 × 低 Z 元素的参考点个数」子块：块内第一维是
        # **高 Z** 元素、第二维是低 Z 元素（用官方实现反解 C6 验证过，见
        # scripts/validate_d3_data.py 的说明）。用 `nref` 当步长会读到错的元素。
        stride = math.isqrt(len(table))
        nref = len(gw_i)
        c6 = 0.0
        dc6_i = 0.0
        dc6_j = 0.0
        high_first = self.index[symbol_i] >= self.index[symbol_j]
        for iref in range(nref):
            for jref in range(len(gw_j)):
                offset = iref + stride * jref if high_first else jref + stride * iref
                refc6 = table[offset]
                c6 += gw_i[iref] * gw_j[jref] * refc6
                dc6_i += dgw_i[iref] * gw_j[jref] * refc6
                dc6_j += gw_i[iref] * dgw_j[jref] * refc6
        return c6, dc6_i, dc6_j

    # ---- 两点项 ------------------------------------------------------------
    def damping(self, symbol_i, symbol_j, r):
        r4r2 = self.r4r2[symbol_i] * self.r4r2[symbol_j]
        rrij = 3.0 * r4r2
        r0 = self.a1 * math.sqrt(rrij) + self.a2
        r0_6 = r0**6
        r0_8 = r0**8
        r6 = r**6
        r8 = r**8
        t6 = 1.0 / (r6 + r0_6)
        t8 = 1.0 / (r8 + r0_8)
        p = self.s6 * t6 + self.s8 * rrij * t8
        dp = -self.s6 * 6.0 * r**5 * t6 * t6 - self.s8 * rrij * 8.0 * r**7 * t8 * t8
        return p, dp, rrij

    # ---- 能量与梯度 --------------------------------------------------------
    def energy_and_gradient(self, symbols, positions):
        n = len(symbols)
        cn = self.coordination_numbers(symbols, positions)
        gw = []
        dgw = []
        for index, symbol in enumerate(symbols):
            w, dw = self.weights_derivative(symbol, cn[index])
            gw.append(w)
            dgw.append(dw)

        energy = 0.0
        gradient = [[0.0, 0.0, 0.0] for _ in range(n)]
        dedcn = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < 1e-12 or r2 > 60.0 * 60.0:
                    continue
                r = math.sqrt(r2)
                c6, dc6_i, dc6_j = self.pair_c6(
                    symbols[i], symbols[j], gw[i], gw[j], dgw[i], dgw[j]
                )
                p, dp, _ = self.damping(symbols[i], symbols[j], r)
                energy -= c6 * p
                # 直接项：dE/dr = -C6 * dp/dr，方向从 j 指向 i。
                magnitude = -c6 * dp
                unit = [dx / r, dy / r, dz / r]
                for axis in range(3):
                    gradient[i][axis] += magnitude * unit[axis]
                    gradient[j][axis] -= magnitude * unit[axis]
                # CN 链式：dE/dCN_i = -p * dC6_i/dCN_i。
                dedcn[i] -= p * dc6_i
                dedcn[j] -= p * dc6_j

        radii = [self.radii[symbol] for symbol in symbols]
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < 1e-12 or r2 > 40.0 * 40.0:
                    continue
                r = math.sqrt(r2)
                rc = radii[i] + radii[j]
                countf = 1.0 / (1.0 + math.exp(-self.kcn * (rc / r - 1.0)))
                dfdr = -self.kcn * rc / r2 * countf * (1.0 - countf)
                magnitude = dedcn[i] * dfdr + dedcn[j] * dfdr
                unit = [dx / r, dy / r, dz / r]
                for axis in range(3):
                    gradient[i][axis] += magnitude * unit[axis]
                    gradient[j][axis] -= magnitude * unit[axis]
        return energy, gradient


class ZeroModel(Model):
    """零阻尼（D3(0)）的同一套实现，只换阻尼函数。

    `zero.f90` 第 197–212 行的定义：

    ```text
    R0_ij = vdw_rad(jzp, izp)               成对表（不是两原子半径相加）
    t6    = (rs6·R0/r)^alp,  t8 = (rs8·R0/r)^(alp+2)
    f6    = 1/(1 + 6·t6),    f8 = 1/(1 + 6·t8)
    E     = −(s6·C6·f6/r⁶ + s8·C6·(C8/C6)·f8/r⁸)
    ```
    """

    def __init__(self, data, functional):
        super().__init__(data, functional)
        self.vdw = {
            key: float(value) * BOHR_PER_ANGSTROM
            for key, value in data["vdw_radii_angstrom"].items()
        }
        params = data["zero_damping_parameters"][functional]
        self.zero_s6 = float(params["s6"])
        self.zero_s8 = float(params["s8"])
        self.rs6 = float(params["rs6"])
        self.rs8 = float(params["rs8"])
        self.alp = float(params["alp"])

    def zero_damping(self, symbol_i, symbol_j, r):
        a, b = self.index[symbol_i], self.index[symbol_j]
        hi, lo = (a, b) if a >= b else (b, a)
        r0 = self.vdw[f"{self.elements[hi]}-{self.elements[lo]}"]
        rrij = 3.0 * self.r4r2[symbol_i] * self.r4r2[symbol_j]
        alp6 = self.alp
        alp8 = self.alp + 2.0
        t6 = (self.rs6 * r0 / r) ** alp6
        t8 = (self.rs8 * r0 / r) ** alp8
        f6 = 1.0 / (1.0 + 6.0 * t6)
        f8 = 1.0 / (1.0 + 6.0 * t8)
        dt6 = -alp6 * t6 / r
        dt8 = -alp8 * t8 / r
        df6 = -6.0 * dt6 * f6 * f6
        df8 = -6.0 * dt8 * f8 * f8
        p = self.zero_s6 * f6 / r**6 + self.zero_s8 * rrij * f8 / r**8
        dp = (
            self.zero_s6 * (df6 / r**6 - 6.0 * f6 / r**7)
            + self.zero_s8 * rrij * (df8 / r**8 - 8.0 * f8 / r**9)
        )
        return p, dp, rrij


def zero_energy_and_gradient(model, symbols, positions):
    """零阻尼的能量与解析梯度（复用 `Model` 的 CN/权重/C6 部分）。"""
    n = len(symbols)
    cn = model.coordination_numbers(symbols, positions)
    gw = []
    dgw = []
    for index, symbol in enumerate(symbols):
        w, dw = model.weights_derivative(symbol, cn[index])
        gw.append(w)
        dgw.append(dw)
    energy = 0.0
    gradient = [[0.0, 0.0, 0.0] for _ in range(n)]
    dedcn = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < 1e-12 or r2 > 60.0 * 60.0:
                continue
            r = math.sqrt(r2)
            c6, dc6_i, dc6_j = model.pair_c6(
                symbols[i], symbols[j], gw[i], gw[j], dgw[i], dgw[j]
            )
            p, dp, _ = model.zero_damping(symbols[i], symbols[j], r)
            energy -= c6 * p
            magnitude = -c6 * dp
            unit = [dx / r, dy / r, dz / r]
            for axis in range(3):
                gradient[i][axis] += magnitude * unit[axis]
                gradient[j][axis] -= magnitude * unit[axis]
            dedcn[i] -= p * dc6_i
            dedcn[j] -= p * dc6_j
    radii = [model.radii[symbol] for symbol in symbols]
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < 1e-12 or r2 > 40.0 * 40.0:
                continue
            r = math.sqrt(r2)
            rc = radii[i] + radii[j]
            countf = 1.0 / (1.0 + math.exp(-model.kcn * (rc / r - 1.0)))
            dfdr = -model.kcn * rc / r2 * countf * (1.0 - countf)
            magnitude = (dedcn[i] + dedcn[j]) * dfdr
            unit = [dx / r, dy / r, dz / r]
            for axis in range(3):
                gradient[i][axis] += magnitude * unit[axis]
                gradient[j][axis] -= magnitude * unit[axis]
    return energy, gradient


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1e-9, help="能量容差（Ha）")
    parser.add_argument("--tolerance-grad", type=float, default=1e-9, help="梯度容差（Ha/Bohr）")
    parser.add_argument(
        "--zero",
        action="store_true",
        help="校验零阻尼（d3zero.yaml）而不是 BJ（d3bj.yaml）",
    )
    args = parser.parse_args()

    data = load_data(DATA)
    geometry = read_csv(GEOMETRY)
    reference = read_csv(REFERENCE)

    by_dimer = {}
    for row in geometry:
        by_dimer.setdefault(row["dimer"], []).append(row)
    reference_by = {}
    for row in reference:
        reference_by.setdefault((row["dimer"], row["method"]), []).append(row)

    worst_energy = (0.0, "")
    worst_gradient = (0.0, "")
    checked = 0
    for dimer, atoms in by_dimer.items():
        symbols = [atom["element"] for atom in atoms]
        positions = [
            [
                float(atom["x_angstrom"]) * BOHR_PER_ANGSTROM,
                float(atom["y_angstrom"]) * BOHR_PER_ANGSTROM,
                float(atom["z_angstrom"]) * BOHR_PER_ANGSTROM,
            ]
            for atom in atoms
        ]
        for method in sorted({key[1] for key in reference_by if key[0] == dimer}):
            model = Model(data, method)
            energy, gradient = model.energy_and_gradient(symbols, positions)
            rows = reference_by[(dimer, method)]
            expected = float(rows[0]["energy_hartree"])
            delta = abs(energy - expected)
            print(f"  {dimer:32s} {method:6s} ours={energy:+.9e} oracle={expected:+.9e} d={delta:.3e}")
            if delta > worst_energy[0]:
                worst_energy = (delta, f"{dimer}/{method}")
            for index, row in enumerate(rows):
                for axis, name in enumerate(("gradient_x", "gradient_y", "gradient_z")):
                    diff = abs(gradient[index][axis] - float(row[name]))
                    if diff > worst_gradient[0]:
                        worst_gradient = (diff, f"{dimer}/{method}/atom{index + 1}/{name}")
            checked += 1

    # ---- 双 Δ 自检：能量必须与解析梯度一致（避免两边一起错）------------------
    model = Model(data, "b3lyp")
    atoms = by_dimer["water_dimer"]
    symbols = [atom["element"] for atom in atoms]
    positions = [
        [
            float(atom["x_angstrom"]) * BOHR_PER_ANGSTROM,
            float(atom["y_angstrom"]) * BOHR_PER_ANGSTROM,
            float(atom["z_angstrom"]) * BOHR_PER_ANGSTROM,
        ]
        for atom in atoms
    ]
    _, analytic = model.energy_and_gradient(symbols, positions)
    step = 1e-5
    worst_fd = 0.0
    for index in range(len(symbols)):
        for axis in range(3):
            shifted = [list(position) for position in positions]
            shifted[index][axis] += step
            plus, _ = model.energy_and_gradient(symbols, shifted)
            shifted[index][axis] -= 2.0 * step
            minus, _ = model.energy_and_gradient(symbols, shifted)
            numeric = (plus - minus) / (2.0 * step)
            worst_fd = max(worst_fd, abs(numeric - analytic[index][axis]))

    if args.zero:
        data_zero = load_data(DATA_ZERO)
        reference = read_csv(REFERENCE_ZERO)
        by_dimer = {}
        for row in geometry:
            by_dimer.setdefault(row["dimer"], []).append(row)
        reference_by = {}
        for row in reference:
            reference_by.setdefault((row["dimer"], row["method"]), []).append(row)
        worst_energy = (0.0, "")
        worst_gradient = (0.0, "")
        checked = 0
        methods = sorted({key[1] for key in reference_by})
        for dimer, atoms in by_dimer.items():
            symbols = [atom["element"] for atom in atoms]
            positions = [
                [
                    float(atom["x_angstrom"]) * BOHR_PER_ANGSTROM,
                    float(atom["y_angstrom"]) * BOHR_PER_ANGSTROM,
                    float(atom["z_angstrom"]) * BOHR_PER_ANGSTROM,
                ]
                for atom in atoms
            ]
            for method in methods:
                rows = reference_by.get((dimer, method))
                if rows is None:
                    continue
                model = ZeroModel(data_zero, method)
                energy, gradient = zero_energy_and_gradient(model, symbols, positions)
                expected = float(rows[0]["energy_hartree"])
                delta = abs(energy - expected)
                if delta > worst_energy[0]:
                    worst_energy = (delta, f"{dimer}/{method}")
                for index, row in enumerate(rows):
                    for axis, name in enumerate(
                        ("gradient_x", "gradient_y", "gradient_z")
                    ):
                        diff = abs(gradient[index][axis] - float(row[name]))
                        if diff > worst_gradient[0]:
                            worst_gradient = (diff, f"{dimer}/{method}/atom{index + 1}")
                checked += 1
        print(f"零阻尼数据：{DATA_ZERO}")
        print(f"官方参考：{REFERENCE_ZERO}（{len(methods)} 个方法：{', '.join(methods)}）")
        print(f"比较了 {checked} 个 (二聚体, 方法) 组合")
        print(
            f"  最大 |ΔE| = {worst_energy[0]:.3e} Ha = "
            f"{worst_energy[0] * HARTREE_TO_KCAL:.3e} kcal/mol（{worst_energy[1]}）"
        )
        print(f"  最大 |ΔG| = {worst_gradient[0]:.3e} Ha/Bohr（{worst_gradient[1]}）")
        failures = []
        if worst_energy[0] > args.tolerance:
            failures.append(f"energy {worst_energy[0]:.3e} Ha > {args.tolerance:g}")
        if worst_gradient[0] > args.tolerance_grad:
            failures.append(f"gradient {worst_gradient[0]:.3e} > {args.tolerance_grad:g}")
        for failure in failures:
            print(f"FAIL: {failure}")
        if failures:
            return 1
        print("ok: 零阻尼数据复现官方实现（能量与解析梯度都在容差内）")
        return 0

    print(f"数据文件：{DATA}")
    print(f"参考：{REFERENCE}（官方 s-dftd3）")
    print(f"比较了 {checked} 个 (二聚体, 方法) 组合，共 {len(reference)} 个原子梯度分量")
    print(
        f"  最大 |ΔE|   = {worst_energy[0]:.3e} Ha = "
        f"{worst_energy[0] * HARTREE_TO_KCAL:.3e} kcal/mol（{worst_energy[1]}）"
    )
    print(f"  最大 |ΔG|   = {worst_gradient[0]:.3e} Ha/Bohr（{worst_gradient[1]}）")
    print(f"  解析梯度 vs 中心差分（水二聚体，B3LYP）：{worst_fd:.3e} Ha/Bohr")

    failures = []
    if worst_energy[0] > args.tolerance:
        failures.append(f"energy {worst_energy[0]:.3e} Ha > {args.tolerance:g}")
    if worst_gradient[0] > args.tolerance_grad:
        failures.append(f"gradient {worst_gradient[0]:.3e} > {args.tolerance_grad:g}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("ok: 数据文件 + 公式复现官方实现（能量与解析梯度都在容差内）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
