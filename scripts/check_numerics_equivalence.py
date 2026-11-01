"""K10 等价性套件（VF-06）：把所有**已实装**的近似路径与解析路径逐项比对，输出差异表。

判据（project-phase2.md §6, VF-06）：「一个脚本跑完（scripts/check_numerics_equivalence.py），
输出差异表；任一超差即门禁红」。

容差来自 **数据**（`qwb-data/numerics/numerics.yaml` 的 `equivalence_tolerance`），
不在脚本里硬编码 —— 改容差就是改验收标准，必须 bump 数据版本。

已经实装的路径：RI-MP2（`method: ri-mp2`）与 canonical MP2（`method: mp2`）。
**未实装**的路径（pseudospectral / IN-03）不静默跳过，而是作为一行
`not implemented` 打出来并计入报告 —— 「没做」与「通过了」在输出里必须能分辨。

用法：
    .venv/Scripts/python.exe scripts/check_numerics_equivalence.py            # 差异表
    .venv/Scripts/python.exe scripts/check_numerics_equivalence.py --report  # 只报告不判红

退出码：0 = 所有已实装路径都在容差内；1 = 有超差；2 = 环境问题（找不到 qwb / 数据）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMERICS = os.path.join(ROOT, "qwb-data", "numerics", "numerics.yaml")
EXAMPLES = os.path.join(ROOT, "qwb-core", "examples")


def load_tolerances() -> dict:
    """从 `numerics.yaml` 读 `equivalence_tolerance`（PyYAML 只有 --check 需要，这里必须有）。"""
    import yaml  # noqa: PLC0415

    with open(NUMERICS, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    tolerance = data.get("equivalence_tolerance") or {}
    if not tolerance:
        raise SystemExit(f"{NUMERICS}: no equivalence_tolerance section")
    return {str(key): float(value) for key, value in tolerance.items()}


def find_qwb() -> list[str]:
    """返回**命令前缀**：优先现成的二进制，否则退回 `cargo run`。

    退回 `cargo run` 是为了让门禁在干净检出上也能跑（`core: test` 会编译 workspace 的
    测试目标，但不保证 `qwb` 可执行文件已经产出）；代价是第一次会多花一次编译时间。
    """
    # **debug 优先**：门禁里的 `core: test` 每次都会重编 debug target，而 release 二进制
    # 可能是一个旧构建（本轮就踩到了：数据字段改名后，门禁里 `data verify` 用新编的
    # 二进制通过，套件却用了陈旧的 release 二进制，报「unknown field」）。
    for candidate in (
        os.path.join(ROOT, "qwb-core", "target", "debug", "qwb.exe"),
        os.path.join(ROOT, "qwb-core", "target", "debug", "qwb"),
        os.path.join(ROOT, "qwb-core", "target", "release", "qwb.exe"),
        os.path.join(ROOT, "qwb-core", "target", "release", "qwb"),
    ):
        if os.path.isfile(candidate):
            return [candidate]
    found = shutil.which("qwb")
    if found:
        return [found]
    cargo = shutil.which("cargo") or shutil.which("cargo.exe")
    if cargo:
        return [
            cargo,
            "run",
            "-q",
            "--manifest-path",
            os.path.join(ROOT, "qwb-core", "Cargo.toml"),
            "-p",
            "qwb-cli",
            "--bin",
            "qwb",
            "--",
        ]
    raise SystemExit("neither qwb nor cargo found: cannot run the equivalence suite")


def run_example(qwb: list[str], example: str, workdir: str) -> dict:
    """跑一个示例，返回 `result.json` 的内容。"""
    out = os.path.join(workdir, os.path.splitext(example)[0])
    command = [*qwb, "run", os.path.join(EXAMPLES, example), "--out", out]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{example}: exit {completed.returncode}\n{completed.stdout[-800:]}\n"
            f"{completed.stderr[-800:]}"
        )
    result_path = os.path.join(out, os.path.splitext(example)[0], "result.json")
    with open(result_path, encoding="utf-8") as handle:
        return json.load(handle)


def total_energy(result: dict) -> float:
    return float(result["energy_hartree"])


def correlation_energy(result: dict) -> float:
    """K2 的 `correlation` 块里相关能叫 `e2_hartree`（总能量是 `e_total_hartree`）。"""
    block = result.get("correlation")
    if not block:
        raise RuntimeError("result has no correlation block")
    return float(block["e2_hartree"])


#: 一个「近似路径 × 参考体系」的比对条目。
#:
#: `kind` 是 `numerics.yaml` 里 `equivalence_tolerance` 的键；`quantity` 决定比哪个量
#: （总能量还是相关能）。`implemented = False` 的条目会作为报告行打出来，不参与判红。
CASES = [
    {
        "name": "RI-MP2 vs canonical MP2 (H2O / def2-SVP)",
        "kind": "ri",
        "quantity": "correlation",
        "reference": "h2o_mp2_def2svp.yaml",
        "approximate": "h2o_ri_mp2_def2svp.yaml",
        "implemented": True,
    },
    {
        "name": "pseudospectral vs analytic (IN-03)",
        "kind": "pseudospectral",
        "quantity": "total",
        "reference": None,
        "approximate": None,
        "implemented": False,
        "reason": "IN-03 (pseudospectral Coulomb/exchange) is not implemented; "
        "`options.numerics` is still rejected by the engine",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="只打印差异表，不因超差而失败")
    args = parser.parse_args()

    tolerances = load_tolerances()
    qwb = find_qwb()
    print(f"qwb = {' '.join(qwb)}")
    print(f"容差（来自 qwb-data/numerics/numerics.yaml）：{tolerances}")
    print()
    header = f"{'case':52s} {'kind':14s} {'reference':>18s} {'approximate':>18s} {'|diff|':>10s} {'tol':>8s}  verdict"
    print(header)
    print("-" * len(header))

    failures = 0
    with tempfile.TemporaryDirectory() as workdir:
        for case in CASES:
            if not case["implemented"]:
                print(
                    f"{case['name']:52s} {case['kind']:14s} {'-':>18s} {'-':>18s} "
                    f"{'':>10s} {tolerances.get(case['kind'], 0.0):8.1e}  not implemented"
                )
                print(f"    reason: {case['reason']}")
                continue
            reference = run_example(qwb, case["reference"], workdir)
            approximate = run_example(qwb, case["approximate"], workdir)
            pick = correlation_energy if case["quantity"] == "correlation" else total_energy
            reference_value = pick(reference)
            approximate_value = pick(approximate)
            difference = abs(approximate_value - reference_value)
            tolerance = tolerances[case["kind"]]
            ok = difference <= tolerance
            if not ok:
                failures += 1
            print(
                f"{case['name']:52s} {case['kind']:14s} {reference_value:18.12f} "
                f"{approximate_value:18.12f} {difference:10.3e} {tolerance:8.1e}  "
                f"{'ok' if ok else 'OVER'}"
            )

    print()
    if failures and not args.report:
        print(f"FAIL: {failures} case(s) exceed the declared equivalence tolerance")
        return 1
    if failures:
        print(f"note: {failures} case(s) exceed the declared tolerance (--report, not failing)")
    else:
        print("ok: every implemented path is within its declared equivalence tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
