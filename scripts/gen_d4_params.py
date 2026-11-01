#!/usr/bin/env python3
"""从 `_d4_staging/*.json`（由 `extract_d4_*.py` 程序化提取自上游，逐文件 sha256 已钉）生成
`qwb-data/dispersion/d4.yaml`（DA-06 的 D4 数据）。

**为什么分两步**：提取（`extract_*.py`）负责"从 LGPL 源码里把数字读出来并证明读对了"
（`scripts/check_d4_extract.py` 用独立解析器复核到 Δ = 0；`validate_d4_data.py` 把整条性质链
对官方 dftd4 的中间量验到 1e-7）。生成（本脚本）负责"把那些数字排成 K1 形状的数据文件"。
这一步**不引入任何新数字** —— 所有值都能追到 `_d4_staging/` 里的某个键。

数据文件刻意**不存** reference C6（上游也不存：它在 `d4.f90:221-235` 由 `alphaiw` 做
`3/π·trapzd` 得到），而是存 `alpha_iw` / `alpha_iw_eeq` 与频率网格、`trapzd` 权重、`thopi`，
让 Rust 侧按上游同样的积分自己算 —— 这样"电荷相关的 C6"是从参数推出来的，不是我们写死的表。
逐层的对照物是 `qwb-scf/tests/reference/d4_properties.csv`（官方 CN/电荷/极化率/C6）与
`d4_dispersion.csv`（官方能量/梯度）。

顶层 `sha256` 由 `qwb-stamp-data` 打（`qwb-registry` 的规范化规则），本脚本不管；`--check`
是**语义**比较（逐项浮点相等，容差 0），与排版无关。

用法：
    python3 scripts/gen_d4_params.py            # 生成/覆盖 qwb-data/dispersion/d4.yaml
    python3 scripts/gen_d4_params.py --check    # 与磁盘上的文件做语义比较（需要 PyYAML）
    qwb-stamp-data qwb-data/dispersion/d4.yaml  # 打 sha256（Rust 侧）
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(ROOT, "qwb-data", "dispersion", "_d4_staging")
# 目的地**暂时**是 staging：`qwb-data/dispersion/*.yaml` 的存在意味着它已经能被 registry
# 加载，而 D4 的 `Dispersion` 类型（EEQ / alpha(iw) / ATM 那些块）还没进 K1 schema。
# 等 `qwb-registry` 的 D4 类型落地、`qwb-gen-schemas` 重新生成、`qwb-stamp-data` 通过之后，
# 把这一行改成 `qwb-data/dispersion/d4.yaml`（同一次提交里加 registry.yaml 的条目）。
OUTPUT = os.path.join(ROOT, "qwb-data", "dispersion", "_d4_staging", "d4-staged.yaml")

REFERENCE = os.path.join(STAGING, "reference_systems.json")
EEQ = os.path.join(STAGING, "eeq2019.json")
ELEMENTS = os.path.join(STAGING, "eeq_elements.json")
DAMPING = os.path.join(STAGING, "damping_parameters.json")
MANIFEST = os.path.join(STAGING, "raw_manifest.json")

MAX_Z = 36
FUNCTIONALS = ["b3lyp", "pbe", "pbe0", "hf", "m06-l"]
UNSUPPORTED = ["m06-2x"]

NOTES = (
    "DFT-D4 (Caldeweyher et al., J. Chem. Phys. 150, 154122 (2019)) parameters for H-Kr, "
    "extracted programmatically from the pinned upstream sources (dftd4 v4.3.1, multicharge "
    "v0.6.0; every file listed in provenance with its sha256). D4 differs from D3 in three "
    "ways that the loader must honour: (1) the damping is rational WITHOUT the zero-damping "
    "alp exponent -- two-body E = sum_{i<j} C6_ij [s6/(R^6+r0^6) + s8*rrij/(R^8+r0^8)] with "
    "r0 = a1*sqrt(rrij)+a2 and rrij = 3*r4r2_i*r4r2_j; alp is only the ATM three-body "
    "exponent; (2) C6 is NOT a tabulated element-pair value -- it is built at setup from the "
    "per-reference dynamic polarizabilities alpha(i*w) by a 3/pi*trapzd Casimir-Polder "
    "integration over the 23 frequencies stored here, so this file stores alpha_iw (not C6); "
    "(3) every reference weight carries the charge scaling zeta(ga, eta_i*gc, q_ref+zeff_i, "
    "q_i+zeff_i), so the model needs EEQ charges -- there are no s6q/s8q/a1q/a2q coefficients "
    "anywhere upstream. The ATM term uses C6 rebuilt with q = 0 (upstream disp.f90:139-141); "
    "using the molecular charges there is off by up to 30 percent. M06-2X has no D4 parameters "
    "upstream (v4.3.1 knows no such functional), so it is recorded under unsupported_functionals "
    "rather than given invented values."
)


def derived_aatoau() -> float:
    """mctc-lib 的 `aatoau = 1/(bohr*1e10)`，bohr 由 CODATA-2018 常数算出（convert.f90:25-35）。

    不在这里重写那四个 CODATA 常数，而是复用 `_d4_staging/validate_d4_data.py` 里已经用于
    验证的那份推导（`mctc_aatoau()`）—— 一个来源，避免两处各写一份近似值。
    """
    import importlib.util  # noqa: PLC0415

    path = os.path.join(STAGING, "validate_d4_data.py")
    if not os.path.isfile(path):
        raise SystemExit(f"gen_d4_params: {path} missing (it carries the CODATA-2018 aatoau derivation)")
    spec = importlib.util.spec_from_file_location("_d4_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "mctc_aatoau"):
        raise SystemExit("gen_d4_params: validate_d4_data.py lost its mctc_aatoau() helper")
    return float(module.mctc_aatoau())


def load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def scalar(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise SystemExit(f"gen_d4_params: `{where}` has no key `{key}`")
    return mapping[key]


def build() -> dict:
    reference = load(REFERENCE)
    eeq = load(EEQ)
    elements = load(ELEMENTS)["elements"]
    damping = load(DAMPING)["functionals"]
    manifest = load(MANIFEST)["files"]

    elements = sorted(elements, key=lambda item: item["Z"])
    if [item["Z"] for item in elements] != list(range(1, MAX_Z + 1)):
        raise SystemExit("gen_d4_params: eeq_elements.json must cover Z = 1..36")
    eeq_elements = {item["Z"]: item for item in sorted(eeq["elements"], key=lambda item: item["Z"])}
    refs = {item["Z"]: item for item in sorted(reference["elements"], key=lambda item: item["Z"])}
    cn = reference["cn_parameters"]
    weighting = reference["weighting"]
    integration = reference["c6_reference"]["integration"]
    scaling = reference["c6_reference"]["polarizability_scaling"]
    ga = scalar(scaling, "ga", "c6_reference.polarizability_scaling")
    gc = scalar(scaling, "gc", "c6_reference.polarizability_scaling")
    thopi = scalar(reference["c6_reference"], "thopi", "c6_reference")
    aatoau = derived_aatoau()

    data: dict = {}
    data["name"] = "d4"
    data["version"] = "1.0.0"
    data["kind"] = "d4"
    data["notes"] = NOTES
    data["citation"] = [
        {
            "title": "A generally applicable atomic-charge dependent London dispersion correction",
            "doi": "10.1063/1.5090222",
        },
        {
            "title": "Extension of the D3 dispersion coefficient model",
            "doi": "10.1063/1.4993215",
        },
    ]
    data["license"] = {
        "spdx": "LGPL-3.0-or-later",
        "commercial_ok": True,
        "redistribute_ok": True,
    }
    data["created"] = "2026-11-10T00:00:00Z"
    data["elements"] = [item["symbol"] for item in elements]

    data["constants"] = {
        "aatoau": float(aatoau),
        "thopi": float(thopi),
        "ga": float(ga),
        "gc": float(gc),
        "wf": float(scalar(weighting, "wf", "weighting")),
    }

    # EEQ 的半径**不做** CN/电荷缩放（multicharge 的 eeq2019 模型里没有 kcnrad），
    # 所以这里不写 kcnrad 字段；D4 的 CN 用 `coordination_number.radius_scaling`（4/3）。
    data["eeq"] = {
        "model": "eeq2019",
        "chi": [float(eeq_elements[z]["chi"]) for z in range(1, MAX_Z + 1)],
        "eta": [float(eeq_elements[z]["eta"]) for z in range(1, MAX_Z + 1)],
        "kcnchi": [float(eeq_elements[z]["kcnchi"]) for z in range(1, MAX_Z + 1)],
        "rad": [float(eeq_elements[z]["rad"]) for z in range(1, MAX_Z + 1)],
    }

    data["coordination_number"] = {
        "kcn": float(scalar(cn, "kcn", "cn_parameters")),
        "cutoff_bohr": float(scalar(cn, "cutoff_bohr", "cn_parameters")),
        "norm_exp": float(scalar(cn, "norm_exp", "cn_parameters")),
        "radius_scaling": float(scalar(cn, "radius_scaling", "cn_parameters")),
        "k4": float(scalar(cn, "k4", "cn_parameters")),
        "k5": float(scalar(cn, "k5", "cn_parameters")),
        "k6": float(scalar(cn, "k6", "cn_parameters")),
    }

    data["frequency_grid"] = {
        "hartree": [float(value) for value in integration["frequencies_hartree"]],
        "trapzd_weights": [float(value) for value in integration["weights"]],
    }

    reference_block = []
    for z in range(1, MAX_Z + 1):
        source = refs[z]
        systems = []
        for item in source["references"]:
            systems.append(
                {
                    "ref": int(item["ref"]),
                    "refsys": int(item["refsys"]),
                    "refcovcn": float(item["refcovcn"]),
                    "clsq": float(item["clsq"]),
                    "ascale": float(item["ascale"]),
                    "alpha_iw": [float(value) for value in item["alphaiw"]],
                    "alpha_iw_eeq": [float(value) for value in item["alphaiw_eeq"]],
                }
            )
        reference_block.append(
            {
                "element": source["symbol"],
                "r4_over_r2": float(source["r4_over_r2"]),
                "r4r2": float(source["r4r2"]),
                "pauling_en": float(source["pauling_en"]),
                "chemical_hardness": float(source["chemical_hardness"]),
                "zeff": float(source["zeff"]),
                "covalent_rad_2009_angstrom": float(source["covalent_rad_2009_angstrom"]),
                "reference_systems": systems,
            }
        )
    data["reference"] = reference_block

    damping_block = {}
    s9_values = set()
    alp_values = set()
    for name in FUNCTIONALS:
        values = damping[name]
        s9_values.add(float(values["s9"]))
        alp_values.add(float(values["alp"]))
        damping_block[name] = {
            "s6": float(values["s6"]),
            "s8": float(values["s8"]),
            "a1": float(values["a1"]),
            "a2": float(values["a2"]),
            "doi": values["doi"],
        }
    if len(s9_values) != 1 or len(alp_values) != 1:
        raise SystemExit("gen_d4_params: s9/alp differ between functionals; the ATM block cannot be shared")
    data["damping_parameters"] = damping_block

    data["unsupported_functionals"] = {
        name: {
            "reason": damping[name]["reason"],
            "checked_files": damping[name]["checked_files"],
        }
        for name in UNSUPPORTED
    }

    data["dispersion"] = {
        "cutoff_bohr": float(reference["cn_parameters"]["cutoffs_bohr"]["disp2"]),
        "atm": {
            "enabled_by_default": True,
            "s9": s9_values.pop(),
            "alp": alp_values.pop(),
            "charge_for_c6": 0.0,
        },
    }

    provenance = []
    for entry in manifest:
        # `raw_manifest.json` 的字段是 local_path / repo / tag / commit / repo_path / url /
        # bytes / sha256；数据文件里只留「文件身份 + 逐字节来源」，`file` 用 upstream 的相对路径。
        provenance.append(
            {
                "file": f"{entry['repo']}/{entry['repo_path']}",
                "url": entry["url"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
            }
        )
    # 每个仓库的 tag/commit 从 manifest 自己汇总（不手写哈希：写错一个字符就是伪造来源）。
    pinned: dict[str, str] = {}
    for entry in manifest:
        pinned[entry["repo"]] = f"{entry['tag']} / {entry['commit']}"
    data["provenance"] = {"upstream": provenance, "pinned": dict(sorted(pinned.items()))}
    return data


# ── 一个够用的 YAML 发射器（文件随后会被 `qwb-stamp-data` 重新序列化，排版只要可读可 diff）──
def emit_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SystemExit("gen_d4_params: non-finite float in output")
        text = repr(value)
        if "e" in text or "E" in text:
            mantissa, _, exponent = text.partition("e" if "e" in text else "E")
            if "." not in mantissa:
                mantissa += ".0"
            text = f"{mantissa}e{exponent}"
        elif "." not in text:
            text += ".0"
        return text
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise SystemExit(f"gen_d4_params: unsupported scalar {type(value)}")


def emit(value, indent: int = 0, out: list[str] | None = None) -> list[str]:
    out = [] if out is None else out
    pad = "  " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                out.append(f"{pad}{key}:")
                emit(item, indent + 1, out)
            elif isinstance(item, list) and not item:
                out.append(f"{pad}{key}: []")
            elif isinstance(item, dict) and not item:
                out.append(f"{pad}{key}: {{}}")
            else:
                out.append(f"{pad}{key}: {emit_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item:
                first = True
                for key, sub in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    first = False
                    if isinstance(sub, (dict, list)) and sub:
                        out.append(f"{prefix}{key}:")
                        emit(sub, indent + 2, out)
                    elif isinstance(sub, list) and not sub:
                        out.append(f"{prefix}{key}: []")
                    else:
                        out.append(f"{prefix}{key}: {emit_scalar(sub)}")
            else:
                out.append(f"{pad}- {emit_scalar(item)}")
    else:
        out.append(f"{pad}{emit_scalar(value)}")
    return out


def semantic_diff(want, got, where: str = "") -> list[str]:
    problems = []
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f"{where}: expected mapping"]
        for key, value in want.items():
            if key not in got:
                if key == "sha256" and where == "":
                    continue  # 由 qwb-stamp-data 打
                problems.append(f"{where}.{key}: missing on disk")
                continue
            problems += semantic_diff(value, got[key], f"{where}.{key}")
        for key in got:
            if key not in want and not (key == "sha256" and where == ""):
                problems.append(f"{where}.{key}: unexpected on disk")
    elif isinstance(want, list):
        if not isinstance(got, list):
            return [f"{where}: expected list"]
        if len(want) != len(got):
            return [f"{where}: length {len(want)} vs {len(got)}"]
        for index, (a, b) in enumerate(zip(want, got)):
            problems += semantic_diff(a, b, f"{where}[{index}]")
    elif isinstance(got, datetime.datetime):
        text = got.isoformat().replace("+00:00", "Z")
        if want != text:
            problems.append(f"{where}: {want!r} vs {text!r}")
    elif isinstance(want, float):
        if not isinstance(got, (int, float)) or float(want) != float(got):
            problems.append(f"{where}: {want!r} vs {got!r}")
    else:
        if want != got:
            problems.append(f"{where}: {want!r} vs {got!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    data = build()
    body = "\n".join(emit(data)) + "\n"
    if args.check:
        import yaml  # noqa: PLC0415 —— 只有 --check 需要
        with open(OUTPUT, encoding="utf-8") as handle:
            on_disk = yaml.safe_load(handle)
        problems = semantic_diff(data, on_disk)
        if problems:
            print(f"{len(problems)} 处与生成器不一致：")
            for problem in problems[:20]:
                print(f"  - {problem}")
            return 1
        print(f"ok: {OUTPUT} matches the generator (semantic, tolerance 0)")
        return 0

    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)  # 旧文件带 sha256：整份重写，避免残留字段
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
    print(f"wrote {OUTPUT} ({len(body)} bytes)")
    print("下一步：qwb-core/target/debug/qwb-stamp-data.exe qwb-data/dispersion/d4.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
