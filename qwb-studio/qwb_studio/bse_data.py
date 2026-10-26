"""从 Basis Set Exchange 取数并转录成 K1 的 `basis` / `ecp` YAML。

这里是**唯一**一份转录实现：`scripts/bse_to_qwb_data.py`、`scripts/bse_ecp_to_qwb_data.py`
与 `qwb_studio.datafetch` 都调它。理由很实际 —— 转录规则一旦有两份，就一定会漂移
（角动量拆分、`region` 白名单、`r^n` 约定、系数列序），而漂移的后果是**静默错数**。

下载与转录分开：
* `download_bse_json()` 只负责把 BSE 的 JSON 取回来（HTTP，`urllib`，不引入新依赖）；
* `transcribe_basis()` / `transcribe_ecp()` 只吃 JSON 字典，**不联网**，所以可以用仓库里的
  导出夹具做离线测试；
* `sha256` 不在这里写 —— K1 的摘要是"规范化内容"的摘要，由 Rust 的 `qwb data sign`
  （`qwb-registry` 的 `sha256::stamp`）统一打戳，Python 侧不复刻那份规则。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

__all__ = [
    "BSE_API",
    "ELEMENT_SYMBOLS",
    "atomic_number",
    "bse_json_url",
    "download_bse_json",
    "split_shells",
    "transcribe_basis",
    "transcribe_ecp",
]

#: BSE 的 JSON 导出接口（`?elements=` 用原子序数，逗号分隔）。
BSE_API = "https://www.basissetexchange.org/api/basis/{basis}/format/json/?elements={elements}"

ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
    20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
    29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
}

_ATOMIC_NUMBER = {symbol: z for z, symbol in ELEMENT_SYMBOLS.items()}


class TranscriptionError(RuntimeError):
    """BSE 导出与 K1 的形态不匹配（拒绝生成，不猜）。"""


def atomic_number(symbol: str) -> int:
    """元素符号 → 原子序数（只支持 H–Kr，与注册表的元素表一致）。"""
    try:
        return _ATOMIC_NUMBER[symbol]
    except KeyError as exc:  # pragma: no cover - 调用方总会先校验
        raise TranscriptionError(f"元素 {symbol!r} 不在 H–Kr 表里") from exc


def bse_json_url(basis: str, elements: list[int]) -> str:
    """BSE 的 JSON 导出 URL（`elements` 是原子序数列表）。"""
    joined = ",".join(str(int(z)) for z in elements)
    return BSE_API.format(basis=basis, elements=joined)


def download_bse_json(basis: str, elements: list[int], *, timeout: float = 60.0) -> dict:
    """从 BSE 取一个基组/ECP 的 JSON。

    只做 HTTP；**不做**任何形态假设（转录在 `transcribe_*` 里做）。
    加 `User-Agent` 是因为 BSE 前置的 CDN 对没有 UA 的请求会挑食。
    """
    url = bse_json_url(basis, elements)
    request = urllib.request.Request(
        url, headers={"User-Agent": "qwb-datafetch/1.0 (+https://github.com/100110101-hub/JDFT)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 固定 https 域名
            payload = response.read()
    except urllib.error.URLError as exc:
        raise TranscriptionError(f"下载失败 {url}: {exc}") from exc
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"{url} 返回的不是 JSON: {exc}") from exc
    if isinstance(data, dict) and data.get("error"):
        raise TranscriptionError(f"BSE 报错（{url}）：{data.get('message')}")
    if not isinstance(data, dict) or "elements" not in data:
        raise TranscriptionError(f"{url} 的返回里没有 elements")
    return data


# ---------------------------------------------------------------------------
# 基组
# ---------------------------------------------------------------------------


def _num(text: str) -> float:
    return float(text)


def _fmt(value: float) -> str:
    """最短的精确往返表示（YAML 里人能读、且不丢精度）。"""
    return repr(float(value))


def split_shells(entry: dict) -> list[dict]:
    """把一个 BSE `electron_shell` 拆成 K1 的 Shell 列表。"""
    if entry.get("function_type") not in ("gto", "gto_spherical"):
        raise TranscriptionError(f"unsupported function_type: {entry.get('function_type')!r}")
    # BSE 的 `region` 只是个标签，不是"这条壳层被裁剪过"的证据：
    #   * `valence`       —— ECP 基组（LANL2DZ 之类）全这么标；
    #   * `polarization`  —— Pople 系列（6-311G(d,p)）把极化函数单列成这样；
    #   * `unknown`       —— 老导出。
    # K1 的 `Shell` 没有"区域"概念，基组 YAML 收录的就是**整个**基组，所以上面三个都按整壳层
    # 转录。真正表示"导出被裁剪"的是 `core` / `semicore`（只给了部分壳层），必须显式拒绝 ——
    # 那种导出放进注册表会得到"少了一批壳层"的基组，而且不会报错。
    region = entry.get("region")
    if region not in (None, "", "valence", "polarization", "unknown"):
        raise TranscriptionError(f"unsupported region: {region!r} (only full-range/valence shells)")

    exps = [_num(e) for e in entry["exponents"]]
    columns = [[_num(c) for c in column] for column in entry["coefficients"]]
    angular = [int(l) for l in entry["angular_momentum"]]

    for column in columns:
        if len(column) != len(exps):
            raise TranscriptionError("coefficient column length != exponent count")

    shells: list[dict] = []
    if len(columns) == len(angular):
        # 业界最常见的形状：每个角动量一列系数（`sp` 复合壳层也长这样）。
        for l, column in zip(angular, columns):
            shells.append({"l": l, "exponents": exps, "columns": [column]})
    elif len(angular) == 1 and len(columns) > 1:
        # 通用收缩：同一个 l、多列系数。
        shells.append({"l": angular[0], "exponents": exps, "columns": columns})
    else:
        raise TranscriptionError(
            f"cannot map angular_momentum={angular} with {len(columns)} coefficient columns"
        )
    return shells


def _citation_lines(citations: list[dict]) -> list[str]:
    lines = ["citation:"]
    for citation in citations:
        lines.append(f"  - title: {json.dumps(citation['title'], ensure_ascii=False)}")
        if citation.get("doi"):
            lines.append(f"    doi: {json.dumps(citation['doi'])}")
        if citation.get("url"):
            lines.append(f"    url: {json.dumps(citation['url'])}")
    return lines


def transcribe_basis(spec: dict, data: dict) -> str:
    """BSE 的基组 JSON → K1 `basis` YAML 文本（不含 `sha256`）。

    `spec` 是 `qwb-data/sources.json` 里的一条 `transcribe` 段：名字/版本/引用/许可/
    `created`/`angular`/`normalization`/`role`/`contraction_type`。**所有元数据都来自数据清单**，
    函数本身不写死任何一个基组的名字或出处。
    """
    per_element: dict[str, list[dict]] = {}
    elements = data["elements"]
    for key in sorted(elements, key=int):
        z = int(key)
        symbol = ELEMENT_SYMBOLS.get(z)
        if symbol is None:
            raise TranscriptionError(f"element Z={z} outside the H..Kr table")
        shells: list[dict] = []
        for entry in elements[key].get("electron_shells") or []:
            shells.extend(split_shells(entry))
        shells.sort(key=lambda shell: shell["l"])
        if not shells:
            raise TranscriptionError(f"element Z={z} has no shells")
        per_element[symbol] = shells

    lines: list[str] = []
    add = lines.append
    add(f"name: {spec['name']}")
    add(f"version: {spec['version']}")
    add(f"notes: {json.dumps(spec['notes'], ensure_ascii=False)}")
    lines.extend(_citation_lines(spec["citations"]))
    add("license:")
    add(f"  spdx: {spec['license']['spdx']}")
    add(f"  commercial_ok: {str(spec['license']['commercial_ok']).lower()}")
    add(f"  redistribute_ok: {str(spec['license']['redistribute_ok']).lower()}")
    add(f"created: {spec['created']}")
    # `elements:` 必须是**块列表**（一行一项），不是 `[H, C, N, O, F]` 这种 flow 列表：
    # `qwb_studio.data_catalog` 为了不整份 YAML 解析（yaml_min 是子集）只按顶层
    # `elements:` 段落扫 `- X` 行，GUI 的元素下拉框就靠它。摘要按"解析后的内容"算，
    # 所以两种写法摘要相同 —— 也就是说写错形态**不会**被摘要钉拦住，只能靠这里的约定。
    add("elements:")
    for symbol in per_element:
        add(f"  - {symbol}")
    add(f"angular: {spec['angular']}")
    add(f"normalization: {spec['normalization']}")
    add(f"role: {spec['role']}")
    add("shells:")
    for symbol, shells in per_element.items():
        add(f"  - element: {symbol}")
        add("    shells:")
        for shell in shells:
            add(f"      - l: {shell['l']}")
            add("        exponents: [" + ", ".join(_fmt(e) for e in shell["exponents"]) + "]")
            add("        coefficients:")
            for column in shell["columns"]:
                add("          - [" + ", ".join(_fmt(c) for c in column) + "]")
            # 收缩类型**逐壳层**判定（不按整个基组一刀切）：一列系数 = 分段收缩，多列 = 通用收缩。
            # K1 的 Segmented 要求"每个收缩只有一个非零系数"，一刀切成 generalized 虽然能过校验，
            # 但会让"一列"的壳层被标成通用收缩 —— 语义不对，读数据的人会被误导。
            forced = spec.get("contraction_type")
            if forced:
                kind = forced
            else:
                kind = "segmented" if len(shell["columns"]) == 1 else "general"
            add(f"        contraction_type: {kind}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ECP
# ---------------------------------------------------------------------------


def transcribe_ecp(spec: dict, data: dict) -> str:
    """BSE 的 ECP JSON → K1 `ecp` YAML 文本（不含 `sha256`）。

    约定：**局域通道 = 导出里最大的 `l`**（EMSL/Stuttgart 的通行约定），其余 `l` 是投影；
    `r_power_convention` 固定写 `"r^n"`（BSE 的 `r_exponents` 就是这个约定，
    Stuttgart 原始表的 `r^(n-2)` 差 2 次，§8.2 要求显式声明）。
    """
    symbol = spec["element"]
    z = atomic_number(symbol)
    entry = data["elements"].get(str(z))
    if entry is None:
        raise TranscriptionError(f"导出文件里没有 {symbol}（Z={z}）")
    potentials = entry.get("ecp_potentials") or []
    if not potentials:
        raise TranscriptionError(f"{symbol} 在导出文件里没有 ecp_potentials")
    n_core = int(entry["ecp_electrons"])

    terms_by_l: dict[int, list[dict]] = {}
    for block in potentials:
        if block.get("ecp_type") != "scalar_ecp":
            raise TranscriptionError(f"unsupported ecp_type: {block.get('ecp_type')!r}")
        angular = [int(l) for l in block["angular_momentum"]]
        if len(angular) != 1:
            raise TranscriptionError(f"ecp block covers several l: {angular}")
        l = angular[0]
        powers = [int(n) for n in block["r_exponents"]]
        exponents = [_num(e) for e in block["gaussian_exponents"]]
        columns = block["coefficients"]
        if len(columns) != 1:
            # 每个 l 只有一组系数；多组会意味着"自旋相关 ECP"，那是另一个数据形态。
            raise TranscriptionError(f"l={l}: {len(columns)} coefficient columns (expected 1)")
        coefficients = [_num(c) for c in columns[0]]
        if not (len(powers) == len(exponents) == len(coefficients)):
            raise TranscriptionError(f"l={l}: powers/exponents/coefficients 长度不一致")
        if any(power < -2 or power > 2 for power in powers):
            raise TranscriptionError(f"l={l}: 出现意外的 r 幂次 {powers}（BSE 约定应为 -2..2）")
        terms_by_l[l] = [
            {"power": power, "exponent": exponent, "coefficient": coefficient}
            for power, exponent, coefficient in zip(powers, exponents, coefficients)
        ]

    local_l = max(terms_by_l)
    lines: list[str] = []
    add = lines.append
    add(f"name: {spec['name']}")
    add(f"version: {spec['version']}")
    add(f"notes: {json.dumps(spec['notes'], ensure_ascii=False)}")
    lines.extend(_citation_lines(spec["citations"]))
    add("license:")
    add(f"  spdx: {spec['license']['spdx']}")
    add(f"  commercial_ok: {str(spec['license']['commercial_ok']).lower()}")
    add(f"  redistribute_ok: {str(spec['license']['redistribute_ok']).lower()}")
    add(f"created: {spec['created']}")
    add(f"element: {symbol}")
    add(f"n_core_electrons: {n_core}")
    add(f"z_eff: {float(z - n_core)}")
    add('r_power_convention: "r^n"')
    add("local_part:")
    add(f'  id: "l={local_l} (BSE local channel: the highest l in the export)"')
    add("  terms:")
    for term in terms_by_l[local_l]:
        add(f"    - power: {term['power']}")
        add(f"      exponent: {_fmt(term['exponent'])}")
        add(f"      coefficient: {_fmt(term['coefficient'])}")
    add("nonlocal_parts:")
    for l in sorted(terms_by_l):
        if l == local_l:
            continue
        add(f"  - l: {l}")
        add(f'    id: "l={l} (BSE projector channel)"')
        add("    terms:")
        for term in terms_by_l[l]:
            add(f"      - power: {term['power']}")
            add(f"        exponent: {_fmt(term['exponent'])}")
            add(f"        coefficient: {_fmt(term['coefficient'])}")
    return "\n".join(lines) + "\n"


def transcribe(kind: str, spec: dict, data: dict) -> str:
    """按 `kind`（`basis` / `ecp`）分派到基组或 ECP 的转录。"""
    if kind == "basis":
        return transcribe_basis(spec, data)
    if kind == "ecp":
        return transcribe_ecp(spec, data)
    raise TranscriptionError(f"unknown data kind {kind!r}")


def write_text(path, text: str) -> None:
    """按 UTF-8 + LF 写文件（跨平台一致的字节）。"""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    with io.open(str(path), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
