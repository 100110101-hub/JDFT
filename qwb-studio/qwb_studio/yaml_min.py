"""极简 YAML 读写（P-01 的 "YAML/JSON roundtrip"）。

project.md §2.4 的 Python 依赖白名单里**没有** PyYAML，而 Studio 的工程文件、
`task.json` 的 YAML 变体与 `.qwb` 内的清单都用 YAML 表达。这个模块只实现模型
层需要的那一小撮 YAML：

* 块状映射（`key: value` / `key:` + 更深的块）、块状序列（`- value` / `-` + 更深的块）；
* 标量：`null`/`~`、`true`/`false`、整数、浮点（含 `1e-05` 这种科学计数）、
  单/双引号字符串（`""` 与 `''` 转义、`\\n` `\\t` `\\\\` `\\"` 转义）；
* 空容器 `{}` 与 `[]`；
* 整行注释，以及未加引号标量里 ` #` 起的行内注释。

**不支持**（遇到会显式报错，绝不静默猜）：锚点/别名、多文档（`---` 之外的 `---`）、
流式非空集合（`[1, 2]` / `{a: 1}`）、多行折叠标量（`|` / `>`）、tab 缩进。
需要这些时请装 PyYAML —— 但本仓库的依赖白名单不允许，所以模型层保持在这个子集内。
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["YamlError", "dumps", "loads"]


class YamlError(ValueError):
    """YAML 文本不在本模块支持的子集内，或语法错误。"""


# 可以不加引号输出的键/字符串：字母或下划线开头，只含安全字符。
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-+/@=]*$")
_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "~"}
_FLOAT = re.compile(r"^[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$")
_SPECIAL_FLOAT = {".nan": float("nan"), ".inf": float("inf"), "-.inf": float("-inf")}
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "0": "\0"}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _quote(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return ".nan"
        if value == float("inf"):
            return ".inf"
        if value == float("-inf"):
            return "-.inf"
        return repr(value)
    if isinstance(value, str):
        if _PLAIN.match(value) and value.lower() not in _RESERVED:
            return value
        return _quote(value)
    raise TypeError(f"cannot serialize {type(value).__name__} to YAML")


def _anchor(text: str) -> str:
    return text if _PLAIN.match(text) and text.lower() not in _RESERVED else _quote(text)


def _inline(value: Any) -> str | None:
    """标量与空容器写成一行；非空容器返回 None（要另起块）。"""
    if isinstance(value, dict):
        return "{}" if not value else None
    if isinstance(value, list):
        return "[]" if not value else None
    if isinstance(value, tuple):
        raise TypeError("serialize tuples as lists first (use model_dump(mode='json'))")
    return _scalar(value)


def _emit(value: Any, indent: int, lines: list[str]) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"YAML mapping keys must be str, got {type(key).__name__}")
            head = f"{pad}{_anchor(key)}:"
            text = _inline(item)
            if text is None:
                lines.append(head)
                _emit(item, indent + 2, lines)
            else:
                lines.append(f"{head} {text}")
    elif isinstance(value, list):
        for item in value:
            text = _inline(item)
            if text is None:
                lines.append(f"{pad}-")
                _emit(item, indent + 2, lines)
            else:
                lines.append(f"{pad}- {text}")
    else:
        text = _inline(value)
        assert text is not None
        lines.append(pad + text)


def dumps(value: Any) -> str:
    """把 dict/list/标量树写成块状 YAML（末尾带换行；键顺序即插入顺序）。"""
    lines: list[str] = []
    _emit(value, 0, lines)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------


def _prepare(text: str) -> list[tuple[int, str, int]]:
    """→ [(缩进, 内容, 行号)]，去掉空行与整行注释。"""
    entries: list[tuple[int, str, int]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped in ("---", "..."):
            raise YamlError(f"document markers / multi-document YAML are not supported (line {number})")
        lead = line[: len(line) - len(line.lstrip())]
        if "\t" in lead:
            raise YamlError(f"tab indentation is not supported (line {number})")
        entries.append((len(lead), stripped, number))
    return entries


def _unquote(text: str) -> tuple[str, int]:
    """解析开头的引号标量 → (解码值, 消耗的字符数)。"""
    quote = text[0]
    out: list[str] = []
    i = 1
    while i < len(text):
        ch = text[i]
        if quote == '"' and ch == "\\":
            if i + 1 >= len(text) or text[i + 1] not in _ESCAPES:
                raise YamlError(f"unsupported escape sequence in {text!r}")
            out.append(_ESCAPES[text[i + 1]])
            i += 2
            continue
        if ch == quote:
            if quote == "'" and i + 1 < len(text) and text[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise YamlError(f"unterminated quoted scalar: {text!r}")


def _scalar_in(text: str, number: int) -> Any:
    text = text.strip()
    if not text:
        return None
    if text[0] in "\"'":
        value, end = _unquote(text)
        rest = text[end:].strip()
        if rest and not rest.startswith("#"):
            raise YamlError(f"trailing garbage after quoted scalar on line {number}: {rest!r}")
        return value
    if text[0] in "|>":
        raise YamlError(f"block scalar ({text[0]}) is not supported (line {number})")
    if text in ("{}", "[]"):
        return {} if text == "{}" else []
    if text[0] in "[{":
        raise YamlError(f"non-empty flow collection is not supported (line {number})")
    if text.startswith("&") or text.startswith("*") or text.startswith("!"):
        raise YamlError(f"anchors/aliases/tags are not supported (line {number})")
    # 未加引号标量里的行内注释。
    comment = text.find(" #")
    if comment >= 0:
        text = text[:comment].rstrip()
    if text in ("null", "Null", "NULL", "~"):
        return None
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    if text in _SPECIAL_FLOAT:
        return _SPECIAL_FLOAT[text]
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if _FLOAT.match(text):
        return float(text)
    return text


def _split_key(text: str, number: int) -> tuple[str, str]:
    if text[0] in "\"'":
        key, end = _unquote(text)
        rest = text[end:].lstrip()
        if not rest.startswith(":"):
            raise YamlError(f"expected ':' after quoted key on line {number}")
        return key, rest[1:]
    colon = text.find(":")
    if colon < 0:
        raise YamlError(f"expected 'key: value', got {text!r} (line {number})")
    return text[:colon].rstrip(), text[colon + 1 :]


def _is_item(text: str) -> bool:
    return text == "-" or text.startswith("- ")


_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


def _looks_like_key(text: str) -> bool:
    """`- name: sto-3g` 这种"序列项里内联映射头"的判定（不能把 `- http://x` 当成键）。"""
    colon = text.find(":")
    if colon <= 0:
        return False
    after = text[colon + 1 :]
    return bool(_KEY.match(text[:colon].rstrip())) and (after == "" or after[0] == " ")


def _parse_sequence(
    entries: list[tuple[int, str, int]], index: int, indent: int
) -> tuple[list[Any], int]:
    out: list[Any] = []
    while index < len(entries):
        cur_indent, text, number = entries[index]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise YamlError(f"unexpected indent {cur_indent} on line {number}")
        if not _is_item(text):
            break
        rest = text[1:].strip()
        if not rest:
            if index + 1 < len(entries) and entries[index + 1][0] > indent:
                value, index = _parse_block(entries, index + 1, entries[index + 1][0])
                out.append(value)
            else:
                out.append(None)
                index += 1
            continue
        if _looks_like_key(rest):
            # `- key: value` + 后续同层键（serde_yaml / 仓库里的数据文件都是这种写法）。
            sub = [(indent + 2, rest, number), *entries[index + 1 :]]
            value, consumed = _parse_mapping(sub, 0, indent + 2)
            out.append(value)
            index += consumed
            continue
        out.append(_scalar_in(rest, number))
        index += 1
    return out, index


def _parse_mapping(entries: list[tuple[int, str, int]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while index < len(entries):
        cur_indent, text, number = entries[index]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise YamlError(f"unexpected indent {cur_indent} on line {number}")
        if _is_item(text):
            raise YamlError(f"sequence item inside mapping without a key (line {number})")
        key, rest = _split_key(text, number)
        if key in out:
            raise YamlError(f"duplicate key {key!r} on line {number}")
        rest = rest.strip()
        if rest:
            out[key] = _scalar_in(rest, number)
            index += 1
            continue
        if index + 1 < len(entries) and entries[index + 1][0] > indent:
            out[key], index = _parse_block(entries, index + 1, entries[index + 1][0])
        elif (
            index + 1 < len(entries)
            and entries[index + 1][0] == indent
            and _is_item(entries[index + 1][1])
        ):
            # **缩进式之外的写法**：`serde_yaml` 把序列写在键的同一缩进上
            # ```yaml
            # atoms:
            # - symbol: O
            # ```
            # 这种"indentless sequence"在 Rust 侧写出的 input.yaml 里到处都是，
            # 不认它就没法读引擎自己的输入文件。
            out[key], index = _parse_sequence(entries, index + 1, indent)
        else:
            out[key] = None
            index += 1
    return out, index


def _parse_block(entries: list[tuple[int, str, int]], index: int, indent: int) -> tuple[Any, int]:
    if _is_item(entries[index][1]):
        return _parse_sequence(entries, index, indent)
    return _parse_mapping(entries, index, indent)


def loads(text: str) -> Any:
    """解析本模块子集内的 YAML 文本；空文档返回 None。"""
    entries = _prepare(text)
    if not entries:
        return None
    value, index = _parse_block(entries, 0, entries[0][0])
    if index != len(entries):
        _, _, number = entries[index]
        raise YamlError(f"unexpected content on line {number}")
    return value
