"""数据目录（GUI 的"下拉框从哪来"）：只读 `qwb-data/registry.yaml`，不复制任何数值。

为什么要这一层而不是在 GUI 里写死 `["sto-3g", "def2-svp", …]`：

* 基组/泛函/网格的名字是**数据**，注册表才是唯一真源（§8.6）；写死在界面里，
  数据加一条界面就落后一次，而且用户会以为能选的东西就是界面里那几条；
* 反过来，能不能跑某种作业是**实现**：`b3lyp` 在 `qwb-data` 里根本不存在
  （`requires.laplacian = true` 被 K1 的 phase-1 语义门禁拒绝，见 status.yaml T3），
  所以下拉框里不该出现它 —— 这一层天然保证"界面上能选的就是数据里有的"。

`hf` 是唯一不在注册表里的方法名：K2 规定 HF 用 `method_name = "hf"`，而
`qwb-data/functional/` 里没有 `hf.yaml`（HF 不是泛函数据）。这里是**唯一**把它加进
方法列表的地方，并把它排在第一位（多数教学场景从 HF 开始）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runner import RunnerError, find_data_root
from .yaml_min import YamlError
from .yaml_min import loads as yaml_loads

__all__ = [
    "GRID_LEVELS",
    "JOB_TYPES",
    "Catalog",
    "CatalogError",
]

#: K2 冻结的 `GridLevel` 取值（`JobInput.grid_level` 的小写拼写）。
#:
#: 这几个是**契约枚举**，不是数据：写在这里有 K2 撑腰，和"网格参数"无关
#: （半径/角度点数在 `qwb-data/grid/*.yaml` 里，由注册表加载）。
GRID_LEVELS: tuple[str, ...] = ("coarse", "normal", "fine", "veryfine")

#: K2 冻结的 `JobType` 取值。`freq` 在引擎侧还没接线（会以 exit 4 拒绝），
#: 所以界面只提供 sp / opt —— 不给用户一个必然失败的选项。
JOB_TYPES: tuple[str, ...] = ("sp", "opt")


class CatalogError(RuntimeError):
    """数据目录读不出来（registry.yaml 缺失/结构不符）。"""


@dataclass(frozen=True)
class Catalog:
    """`qwb-data` 里**实际注册**的基组、泛函、网格名。"""

    root: Path
    bases: tuple[str, ...]
    functionals: tuple[str, ...]
    grids: tuple[str, ...]

    @property
    def methods(self) -> tuple[str, ...]:
        """方法名下拉框的取值：`hf` + 注册表里的泛函。"""
        return ("hf", *self.functionals)

    @classmethod
    def open(cls, root: str | Path | None = None) -> "Catalog":
        """读 `registry.yaml`。`root=None` 时按 `QWB_DATA_ROOT` / 向上搜索定位。"""
        try:
            data_root = find_data_root(root)
        except RunnerError as exc:
            raise CatalogError(str(exc)) from exc
        manifest = data_root / "registry.yaml"
        if not manifest.is_file():
            raise CatalogError(f"{manifest} 不存在：这不是一个数据根")
        try:
            document = yaml_loads(manifest.read_text(encoding="utf-8"))
        except (OSError, YamlError) as exc:
            raise CatalogError(f"读不了 {manifest}: {exc}") from exc
        if not isinstance(document, dict):
            raise CatalogError(f"{manifest} 的顶层不是映射")

        def names(key: str) -> tuple[str, ...]:
            entries = document.get(key) or []
            if not isinstance(entries, list):
                raise CatalogError(f"{manifest} 的 {key} 不是列表")
            out = []
            for entry in entries:
                if not isinstance(entry, dict) or "name" not in entry:
                    raise CatalogError(f"{manifest} 的 {key} 条目缺 name: {entry!r}")
                out.append(str(entry["name"]))
            return tuple(out)

        bases = names("bases")
        functionals = names("functionals")
        grids = names("grids")
        if not bases or not functionals or not grids:
            raise CatalogError(
                f"{manifest} 至少要各有一个基组/泛函/网格（现在是 {len(bases)}/{len(functionals)}/{len(grids)}）"
            )
        return cls(root=data_root, bases=bases, functionals=functionals, grids=grids)

    def elements(self, basis: str) -> tuple[str, ...]:
        """某个基组覆盖哪些元素（`<basis>.yaml` 顶层的 `elements:`）。

        建分子时元素下拉框只该给**这个基组能算的元素** —— 否则用户能选 Cr，
        然后在提交时才拿到 `data load failed`。

        这里不整份 `yaml_loads` 基组文件：基组文件里有 `shells:` 那种"列表里套列表
        套映射"的结构，`yaml_min` 是**子集**解析器（project.md §2.4 的取舍），
        故意不覆盖那部分。只扫顶层 `elements:` 这一段的 `- X` 项，够用且不会假装
        自己解析了整份文件。
        """
        entry = self._basis_path(basis)
        try:
            text = entry.read_text(encoding="utf-8")
        except OSError as exc:
            raise CatalogError(f"读不了 {entry}: {exc}") from exc
        return _top_level_element_list(text, entry)

    def _basis_path(self, basis: str) -> Path:
        if basis not in self.bases:
            raise CatalogError(f"数据里没有基组 {basis!r}（有：{', '.join(self.bases)}）")
        manifest = yaml_loads((self.root / "registry.yaml").read_text(encoding="utf-8"))
        for entry in manifest["bases"]:
            if entry["name"] == basis:
                return self.root / str(entry["path"])
        raise CatalogError(f"registry.yaml 里找不到基组 {basis!r} 的 path")


def _top_level_element_list(text: str, source: Path) -> tuple[str, ...]:
    """抽出顶层 `elements:` 的项（块列表两种缩进都支持，也接受 `[H, C, N]` flow 写法）。"""
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not in_block:
            if line.rstrip() == "elements:":
                in_block = True
                continue
            # 容错：`elements: [H, C, N, O, F]`（flow 列表）。数据文件的约定是块列表
            # （见 qwb_studio.bse_data 的注释），但少一行解析就少一个"明明有却读不到"。
            if stripped.startswith("elements:") and "[" in stripped:
                inner = stripped.split("[", 1)[1].rsplit("]", 1)[0]
                values = [item.strip().strip("\"'") for item in inner.split(",")]
                out = [value for value in values if value]
                break
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            value = stripped[2:].strip().strip("\"'")
            if not value:
                raise CatalogError(f"{source} 的 elements 里有一项是空的")
            out.append(value)
            continue
        # 第一个不属于该块的键：结束
        break
    if not out:
        raise CatalogError(f"{source} 没有顶层 elements 列表")
    return tuple(out)
