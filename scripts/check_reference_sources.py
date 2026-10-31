"""DA-08 门禁：参考集 CSV 必须**逐行带来源**，且每个文件都有可重建的生成脚本。

判据原文：「每行有 `source_doi`/`source_url`；脚本可重建（`scripts/gen_*.py --check` 绿）；
缺来源即门禁失败」。本脚本负责前两句的**静态**部分（不需要 PySCF，Windows 门禁也能跑）：

1. 文件存在、表头最后两列就是 `source_doi,source_url`（加在末尾，旧读者按位置取列不受影响）；
2. 每一行的两列都非空，且 DOI 形如 `10.xxxx/...`、URL 形如 `https://...`；
3. DOI/URL 必须命中 [`reference_sources.PYSCF_DOI`] / `PYSCF_URL` 这个**白名单** ——
   不许出现 `TODO`、`n/a`、或者"某个 DOI 看起来挺像"；
4. 每个文件都有对应的生成脚本，且脚本里真有 `--check` 分支。

「数值能不能重算出来」由 `--check` 自己证明（需要 PySCF，在 WSL 里跑：见 DA-08 的账本条目）。
两条合起来才是完整的「可重建」：静态检查保证来源与脚本在位，`--check` 保证数字没被手改。
"""

import csv
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_sources  # noqa: E402

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
URL_PREFIX = "https://"
KNOWN_SOURCES = {
    (reference_sources.PYSCF_DOI, reference_sources.PYSCF_URL),
}


def check_family(relative: str) -> list[str]:
    problems = []
    path = os.path.join(reference_sources.ROOT, relative)
    if not os.path.isfile(path):
        return [f"{relative}: missing file"]
    rows = list(csv.reader(io.open(path, encoding="utf-8", newline="")))
    if not rows:
        return [f"{relative}: empty file"]
    header = rows[0]
    expected_tail = list(reference_sources.SOURCE_COLUMNS)
    if header[-len(expected_tail):] != expected_tail:
        problems.append(
            f"{relative}: header must end with {expected_tail}, got {header[-len(expected_tail):]}"
        )
        return problems
    doi_index = header.index("source_doi")
    url_index = header.index("source_url")
    for lineno, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        if len(row) != len(header):
            problems.append(f"{relative}:{lineno}: {len(row)} fields, header has {len(header)}")
            continue
        doi = row[doi_index].strip()
        url = row[url_index].strip()
        if not DOI_PATTERN.match(doi):
            problems.append(f"{relative}:{lineno}: malformed source_doi {doi!r}")
        if not url.startswith(URL_PREFIX):
            problems.append(f"{relative}:{lineno}: malformed source_url {url!r}")
        if (doi, url) not in KNOWN_SOURCES:
            problems.append(f"{relative}:{lineno}: unknown source {doi!r} / {url!r}")
    return problems


def check_generator(relative: str) -> list[str]:
    generator = reference_sources.GENERATORS[relative]
    path = os.path.join(reference_sources.ROOT, generator)
    if not os.path.isfile(path):
        return [f"{relative}: generator {generator} is missing"]
    text = io.open(path, encoding="utf-8").read()
    problems = []
    if '"--check"' not in text:
        problems.append(f"{relative}: generator {generator} has no --check branch")
    if "reference_sources" not in text:
        problems.append(f"{relative}: generator {generator} does not use reference_sources")
    return problems


def main() -> int:
    problems = []
    for relative in reference_sources.FAMILIES:
        problems += check_family(relative)
        problems += check_generator(relative)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        print(f"{len(problems)} problem(s) in {len(reference_sources.FAMILIES)} family file(s)")
        return 1
    total = 0
    for relative in reference_sources.FAMILIES:
        path = os.path.join(reference_sources.ROOT, relative)
        with io.open(path, encoding="utf-8", newline="") as handle:
            total += max(0, sum(1 for _ in handle) - 1)
    print(
        f"ok: {len(reference_sources.FAMILIES)} reference family files, "
        f"{total} rows, every row carries source_doi/source_url"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
