# -*- coding: utf-8 -*-
"""从 CCCBDB 抓「模型化学 + 能量 + 几何」快照，写成 JSON（一次性抓取，之后不再联网）。

用法：python scripts/fetch_cccbdb_reference.py out.json

注意：CCCBDB 对高频请求回 429，所以要限速 + 退避重试（本脚本 2s/请求）。
"""
import json
import re
import sys
import time

import urllib.request

CAS = {"H2": "1333740", "H2O": "7732185", "CO": "630080", "C6H6": "71432", "CH4": "74828", "NH3": "7664417", "N2": "7727379"}
METHODS = {"HF": 1, "LSDA": 51, "PBEPBE": 61, "BLYP": 13, "B1B95": 55, "B3LYP": 8}
BASES = {"STO-3G": 20, "6-31G*": 1, "6-311G**": 21}

NUMBER = re.compile(r"-?\d+\.\d+")


USER_AGENT = "Mozilla/5.0 (qwb reference fetch; research)"


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


BASE = "https://cccbdb.nist.gov/"


def get_retry(url, tries=6):
    for k in range(tries):
        try:
            return get(url)
        except Exception as exc:  # noqa: BLE001
            wait = 3.0 * (2 ** k)
            print(f"   retry {k + 1}/{tries} after {wait:.0f}s ({exc})")
            time.sleep(wait)
    raise RuntimeError(f"give up on {url}")


def flat_cells(html):
    cells = []
    for tr in re.findall(r"(?is)<tr.*?</tr>", html):
        for td in re.findall(r"(?is)<td.*?</td>", tr):
            txt = re.sub(r"<[^>]+>", " ", td).replace("&nbsp;", " ").strip()
            txt = re.sub(r"\s+", " ", txt)
            cells.append(txt)
    return cells


def after(cells, label, numeric=True, start=0):
    """在扁平单元格序列里找 label 之后第一个纯数字单元格。"""
    for i in range(start, len(cells)):
        if cells[i].startswith(label):
            for j in range(i + 1, min(i + 6, len(cells))):
                c = cells[j].strip()
                if numeric:
                    if re.fullmatch(r"-?\d+\.\d+", c):
                        return float(c), j
                elif c:
                    return c, j
    return None, start


def extract(html):
    cells = flat_cells(html)
    name, _ = after(cells, "using model chemistry:", numeric=False)
    e0, _ = after(cells, "Energy at 0K")
    enuc, _ = after(cells, "Nuclear repulsion energy")
    # 只扫 "Cartesians" 之后的**第一张表**：后面那张 "Atom - Atom Distances" 矩阵的
    # 首列也是 "C1"/"O1" 这种标签，跟着三个数就是距离，混进来会把核排斥能算成好几倍。
    tail = html.split("Cartesians", 1)[1]
    m = re.search(r"(?is)<table.*?</table>", tail)
    geo = m.group(0) if m else ""
    atoms = []
    cells_geo = flat_cells(geo)
    for i, c in enumerate(cells_geo):
        if re.fullmatch(r"[A-Z][a-z]?\d+", c) and i + 3 < len(cells_geo):
            xyz = cells_geo[i + 1:i + 4]
            if all(re.fullmatch(r"-?\d+\.\d+", v.strip()) for v in xyz):
                atoms.append([c, float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return {"model_chemistry": name, "energy_0k_hartree": e0,
            "nuclear_repulsion_hartree": enuc, "cartesian_angstrom": atoms}


Z = {"H": 1, "C": 6, "N": 7, "O": 8}
BOHR = 0.529177210903


def enuc_of(atoms):
    total = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            zi = Z[atoms[i][0].rstrip("0123456789")]
            zj = Z[atoms[j][0].rstrip("0123456789")]
            d2 = sum((atoms[i][k] - atoms[j][k]) ** 2 for k in (1, 2, 3))
            total += zi * zj / (d2 ** 0.5)
    return total * BOHR


def repair(path):
    """重抓自检不过的条目（几何错位的那些）。"""
    data = json.load(open(path, encoding="utf-8"))
    for e in data["entries"]:
        atoms = [[a[0], float(a[1]), float(a[2]), float(a[3])] for a in e["cartesian_angstrom"]]
        bad = abs(enuc_of(atoms) - e["nuclear_repulsion_hartree"]) > 1e-4
        if not bad:
            continue
        print(f"repair {e['species']} {e['method']}/{e['basis']}")
        info = extract(get_retry(e["url"]))
        time.sleep(2.0)
        if abs(enuc_of([["".join(c for c in a[0] if c.isalpha()), float(a[1]), float(a[2]),
                         float(a[3])] for a in info["cartesian_angstrom"]])
               - info["nuclear_repulsion_hartree"]) < 1e-4:
            e["cartesian_angstrom"] = info["cartesian_angstrom"]
            e["energy_0k_hartree"] = info["energy_0k_hartree"]
            e["nuclear_repulsion_hartree"] = info["nuclear_repulsion_hartree"]
            print("   ok")
        else:
            print("   still bad")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    print("repaired in place:", path)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--repair":
        repair(sys.argv[2])
        return
    out_path = sys.argv[1] if len(sys.argv) > 1 else "cccbdb.json"
    data = {"source": "NIST CCCBDB Release 22 (May 2022), SRD 101",
            "url_base": "https://cccbdb.nist.gov/", "entries": []}
    for sp, cas in CAS.items():
        for mname, mcode in METHODS.items():
            for bname, bcode in BASES.items():
                url = f"energy3x.asp?method={mcode}&basis={bcode}&casno={cas}&charge=0"
                try:
                    info = extract(get_retry(BASE + url))
                except Exception as exc:  # noqa: BLE001
                    print(f"skip {sp} {mname}/{bname}: {exc}")
                    continue
                finally:
                    time.sleep(2.0)
                if not info["cartesian_angstrom"] or info["energy_0k_hartree"] is None:
                    print(f"empty {sp} {mname}/{bname}: {info}")
                    continue
                info.update({"species": sp, "casno": cas, "method": mname, "basis": bname,
                             "url": "https://cccbdb.nist.gov/" + url})
                data["entries"].append(info)
                print(f"{sp:5} {mname:7} {bname:9} E={info['energy_0k_hartree']:.6f} "
                      f"Enuc={info['nuclear_repulsion_hartree']:.6f} "
                      f"({info['model_chemistry']}) n={len(info['cartesian_angstrom'])}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    print("wrote", out_path, len(data["entries"]), "entries")


if __name__ == "__main__":
    main()
