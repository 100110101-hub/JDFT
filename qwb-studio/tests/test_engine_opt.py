"""T8 的引擎侧判据（Python 绑定）：`opt` 作业真的能跑，且结果是终点上的量。

这一条与 `qwb-core/crates/qwb-engine/tests/t8_opt_jobtype.rs` 是同一个判据的两个入口
（Python API 与 Rust API）——K6 的 `qwb_engine` 是产品面，`JobType::Opt` 接上了它才
算"GUI 能选 opt"。

判据：
* `jobtype="opt"` 能跑通 H2/STO-3G/PBE（约 3–4 秒），返回 `scf_converged=True`；
* 返回的能量低于同几何的单点能（真的往下走了）；
* `gradient_hartree_per_bohr` 存在且最大分量 < 4.5e-4 Ha/Bohr（T3 判据）；
* opt 结果带轨道与偶极（3D 面板要用）。
"""

from __future__ import annotations

from pathlib import Path

import qwb_engine as qe

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "qwb-data"


def _stretched_h2() -> qe.Molecule:
    # 0.80 Å 的 H–H：明显长于平衡值，优化必须动
    return qe.Molecule(
        [
            qe.Atom("H", (0.0, 0.0, -0.40)),
            qe.Atom("H", (0.0, 0.0, 0.40)),
        ],
        charge=0,
        multiplicity=1,
    )


def _job(jobtype: str) -> qe.JobInput:
    return qe.JobInput(_stretched_h2(), "sto-3g", "pbe", jobtype=jobtype)


def test_opt_jobtype_runs_and_reports_the_final_state() -> None:
    engine = qe.Engine(str(DATA_ROOT))

    opt = engine.run(_job("opt"))
    assert opt.scf_converged is True
    assert opt.gradient_hartree_per_bohr is not None, "opt 结果必须带终点梯度"
    worst = max(
        abs(value) for row in opt.gradient_hartree_per_bohr for value in row
    )
    assert worst < 4.5e-4, f"终点最大力 {worst:.3e} Ha/Bohr 超过 T3 判据"
    assert opt.mo_energies_hartree is not None
    assert opt.mo_coeffs is not None
    assert opt.dipole_debye is not None

    # 同几何的单点能必须更高（否则优化没起作用）
    sp = engine.run(_job("sp"))
    assert opt.energy_hartree < sp.energy_hartree, (
        f"opt 能量 {opt.energy_hartree:.10f} 没有低于初值的单点能 "
        f"{sp.energy_hartree:.10f}"
    )
    # opt 的结果是**终点**上的：它不该与初值的单点能相等（但也不该差太多）
    assert abs(opt.energy_hartree - sp.energy_hartree) < 1.0e-2


def test_hf_method_name_is_accepted_by_the_binding() -> None:
    """K2 规定 HF 用 `method_name="hf"`；数据层没有 hf.yaml，引擎自己翻译。"""
    engine = qe.Engine(str(DATA_ROOT))
    h2 = qe.Molecule(
        [qe.Atom("H", (0.0, 0.0, -0.37)), qe.Atom("H", (0.0, 0.0, 0.37))],
        charge=0,
        multiplicity=1,
    )
    result = engine.run(qe.JobInput(h2, "sto-3g", "hf", grid_level="coarse"))
    # 外部参考：qwb-scf/tests/reference/h2_sto3g.csv 的 hf_total_energy = -1.116759
    assert abs(result.energy_hartree - (-1.116759)) < 1.0e-3
