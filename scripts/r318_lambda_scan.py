"""诊断：顺磁屏蔽参考（有限场）对步长 lambda 的敏感性。

上一轮我们的解析实现（耦合 CPHF）与参考差 1.3–1.4，且**两个核需要的因子不同**。
一个可能的解释是**参考自身的非线性误差**：`lambda = mu = 0.02` 对一个 −236 ppm 量级的响应
来说是非常大的场。这个脚本就测这一件事：把 lambda 从 0.02 缩到 0.00125，看参考值收敛到哪里。

只算水/STO-3G（我们有的基组），不做别的。
"""

import sys

import numpy as np
from pyscf import gto, scf

C = 137.035999084
WATER = [
    ("O", [0.0, 0.0, 0.117790]),
    ("H", [0.0, 0.755450, -0.471160]),
    ("H", [0.0, -0.755450, -0.471160]),
]


def load_yaml_basis(name):
    import yaml

    with open(f"qwb-data/basis/{name}.yaml", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    out = {}
    for entry in data["shells"]:
        shells = []
        for shell in entry["shells"]:
            l = shell["l"]
            exponents = np.array(shell["exponents"], dtype=float)
            for row in np.array(shell["coefficients"], dtype=float):
                shells.append([l] + [[float(e), float(c)] for e, c in zip(exponents, row)])
        out[entry["element"]] = shells
    return out


def main():
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    mol.basis = load_yaml_basis("sto-3g")
    mol.cart = False
    mol.build()
    h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    mf = scf.RHF(mol).run(conv_tol=1e-12, verbose=0)
    density = mf.make_rdm1()
    coords = mol.atom_coords(unit="Bohr")
    symbols = [a[0] for a in WATER]

    def energy(perturbations, scales):
        hb = h0.copy()
        for operator, scale in zip(perturbations, scales):
            hb = hb + 1j * scale * operator
        zeros = np.zeros_like(hb)
        solver = scf.GHF(mol)
        solver.get_hcore = lambda *args: np.block([[hb, zeros], [zeros, hb]])
        solver.conv_tol = 1e-11
        solver.max_cycle = 300
        solver.run(verbose=0)
        assert solver.converged, "complex GHF did not converge"
        return solver.e_tot

    print(f"{'nucleus':>8} {'lambda':>9} {'sigma_p (ppm)':>16}")
    for index, ra in enumerate(coords):
        shifted = [[s, list(c - ra)] for s, c in zip(symbols, coords)]
        m2 = gto.M(atom=shifted, basis="sto-3g", unit="Bohr", verbose=0)
        m2.basis = load_yaml_basis("sto-3g")
        m2.build()
        irp = m2.intor("int1e_irp").reshape(3, 3, mol.nao, mol.nao)
        hyper_all = -m2.intor("int1e_prinvxp")
        for lam in (0.02, 0.01, 0.005, 0.0025, 0.00125):
            isotropic = 0.0
            for component in range(3):
                other = [(component + 1) % 3, (component + 2) % 3]
                ang = 0.5 * (irp[other[0], other[1]] - irp[other[1], other[0]])
                hyper = hyper_all[component]
                isotropic += (
                    energy((ang, hyper), (lam, lam))
                    - energy((ang, hyper), (lam, -lam))
                    - energy((ang, hyper), (-lam, lam))
                    + energy((ang, hyper), (-lam, -lam))
                ) / (4.0 * lam * lam)
            sigma_p = -isotropic / 3.0 / (C * C) * 1e6
            print(f"{symbols[index]:>8} {lam:9.5f} {sigma_p:16.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
