"""RS-04 acceptance target: paramagnetic shielding by finite field (cc-pVTZ water).

No NMR reference implementation exists in this environment (the installed
PySCF has no `pyscf.prop`), so the target is generated self-consistently:
sigma^p is a mixed second derivative of the energy in two *imaginary*
one-electron perturbations, which a complex SCF computes directly.

    H(lam, mu) = H0 + i (lam W + mu V)
    W = (1/2) L_O          O = nucleus  (nucleus-centred common-origin gauge)
    V = L_A / |r-R_A|^3
    sigma^p_iso = -(1/3c^2) sum_alpha d2E/dlam dmu
    sigma_dia   = 1/(3c^2) sum P <1/r_A>      (int1e_rinv)

Both operators come from cint (`int1e_irp`, `int1e_prinvxp`) with the same
conventions as the Rust layer (`angmom::{angular_momentum_integrals,
hyperfine_integrals}`).

The 4-point stencil must scale the two perturbations *independently*: tying
both to one lambda makes E = E0 + lam^2 q(W +- V) and the two halves cancel
exactly (a real bug, recorded in docs/choices.md 八十一).

Usage: python3 scripts/pyscf_paramagnetic_reference.py [--basis cc-pvtz]
Output: qwb-core/crates/qwb-response/tests/reference/paramagnetic.csv
"""
import argparse
import io
import os

import numpy as np
from pyscf import gto, scf

C = 137.035999084
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "qwb-core", "crates", "qwb-response", "tests",
                   "reference", "paramagnetic.csv")
WATER = [("O", [0.0, 0.0, 0.117790]),
         ("H", [0.0, 0.755450, -0.471160]),
         ("H", [0.0, -0.755450, -0.471160])]


def main(basis):
    mol = gto.M(atom=[[s, list(c)] for s, c in WATER], basis=basis, verbose=0)
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
        solver.max_cycle = 200
        solver.run(verbose=0)
        return solver.e_tot

    lam = mu = 0.02
    lines = ["system,basis,nucleus_index,symbol,sigma_dia_ppm,"
             "sigma_p_ppm,sigma_total_ppm,lambda,mu"]
    for index, ra in enumerate(coords):
        shifted = [[s, list(c - ra)] for s, c in zip(symbols, coords)]
        m2 = gto.M(atom=shifted, basis=basis, unit="Bohr", verbose=0)
        sigma_dia = (np.einsum("ij,ij->", density, m2.intor("int1e_rinv"))
                     / (3.0 * C * C) * 1e6)
        irp = m2.intor("int1e_irp").reshape(3, 3, mol.nao, mol.nao)
        hyper_all = -m2.intor("int1e_prinvxp")
        isotropic = 0.0
        for component in range(3):
            other = [(component + 1) % 3, (component + 2) % 3]
            ang = 0.5 * (irp[other[0], other[1]] - irp[other[1], other[0]])
            hyper = hyper_all[component]
            isotropic += (
                energy((ang, hyper), (lam, mu))
                - energy((ang, hyper), (lam, -mu))
                - energy((ang, hyper), (-lam, mu))
                + energy((ang, hyper), (-lam, -mu))
            ) / (4.0 * lam * mu)
        sigma_p = -isotropic / 3.0 / (C * C) * 1e6
        lines.append(
            f"water,{basis},{index},{symbols[index]},{sigma_dia:.6f},"
            f"{sigma_p:.6f},{sigma_dia + sigma_p:.6f},{lam},{mu}"
        )
        print(f"  {symbols[index]}{index}: dia {sigma_dia:.3f}  para {sigma_p:.3f}  "
              f"total {sigma_dia + sigma_p:.3f} ppm")
    io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="cc-pvtz")
    main(parser.parse_args().basis)
