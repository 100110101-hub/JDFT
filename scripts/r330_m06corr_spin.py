#!/usr/bin/env python3
"""r330 -- spin-resolved (zeta-dependent) M06 / M06-L / M06-2X *correlation*:
candidate formula + point-by-point numerical validation against libxc (PySCF).

Sources actually read (see also the header of the final report):
  * libxc 4.3.4  src/mgga_c_m06l.c  -- parameter tables (par_m06l/par_m06/
    par_m062x) and the LSDA auxiliary functional XC_LDA_C_PW_MOD.
      https://gitlab.com/libxc/libxc/-/raw/4.3.4/src/mgga_c_m06l.c
      (MPL-2.0, "This Source Code Form is subject to the terms of the Mozilla
       Public License, v. 2.0")
  * NWChem src/nwxc/nwxc_c_m06.F  (M06/M06-L/M06-2X correlation, same-spin
    nwxc_m06css + opposite-spin Stoll remainder) and src/nwxc/nwxc_c_vs98.F
    (the gtv4 part, same-spin per channel + opposite-spin total).
      https://raw.githubusercontent.com/nwchemgit/nwchem/master/src/nwxc/nwxc_c_m06.F
      https://raw.githubusercontent.com/nwchemgit/nwchem/master/src/nwxc/nwxc_c_vs98.F

Run from Windows:
  wsl -e bash -lc "cd /mnt/d/python_II/Jaguar && python3 scripts/r330_m06corr_spin.py"

Dependencies: numpy + pyscf only.

Conventions
-----------
Everything is written for the energy *per volume* e = rho*eps, in the per-spin
reduced variables (no 2^(2/3) rescaling):

  x^2_s = sigma_ss / rho_s^(8/3)          ("chi_sigma")
  t_s   = tau_s   / rho_s^(5/3)           (tau_s = 1/2 sum_i |grad phi_i|^2)
  Z_s   = 2*(t_s - K'),  K' = (3/10)(6 pi^2)^(2/3) = 2^(2/3) K

libxc returns exc (per particle) plus vrho = de/drho_s, vsigma = de/dsigma,
vtau = de/dtau_s; forward-mode dual numbers with 7 partials
(rho_a, rho_b, sig_aa, sig_ab, sig_bb, tau_a, tau_b) give exactly those.
"""

from __future__ import annotations

import numpy as np
from pyscf.dft import libxc

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
PI = np.pi
K_FACTOR = 0.3 * (3.0 * PI * PI) ** (2.0 / 3.0)      # K  = (3/10)(3 pi^2)^(2/3)
K_PRIME = 0.3 * (6.0 * PI * PI) ** (2.0 / 3.0)       # K' = (3/10)(6 pi^2)^(2/3)
F_SECOND = 8.0 / (9.0 * (2.0 ** (4.0 / 3.0) - 2.0))  # f''(0) of PW92
F43M2 = 2.0 ** (4.0 / 3.0) - 2.0

PW92 = {  # identical to qwb-xc::gga::Pw92Set
    "para":  (0.0310907, 0.21370, [7.5957, 3.5876, 1.6382, 0.49294]),
    "ferro": (0.01554535, 0.20548, [14.1189, 6.1977, 3.3662, 0.62517]),
    "alpha": (0.0168869, 0.11125, [10.357, 3.6231, 0.88026, 0.49671]),
}

PARAMS = {  # libxc mgga_c_m06l.c: par_m06l / par_m06 / par_m062x
    "MGGA_C_M06_L": dict(
        gamma_ss=0.06, gamma_ab=0.0031, alpha_ss=0.00515088, alpha_ab=0.00304966,
        css=[5.349466e-01, 5.396620e-01, -3.161217e01, 5.149592e01, -2.919613e01],
        cab=[6.042374e-01, 1.776783e02, -2.513252e02, 7.635173e01, -1.255699e01],
        dss=[4.650534e-01, 1.617589e-01, 1.833657e-01, 4.692100e-04, -4.990573e-03, 0.0],
        dab=[3.957626e-01, -5.614546e-01, 1.403963e-02, 9.831442e-04, -3.577176e-03, 0.0],
    ),
    "MGGA_C_M06": dict(
        gamma_ss=0.06, gamma_ab=0.0031, alpha_ss=0.00515088, alpha_ab=0.00304966,
        css=[5.094055e-01, -1.491085e00, 1.723922e01, -3.859018e01, 2.845044e01],
        cab=[3.741539e00, 2.187098e02, -4.531252e02, 2.936479e02, -6.287470e01],
        dss=[4.905945e-01, -1.437348e-01, 2.357824e-01, 1.871015e-03, -3.788963e-03, 0.0],
        dab=[-2.741539e00, -6.720113e-01, -7.932688e-02, 1.918681e-03, -2.032902e-03, 0.0],
    ),
    "MGGA_C_M06_2X": dict(
        gamma_ss=0.06, gamma_ab=0.0031, alpha_ss=0.00515088, alpha_ab=0.00304966,
        css=[3.097855e-01, -5.528642e00, 1.347420e01, -3.213623e01, 2.846742e01],
        cab=[8.833596e-01, 3.357972e01, -7.043548e01, 4.978271e01, -1.852891e01],
        dss=[6.902145e-01, 9.847204e-02, 2.214797e-01, -1.968264e-03, -6.775479e-03, 0.0],
        dab=[1.166404e-01, -9.120847e-02, -6.726189e-02, 6.720580e-05, 8.448011e-04, 0.0],
    ),
}

# 5 spin blocks of qwb-core/crates/qwb-xc/tests/reference/spin_mgga_pointwise.csv
CSV_POINTS = [
    # rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b
    (1.0, 1.0, 3.174802103936e-01, 3.174802103936e-03, 3.174802103936e-01,
     3.445480800226, 3.445480800226),                                  # closed_shell_dense
    (1.0e-3, 1.0e-3, 6.349604207873e-14, 6.349604207873e-21, 6.349604207873e-14,
     3.014795700198e-05, 3.014795700198e-05),                          # closed_shell_sparse
    (1.2, 5.5e-1, 3.557881261808e-01, 0.0, 8.894703154519e-02,
     5.252560943125, 1.431111612625),                                  # weakly_polarized
    (1.5, 1.0e-8, 2.948334120900e-01, 0.0, 0.0,
     7.336642751834, 1.0e-12),                                         # fully_polarized
    (8.0e-1, 4.5e-1, 7.252482552520e-01, 2.220610202764e-02, 2.719680957195e-01,
     3.167173700201, 1.213974116377),                                  # large_gradient
]


# --------------------------------------------------------------------------
# forward-mode dual numbers, batched over N points, 7 partials
# --------------------------------------------------------------------------
NV = 7
I_RA, I_RB, I_SAA, I_SAB, I_SBB, I_TA, I_TB = range(NV)


class D:
    __slots__ = ("v", "d")

    def __init__(self, v, d=None):
        self.v = np.asarray(v, dtype=float)
        self.d = np.zeros((self.v.shape[0], NV)) if d is None else d

    @staticmethod
    def const(c, n):
        return D(np.full(n, float(c)))

    @staticmethod
    def var(v, index):
        d = np.zeros((v.shape[0], NV))
        d[:, index] = 1.0
        return D(v, d)

    def __add__(self, o):
        if isinstance(o, D):
            return D(self.v + o.v, self.d + o.d)
        return D(self.v + o, self.d.copy())

    __radd__ = __add__

    def __neg__(self):
        return D(-self.v, -self.d)

    def __sub__(self, o):
        if isinstance(o, D):
            return D(self.v - o.v, self.d - o.d)
        return D(self.v - o, self.d.copy())

    def __rsub__(self, o):
        return D(o - self.v, -self.d)

    def __mul__(self, o):
        if isinstance(o, D):
            return D(self.v * o.v, self.d * o.v[:, None] + self.v[:, None] * o.d)
        return D(self.v * o, self.d * o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, D):
            inv = 1.0 / o.v
            v = self.v * inv
            return D(v, (self.d - v[:, None] * o.d) * inv[:, None])
        return D(self.v / o, self.d / o)

    def __rtruediv__(self, o):
        inv = 1.0 / self.v
        v = o * inv
        return D(v, -(v[:, None] * self.d) * inv[:, None])

    def powf(self, p):
        return D(self.v ** p, (p * self.v ** (p - 1.0))[:, None] * self.d)

    def ln(self):
        return D(np.log(self.v), self.d / self.v[:, None])

    def neg_expm1(self):                     # -expm1(x) = 1 - exp(x)
        e = np.exp(self.v)
        return D(1.0 - e, -e[:, None] * self.d)


def sqrt(x):
    return x.powf(0.5)


# --------------------------------------------------------------------------
# LSDA correlation: modified PW92 with the zeta^4 interpolation (f_pw)
# --------------------------------------------------------------------------
def pw92_form(rs, name):
    a, alpha, betas = PW92[name]
    root = sqrt(rs)
    b = betas[0] * root + betas[1] * rs + betas[2] * rs * root + betas[3] * rs * rs
    one = D.const(1.0, rs.v.shape[0])
    log = (one + D.const(1.0, rs.v.shape[0]) / (D.const(2.0 * a, rs.v.shape[0]) * b)).ln()
    return (-2.0 * a) * (one + alpha * rs) * log


def f_pw(rs, zeta, opz, omz):
    """Modified PW92 correlation per particle, zeta^4 interpolation.

    ``opz = 1+zeta`` and ``omz = 1-zeta`` are passed in *separately* because
    for a nearly fully polarized point (rho_b/rho ~ 1e-8) forming ``1-zeta``
    from ``zeta`` loses ~8 digits; ``2*rho_s/rho`` is exact to machine
    precision.  libxc's maple code uses the same (opz, omz) representation.
    """
    one = D.const(1.0, rs.v.shape[0])
    ep = pw92_form(rs, "para")
    ef = pw92_form(rs, "ferro")
    alpha_c = -pw92_form(rs, "alpha")
    f = (opz.powf(4.0 / 3.0) + omz.powf(4.0 / 3.0) - 2.0) / F43M2
    z4 = zeta.powf(4.0)
    return ep * (one - f * z4) + ef * (f * z4) + alpha_c / F_SECOND * (f * (one - z4))


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def horner(c, u, n):
    poly = None
    for coef in reversed(c):
        poly = D.const(coef, n) if poly is None else poly * u + coef
    return poly


def b97_g(gamma, c, x2):
    """sum_i c_i u^i, u = gamma*x2/(1+gamma*x2)."""
    n = x2.v.shape[0]
    one = D.const(1.0, n)
    u = (gamma * x2) / (one + gamma * x2)
    return horner(c, u, n)


def gtv4(alpha, dvec, x2, z):
    """d0/g + (d1 x2 + d2 z)/g^2 + (d3 x2^2 + d4 x2 z + d5 z^2)/g^3,
    g = 1 + alpha (x2 + z)."""
    one = D.const(1.0, x2.v.shape[0])
    g = one + alpha * (x2 + z)
    g2 = g * g
    g3 = g2 * g
    return (dvec[0] / g
            + (dvec[1] * x2 + dvec[2] * z) / g2
            + (dvec[3] * x2 * x2 + dvec[4] * x2 * z + dvec[5] * z * z) / g3)


def fermi_d(x2, t):
    return (8.0 * t - x2) / (8.0 * t)


def fermi_d_corrected(x2, t):
    """Fermi_D * (1 - exp(-4 t^2/c^2)) with c = 1e-10 (inert for t > 1e-8)."""
    cnst = 1.0e-10
    return fermi_d(x2, t) * ((-4.0 / (cnst * cnst)) * t * t).neg_expm1()


# --------------------------------------------------------------------------
# candidate spin-resolved formula
# --------------------------------------------------------------------------
def m06_corr_e_vol(p, rho_a, rho_b, saa, sab, sbb, ta, tb):
    n = rho_a.v.shape[0]
    one = D.const(1.0, n)
    two = D.const(2.0, n)
    zero = D.const(0.0, n)
    pi34 = D.const(3.0 / (4.0 * PI), n)

    rho = rho_a + rho_b
    zeta = (rho_a - rho_b) / rho
    opz = (2.0 * rho_a) / rho          # 1+zeta, cancellation free
    omz = (2.0 * rho_b) / rho          # 1-zeta, cancellation free
    rs = (pi34 / rho).powf(1.0 / 3.0)

    def channel(rho_s, sig_s, tau_s):
        x2 = sig_s / rho_s.powf(8.0 / 3.0)
        t = tau_s / rho_s.powf(5.0 / 3.0)
        z = 2.0 * (t - K_PRIME)
        rs_s = (pi34 / rho_s).powf(1.0 / 3.0)
        e_lda_pol = rho_s * f_pw(rs_s, one, two, zero)   # zeta = 1, per volume
        return (e_lda_pol * (fermi_d_corrected(x2, t) * b97_g(p["gamma_ss"], p["css"], x2)
                             + fermi_d(x2, t) * gtv4(p["alpha_ss"], p["dss"], x2, z)),
                x2, z)

    e_a, x2a, za = channel(rho_a, saa, ta)
    e_b, x2b, zb = channel(rho_b, sbb, tb)

    e_same_lda = (rho_a * f_pw((pi34 / rho_a).powf(1.0 / 3.0), one, two, zero)
                  + rho_b * f_pw((pi34 / rho_b).powf(1.0 / 3.0), one, two, zero))
    e_stoll = rho * f_pw(rs, zeta, opz, omz) - e_same_lda

    xab = x2a + x2b
    zab = za + zb
    e_opp = e_stoll * (b97_g(p["gamma_ab"], p["cab"], xab)
                       + gtv4(p["alpha_ab"], p["dab"], xab, zab))
    return e_a + e_b + e_opp


# --------------------------------------------------------------------------
# libxc oracle
# --------------------------------------------------------------------------
def libxc_eval(libxc_id, rho_a, rho_b, saa, sab, sbb, ta, tb):
    npts = rho_a.shape[0]
    blocks = np.zeros((2, 6, npts))
    blocks[0, 0] = rho_a
    blocks[1, 0] = rho_b
    blocks[0, 5] = ta
    blocks[1, 5] = tb
    # rows 1..3 are the gradient *components*; sigma_ss = |g_s|^2,
    # sigma_ab = g_a . g_b.  Encode (saa, sab, sbb) with
    #   g_a = (sqrt(saa), 0, 0)
    #   g_b = (sab/sqrt(saa), sqrt(sbb - sab^2/saa), 0)
    ga = np.sqrt(np.maximum(saa, 0.0))
    blocks[0, 1] = ga
    gb1 = np.where(ga > 0.0, sab / np.where(ga > 0.0, ga, 1.0), 0.0)
    blocks[1, 1] = gb1
    blocks[1, 2] = np.sqrt(np.maximum(sbb - gb1 ** 2, 0.0))
    exc, vxc, _, _ = libxc.eval_xc(libxc_id, blocks, spin=1, deriv=1)
    return (rho_a + rho_b) * exc, vxc


def compare(libxc_id, samples, verbose=True):
    p = PARAMS[libxc_id]
    cols = list(zip(*samples))
    rho_a, rho_b, saa, sab, sbb, ta, tb = [np.asarray(c, dtype=float) for c in cols]
    n = rho_a.shape[0]

    e_ref, vxc = libxc_eval(libxc_id, rho_a, rho_b, saa, sab, sbb, ta, tb)
    e_mine = m06_corr_e_vol(p, D.var(rho_a, I_RA), D.var(rho_b, I_RB),
                            D.var(saa, I_SAA), D.var(sab, I_SAB), D.var(sbb, I_SBB),
                            D.var(ta, I_TA), D.var(tb, I_TB))

    dev = {
        "e": np.abs(e_mine.v - e_ref),
        "v_rho": np.maximum(np.abs(e_mine.d[:, I_RA] - vxc[0][:, 0]),
                            np.abs(e_mine.d[:, I_RB] - vxc[0][:, 1])),
        "v_sigma": np.maximum.reduce([
            np.abs(e_mine.d[:, I_SAA] - vxc[1][:, 0]),
            np.abs(e_mine.d[:, I_SAB] - vxc[1][:, 1]),
            np.abs(e_mine.d[:, I_SBB] - vxc[1][:, 2]),
        ]),
        "v_tau": np.maximum(np.abs(e_mine.d[:, I_TA] - vxc[3][:, 0]),
                            np.abs(e_mine.d[:, I_TB] - vxc[3][:, 1])),
    }
    if verbose:
        zeta = (rho_a - rho_b) / (rho_a + rho_b)
        sav = saa / rho_a ** (8.0 / 3.0)
        sbv = sbb / rho_b ** (8.0 / 3.0)
        tav = ta / rho_a ** (5.0 / 3.0)
        tbv = tb / rho_b ** (5.0 / 3.0)
        print(f"\n=== {libxc_id} ===")
        print(f"{'#':>2} {'rho_a':>9} {'rho_b':>9} {'zeta':>7} {'x2_a':>9} {'x2_b':>9}"
              f" {'t_a':>7} {'t_b':>7} | {'|de|':>9} {'|dv_rho|':>9}"
              f" {'|dv_sig|':>9} {'|dv_tau|':>9}")
        for i in range(n):
            print(f"{i:>2} {rho_a[i]:9.3e} {rho_b[i]:9.3e} {zeta[i]:7.3f} {sav[i]:9.2e}"
                  f" {sbv[i]:9.2e} {tav[i]:7.3f} {tbv[i]:7.3f} | {dev['e'][i]:9.2e}"
                  f" {dev['v_rho'][i]:9.2e} {dev['v_sigma'][i]:9.2e} {dev['v_tau'][i]:9.2e}")
    return {k: float(np.max(v)) for k, v in dev.items()}


def invented_points():
    """zeta = 0, .2, .5, .8, ~1 with small and large reduced gradients;
    tau = alpha*tau_uniform(rho_s), alpha >= 1 (physically sensible)."""
    k_unif = K_PRIME
    spec = [  # rho_tot, zeta, x2_a, x2_b, tau/tau_unif
        (1.0, 0.0, 1.0e-2, 1.0e-2, 1.05),
        (1.0, 0.0, 5.0, 5.0, 1.40),
        (0.7, 0.2, 1.0e-1, 2.0e-1, 1.02),
        (0.7, 0.2, 8.0, 3.0, 1.55),
        (1.4, 0.5, 3.0e-2, 4.0e-3, 1.10),
        (1.4, 0.5, 12.0, 0.5, 1.70),
        (0.5, 0.8, 0.2, 3.0e-2, 1.25),
        (0.5, 0.8, 20.0, 2.0, 1.60),
        (0.9, 0.999, 0.5, 1.0e-4, 1.30),
        (2.5, 0.0, 30.0, 30.0, 1.80),
        (0.35, 0.35, 1.5, 0.4, 1.15),
        (3.0, 0.65, 0.05, 6.0, 1.45),
    ]
    pts = []
    for rho_tot, zeta, x2a, x2b, alpha in spec:
        ra = 0.5 * rho_tot * (1.0 + zeta)
        rb = 0.5 * rho_tot * (1.0 - zeta)
        saa = x2a * ra ** (8.0 / 3.0)
        sbb = x2b * rb ** (8.0 / 3.0)
        sab = 0.5 * np.sqrt(saa * sbb)
        ta = alpha * k_unif * ra ** (5.0 / 3.0)
        tb = alpha * k_unif * rb ** (5.0 / 3.0)
        pts.append((ra, rb, saa, sab, sbb, ta, tb))
    return pts


def main():
    pts = CSV_POINTS + invented_points()
    print(f"{len(pts)} points = {len(CSV_POINTS)} from spin_mgga_pointwise.csv"
          f" + {len(invented_points())} invented")
    worst = {}
    for libxc_id in ("MGGA_C_M06_L", "MGGA_C_M06", "MGGA_C_M06_2X"):
        worst[libxc_id] = compare(libxc_id, pts)
    print("\n=========== max |deviation| over all points (energy per volume) ===========")
    print(f"{'component':<16} {'|de|':>12} {'|dv_rho|':>12} {'|dv_sigma|':>12} {'|dv_tau|':>12}")
    for k, m in worst.items():
        print(f"{k:<16} {m['e']:12.3e} {m['v_rho']:12.3e} {m['v_sigma']:12.3e} {m['v_tau']:12.3e}")


if __name__ == "__main__":
    main()
