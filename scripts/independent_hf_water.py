"""独立实现：H2O/STO-3G 的 Hartree-Fock，用来和 Rust 侧对照。

**状态（WIP，2026-10-23）**：已验证可用的部分——
  * S 矩阵（自己的 1D 高斯递推）：与 Rust 逐位相同（S[0,0] = 1.0000000000485125，
    S[2,2] = 0.9999999999800491）；
  * T 矩阵：T[0,0] = 29.003204066085107，与 Rust 完全相同；
  * (ss|ss) 类 2e 积分：(00|00) = 4.78506575227999，与 Rust 完全相同。
这三项一致说明「S/T/s-s 2e 存在共享错误」的假设不成立。

**尚未跑通**：p 型核吸引与 p 型 2e 的导数复合（脚本里 primitive_nuclear / primitive_2e
的递归复合仍有 bug，运行到该处会抛错）。下一步只需修这两处函数，就能得到完整的
独立 HF 能量，从而把「Rust 侧差 2.36 Ha」定位到具体矩阵元素。

下面是原始说明：

刻意**不复用** Rust 的任何算法：
  * 重叠/动能：自己的 1D 高斯递推；
  * 双电子积分：只实现 (ss|ss) 闭式，再对**中心做数值微分**生成 p 型
    （(x−A_x)e^{…} = (1/2α)∂_{A_x}e^{…}）；完全不碰 Hermite 展开；
  * 核吸引：用「极窄 s 函数压到核上」的双电子极限；
  * SCF：自己的 S^{-1/2} 与简单迭代。

输出中间量（S、T、V、若干张量元素、能量、轨道能），便于逐级定位与 Rust 的分叉点。
"""

import math
import sys

import numpy as np

PI = math.pi

# STO-3G（Hehre–Stewart–Pople 1969）；系数用于**已归一化**的原函数。
STO3G = {
    "H": [("s", [3.425250914, 0.6239137298, 0.168855404],
                [0.1543289673, 0.5353281423, 0.4446345422])],
    "O": [
        ("s", [130.7093214, 23.80886605, 6.443608313],
              [0.1543289673, 0.5353281423, 0.4446345422]),
        ("s", [5.033151319, 1.169596125, 0.38038896],
              [-0.09996722919, 0.3995128261, 0.7001154689]),
        ("p", [5.033151319, 1.169596125, 0.38038896],
              [0.155916275, 0.6076837186, 0.3919573931]),
    ],
}

BOHR = 1.0 / 0.529177210903


def norm(alpha, powers):
    """笛卡尔高斯 (x^a y^b z^c) e^{-αr²} 的归一化常数。"""
    l = sum(powers)
    num = (2.0 * alpha / PI) ** 1.5 * (4.0 * alpha) ** l
    den = 1.0
    for n in powers:
        den *= double_factorial(n)
    return math.sqrt(num / den)


def double_factorial(n):
    if n <= 0:
        return 1.0
    out = 1.0
    while n > 0:
        out *= n
        n -= 2
    return out


def boys(m_max, x):
    """F_m(x), m=0..m_max，向下递推。"""
    out = [0.0] * (m_max + 1)
    if x < 1e-12:
        for m in range(m_max + 1):
            out[m] = 1.0 / (2 * m + 1)
        return out
    out[m_max] = 0.0
    for m in range(m_max, 0, -1):
        out[m - 1] = (2 * x * out[m] + math.exp(-x)) / (2 * m - 1)
    return out


def ssss(alpha, A, beta, B, gamma, C, delta, D):
    """(ss|ss) 闭式。"""
    p = alpha + beta
    q = gamma + delta
    P = [(alpha * A[i] + beta * B[i]) / p for i in range(3)]
    Q = [(gamma * C[i] + delta * D[i]) / q for i in range(3)]
    ab2 = sum((A[i] - B[i]) ** 2 for i in range(3))
    cd2 = sum((C[i] - D[i]) ** 2 for i in range(3))
    pq = p + q
    rho = p * q / pq
    r2 = sum((P[i] - Q[i]) ** 2 for i in range(3))
    e = math.exp(-alpha * beta / p * ab2 - gamma * delta / q * cd2)
    return 2.0 * PI ** 2.5 / (p * q * math.sqrt(pq)) * e * boys(0, rho * r2)[0]


# ---- 一维重叠递推（自己的实现，不用 Hermite） ----
def overlap_1d(i, j, alpha, beta, A, B):
    p = alpha + beta
    P = (alpha * A + beta * B) / p
    # table[n][m] = 1D 重叠 <x^n on A | x^m on B>
    table = [[0.0] * (j + 1) for _ in range(i + 1)]
    table[0][0] = math.sqrt(PI / p) * math.exp(-alpha * beta * (A - B) ** 2 / p)
    for ii in range(0, i + 1):
        for jj in range(0, j + 1):
            if ii == 0 and jj == 0:
                continue
            if ii > 0:
                v = (P - A) * table[ii - 1][jj]
                if ii >= 2:
                    v += (ii - 1) / (2.0 * p) * table[ii - 2][jj]
                if jj >= 1:
                    v += jj / (2.0 * p) * table[ii - 1][jj - 1]
                table[ii][jj] = v
            else:
                v = (P - B) * table[ii][jj - 1]
                if jj >= 2:
                    v += (jj - 1) / (2.0 * p) * table[ii][jj - 2]
                table[ii][jj] = v
    return table[i][j]


def overlap_prim(pa, A, alpha, pb, B, beta):
    return math.prod(overlap_1d(pa[k], pb[k], alpha, beta, A[k], B[k]) for k in range(3))


def kinetic_prim(pa, A, alpha, pb, B, beta):
    """<pa| -1/2 ∇² |pb>，用逐轴二阶导处理 pb。"""
    total = 0.0
    for axis in range(3):
        n = pb[axis]
        # ∂²/∂x²[ u^n e^{-βu²} ] = [n(n-1)u^{n-2} - 2β(2n+1)u^n + 4β²u^{n+2}] e
        terms = []
        if n >= 2:
            terms.append((n - 2, n * (n - 1.0)))
        terms.append((n, -2.0 * beta * (2.0 * n + 1.0)))
        terms.append((n + 2, 4.0 * beta * beta))
        acc = 0.0
        for new_power, coef in terms:
            product = coef
            for ax in range(3):
                pb_new = list(pb)
                if ax == axis:
                    pb_new[ax] = new_power
                product *= overlap_1d(pa[ax], pb_new[ax], alpha, beta, A[ax], B[ax])
            acc += product
        total += acc
    return -0.5 * total


# ---- 双电子：s 型闭式 + 中心数值微分 ----
STEP = 1.0e-2


def deriv(f, axis, amount):
    """对某个中心坐标做 Richardson 中心差分。"""
    x = 0.0
    d = lambda h: (f(h) - f(-h)) / (2 * h)
    return (4.0 * d(STEP * 0.5) - d(STEP)) / 3.0 if amount else 0.0


def ssss_terms(alpha, A, beta, B, gamma, C, delta, D):
    """返回 (值, 各中心一阶导所需的中间量)。"""
    p = alpha + beta
    q = gamma + delta
    P = [(alpha * A[i] + beta * B[i]) / p for i in range(3)]
    Q = [(gamma * C[i] + delta * D[i]) / q for i in range(3)]
    ab2 = sum((A[i] - B[i]) ** 2 for i in range(3))
    cd2 = sum((C[i] - D[i]) ** 2 for i in range(3))
    pq = p + q
    rho = p * q / pq
    r2 = sum((P[i] - Q[i]) ** 2 for i in range(3))
    e = math.exp(-alpha * beta / p * ab2 - gamma * delta / q * cd2)
    pref = 2.0 * PI ** 2.5 / (p * q * math.sqrt(pq))
    f0 = boys(1, rho * r2)[0]
    f1 = boys(1, rho * r2)[1]
    return dict(p=p, q=q, P=P, Q=Q, pq=pq, rho=rho, r2=r2, e=e, pref=pref,
                f0=f0, f1=f1, ab2=ab2, cd2=cd2)


def ssss(alpha, A, beta, B, gamma, C, delta, D):
    t = ssss_terms(alpha, A, beta, B, gamma, C, delta, D)
    return t["pref"] * t["e"] * t["f0"]


def deriv_center(alpha, A, beta, B, gamma, C, delta, D, which, axis):
    """∂/∂(center_which[axis]) (ss|ss)，解析式（只支持一阶，s/p 足够）。"""
    t = ssss_terms(alpha, A, beta, B, gamma, C, delta, D)
    p, q, P, Q = t["p"], t["q"], t["P"], t["Q"]
    centers = {"a": A, "b": B, "c": C, "d": D}[which]
    # ∂E/∂x
    if which in ("a", "b"):
        coef = -alpha * beta / p
        other = B if which == "a" else A
        dE = 2.0 * coef * (centers[axis] - other[axis])
        sign = 1.0 if which == "a" else -1.0  # ∂P/∂A = α/p, ∂P/∂B = β/p
        dP = (alpha if which == "a" else beta) / p
        del sign
    else:
        coef = -gamma * delta / q
        other = D if which == "c" else C
        dE = 2.0 * coef * (centers[axis] - other[axis])
        dP = (gamma if which == "c" else delta) / q
    # ∂T/∂x = 2ρ(P_x−Q_x)(∂P_x/∂x)
    dT = 2.0 * t["rho"] * (P[axis] - Q[axis]) * dP
    # ∂F0/∂x = −F1·∂T/∂x
    dF0 = -t["f1"] * dT
    return t["pref"] * (t["e"] * dE * t["f0"] + t["e"] * dF0)


def _fd(f, x):
    """自适应步长的 Richardson 一阶导（步长按被积函数最紧的指数缩放）。"""
    h = f.step
    d = lambda s: (f(x + s) - f(x - s)) / (2.0 * s)
    return (4.0 * d(h * 0.5) - d(h)) / 3.0


def primitive_2e(pa, A, alpha, pb, B, beta, pc, C, gamma, pd, D, delta):
    """任意 s/p 的 (ab|cd)：对 (ss|ss) 做中心导数（递归复合，自适应步长）。"""
    centers0 = {"a": list(A), "b": list(B), "c": list(C), "d": list(D)}
    derivs = []
    for label, powers in (("a", pa), ("b", pb), ("c", pc), ("d", pd)):
        for axis in range(3):
            for _ in range(powers[axis]):
                derivs.append((label, axis))
    # 步长：最紧的高斯宽度的一个小分数
    step = 0.01  # 固定步长：太小会被舍入放大（每层导数都除以 h）

    def ev(pending, centers):
        if not pending:
            return ssss(alpha, centers["a"], beta, centers["b"],
                        gamma, centers["c"], delta, centers["d"])
        (label, axis), rest = pending[0], pending[1:]
        origin = centers[label][axis]

        def moved(t):
            c2 = {k: list(v) for k, v in centers.items()}
            c2[label][axis] = t
            return ev(rest, c2)

        # 5 点四阶中心差分
        raw = (moved(origin - 2 * step) - 8.0 * moved(origin - step)
               + 8.0 * moved(origin + step) - moved(origin + 2 * step)) / (12.0 * step)
        return raw

    scale = 1.0
    for label, powers in (("a", pa), ("b", pb), ("c", pc), ("d", pd)):
        exponent = {"a": alpha, "b": beta, "c": gamma, "d": delta}[label]
        scale *= (2.0 * exponent) ** (-sum(powers))
    return ev(derivs, centers0) * scale


def v_ss(alpha, A, beta, B, C, Z):
    """(ss|1/r_C) 闭式：−Z(2π/p)E_AB F₀(p|P−C|²)。"""
    p = alpha + beta
    P = [(alpha * A[i] + beta * B[i]) / p for i in range(3)]
    ab2 = sum((A[i] - B[i]) ** 2 for i in range(3))
    r2 = sum((P[i] - C[i]) ** 2 for i in range(3))
    return -Z * 2.0 * PI / p * math.exp(-alpha * beta / p * ab2) * boys(0, p * r2)[0]

def primitive_nuclear(pa, A, alpha, pb, B, beta, nuclei):
    """(ab|Σ_C −Z_C/r_C)：对 s-s 闭式做中心导数（递归复合）。"""
    derivs = []
    for label, powers in (("a", pa), ("b", pb)):
        for axis in range(3):
            for _ in range(powers[axis]):
                derivs.append((label, axis))
    step = 0.01

    def base(centers):
        return sum(v_ss(alpha, centers["a"], beta, centers["b"], C, Z) for C, Z in nuclei)

    def ev(pending, centers):
        if not pending:
            return base(centers)
        (label, axis), rest = pending[0], pending[1:]
        origin = centers[label][axis]

        def moved(t):
            c2 = {k: list(v) for k, v in centers.items()}
            c2[label][axis] = t
            return ev(rest, c2)

        return (moved(origin - 2 * step) - 8.0 * moved(origin - step)
                + 8.0 * moved(origin + step) - moved(origin + 2 * step)) / (12.0 * step)

    centers0 = {"a": list(A), "b": list(B)}
    raw = ev(derivs, centers0)
    # 每个一阶导数对应 1/(2α) 或 1/(2β)
    scale = 1.0
    for label, powers in (("a", pa), ("b", pb)):
        exponent = alpha if label == "a" else beta
        scale *= (2.0 * exponent) ** (-sum(powers))
    return raw * scale

def main():
    molecule = [("O", [0.0, 0.0, 0.0]), ("H", [0.0, 0.757, 0.586]), ("H", [0.0, -0.757, 0.586])]
    atoms = [(sym, [c * BOHR for c in pos]) for sym, pos in molecule]

    # 组装 AO 列表：(中心, 指数, 系数, 幂次)
    aos = []
    for sym, center in atoms:
        for kind, exps, coefs in STO3G[sym]:
            if kind == "s":
                powers = [(0, 0, 0)]
            else:
                powers = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            for pw in powers:
                aos.append((center, exps, coefs, pw))
    n = len(aos)
    print(f"n_ao = {n}")

    S = np.zeros((n, n))
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ci, ei, wi, pi = aos[i]
            cj, ej, wj, pj = aos[j]
            for a, ca in zip(ei, wi):
                na = norm(a, pi)
                for b, cb in zip(ej, wj):
                    nb = norm(b, pj)
                    S[i, j] += ca * cb * na * nb * overlap_prim(pi, ci, a, pj, cj, b)
                    T[i, j] += ca * cb * na * nb * kinetic_prim(pi, ci, a, pj, cj, b)
    print("S[0,0] =", S[0, 0], " S[0,5] =", S[0, 5], " S[2,2] =", S[2, 2])
    print("T[0,0] =", T[0, 0], " T[1,1] =", T[1, 1], " T[2,2] =", T[2, 2])

    # 核吸引：极窄 s 压到核上的双电子极限
    gamma_tight = 1.0e5
    norm_tight = norm(gamma_tight, (0, 0, 0))
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ci, ei, wi, pi = aos[i]
            cj, ej, wj, pj = aos[j]
            for a, ca in zip(ei, wi):
                na = norm(a, pi)
                for b, cb in zip(ej, wj):
                    nb = norm(b, pj)
                    weight = ca * cb * na * nb
                    V[i, j] += weight * primitive_nuclear(
                        pi, ci, a, pj, cj, b,
                        [(ac, 8.0 if asy == "O" else 1.0) for asy, ac in atoms])
    print("V[0,0] =", V[0, 0], " V[1,1] =", V[1, 1], " V[2,2] =", V[2, 2])

    H = T + V
    print("H[0,0] =", H[0, 0])

    # 双电子张量（用对称性少算）
    eri = np.zeros((n, n, n, n))
    count = 0
    for i in range(n):
        for j in range(i, n):
            for k in range(n):
                for l in range(k, n):
                    ci, ei, wi, pi = aos[i]
                    cj, ej, wj, pj = aos[j]
                    ck, ek, wk, pk = aos[k]
                    cl, el, wl, pl = aos[l]
                    value = 0.0
                    for a, ca in zip(ei, wi):
                        na = norm(a, pi)
                        for b, cb in zip(ej, wj):
                            nb = norm(b, pj)
                            for g, cg in zip(ek, wk):
                                nc = norm(g, pk)
                                for d, cd in zip(el, wl):
                                    nd = norm(d, pl)
                                    value += ca * cb * cg * cd * na * nb * nc * nd * primitive_2e(
                                        pi, ci, a, pj, cj, b, pk, ck, g, pl, cl, d)
                    eri[i, j, k, l] = value
                    eri[j, i, k, l] = value
                    eri[i, j, l, k] = value
                    eri[j, i, l, k] = value
                    eri[k, l, i, j] = value
                    eri[l, k, i, j] = value
                    eri[k, l, j, i] = value
                    eri[l, k, j, i] = value
                    count += 1
                    if count % 100 == 0:
                        print(".", end="", flush=True)
    print(f"\nunique quartets: {count}")
    print("(00|22) =", eri[0, 0, 2, 2], " (00|00) =", eri[0, 0, 0, 0])

    # 核排斥
    e_nuc = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            r = math.dist(atoms[i][1], atoms[j][1])
            zi = 8.0 if atoms[i][0] == "O" else 1.0
            zj = 8.0 if atoms[j][0] == "O" else 1.0
            e_nuc += zi * zj / r

    # 广义本征值：S^{-1/2}
    w, U = np.linalg.eigh(S)
    X = U @ np.diag(w ** -0.5) @ U.T

    n_occ = 5
    P = np.zeros((n, n))
    Hp = X.T @ H @ X
    _, Cp = np.linalg.eigh(Hp)
    C = X @ Cp
    for o in range(n_occ):
        P += 2.0 * np.outer(C[:, o], C[:, o])

    energy = None
    for step in range(200):
        J = np.einsum("ijkl,kl->ij", eri, P)
        K = np.einsum("ikjl,kl->ij", eri, P)
        F = H + J - 0.5 * K
        Fp = X.T @ F @ X
        _, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        P_new = np.zeros((n, n))
        for o in range(n_occ):
            P_new += 2.0 * np.outer(C[:, o], C[:, o])
        Jn = np.einsum("ijkl,kl->ij", eri, P_new)
        Kn = np.einsum("ikjl,kl->ij", eri, P_new)
        e = np.sum(P_new * H) + 0.5 * np.sum(P_new * Jn) - 0.25 * np.sum(P_new * Kn) + e_nuc
        delta = abs(e - energy) if energy is not None else float("inf")
        energy = e
        P = P_new
        if delta < 1e-11:
            print(f"converged in {step + 1} steps")
            break

    print(f"E = {energy:.10f}")
    print(f"E_nuc = {e_nuc:.10f}")
    print("MO =", np.round(np.linalg.eigvalsh(X.T @ (H + np.einsum('ijkl,kl->ij', eri, P)
                                                      - 0.5 * np.einsum('ikjl,kl->ij', eri, P)) @ X), 6))
    print(f"reference: HF/STO-3G H2O ~ -75.20 (opt geom) / ours(Rust) = -72.8758")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
