//! S-01 验收（project.md §6.5）：Fock 构建**与直接乘对照 < 1e-12**。
//!
//! 生产路径是四重循环；这里的参照路径把 `(ij|kl)` **展成矩阵**再用
//! `qwb_linalg::matmul` 乘一次。两条实现互不相干，差值就是验收要求的量。
//!
//! 另外用 H2/STO-3G 的真实张量做物理一致性检查：
//! 单占据 MO、`P = 2ccᵀ` 时应有 `E_2e = 2·(11|11)·?` —— 下面按代数恒等式断言。

use std::path::{Path, PathBuf};

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::{TwoElectronIntegrals, two_electron};
use qwb_linalg::{Matrix, matmul};
use qwb_registry::Registry;
use qwb_scf::{build_fock, two_electron_energy};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn h2() -> (InternalBasis, Molecule) {
    let registry = Registry::open(repo_root().join("qwb-data")).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    );
    let basis = InternalBasis::build(basis_k1, &molecule).expect("build");
    (basis, molecule)
}

/// 参照实现：把 `(ij|kl)` 展成 `(ij) × (kl)` 矩阵，用一次矩阵乘做库仑项；
/// 交换项把张量的指标顺序重排后再乘。
fn reference_jk(tensor: &TwoElectronIntegrals, density: &Matrix) -> (Matrix, Matrix) {
    let n = tensor.n_functions;
    let mut density_flat = vec![0.0; n * n];
    for k in 0..n {
        for l in 0..n {
            density_flat[k * n + l] = density[(k, l)];
        }
    }

    // J: M[ij, kl] * P[kl]
    let mut j_matrix = Matrix::zeros(n * n, n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    j_matrix[(i * n + j, k * n + l)] = tensor.get(i, j, k, l);
                }
            }
        }
    }
    let p_column = Matrix::from_column_major(n * n, 1, density_flat.clone()).expect("column");
    let j_flat = matmul(&j_matrix, &p_column).expect("J = M P");
    let mut j = Matrix::zeros(n, n);
    for i in 0..n {
        for jx in 0..n {
            j[(i, jx)] = j_flat[(i * n + jx, 0)];
        }
    }

    // K: 用 K[i,j] = Σ_kl (ik|jl) P[k,l]，同样展成矩阵（指标重排后一次乘）。
    let mut k_matrix = Matrix::zeros(n * n, n * n);
    for i in 0..n {
        for jx in 0..n {
            for k in 0..n {
                for l in 0..n {
                    k_matrix[(i * n + jx, k * n + l)] = tensor.get(i, k, jx, l);
                }
            }
        }
    }
    let k_flat = matmul(&k_matrix, &p_column).expect("K = M P");
    let mut k = Matrix::zeros(n, n);
    for i in 0..n {
        for jx in 0..n {
            k[(i, jx)] = k_flat[(i * n + jx, 0)];
        }
    }
    (j, k)
}

#[test]
fn fock_matches_the_matrix_multiply_reference() {
    let (basis, molecule) = h2();
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let one = one_electron(&basis, &molecule).expect("1e");

    // 一个物理上合理的密度：H2 的 1σ MO 近似取成两个 AO 的等权组合，
    // 双占据 → P = 2·(0.5, 0.5)ᵀ(0.5, 0.5) = [[0.5,0.5],[0.5,0.5]]。
    let mut density = Matrix::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            density[(i, j)] = 0.5;
        }
    }

    let fock = build_fock(&one.core_hamiltonian, &tensor, &density).expect("fock");
    let (j_ref, k_ref) = reference_jk(&tensor, &density);

    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (fock.coulomb[(i, j)] - j_ref[(i, j)]).abs() < 1e-12,
                "J({i},{j}) = {} vs 直接乘 {}",
                fock.coulomb[(i, j)],
                j_ref[(i, j)]
            );
            assert!(
                (fock.exchange[(i, j)] - k_ref[(i, j)]).abs() < 1e-12,
                "K({i},{j}) = {} vs 直接乘 {}",
                fock.exchange[(i, j)],
                k_ref[(i, j)]
            );
        }
    }

    // F = H + 2J − K
    for i in 0..2 {
        for j in 0..2 {
            let expected = one.core_hamiltonian[(i, j)] + j_ref[(i, j)] - 0.5 * k_ref[(i, j)];
            assert!((fock.fock[(i, j)] - expected).abs() < 1e-12, "F({i},{j})");
        }
    }

    // J、K、F 都对称（真实张量与对称密度下）。
    for i in 0..2 {
        for j in 0..2 {
            assert!((fock.coulomb[(i, j)] - fock.coulomb[(j, i)]).abs() < 1e-12);
            assert!((fock.exchange[(i, j)] - fock.exchange[(j, i)]).abs() < 1e-12);
            assert!((fock.fock[(i, j)] - fock.fock[(j, i)]).abs() < 1e-12);
        }
    }
}

#[test]
fn two_electron_energy_satisfies_the_rf_identity() {
    // 单占据 MO、P = 2ccᵀ：J_ij = (ij|kl)P_kl，K_ij = (ik|jl)P_kl。
    // 用 c = (1/√2, 1/√2)（对称 MO）时，由对称性 J = K 与 P 同形，
    // 于是 E_2e = ½Tr[P(2J−K)] = ½Tr[P·J]，与直接用张量算的库仑能一致。
    let (basis, molecule) = h2();
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let one = one_electron(&basis, &molecule).expect("1e");

    let c = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
    let mut density = Matrix::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            density[(i, j)] = 2.0 * c[i] * c[j];
        }
    }

    let fock = build_fock(&one.core_hamiltonian, &tensor, &density).expect("fock");
    let energy = two_electron_energy(&density, &fock);

    // 独立算式：E_2e = ½ Σ P_ij P_kl (2(ij|kl) − (ik|jl))。
    let mut expected = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    let p_ij = density[(i, j)];
                    let p_kl = density[(k, l)];
                    expected +=
                        0.5 * p_ij * p_kl * (tensor.get(i, j, k, l) - 0.5 * tensor.get(i, k, j, l));
                }
            }
        }
    }
    assert!(
        (energy - expected).abs() < 1e-12,
        "E_2e = {energy} vs 独立求和 {expected}"
    );
    assert!(energy > 0.0, "电子排斥能必须为正");
}

#[test]
fn shape_mismatch_is_rejected() {
    let (basis, molecule) = h2();
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let wrong = Matrix::zeros(3, 3);
    let ok = Matrix::zeros(2, 2);
    assert!(build_fock(&ok, &tensor, &wrong).is_err());
    assert!(build_fock(&wrong, &tensor, &ok).is_err());
}
