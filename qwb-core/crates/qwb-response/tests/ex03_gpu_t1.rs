//! EX-03：feature `gpu-kernel` 下，GPU 核算出的 T1 必须与 CPU 四重循环一致（< 1e-12）。
#![cfg(feature = "gpu-kernel")]

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::dipole::dipole_integrals;
use qwb_integrals::two_electron::two_electron;
use qwb_registry::Registry;
use qwb_response::mo_integrals::{build_cphf_integrals, build_cphf_integrals_with_gpu_t1};

fn data_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

#[test]
fn gpu_t1_matches_the_cpu_reference() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    );
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule).unwrap();
    let tensor = two_electron(&basis, &molecule).unwrap();
    // **非平凡**系数：单位矩阵会让变换退化成「复制 ERI 元素」，两边必然逐位相同，
    // 比较就没有意义（第一版正是 0.0）。用确定性伪随机数，确保真的做两次收缩。
    let n = basis.n_functions();
    let mut seed = 2024_u64;
    let mut coefficients = qwb_linalg::Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            coefficients.set(i, j, ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5);
        }
    }
    let dipoles = dipole_integrals(&basis, &molecule).unwrap();
    let n_occ = 2;

    let cpu = build_cphf_integrals(&tensor, &coefficients, &dipoles, n_occ).unwrap();
    let gpu = build_cphf_integrals_with_gpu_t1(&tensor, &coefficients, &dipoles, n_occ).unwrap();

    let n_pairs = n_occ * (n - n_occ);
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for p in 0..n_pairs {
        for q in 0..n_pairs {
            let a = cpu.t1(p, q);
            scale = scale.max(a.abs());
            worst = worst.max((a - gpu.t1(p, q)).abs());
        }
    }
    println!("T1: worst |gpu - cpu| = {worst:.3e}（scale {scale:.3e}）");
    assert!(scale > 0.0, "参考值全零，比较没有意义");
    assert!(
        worst / scale < 1.0e-12,
        "T1 不一致：{worst:.3e}（相对 {:.3e}）",
        worst / scale
    );
}
