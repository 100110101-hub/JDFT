//! IN-05 验收：**并行 Fock 构建必须与串行逐位相同**，并实测加速比。
//!
//! 判据是「逐位」（`==`），不是「容差内」：并行只换了「谁算哪一行」，没换任何元素的
//! 加法顺序。若哪天有人改成多线程累加到同一元素上，这个测试会立刻红。
use std::time::Instant;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_registry::Registry;
use qwb_scf::{build_fock, build_fock_serial};

fn data_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    )
}

#[test]
fn parallel_fock_is_bit_identical_to_the_serial_reference() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule).unwrap();
    let n = basis.n_functions();
    let tensor = two_electron(&basis, &molecule).unwrap();
    let hcore = qwb_integrals::one_electron::one_electron(&basis, &molecule)
        .unwrap()
        .core_hamiltonian;
    // 非平凡密度：单位矩阵会让很多项恰好为 0，掩盖排序问题。
    let mut density = qwb_linalg::Matrix::zeros(n, n);
    let mut seed = 424_242_u64;
    for i in 0..n {
        for j in 0..n {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            density.set(i, j, ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5);
        }
    }

    let parallel = build_fock(&hcore, &tensor, &density).unwrap();
    let serial = build_fock_serial(&hcore, &tensor, &density).unwrap();

    let mut mismatches = 0usize;
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            // 逐位比较：`to_bits` 相同才算过。
            for (label, a, b) in [
                ("j", parallel.coulomb[(i, j)], serial.coulomb[(i, j)]),
                ("k", parallel.exchange[(i, j)], serial.exchange[(i, j)]),
                ("f", parallel.fock[(i, j)], serial.fock[(i, j)]),
            ] {
                if a.to_bits() != b.to_bits() {
                    mismatches += 1;
                    worst = worst.max((a - b).abs());
                    println!("{label}({i},{j}): parallel {a:.17e} vs serial {b:.17e}");
                }
            }
        }
    }
    println!("n = {n}: 逐位不一致元素 {mismatches} 个（最大差 {worst:.3e}）");
    assert_eq!(mismatches, 0, "并行与串行必须逐位相同");
}

#[test]
fn parallel_fock_reports_a_measured_speedup() {
    // 真实体系（def2-SVP 水，n = 25）+ 多次重复：单次太小测不准，重复把噪声摊掉。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = InternalBasis::build(registry.get_basis("def2-svp").unwrap(), &molecule).unwrap();
    let n = basis.n_functions();
    let tensor = two_electron(&basis, &molecule).unwrap();
    let hcore = qwb_integrals::one_electron::one_electron(&basis, &molecule)
        .unwrap()
        .core_hamiltonian;
    let mut density = qwb_linalg::Matrix::zeros(n, n);
    let mut seed = 99_u64;
    for i in 0..n {
        for j in 0..n {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            density.set(i, j, ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5);
        }
    }

    let repeats = 40;
    let started = Instant::now();
    let mut last_serial = build_fock_serial(&hcore, &tensor, &density).unwrap();
    for _ in 1..repeats {
        last_serial = build_fock_serial(&hcore, &tensor, &density).unwrap();
    }
    let serial_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut last_parallel = build_fock(&hcore, &tensor, &density).unwrap();
    for _ in 1..repeats {
        last_parallel = build_fock(&hcore, &tensor, &density).unwrap();
    }
    let parallel_seconds = started.elapsed().as_secs_f64();

    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((last_serial.fock[(i, j)] - last_parallel.fock[(i, j)]).abs());
        }
    }
    println!(
        "n = {n}, {repeats} 次: serial {serial_seconds:.3} s, parallel {parallel_seconds:.3} s, 加速 {:.2}x, 最大差 {worst:.3e}",
        serial_seconds / parallel_seconds
    );
    assert_eq!(worst, 0.0, "加速测试里也必须逐位相同");
}
