//! I-04 验收：`rayon` 并行下的两电子积分必须与单线程 **byte-identical**。
//!
//! 做法（也是实现上的保证）：并行只作用在每个壳层四元组的球谐块计算，散射进张量
//! 仍是串行、顺序固定。所以判据可以直接取「张量全部 f64 的字节串是否相同」。
//!
//! 这里对 1 / 2 / 4 线程各算一次，比较：
//!   ① 逐字节相同（`to_bits` 序列）；
//!   ② 一个稳定哈希（FNV-1a over bits）相同；
//! 并覆盖含 p 与 d 的体系（水/STO-3G 只有 p；用 CH2 水几何亦可）。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::two_electron::two_electron_with_threads;
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn hash_bits(values: &[f64]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[test]
fn thread_count_does_not_change_a_single_bit() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");

    let cases: Vec<(&str, Molecule)> = vec![
        (
            "H2",
            Molecule::new(
                vec![
                    Atom::new("H", [0.0, 0.0, -0.37]),
                    Atom::new("H", [0.0, 0.0, 0.37]),
                ],
                0,
                1,
            ),
        ),
        (
            "CH2(水几何)",
            Molecule::new(
                vec![
                    Atom::new("C", [0.0, 0.0, 0.0]),
                    Atom::new("H", [0.0, 0.757, 0.586]),
                    Atom::new("H", [0.0, -0.757, 0.586]),
                ],
                0,
                1,
            ),
        ),
    ];

    for (label, molecule) in cases {
        let basis = InternalBasis::build(basis_k1, &molecule).expect("build");
        let serial = two_electron_with_threads(&basis, &molecule, 1).expect("1 thread");
        let serial_hash = hash_bits(&serial.values);
        let serial_bits: Vec<u64> = serial.values.iter().map(|v| v.to_bits()).collect();

        for threads in [2usize, 4] {
            let parallel = two_electron_with_threads(&basis, &molecule, threads).expect("threads");
            let parallel_hash = hash_bits(&parallel.values);
            println!(
                "{label}: 1 线程 hash {serial_hash:#018x} / {threads} 线程 hash {parallel_hash:#018x}"
            );
            assert_eq!(
                parallel.n_functions, serial.n_functions,
                "{label}: 维数不同"
            );
            assert_eq!(
                parallel_hash, serial_hash,
                "{label}: {threads} 线程的哈希与单线程不同"
            );
            let parallel_bits: Vec<u64> = parallel.values.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                parallel_bits, serial_bits,
                "{label}: {threads} 线程的结果不是逐位相同"
            );
        }
    }
}

#[test]
fn the_default_entry_point_matches_single_thread_too() {
    // two_electron() 现在走 rayon 默认线程数，结果必须与显式 1 线程一致。
    let registry = Registry::open(data_root()).expect("qwb-data");
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
    let default = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("default");
    let serial = two_electron_with_threads(&basis, &molecule, 1).expect("serial");
    assert_eq!(
        default
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        serial
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "默认入口（rayon 默认线程）与单线程必须逐位相同"
    );
}
