//! T1：注册参考值集合（`project.md` §7.0）。
//!
//! CSV 是唯一真值来源——测试只负责把每一行与实际计算对上，不在这里另写常数。
//! 新增一行就必须在下面的 `match` 里补一个分支，否则测试直接失败（防止参考值
//! 悄悄漂移）。

use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_from_data, rks_scf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    )
}

struct Reference {
    quantity: String,
    value: f64,
    tolerance: f64,
    source: String,
}

fn load_references() -> Vec<Reference> {
    let path = manifest_dir().join("tests/reference/h2_sto3g.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("读不到 {}: {err}", path.display()));
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("quantity,") {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 4, "参考值行格式错误: {line}");
        out.push(Reference {
            quantity: fields[0].to_string(),
            value: fields[1].parse().expect("value"),
            tolerance: fields[2].parse().expect("tolerance"),
            source: fields[3].to_string(),
        });
    }
    out
}

#[test]
fn registered_reference_values_are_reproduced() {
    let registry =
        Registry::open(manifest_dir().join("../../../qwb-data")).expect("qwb-data 必须能加载");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = h2();
    let basis = qwb_basis::InternalBasis::build(basis_k1, &molecule).expect("build");
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &molecule).expect("grid");

    // 空 terms ⇒ 闭壳层 HF（F = H + J − ½K）。
    let mut hf_options = RksOptions::default();
    hf_options.energy_threshold = 1.0e-10;
    hf_options.max_iter = 200;
    let hf = rks_scf(&basis, &molecule, &tensor, &grid, &[], hf_options).expect("hf");

    let mut lda_options = RksOptions::default();
    lda_options.energy_threshold = 1.0e-10;
    let lda = rks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "svwn5",
        "grid-normal",
        lda_options,
    )
    .expect("lda");

    let references = load_references();
    assert!(
        references.len() >= 8,
        "参考值集合不完整，只读到 {} 行",
        references.len()
    );

    for reference in references {
        let computed = match reference.quantity.as_str() {
            "hf_total_energy" => hf.energy_hartree,
            "svwn5_total_energy" => lda.energy_hartree,
            "nuclear_repulsion" => lda.nuclear_repulsion_hartree,
            "overlap_01" => one.overlap[(0, 1)],
            "eri_0000" => tensor.get(0, 0, 0, 0),
            "eri_0001" => tensor.get(0, 0, 0, 1),
            "eri_0011" => tensor.get(0, 0, 1, 1),
            "eri_0101" => tensor.get(0, 1, 0, 1),
            other => {
                panic!("参考值 CSV 里出现未知量 `{other}`：必须在 t01_reference 的 match 里补分支")
            }
        };
        let difference = (computed - reference.value).abs();
        println!(
            "{:<20} computed {computed:>15.10}  reference {:>15.10}  diff {difference:.3e}  tol {:.1e}  [{}]",
            reference.quantity, reference.value, reference.tolerance, reference.source
        );
        assert!(
            difference <= reference.tolerance,
            "{}: 计算值 {computed:.10}, 参考值 {:.10}, 差 {difference:.3e} > 容差 {:.1e}（来源: {}）",
            reference.quantity,
            reference.value,
            reference.tolerance,
            reference.source
        );
    }
}
