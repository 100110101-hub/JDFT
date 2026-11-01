//! SC-06：**开壳层 UKS × meta-GGA 的整条链路**（τ 积累 + τ 的 Fock 矩阵元 + Ξ 插值）。
//!
//! 逐点测试（`qwb-xc/tests/sc06_spin_mgga.rs`、`sc06_spin_m06_correlation.rs`）证明的是
//! 公式与导数；这里证明的是**接线**：`uks` 从 `P^σ` 积累 `τ_σ`、在 Fock 里加
//! `½(∂e/∂τ_σ)∇φ_μ·∇φ_ν`、再交给自旋相关的 M06 交换/相关（含 ζ 插值的 Stoll 分解）。
//! 三者任何一处写错，总能量都会跑偏远超判据。
//!
//! 参考值：`tests/reference/uks_mgga.csv`（`scripts/pyscf_uks_mgga_reference.py` 用
//! PySCF 2.14.0 / libxc 7.0.0 的 `dft.UKS` 在 O₂ 三重态 / STO-3G 上算出，带
//! `source_doi`/`source_url`）。几何、基组、电荷、多重度、内存网格全部从 CSV 读，
//! 避免测试与生成脚本各写一份而漂移。
//!
//! 判据 0.5 mHa（与 FX-02 的 M06 水分子同一条）：本轮实测两端差 1e-4 量级，主因是
//! 网格（PySCF level 3 vs `grid-normal`），不是公式。

use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::uks::{UksOptions, uks_from_data};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/uks_mgga.csv")
}

/// `O 0.0 0.0 0.0; O 0.0 0.0 1.208`（Å）→ `Molecule`。
///
/// 只支持本参考文件里那种「符号 + 三个坐标」的写法；解析不了就让测试红，而不是静默跳过。
fn molecule_from_geometry(geometry: &str, charge: i32, spin_2s: u32) -> Molecule {
    let mut atoms = Vec::new();
    for chunk in geometry.split(';') {
        let fields: Vec<&str> = chunk.split_whitespace().collect();
        assert_eq!(fields.len(), 4, "geometry chunk `{chunk}`");
        let symbol = fields[0];
        let mut xyz = [0.0_f64; 3];
        for (axis, field) in fields[1..].iter().enumerate() {
            xyz[axis] = field.parse::<f64>().expect("coordinate");
        }
        atoms.push(Atom::new(symbol, xyz));
    }
    let multiplicity = spin_2s + 1;
    Molecule::new(atoms, charge, multiplicity)
}

fn read_rows(path: &std::path::Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .expect("header")
        .split(',')
        .map(str::trim)
        .map(String::from)
        .collect();
    let rows = lines
        .map(|line| line.split(',').map(str::trim).map(String::from).collect())
        .collect();
    (header, rows)
}

fn column(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|field| field == name)
        .unwrap_or_else(|| panic!("missing column `{name}`"))
}

#[test]
fn open_shell_uks_meta_gga_matches_pyscf() {
    let (header, rows) = read_rows(&reference_path());
    let functional_column = column(&header, "functional");
    let geometry_column = column(&header, "geometry_angstrom");
    let basis_column = column(&header, "basis");
    let charge_column = column(&header, "charge");
    let spin_column = column(&header, "spin_2s");
    let energy_column = column(&header, "energy_hartree");
    assert_eq!(rows.len(), 2, "expected m06-l and m06-2x references");

    let registry = Registry::open(data_root()).expect("qwb-data");
    for row in &rows {
        let functional = &row[functional_column];
        let molecule = molecule_from_geometry(
            &row[geometry_column],
            row[charge_column].parse::<i32>().expect("charge"),
            row[spin_column].parse::<u32>().expect("spin_2s"),
        );
        let mut options = UksOptions::default();
        options.max_iter = 300;
        let result = uks_from_data(
            &registry,
            &molecule,
            &row[basis_column],
            functional,
            "grid-normal",
            options,
        )
        .unwrap_or_else(|error| panic!("uks + {functional} 应当跑通：{error}"));
        let expected = row[energy_column].parse::<f64>().expect("energy");
        let delta = result.energy_hartree - expected;
        println!(
            "{functional}: E = {:.10} Ha（PySCF {expected:.10}，差 {:+.2e}），E_xc = {:.10}，\
             n_α/β = {}/{}, ⟨S²⟩ = {:.6}, iters = {}, converged = {}",
            result.energy_hartree,
            delta,
            result.xc_energy_hartree,
            result.n_alpha,
            result.n_beta,
            result.s_squared,
            result.iterations,
            result.scf_converged,
        );
        assert!(result.scf_converged, "{functional}: UKS 没有收敛");
        assert_eq!(
            (result.n_alpha, result.n_beta),
            (9, 7),
            "{functional}: 三重态 O2 应当是 9α/7β"
        );
        assert!(
            delta.abs() < 5.0e-4,
            "{functional}: E = {:.10} vs PySCF {expected:.10}（判据 0.5 mHa）",
            result.energy_hartree
        );
    }
}
