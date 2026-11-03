//! RS-04（GIAO 屏蔽）：逐核**绝对**屏蔽值的验收 —— 对照 PySCF 的 GIAO-HF/CPHF。
//!
//! # 参考值
//!
//! `tests/reference/giao_nmr.csv`（由 `scripts/pyscf_giao_nmr_reference.py` 生成：PySCF 2.14 的
//! libcint GIAO 积分 + 转写的 `pyscf/prop/nmr/rhf.py` v1.7.6；那个转写由 PySCF 自己写在文件里
//! 的自检值钉住，见脚本头部）。判据 **0.1 ppm**（任务书的 A1）。
//!
//! CSV 里还有水的 **6-31G** 一行（O 326.753598、H 32.725715），但本仓库的 `qwb-data/basis/`
//! **没有 6-31G**（只有 sto-3g / 6-311g-dp / def2-svp / def2-tzvp / lanl2dz 及 RI 变体），
//! 而 `qwb_registry::basis::BasisSet` 是 `#[non_exhaustive]`、`InternalBasis` 的字段私有，
//! 测试侧既构造不出基组、也不该往 `qwb-data`（另一层、带 sha256/清单）里加数据 —— 所以这一行
//! **无法在本轮验证**，见报告的"未能验证"。能验的两行都验了。
//!
//! # 三条判据
//!
//! 1. **A1（绝对值）**：水/STO-3G 与甲烷/STO-3G，逐核各向同性 + 三个对角分量都进 0.1 ppm。
//! 2. **A1 的中间量**：抗磁张量单独对（O 359.0486/393.5013/380.5008、H 23.2536/38.2721/30.5263、
//!    H 的 yz/zy 7.5044/8.4207 ppm）。这些数**不在** CSV 里（脚本有意只存总和，因为 dia/para 的
//!    分解不是规范不变量），它们是把 oracle 的 `giao_dia` / `giao_para` 单独调出来打印得到的：
//!
//!    ```text
//!    python3 -c "从 pyscf_giao_nmr_reference 导入 make_scf/giao_dia/giao_para/solve_mo1，\
//!                对 SYSTEMS['water'] + 'sto-3g' 调一遍，乘 nist.ALPHA**2*1e6"
//!    ```
//!
//!    （本条只用 oracle 的**代码**，不重推它的公式。）顺磁张量用同一段 `giao_para` 的输出核对
//!    —— 它与"总和 − 抗磁"一致，两条独立断言能抓住互相抵消的错误。
//! 3. **A2（规范不变性）**：GIAO 里"换规范原点"没有这个自由度（London 相位里的原点到 `R_μ−R_ν`
//!    就抵消了），所以规范检验做成**换坐标原点**：把整个分子平移后重算，总和必须不动。
//!    判据 < 1 ppm（非 GIAO 的共原实现是 7.1229 ppm）。dia/para 的分解**会**动，只有总和不动
//!    —— 这正好也证明了分解确实依赖绝对原点（`giao_dia.rs` 记的那一条）。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_response::giao_nmr::GiaoShielding;
use qwb_response::mo_integrals::build_cphf_integrals;

/// A1 的判据：逐核各向同性屏蔽与 PySCF 的差（ppm）。
const ABSOLUTE_TOLERANCE_PPM: f64 = 0.1;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/giao_nmr.csv")
}

/// 与 `scripts/pyscf_giao_nmr_reference.py` 的 `SYSTEMS` 逐字一致的几何（Å）。
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

fn methane() -> Molecule {
    let t = 0.629;
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [t, t, t]),
            Atom::new("H", [-t, -t, t]),
            Atom::new("H", [-t, t, -t]),
            Atom::new("H", [t, -t, -t]),
        ],
        0,
        1,
    )
}

/// 把整个分子平移 `shift`（Å）：只换坐标原点，物理不变。
fn translated(molecule: &Molecule, shift: [f64; 3]) -> Molecule {
    let atoms = molecule
        .atoms
        .iter()
        .map(|atom| {
            let p = atom.position_angstrom;
            Atom::new(
                &atom.symbol,
                [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]],
            )
        })
        .collect();
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

/// CSV 的一行（`sigma_iso_ppm` + 三个对角分量）。
#[derive(Debug, Clone, Copy)]
struct Reference {
    iso: f64,
    xx: f64,
    yy: f64,
    zz: f64,
}

/// `(system, basis, atom_index) -> Reference`。
fn references() -> BTreeMap<(String, String, usize), Reference> {
    let text = std::fs::read_to_string(reference_path()).expect("giao_nmr.csv");
    let mut out = BTreeMap::new();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split(',').collect();
    let column = |name: &str| {
        header
            .iter()
            .position(|item| *item == name)
            .unwrap_or_else(|| panic!("column {name}"))
    };
    let (s_at, b_at, a_at) = (column("system"), column("basis"), column("atom_index"));
    let (iso_at, xx_at, yy_at, zz_at) = (
        column("sigma_iso_ppm"),
        column("sigma_xx_ppm"),
        column("sigma_yy_ppm"),
        column("sigma_zz_ppm"),
    );
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        out.insert(
            (
                fields[s_at].to_string(),
                fields[b_at].to_string(),
                fields[a_at].parse().expect("atom_index"),
            ),
            Reference {
                iso: fields[iso_at].parse().expect("iso"),
                xx: fields[xx_at].parse().expect("xx"),
                yy: fields[yy_at].parse().expect("yy"),
                zz: fields[zz_at].parse().expect("zz"),
            },
        );
    }
    out
}

/// 跑一次 SCF + GIAO，返回逐核屏蔽张量。
fn shielding(molecule: &Molecule, basis_name: &str) -> Vec<GiaoShielding> {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), molecule)
        .expect("grid");
    let mut options = qwb_scf::rks::RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-11;
    options.density_threshold = 1.0e-9;
    let scf = qwb_scf::rks::rks_scf(&basis, molecule, &tensor, &grid, &[], options).expect("HF");
    let electrons: i64 = molecule
        .atoms
        .iter()
        .map(|atom| i64::from(qwb_registry::elements::atomic_number(&atom.symbol).unwrap_or(0)))
        .sum::<i64>()
        - i64::from(molecule.charge);
    let n_occupied = usize::try_from(electrons / 2).expect("occupied count");
    let energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();
    let dipole_ao = qwb_integrals::dipole::dipole_integrals(&basis, molecule).expect("dipole");
    let integrals =
        build_cphf_integrals(&tensor, &scf.coefficients, &dipole_ao, n_occupied).expect("CPHF");
    qwb_response::giao_shielding_ppm(
        &basis,
        molecule,
        &scf.coefficients,
        &energies,
        n_occupied,
        &integrals,
        &tensor,
    )
    .expect("GIAO shielding")
}

/// A1：一个体系逐核对照 CSV。
///
/// 先**打印每一行**再统一断言（一次失败就能看到整张表；判据还是 0.1 ppm，没有放宽）。
fn check_against_the_reference(
    system: &str,
    basis_name: &str,
    molecule: &Molecule,
) -> Vec<GiaoShielding> {
    let expected = references();
    let computed = shielding(molecule, basis_name);
    assert_eq!(computed.len(), molecule.n_atoms());
    let mut worst_iso = 0.0_f64;
    let mut worst_component = 0.0_f64;
    let mut where_worst = String::new();
    for (index, ours) in computed.iter().enumerate() {
        let reference = expected[&(system.to_string(), basis_name.to_string(), index)];
        let difference = (ours.sigma_iso_ppm - reference.iso).abs();
        worst_iso = worst_iso.max(difference);
        println!(
            "  {system}/{basis_name} {}#{index}: σ_iso {:.6} ppm vs PySCF {:.6}（|Δ| {:.2e}）\
             | 分量 {:.6}/{:.6}/{:.6} vs {:.6}/{:.6}/{:.6}",
            ours.symbol,
            ours.sigma_iso_ppm,
            reference.iso,
            difference,
            ours.total_ppm[0][0],
            ours.total_ppm[1][1],
            ours.total_ppm[2][2],
            reference.xx,
            reference.yy,
            reference.zz,
        );
        // 三个对角分量与各向同性一起进 0.1 ppm：迹对不代表分量对。
        for (component, value) in [
            (ours.total_ppm[0][0], reference.xx),
            (ours.total_ppm[1][1], reference.yy),
            (ours.total_ppm[2][2], reference.zz),
        ] {
            let component_difference = (component - value).abs();
            if component_difference > worst_component {
                worst_component = component_difference;
                where_worst = format!("{}#{index}", ours.symbol);
            }
        }
    }
    println!(
        "  {system}/{basis_name}: worst |Δ_iso| = {worst_iso:.3e} ppm、\
         worst |Δ_分量| = {worst_component:.3e} ppm（{where_worst}；判据 {ABSOLUTE_TOLERANCE_PPM}）"
    );
    assert!(
        worst_iso < ABSOLUTE_TOLERANCE_PPM,
        "{system}/{basis_name}: 各向同性 GIAO 屏蔽与 PySCF 参考差 {worst_iso:.3e} ppm\
         （判据 {ABSOLUTE_TOLERANCE_PPM}）"
    );
    assert!(
        worst_component < ABSOLUTE_TOLERANCE_PPM,
        "{system}/{basis_name}: 张量分量与 PySCF 参考差 {worst_component:.3e} ppm\
         （{where_worst}；判据 {ABSOLUTE_TOLERANCE_PPM}）"
    );
    computed
}

#[test]
fn the_water_sto3g_shielding_matches_the_pyscf_reference() {
    check_against_the_reference("water", "sto-3g", &water());
}

#[test]
fn the_methane_sto3g_shielding_matches_the_pyscf_reference() {
    check_against_the_reference("methane", "sto-3g", &methane());
}

/// **中间量**：抗磁张量逐分量对照 oracle 的 `giao_dia`（水/STO-3G）。
///
/// 抗磁项不经过 CPHF，所以这一条单独把"`a11`/`a01` 的装配 + 迹修正 + ppm 常数"钉住；
/// 顺磁项由同一段 oracle 的 `giao_para` 输出核对。两段的和就是上面 A1 的 CSV 值
/// （365.448064/368.458917/363.111310 → iso 365.672764）。
#[test]
fn the_diamagnetic_and_paramagnetic_tensors_match_the_oracle_split() {
    let computed = shielding(&water(), "sto-3g");
    // oracle（`giao_dia` × nist.ALPHA**2*1e6，水/STO-3G）：
    let expected_dia = [
        ([359.048_612, 393.501_291, 380.500_838], [0.0, 0.0]),
        (
            [23.253_612, 38.272_146, 30.526_328],
            [-7.504_439, -8.420_717],
        ),
        ([23.253_612, 38.272_146, 30.526_328], [7.504_439, 8.420_717]),
    ];
    // oracle（`giao_para`，同一段代码）：
    let expected_para = [
        ([6.399_452, -25.042_374, -17.389_528], [0.0, 0.0]),
        ([3.750_108, 2.436_741, 2.807_353], [3.203_295, 1.371_276]),
        ([3.750_108, 2.436_741, 2.807_353], [-3.203_295, -1.371_276]),
    ];
    // 参考给到 4–6 位小数，判据取 0.01 ppm（Å→Bohr 常数差带来的相对 1.84e-8 只有 1e-5 ppm）。
    let tolerance = 1.0e-2_f64;
    let mut worst = 0.0_f64;
    for (index, ours) in computed.iter().enumerate() {
        let (dia_diagonal, dia_off) = expected_dia[index];
        let (para_diagonal, para_off) = expected_para[index];
        for component in 0..3 {
            let dia = ours.dia_ppm[component][component];
            let para = ours.para_ppm[component][component];
            println!(
                "  {}#{index} comp={component}: dia {dia:.6} vs {:.6} | para {para:.6} vs {:.6}",
                ours.symbol, dia_diagonal[component], para_diagonal[component]
            );
            worst = worst.max((dia - dia_diagonal[component]).abs());
            worst = worst.max((para - para_diagonal[component]).abs());
        }
        // yz / zy：`a11` 对 α↔β 不对称，所以这两个元素本来就**不等**。
        for (alpha, beta, expected) in [(1, 2, dia_off[0]), (2, 1, dia_off[1])] {
            let value = ours.dia_ppm[alpha][beta];
            println!(
                "  {}#{index} dia[{alpha}][{beta}] = {value:.6} vs {expected:.6}",
                ours.symbol
            );
            worst = worst.max((value - expected).abs());
        }
        for (alpha, beta, expected) in [(1, 2, para_off[0]), (2, 1, para_off[1])] {
            let value = ours.para_ppm[alpha][beta];
            println!(
                "  {}#{index} para[{alpha}][{beta}] = {value:.6} vs {expected:.6}",
                ours.symbol
            );
            worst = worst.max((value - expected).abs());
        }
    }
    println!("  worst |Δ| = {worst:.3e} ppm（判据 {tolerance}）");
    assert!(
        worst < tolerance,
        "dia/para 分解与 oracle 的 giao_dia/giao_para 差 {worst:.3e} ppm（判据 {tolerance}）"
    );
}

/// **A2：规范不变性** —— 换坐标原点（把整个分子平移）后总和必须不动。
///
/// 判据 < 1 ppm（任务书 A2）；非 GIAO 的共原实现是 7.1229 ppm
/// （`rs04_paramagnetic.rs::the_common_origin_gauge_error_is_measured` 实测）。
/// 顺带把 dia/para 的分解也打出来：分解**会**动，这正是"只有总和不依赖绝对原点"的证据。
#[test]
fn the_giao_shielding_is_invariant_under_a_change_of_coordinate_origin() {
    let shifts = [[0.0, 0.0, 0.0], [0.37, -0.21, 0.43], [-0.61, 0.52, -0.28]];
    let mut totals: Vec<Vec<GiaoShielding>> = Vec::new();
    for shift in shifts {
        let molecule = translated(&water(), shift);
        totals.push(shielding(&molecule, "sto-3g"));
    }
    let n_atoms = totals[0].len();
    let mut worst_total = 0.0_f64;
    let mut worst_dia = 0.0_f64;
    let mut worst_para = 0.0_f64;
    for index in 0..n_atoms {
        let mut iso: Vec<f64> = Vec::new();
        let mut dia: Vec<f64> = Vec::new();
        let mut para: Vec<f64> = Vec::new();
        for values in &totals {
            let entry = &values[index];
            iso.push(entry.sigma_iso_ppm);
            dia.push((entry.dia_ppm[0][0] + entry.dia_ppm[1][1] + entry.dia_ppm[2][2]) / 3.0);
            para.push((entry.para_ppm[0][0] + entry.para_ppm[1][1] + entry.para_ppm[2][2]) / 3.0);
        }
        let spread = |values: &[f64]| {
            values.iter().cloned().fold(f64::MIN, f64::max)
                - values.iter().cloned().fold(f64::MAX, f64::min)
        };
        let (iso_spread, dia_spread, para_spread) = (spread(&iso), spread(&dia), spread(&para));
        worst_total = worst_total.max(iso_spread);
        worst_dia = worst_dia.max(dia_spread);
        worst_para = worst_para.max(para_spread);
        println!(
            "  {}#{index}: σ_iso {iso:?}；总 spread {iso_spread:.3e} ppm\
             （分解：dia spread {dia_spread:.3e}、para spread {para_spread:.3e} ppm）",
            totals[0][index].symbol
        );
    }
    println!(
        "  worst total spread = {worst_total:.3e} ppm（判据 1）; \
         worst dia spread = {worst_dia:.3e}, worst para spread = {worst_para:.3e} ppm"
    );
    assert!(
        worst_total < 1.0,
        "GIAO 总屏蔽随坐标原点变化 {worst_total:.3e} ppm（判据 1 ppm）"
    );
    // 分解确实依赖绝对原点：如果 dia/para 的 spread 也是零，那说明平移根本没生效
    // （或者 `a01` 的绝对 `r` 被悄悄换成了相对 `r`），这条断言就是为了不让上一行空过。
    assert!(
        worst_dia > 1.0e-3 || worst_para > 1.0e-3,
        "dia/para 的分解也一动没动（{worst_dia:.3e} / {worst_para:.3e} ppm）—— 平移没有生效？"
    );
}
