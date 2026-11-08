//! I-05 的**外部**对照：LANL2DZ 的 ECP 矩阵按 `l` 通道与 PySCF 2.14.0 逐元比对。
//!
//! # 为什么需要这一条（原来的 I-05 为什么不够）
//!
//! `tests/i05_ecp.rs` 判的是"本项目的两条实现互相一致"（解析 vs 半数值）。两条实现共享
//! 同一套**数据解释**，所以数据解释一旦错，两条会一起错、而且错得一模一样：
//!
//! * 2026-11 实测：Zn 的一中心 ECP 矩阵，两条实现互差 `3e-14`，而与 PySCF 的
//!   `Tr(S⁻¹V_ecp)` 是 `+0.485` vs `-24.797`；
//! * 根因有两层：BSE 的 `r_exponents` 是 `r^(n-2)` 约定（数据里却声明成 `r^n`），
//!   以及数据里的 `l < L` 块**已经是** `ΔV_l = V_l − V_L`（实现里又减了一次 `V_L`）。
//!   两条加起来把 Zn 的 3d 通道压到 1/18，ZnO 因此高出 oracle 约 20 Ha。
//!
//! # 参考值怎么来的
//!
//! `scripts/pyscf_ecp_channels_reference.py`（PySCF 2.14.0 / libcint）把 `ECPscalar`
//! 按通道拆开：`local` = 只有局域块的矩阵，`delta_l{l}` = `V_l P_l`（带该通道的矩阵减去
//! `local`），`total` = `ecp='lanl2dz'` 的全量矩阵。`total` 必须是
//! `local + Σ_l delta_l`（脚本里同一次运行验证到 `8.9e-16`）。
//!
//! 每一行是 `(element, channel, ao_i, ao_j)` 的一个矩阵元；只收同 `l` 的 `(i, j)` 对
//! （球对称算符在一中心下按 `l` 分块，跨块元恒为 0）。AO 顺序**两边逐位一致**：整体矩阵的
//! 逐元差实测 Zn `4.1e-13`、Cu `1.3e-13`（矩阵尺度 `5.2` / `3.6`），所以下标可以对齐用。
//!
//! # 判据
//!
//! 每个元与 CSV 差 < 1e-9 Ha；实测最大差见 `println!` 那行（小 3 个量级左右）。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_registry::ecp::Ecp;
use qwb_scf::ecp::{EcpOperator, matrix_elements};

/// 与 CSV 的容差（Ha）。实测最大差 ~4e-13，见 `per_channel_matrices_match_pyscf`。
const TOLERANCE: f64 = 1.0e-9;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn registry() -> Registry {
    Registry::open(repo_root().join("qwb-data")).expect("qwb-data")
}

fn operator(name: &str, molecule: &Molecule) -> EcpOperator {
    let path = repo_root()
        .join("qwb-data")
        .join("ecp")
        .join(format!("{name}.yaml"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("读不到 {}: {err}", path.display()));
    let ecp: Ecp = serde_yaml::from_str(&text).expect("K1 ecp");
    EcpOperator::from_ecp(&ecp, molecule).expect("operator")
}

/// `(element, channel, ao_i, ao_j) -> value`
type Reference = BTreeMap<(String, String, usize, usize), f64>;

/// 一个元素的通道矩阵 + 每个 `l` 的 AO 下标列表。
type Channels = (Vec<(String, Matrix)>, BTreeMap<u32, Vec<usize>>);

fn reference() -> Reference {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/ecp_lanl2dz_channels.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("读不到 {}: {err}", path.display()));
    let mut out = Reference::new();
    let mut header: Vec<&str> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if header.is_empty() {
            header = fields;
            continue;
        }
        let at = |name: &str| {
            header
                .iter()
                .position(|field| *field == name)
                .unwrap_or_else(|| panic!("CSV 缺 {name} 列"))
        };
        let key = (
            fields[at("element")].to_string(),
            fields[at("channel")].to_string(),
            fields[at("ao_i")].parse().expect("ao_i"),
            fields[at("ao_j")].parse().expect("ao_j"),
        );
        out.insert(key, fields[at("value_hartree")].parse().expect("value"));
    }
    out
}

fn subtract(left: &Matrix, right: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(left.nrows(), left.ncols());
    for i in 0..left.nrows() {
        for j in 0..left.ncols() {
            out[(i, j)] = left[(i, j)] - right[(i, j)];
        }
    }
    out
}

/// 一个元素各通道的矩阵 + 每个 `l` 的 AO 下标列表（与 CSV 的 channel 同名）。
fn channel_matrices(element: &str, ecp_name: &str) -> Channels {
    let registry = registry();
    let molecule = Molecule::new(vec![Atom::new(element, [0.0, 0.0, 0.0])], 0, 1);
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("lanl2dz").expect("lanl2dz"), &molecule)
            .expect("basis");
    let level = registry.get_grid("grid-veryfine").expect("grid-veryfine");
    let full = operator(ecp_name, &molecule);

    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for shell in basis.shells() {
        let components = 2 * (shell.l as usize) + 1;
        let contractions = shell.coefficients.len() / shell.exponents.len().max(1);
        for k in 0..contractions * components {
            groups
                .entry(shell.l)
                .or_default()
                .push(shell.first_function + k);
        }
    }

    let total = matrix_elements(&basis, &full, level).expect("total");
    let mut local_operator = full.clone();
    local_operator.projectors.clear();
    let local = matrix_elements(&basis, &local_operator, level).expect("local");

    let mut out = vec![
        ("total".to_string(), total),
        ("local".to_string(), local.clone()),
    ];
    for channel in &full.projectors {
        let mut single = full.clone();
        single.projectors = vec![channel.clone()];
        let with_channel = matrix_elements(&basis, &single, level).expect("channel");
        out.push((
            format!("delta_l{}", channel.l),
            subtract(&with_channel, &local),
        ));
    }
    (out, groups)
}

#[test]
fn per_channel_matrices_match_pyscf() {
    let reference = reference();
    let mut worst = 0.0_f64;
    let mut worst_where = String::new();
    let mut checked = 0usize;

    for (element, ecp_name) in [("Zn", "lanl2dz-zn"), ("Cu", "lanl2dz-cu")] {
        let (matrices, groups) = channel_matrices(element, ecp_name);
        for (channel, matrix) in matrices {
            for (l, functions) in &groups {
                for i in functions {
                    for j in functions {
                        let key = (element.to_string(), channel.clone(), *i, *j);
                        let expected = *reference
                            .get(&key)
                            .unwrap_or_else(|| panic!("CSV 里没有 {key:?}"));
                        let value = matrix[(*i, *j)];
                        let difference = (value - expected).abs();
                        checked += 1;
                        if difference > worst {
                            worst = difference;
                            worst_where = format!(
                                "{element} {channel} l={l} ({i},{j})：本项目 {value:.17e} vs PySCF {expected:.17e}"
                            );
                        }
                    }
                }
            }
        }
    }

    println!(
        "按通道与 PySCF 比对了 {checked} 个矩阵元；最大绝对差 {worst:.3e} Ha（容差 {TOLERANCE:.0e}）"
    );
    println!("最大偏差处：{worst_where}");
    assert!(
        checked > 1000,
        "比对的矩阵元太少（{checked}），参考集可能不完整"
    );
    assert!(
        worst < TOLERANCE,
        "ECP 通道与 PySCF 的最大差 {worst:.3e} Ha 超过 {TOLERANCE:.0e}：{worst_where}"
    );
}
