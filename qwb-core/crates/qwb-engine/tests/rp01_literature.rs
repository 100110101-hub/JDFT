//! RP-01 的**外部尺子**：三个反应的文献势垒，逐行钉在来源自己说过的话上（DA-08）。
//!
//! `project-phase2.md` §6 的 RP-01 判据是「三个反应（HCN↔HNC、CH₃NC↔CH₃CN、H₂CO→H₂+CO）的
//! 势垒与**文献**差 < 1.5 kcal/mol（DFT 水平）」。判据要成立，先得有三条可引用的文献值，
//! 而且必须知道每个数的**理论级别**与**含不含零点能** —— 否则「差 1.5 kcal/mol」是一句空话。
//!
//! 数据在 `tests/reference/rp01_literature_barriers.csv`（生成器
//! `scripts/gen_rp01_literature.py`，逐行带 `source_doi`/`source_url`）。这个文件里**不另写
//! 一份**文献值：读数、判级别、比零点能口径，全部走那份 CSV。文献值的可读范围与三条来源的
//! 原话见生成器的模块文档（哪一条只有摘要、哪一条读到了全文，都写在那儿）。
//!
//! # 这一份为什么是「测试」而不是「参考数据」
//!
//! 参考集（DA-08）钉的是**外部实现/实验**给的值。这一份的每一行都是论文里**手抄**的数
//! （`--check` 证明的是 CSV 与生成器常量逐字一致，不是「能被重算」）。本项目的实测值**不在**
//! 这份 CSV 里 —— 那是 `rp01_barriers_gap.rs` 里按实测钉住的常数，两者分开才不会出现
//! 「拿自算值冒充文献值」这种最难查的错误。
use std::path::PathBuf;

/// CSV 的一行（只保留这个测试要用的列）。
struct LiteratureRow {
    reaction: String,
    role: String,
    barrier_kcal_per_mol: f64,
    uncertainty_kcal_per_mol: Option<f64>,
    zpe_included: String,
    level_of_theory: String,
    quote: String,
    source_doi: String,
    source_url: String,
}

/// 极简 CSV 切分：字段里**有逗号**（三份来源的引文都有），所以不能 `split(',')`。
/// 支持双引号包起来的字段与 `""` 转义 —— 只需要这么多。
fn split_csv(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '"' if quoted => {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            }
            '"' => quoted = true,
            ',' if !quoted => fields.push(std::mem::take(&mut current)),
            other => current.push(other),
        }
    }
    fields.push(current);
    fields
}

fn rows() -> Vec<LiteratureRow> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference/rp01_literature_barriers.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("读不到 {}: {error}", path.display()));
    let header = split_csv(text.lines().next().expect("表头"));
    let index = |name: &str| {
        header
            .iter()
            .position(|field| field == name)
            .unwrap_or_else(|| panic!("表头里没有 {name}：{header:?}"))
    };
    let (reaction, role) = (index("reaction"), index("role"));
    let (barrier, uncertainty) = (
        index("barrier_kcal_per_mol"),
        index("uncertainty_kcal_per_mol"),
    );
    let (zpe, level) = (index("zpe_included"), index("level_of_theory"));
    let (quote, doi, url) = (index("quote"), index("source_doi"), index("source_url"));
    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_csv(line);
        assert_eq!(
            fields.len(),
            header.len(),
            "第 {} 行的列数不对",
            line_no + 1
        );
        out.push(LiteratureRow {
            reaction: fields[reaction].clone(),
            role: fields[role].clone(),
            barrier_kcal_per_mol: fields[barrier]
                .parse()
                .unwrap_or_else(|_| panic!("第 {} 行的势垒不是数", line_no + 1)),
            uncertainty_kcal_per_mol: if fields[uncertainty].is_empty() {
                None
            } else {
                Some(fields[uncertainty].parse().expect("uncertainty"))
            },
            zpe_included: fields[zpe].clone(),
            level_of_theory: fields[level].clone(),
            quote: fields[quote].clone(),
            source_doi: fields[doi].clone(),
            source_url: fields[url].clone(),
        });
    }
    out
}

/// 每条文献值都必须带上「理论级别 + 零点能口径 + 来源原话 + DOI/URL」四件事，
/// 而且三个反应各有**恰好一条** `computed` 行（实验值只作参照，不是判据要比的那一栏）。
#[test]
fn every_pinned_barrier_carries_its_level_and_its_zero_point_scope() {
    let rows = rows();
    assert_eq!(rows.len(), 5, "三条文献值 + 两条它们引的实验值");

    for row in &rows {
        assert!(
            !row.level_of_theory.is_empty(),
            "{} / {}: 没有理论级别",
            row.reaction,
            row.role
        );
        assert!(
            ["yes", "no", "not_stated"].contains(&row.zpe_included.as_str()),
            "{} / {}: zpe_included = {:?} 不是 yes/no/not_stated",
            row.reaction,
            row.role,
            row.zpe_included
        );
        assert!(
            row.quote.len() > 20,
            "{} / {}: 引文太短（必须是来源自己的话）",
            row.reaction,
            row.role
        );
        assert!(
            row.source_doi.starts_with("10."),
            "{} / {}: DOI",
            row.reaction,
            row.role
        );
        assert!(
            row.source_url.starts_with("https://"),
            "{} / {}: URL",
            row.reaction,
            row.role
        );
        assert!(row.barrier_kcal_per_mol.is_finite() && row.barrier_kcal_per_mol > 0.0);
    }

    for reaction in ["r1_hcn_hcnc", "r2_ch3nc_ch3cn", "r3_h2co_h2_co"] {
        let computed: Vec<&LiteratureRow> = rows
            .iter()
            .filter(|row| row.reaction == reaction && row.role == "computed")
            .collect();
        assert_eq!(computed.len(), 1, "{reaction}: 应当恰好一条 computed 行");
    }

    let computed = |reaction: &str| -> &LiteratureRow {
        rows.iter()
            .find(|row| row.reaction == reaction && row.role == "computed")
            .unwrap_or_else(|| panic!("{reaction} 没有 computed 行"))
    };

    // ── 三条钉住的数（与生成器里的常量逐字一致）─────────────────────────────
    // HCN↔HNC：16 809.4 cm⁻¹ = 48.0605 kcal/mol（ic-MRCI/aug-cc-pCV6Z + BODC + 相对论；
    // **不含** ZPE —— 来源的 Table 13 把 barrier 拆成 ∆E_BO + ∆E_BODC + ∆E_rel 三项，
    // ZPE 在 Table 12 里是单独一行）。
    let hcn = computed("r1_hcn_hcnc");
    assert!(
        (hcn.barrier_kcal_per_mol - 48.0605).abs() < 1.0e-3,
        "{}",
        hcn.barrier_kcal_per_mol
    );
    assert_eq!(hcn.zpe_included, "no", "HCN↔HNC 的文献势垒是电子势垒");
    assert!(hcn.level_of_theory.contains("ic-MRCI"));
    assert!(hcn.source_doi == "10.1021/acs.jpca.7b10483");

    // CH₃NC↔CH₃CN：来源只给「∼41 kcal/mol」，并且自己说四重激发值 1 kcal/mol —— 所以这条
    // 尺子本身的精度（±1）就吃掉了 1.5 kcal/mol 判据的大半；ZPE 口径来源没写。
    let mecn = computed("r2_ch3nc_ch3cn");
    assert!((mecn.barrier_kcal_per_mol - 41.0).abs() < 1.0e-9);
    assert_eq!(mecn.uncertainty_kcal_per_mol, Some(1.0));
    assert_eq!(mecn.zpe_included, "not_stated");
    assert!(mecn.level_of_theory.contains("SDQ-MBPT(4)/DZP"));
    assert!(mecn.source_doi == "10.1063/1.436568");

    // H₂CO→H₂+CO：81.9 ± 0.3 kcal/mol，**含** ZPE（来源原话 "including zero point vibrational
    // effects"）—— 与本项目的电子势垒不可直接比。
    let h2co = computed("r3_h2co_h2_co");
    assert!((h2co.barrier_kcal_per_mol - 81.9).abs() < 1.0e-9);
    assert_eq!(h2co.uncertainty_kcal_per_mol, Some(0.3));
    assert_eq!(h2co.zpe_included, "yes");
    assert!(h2co.source_doi == "10.1063/1.481788");
}
