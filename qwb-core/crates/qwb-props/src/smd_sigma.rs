//! SV-03 的第三层：**每原子 CDS 表面张力 `STS`**（上游 `SMXCDS`）。
//!
//! # 为什么不是"查表乘面积"
//!
//! `E_CDS = Σ_k (STS_k + CSSIGM)·A_k·1e-3`（kcal/mol），其中 `A_k` 是 `DAREAL` 算的表面分解
//! （本 crate 的 [`crate::smd_surface`]），而 `STS_k` 是**每原子**的表面张力，由四块拼出来：
//!
//! ```text
//! STS_i = SIGMA(Z_i)                                   （零阶表，逐元素）
//!       + Σ_j COT_ij·HSIGMA(Z_j)                       （只对 H：H 与重原子成键的修正）
//!       + 成对项 SIGMA(100+pair)                        （按"原子 i 与邻居 j 的元素"选 pair）
//! ```
//!
//! 成键权重是 `COT_ij = exp(Δ/(d_ij − (R_ij + Δ)))`（`d ≥ R + Δ` 时为 0），`R_ij` 来自
//! 15×15 的 `RKKVAL`、`Δ = 0.30 Å`；两处特例在 `SMXCDS` 里是硬写的：
//!
//! * **O–C**（`II=8, JJ=6` 或反过来）：`Δ = 0.10`、`R = 1.330`（而不是 `RKKVAL` 的 1.84）；
//! * **O–O**（`ISTS = 6`）：`Δ = 0.30`、`R = 1.80`（而不是 `RKKVAL` 的 2.75）。
//!
//! N 与 C 的成对项还各有一个**软核**：`W(d; R, Δ) = exp(1 − 1/(1 − x))/e`，`x = (d − R)/Δ`
//! （`d ≥ R + Δ` 时为 0）。C 用 `Σ_C-neighbors W` 乘 `SIGMA(101)`（C–C）与 `SIGMA(102)`
//! （C–C(2)），N 用 `(Σ COT·(Σ W)²)^1.3 · SIGMA(105)`、三键 `Σ W(d; 1.225, 0.065)·SIGMA(116)`。
//!
//! # 覆盖范围与不编造
//!
//! 分支按上游逐元素写：H(1)、C(6)、N(7)、O(8)、S(16)。这五个元素的成对项覆盖了
//! 上游全部**非零**的 pair 值（水：1/3/5/6/14/16；非水：1/3/4/5/6/10）。上游还有
//! O–Si(117)、O–S(118)、S–S(107)、S–P(115)、H–N(2)(112)、H–O(2)(113)、N–C(2)(111)
//! 这些分支，它们在本仓库的数据里**全为 0**，照原样实现只会让代码多七条永不触发的路径 ——
//! 所以这里不写，但测试会断言这些数据项确实是 0（一旦上游数据变了，那条断言会红）。
//!
//! 单位：`SIGMA`/`HSIGMA` 是 cal/(mol·Å²)，`A` 是 Å²，乘 `1e-3` 得 kcal/mol（与上游
//! `CDS_EG` 的 `*0.001D0` 一致）。
#![deny(unsafe_code)]

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::elements::atomic_number;
use qwb_registry::smd_atomic::SmdAtomic;

use crate::smd::{SmdDescriptors, cds_constant_term, cds_sigma};

/// 默认成键带宽度 `Δ`（Å）。上游 `SMXCDS`：`DELTAR = 0.30D0`。
const DEFAULT_DELTA: f64 = 0.30;
/// 上游 `SMXCDS` 里硬写的 `ISTS`；它打开 O–O 与 O–C 两处特例（见模块文档）。
const ISTS: usize = 6;

/// 成键接触权重 `COT = exp(Δ/(d − (R + Δ)))`；`d ≥ R + Δ` 时为 0。
fn cot_weight(distance: f64, reference: f64, delta: f64) -> f64 {
    let cutoff = reference + delta;
    if distance >= cutoff {
        0.0
    } else {
        (delta / (distance - cutoff)).exp()
    }
}

/// 上游的软核 `W = exp(1 − 1/(1 − x))/e`，`x = (d − R)/Δ`；`d ≥ R + Δ` 时为 0。
fn soft_weight(distance: f64, reference: f64, delta: f64) -> f64 {
    let cutoff = reference + delta;
    if distance >= cutoff {
        0.0
    } else {
        let x = (distance - reference) / delta;
        (1.0 - 1.0 / (1.0 - x)).exp() / std::f64::consts::E
    }
}

/// 每原子 CDS 表面张力 `STS`（cal/(mol·Å²)），顺序与 `molecule.atoms` 一致。
///
/// `water` 选择水相（`ICDS = 1`）或非水相（`ICDS = 2`，需要六个 Abraham 描述符）。
///
/// # Errors
///
/// 元素不在数据范围内、非水相缺描述符、`RKKVAL` 表里出现零参考键长等 → [`EngineError::Data`]。
pub fn smd_sts(
    data: &SmdAtomic,
    molecule: &Molecule,
    water: bool,
    descriptors: Option<&SmdDescriptors>,
) -> Result<Vec<f64>, EngineError> {
    let context = Context::new(data, molecule, water, descriptors)?;
    let mut sts = Vec::with_capacity(context.natoms);
    for index in 0..context.natoms {
        let mut value = cds_sigma(data, context.symbols[index], water, descriptors)
            .map_err(EngineError::Data)?;
        value += match context.atomic_numbers[index] {
            1 => context.hydrogen_contribution(index)?,
            6 => context.carbon_contribution(index)?,
            7 => context.nitrogen_contribution(index)?,
            8 => context.oxygen_contribution(index)?,
            16 => context.sulfur_contribution(index)?,
            _ => 0.0,
        };
        sts.push(value);
    }
    Ok(sts)
}

/// CDS 项的能量（kcal/mol）：`Σ_k (STS_k + CSSIGM)·A_k·1e-3`。
///
/// `areas_angstrom2` 与 `sts` 必须同序同长（每原子）；`cssigm` 由
/// [`cds_constant_term`] 给出（水相恒 0）。
///
/// # Errors
///
/// 长度不一致或出现非有限值。
pub fn cds_energy_kcal_from_sts(
    sts: &[f64],
    cssigm: f64,
    areas_angstrom2: &[f64],
) -> Result<f64, String> {
    if sts.len() != areas_angstrom2.len() {
        return Err(format!(
            "smd: {} surface tensions for {} surface areas",
            sts.len(),
            areas_angstrom2.len()
        ));
    }
    let mut energy = 0.0;
    for (sigma, area) in sts.iter().zip(areas_angstrom2) {
        if !sigma.is_finite() || !area.is_finite() || *area < 0.0 {
            return Err("smd: surface tension and area must be finite, area >= 0".to_string());
        }
        energy += (sigma + cssigm) * area;
    }
    Ok(energy / 1000.0)
}

/// 一次算完的便捷入口：`STS` + `CSSIGM` + 面积 → CDS 能量（kcal/mol）。
///
/// # Errors
///
/// 同 [`smd_sts`] 与 [`cds_energy_kcal_from_sts`]。
pub fn smd_cds_energy_kcal(
    data: &SmdAtomic,
    molecule: &Molecule,
    areas_angstrom2: &[f64],
    water: bool,
    descriptors: Option<&SmdDescriptors>,
) -> Result<f64, EngineError> {
    let sts = smd_sts(data, molecule, water, descriptors)?;
    let cssigm = cds_constant_term(data, water, descriptors);
    cds_energy_kcal_from_sts(&sts, cssigm, areas_angstrom2).map_err(EngineError::Data)
}

/// 一次调用的公共上下文（符号、原子序数、坐标、键长）。
struct Context<'a> {
    data: &'a SmdAtomic,
    molecule: &'a Molecule,
    water: bool,
    descriptors: Option<&'a SmdDescriptors>,
    symbols: Vec<&'a str>,
    atomic_numbers: Vec<usize>,
    natoms: usize,
}

impl<'a> Context<'a> {
    fn new(
        data: &'a SmdAtomic,
        molecule: &'a Molecule,
        water: bool,
        descriptors: Option<&'a SmdDescriptors>,
    ) -> Result<Self, EngineError> {
        let symbols: Vec<&'a str> = molecule
            .atoms
            .iter()
            .map(|atom| atom.symbol.as_str())
            .collect();
        let mut atomic_numbers = Vec::with_capacity(symbols.len());
        for symbol in &symbols {
            let number = atomic_number(symbol)
                .ok_or_else(|| EngineError::Data(format!("smd: unknown element `{symbol}`")))?;
            atomic_numbers.push(number as usize);
        }
        Ok(Self {
            data,
            molecule,
            water,
            descriptors,
            natoms: symbols.len(),
            symbols,
            atomic_numbers,
        })
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        let a = self.molecule.atoms[i].position_angstrom;
        let b = self.molecule.atoms[j].position_angstrom;
        let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// 元素 → 上游 `NATCNV` 类型（0 = 不参与成键权重）。
    fn element_type(&self, symbol: &str) -> usize {
        self.data
            .cds
            .bonds
            .element_types
            .get(symbol)
            .copied()
            .unwrap_or(0)
    }

    /// 两个原子之间的 `RKKVAL` 参考键长，**已经**应用 `SMXCDS` 的两处特例。
    ///
    /// 返回 `None` 表示不参与成键权重（类型为 0 或参考键长为 0）。
    fn reference(&self, i: usize, j: usize) -> Option<(f64, f64)> {
        let (zi, zj) = (self.atomic_numbers[i], self.atomic_numbers[j]);
        // O–C（上游 `II=8, JJ=6` 或反过来）：Δ = 0.10、R = 1.330。
        if (zi == 8 && zj == 6) || (zi == 6 && zj == 8) {
            return Some((1.330, 0.10));
        }
        // O–O（ISTS = 6）：Δ = 0.30、R = 1.80。
        if ISTS == 6 && zi == 8 && zj == 8 {
            return Some((1.800, 0.30));
        }
        let ti = self.element_type(self.symbols[i]);
        let tj = self.element_type(self.symbols[j]);
        if ti == 0 || tj == 0 {
            return None;
        }
        let reference = self.data.cds.bonds.rkkval_angstrom[ti - 1][tj - 1];
        if reference <= 0.0 {
            return None;
        }
        Some((reference, DEFAULT_DELTA))
    }

    fn contact(&self, i: usize, j: usize) -> f64 {
        match self.reference(i, j) {
            Some((reference, delta)) => cot_weight(self.distance(i, j), reference, delta),
            None => 0.0,
        }
    }

    /// `HSIGMA(Z_j)`：水相查表；非水相 `HSIGMA_N·SOLN + HSIGMA_A·SOLA + HSIGMA_B·SOLB`。
    fn hsigma_at(&self, symbol: &str) -> Result<f64, EngineError> {
        if self.water {
            return self
                .data
                .cds
                .water
                .hsigma_cal_per_mol_angstrom2
                .get(symbol)
                .copied()
                .ok_or_else(|| {
                    EngineError::Data(format!("smd: no aqueous HSIGMA for `{symbol}`"))
                });
        }
        let descriptors = self.descriptors.ok_or_else(|| {
            EngineError::Data(
                "smd: the nonaqueous HSIGMA needs the solvent Abraham descriptors".to_string(),
            )
        })?;
        let index = self
            .data
            .elements
            .iter()
            .position(|candidate| candidate == symbol)
            .ok_or_else(|| {
                EngineError::Data(format!("smd: `{symbol}` outside the HSIGMA table"))
            })?;
        let table = &self.data.cds.nonaqueous;
        Ok(table.hsigma_n[index] * descriptors.soln
            + table.hsigma_a[index] * descriptors.sola
            + table.hsigma_b[index] * descriptors.solb)
    }

    /// 成对修正 `SIGMA(100 + pair)`（`pair` 为上游的 1 基编号）。
    fn pair_sigma(&self, pair: usize) -> Result<f64, EngineError> {
        let slot = pair
            .checked_sub(1)
            .filter(|index| *index < 50)
            .ok_or_else(|| EngineError::Data(format!("smd: pair index {pair} outside 1..=50")))?;
        if self.water {
            return self
                .data
                .cds
                .water
                .pair_sigma_cal_per_mol_angstrom2
                .get(slot)
                .copied()
                .ok_or_else(|| {
                    EngineError::Data(format!("smd: the aqueous pair block has no slot {slot}"))
                });
        }
        let descriptors = self.descriptors.ok_or_else(|| {
            EngineError::Data(
                "smd: the nonaqueous pair sigma needs the solvent Abraham descriptors".to_string(),
            )
        })?;
        let table = &self.data.cds.nonaqueous;
        Ok(table.pair_sigma_n[slot] * descriptors.soln
            + table.pair_sigma_a[slot] * descriptors.sola
            + table.pair_sigma_b[slot] * descriptors.solb)
    }

    /// H：`Σ_j COT_ij·HSIGMA(Z_j)`，外加两条"未使用"的二阶项（`SIGMA(112/113)`，数据为 0）。
    fn hydrogen_contribution(&self, i: usize) -> Result<f64, EngineError> {
        let mut value = 0.0;
        for j in 0..self.natoms {
            if j == i {
                continue;
            }
            let cot = self.contact(i, j);
            if cot == 0.0 {
                continue;
            }
            value += cot * self.hsigma_at(self.symbols[j])?;
            match self.atomic_numbers[j] {
                // H–N–N：`SIGMA(112)`（上游标为 NOT USED BY SMD）。
                7 => {
                    for k in 0..self.natoms {
                        if k != j && self.atomic_numbers[k] == 7 {
                            value += cot * self.contact(j, k) * self.pair_sigma(12)?;
                        }
                    }
                }
                // H–O–H：`SIGMA(113)`（上游标为 NOT USED BY SMD）。
                8 => {
                    for k in 0..self.natoms {
                        if k != i && self.atomic_numbers[k] == 1 {
                            value += cot * self.contact(j, k) * self.pair_sigma(13)?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(value)
    }

    /// O：`Σ_j COT_ij·SIGMA(O–X)`，X ∈ {C 3、N 6、S 18、Si 17、P 14、O 4}。
    fn oxygen_contribution(&self, i: usize) -> Result<f64, EngineError> {
        let mut value = 0.0;
        for j in 0..self.natoms {
            if j == i {
                continue;
            }
            let cot = self.contact(i, j);
            if cot == 0.0 {
                continue;
            }
            let pair = match self.atomic_numbers[j] {
                6 => Some(3),
                7 => Some(6),
                16 => Some(18),
                14 => Some(17),
                15 => Some(14),
                8 => {
                    if ISTS == 6 {
                        Some(4)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(pair) = pair {
                value += cot * self.pair_sigma(pair)?;
            }
        }
        Ok(value)
    }

    /// N：`RTKKS^1.3·SIGMA(105) + RTKKS2·SIGMA(111) + RTKK_triple·SIGMA(116)`。
    fn nitrogen_contribution(&self, i: usize) -> Result<f64, EngineError> {
        let mut rtkks = 0.0;
        let mut rtkks2 = 0.0;
        for j in 0..self.natoms {
            if j == i || self.atomic_numbers[j] != 6 {
                continue;
            }
            let cot = self.contact(i, j);
            if cot == 0.0 {
                continue;
            }
            let carbon_type = self.element_type(self.symbols[j]);
            let mut inner = 0.0;
            let mut inner_oxygen = 0.0;
            for k in 0..self.natoms {
                if k == i || k == j {
                    continue;
                }
                let kind = self.element_type(self.symbols[k]);
                if kind == 0 || carbon_type == 0 {
                    continue;
                }
                let reference = self.data.cds.bonds.rkkval_angstrom[kind - 1][carbon_type - 1];
                if reference <= 0.0 {
                    continue;
                }
                let weight = soft_weight(self.distance(j, k), reference, DEFAULT_DELTA);
                inner += weight;
                if self.atomic_numbers[k] == 8 {
                    inner_oxygen += weight;
                }
            }
            rtkks += cot * inner * inner;
            rtkks2 += cot * inner_oxygen;
        }
        let mut value = rtkks.powf(1.3) * self.pair_sigma(5)?;
        value += rtkks2 * self.pair_sigma(11)?;
        // N–C 三键：R = 1.225、Δ = 0.065（上游的 `DELTAR = 0.065D0` 分支）。
        let mut triple = 0.0;
        for j in 0..self.natoms {
            if j == i || self.atomic_numbers[j] != 6 {
                continue;
            }
            triple += soft_weight(self.distance(i, j), 1.225, 0.065);
        }
        value += triple * self.pair_sigma(16)?;
        Ok(value)
    }

    /// C：`RTKK·SIGMA(101) + RTKK2·SIGMA(102) + (Σ_{N} COT)²·SIGMA(110)`。
    fn carbon_contribution(&self, i: usize) -> Result<f64, EngineError> {
        let carbon_type = self.element_type(self.symbols[i]);
        if carbon_type == 0 {
            return Ok(0.0);
        }
        let mut value = 0.0;
        for (pair, reference, delta) in [
            (
                1usize,
                self.data.cds.bonds.rkkval_angstrom[carbon_type - 1][carbon_type - 1],
                DEFAULT_DELTA,
            ),
            (2, 1.27, 0.07),
        ] {
            let mut kernel = 0.0;
            for j in 0..self.natoms {
                if j == i || self.atomic_numbers[j] != 6 {
                    continue;
                }
                kernel += soft_weight(self.distance(i, j), reference, delta);
            }
            value += kernel * self.pair_sigma(pair)?;
        }
        let mut contacts = 0.0;
        for j in 0..self.natoms {
            if j == i || self.atomic_numbers[j] != 7 {
                continue;
            }
            contacts += self.contact(i, j);
        }
        value += contacts * contacts * self.pair_sigma(10)?;
        Ok(value)
    }

    /// S：`Σ_j COT_ij·SIGMA(S–S 7 / S–P 15)`（上游两条都标 NOT USED BY SMD，数据为 0）。
    fn sulfur_contribution(&self, i: usize) -> Result<f64, EngineError> {
        let mut value = 0.0;
        for j in 0..self.natoms {
            if j == i {
                continue;
            }
            let cot = self.contact(i, j);
            if cot == 0.0 {
                continue;
            }
            match self.atomic_numbers[j] {
                16 => value += cot * self.pair_sigma(7)?,
                15 => value += cot * self.pair_sigma(15)?,
                _ => {}
            }
        }
        Ok(value)
    }
}
