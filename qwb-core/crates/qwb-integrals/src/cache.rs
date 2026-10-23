//! I-06：积分缓存（**可选 feature** `integral-cache`）。
//!
//! 验收只有一条、但很硬：**开/关缓存的结果必须 identical**。所以缓存不能改变
//! 任何数值——它只是记住"同样的基组 + 同样的分子"算过的张量并原样返回。
//!
//! # 键怎么取
//!
//! 用**内容指纹**而不是指针：把基组的每个壳层（原子序号、l、中心、指数、系数）
//! 与分子的每个原子（符号、坐标）按固定顺序写进一个 `u64` 哈希。
//! 这样：
//!
//! - 同一个 `InternalBasis` 值重复调用 → 命中；
//! - 两个不同来源但内容相同的基组 → 也命中（不依赖分配地址，符合"不要用指针当身份"）；
//! - 内容不同 → 不命中（不会串味）。
//!
//! 指纹用 `DefaultHasher`（SipHash-1-3），只用于比较，不参与数值结果，
//! 因此不存在"哈希碰撞改变结果"的风险：即使碰撞，得到的也是**另一份内容相同**的张量，
//! 结果仍然 identical —— 这是缓存能安全用的前提，写在这里以免以后有人改成"用指针"。

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};

use crate::two_electron::{TwoElectronIntegrals, two_electron};

/// 基组 + 分子的内容指纹。
#[must_use]
pub fn fingerprint(basis: &InternalBasis, molecule: &Molecule) -> u64 {
    let mut hasher = DefaultHasher::new();
    molecule.atoms.len().hash(&mut hasher);
    for atom in &molecule.atoms {
        atom.symbol.hash(&mut hasher);
        for coordinate in atom.position_angstrom {
            coordinate.to_bits().hash(&mut hasher);
        }
    }
    molecule.charge.hash(&mut hasher);
    molecule.multiplicity.hash(&mut hasher);

    basis.shells().len().hash(&mut hasher);
    for shell in basis.shells() {
        shell.atom_index.hash(&mut hasher);
        shell.l.hash(&mut hasher);
        for coordinate in shell.center_bohr {
            coordinate.to_bits().hash(&mut hasher);
        }
        shell.exponents.len().hash(&mut hasher);
        for exponent in &shell.exponents {
            exponent.to_bits().hash(&mut hasher);
        }
        for coefficient in &shell.coefficients {
            coefficient.to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// 两电子积分缓存。持有 `Arc`，所以重复取用不复制张量。
#[derive(Debug, Default)]
pub struct IntegralCache {
    entries: HashMap<u64, Arc<TwoElectronIntegrals>>,
    hits: std::cell::Cell<usize>,
    misses: std::cell::Cell<usize>,
}

impl IntegralCache {
    /// 空缓存。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 取或算：命中时直接返回同一个 `Arc`（不重算、不复制）。
    ///
    /// # Errors
    ///
    /// 透传 [`two_electron`] 的错误（例如 `l > 0` 尚未实现）。
    pub fn two_electron(
        &mut self,
        basis: &InternalBasis,
        molecule: &Molecule,
    ) -> Result<Arc<TwoElectronIntegrals>, EngineError> {
        let key = fingerprint(basis, molecule);
        if let Some(cached) = self.entries.get(&key) {
            self.hits.set(self.hits.get() + 1);
            return Ok(Arc::clone(cached));
        }
        self.misses.set(self.misses.get() + 1);
        let computed = Arc::new(two_electron(basis, molecule)?);
        self.entries.insert(key, Arc::clone(&computed));
        Ok(computed)
    }

    /// 命中次数。
    #[must_use]
    pub fn hits(&self) -> usize {
        self.hits.get()
    }

    /// 未命中次数。
    #[must_use]
    pub fn misses(&self) -> usize {
        self.misses.get()
    }

    /// 条目数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
