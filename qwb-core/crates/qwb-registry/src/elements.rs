//! 元素符号 → 原子序数（H..Kr）。
//!
//! 只覆盖第一阶段需要的元素（§2.2：不做周期表，T5 到 Kr）。表放在代码里而不是数据里，
//! 因为它是**物理常数**而不是可再分发数据：不涉及 citation/license 门禁。

/// 第一阶段支持的元素符号（原子序数 1..=36，H..Kr）。
pub const ELEMENTS: [&str; 36] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr",
];

/// 元素符号 → 原子序数；未知或超出第一阶段范围返回 `None`。
#[must_use]
pub fn atomic_number(symbol: &str) -> Option<u32> {
    ELEMENTS
        .iter()
        .position(|s| *s == symbol)
        .map(|i| u32::try_from(i + 1).expect("element index fits in u32"))
}

/// 原子序数是否在第一阶段范围内。
#[must_use]
pub fn is_supported(symbol: &str) -> bool {
    atomic_number(symbol).is_some()
}

#[cfg(test)]
mod tests {
    use super::{atomic_number, is_supported};

    #[test]
    fn known_symbols() {
        assert_eq!(atomic_number("H"), Some(1));
        assert_eq!(atomic_number("C"), Some(6));
        assert_eq!(atomic_number("O"), Some(8));
        assert_eq!(atomic_number("Cu"), Some(29));
        assert_eq!(atomic_number("Kr"), Some(36));
    }

    #[test]
    fn unknown_symbols_are_rejected() {
        assert_eq!(atomic_number("h"), None);
        assert_eq!(atomic_number("Xx"), None);
        assert_eq!(atomic_number(""), None);
        assert!(!is_supported("U"));
        assert!(is_supported("Zn"));
    }
}
