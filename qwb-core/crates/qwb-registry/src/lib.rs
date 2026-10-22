//! K1 数据模型（project.md §4 K1）：基组 / ECP / 泛函 / 网格 / 溶剂 / 分子 / 注册表。
//!
//! # 本 crate 的职责边界
//!
//! - **Rust 类型是唯一真相源**：schema 由 `schemars` 从这些类型生成（C-01 定案），
//!   因此 `qwb-contracts/schema/*.schema.json` 与代码不可能不一致。
//! - 反序列化一律 `#[serde(deny_unknown_fields)]`：拼错字段名必须报错，不能静默忽略。
//! - 加载器（`Registry`、`load_pair`）属于 D-01；本 crate 目前只提供类型、schema 生成
//!   与**文件校验**（`data verify` 的后端）。
//!
//! # 单位（§12.2）
//!
//! 数据文件里的长度是 **Bohr**（基组指数、ECP 指数、网格 cutoff），温度是 K，
//! 介电常数是无量纲。分子坐标例外：`.xyz`/`.mol` 与 `engine-api` 用 Å，字段名后缀
//! `_angstrom` 的地方就是 Å。
#![deny(unsafe_code)]
#![deny(missing_docs)]

#[macro_use]
mod macros;

pub mod basis;
pub mod common;
pub mod ecp;
pub mod elements;
pub mod functional;
pub mod grid;
pub mod loader;
pub mod molecule;
pub mod registry;
pub mod schema;
pub mod sha256;
pub mod solvent;
pub mod verify;

pub use basis::{Angular, BasisRole, BasisSet, ContractionType, Normalization, Shell, ShellBlock};
pub use common::{Citation, DataCommon, License};
pub use ecp::{Ecp, EcpLocalPart, EcpNonlocalPart, EcpTerm, RPowerConvention};
pub use functional::{Functional, FunctionalComponent, FunctionalRequires, StabilityRules};
pub use grid::{GridLevelDef, GridLevelName};
pub use loader::Registry;
pub use molecule::{MoleculeAtom, MoleculeFile};
pub use registry::{RegistryEntry, RegistryManifest};
pub use solvent::{CavityModel, Solvent};
pub use verify::{DataKind, Issue, Report, Severity};

/// 本 crate 实现的 K1 版本。
pub const K1_VERSION: &str = "1.0.0";
