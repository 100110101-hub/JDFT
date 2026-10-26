//! qwb-engine-api: 冻结的引擎接口。任何破坏性变更需经 CHANGELOG。
//!
//! 本 crate 只承载契约（project.md K2）：数据传递类型、错误枚举、[`Reporter`] 观测面，
//! 以及 [`Engine`] 门面。**引擎实现不在本 crate 内** —— 积分、XC、SCF、梯度分别由
//! `qwb-integrals` / `qwb-xc` / `qwb-scf` / `qwb-gradient` 提供，并在顶层
//! `qwb-engine` 里以 [`EngineBackend`] 注入。
//!
//! 冻结约定（K2）：
//!
//! - 公共 struct/enum 一律 `#[non_exhaustive]`，防止下游依赖字段全集。
//! - 公共字段后缀 `_hartree` / `_bohr` / `_debye` / `_angstrom`，禁止裸 `energy`。
//! - 错误 enum 只允许新增变体，禁止删除。
//! - 所有跨线程类型 `Send + Sync`。
//! - [`Engine::run`] 接受 `&dyn Reporter`，引擎不得直接写 stdout。
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc)]
#![deny(clippy::missing_panics_doc)]

pub mod correlation;
pub mod engine;
pub mod error;
pub mod input;
pub mod molecule;
pub mod properties;
pub mod registry;
pub mod reporter;
pub mod result;
pub mod solvation;

pub use correlation::{CorrelationResult, Mp2Kind, Mp2Options, ScfType, SpinResult};
pub use engine::{Engine, EngineBackend, SUPPORTED_FEATURES};
pub use error::EngineError;
pub use input::{ExcitedStatesOptions, GridLevel, JobInput, JobType, TaskOptions};
pub use molecule::{Atom, Molecule};
pub use properties::{
    ExcitedStateMethod, ExcitedStates, Multipoles, NmrReference, NmrShieldings, NucleusShielding,
    Polarizability, SumRules, Transition, TransitionContribution,
};
pub use registry::Registry;
pub use reporter::Reporter;
pub use result::{DataSnapshot, EnvInfo, JobResult};
pub use solvation::{SolvationResult, SolventModel};
