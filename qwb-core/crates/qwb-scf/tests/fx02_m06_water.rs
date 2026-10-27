//! FX-02 / DA-01 / DA-02：**M06-2X 与 M06-L 的端到端验收**（H2O，STO-3G）。
//!
//! 判据（project-phase2.md §6.3 FX-02 / §7 M2）：与 PySCF 黑盒同值 < 0.5 mHa。
//!
//! ```text
//! H2O/STO-3G   M06-2X: PySCF −75.3129666765   本实现 −75.31298...
//! H2O/STO-3G   M06-L : PySCF −75.3055076948
//! ```
//!
//! 同几何下 def2-SVP 的实测（grid-coarse）也在 0.5 mHa 内：
//! M06-2X −76.3231912（PySCF −76.3231686，差 2.3e-5）、
//! M06-L −76.3499177（PySCF −76.3499458，差 2.8e-5）。
//!
//! 注意这里只验**能量**：公式层面的交换/相关（含三个偏导）由
//! `qwb-xc/tests/fx02_m06_exchange.rs` 逐点钉死（< 1e-9）。

use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_from_data};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

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

#[test]
fn m06_functionals_match_pyscf_on_water() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    // PySCF 2.14.0（libxc）在 H2O/STO-3G、同一几何上的值
    let reference = [("m06-2x", -75.312_966_676_5), ("m06-l", -75.305_507_694_8)];
    for (functional, expected) in reference {
        let mut options = RksOptions::default();
        options.max_iter = 300;
        let result = rks_from_data(
            &registry,
            &molecule,
            "sto-3g",
            functional,
            "grid-normal",
            options,
        )
        .unwrap_or_else(|error| panic!("{functional} 应当跑通：{error}"));
        println!(
            "{functional}: E = {:.10} Ha（PySCF {expected:.10}，差 {:+.2e}），E_xc = {:.10}，iters = {}",
            result.energy_hartree,
            result.energy_hartree - expected,
            result.exchange_correlation_hartree,
            result.iterations
        );
        assert!(
            (result.energy_hartree - expected).abs() < 5.0e-4,
            "{functional}: E = {:.10} vs PySCF {expected:.10}（判据 0.5 mHa）",
            result.energy_hartree
        );
    }
}
