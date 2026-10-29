# -*- coding: utf-8 -*-
"""RS-05: hot-start the displaced SCFs in the Hessian from the reference density."""
import io

# ---------------- 1) OptOptions gets the initial density
path = 'qwb-core/crates/qwb-engine/src/opt.rs'
text = io.open(path, encoding='utf-8').read()
old = """    /// 自旋设置（`None`/`Rks`/`Rhf` = 闭壳层；`Uhf` = 开壳层 UHF，SC-02）。
    pub scf_type: Option<qwb_engine_api::ScfType>,
}"""
new = """    /// 自旋设置（`None`/`Rks`/`Rhf` = 闭壳层；`Uhf` = 开壳层 UHF，SC-02）。
    pub scf_type: Option<qwb_engine_api::ScfType>,
    /// 每一步 SCF 的热启动密度（`None` = 用 core guess）。
    ///
    /// 用途是 Hessian：**同一条势能面上相邻几何**的密度几乎一样，从参考密度出发
    /// SCF 迭代数能从十几次掉到几次（RS-05 的墙钟优化，见 `docs/choices.md` 七十二）。
    pub scf_initial_density: Option<qwb_linalg::Matrix>,
}"""
assert text.count(old) == 1, 'OptOptions anchor'
text = text.replace(old, new)

old = """            scf_type: None,"""
new = """            scf_type: None,
            scf_initial_density: None,"""
assert text.count(old) == 1, 'Default anchor'
text = text.replace(old, new)

old = """        let mut options = RksOptions::default();
        options.max_iter = self.scf_max_iter;
        options.energy_threshold = self.scf_energy_threshold;
        options.density_threshold = self.scf_density_threshold;
        options
    }"""
new = """        let mut options = RksOptions::default();
        options.max_iter = self.scf_max_iter;
        options.energy_threshold = self.scf_energy_threshold;
        options.density_threshold = self.scf_density_threshold;
        if let Some(density) = &self.scf_initial_density {
            options.resume_from = Some(qwb_scf::checkpoint::ScfCheckpoint::new(density, None, 0));
        }
        options
    }"""
assert text.count(old) == 1, 'scf_options anchor'
text = text.replace(old, new)
io.open(path, 'w', encoding='utf-8', newline='').write(text)

# ---------------- 2) hessian.rs: reference density + pass it to the displaced evaluations
path = 'qwb-core/crates/qwb-engine/src/hessian.rs'
text = io.open(path, encoding='utf-8').read()

old = """    let natoms = molecule.atoms.len();
    let dof = 3 * natoms;
    // 每个位移需要 ±δ 两个几何的梯度
    let displacements: Vec<(usize, usize)> = (0..dof).map(|k| (k / 3, k % 3)).collect();"""
new = """    let natoms = molecule.atoms.len();
    let dof = 3 * natoms;
    // RS-05：先用参考几何把 SCF 解出来，之后每个位移都从这条密度**热启动** ——
    // 同一条势能面上相邻几何的密度差别很小，SCF 迭代数能掉一大截（choices 七十二）。
    let mut options = options.clone();
    options.scf_initial_density = reference_density(registry, molecule, options_ref(options_ref_arg(&options)))?;
    let options = &options;
    // 每个位移需要 ±δ 两个几何的梯度
    let displacements: Vec<(usize, usize)> = (0..dof).map(|k| (k / 3, k % 3)).collect();"""
assert text.count(old) == 1, 'hessian anchor'
text = text.replace(old, new)
io.open(path, 'w', encoding='utf-8', newline='').write(text)
print('ok')
