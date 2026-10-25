//! `qwb_engine`：QuantumWorkbench 引擎的 Python 绑定（K6 冻结接口 / P-06）。
//!
//! 实现约定（都来自 `qwb-contracts/studio/python/qwb_engine.pyi`）：
//!
//! * **GIL 释放**：`Engine::run` 在 `Python::allow_threads` 里跑完整的一次 SCF，
//!   Python 侧的多线程因此是真的并行（`tests/test_engine_binding.py` 有并发用例）；
//! * **Reporter 批量推送**：引擎跑的时候 GIL 已经放开，所以不能在迭代中逐行回调 Python。
//!   Rust 侧先把日志/进度收进缓冲区，**整批**交给 Python 的 reporter；
//! * **矩阵零拷贝不做**：`JobResult` 的矩阵通过 `numpy.array(list)` 构造（< 500×500 可接受，R-18）；
//! * **错误映射**：`EngineError` 的四个子类与 K5 退出码一一对应，实例上带 `exit_code`。
//!
//! 这个 crate 是**独立 workspace**（`Cargo.toml` 里有空 `[workspace]`），原因见那里的注释：
//! `extension-module` 的链接属性不能传染给 `qwb-core`。

#![deny(unsafe_code)]
// pyo3 0.22 的宏会展开出 `cfg(feature = "gil-refs")`（上游遗留），本 crate 不声明该 feature。
#![allow(unexpected_cfgs)]
// pyo3 的 #[pymethods] 包装层会在宏展开里做一次 PyErr -> PyErr 的 ?，clippy 会点名到
// 我们的函数签名那一行。这是上游生成代码，不是本 crate 的写法。
#![allow(clippy::useless_conversion)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use qwb_engine_api::{
    Atom as ApiAtom, DataSnapshot as ApiDataSnapshot, Engine as ApiEngine, EngineError as ApiError,
    EnvInfo as ApiEnvInfo, GridLevel, JobInput as ApiJobInput, JobResult as ApiJobResult, JobType,
    Molecule as ApiMolecule, Reporter as ApiReporter, TaskOptions as ApiTaskOptions,
};

// ---------------------------------------------------------------------------
// K5 退出码对应的异常族
// ---------------------------------------------------------------------------

create_exception!(qwb_engine, EngineError, PyException);
create_exception!(qwb_engine, InvalidInputError, EngineError);
create_exception!(qwb_engine, ScfNotConvergedError, EngineError);
create_exception!(qwb_engine, DataError, EngineError);
create_exception!(qwb_engine, UnsupportedError, EngineError);

fn error_with_code(py: Python<'_>, err: PyErr, code: i32) -> PyErr {
    if let Ok(value) = err.value_bound(py).downcast() {
        let _ = value.setattr("exit_code", code);
    }
    err
}

/// 把引擎错误翻译成 K6 的异常族（带 K5 退出码）。
fn to_py_error(py: Python<'_>, error: ApiError) -> PyErr {
    let message = error.to_string();
    match error {
        ApiError::InvalidInput(_) => {
            error_with_code(py, PyErr::new::<InvalidInputError, _>(message), 1)
        }
        ApiError::ScfNotConverged { iters, max } => {
            let err = PyErr::new::<ScfNotConvergedError, _>(message);
            if let Ok(value) = err.value_bound(py).downcast() {
                let _ = value.setattr("exit_code", 2);
                let _ = value.setattr("iters", iters);
                let _ = value.setattr("max_iters", max);
            }
            err
        }
        ApiError::Data(_) => error_with_code(py, PyErr::new::<DataError, _>(message), 3),
        ApiError::Unsupported(_) | ApiError::Io(_) | ApiError::Internal(_) => {
            error_with_code(py, PyErr::new::<UnsupportedError, _>(message), 4)
        }
        _ => error_with_code(py, PyErr::new::<UnsupportedError, _>(message), 4),
    }
}

// ---------------------------------------------------------------------------
// Atom / Molecule
// ---------------------------------------------------------------------------

/// 单个原子（坐标单位 Å）。
#[pyclass(module = "qwb_engine")]
#[derive(Debug, Clone)]
pub struct Atom {
    /// 元素符号。
    #[pyo3(get, set)]
    pub symbol: String,
    /// Å 坐标。
    #[pyo3(get, set)]
    pub position_angstrom: (f64, f64, f64),
}

impl Atom {
    fn to_api(&self) -> ApiAtom {
        ApiAtom::new(
            self.symbol.clone(),
            [
                self.position_angstrom.0,
                self.position_angstrom.1,
                self.position_angstrom.2,
            ],
        )
    }
}

#[pymethods]
impl Atom {
    #[new]
    fn new(symbol: String, position_angstrom: (f64, f64, f64)) -> Self {
        Self {
            symbol,
            position_angstrom,
        }
    }

    /// 以 Bohr 为单位的坐标。
    fn position_bohr(&self) -> (f64, f64, f64) {
        let [x, y, z] = self.to_api().position_bohr();
        (x, y, z)
    }

    fn __repr__(&self) -> String {
        format!(
            "Atom(\"{}\", ({}, {}, {}))",
            self.symbol,
            self.position_angstrom.0,
            self.position_angstrom.1,
            self.position_angstrom.2
        )
    }
}

/// 分子：原子 + 总电荷 + 自旋多重度。
#[pyclass(module = "qwb_engine")]
#[derive(Debug)]
pub struct Molecule {
    /// 原子列表。
    #[pyo3(get, set)]
    pub atoms: Vec<Py<Atom>>,
    /// 总电荷。
    #[pyo3(get, set)]
    pub charge: i32,
    /// 自旋多重度。
    #[pyo3(get, set)]
    pub multiplicity: u32,
}

impl Molecule {
    fn to_api(&self, py: Python<'_>) -> PyResult<ApiMolecule> {
        let mut atoms = Vec::with_capacity(self.atoms.len());
        for atom in &self.atoms {
            atoms.push(atom.borrow(py).to_api());
        }
        Ok(ApiMolecule::new(atoms, self.charge, self.multiplicity))
    }
}

#[pymethods]
impl Molecule {
    #[new]
    #[pyo3(signature = (atoms, charge = 0, multiplicity = 1))]
    fn new(atoms: Vec<Py<Atom>>, charge: i32, multiplicity: u32) -> Self {
        Self {
            atoms,
            charge,
            multiplicity,
        }
    }

    /// 原子数。
    fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// 语义校验（空分子、未知元素、多重度与电子数不匹配等）。
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        let outcome = self.to_api(py)?.validate();
        outcome.map_err(|error| to_py_error(py, error))
    }

    fn __len__(&self) -> usize {
        self.atoms.len()
    }
}

// ---------------------------------------------------------------------------
// TaskOptions
// ---------------------------------------------------------------------------

/// 数值控制参数（全部有默认值）。
#[pyclass(module = "qwb_engine")]
#[derive(Debug, Clone)]
pub struct TaskOptions {
    /// SCF 最大迭代步数。
    #[pyo3(get, set)]
    pub scf_max_iter: u32,
    /// 能量收敛阈值（Hartree）。
    #[pyo3(get, set)]
    pub scf_energy_threshold_hartree: f64,
    /// 密度矩阵收敛阈值。
    #[pyo3(get, set)]
    pub scf_density_threshold: f64,
    /// DIIS 历史长度。
    #[pyo3(get, set)]
    pub diis_history: u32,
    /// 线程数；0 = 自动。
    #[pyo3(get, set)]
    pub threads: u32,
    /// 几何优化最大步数。
    #[pyo3(get, set)]
    pub max_opt_steps: u32,
    /// 优化收敛阈值（最大力分量，Hartree/Bohr）。
    #[pyo3(get, set)]
    pub opt_gradient_threshold_hartree_per_bohr: f64,
    /// 数值 Hessian 位移步长（Å）。
    #[pyo3(get, set)]
    pub freq_step_angstrom: f64,
    /// 与基组配对的 ECP 名称；None = 全电子。
    #[pyo3(get, set)]
    pub ecp_name: Option<String>,
}

impl TaskOptions {
    fn to_api(&self) -> ApiTaskOptions {
        let mut out = ApiTaskOptions::default();
        out.scf_max_iter = self.scf_max_iter;
        out.scf_energy_threshold_hartree = self.scf_energy_threshold_hartree;
        out.scf_density_threshold = self.scf_density_threshold;
        out.diis_history = self.diis_history;
        out.threads = self.threads;
        out.max_opt_steps = self.max_opt_steps;
        out.opt_gradient_threshold_hartree_per_bohr = self.opt_gradient_threshold_hartree_per_bohr;
        out.freq_step_angstrom = self.freq_step_angstrom;
        out.ecp_name = self.ecp_name.clone();
        out
    }

    fn from_api(options: &ApiTaskOptions) -> Self {
        Self {
            scf_max_iter: options.scf_max_iter,
            scf_energy_threshold_hartree: options.scf_energy_threshold_hartree,
            scf_density_threshold: options.scf_density_threshold,
            diis_history: options.diis_history,
            threads: options.threads,
            max_opt_steps: options.max_opt_steps,
            opt_gradient_threshold_hartree_per_bohr: options
                .opt_gradient_threshold_hartree_per_bohr,
            freq_step_angstrom: options.freq_step_angstrom,
            ecp_name: options.ecp_name.clone(),
        }
    }
}

#[pymethods]
impl TaskOptions {
    #[new]
    fn new() -> Self {
        Self::from_api(&ApiTaskOptions::default())
    }

    /// 取值区间校验。
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        let outcome = self.to_api().validate();
        outcome.map_err(|error| to_py_error(py, error))
    }
}

// ---------------------------------------------------------------------------
// JobInput
// ---------------------------------------------------------------------------

fn parse_grid_level(text: &str) -> PyResult<GridLevel> {
    Ok(match text.to_ascii_lowercase().as_str() {
        "coarse" => GridLevel::Coarse,
        "normal" => GridLevel::Normal,
        "fine" => GridLevel::Fine,
        "veryfine" => GridLevel::VeryFine,
        other => {
            return Err(PyErr::new::<InvalidInputError, _>(format!(
                "unknown grid_level `{other}`: expected coarse|normal|fine|veryfine"
            )))
        }
    })
}

fn parse_job_type(text: &str) -> PyResult<JobType> {
    Ok(match text.to_ascii_lowercase().as_str() {
        "sp" => JobType::Sp,
        "opt" => JobType::Opt,
        "freq" => JobType::Freq,
        other => {
            return Err(PyErr::new::<InvalidInputError, _>(format!(
                "unknown jobtype `{other}`: expected sp|opt|freq"
            )))
        }
    })
}

/// 一次作业的完整输入。
#[pyclass(module = "qwb_engine")]
#[derive(Debug)]
pub struct JobInput {
    /// 分子。
    #[pyo3(get, set)]
    pub molecule: Py<Molecule>,
    /// 基组键名。
    #[pyo3(get, set)]
    pub basis_name: String,
    /// 泛函键名（HF 用 `"hf"`）。
    #[pyo3(get, set)]
    pub method_name: String,
    /// 网格等级（`coarse|normal|fine|veryfine`）。
    pub grid_level: String,
    /// 溶剂键名（`"vacuum"` = 气相）。
    #[pyo3(get, set)]
    pub solvent: String,
    /// 作业类型（`sp|opt|freq`）。
    pub jobtype: String,
    /// 数值控制。
    #[pyo3(get, set)]
    pub options: Py<TaskOptions>,
}

impl JobInput {
    fn to_api(&self, py: Python<'_>) -> PyResult<ApiJobInput> {
        let molecule = self.molecule.borrow(py).to_api(py)?;
        let options = self.options.borrow(py).to_api();
        Ok(ApiJobInput::new(
            molecule,
            self.basis_name.clone(),
            self.method_name.clone(),
            parse_grid_level(&self.grid_level)?,
            self.solvent.clone(),
            parse_job_type(&self.jobtype)?,
            options,
        ))
    }
}

#[pymethods]
impl JobInput {
    #[new]
    #[pyo3(signature = (molecule, basis_name, method_name, grid_level = "normal", solvent = "vacuum", jobtype = "sp", options = None))]
    #[allow(clippy::too_many_arguments)] // K6 冻结的签名就是这么多参数
    fn new(
        py: Python<'_>,
        molecule: Py<Molecule>,
        basis_name: String,
        method_name: String,
        grid_level: &str,
        solvent: &str,
        jobtype: &str,
        options: Option<Py<TaskOptions>>,
    ) -> PyResult<Self> {
        parse_grid_level(grid_level)?;
        parse_job_type(jobtype)?;
        let options = match options {
            Some(options) => options,
            None => Py::new(py, TaskOptions::new())?,
        };
        Ok(Self {
            molecule,
            basis_name,
            method_name,
            grid_level: grid_level.to_string(),
            solvent: solvent.to_string(),
            jobtype: jobtype.to_string(),
            options,
        })
    }

    #[getter]
    fn grid_level(&self) -> &str {
        &self.grid_level
    }

    #[setter]
    fn set_grid_level(&mut self, value: &str) -> PyResult<()> {
        parse_grid_level(value)?;
        self.grid_level = value.to_string();
        Ok(())
    }

    #[getter]
    fn jobtype(&self) -> &str {
        &self.jobtype
    }

    #[setter]
    fn set_jobtype(&mut self, value: &str) -> PyResult<()> {
        parse_job_type(value)?;
        self.jobtype = value.to_string();
        Ok(())
    }

    /// 语义校验（分子、键名、选项区间）。
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        let outcome = self.to_api(py)?.validate();
        outcome.map_err(|error| to_py_error(py, error))
    }
}

// ---------------------------------------------------------------------------
// DataSnapshot / EnvInfo / JobResult
// ---------------------------------------------------------------------------

/// 数据快照：相对路径 -> sha256。
#[pyclass(module = "qwb_engine")]
#[derive(Debug, Clone, Default)]
pub struct DataSnapshot {
    entries: Vec<(String, String)>,
}

impl DataSnapshot {
    fn from_api(snapshot: &ApiDataSnapshot) -> Self {
        Self {
            entries: snapshot
                .iter()
                .map(|(path, digest)| (path.to_string(), digest.to_string()))
                .collect(),
        }
    }

    fn to_api(&self) -> ApiDataSnapshot {
        let mut snapshot = ApiDataSnapshot::new();
        for (path, digest) in &self.entries {
            snapshot.insert(path.clone(), digest.clone());
        }
        snapshot
    }
}

#[pymethods]
impl DataSnapshot {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// 记录一条 `path -> sha256`。
    fn insert(&mut self, path: String, sha256: String) {
        self.entries.retain(|(existing, _)| existing != &path);
        self.entries.push((path, sha256));
    }

    /// 读取某路径的摘要。
    fn get(&self, path: &str) -> Option<String> {
        self.entries
            .iter()
            .find(|(existing, _)| existing == path)
            .map(|(_, digest)| digest.clone())
    }

    /// 全部条目（按插入顺序）。
    fn items(&self) -> Vec<(String, String)> {
        self.entries.clone()
    }

    /// 每个摘要都必须是 64 位小写 sha256。
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        let outcome = self.to_api().validate();
        outcome.map_err(|error| to_py_error(py, error))
    }

    fn __len__(&self) -> usize {
        self.entries.len()
    }
}

/// 复现环境（`env.json`）。
#[pyclass(module = "qwb_engine")]
#[derive(Debug, Clone)]
pub struct EnvInfo {
    inner: ApiEnvInfo,
}

#[pymethods]
impl EnvInfo {
    #[getter]
    fn engine_version(&self) -> String {
        self.inner.engine_version.clone()
    }

    #[getter]
    fn rustc_version(&self) -> String {
        self.inner.rustc_version.clone()
    }

    #[getter]
    fn cargo_lock_sha256(&self) -> String {
        self.inner.cargo_lock_sha256.clone()
    }

    #[getter]
    fn data_snapshot(&self) -> DataSnapshot {
        DataSnapshot::from_api(&self.inner.data_snapshot)
    }

    #[getter]
    fn threads(&self) -> u32 {
        self.inner.threads
    }

    #[getter]
    fn os(&self) -> String {
        self.inner.os.clone()
    }

    #[getter]
    fn created_at_utc(&self) -> String {
        self.inner.created_at_utc.clone()
    }

    #[getter]
    fn toolchain_channel(&self) -> Option<String> {
        self.inner.toolchain_channel.clone()
    }

    #[getter]
    fn crate_versions(&self) -> std::collections::BTreeMap<String, String> {
        self.inner.crate_versions.clone()
    }

    /// G5 复现门禁。
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        let outcome = self.inner.validate();
        outcome.map_err(|error| to_py_error(py, error))
    }
}

fn numpy_array(py: Python<'_>, value: Vec<Vec<f64>>) -> PyResult<PyObject> {
    let module = py.import_bound("numpy")?;
    Ok(module.getattr("array")?.call1((value,))?.into_py(py))
}

fn numpy_vector(py: Python<'_>, value: Vec<f64>) -> PyResult<PyObject> {
    let module = py.import_bound("numpy")?;
    Ok(module.getattr("array")?.call1((value,))?.into_py(py))
}

/// 一次作业的结果（`result.json` 的形态）。
#[pyclass(module = "qwb_engine")]
#[derive(Debug, Clone)]
pub struct JobResult {
    inner: ApiJobResult,
}

#[pymethods]
impl JobResult {
    #[getter]
    fn energy_hartree(&self) -> f64 {
        self.inner.energy_hartree
    }

    #[getter]
    fn gradient_hartree_per_bohr(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.gradient_hartree_per_bohr {
            None => Ok(py.None()),
            Some(matrix) => {
                let rows = (0..matrix.nrows())
                    .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
                    .collect();
                numpy_array(py, rows)
            }
        }
    }

    #[getter]
    fn hessian_hartree_per_bohr2(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.hessian_hartree_per_bohr2 {
            None => Ok(py.None()),
            Some(matrix) => {
                let rows = (0..matrix.nrows())
                    .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
                    .collect();
                numpy_array(py, rows)
            }
        }
    }

    #[getter]
    fn dipole_debye(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.dipole_debye {
            None => Ok(py.None()),
            Some(vector) => numpy_vector(py, vector.iter().copied().collect()),
        }
    }

    #[getter]
    fn mo_energies_hartree(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.mo_energies_hartree {
            None => Ok(py.None()),
            Some(vector) => numpy_vector(py, vector.iter().copied().collect()),
        }
    }

    #[getter]
    fn mo_coeffs(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.mo_coeffs {
            None => Ok(py.None()),
            Some(matrix) => {
                let rows = (0..matrix.nrows())
                    .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
                    .collect();
                numpy_array(py, rows)
            }
        }
    }

    #[getter]
    fn scf_converged(&self) -> bool {
        self.inner.scf_converged
    }

    #[getter]
    fn data_snapshot(&self) -> DataSnapshot {
        DataSnapshot::from_api(&self.inner.data_snapshot)
    }

    #[getter]
    fn log(&self) -> String {
        self.inner.log.clone()
    }

    #[getter]
    fn env(&self) -> EnvInfo {
        EnvInfo {
            inner: self.inner.env.clone(),
        }
    }

    /// 序列化成 `result.json` 的文本（K4：矩阵 list-of-list）。
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|error| {
            PyErr::new::<DataError, _>(format!("cannot serialise JobResult: {error}"))
        })
    }

    /// 从 `result.json` 的文本还原。
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        serde_json::from_str::<ApiJobResult>(text)
            .map(|inner| Self { inner })
            .map_err(|error| PyErr::new::<DataError, _>(format!("cannot parse JobResult: {error}")))
    }
}

// ---------------------------------------------------------------------------
// Reporter 桥：Rust 侧缓冲，跑完之后整批交给 Python
// ---------------------------------------------------------------------------

struct BatchReporter {
    cancelled: Arc<AtomicBool>,
    lines: Mutex<Vec<String>>,
    progress: Mutex<Vec<(String, f64)>>,
}

impl ApiReporter for BatchReporter {
    fn progress(&self, stage: &str, pct: f64) {
        if let Ok(mut guard) = self.progress.lock() {
            if guard.len() < 4096 {
                guard.push((stage.to_string(), pct));
            }
        }
    }

    fn log(&self, line: &str) {
        if let Ok(mut guard) = self.lines.lock() {
            guard.push(line.to_string());
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// 引擎门面。
#[pyclass(module = "qwb_engine")]
pub struct Engine {
    inner: ApiEngine,
    root: String,
    cancelled: Arc<AtomicBool>,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (registry_path, *, features = None))]
    fn new(py: Python<'_>, registry_path: &str, features: Option<Vec<String>>) -> PyResult<Self> {
        let registry =
            qwb_registry::Registry::open(registry_path).map_err(|error| to_py_error(py, error))?;
        let mut handle = registry.engine_handle();
        if let Some(features) = features {
            handle = handle.with_declared_features(features);
        }
        let backend = Box::new(::qwb_engine::RksBackend::new(registry));
        let engine =
            ApiEngine::with_backend(&handle, backend).map_err(|error| to_py_error(py, error))?;
        Ok(Self {
            inner: engine,
            root: handle.root().display().to_string(),
            cancelled: Arc::new(AtomicBool::new(false)),
        })
    }

    /// 数据根目录。
    #[getter]
    fn registry_root(&self) -> &str {
        &self.root
    }

    /// 执行一次作业；**整次计算在 `allow_threads` 里跑**（K6：必须释放 GIL）。
    #[pyo3(signature = (job_input, reporter = None))]
    fn run(
        &self,
        py: Python<'_>,
        job_input: &JobInput,
        reporter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<JobResult> {
        let input = job_input.to_api(py)?;
        input.validate().map_err(|error| to_py_error(py, error))?;
        let bridge = BatchReporter {
            cancelled: Arc::clone(&self.cancelled),
            lines: Mutex::new(Vec::new()),
            progress: Mutex::new(Vec::new()),
        };
        let result = py.allow_threads(|| self.inner.run(&input, &bridge));
        // 整批回调：跑的时候没有 GIL，所以这里一次性推。
        if let Some(reporter) = reporter {
            if let Ok(lines) = bridge.lines.lock() {
                for line in lines.iter() {
                    let _ = reporter.call_method1("log", (line,));
                }
            }
            if let Ok(entries) = bridge.progress.lock() {
                for (stage, pct) in entries.iter() {
                    let _ = reporter.call_method1("progress", (stage, *pct));
                }
            }
        }
        match result {
            Ok(job_result) => Ok(JobResult { inner: job_result }),
            Err(error) => Err(to_py_error(py, error)),
        }
    }

    /// 请求取消当前作业（下一次迭代边界生效）。
    fn request_cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.inner.request_cancel();
    }
}

// ---------------------------------------------------------------------------
// 便捷入口 + 模块
// ---------------------------------------------------------------------------

/// 一次性便捷入口：`run_job(input, registry_path)`。
#[pyfunction]
fn run_job(py: Python<'_>, job_input: &JobInput, registry_path: &str) -> PyResult<JobResult> {
    let engine = Engine::new(py, registry_path, None)?;
    engine.run(py, job_input, None)
}

#[pymodule]
fn qwb_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<Atom>()?;
    m.add_class::<Molecule>()?;
    m.add_class::<TaskOptions>()?;
    m.add_class::<JobInput>()?;
    m.add_class::<DataSnapshot>()?;
    m.add_class::<EnvInfo>()?;
    m.add_class::<JobResult>()?;
    m.add_class::<Engine>()?;

    m.add("EngineError", py.get_type_bound::<EngineError>())?;
    m.add(
        "InvalidInputError",
        py.get_type_bound::<InvalidInputError>(),
    )?;
    m.add(
        "ScfNotConvergedError",
        py.get_type_bound::<ScfNotConvergedError>(),
    )?;
    m.add("DataError", py.get_type_bound::<DataError>())?;
    m.add("UnsupportedError", py.get_type_bound::<UnsupportedError>())?;

    let features = PyTuple::new_bound(py, qwb_engine_api::SUPPORTED_FEATURES.iter());
    m.add("BUILTIN_REGISTRY_FEATURES", features)?;

    m.add_function(wrap_pyfunction!(run_job, m)?)?;
    // 只列**运行时真有对象**的名字。桩里的 `GridLevel` / `JobType` / `Reporter` 是
    // `typing.Literal` 与 `Protocol`（类型层面的东西），运行时不存在，列进来会让
    // `from qwb_engine import *` 直接 AttributeError —— 这一点由
    // `tests/test_engine_binding.py` 的接口一致性用例钉住。
    m.add(
        "__all__",
        vec![
            "Atom",
            "Molecule",
            "TaskOptions",
            "JobInput",
            "DataSnapshot",
            "EnvInfo",
            "JobResult",
            "EngineError",
            "DataError",
            "InvalidInputError",
            "ScfNotConvergedError",
            "UnsupportedError",
            "BUILTIN_REGISTRY_FEATURES",
            "Engine",
            "run_job",
        ],
    )?;
    Ok(())
}
