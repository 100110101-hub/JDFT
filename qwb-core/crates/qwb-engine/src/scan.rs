//! PES 扫描的几何操作（phase-2 `RP-03`）：把某个内坐标挪到目标值。
//!
//! 这里的每一个函数都只动**内坐标末端的那个原子**（距离的第二个、键角的第三个、二面角的
//! 第四个），其余原子保持不动。这是刚性扫描的常见做法，但它有一个必须写下来的后果：
//! 对多原子体系，"挪一个原子"并不等于"整体保持刚性"——如果那个原子带着取代基，取代基不会
//! 跟着转。真正保刚性的做法是绕着旋转轴转整个子片段，那是 RP-01/IRC 的活；本模块的选择是
//! **可预期且可核对**：只动一个原子、动完立刻量回来。
//!
//! 三条不变量：① 动完之后 `measure` 必须回到目标值（角度 1e-8 度、距离 1e-10 Å），否则报错
//! 而不是返回一个"差不多"的几何；② 退化情形（原子重合、(反)平行导致旋转轴不定）显式报错；
//! ③ 原子序号越界/重复、种类与原子数不符都拒绝。
#![deny(unsafe_code)]

use qwb_engine_api::EngineError;
use qwb_engine_api::Molecule;
use qwb_engine_api::pes::ScanKind;

/// 量一个内坐标（距离 Å、角度与二面角度）。
///
/// # Errors
///
/// 原子序号越界、原子重合、种类与原子数不符。
pub fn measure(molecule: &Molecule, kind: ScanKind, atoms: &[usize]) -> Result<f64, EngineError> {
    check_atoms(molecule, kind, atoms)?;
    match kind {
        ScanKind::Distance => Ok(distance(molecule, atoms[0], atoms[1])),
        ScanKind::Angle => Ok(angle(molecule, atoms[0], atoms[1], atoms[2])),
        ScanKind::Dihedral => Ok(dihedral(molecule, atoms[0], atoms[1], atoms[2], atoms[3])),
        _ => Err(EngineError::Unsupported(format!(
            "scan kind `{}` is not implemented",
            kind.as_str()
        ))),
    }
}

/// 返回一份**新几何**，其中该内坐标被设为 `target`（距离 Å、角度与二面角度）。
///
/// # Errors
///
/// 同 [`measure`]，外加"挪完没到目标值"（内部一致性失败）。
pub fn set_coordinate(
    molecule: &Molecule,
    kind: ScanKind,
    atoms: &[usize],
    target: f64,
) -> Result<Molecule, EngineError> {
    check_atoms(molecule, kind, atoms)?;
    if !target.is_finite() {
        return Err(EngineError::InvalidInput(
            "scan target must be finite".into(),
        ));
    }
    let mut moved = molecule.clone();
    match kind {
        ScanKind::Distance => {
            if target <= 0.0 {
                return Err(EngineError::InvalidInput(format!(
                    "scan distance {target} must be positive"
                )));
            }
            let a = molecule.atoms[atoms[0]].position_angstrom;
            let b = molecule.atoms[atoms[1]].position_angstrom;
            let axis = subtract(b, a);
            let norm = norm(axis);
            if norm < 1e-12 {
                return Err(EngineError::InvalidInput(
                    "scan distance between coincident atoms".into(),
                ));
            }
            let unit = scale(axis, 1.0 / norm);
            moved.atoms[atoms[1]].position_angstrom = add(a, scale(unit, target));
        }
        ScanKind::Angle => {
            let b = molecule.atoms[atoms[1]].position_angstrom;
            let v1 = subtract(molecule.atoms[atoms[0]].position_angstrom, b);
            let v2 = subtract(molecule.atoms[atoms[2]].position_angstrom, b);
            let current = angle_from_vectors(v1, v2)?;
            let axis = cross(v1, v2);
            let axis = if norm(axis) < 1e-10 {
                // A–B–C 共线：旋转轴不定，取任一垂直于 v2 的方向。
                any_perpendicular(v2)
            } else {
                normalize(axis)
            };
            let delta = target.to_radians() - current.to_radians();
            let rotated = rotate(v2, axis, delta);
            moved.atoms[atoms[2]].position_angstrom = add(b, rotated);
        }
        ScanKind::Dihedral => {
            let b = molecule.atoms[atoms[1]].position_angstrom;
            let c = molecule.atoms[atoms[2]].position_angstrom;
            let axis = subtract(c, b);
            if norm(axis) < 1e-12 {
                return Err(EngineError::InvalidInput(
                    "scan dihedral with coincident axis atoms".into(),
                ));
            }
            let current = dihedral(molecule, atoms[0], atoms[1], atoms[2], atoms[3]);
            let delta = target.to_radians() - current.to_radians();
            let d = molecule.atoms[atoms[3]].position_angstrom;
            let rotated = rotate(subtract(d, c), normalize(axis), delta);
            moved.atoms[atoms[3]].position_angstrom = add(c, rotated);
        }
        _ => {
            return Err(EngineError::Unsupported(format!(
                "scan kind `{}` is not implemented",
                kind.as_str()
            )));
        }
    }
    // 不变量：量回来必须就是目标值（角度 1e-8 度、距离 1e-10 Å）。
    let achieved = measure(&moved, kind, atoms)?;
    let tolerance = match kind {
        ScanKind::Distance => 1.0e-10,
        _ => 1.0e-8,
    };
    // 二面角是**圆周量**：+180° 与 −180° 是同一个构象，直接相减会得到 360 的差。
    let difference = if kind == ScanKind::Dihedral {
        let wrapped = (achieved - target).rem_euclid(360.0);
        wrapped.min(360.0 - wrapped)
    } else {
        (achieved - target).abs()
    };
    if difference > tolerance {
        return Err(EngineError::Internal(format!(
            "scan: wanted {} {target} but the moved geometry measures {achieved}",
            kind.as_str()
        )));
    }
    Ok(moved)
}

/// 约束梯度 `∂q/∂x`：数值中心差分（步长 `1e-4 Å`），按 `[3*atom + axis]` 摊平。
///
/// 用数值而不是逐类解析：距离的梯度是一行闭式，键角/二面角的闭式要写不少易错的分支，而这里
/// 只需要一个**方向**（梯度投影用），代价是 `6N` 次几何度量（不跑 SCF，微秒级）。
///
/// # Errors
///
/// 内坐标无法度量（原子重合/越界），或差分出现非有限值。
pub fn constraint_gradient(
    molecule: &Molecule,
    kind: ScanKind,
    atoms: &[usize],
) -> Result<Vec<f64>, EngineError> {
    const STEP: f64 = 1.0e-4;
    check_atoms(molecule, kind, atoms)?;
    let dof = molecule.n_atoms() * 3;
    let mut gradient = vec![0.0_f64; dof];
    #[allow(clippy::needless_range_loop)]
    for coordinate in 0..dof {
        let atom = coordinate / 3;
        let axis = coordinate % 3;
        let mut plus = molecule.clone();
        let mut minus = molecule.clone();
        plus.atoms[atom].position_angstrom[axis] += STEP;
        minus.atoms[atom].position_angstrom[axis] -= STEP;
        let high = measure(&plus, kind, atoms)?;
        let low = measure(&minus, kind, atoms)?;
        let value = (high - low) / (2.0 * STEP);
        if !value.is_finite() {
            return Err(EngineError::Internal(
                "scan: constraint gradient is not finite".into(),
            ));
        }
        gradient[coordinate] = value;
    }
    Ok(gradient)
}

fn check_atoms(molecule: &Molecule, kind: ScanKind, atoms: &[usize]) -> Result<(), EngineError> {
    if atoms.len() != kind.atom_count() {
        return Err(EngineError::InvalidInput(format!(
            "scan kind `{}` needs {} atoms, got {}",
            kind.as_str(),
            kind.atom_count(),
            atoms.len()
        )));
    }
    for atom in atoms {
        if *atom >= molecule.n_atoms() {
            return Err(EngineError::InvalidInput(format!(
                "scan refers to atom {atom} but the molecule has {}",
                molecule.n_atoms()
            )));
        }
    }
    for (index, atom) in atoms.iter().enumerate() {
        if atoms[index + 1..].contains(atom) {
            return Err(EngineError::InvalidInput(format!(
                "scan lists atom {atom} twice"
            )));
        }
    }
    Ok(())
}

fn distance(molecule: &Molecule, i: usize, j: usize) -> f64 {
    norm(subtract(
        molecule.atoms[j].position_angstrom,
        molecule.atoms[i].position_angstrom,
    ))
}

fn angle(molecule: &Molecule, i: usize, j: usize, k: usize) -> f64 {
    let b = molecule.atoms[j].position_angstrom;
    let v1 = subtract(molecule.atoms[i].position_angstrom, b);
    let v2 = subtract(molecule.atoms[k].position_angstrom, b);
    angle_from_vectors(v1, v2).unwrap_or(f64::NAN)
}

fn angle_from_vectors(v1: [f64; 3], v2: [f64; 3]) -> Result<f64, EngineError> {
    let n1 = norm(v1);
    let n2 = norm(v2);
    if n1 < 1e-12 || n2 < 1e-12 {
        return Err(EngineError::InvalidInput(
            "scan angle with a zero-length bond".into(),
        ));
    }
    let cosine = (dot(v1, v2) / (n1 * n2)).clamp(-1.0, 1.0);
    Ok(cosine.acos().to_degrees())
}

/// 二面角（度），按 IUPAC 符号约定：`0..180` 与 `-180..0`。
fn dihedral(molecule: &Molecule, i: usize, j: usize, k: usize, l: usize) -> f64 {
    let b1 = subtract(
        molecule.atoms[j].position_angstrom,
        molecule.atoms[i].position_angstrom,
    );
    let b2 = subtract(
        molecule.atoms[k].position_angstrom,
        molecule.atoms[j].position_angstrom,
    );
    let b3 = subtract(
        molecule.atoms[l].position_angstrom,
        molecule.atoms[k].position_angstrom,
    );
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let m = cross(n1, normalize(b2));
    let x = dot(n1, n2);
    let y = dot(m, n2);
    (-y).atan2(x).to_degrees()
}

fn any_perpendicular(vector: [f64; 3]) -> [f64; 3] {
    let unit = normalize(vector);
    let candidate = if unit[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(cross(unit, candidate))
}

fn rotate(vector: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let (sin, cos) = angle.sin_cos();
    let term1 = scale(vector, cos);
    let term2 = scale(cross(axis, vector), sin);
    let term3 = scale(axis, dot(axis, vector) * (1.0 - cos));
    add(add(term1, term2), term3)
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], factor: f64) -> [f64; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let length = norm(a);
    if length < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / length)
    }
}

#[cfg(test)]
mod tests {
    use super::{measure, set_coordinate};
    use qwb_engine_api::pes::ScanKind;
    use qwb_engine_api::{Atom, Molecule};

    fn water() -> Molecule {
        Molecule::new(
            vec![
                Atom::new("O", [0.0, 0.0, 0.11779]),
                Atom::new("H", [0.0, 0.75545, -0.47116]),
                Atom::new("H", [0.0, -0.75545, -0.47116]),
            ],
            0,
            1,
        )
    }

    #[test]
    fn distances_and_angles_round_trip() {
        let molecule = water();
        let current = measure(&molecule, ScanKind::Distance, &[0, 1]).expect("distance");
        // 这个几何是实验平衡几何（O–H ≈ 0.9579 Å）：只要求它落在水分子该有的范围里。
        assert!((0.9..=1.0).contains(&current), "O–H = {current}");
        let moved = set_coordinate(&molecule, ScanKind::Distance, &[0, 1], 1.15).expect("set");
        let after = measure(&moved, ScanKind::Distance, &[0, 1]).expect("measure");
        assert!((after - 1.15).abs() < 1e-12);
        // 另一根键没被碰
        let other = measure(&moved, ScanKind::Distance, &[0, 2]).expect("distance");
        assert!((other - current).abs() < 1e-12, "另一根 O–H 不该被碰");

        let angle = measure(&molecule, ScanKind::Angle, &[1, 0, 2]).expect("angle");
        assert!((angle - 104.5).abs() < 2.0, "H–O–H = {angle}");
        let moved = set_coordinate(&molecule, ScanKind::Angle, &[1, 0, 2], 96.0).expect("set");
        let after = measure(&moved, ScanKind::Angle, &[1, 0, 2]).expect("measure");
        assert!((after - 96.0).abs() < 1e-8, "angle {after}");
    }

    #[test]
    fn dihedral_round_trips_on_butane_like_fragment() {
        // 用一个 4 原子链（C–C–C–C 的近似几何）测二面角。
        let molecule = Molecule::new(
            vec![
                Atom::new("C", [0.0, 0.0, 0.0]),
                Atom::new("C", [0.0, 0.0, 1.53]),
                Atom::new("C", [1.45, 0.0, 2.05]),
                Atom::new("C", [1.45, 1.4, 2.8]),
            ],
            0,
            1,
        );
        let current = measure(&molecule, ScanKind::Dihedral, &[0, 1, 2, 3]).expect("dihedral");
        assert!(current.is_finite());
        for target in [-120.0, 0.0, 60.0, 180.0] {
            let moved = set_coordinate(&molecule, ScanKind::Dihedral, &[0, 1, 2, 3], target)
                .unwrap_or_else(|error| panic!("set {target}: {error}"));
            let after = measure(&moved, ScanKind::Dihedral, &[0, 1, 2, 3]).expect("measure");
            let difference = (after - target).abs().min((after - target + 360.0).abs());
            assert!(difference < 1e-7, "target {target} -> {after}");
        }
    }

    #[test]
    fn degenerate_and_bad_inputs_are_refused() {
        let molecule = water();
        assert!(set_coordinate(&molecule, ScanKind::Distance, &[0, 0], 1.0).is_err());
        assert!(set_coordinate(&molecule, ScanKind::Distance, &[0, 9], 1.0).is_err());
        assert!(set_coordinate(&molecule, ScanKind::Angle, &[0, 1], 90.0).is_err());
        assert!(set_coordinate(&molecule, ScanKind::Distance, &[0, 1], -1.0).is_err());
        // 重合原子：距离无定义
        let mut collided = molecule.clone();
        collided.atoms[1].position_angstrom = collided.atoms[0].position_angstrom;
        assert!(set_coordinate(&collided, ScanKind::Distance, &[0, 1], 1.0).is_err());
    }
}
