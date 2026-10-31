# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., best follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-10-31

### Changed

- **Results of membranes, plates and shells change.** The material stiffness matrix of the layers is obtained by condensing out the strain through the thickness of the 3d stiffness matrix, according to the zero normal stress through the thickness. Previously, the in-plane components were taken from the 3d stiffness matrix as they are, which constrained the strain through the thickness to zero and overestimated the stiffness, by 22% for an isotropic material with a Poisson's ratio of 0.3 (and without bounds as it approaches 0.5). The stiffness of an isotropic plate is now the classical `E * t**3 / (12 * (1 - nu**2))`, and that of an orthotropic layer is the reduced stiffness of classical laminate theory.
- The shear stress distribution and the shear correction factors of shells and plates use the transformed, reduced stiffness of the layers, the same as the ABDS matrix. Previously, they used the 3d stiffness of the layers in the material frame, and the in-plane shear modulus in place of the transverse one in the y direction, so for layers with different orientations the shear stresses didn't add up to the shear forces, and the shear correction factors could even exceed 1.

## [1.1.0] - 2026-10-31

### Changed

- Relaxed the dependency bounds: no upper caps, and the minimums are the oldest versions the test suite passes with (checked by a lowest-direct CI job). NumPy 2 is supported.
- Python 3.13 is supported, and `requires-python` is `>=3.10` without an upper bound.
- The directly imported packages (`sigmaepsilon.core`, `sigmaepsilon.math`, NumPy, SciPy, Numba, SymPy) are declared as dependencies.
- Requires `sigmaepsilon.mesh>=3.3.0`.

### Fixed

- Replaced `np.Infinity`, which was removed in NumPy 2, with `np.inf`.
- `Tensor2x3`, and with it `SmallStrainTensor` and `CauchyStressTensor`, rejected 3x3 matrices and 6-vectors, as the shape checks were those of a 4th-order tensor.
- `ElasticityTensor` can be created from a dictionary of engineering constants.
- Integer values for `z` in the postprocessing of surfaces (e.g. `z=[-1, 0, 1]`) were truncated during the mapping to the thickness, so every point was evaluated at the reference surface.
- Fitting the parameters of `HuberMisesHenckyFailureCriterion` had no effect, as it couldn't be instantiated from parameters. It accepts `params` now.
- The binary genetic algorithm maximized the error when fitting failure criteria, since `sigmaepsilon.math` 2.2.0 changed it to maximize by default.
- `HuberMisesHenckyFailureCriterion` accepts a single `FunctionEvaluator` for every backend.
- Setting the `params` of `HuberMisesHenckyFailureCriterion` to a string raises a `TypeError`.

## [1.0.0] - 2023-09-27

This is the first release.

### Added

- Linear elastic analysis of Euler-Bernoulli and Timoshenko-Ehrenfest beams
- Linear elastic analysis of Kirchhoff-Love and Uflyand-Mindlin plates
- Support for concentrated and distributed loads
