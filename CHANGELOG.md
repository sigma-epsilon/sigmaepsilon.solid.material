# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., best follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-10-31

### Changed

- Relaxed the dependency bounds: no upper caps, and the minimums are the oldest versions the test suite passes with (checked by a lowest-direct CI job). NumPy 2 is supported.
- Python 3.13 is supported, and `requires-python` is `>=3.10` without an upper bound.
- The directly imported packages (`sigmaepsilon.core`, `sigmaepsilon.math`, NumPy, SciPy, Numba, SymPy) are declared as dependencies.
- Requires `sigmaepsilon.mesh>=3.3.0`.

### Fixed

- Replaced `np.Infinity`, which was removed in NumPy 2, with `np.inf`.

## [1.0.0] - 2023-09-27

This is the first release.

### Added

- Linear elastic analysis of Euler-Bernoulli and Timoshenko-Ehrenfest beams
- Linear elastic analysis of Kirchhoff-Love and Uflyand-Mindlin plates
- Support for concentrated and distributed loads
