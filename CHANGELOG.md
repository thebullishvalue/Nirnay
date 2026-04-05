# Changelog

All notable changes to the NIRNAY (निर्णय) project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.1] — 2026-04-05

### Changed
- Synchronized version numbers across `app.py`, `nirnay_core.py`, and `README.md` (previously inconsistent at `v1.1.0` vs `1.0.0`)
- Verified `nirnay_core.py` standalone library module is production-ready with full type hints and dataclass exports

### Changed (Production Preparation)
- Full codebase audit performed across all source and documentation files
- Confirmed all 4 project files (`app.py`, `nirnay_core.py`, `README.md`, `requirements.txt`) are active with zero dead or orphaned files
- Verified no dead code paths — all utility functions (`sigmoid`, `zscore_clipped`, `calculate_atr`) are actively referenced by the analysis pipeline
- Confirmed `app.py` operates as a self-contained Streamlit application with inline regime intelligence (HMM, GARCH, CUSUM, Kalman Filter)
- Confirmed `nirnay_core.py` serves as the reusable OOP library with `NirnayEngine`, `MSFCalculator`, `MMRCalculator`, and all regime detectors

### Fixed
- `nirnay_core.py` docstring version updated from `1.0.0` to `1.1.1` (was stale)
- `README.md` version badge updated from `1.1.0` to `1.1.1`

---

## [1.1.0] — Previous Release

### Features
- Full NIRNAY (MSF + MMR + Regime Intelligence) unified analysis engine
- ETF Screener: 30 curated global ETFs with single-day and time-series modes
- Market Screener: F&O stocks, index constituents, commodities, and currency pairs
- Chart Analysis: Deep-dive with candlestick, oscillator, and HMM probability charts
- Regime Intelligence: HMM state detection, GARCH volatility regimes, CUSUM change points, Kalman filtering
