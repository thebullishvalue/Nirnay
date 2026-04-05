# NIRNAY (निर्णय) — Unified Market Analysis

**Quantitative Signal + Regime Intelligence System**
A Pragyam Product Family Member | @thebullishvalue

Version 1.1.1

---

## Overview

NIRNAY combines signal generation (MSF + MMR) with regime intelligence (HMM, GARCH, CUSUM, Kalman) into a single market analysis platform. Built on Streamlit, it provides screeners, time-series tracking, and deep-dive chart analysis for ETFs, equities, commodities, and currencies.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NIRNAY Application                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │    app.py        │         │   nirnay_core.py     │  │
│  │  (Streamlit App) │         │  (Analysis Library)  │  │
│  │                  │         │                      │  │
│  │  • UI & Layout   │         │  • NirnayEngine      │  │
│  │  • Data Fetching │         │  • MSFCalculator     │  │
│  │  • Charting      │         │  • MMRCalculator     │  │
│  │  • Screener Logic│         │  • AdaptiveKalman    │  │
│  │  • Inline Models │         │  • AdaptiveHMM       │  │
│  │                  │         │  • GARCHDetector     │  │
│  │  Regime Models:  │         │  • CUSUMDetector     │  │
│  │  • AdaptiveHMM   │         │  • MathUtils         │  │
│  │  • GARCHDetector │         │  • run_batch_analysis│  │
│  │  • CUSUMDetector │         │                      │  │
│  │  • KalmanFilter  │         │  Fully typed,        │  │
│  │                  │         │  dataclass-based     │  │
│  └──────────────────┘         └──────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Data Layer                          │   │
│  │                                                  │   │
│  │  Yahoo Finance (yfinance)  —  Price & Volume     │   │
│  │  Stooq HTTP API            —  Bond Yields         │   │
│  │  NSE Indices / Wikipedia   —  Index Constituents  │   │
│  │                                                  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Two-Module Design

**`app.py` — Self-Contained Streamlit Application**

The primary entry point. A fully independent Streamlit app with inline implementations of all regime intelligence models (HMM, GARCH, CUSUM, Kalman Filter). Contains the complete UI layer, data fetching pipeline, chart generation, and screener logic. Runs standalone: `streamlit run app.py`.

**`nirnay_core.py` — Reusable Analysis Library**

A production-ready, OOP-style Python library with full type hints, dataclasses, and clean exports. Implements the same signal generation and regime detection algorithms in a class-based architecture designed for use by external tools, tests, or other applications. Exports `NirnayEngine`, `MSFCalculator`, `MMRCalculator`, `AdaptiveKalmanFilter`, `AdaptiveHMM`, `GARCHDetector`, `CUSUMDetector`, `MathUtils`, and `run_batch_analysis`.

Both modules implement the same core algorithms independently — `app.py` uses a functional style optimized for Streamlit's execution model, while `nirnay_core.py` uses an object-oriented design for library reuse.

---

## Analysis Methodology

### Signal Generation

**MSF — Market Structure & Flow**

| Component | Description |
|-----------|-------------|
| Momentum | Rate-of-change dynamics |
| Microstructure | Price efficiency metrics |
| Trend | Directional bias detection |
| Flow | Volume-weighted signal |

**MMR — Macro-Market Regression**

| Driver | Source |
|--------|--------|
| Bond Markets | US 10Y yield, India 10Y yield |
| Currencies | DXY, USD/INR |
| Commodities | Gold, Crude Oil |

### Regime Intelligence

| Model | Purpose |
|-------|---------|
| HMM (Hidden Markov Model) | 3-state regime discovery (Bull / Neutral / Bear) with online learning |
| GARCH(1,1) | Volatility regime classification (Low / Normal / High / Extreme) |
| CUSUM | Cumulative-sum change-point detection for structural breaks |
| Kalman Filter | Adaptive signal smoothing and noise estimation |

---

## Signal Interpretation

| Zone | Range | Interpretation |
|------|-------|----------------|
| Oversold | < −5 | Potential buying opportunity |
| Neutral | −5 to +5 | No clear directional bias |
| Overbought | > +5 | Potential selling opportunity |

### Regime States

| State | Condition |
|-------|-----------|
| BULL | Strong bullish (P > 0.6) |
| WEAK_BULL | Moderate bullish bias |
| NEUTRAL | No clear direction |
| WEAK_BEAR | Moderate bearish bias |
| BEAR | Strong bearish (P > 0.6) |
| TRANSITION | CUSUM change point detected |

---

## Application Modules

### ETF Screener

- Full MSF + MMR + Regime analysis across 30 curated global ETFs
- Single-day and time-series modes
- Macro correlation analysis and HMM regime detection

### Market Screener

- MSF-based signal analysis for Indian F&O stocks, index constituents, commodities, and currencies
- Universe options: India Indexes, US Indexes, Commodities, Currency
- Single-day and time-series tracking
- Volatility regime classification (GARCH)

### Chart Analysis

- Individual security deep-dive
- Price candlestick and oscillator charts
- HMM state probability visualization
- CUSUM change-point detection
- Macro driver correlation breakdown

---

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Lookback Period | 20 | 10–50 | MSF calculation window |
| ROC Length | 14 | 5–30 | Rate-of-change period |
| Regime Sensitivity | 1.5 | 0.5–3.0 | Adaptive weighting power |
| Base MSF Weight | 0.5 | 0.0–1.0 | MSF vs MMR base allocation |

---

## File Structure

```
Nirnay/
├── app.py            # Streamlit application (UI + data fetching + inline models)
├── nirnay_core.py    # Standalone analysis library (OOP engine, typed, exportable)
├── requirements.txt  # Python dependencies
├── CHANGELOG.md      # Version history and release notes
└── README.md         # This file — documentation and architecture
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a full history of changes.

### Latest — v1.1.1 (2026-04-05)
- Synchronized version numbers across all files
- Full codebase audit and production preparation
- Created CHANGELOG.md for version tracking

---

## Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.31
- plotly >= 5.18.0
- requests >= 2.31.0
- lxml >= 4.9.0
- beautifulsoup4 >= 4.12.0

---

## License

Proprietary — Pragyam Product Family, @thebullishvalue
