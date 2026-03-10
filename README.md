# NIRNAY (निर्णय) — Unified Market Analysis

**Quantitative Signal + Regime Intelligence System**
A Pragyam Product Family Member | Hemrek Capital

Version 1.1.0

---

## Overview

NIRNAY combines signal generation (MSF + MMR) with regime intelligence (HMM, GARCH, CUSUM, Kalman) into a single market analysis platform. Built on Streamlit, it provides screeners, time-series tracking, and deep-dive chart analysis for ETFs, equities, commodities, and currencies.

---

## Modules

### ETF Screener

- Full MSF + MMR + Regime analysis across 30 curated global ETFs
- Single-day and time-series modes
- Macro correlation analysis and HMM regime detection

### Market Screener

- MSF-based signal analysis for Indian F&O stocks, index constituents, commodities, and currencies
- Single-day and time-series tracking
- Volatility regime classification (GARCH)

### Chart Analysis

- Individual security deep-dive
- Price candlestick and oscillator charts
- HMM state probability visualization
- CUSUM change-point detection
- Macro driver correlation breakdown

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
├── app.py            # Streamlit application (UI + analysis engine)
├── nirnay_core.py    # Standalone analysis core (dataclasses, engines)
├── requirements.txt  # Python dependencies
└── README.md         # Documentation
```

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

Proprietary — Pragyam Product Family, Hemrek Capital
