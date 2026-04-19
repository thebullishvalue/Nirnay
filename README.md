# NIRNAY (निर्णय) — Unified Market Analysis Terminal

**Quantitative Signal + Regime Intelligence System**

A precision-grade market analysis tool combining multi-engine signal generation with hidden Markov regime detection. Powered by the Obsidian Quant institutional research terminal design language.

---

## System Overview

NIRNAY is built on three core engines:

### MSF (Market Strength Factor)
Internal price structure decomposition. Isolates institutional flow from noise via momentum ROC and efficiency ratios.

### MMR (Macro-Micro Regime)  
Macro correlation analysis. Tracks bond yields, currencies, and commodities to identify regime shifts and macro regime alignment.

### HMM + GARCH + CUSUM
Hidden Markov Model for regime classification, GARCH for volatility regime detection, and CUSUM for change point detection across market data.

---

## Modes

### ETF Screener
Fixed universe analysis across 100+ curated ETFs (broad market, sector, thematic, commodity, and currency plays).
- **Single Day**: Snapshot signal analysis for current date
- **Time Series**: Signal evolution tracking over configurable date ranges

### Market Screener
F&O and equity index constituent analysis (NIFTY 50, NIFTY 100, US S&P 500, commodities, currencies).
- **Single Day**: Constituent-level signal decomposition
- **Time Series**: Market breadth and regime trends

---

## Analysis Outputs

Per-asset and cross-sectional metrics:
- **Signal Score** (-10 to +10): Unified conviction combining MSF + MMR
- **Zone Classification**: Oversold / Neutral / Overbought
- **Regime State**: Bull / Neutral / Bear (HMM-derived)
- **Volatility Regime**: Low / Normal / High / Extreme (GARCH)
- **Signal Triggers**: Buy/Sell regime-change crossovers
- **Divergence Detection**: Divergence persistence and signal conflicts
- **Breadth Metrics**: % of universe in each zone

---

## Design System

**Obsidian Quant Terminal** — Precision-instrument aesthetic for quantitative finance:
- **Typography**: Syne (display), JetBrains Mono (data)
- **Palette**: Obsidian (#0A0E17), Amber Gold (#D4A853), Emerald (#34D399), Rose (#FB7185), Cyan (#22D3EE)
- **Surfaces**: Frameless glass panels with thin border strokes
- **Charts**: Plotly with custom spike design, institutional-grade aesthetics

---

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

---

## Interface

### Sidebar
- **Mode Selection**: Home / ETF Screener / Market Screener
- **Analysis Type**: Single Day or Time Series
- **Date Selection**: Single date or date range
- **Action Button**: Dynamic "RUN" button reflecting current selections
- **System Info**: Engine, data feed, and version

### Main View
- **Landing Page**: System overview and mode selection prompts
- **ETF Screener**: Universe summary, signal ranking, cross-sectional analysis
- **Market Screener**: Index composition, regime distribution, constituent drill-down
- **Time Series**: Signal momentum, regime evolution, divergence trends

---

## Version

**Current: 7.2.0**

See CHANGELOG.md for detailed update history.

---

## Company

Built by **@thebullishvalue** — quantitative research and portfolio intelligence platform.

---

## Technical Stack

- **Backend**: Python 3.12, Streamlit
- **Data**: yfinance (live market data), macro feeds (10Y yields, USD/INR, commodities)
- **ML/Stats**: scikit-learn, numpy, pandas, statsmodels (HMM, GARCH)
- **Visualization**: Plotly
- **Design**: CSS-in-JS via Streamlit markdown injection

---

## License

Proprietary. All rights reserved.
