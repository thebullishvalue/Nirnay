# ◈ NIRNAY (निर्णय) - Decisive Market Intelligence

**Unified Quantitative Market Intelligence System**  
**A Pragyam Product Family Member**

Version: 1.0.0

---

## Overview

NIRNAY (Sanskrit: "transcendent wisdom") is a unified market intelligence system that synthesizes two powerful analytical frameworks:

1. **Signal Generation** (from UMA - Unified Market Analysis)
   - MSF: Market Strength Factor
   - MMR: Macro-Micro Regime
   - Adaptive weighting based on signal clarity

2. **Regime Intelligence** (from AVASTHA - Adaptive Regime Detection)
   - HMM: Hidden Markov Model for state discovery
   - Kalman Filter: Adaptive signal smoothing
   - GARCH: Volatility regime detection
   - CUSUM: Change point detection
   - Bayesian confidence scoring

**The Key Innovation:** Signals are interpreted in the context of market regime, and ALL thresholds are adaptive (percentile-based), not fixed.

---

## Architecture

```
NIRNAY
│
├── nirnay_core.py      # Unified intelligence engine
│   ├── Signal Generation
│   │   ├── MSFCalculator (Momentum, Microstructure, Trend, Flow)
│   │   └── MMRCalculator (Macro correlation-based)
│   │
│   ├── Regime Intelligence
│   │   ├── AdaptiveHMM (state discovery)
│   │   ├── AdaptiveKalmanFilter (signal smoothing)
│   │   ├── GARCHDetector (volatility regime)
│   │   └── CUSUMDetector (change points)
│   │
│   └── NirnayEngine (unified analysis)
│
├── data_engine.py      # Multi-universe data fetching
│   ├── ETF Universe (30 sectoral ETFs)
│   ├── F&O Stocks (~200+ liquid stocks)
│   ├── Index Constituents (16 NSE indices)
│   └── Macro Data (bonds, forex, commodities)
│
├── charts.py           # Visualization components
│   ├── Price charts with signals
│   ├── Oscillator panels
│   ├── Regime gauges
│   ├── HMM probability charts
│   └── Heatmaps & distributions
│
└── app.py              # Streamlit application
    ├── Dashboard Mode
    ├── Chart Analysis Mode
    ├── Screener Mode
    └── Regime Detection Mode
```

---

## How It Works

### 1. Signal Generation (MSF + MMR)

**MSF (Market Strength Factor)** combines four components:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Momentum | ~33% | Rate of change, normalized via sigmoid |
| Microstructure | ~33% | Volume-weighted direction vs impact |
| Trend | ~33% | Multi-timeframe trend composite |
| Flow | ~33% | Accumulation/Distribution + Regime counting |

```
MSF = sigmoid(momentum + structure + flow)
```

**MMR (Macro-Micro Regime)** measures deviation from macro-predicted value:

```
y_predicted = Σ(βᵢ × xᵢ) weighted by R²
MMR = sigmoid(zscore(actual - predicted))
```

**Unified Signal** uses adaptive weighting:
```
Unified = (MSF_weight × MSF) + (MMR_weight × MMR) × agreement_multiplier
```

### 2. Regime Intelligence

**Hidden Markov Model** discovers three latent states:
- State 0: Bull
- State 1: Neutral  
- State 2: Bear

Using the Forward Algorithm:
```
P(State | Observations) ∝ P(Observation | State) × P(State | Previous)
```

**Kalman Filter** smooths signals while adapting to noise:
```
estimate = prediction + kalman_gain × (measurement - prediction)
```

**GARCH Volatility** adjusts signal sensitivity:
```
σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}
multiplier = f(current_vol / long_term_vol)
```

**CUSUM Change Points** detect structural breaks:
```
S⁺_t = max(0, S⁺_{t-1} + z - drift)
Change when S⁺_t > threshold
```

### 3. Adaptive Thresholds

**The Problem with Fixed Thresholds:**
```python
# Traditional (WRONG)
if signal > 5: return "OVERBOUGHT"
```

**NIRNAY's Adaptive Approach:**
```python
# Adaptive (CORRECT)
overbought_threshold = percentile(signal_history, 80)
if signal > overbought_threshold: return "OVERBOUGHT"
```

Thresholds automatically adapt to:
- Different market regimes
- Changing volatility
- Asset-specific characteristics

### 4. Regime-Aware Signal Interpretation

The same signal means different things in different regimes:

| Signal | Bull Regime | Bear Regime | Interpretation |
|--------|-------------|-------------|----------------|
| Oversold | Strong buy | Cautious buy | Aligned vs counter-trend |
| Overbought | Cautious sell | Strong sell | Counter-trend vs aligned |

---

## Signal Types

| Signal | Description | Position Factor |
|--------|-------------|-----------------|
| STRONG_BUY | <10th percentile, bullish regime | 100% |
| BUY | 10-25th percentile | 75% |
| WEAK_BUY | 25-40th percentile | 50% |
| NEUTRAL | 40-60th percentile | 0% |
| WEAK_SELL | 60-75th percentile | -50% |
| SELL | 75-90th percentile | -75% |
| STRONG_SELL | >90th percentile, bearish regime | -100% |

---

## Market Regimes

| Regime | Description | Typical Action |
|--------|-------------|----------------|
| STRONG_BULL 🚀 | Exceptional bullish | Aggressive longs |
| BULL 🐂 | Clear uptrend | Maintain longs |
| WEAK_BULL 📈 | Mild bullish | Cautious longs |
| NEUTRAL 📊 | No clear direction | Reduce exposure |
| WEAK_BEAR 📉 | Mild bearish | Cautious |
| BEAR 🐻 | Clear downtrend | Defensive |
| CRISIS 🔥 | Extreme bearish | Maximum defense |
| TRANSITION ⚡ | Regime change in progress | Wait for clarity |

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Usage Modes

### 1. Dashboard Mode
Quick overview of ETF universe with signal distribution.

### 2. Chart Analysis Mode
Deep-dive into a single symbol with:
- Price chart with signals
- Oscillator panels (MSF, MMR, Unified)
- Component radar
- Regime analysis
- Macro driver correlation

### 3. Screener Mode
Scan entire universe for opportunities:
- Heatmap visualization
- Signal distribution
- Ranking charts
- Full data table

### 4. Regime Detection Mode
Market-wide regime analysis:
- Dominant regime identification
- Bull/Bear percentage breakdown
- Confidence distribution
- Regime distribution chart

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| MSF Length | 20 | Lookback for MSF calculations |
| ROC Length | 14 | Rate of change period |
| Regime Sensitivity | 1.0 | Scaling for regime detection |
| Base Weight | 0.6 | MSF vs MMR base allocation |

---

## Output: NirnayResult

```python
@dataclass
class NirnayResult:
    signal: SignalType          # STRONG_BUY to STRONG_SELL
    signal_strength: float      # -1 to +1
    components: SignalComponents  # MSF, MMR, Momentum, Micro, Flow
    regime: RegimeState         # Regime, HMM probs, volatility
    thresholds: AdaptiveThresholds  # Percentile-based boundaries
    action: str                 # Human-readable recommendation
    position_size_factor: float # 0 to 1 based on confidence
    warnings: List[str]         # Risk alerts
    macro_drivers: List[Dict]   # Top correlated macro factors
```

---

## Key Advantages

### vs Traditional Technical Analysis
✅ Probabilistic (confidence levels)  
✅ Regime-aware interpretation  
✅ Adaptive thresholds  
✅ Multi-factor synthesis  

### vs Fixed-Threshold Systems
✅ No "magic numbers"  
✅ Automatically adapts  
✅ Works across market conditions  
✅ Reduces false signals  

### vs Single-Factor Systems
✅ Momentum + Microstructure + Trend + Flow + Macro  
✅ Agreement multiplier boosts aligned signals  
✅ Divergence detection  
✅ Change point awareness  

---

## Mathematical Foundation

### Hidden Markov Model
- **States:** S = {Bull, Neutral, Bear}
- **Transition Matrix:** A[i,j] = P(S_t = j | S_{t-1} = i)
- **Emission:** B[j](o) = N(o; μ_j, σ_j)
- **Forward Algorithm:** α_t(j) = Σᵢ[α_{t-1}(i) × A[i,j]] × B[j](o_t)

### Kalman Filter
- **State:** x_t = x_{t-1} + w_t
- **Observation:** z_t = x_t + v_t
- **Gain:** K = P / (P + R)
- **Update:** x̂ = x̂_prev + K × (z - x̂_prev)

### GARCH(1,1)
- **Variance:** σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}
- **Persistence:** α + β ≈ 0.95

### Bayesian Confidence
- **Posterior:** P(Regime | Data) ∝ P(Data | Regime) × P(Regime)
- **Combines:** HMM certainty + factor agreement + data sufficiency

---

## License

Proprietary - Pragyam Product Family

---

## Etymology

**NIRNAY (निर्णय)** in Sanskrit/Hindi means:
- "Decision"
- "Judgment" 
- "Determination"
- "Verdict"

NIRNAY represents the decisive moment when analysis transforms into action. While signals show possibilities and regimes reveal context, NIRNAY is the **judgment** that synthesizes everything into a clear decision.

The synthesis of **UMA** (Unified Market Analysis) and **AVASTHA** (अवस्था - State/Condition) creates **NIRNAY** - the decisive judgment that emerges from understanding both signals AND context.
