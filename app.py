"""
NIRNAY (निर्णय) - Unified Market Analysis | A Pragyam Product Family Member
Quantitative Signal + Regime Intelligence System

Combines:
- MSF (Market Strength Factor) - Price structure analysis
- MMR (Macro-Micro Regime) - Macro correlation analysis
- HMM (Hidden Markov Model) - Regime state detection
- GARCH - Volatility regime analysis
- CUSUM - Change point detection
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import requests
import io
import urllib3

# Obsidian Quant Design System
import ui.theme as theme
import ui.components as comps

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NIRNAY | Unified Market Analysis",
    layout="wide",
    page_icon="📈",
    page_icon=None,
    initial_sidebar_state="collapsed"
)

from core.config import (
    VERSION, PRODUCT_NAME, COMPANY,
    COLOR_GREEN, COLOR_RED, COLOR_GOLD, COLOR_CYAN, COLOR_AMBER, COLOR_PURPLE, COLOR_MUTED,
    UI_CHART_HEIGHT_SMALL, UI_CHART_HEIGHT_MEDIUM, UI_CHART_HEIGHT_LARGE, UI_CHART_HEIGHT_XLARGE, UI_CHART_HEIGHT_STACKED,
    UI_BREADTH_HIGH, UI_CONVICTION_STRONG, UI_CONVICTION_MODERATE,
)
from ui.theme import chart_layout, style_axes
from universe import (
    ETF_UNIVERSE, INDIA_INDEX_LIST, US_INDEX_LIST, MARKET_UNIVERSE_OPTIONS,
    COMMODITY_TICKERS, CURRENCY_TICKERS, get_fno_stock_list, get_index_stock_list,
    get_commodity_list, get_currency_list
)


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM INJECTION
# ══════════════════════════════════════════════════════════════════════════════

theme.inject_css()

# Initialize session state for analysis tracking
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════

SCREENER_SYMBOLS = [
    "SENSEXIETF.NS", "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS", "CPSEETF.NS",
    "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS", "MOREALTY.NS", "PSUBNKIETF.NS",
    "MASPTOP50.NS", "FMCGIETF.NS", "BANKIETF.NS", "ITIETF.NS", "EVINDIA.NS",
    "MNC.NS", "FINIETF.NS", "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS",
    "ECAPINSURE.NS", "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS"
]

SYMBOL_NAMES = {
    "SENSEXIETF.NS": "SENSEX", "NIFTYIETF.NS": "NIFTY 50", "MON100.NS": "NIFTY 100",
    "MAKEINDIA.NS": "Make India", "SILVERIETF.NS": "Silver", "HEALTHIETF.NS": "Healthcare",
    "CONSUMIETF.NS": "Consumer", "GOLDIETF.NS": "Gold", "INFRAIETF.NS": "Infra",
    "CPSEETF.NS": "CPSE", "TNIDETF.NS": "TN Index", "COMMOIETF.NS": "Commodities",
    "MODEFENCE.NS": "Defence", "MOREALTY.NS": "Realty", "PSUBNKIETF.NS": "PSU Bank",
    "MASPTOP50.NS": "Top 50", "FMCGIETF.NS": "FMCG", "BANKIETF.NS": "Banking",
    "ITIETF.NS": "IT", "EVINDIA.NS": "EV India", "MNC.NS": "MNC",
    "FINIETF.NS": "Financial", "AUTOIETF.NS": "Auto", "PVTBANIETF.NS": "Pvt Bank",
    "MONIFTY500.NS": "NIFTY 500", "ECAPINSURE.NS": "Insurance", "MIDCAPIETF.NS": "Midcap",
    "MOSMALL250.NS": "Smallcap", "OILIETF.NS": "Oil & Gas", "METALIETF.NS": "Metal"
}

MACRO_SYMBOLS_STOOQ = {
    "India 10Y": "10YINY.B", "India 02Y": "2YINY.B",
    "US 30Y": "30YUSY.B", "US 10Y": "10YUSY.B", "US 05Y": "5YUSY.B", "US 02Y": "2YUSY.B",
    "UK 30Y": "30YUKY.B", "UK 10Y": "10YUKY.B", "UK 05Y": "5YUKY.B", "UK 02Y": "2YUKY.B",
    "EU (DE) 30Y": "30YDEY.B", "EU (DE) 10Y": "10YDEY.B", "EU (DE) 05Y": "5YDEY.B", "EU (DE) 02Y": "2YDEY.B",
    "China 10Y": "10YCNY.B", "China 02Y": "2YCNY.B",
    "Japan 30Y": "30YJPY.B", "Japan 10Y": "10YJPY.B", "Japan 02Y": "2YJPY.B",
    "Singapore 10Y": "10YSGY.B",
}

MACRO_SYMBOLS_YF = {
    "Dollar Index": "DX-Y.NYB", "Crude Oil": "CL=F", "Brent Crude": "BZ=F",
    "USD/INR": "INR=X", "GBP/INR": "GBPINR=X", "EUR/INR": "EURINR=X",
    "SGD/INR": "SGDINR=X", "JPY/INR": "JPYINR=X", "Gold": "GC=F", "Silver": "SI=F"
}

MACRO_SYMBOLS = {**MACRO_SYMBOLS_STOOQ, **MACRO_SYMBOLS_YF}

# ══════════════════════════════════════════════════════════════════════════════
# SPREAD SCREENER CONSTANTS (imported from universe module)
# ══════════════════════════════════════════════════════════════════════════════

# Combined list for backward compatibility
INDEX_LIST = INDIA_INDEX_LIST + US_INDEX_LIST

def get_display_name(symbol):
    """Map ticker symbol to human-readable display name."""
    if symbol in COMMODITY_TICKERS:
        return COMMODITY_TICKERS[symbol]
    if symbol in CURRENCY_TICKERS:
        return CURRENCY_TICKERS[symbol]
    return SYMBOL_NAMES.get(symbol, symbol.replace(".NS", ""))

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE SELECTION FUNCTIONS (for Spread Screener)
# ══════════════════════════════════════════════════════════════════════════════

def get_fno_stock_list():
    """Return list of major F&O stocks (most liquid NSE derivatives universe)."""
    fno_stocks = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "ICICIBANK.NS",
        "KOTAK.NS", "BAJAJFINSV.NS", "ITC.NS", "LT.NS", "MARUTI.NS",
        "ASIANPAINT.NS", "SUNPHARMA.NS", "WIPRO.NS", "ADANIPORT.NS", "ADANIGREEN.NS",
        "POWERGRID.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "SBIN.NS", "AXISBANK.NS",
        "ULTRACEMCO.NS", "HCLTECH.NS", "BHARATIARTL.NS", "TECHM.NS", "BAJAJ-AUTO.NS",
        "HEROMOTOCO.NS", "M&M.NS", "EICHERMOT.NS", "HINDALCO.NS", "NTPC.NS"
    ]
    return fno_stocks, f"✓ {len(fno_stocks)} F&O stocks available"


INDIA_INDEX_WIKI_MAP = {
    "NIFTY 50": "https://en.wikipedia.org/wiki/NIFTY_50",
    "NIFTY NEXT 50": "https://en.wikipedia.org/wiki/NIFTY_Next_50",
    "NIFTY 500": "https://en.wikipedia.org/wiki/NIFTY_500",
    # NIFTY 100 = NIFTY 50 + NIFTY NEXT 50 (constructed from both pages)
}


def _fetch_india_index_from_wikipedia(index):
    """Fallback: Fetch Indian index constituents from Wikipedia when niftyindices.com is unreachable"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    def _parse_wiki_table(url, min_count=10):
        """Parse a Wikipedia page and extract NSE symbols from the constituent table"""
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        for tbl in tables:
            if 'Symbol' in tbl.columns:
                symbols = tbl['Symbol'].dropna().astype(str).str.strip().tolist()
                symbols = [s for s in symbols if s and len(s) <= 20 and s != 'nan']
                if len(symbols) >= min_count:
                    return symbols
        return None

    try:
        # NIFTY 100 is constructed from NIFTY 50 + NIFTY NEXT 50
        if index == "NIFTY 100":
            n50 = _parse_wiki_table(INDIA_INDEX_WIKI_MAP["NIFTY 50"], min_count=40)
            nn50 = _parse_wiki_table(INDIA_INDEX_WIKI_MAP["NIFTY NEXT 50"], min_count=40)
            if n50 and nn50:
                combined = list(dict.fromkeys(n50 + nn50))  # deduplicate preserving order
                symbols_ns = [s + ".NS" for s in combined]
                return symbols_ns, f"⚠ niftyindices.com unavailable → Loaded {len(symbols_ns)} NIFTY 100 constituents from Wikipedia (NIFTY 50 + Next 50)"
            return None, "Wikipedia fallback failed for NIFTY 100"

        # NIFTY 200 — use NIFTY 500 Wikipedia page (first 200 by order)
        if index == "NIFTY 200":
            symbols = _parse_wiki_table(INDIA_INDEX_WIKI_MAP["NIFTY 500"], min_count=100)
            if symbols:
                symbols_200 = symbols[:200]
                symbols_ns = [s + ".NS" for s in symbols_200]
                return symbols_ns, f"⚠ niftyindices.com unavailable → Loaded {len(symbols_ns)} NIFTY 200 constituents from Wikipedia (top 200 of NIFTY 500)"
            return None, "Wikipedia fallback failed for NIFTY 200"

        # Direct Wikipedia lookup for NIFTY 50, NIFTY NEXT 50, NIFTY 500
        wiki_url = INDIA_INDEX_WIKI_MAP.get(index)
        if wiki_url:
            min_expected = {"NIFTY 50": 40, "NIFTY NEXT 50": 40, "NIFTY 500": 400}.get(index, 10)
            symbols = _parse_wiki_table(wiki_url, min_count=min_expected)
            if symbols:
                symbols_ns = [s + ".NS" for s in symbols]
                return symbols_ns, f"⚠ niftyindices.com unavailable → Loaded {len(symbols_ns)} {index} constituents from Wikipedia"
            return None, f"Wikipedia fallback: could not parse {index} table"

        # No Wikipedia fallback available for this index (sectoral/midcap)
        return None, None  # Signal: no fallback available

    except Exception as e:
        return None, f"Wikipedia fallback error: {e}"


@st.cache_data(ttl=3600, show_spinner=False)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_data(stock_list, end_date=None, days_back=100, include_live=True):
    """Batch download for spread screener with optional live data for current day"""
    if end_date is None:
        end_date = datetime.date.today()
    
    # Add buffer for end date to ensure we get the requested date
    download_end = end_date + datetime.timedelta(days=5)
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    
    try:
        all_data = yf.download(
            stock_list,
            start=start_date,
            end=download_end,
            progress=False,
            auto_adjust=True,
            group_by='ticker'
        )
        
        if all_data.empty:
            return None, "No data returned"
            
        if isinstance(all_data, pd.DataFrame) and isinstance(all_data.columns, pd.MultiIndex):
            data_dict = {}
            for ticker in stock_list:
                try:
                    ticker_df = all_data.xs(ticker, level=0, axis=1)
                    if not ticker_df.empty and not ticker_df['Close'].isnull().all():
                        data_dict[ticker] = ticker_df.copy()
                except KeyError:
                    pass

        elif isinstance(all_data, dict):
            data_dict = {t:df.copy() for t,df in all_data.items() if not df.empty and not df['Close'].isnull().all()}

        else:
             return None, "Unexpected data structure"
        
        # Fetch live data for today if requested and end_date is today
        if include_live and end_date == datetime.date.today() and data_dict:
            today_ts = pd.Timestamp(datetime.date.today())
            
            # Check if today's data is missing from at least one ticker
            sample_df = list(data_dict.values())[0]
            sample_df.index = pd.to_datetime(sample_df.index)
            if sample_df.index.tz is not None:
                sample_df.index = sample_df.index.tz_localize(None)
            
            has_today = any(idx.date() == datetime.date.today() for idx in sample_df.index)
            
            if not has_today:
                # Fetch live data for all tickers
                try:
                    live_data = yf.download(
                        list(data_dict.keys()),
                        period="1d",
                        progress=False,
                        auto_adjust=True,
                        group_by='ticker'
                    )
                    
                    if not live_data.empty:
                        if isinstance(live_data, pd.DataFrame) and isinstance(live_data.columns, pd.MultiIndex):
                            for ticker in data_dict.keys():
                                try:
                                    live_ticker = live_data.xs(ticker, level=0, axis=1)
                                    if not live_ticker.empty and not live_ticker['Close'].isnull().all():
                                        # Append live data to historical
                                        hist_df = data_dict[ticker]
                                        hist_df.index = pd.to_datetime(hist_df.index)
                                        if hist_df.index.tz is not None:
                                            hist_df.index = hist_df.index.tz_localize(None)
                                        
                                        live_ticker.index = pd.to_datetime(live_ticker.index)
                                        if live_ticker.index.tz is not None:
                                            live_ticker.index = live_ticker.index.tz_localize(None)
                                        
                                        # Only append if not already present
                                        new_dates = live_ticker.index.difference(hist_df.index)
                                        if len(new_dates) > 0:
                                            data_dict[ticker] = pd.concat([hist_df, live_ticker.loc[new_dates]]).sort_index()
                                except KeyError:
                                    pass
                        
                        return data_dict, f"✓ Downloaded {len(data_dict)} tickers (with live data)"
                except Exception:
                    pass  # Fall through to return historical data only
            
        return data_dict, f"✓ Downloaded {len(data_dict)} tickers"

    except Exception as e:
        return None, f"Download error: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x, scale=1.0):
    """Sigmoid activation for signal normalization."""
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0

def zscore_clipped(series, window, clip=3.0):
    """Calculate clipped z-score to identify extreme values."""
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)

def calculate_atr(df, length=14):
    """Calculate Average True Range for volatility measurement."""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


# ══════════════════════════════════════════════════════════════════════════════
# REGIME INTELLIGENCE (from AVASTHA)
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveHMM:
    """Hidden Markov Model for regime state discovery"""
    
    def __init__(self):
        self.n_states = 3
        self.transition_matrix = np.array([
            [0.85, 0.10, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85]
        ])
        self.emission_means = np.array([0.6, 0.0, -0.6])
        self.emission_stds = np.array([0.3, 0.25, 0.3])
        self.state_probabilities = np.array([0.33, 0.34, 0.33])
        self.observation_history = []
        self.state_history = []
    
    def _gaussian_pdf(self, x, mean, std):
        """Compute Gaussian PDF for emission probability."""
        if std < 1e-8:
            return 1.0 if abs(x - mean) < 1e-8 else 0.0
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    def update(self, observation):
        """Forward pass: ingest observation, update state probabilities, adapt parameters. Returns regime dict."""
        self.observation_history.append(observation)
        
        # Forward step
        predicted = self.transition_matrix.T @ self.state_probabilities
        emissions = np.array([self._gaussian_pdf(observation, self.emission_means[s], self.emission_stds[s]) for s in range(3)])
        updated = emissions * predicted
        total = updated.sum()
        if total > 1e-10:
            updated /= total
        else:
            updated = np.array([0.33, 0.34, 0.33])
        
        self.state_probabilities = updated
        most_likely = np.argmax(updated)
        self.state_history.append(most_likely)
        
        # Adapt parameters online
        if len(self.observation_history) >= 10:
            recent_obs = np.array(self.observation_history[-50:])
            recent_states = self.state_history[-len(recent_obs):]
            for state in range(3):
                mask = np.array(recent_states) == state
                if mask.sum() >= 2:
                    state_obs = recent_obs[mask]
                    self.emission_means[state] = 0.9 * self.emission_means[state] + 0.1 * np.mean(state_obs)
                    self.emission_stds[state] = 0.9 * self.emission_stds[state] + 0.1 * max(np.std(state_obs), 0.1)
        
        return {"BULL": updated[0], "NEUTRAL": updated[1], "BEAR": updated[2]}
    
    def reset(self):
        """Reset state history and probabilities to uniform."""
        self.state_probabilities = np.array([0.33, 0.34, 0.33])
        self.observation_history = []
        self.state_history = []


class GARCHDetector:
    """GARCH-inspired volatility regime detection"""
    
    def __init__(self):
        self.current_variance = 0.04
        self.omega = 0.0001
        self.alpha = 0.1
        self.beta = 0.85
        self.long_term_mean = 0.04
        self.shock_history = []
    
    def update(self, shock):
        """Update variance estimate from price shock. Returns current volatility."""
        self.shock_history.append(shock)
        shock_sq = shock ** 2
        new_var = self.omega + self.alpha * shock_sq + self.beta * self.current_variance
        self.current_variance = np.clip(new_var, 0.001, 1.0)
        
        if len(self.shock_history) >= 10:
            realized = np.var(self.shock_history[-min(50, len(self.shock_history)):])
            self.long_term_mean = 0.95 * self.long_term_mean + 0.05 * realized
        
        return np.sqrt(self.current_variance)
    
    def get_regime(self):
        """Classify volatility regime (LOW/NORMAL/HIGH/EXTREME) with risk multiplier."""
        current_vol = np.sqrt(self.current_variance)
        long_term_vol = np.sqrt(self.long_term_mean)
        ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        if ratio < 0.6:
            return "LOW", 1.3
        elif ratio < 0.9:
            return "NORMAL", 1.0
        elif ratio < 1.4:
            return "HIGH", 0.8
        else:
            return "EXTREME", 0.6
    
    def reset(self):
        """Reset variance and shock history."""
        self.current_variance = 0.04
        self.shock_history = []


class CUSUMDetector:
    """CUSUM change point detection"""
    
    def __init__(self, threshold=4.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self.positive_cusum = 0.0
        self.negative_cusum = 0.0
        self.value_history = []
        self.running_mean = 0.0
        self.running_std = 1.0
    
    def update(self, value):
        """Ingest value and test for change points via CUSUM. Returns True if change detected."""
        self.value_history.append(value)
        
        if len(self.value_history) >= 3:
            recent = self.value_history[-min(20, len(self.value_history)):]
            self.running_mean = np.mean(recent)
            self.running_std = max(np.std(recent), 0.1)
        
        z = (value - self.running_mean) / self.running_std
        
        self.positive_cusum = max(0, self.positive_cusum + z - self.drift)
        self.negative_cusum = max(0, self.negative_cusum - z - self.drift)
        
        change_detected = self.positive_cusum > self.threshold or self.negative_cusum > self.threshold
        
        if change_detected:
            self.positive_cusum = 0
            self.negative_cusum = 0
        
        return change_detected
    
    def reset(self):
        """Reset CUSUM accumulators and value history."""
        self.positive_cusum = 0.0
        self.negative_cusum = 0.0
        self.value_history = []


class AdaptiveKalmanFilter:
    """Kalman filter for signal smoothing"""
    
    def __init__(self, process_var=0.01, measurement_var=0.1):
        self.estimate = 0.0
        self.error_covariance = 1.0
        self.process_variance = process_var
        self.measurement_variance = measurement_var
        self.innovation_history = []
    
    def update(self, measurement):
        """Kalman filter step: ingest measurement, compute gain, update estimate. Returns smoothed estimate."""
        predicted_estimate = self.estimate
        predicted_covariance = self.error_covariance + self.process_variance
        
        innovation = measurement - predicted_estimate
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > 50:
            self.innovation_history.pop(0)
        
        innovation_cov = predicted_covariance + self.measurement_variance
        kalman_gain = predicted_covariance / innovation_cov
        
        self.estimate = predicted_estimate + kalman_gain * innovation
        self.error_covariance = (1 - kalman_gain) * predicted_covariance
        
        if len(self.innovation_history) >= 5:
            innovation_var = np.var(self.innovation_history[-min(20, len(self.innovation_history)):])
            self.measurement_variance = 0.9 * self.measurement_variance + 0.1 * innovation_var
        
        return self.estimate
    
    def reset(self, initial=0.0):
        """Reset filter state and covariance to initial value."""
        self.estimate = initial
        self.error_covariance = 1.0
        self.innovation_history = []


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_stooq_symbol(symbol, start_date, end_date):
    """Fetch single symbol from Stooq via direct HTTP request (Python 3.12+ compatible)"""
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start_date.strftime('%Y%m%d')}&d2={end_date.strftime('%Y%m%d')}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and len(response.text) > 50:
            df = pd.read_csv(io.StringIO(response.text))
            if 'Date' in df.columns and 'Close' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                return df['Close']
    except Exception:
        pass
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_macro_data(days_back=100):
    """Fetch macro indicators (yields, FX, commodities) from Stooq and Yahoo Finance."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    
    # Fetch from Stooq via direct HTTP requests (replaces pandas_datareader)
    stooq_df = pd.DataFrame()
    for name, symbol in MACRO_SYMBOLS_STOOQ.items():
        series = fetch_stooq_symbol(symbol, start_date, end_date)
        if series is not None and len(series) > 0:
            stooq_df[symbol] = series
    
    if not stooq_df.empty:
        stooq_df = stooq_df.sort_index()

    yf_df = pd.DataFrame()
    try:
        yf_tickers = list(MACRO_SYMBOLS_YF.values())
        yf_raw = yf.download(yf_tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if not yf_raw.empty:
            if isinstance(yf_raw.columns, pd.MultiIndex):
                if 'Close' in yf_raw.columns.get_level_values(0):
                    yf_df = yf_raw['Close']
                elif 'Adj Close' in yf_raw.columns.get_level_values(0):
                    yf_df = yf_raw['Adj Close']
            else:
                if 'Close' in yf_raw.columns:
                    yf_df = yf_raw[['Close']]
                else:
                    yf_df = yf_raw
            if yf_df.index.tz is not None:
                yf_df.index = yf_df.index.tz_localize(None)
            yf_df = yf_df.sort_index()
            
            # Fetch live data for today if missing
            has_today = any(idx.date() == datetime.date.today() for idx in yf_df.index)
            if not has_today:
                try:
                    live_yf = yf.download(yf_tickers, period="1d", progress=False, auto_adjust=False)
                    if not live_yf.empty:
                        if isinstance(live_yf.columns, pd.MultiIndex):
                            if 'Close' in live_yf.columns.get_level_values(0):
                                live_yf = live_yf['Close']
                            elif 'Adj Close' in live_yf.columns.get_level_values(0):
                                live_yf = live_yf['Adj Close']
                        if live_yf.index.tz is not None:
                            live_yf.index = live_yf.index.tz_localize(None)
                        new_dates = live_yf.index.difference(yf_df.index)
                        if len(new_dates) > 0:
                            yf_df = pd.concat([yf_df, live_yf.loc[new_dates]]).sort_index()
                except Exception:
                    pass
    except Exception:
        pass

    if not stooq_df.empty and not yf_df.empty:
        combined_macro = pd.concat([stooq_df, yf_df], axis=1).sort_index()
    elif not stooq_df.empty:
        combined_macro = stooq_df
    elif not yf_df.empty:
        combined_macro = yf_df
    else:
        return pd.DataFrame()
    return combined_macro.ffill()


def fetch_ticker_data(target_ticker, macro_df, days_back=100, include_live=True):
    """Fetch price data for a ticker and merge with macro indicators. Returns combined OHLCV + macro series."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    try:
        target_df = yf.download(target_ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if target_df.empty:
            return None
        if isinstance(target_df.columns, pd.MultiIndex):
            target_df.columns = target_df.columns.get_level_values(0)
        target_df = target_df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
        if target_df.index.tz is not None:
            target_df.index = target_df.index.tz_localize(None)
        
        # Fetch live data for today if requested
        if include_live:
            has_today = any(idx.date() == datetime.date.today() for idx in target_df.index)
            if not has_today:
                try:
                    live_df = yf.download(target_ticker, period="1d", progress=False, auto_adjust=False)
                    if not live_df.empty:
                        if isinstance(live_df.columns, pd.MultiIndex):
                            live_df.columns = live_df.columns.get_level_values(0)
                        live_df = live_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        if live_df.index.tz is not None:
                            live_df.index = live_df.index.tz_localize(None)
                        # Append only new dates
                        new_dates = live_df.index.difference(target_df.index)
                        if len(new_dates) > 0:
                            target_df = pd.concat([target_df, live_df.loc[new_dates]]).sort_index()
                except Exception:
                    pass
        
        combined = target_df.join(macro_df, how='left').ffill()
        return combined
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def calculate_msf(df, length=20, roc_len=14, clip=3.0):
    """Calculate Market Strength Factor: composite of momentum, microstructure, and flow analysis. Returns (msf_signal, micro, momentum, flow)."""
    close = df['Close']
    
    roc_raw = close.pct_change(roc_len, fill_method=None)
    roc_z = zscore_clipped(roc_raw, length, clip)
    momentum_norm = sigmoid(roc_z, 1.5)
    
    intrabar_dir = (df['High'] + df['Low']) / 2 - df['Open']
    vol_ma = df['Volume'].rolling(length).mean()
    vol_ratio = (df['Volume'] / vol_ma).fillna(1.0)
    
    vw_direction = (intrabar_dir * vol_ratio).rolling(length).mean()
    price_change_imp = close.diff(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(length).mean()
    
    micro_raw = vw_direction - vw_impact
    micro_z = zscore_clipped(micro_raw, length, clip)
    micro_norm = sigmoid(micro_z, 1.5)
    
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(length).mean()
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, length, clip)
    
    mom_accel_raw = close.diff(5).diff(5)
    mom_accel_z = zscore_clipped(mom_accel_raw, length, clip)
    
    atr = calculate_atr(df, 14)
    vol_adj_mom_raw = close.diff(5) / atr
    vol_adj_mom_z = zscore_clipped(vol_adj_mom_raw, length, clip)
    
    mean_rev_z = zscore_clipped(close - trend_slow, length, clip)
    
    composite_trend_z = (trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z) / np.sqrt(4.0)
    composite_trend_norm = sigmoid(composite_trend_z, 1.5)
    
    typical_price = (df['High'] + df['Low'] + close) / 3
    mf = typical_price * df['Volume']
    mf_pos = np.where(close > close.shift(1), mf, 0)
    mf_neg = np.where(close < close.shift(1), mf, 0)
    
    mf_pos_smooth = pd.Series(mf_pos, index=df.index).rolling(length).mean()
    mf_neg_smooth = pd.Series(mf_neg, index=df.index).rolling(length).mean()
    mf_total = mf_pos_smooth + mf_neg_smooth
    
    accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
    accum_ratio = accum_ratio.fillna(0.5)
    accum_norm = 2.0 * (accum_ratio - 0.5)
    
    pct_change = close.pct_change(fill_method=None)
    threshold = 0.0033
    regime_signals = np.select([pct_change > threshold, pct_change < -threshold], [1, -1], default=0)
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length).mean()
    regime_z = zscore_clipped(regime_raw, length, clip)
    regime_norm = sigmoid(regime_z, 1.5)
    
    osc_momentum = momentum_norm
    osc_structure = (micro_norm + composite_trend_norm) / np.sqrt(2.0)
    osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
    
    msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
    msf_signal = sigmoid(msf_raw * np.sqrt(3.0), 1.0)
    
    return msf_signal, micro_norm, momentum_norm, accum_norm


def calculate_mmr(df, length=20, num_vars=5):
    """Calculate Macro-Micro Regime: rolling regression of price against top macro correlates. Returns (mmr_signal, driver_details, model_quality)."""
    available_macros = [v for v in MACRO_SYMBOLS.values() if v in df.columns]
    target = df['Close']
    
    if len(df) < length + 10 or not available_macros:
        return pd.Series(0, index=df.index), [], pd.Series(0, index=df.index)

    correlations = df[available_macros].corrwith(target).abs().sort_values(ascending=False)
    top_drivers = correlations.head(num_vars).index.tolist()
    
    preds = []
    r2_sum = 0
    r2_sq_sum = 0
    y_mean = target.rolling(length).mean()
    y_std = target.rolling(length).std()
    
    driver_details = []

    for ticker in top_drivers:
        x = df[ticker]
        x_mean = x.rolling(length).mean()
        x_std = x.rolling(length).std()
        roll_corr = x.rolling(length).corr(target)
        slope = roll_corr * (y_std / x_std)
        intercept = y_mean - (slope * x_mean)
        
        pred = (slope * x) + intercept
        r2 = roll_corr ** 2
        
        preds.append(pred * r2)
        r2_sum += r2
        r2_sq_sum += r2 ** 2
        
        name = next((k for k, v in MACRO_SYMBOLS.items() if v == ticker), ticker)
        driver_details.append({"Symbol": ticker, "Name": name, "Correlation": round(df[ticker].corr(target), 4)})

    r2_sum = r2_sum.replace(0, np.nan)
    
    if len(preds) > 0:
        y_predicted = sum(preds) / r2_sum
    else:
        y_predicted = y_mean

    deviation = target - y_predicted
    mmr_z = zscore_clipped(deviation, length, 3.0)
    mmr_signal = sigmoid(mmr_z, 1.5)
    
    model_r2 = r2_sq_sum / r2_sum
    mmr_quality = np.sqrt(model_r2.fillna(0))
    
    return mmr_signal, driver_details, mmr_quality


def run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight):
    """Unified analysis: compute MSF + MMR with adaptive weighting, agreement signals, and divergence markers. Mutates df in-place."""
    df['MSF'], df['Micro'], df['Momentum'], df['Flow'] = calculate_msf(df, length, roc_len)
    df['MMR'], drivers, df['MMR_Quality'] = calculate_mmr(df, length, num_vars=5)
    
    msf_clarity = df['MSF'].abs()
    mmr_clarity = df['MMR'].abs()
    msf_clarity_scaled = msf_clarity.pow(regime_sensitivity)
    mmr_clarity_scaled = (mmr_clarity * df['MMR_Quality']).pow(regime_sensitivity)
    clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
    
    msf_w_adaptive = msf_clarity_scaled / clarity_sum
    mmr_w_adaptive = mmr_clarity_scaled / clarity_sum
    
    msf_w_final = 0.5 * base_weight + 0.5 * msf_w_adaptive
    mmr_w_final = 0.5 * (1.0 - base_weight) + 0.5 * mmr_w_adaptive
    w_sum = msf_w_final + mmr_w_final
    msf_w_norm = msf_w_final / w_sum
    mmr_w_norm = mmr_w_final / w_sum
    
    unified_signal = (msf_w_norm * df['MSF']) + (mmr_w_norm * df['MMR'])
    
    agreement = df['MSF'] * df['MMR']
    agree_strength = agreement.abs()
    multiplier = np.where(agreement > 0, 1.0 + 0.2 * agree_strength, 1.0 - 0.1 * agree_strength)
    
    df['Unified'] = (unified_signal * multiplier).clip(-1.0, 1.0)
    df['Unified_Osc'] = df['Unified'] * 10
    df['MSF_Osc'] = df['MSF'] * 10
    df['MMR_Osc'] = df['MMR'] * 10
    df['MSF_Weight'] = msf_w_norm
    df['MMR_Weight'] = mmr_w_norm
    df['Agreement'] = agreement
    
    strong_agreement = agreement > 0.3
    df['Buy_Signal'] = strong_agreement & (df['Unified_Osc'] < -5)
    df['Sell_Signal'] = strong_agreement & (df['Unified_Osc'] > 5)
    
    osc_rising = df['Unified_Osc'] > df['Unified_Osc'].shift(1)
    price_falling = df['Close'] < df['Close'].shift(1)
    osc_falling = df['Unified_Osc'] < df['Unified_Osc'].shift(1)
    price_rising = df['Close'] > df['Close'].shift(1)

    df['Bullish_Div'] = osc_rising & price_falling & (df['Unified_Osc'] < -5)
    df['Bearish_Div'] = osc_falling & price_rising & (df['Unified_Osc'] > 5)
    
    # Vectorized condition assignment (faster than loop)
    df['Condition'] = np.where(df['Unified_Osc'] < -5, 'Oversold', 
                               np.where(df['Unified_Osc'] > 5, 'Overbought', 'Neutral'))
    
    # === REGIME INTELLIGENCE (from AVASTHA) ===
    hmm = AdaptiveHMM()
    garch = GARCHDetector()
    cusum = CUSUMDetector()
    kalman = AdaptiveKalmanFilter()
    
    regimes = []
    hmm_bulls = []
    hmm_bears = []
    vol_regimes = []
    change_points = []
    confidences = []
    signal_history = []
    
    unified_vals = df['Unified'].values
    
    for i in range(len(df)):
        sig = unified_vals[i] if not np.isnan(unified_vals[i]) else 0
        
        # Kalman filter for smoothing
        filtered = kalman.update(sig)
        
        # GARCH for volatility regime
        shock = sig - signal_history[-1] if signal_history else 0
        garch.update(shock)
        vol_regime, _ = garch.get_regime()
        
        # HMM for market state
        hmm_probs = hmm.update(filtered)
        
        # CUSUM for change point detection
        change = cusum.update(filtered)
        
        # Determine regime
        bull_p = hmm_probs['BULL']
        bear_p = hmm_probs['BEAR']
        
        if change:
            regime = "TRANSITION"
        elif bull_p > 0.6:
            regime = "BULL"
        elif bear_p > 0.6:
            regime = "BEAR"
        elif bull_p > 0.4:
            regime = "WEAK_BULL"
        elif bear_p > 0.4:
            regime = "WEAK_BEAR"
        else:
            regime = "NEUTRAL"
        
        regimes.append(regime)
        hmm_bulls.append(bull_p)
        hmm_bears.append(bear_p)
        vol_regimes.append(vol_regime)
        change_points.append(change)
        confidences.append(max(bull_p, bear_p, hmm_probs['NEUTRAL']))
        signal_history.append(sig)
    
    df['Regime'] = regimes
    df['HMM_Bull'] = hmm_bulls
    df['HMM_Bear'] = hmm_bears
    df['Vol_Regime'] = vol_regimes
    df['Change_Point'] = change_points
    df['Confidence'] = confidences

    return df, drivers


# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_price_chart(df, symbol):
    """Candlestick chart with 20/50 moving average overlays."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color=COLOR_GREEN, decreasing_line_color=COLOR_RED,
        increasing_fillcolor='rgba(52,211,153,0.08)', decreasing_fillcolor='rgba(251,113,133,0.08)', name='Price'
    ))
    ma20 = df['Close'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma20, mode='lines', name='MA20', line=dict(color=COLOR_GOLD, width=1.5)))
    ma50 = df['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma50, mode='lines', name='MA50', line=dict(color=COLOR_CYAN, width=1.5)))
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_XLARGE), xaxis_rangeslider_visible=False)
    style_axes(fig, x_title="Date", y_title="Price")
    return fig


def create_oscillator_chart(df):
    """Unified signal oscillator with MSF/MMR components and buy/sell markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Unified_Osc'].clip(lower=0),
        fill='tozeroy', fillcolor='rgba(251,113,133,0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Unified_Osc'].clip(upper=0),
        fill='tozeroy', fillcolor='rgba(52,211,153,0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Unified_Osc'], mode='lines', name='Unified Signal',
        line=dict(color=COLOR_MUTED, width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MSF_Osc'], mode='lines', name='MSF (Internal)',
        line=dict(color=COLOR_GOLD, width=1.2, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MMR_Osc'], mode='lines', name='MMR (Macro)',
        line=dict(color=COLOR_CYAN, width=1.2, dash='dot')
    ))
    buys = df[df['Buy_Signal']]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys['Unified_Osc'], mode='markers', name='Buy Signal',
            marker=dict(size=6, color=COLOR_GREEN)
        ))
    sells = df[df['Sell_Signal']]
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells['Unified_Osc'], mode='markers', name='Sell Signal',
            marker=dict(size=6, color=COLOR_RED)
        ))
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig, y_title="Oscillator", y_range=[-10, 10])
    return fig


def create_gauge_chart(value):
    """Gauge indicator for signal strength (-10 to +10) with color zones."""
    color = COLOR_GREEN if value < -5 else COLOR_RED if value > 5 else COLOR_MUTED
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(font=dict(size=32, color=color, family='JetBrains Mono'), suffix=""),
        gauge=dict(
            axis=dict(range=[-10, 10], tickwidth=1, tickcolor='rgba(255,255,255,0.08)', tickvals=[-10, -5, 0, 5, 10], tickfont=dict(size=10, color='#64748B')),
            bar=dict(color=color, thickness=0.3), bgcolor='rgba(0,0,0,0)', borderwidth=1, bordercolor='rgba(255,255,255,0.08)',
            steps=[dict(range=[-10, -5], color='rgba(52,211,153,0.08)'), dict(range=[-5, 5], color='rgba(255,255,255,0.03)'), dict(range=[5, 10], color='rgba(251,113,133,0.08)')],
            threshold=dict(line=dict(color='white', width=1), thickness=0.8, value=value)
        )
    ))
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False))
    return fig


def create_heatmap_chart(results_df):
    """Grid heatmap of signal strengths across screener universe."""
    symbols = results_df['DisplayName'].tolist()
    scores = results_df['Signal'].tolist()
    n_cols = 6
    n_rows = int(np.ceil(len(symbols) / n_cols))
    while len(symbols) < n_cols * n_rows:
        symbols.append("")
        scores.append(0)
    symbols_grid = np.array(symbols).reshape(n_rows, n_cols)
    scores_grid = np.array(scores).reshape(n_rows, n_cols)
    colorscale = [[0, COLOR_GREEN], [0.25, 'rgba(52,211,153,0.6)'], [0.5, 'rgba(255,255,255,0.1)'], [0.75, 'rgba(251,113,133,0.6)'], [1, COLOR_RED]]
    normalized_scores = (scores_grid + 10) / 20

    fig = go.Figure(data=go.Heatmap(
        z=normalized_scores,
        text=[[f"{s}<br>{v:.1f}" if s else "" for s, v in zip(row_s, row_v)] for row_s, row_v in zip(symbols_grid, scores_grid)],
        texttemplate="%{text}", textfont=dict(size=11, color='#F1F5F9', family='JetBrains Mono'),
        colorscale=colorscale, showscale=False, hovertemplate="<b>%{text}</b><extra></extra>", xgap=3, ygap=3
    ))
    fig.update_layout(
        **chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=False),
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange='reversed'),
    )
    return fig


def create_distribution_chart(results_df):
    """Histogram of Signal values across the universe."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=results_df['Signal'], nbinsx=25,
        marker=dict(color=COLOR_GOLD, line=dict(color='rgba(255,255,255,0.05)', width=0.5)),
        opacity=0.8
    ))
    fig.add_vline(x=-5, line=dict(color=COLOR_GREEN, width=1.5, dash='dot'))
    fig.add_vline(x=5, line=dict(color=COLOR_RED, width=1.5, dash='dot'))
    fig.add_vline(x=0, line=dict(color='rgba(255,255,255,0.1)', width=1))
    fig.add_vrect(x0=-10, x1=-5, fillcolor='rgba(52,211,153,0.04)', line_width=0)
    fig.add_vrect(x0=5, x1=10, fillcolor='rgba(251,113,133,0.04)', line_width=0)

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL), bargap=0.1)
    style_axes(fig, x_title="Signal Strength", y_title="Count", y_range=[0, None])
    return fig


def create_sector_radar(results_df):
    """Polar radar chart aggregating signals by market sector."""
    sectors = {
        'Index': ['SENSEX', 'NIFTY 50', 'NIFTY 100', 'NIFTY 500', 'Top 50', 'Midcap', 'Smallcap'],
        'Banking': ['Banking', 'Pvt Bank', 'PSU Bank', 'Financial', 'Insurance'],
        'Commodities': ['Gold', 'Silver', 'Oil & Gas', 'Metal', 'Commodities'],
        'Defensive': ['Healthcare', 'Consumer', 'FMCG'],
        'Cyclical': ['Auto', 'Infra', 'Realty', 'IT'],
        'Thematic': ['Defence', 'EV India', 'Make India', 'MNC', 'CPSE']
    }
    sector_scores = {}
    for sector, symbols in sectors.items():
        matching = results_df[results_df['DisplayName'].isin(symbols)]
        sector_scores[sector] = matching['Signal'].mean() if not matching.empty else 0

    categories = list(sector_scores.keys())
    values = list(sector_scores.values())
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', fillcolor='rgba(212,168,83,0.12)',
        line=dict(color=COLOR_GOLD, width=2), marker=dict(size=8, color=COLOR_GOLD)
    ))
    fig.update_layout(
        **chart_layout(height=UI_CHART_HEIGHT_LARGE, show_legend=False),
        polar=dict(
            radialaxis=dict(visible=True, range=[-10, 10], tickvals=[-10, -5, 0, 5, 10], gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.05)', tickfont=dict(size=9, color='#64748B')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.05)', tickfont=dict(size=10, color='#94A3B8')),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig


def create_scatter_matrix(results_df):
    """Scatter plot of MSF vs MMR with quadrant zones (buy/sell/neutral)."""
    fig = go.Figure()
    colors = results_df['Zone'].map({'Oversold': COLOR_GREEN, 'Overbought': COLOR_RED, 'Neutral': COLOR_MUTED})
    fig.add_trace(go.Scatter(
        x=results_df['MSF'], y=results_df['MMR'], mode='markers',
        marker=dict(size=10, color=colors, line=dict(color='rgba(255,255,255,0.05)', width=0.5), opacity=0.85),
        text=results_df['DisplayName'], hovertemplate="<b>%{text}</b><br>MSF: %{x:.2f}<br>MMR: %{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=0, line=dict(color=COLOR_GOLD, width=0.5, dash='dot'))
    fig.add_vline(x=0, line=dict(color=COLOR_GOLD, width=0.5, dash='dot'))

    fig.add_annotation(x=8, y=8, text="SELL", font=dict(size=9, color='rgba(251,113,133,0.4)', family='JetBrains Mono'), showarrow=False)
    fig.add_annotation(x=-8, y=-8, text="BUY", font=dict(size=9, color='rgba(52,211,153,0.4)', family='JetBrains Mono'), showarrow=False)

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
    style_axes(fig, x_title="Internal Momentum (MSF)", y_title="Macro Regression (MMR)", y_range=[-12, 12])
    return fig


def create_ranking_chart(results_df, top_n=10):
    """Horizontal bar chart of top buy (green) and top sell (red) signals."""
    sorted_df = results_df.sort_values('Signal')
    bottom = sorted_df.head(top_n//2)
    top = sorted_df.tail(top_n//2)
    combined = pd.concat([bottom, top])
    colors = [COLOR_GREEN if v < 0 else COLOR_RED for v in combined['Signal']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=combined['DisplayName'], x=combined['Signal'], orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.05)', width=0.5)),
        text=[f"{v:.1f}" for v in combined['Signal']], textposition='outside',
        textfont=dict(size=9, color='#64748B', family='JetBrains Mono')
    ))
    fig.add_vline(x=0, line=dict(color=COLOR_GOLD, width=0.5))
    fig.add_vline(x=-5, line=dict(color=COLOR_GREEN, width=1, dash='dot'))
    fig.add_vline(x=5, line=dict(color=COLOR_RED, width=1, dash='dot'))

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
    style_axes(fig, x_title="Signal Score", y_range=[None, None])
    fig.update_yaxes(showgrid=False)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS & MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    """Render the main masthead header (matches Pragyam design)."""
    comps.render_header("NIRNAY", "Quantitative Signal + Regime Intelligence System")


def render_sidebar():
    """Render navigation sidebar with mode selection, parameters, and run button."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:0.75rem 0 1rem 0;">
                <div style="font-family:var(--display);font-size:1.5rem;font-weight:700;color:var(--amber);letter-spacing:0.06em;">NIRNAY</div>
                <div style="font-family:var(--data);color:var(--ink-tertiary);font-size:0.65rem;margin-top:0.2rem;letter-spacing:0.08em;text-transform:uppercase;">निर्णय | Market Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
        mode = st.radio("Analysis Mode", ["Home", "ETF Screener", "Market Screener"], label_visibility="collapsed")

        # Reset analysis flag when mode changes
        st.session_state.analysis_completed = False
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ETF Screener specific options (fixed ETF universe)
        etf_mode = None
        etf_date = None
        etf_start_date = None
        etf_end_date = None
        
        if "ETF" in mode:
            st.markdown('<div class="sidebar-title">Analysis Type</div>', unsafe_allow_html=True)
            etf_mode = st.radio(
                "Select ETF Mode",
                ["Single Day", "Time Series"],
                label_visibility="collapsed",
                help="Single Day: Analyze one date | Time Series: Track signals over a date range"
            )
            
            if "Single" in etf_mode:
                st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
                etf_date = st.date_input(
                    "ETF Analysis Date",
                    datetime.date.today(),
                    max_value=datetime.date.today(),
                    help="Select the date for signal analysis (defaults to today)"
                )
            else:
                st.markdown('<div class="sidebar-title">Date Range</div>', unsafe_allow_html=True)
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    etf_start_date = st.date_input(
                        "ETF Start Date",
                        datetime.date.today() - datetime.timedelta(days=100),
                        max_value=datetime.date.today(),
                        help="Start of analysis period"
                    )
                with col_e2:
                    etf_end_date = st.date_input(
                        "ETF End Date",
                        datetime.date.today(),
                        max_value=datetime.date.today(),
                        help="End of analysis period"
                    )
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Market Screener specific options (F&O / Index universe)
        spread_universe = None
        spread_index = None
        spread_date = None
        spread_mode = None
        spread_start_date = None
        spread_end_date = None
        
        if "Market" in mode:
            st.markdown('<div class="sidebar-title">Universe Selection</div>', unsafe_allow_html=True)
            spread_universe = st.selectbox(
                "Analysis Universe",
                MARKET_UNIVERSE_OPTIONS,
                help="Choose India/US index constituents, Commodities, or Currency pairs"
            )
            if spread_universe == "India Indexes":
                spread_index = st.selectbox(
                    "Select Index",
                    INDIA_INDEX_LIST,
                    index=INDIA_INDEX_LIST.index("NIFTY 50"),
                    help="F&O Stocks or select a specific NIFTY index for constituent analysis"
                )
            elif spread_universe == "US Indexes":
                spread_index = st.selectbox(
                    "Select Index",
                    US_INDEX_LIST,
                    help="Select the US index for constituent analysis"
                )
            
            st.markdown('<div class="sidebar-title">Analysis Type</div>', unsafe_allow_html=True)
            spread_mode = st.radio(
                "Select Mode",
                ["Single Day", "Time Series"],
                label_visibility="collapsed",
                help="Single Day: Analyze one date | Time Series: Track signals over a date range"
            )
            
            if "Single" in spread_mode:
                st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
                spread_date = st.date_input(
                    "Select Date",
                    datetime.date.today(),
                    max_value=datetime.date.today(),
                    help="Select the date for signal analysis (defaults to today)"
                )
            else:
                st.markdown('<div class="sidebar-title">Date Range</div>', unsafe_allow_html=True)
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    spread_start_date = st.date_input(
                        "Start Date",
                        datetime.date.today() - datetime.timedelta(days=100),
                        max_value=datetime.date.today(),
                        help="Start of analysis period"
                    )
                with col_d2:
                    spread_end_date = st.date_input(
                        "End Date",
                        datetime.date.today(),
                        max_value=datetime.date.today(),
                        help="End of analysis period"
                    )
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Dynamic action button based on mode selection (hidden on Home page)
        run_clicked = False

        if mode != "Home":
            button_text = "SELECT MODE"
            button_disabled = True

            if "ETF" in mode:
                if etf_mode and "Single" in etf_mode:
                    button_text = "RUN ETF SCREENER"
                    button_disabled = False
                elif etf_mode and "Time Series" in etf_mode:
                    button_text = "RUN ETF TIME SERIES"
                    button_disabled = False
            elif "Market" in mode:
                if spread_mode and "Single" in spread_mode:
                    button_text = "RUN MARKET SCREENER"
                    button_disabled = False
                elif spread_mode and "Time Series" in spread_mode:
                    button_text = "RUN MARKET TIME SERIES"
                    button_disabled = False

            run_clicked = st.button(button_text, type="primary", width='stretch', disabled=button_disabled, key="sidebar_run_btn")
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="system-spec">
            <div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">{VERSION}</span></div>
            <div class="spec-row"><span class="spec-label">Engine</span><span class="spec-value">MSF + MMR + HMM</span></div>
            <div class="spec-row"><span class="spec-label">Data</span><span class="spec-value">Live Market Feed</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Default indicator parameters (no longer user-configurable)
        length = 20
        roc_len = 14
        regime_sensitivity = 1.5
        base_weight = 0.5

        return mode, length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_date, spread_mode, spread_start_date, spread_end_date, etf_mode, etf_date, etf_start_date, etf_end_date, run_clicked


def run_home_page():
    """Render landing page with system overview — exact Pragyam design adapted for Nirnay."""
    comps.section_gap()

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        st.markdown("""
        <div class='system-card portfolio'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                SIGNALS
            </h3>
            <p>Internal Market Structure & Flow analysis with momentum ROC dynamics, price microstructure decomposition, and institutional flow patterns.</p>
            <div class='spec'>
                <span>Engines:</span> MSF (Market Structure) + MMR (Macro Regression)<br>
                <span>Indicators:</span> ROC · Z-Score · Efficiency Ratio<br>
                <span>Signals:</span> Flow quality + Structural strength<br>
                <span>Modes:</span> Single Day + Time Series analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='system-card regime'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                REGIME
            </h3>
            <p>Adaptive market regime detection using Hidden Markov Models for state discovery, GARCH for volatility regime, and CUSUM for change points.</p>
            <div class='spec'>
                <span>Detection:</span> HMM state identification<br>
                <span>Volatility:</span> GARCH regime classification<br>
                <span>Change Points:</span> CUSUM anomaly detection<br>
                <span>Sensitivity:</span> Configurable regime threshold
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='system-card strategies'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                SCREENER
            </h3>
            <p>Global multi-instrument screener across ETF universe, equity indices, commodities, and currency markets with unified signal overlay.</p>
            <div class='spec'>
                <span>Coverage:</span> 30 ETFs · 17 Indices · 24 Commodities<br>
                <span>Currencies:</span> 25 major FX pairs<br>
                <span>Analysis:</span> Single symbol or universe-wide<br>
                <span>Output:</span> Signal scores + regime state
            </div>
        </div>
        """, unsafe_allow_html=True)

    comps.section_gap()

    st.markdown("""
    <div class='landing-prompt'>
        <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
            SELECT ANALYSIS MODE
        </h4>
        <p>Choose from the <strong>Sidebar</strong>: <strong>ETF Screener</strong> (curated universe) or
           <strong>Market Screener</strong> (equity indices · commodities · currencies).<br>
           Select <strong>Single Day</strong> for current signals or <strong>Time Series</strong> to track evolution over time.<br>
           <span style="color:var(--ink-secondary); font-size:0.85em; margin-top:0.5rem; display:inline-block;">System will decompose price structure · Detect market regime · Score signal strength</span></p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point: render header, sidebar, and delegate to selected mode."""
    mode, length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_date, spread_mode, spread_start_date, spread_end_date, etf_mode, etf_date, etf_start_date, etf_end_date, run_clicked = render_sidebar()
    
    # Only show main header on Home page
    if "Home" in mode:
        comps.render_header("NIRNAY", "Quantitative Signal + Regime Intelligence System")

    # Set analysis completion flag when run button is clicked
    if run_clicked:
        st.session_state.analysis_completed = True

    if "Home" in mode:
        run_home_page()
    elif "ETF" in mode:
        # ETF Screener (fixed ETF universe)
        if etf_mode and "Time Series" in etf_mode:
            run_etf_timeseries_mode(length, roc_len, regime_sensitivity, base_weight, etf_start_date, etf_end_date, run_clicked or st.session_state.analysis_completed)
        else:
            run_etf_screener_mode(length, roc_len, regime_sensitivity, base_weight, etf_date, run_clicked or st.session_state.analysis_completed)
    elif "Market" in mode:
        # Market Screener (F&O / Index universe)
        if spread_mode and "Time Series" in spread_mode:
            run_market_timeseries_mode(length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_start_date, spread_end_date, run_clicked or st.session_state.analysis_completed)
        else:
            run_market_screener_mode(length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_date, run_clicked or st.session_state.analysis_completed)

    # Dynamic footer with current IST time
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown(
        f'<div class="app-footer">'
        f'<div class="content">'
        f'&copy; {ist_now.year} <strong>{PRODUCT_NAME}</strong> &nbsp;&middot;&nbsp; {COMPANY} &nbsp;&middot;&nbsp; v{VERSION}'
        f'<br>'
        f'<span style="opacity:0.7;">{current_time_ist}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )



def run_etf_screener_mode(length, roc_len, regime_sensitivity, base_weight, analysis_date, run_clicked):
    """ETF Screener: NIRNAY analysis on fixed ETF universe with date selection"""
    
    # Format analysis date
    if analysis_date is None:
        analysis_date = datetime.date.today()
    analysis_date_str = analysis_date.strftime("%d %b %Y")
    is_today = analysis_date == datetime.date.today()

    if not run_clicked:
        # Top spacing
        comps.section_gap()
        comps.section_gap()

        comps.render_section_header(
            "ETF Screener — Fixed Universe",
            f"Full NIRNAY (MSF + MMR) analysis across {len(SCREENER_SYMBOLS)} ETFs · Analysis Date: {analysis_date_str} {'(Today)' if is_today else ''}",
            icon="grid",
            accent="cyan"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Analysis Overview Section
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            comps.render_metric_card("Universe", f"{len(SCREENER_SYMBOLS)}", "Curated ETFs", "info")
        with col2:
            comps.render_metric_card("Signal Engines", "2", "MSF + MMR", "cyan")
        with col3:
            comps.render_metric_card("Output Metrics", "8", "Signal + Regime + Zone", "warning")
        with col4:
            comps.render_metric_card("Analysis Mode", "Single Day", f"Date: {analysis_date_str}", "neutral")

        comps.section_gap()
        comps.section_gap()

        # Analysis Framework Section
        comps.render_section_header(
            "Analysis Framework",
            "Market Structure (MSF) + Macro Regression (MMR) + Regime Intelligence (HMM/GARCH/CUSUM)",
            icon="layers",
            accent="emerald"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #06B6D4; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Market Structure</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    Internal price structure via momentum ROC, efficiency ratio, and microstructure decomposition.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(212, 168, 83, 0.08); border: 1px solid rgba(212, 168, 83, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #D4A853; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Macro Regression</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    Macro correlation tracking bond yields, currencies, and commodity flows for regime shifts.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #A855F7; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Regime Intelligence</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    HMM state evolution, volatility regime distribution, and change point timeline.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Validate analysis date
    if analysis_date > datetime.date.today():
        st.error("⚠️ Analysis date cannot be in the future.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    if run_clicked:
        progress_slot = st.empty()

        # Fetch macro data with buffer for historical analysis
        days_back = 100 + (datetime.date.today() - analysis_date).days
        theme.progress_bar(progress_slot, 5, "Initialization", "Fetching global macro data...")
        macro_df = fetch_macro_data(days_back=days_back)

        results = []
        total = len(SCREENER_SYMBOLS)

        for i, symbol in enumerate(SCREENER_SYMBOLS):
            pct = int(5 + (95 * (i + 1) / total))
            theme.progress_bar(progress_slot, pct, f"Scanning {get_display_name(symbol)}", f"{i+1}/{total} ETFs")
            df = fetch_ticker_data(symbol, macro_df, days_back=days_back)
            
            if df is not None and len(df) > length + 5:
                try:
                    df, _ = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    
                    # Find the row for the analysis date
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Get the closest date on or before analysis_date
                    analysis_datetime = pd.Timestamp(analysis_date)
                    valid_dates = df.index[df.index <= analysis_datetime]
                    
                    if len(valid_dates) == 0:
                        continue
                    
                    target_date = valid_dates[-1]
                    target_idx = df.index.get_loc(target_date)
                    
                    if target_idx < 1:
                        continue
                    
                    last_row = df.iloc[target_idx]
                    prev_row = df.iloc[target_idx - 1]
                    price_change = ((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                    
                    signal_str = "BUY" if last_row['Buy_Signal'] else "SELL" if last_row['Sell_Signal'] else "-"
                    div_str = "BULL" if last_row['Bullish_Div'] else "BEAR" if last_row['Bearish_Div'] else "-"
                    
                    results.append({
                        "Symbol": symbol, "DisplayName": get_display_name(symbol),
                        "Price": round(last_row['Close'], 2),
                        "Change": round(price_change, 2),
                        "Signal": round(last_row['Unified_Osc'], 2),
                        "MSF": round(last_row['MSF_Osc'], 2),
                        "MMR": round(last_row['MMR_Osc'], 2),
                        "Zone": last_row['Condition'],
                        "Trigger": signal_str,
                        "Divergence": div_str,
                        "Agreement": round(last_row['Agreement'], 3),
                        # NEW: Regime Intelligence columns
                        "Regime": last_row['Regime'],
                        "HMM_Bull": round(last_row['HMM_Bull'], 2),
                        "HMM_Bear": round(last_row['HMM_Bear'], 2),
                        "Vol_Regime": last_row['Vol_Regime'],
                        "Confidence": round(last_row['Confidence'], 2),
                        "Change_Point": last_row['Change_Point']
                    })
                except Exception:
                    pass

        progress_slot.empty()

        if results:
            st.toast(f"ETF Scan Complete! Analyzed {len(results)}/{total} ETFs")
            results_df = pd.DataFrame(results)
            
            # Calculate summary stats
            n_oversold = len(results_df[results_df['Zone'] == 'Oversold'])
            n_overbought = len(results_df[results_df['Zone'] == 'Overbought'])
            n_neutral = len(results_df[results_df['Zone'] == 'Neutral'])
            n_buys = len(results_df[results_df['Trigger'] == 'BUY'])
            n_sells = len(results_df[results_df['Trigger'] == 'SELL'])
            avg_signal = results_df['Signal'].mean()
            
            # NEW: Calculate HMM regime distribution
            n_bull = len(results_df[results_df['Regime'].str.contains('BULL', na=False)])
            n_bear = len(results_df[results_df['Regime'].str.contains('BEAR', na=False)])
            n_transition = len(results_df[results_df['Regime'] == 'TRANSITION'])
            dominant_regime = results_df['Regime'].mode().iloc[0] if len(results_df) > 0 else "NEUTRAL"
            regime_color = "success" if "BULL" in dominant_regime else "danger" if "BEAR" in dominant_regime else "warning" if dominant_regime == "TRANSITION" else "neutral"
            
            # Metrics row
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                comps.render_metric_card("Universe", f"{len(results)}", "ETFs Analyzed", "info")
            with c2:
                comps.render_metric_card("Oversold", f"{n_oversold}", "Buy Zone", "success")
            with c3:
                comps.render_metric_card("Overbought", f"{n_overbought}", "Sell Zone", "danger")
            with c4:
                comps.render_metric_card("Buy Signals", f"{n_buys}", "Confirmed", "primary")
            with c5:
                comps.render_metric_card("Sell Signals", f"{n_sells}", "Confirmed", "warning")
            with c6:
                comps.render_metric_card("HMM Regime", dominant_regime, f"Bull: {n_bull} | Bear: {n_bear}", regime_color)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Fresh institutional research design: Executive → Opportunities → Analysis → Context → Data
            tab_summary, tab_opps, tab_signals, tab_structure, tab_data = st.tabs(["Summary", "Opportunities", "Signal Analysis", "Market Structure", "Data Explorer"])

            with tab_summary:
                # Executive snapshot for decision makers
                comps.render_section_header("Executive Summary", "Key performance indicators and regime context", icon="list", accent="cyan")

                # KPI metrics in grid
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
                with metric_col1:
                    comps.render_metric_card("Universe", f"{len(results_df)}", "ETFs", "info")
                with metric_col2:
                    comps.render_metric_card("Buy Signals", f"{n_buys}", "Confirmed", "success")
                with metric_col3:
                    comps.render_metric_card("Sell Signals", f"{n_sells}", "Confirmed", "danger")
                with metric_col4:
                    oversold_pct = f"{n_oversold/len(results_df)*100:.0f}%"
                    comps.render_metric_card("Oversold", oversold_pct, f"{n_oversold} ETFs", "info")
                with metric_col5:
                    overbought_pct = f"{n_overbought/len(results_df)*100:.0f}%"
                    comps.render_metric_card("Overbought", overbought_pct, f"{n_overbought} ETFs", "warning")
                with metric_col6:
                    comps.render_metric_card("Avg Signal", f"{results_df['Signal'].mean():.1f}", "Population mean", "neutral")

                st.markdown("<br>", unsafe_allow_html=True)

                # Regime snapshot
                comps.render_section_header("Market Regime — HMM State & Volatility", "Current market conditions and probability states", icon="layers", accent="emerald")
                regime_col1, regime_col2 = st.columns(2)

                with regime_col1:
                    regime_counts = results_df['Regime'].value_counts()
                    regime_colors = {'BULL': '#10b981', 'WEAK_BULL': '#34d399', 'NEUTRAL': '#888888', 'WEAK_BEAR': '#fbbf24', 'BEAR': '#ef4444', 'TRANSITION': '#a855f7'}
                    fig_regime = go.Figure(go.Pie(
                        labels=regime_counts.index, values=regime_counts.values, hole=0.5,
                        marker=dict(colors=[regime_colors.get(r, '#888888') for r in regime_counts.index], line=dict(color='#1A1A1A', width=2)),
                        textinfo='label+percent', textfont=dict(size=11, color='white')
                    ))
                    fig_regime.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False), title=dict(text='HMM Regime', font=dict(size=12, color='#888888')))
                    st.plotly_chart(fig_regime, width='stretch', key="regime_pie")

                with regime_col2:
                    vol_counts = results_df['Vol_Regime'].value_counts()
                    vol_colors = {'LOW': '#10b981', 'NORMAL': '#888888', 'HIGH': '#f59e0b', 'EXTREME': '#ef4444'}
                    fig_vol = go.Figure(go.Pie(
                        labels=vol_counts.index, values=vol_counts.values, hole=0.5,
                        marker=dict(colors=[vol_colors.get(v, '#888888') for v in vol_counts.index], line=dict(color='#1A1A1A', width=2)),
                        textinfo='label+percent', textfont=dict(size=11, color='white')
                    ))
                    fig_vol.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False), title=dict(text='Volatility Regime', font=dict(size=12, color='#888888')))
                    st.plotly_chart(fig_vol, width='stretch', key="vol_pie")

                st.markdown("<br>", unsafe_allow_html=True)

                # Signal distribution
                comps.render_section_header("Signal Distribution", "Cross-sectional view of signal spread", icon="trending-up", accent="amber")
                st.plotly_chart(create_distribution_chart(results_df), width='stretch', key="distribution")

            with tab_opps:
                # Ranked opportunities for traders
                comps.render_section_header("Opportunity Ranking", "All assets ranked by signal strength and conviction", icon="target", accent="amber")

                opp_tab1, opp_tab2 = st.tabs(["Buy Setup", "Sell Setup"])

                with opp_tab1:
                    # Buy opportunities
                    buy_data = results_df[(results_df['Trigger'] == 'BUY') | (results_df['Zone'] == 'Oversold')].copy()
                    buy_data = buy_data.sort_values('Signal').head(20)

                    if not buy_data.empty:
                        buy_display = buy_data[['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'MMR', 'Zone']].copy()
                        buy_display.columns = ['ETF', 'Price', 'Chg %', 'Signal', 'MSF', 'MMR', 'Zone']
                        st.dataframe(buy_display, width="stretch", hide_index=True)
                        st.caption(f"Showing {len(buy_data)} buy setup candidates. Sorted by signal strength (lowest to highest).")
                    else:
                        st.info("No buy opportunities in current scan.")

                with opp_tab2:
                    # Sell opportunities
                    sell_data = results_df[(results_df['Trigger'] == 'SELL') | (results_df['Zone'] == 'Overbought')].copy()
                    sell_data = sell_data.sort_values('Signal', ascending=False).head(20)

                    if not sell_data.empty:
                        sell_display = sell_data[['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'MMR', 'Zone']].copy()
                        sell_display.columns = ['ETF', 'Price', 'Chg %', 'Signal', 'MSF', 'MMR', 'Zone']
                        st.dataframe(sell_display, width="stretch", hide_index=True)
                        st.caption(f"Showing {len(sell_data)} sell setup candidates. Sorted by signal strength (highest to lowest).")
                    else:
                        st.info("No sell opportunities in current scan.")

            with tab_signals:
                # Deep signal analysis
                comps.render_section_header("Signal Decomposition", "MSF vs MMR component analysis and divergence alerts", icon="layers", accent="violet")

                signal_tab1, signal_tab2, signal_tab3 = st.tabs(["Components", "Divergences", "Extremes"])

                with signal_tab1:
                    # MSF vs MMR breakdown
                    comp_col1, comp_col2 = st.columns(2)

                    with comp_col1:
                        st.markdown("**MSF-Dominant (Structure)**")
                        msf_heavy = results_df[results_df['MSF'].abs() > results_df['MMR'].abs()].sort_values('MSF').head(12)
                        if not msf_heavy.empty:
                            msf_display = msf_heavy[['DisplayName', 'Price', 'Signal', 'MSF', 'MMR']].copy()
                            msf_display.columns = ['ETF', 'Price', 'Signal', 'MSF', 'MMR']
                            st.dataframe(msf_display, width="stretch", hide_index=True, height=300)
                        else:
                            st.caption("No MSF-dominant signals found.")

                    with comp_col2:
                        st.markdown("**MMR-Dominant (Macro)**")
                        mmr_heavy = results_df[results_df['MMR'].abs() > results_df['MSF'].abs()].sort_values('MMR').head(12)
                        if not mmr_heavy.empty:
                            mmr_display = mmr_heavy[['DisplayName', 'Price', 'Signal', 'MSF', 'MMR']].copy()
                            mmr_display.columns = ['ETF', 'Price', 'Signal', 'MSF', 'MMR']
                            st.dataframe(mmr_display, width="stretch", hide_index=True, height=300)
                        else:
                            st.caption("No MMR-dominant signals found.")

                with signal_tab2:
                    # Divergence analysis
                    bull_divs = results_df[results_df['Divergence'] == 'BULL']
                    bear_divs = results_df[results_df['Divergence'] == 'BEAR']

                    if not bull_divs.empty or not bear_divs.empty:
                        div_col1, div_col2 = st.columns(2)

                        with div_col1:
                            if not bull_divs.empty:
                                st.markdown("**Bullish Divergences** (Price down, Signal up)")
                                div_display = bull_divs[['DisplayName', 'Price', 'Signal', 'Divergence']].copy()
                                div_display.columns = ['ETF', 'Price', 'Signal', 'Type']
                                st.dataframe(div_display, width="stretch", hide_index=True)
                            else:
                                st.caption("No bullish divergences detected.")

                        with div_col2:
                            if not bear_divs.empty:
                                st.markdown("**Bearish Divergences** (Price up, Signal down)")
                                div_display = bear_divs[['DisplayName', 'Price', 'Signal', 'Divergence']].copy()
                                div_display.columns = ['ETF', 'Price', 'Signal', 'Type']
                                st.dataframe(div_display, width="stretch", hide_index=True)
                            else:
                                st.caption("No bearish divergences detected.")
                    else:
                        st.info("No divergences detected in current scan.")

                with signal_tab3:
                    # Signal extremes chart
                    st.plotly_chart(create_ranking_chart(results_df, 15), width='stretch', key="ranking_15")

            with tab_structure:
                # Market context and structure
                comps.render_section_header("Market Composition", "Universe statistics, regime breakdown, and performance", icon="grid", accent="info")

                struct_tab1, struct_tab2, struct_tab3 = st.tabs(["Statistics", "Regime Mapping", "Performance"])

                with struct_tab1:
                    # Statistical breakdown
                    stat_col1, stat_col2 = st.columns(2)

                    with stat_col1:
                        st.markdown("**Signal Metrics**")
                        signal_stats = {
                            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Q1 (25%)", "Q3 (75%)"],
                            "Value": [
                                f"{results_df['Signal'].mean():.2f}",
                                f"{results_df['Signal'].median():.2f}",
                                f"{results_df['Signal'].std():.2f}",
                                f"{results_df['Signal'].min():.2f}",
                                f"{results_df['Signal'].max():.2f}",
                                f"{results_df['Signal'].quantile(0.25):.2f}",
                                f"{results_df['Signal'].quantile(0.75):.2f}",
                            ]
                        }
                        st.dataframe(pd.DataFrame(signal_stats), width="stretch", hide_index=True)

                    with stat_col2:
                        st.markdown("**Zone Distribution**")
                        zone_stats = {
                            "Zone": ["Oversold", "Neutral", "Overbought"],
                            "Count": [n_oversold, n_neutral, n_overbought],
                            "Pct": [f"{n_oversold/len(results_df)*100:.1f}%", f"{n_neutral/len(results_df)*100:.1f}%", f"{n_overbought/len(results_df)*100:.1f}%"]
                        }
                        st.dataframe(pd.DataFrame(zone_stats), width="stretch", hide_index=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Trigger Summary**")
                        trigger_stats = {
                            "Trigger": ["Buy", "Sell", "Neutral"],
                            "Count": [n_buys, n_sells, len(results_df) - n_buys - n_sells],
                            "Pct": [f"{n_buys/len(results_df)*100:.1f}%", f"{n_sells/len(results_df)*100:.1f}%", f"{(len(results_df)-n_buys-n_sells)/len(results_df)*100:.1f}%"]
                        }
                        st.dataframe(pd.DataFrame(trigger_stats), width="stretch", hide_index=True)

                with struct_tab2:
                    # Regime breakdown
                    regime_col1, regime_col2 = st.columns(2)

                    with regime_col1:
                        st.markdown("**Bullish Regime ETFs**")
                        bull_etfs = results_df[results_df['Regime'].str.contains('BULL', na=False)].sort_values('HMM_Bull', ascending=False).head(15)
                        if not bull_etfs.empty:
                            bull_display = bull_etfs[['DisplayName', 'Regime', 'HMM_Bull', 'HMM_Bear']].copy()
                            bull_display.columns = ['ETF', 'Regime', 'P(Bull)', 'P(Bear)']
                            st.dataframe(bull_display, width="stretch", hide_index=True, height=350)
                        else:
                            st.info("No ETFs in bullish regime.")

                    with regime_col2:
                        st.markdown("**Bearish Regime ETFs**")
                        bear_etfs = results_df[results_df['Regime'].str.contains('BEAR', na=False)].sort_values('HMM_Bear', ascending=False).head(15)
                        if not bear_etfs.empty:
                            bear_display = bear_etfs[['DisplayName', 'Regime', 'HMM_Bull', 'HMM_Bear']].copy()
                            bear_display.columns = ['ETF', 'Regime', 'P(Bull)', 'P(Bear)']
                            st.dataframe(bear_display, width="stretch", hide_index=True, height=350)
                        else:
                            st.info("No ETFs in bearish regime.")

                    # Change points
                    change_points = results_df[results_df['Change_Point'] == True]
                    if not change_points.empty:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Regime Change Points Detected**")
                        cp_display = change_points[['DisplayName', 'Regime', 'Signal', 'Confidence']].copy()
                        cp_display.columns = ['ETF', 'Regime', 'Signal', 'Confidence']
                        st.dataframe(cp_display, width="stretch", hide_index=True)

                with struct_tab3:
                    # Performance metrics
                    perf_col1, perf_col2 = st.columns(2)

                    with perf_col1:
                        st.markdown("**Top Gainers Today**")
                        gainers = results_df.nlargest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']].copy()
                        gainers.columns = ['ETF', 'Price', 'Chg %', 'Signal']
                        st.dataframe(gainers, width="stretch", hide_index=True)

                    with perf_col2:
                        st.markdown("**Top Losers Today**")
                        losers = results_df.nsmallest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']].copy()
                        losers.columns = ['ETF', 'Price', 'Chg %', 'Signal']
                        st.dataframe(losers, width="stretch", hide_index=True)

            with tab_data:
                
                # Filter options
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                with filter_col1:
                    zone_filter = st.multiselect("Filter by Zone", ["Oversold", "Neutral", "Overbought"], default=["Oversold", "Neutral", "Overbought"], key="etf_zone_filter")
                with filter_col2:
                    signal_filter = st.multiselect("Filter by Trigger", ["BUY", "SELL", "-"], default=["BUY", "SELL", "-"], key="etf_signal_filter")
                with filter_col3:
                    sort_by = st.selectbox("Sort by", ["Signal", "Change", "Price", "DisplayName", "Regime", "Confidence"], index=0, key="etf_sort_by")
                
                # Apply filters
                filtered_df = results_df[
                    (results_df['Zone'].isin(zone_filter)) & 
                    (results_df['Trigger'].isin(signal_filter))
                ].sort_values(sort_by, ascending=(sort_by == 'DisplayName'))
                
                # Updated display columns with Regime Intelligence
                display_cols = ['DisplayName', 'Price', 'Change', 'Signal', 'Zone', 'Trigger', 'Regime', 'Vol_Regime', 'Confidence']
                display_df = filtered_df[display_cols].copy()
                display_df.columns = ['ETF', 'Price', 'Chg %', 'Signal', 'Zone', 'Trigger', 'HMM Regime', 'Vol Regime', 'Conf']
                
                st.dataframe(display_df, width="stretch", hide_index=True, height=400)
                
                st.markdown("<br>", unsafe_allow_html=True)
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Report (CSV)",
                    data=csv_data,
                    file_name=f"nirnay_etf_screener_{analysis_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No data retrieved. Please check your internet connection.")


def run_market_screener_mode(length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_date, run_clicked):
    """Market Screener: Full NIRNAY analysis (MSF + MMR + Regime) on F&O / Index stocks"""
    
    # Format analysis date
    analysis_date = spread_date if spread_date else datetime.date.today()
    analysis_date_str = analysis_date.strftime("%d %b %Y")
    is_today = analysis_date == datetime.date.today()

    # Display universe info (needed for both description and results)
    if spread_universe == "India Indexes" and spread_index == "F&O Stocks":
        universe_title = "F&O Stocks"
        universe_desc = "Full NIRNAY (MSF + MMR + Regime) analysis across all F&O securities from NSE."
    elif spread_universe in ("India Indexes", "US Indexes"):
        universe_title = spread_index if spread_index else "Index"
        universe_desc = f"Full NIRNAY (MSF + MMR + Regime) analysis across all constituents of {universe_title}."
    elif spread_universe == "Commodities":
        universe_title = "Commodities"
        universe_desc = f"Full NIRNAY (MSF + MMR + Regime) analysis across {len(COMMODITY_TICKERS)} commodity futures."
    elif spread_universe == "Currency":
        universe_title = "Currency"
        universe_desc = f"Full NIRNAY (MSF + MMR + Regime) analysis across {len(CURRENCY_TICKERS)} currency pairs."
    else:
        universe_title = spread_universe or "Unknown"
        universe_desc = "Full NIRNAY analysis."

    accent_color = "cyan" if spread_universe == "India Indexes" else ("amber" if spread_universe == "Commodities" else ("violet" if spread_universe == "Currency" else "emerald"))

    if not run_clicked:
        # Top spacing
        comps.section_gap()
        comps.section_gap()
        comps.render_section_header(
            f"Market Screener — {universe_title}",
            f"{universe_desc} Analysis Date: {analysis_date_str} {'(Today)' if is_today else ''}",
            icon="target",
            accent=accent_color
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Universe Overview Section
        col1, col2, col3, col4 = st.columns(4)

        # Dynamically show metrics based on selected universe
        if spread_universe == "India Indexes" and spread_index == "F&O Stocks":
            with col1:
                comps.render_metric_card("Universe", "F&O", "NSE Securities", "info")
            with col2:
                comps.render_metric_card("Signal Engines", "3", "MSF + MMR + Regime", "cyan")
            with col3:
                comps.render_metric_card("Output Metrics", "10", "Signal + Regime + HMM", "warning")
            with col4:
                comps.render_metric_card("Analysis Mode", "Single Day", f"Date: {analysis_date_str}", "neutral")
        elif spread_universe in ("India Indexes", "US Indexes"):
            num_symbols = len(get_index_stock_list(spread_index)[0]) if get_index_stock_list(spread_index)[0] else 0
            with col1:
                comps.render_metric_card("Universe", spread_index, f"{num_symbols} constituents" if num_symbols else "Index", "info")
            with col2:
                comps.render_metric_card("Signal Engines", "3", "MSF + MMR + Regime", "cyan")
            with col3:
                comps.render_metric_card("Output Metrics", "10", "Signal + Regime + HMM", "warning")
            with col4:
                comps.render_metric_card("Analysis Mode", "Single Day", f"Date: {analysis_date_str}", "neutral")
        elif spread_universe == "Commodities":
            with col1:
                comps.render_metric_card("Universe", "Commodities", f"{len(COMMODITY_TICKERS)} futures", "info")
            with col2:
                comps.render_metric_card("Signal Engines", "3", "MSF + MMR + Regime", "amber")
            with col3:
                comps.render_metric_card("Output Metrics", "10", "Signal + Regime + HMM", "warning")
            with col4:
                comps.render_metric_card("Analysis Mode", "Single Day", f"Date: {analysis_date_str}", "neutral")
        elif spread_universe == "Currency":
            with col1:
                comps.render_metric_card("Universe", "FX Pairs", f"{len(CURRENCY_TICKERS)} pairs", "info")
            with col2:
                comps.render_metric_card("Signal Engines", "3", "MSF + MMR + Regime", "violet")
            with col3:
                comps.render_metric_card("Output Metrics", "10", "Signal + Regime + HMM", "warning")
            with col4:
                comps.render_metric_card("Analysis Mode", "Single Day", f"Date: {analysis_date_str}", "neutral")

        comps.section_gap()
        comps.section_gap()

        comps.render_section_header(
            "Analysis Framework",
            "Market Structure (MSF) + Macro Regression (MMR) decomposition with regime-aware signal scoring",
            icon="layers",
            accent="emerald"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #06B6D4; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Market Structure</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    Internal price structure via momentum ROC, efficiency ratio, and microstructure decomposition.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(212, 168, 83, 0.08); border: 1px solid rgba(212, 168, 83, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #D4A853; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Macro Regression</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    Macro correlation tracking bond yields, currencies, and commodity flows for regime shifts.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #A855F7; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Regime Intelligence</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    HMM state evolution, volatility regime distribution, and change point timeline.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Validate analysis date
    if analysis_date > datetime.date.today():
        st.error("⚠️ Analysis date cannot be in the future.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    if run_clicked:
        progress_slot = st.empty()

        # Fetch stock list based on universe selection
        theme.progress_bar(progress_slot, 5, f"Fetching {universe_title} list", "Initializing analysis...")

        if spread_universe == "India Indexes" and spread_index == "F&O Stocks":
            stock_list, fetch_msg = get_fno_stock_list()
        elif spread_universe in ("India Indexes", "US Indexes"):
            stock_list, fetch_msg = get_index_stock_list(spread_index)
        elif spread_universe == "Commodities":
            stock_list, fetch_msg = get_commodity_list()
        elif spread_universe == "Currency":
            stock_list, fetch_msg = get_currency_list()
        else:
            stock_list, fetch_msg = None, "Unknown universe"

        if not stock_list:
            st.error(f"Failed to fetch stock list: {fetch_msg}")
            progress_slot.empty()
            return

        st.toast(fetch_msg)
        total_stocks = len(stock_list)

        # Batch download data
        theme.progress_bar(progress_slot, 10, "Downloading Data", f"{total_stocks} securities to analyze")

        data_dict, batch_msg = fetch_batch_data(stock_list, end_date=analysis_date, days_back=100)

        if data_dict is None:
            st.error(f"Failed to download data: {batch_msg}")
            progress_slot.empty()
            return

        st.toast(batch_msg)

        # Fetch macro data ONCE for all stocks (VIX, DXY, rates are market-wide)
        theme.progress_bar(progress_slot, 15, "Macro Data", "Fetching global macro factors for MMR...")
        macro_df = fetch_macro_data(days_back=100)

        # Process each stock
        results = []
        valid_tickers = list(data_dict.keys())
        total_valid = len(valid_tickers)

        for i, ticker in enumerate(valid_tickers):
            pct = 15 + int(75 * (i + 1) / total_valid)
            theme.progress_bar(progress_slot, pct, f"Analyzing {ticker.replace('.NS', '')}", f"{i+1}/{total_valid} securities")
            
            df = data_dict[ticker]
            
            if df is not None and len(df) > length + 5:
                try:
                    # Merge macro data with stock data for MMR
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.join(macro_df, how='left').ffill()
                    
                    # Run FULL analysis (MSF + MMR + Regime Intelligence)
                    df, _ = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    
                    # Get the closest date on or before analysis_date
                    analysis_datetime = pd.Timestamp(analysis_date)
                    valid_dates = df.index[df.index <= analysis_datetime]
                    
                    if len(valid_dates) == 0:
                        continue  # No data for this date
                    
                    target_date = valid_dates[-1]
                    target_idx = df.index.get_loc(target_date)
                    
                    if target_idx < 1:
                        continue  # Need at least one previous row for change calculation
                    
                    last_row = df.iloc[target_idx]
                    prev_row = df.iloc[target_idx - 1]
                    price_change = ((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                    
                    signal_str = "BUY" if last_row['Buy_Signal'] else "SELL" if last_row['Sell_Signal'] else "-"
                    div_str = "BULL" if last_row['Bullish_Div'] else "BEAR" if last_row['Bearish_Div'] else "-"
                    
                    results.append({
                        "Symbol": ticker,
                        "DisplayName": ticker.replace(".NS", ""),
                        "Price": round(last_row['Close'], 2),
                        "Change": round(price_change, 2),
                        "Signal": round(last_row['Unified_Osc'], 2),
                        "MSF": round(last_row['MSF_Osc'], 2),
                        "MMR": round(last_row['MMR_Osc'], 2),  # Now populated with actual MMR
                        "Zone": last_row['Condition'],
                        "Trigger": signal_str,
                        "Divergence": div_str,
                        "Agreement": round(last_row['Agreement'], 3),
                        # Regime Intelligence columns
                        "Regime": last_row['Regime'],
                        "HMM_Bull": round(last_row['HMM_Bull'], 2),
                        "HMM_Bear": round(last_row['HMM_Bear'], 2),
                        "Vol_Regime": last_row['Vol_Regime'],
                        "Confidence": round(last_row['Confidence'], 2),
                        "Change_Point": last_row['Change_Point']
                    })
                except Exception:
                    pass

        progress_slot.empty()

        if results:
            st.toast(f"Market Scan Complete! Analyzed {len(results)}/{total_stocks} stocks")
            results_df = pd.DataFrame(results)
            
            # Calculate summary stats
            n_oversold = len(results_df[results_df['Zone'] == 'Oversold'])
            n_overbought = len(results_df[results_df['Zone'] == 'Overbought'])
            n_neutral = len(results_df[results_df['Zone'] == 'Neutral'])
            n_buys = len(results_df[results_df['Trigger'] == 'BUY'])
            n_sells = len(results_df[results_df['Trigger'] == 'SELL'])
            avg_signal = results_df['Signal'].mean()
            
            # NEW: Calculate HMM regime distribution
            n_bull = len(results_df[results_df['Regime'].str.contains('BULL', na=False)])
            n_bear = len(results_df[results_df['Regime'].str.contains('BEAR', na=False)])
            n_transition = len(results_df[results_df['Regime'] == 'TRANSITION'])
            dominant_regime = results_df['Regime'].mode().iloc[0] if len(results_df) > 0 else "NEUTRAL"
            regime_color = "success" if "BULL" in dominant_regime else "danger" if "BEAR" in dominant_regime else "warning" if dominant_regime == "TRANSITION" else "neutral"
            
            # Metrics row
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                comps.render_metric_card("Universe", f"{len(results)}", f"{universe_title} Analyzed", "info")
            with c2:
                comps.render_metric_card("Oversold", f"{n_oversold}", "Buy Zone", "success")
            with c3:
                comps.render_metric_card("Overbought", f"{n_overbought}", "Sell Zone", "danger")
            with c4:
                comps.render_metric_card("Buy Signals", f"{n_buys}", "Confirmed", "primary")
            with c5:
                comps.render_metric_card("Sell Signals", f"{n_sells}", "Confirmed", "warning")
            with c6:
                comps.render_metric_card("HMM Regime", dominant_regime, f"Bull: {n_bull} | Bear: {n_bear}", regime_color)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Fresh institutional research design: Executive → Opportunities → Analysis → Context → Data
            tab_summary, tab_opps, tab_signals, tab_structure, tab_data = st.tabs(["Summary", "Opportunities", "Signal Analysis", "Market Structure", "Data Explorer"])

            with tab_summary:
                # Executive snapshot for decision makers
                comps.render_section_header("Executive Summary", "Key performance indicators and regime context", icon="list", accent="cyan")

                # KPI metrics in grid
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
                with metric_col1:
                    comps.render_metric_card("Universe", f"{len(results_df)}", "Securities", "info")
                with metric_col2:
                    comps.render_metric_card("Buy Signals", f"{n_buys}", "Confirmed", "success")
                with metric_col3:
                    comps.render_metric_card("Sell Signals", f"{n_sells}", "Confirmed", "danger")
                with metric_col4:
                    oversold_pct = f"{n_oversold/len(results_df)*100:.0f}%"
                    comps.render_metric_card("Oversold", oversold_pct, f"{n_oversold} assets", "info")
                with metric_col5:
                    overbought_pct = f"{n_overbought/len(results_df)*100:.0f}%"
                    comps.render_metric_card("Overbought", overbought_pct, f"{n_overbought} assets", "warning")
                with metric_col6:
                    comps.render_metric_card("Avg Signal", f"{results_df['Signal'].mean():.1f}", "Population mean", "neutral")

                st.markdown("<br>", unsafe_allow_html=True)

                # Regime snapshot
                comps.render_section_header("Market Regime — HMM State & Volatility", "Current market conditions and probability states", icon="layers", accent="emerald")
                regime_col1, regime_col2 = st.columns(2)

                with regime_col1:
                    regime_counts = results_df['Regime'].value_counts()
                    regime_colors = {'BULL': '#10b981', 'WEAK_BULL': '#34d399', 'NEUTRAL': '#888888', 'WEAK_BEAR': '#fbbf24', 'BEAR': '#ef4444', 'TRANSITION': '#a855f7'}
                    fig_regime = go.Figure(go.Pie(
                        labels=regime_counts.index, values=regime_counts.values, hole=0.5,
                        marker=dict(colors=[regime_colors.get(r, '#888888') for r in regime_counts.index], line=dict(color='#1A1A1A', width=2)),
                        textinfo='label+percent', textfont=dict(size=11, color='white')
                    ))
                    fig_regime.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False), title=dict(text='HMM Regime', font=dict(size=12, color='#888888')))
                    st.plotly_chart(fig_regime, width='stretch', key="regime_pie")

                with regime_col2:
                    vol_counts = results_df['Vol_Regime'].value_counts()
                    vol_colors = {'LOW': '#10b981', 'NORMAL': '#888888', 'HIGH': '#f59e0b', 'EXTREME': '#ef4444'}
                    fig_vol = go.Figure(go.Pie(
                        labels=vol_counts.index, values=vol_counts.values, hole=0.5,
                        marker=dict(colors=[vol_colors.get(v, '#888888') for v in vol_counts.index], line=dict(color='#1A1A1A', width=2)),
                        textinfo='label+percent', textfont=dict(size=11, color='white')
                    ))
                    fig_vol.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False), title=dict(text='Volatility Regime', font=dict(size=12, color='#888888')))
                    st.plotly_chart(fig_vol, width='stretch', key="vol_pie")

                st.markdown("<br>", unsafe_allow_html=True)

                # Signal distribution
                comps.render_section_header("Signal Distribution", "Cross-sectional view of signal spread", icon="trending-up", accent="amber")
                st.plotly_chart(create_distribution_chart(results_df), width='stretch', key="distribution")

            with tab_opps:
                # Ranked opportunities for traders
                comps.render_section_header("Opportunity Ranking", "All securities ranked by signal strength and conviction", icon="target", accent="amber")

                opp_tab1, opp_tab2 = st.tabs(["Buy Setup", "Sell Setup"])

                with opp_tab1:
                    # Buy opportunities
                    buy_data = results_df[(results_df['Trigger'] == 'BUY') | (results_df['Zone'] == 'Oversold')].copy()
                    buy_data = buy_data.sort_values('Signal').head(25)

                    if not buy_data.empty:
                        buy_display = buy_data[['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'Zone']].copy()
                        buy_display.columns = ['Symbol', 'Price', 'Chg %', 'Signal', 'MSF', 'Zone']
                        st.dataframe(buy_display, width="stretch", hide_index=True)
                        st.caption(f"Showing {len(buy_data)} buy setup candidates. Sorted by signal strength (lowest to highest).")
                    else:
                        st.info("No buy opportunities in current scan.")

                with opp_tab2:
                    # Sell opportunities
                    sell_data = results_df[(results_df['Trigger'] == 'SELL') | (results_df['Zone'] == 'Overbought')].copy()
                    sell_data = sell_data.sort_values('Signal', ascending=False).head(25)

                    if not sell_data.empty:
                        sell_display = sell_data[['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'Zone']].copy()
                        sell_display.columns = ['Symbol', 'Price', 'Chg %', 'Signal', 'MSF', 'Zone']
                        st.dataframe(sell_display, width="stretch", hide_index=True)
                        st.caption(f"Showing {len(sell_data)} sell setup candidates. Sorted by signal strength (highest to lowest).")
                    else:
                        st.info("No sell opportunities in current scan.")

            with tab_signals:
                # Deep signal analysis
                comps.render_section_header("Signal Decomposition", "MSF vs MMR component analysis and divergence alerts", icon="layers", accent="violet")

                signal_tab1, signal_tab2, signal_tab3 = st.tabs(["Components", "Divergences", "Extremes"])

                with signal_tab1:
                    # MSF vs MMR breakdown
                    comp_col1, comp_col2 = st.columns(2)

                    with comp_col1:
                        st.markdown("**MSF-Dominant (Structure)**")
                        msf_heavy = results_df[results_df['MSF'].abs() > results_df['MMR'].abs()].sort_values('MSF').head(15)
                        if not msf_heavy.empty:
                            msf_display = msf_heavy[['DisplayName', 'Price', 'Signal', 'MSF']].copy()
                            msf_display.columns = ['Symbol', 'Price', 'Signal', 'MSF']
                            st.dataframe(msf_display, width="stretch", hide_index=True, height=350)
                        else:
                            st.caption("No MSF-dominant signals found.")

                    with comp_col2:
                        st.markdown("**MMR-Dominant (Macro)**")
                        mmr_heavy = results_df[results_df['MMR'].abs() > results_df['MSF'].abs()].sort_values('MMR').head(15)
                        if not mmr_heavy.empty:
                            mmr_display = mmr_heavy[['DisplayName', 'Price', 'Signal', 'MMR']].copy()
                            mmr_display.columns = ['Symbol', 'Price', 'Signal', 'MMR']
                            st.dataframe(mmr_display, width="stretch", hide_index=True, height=350)
                        else:
                            st.caption("No MMR-dominant signals found.")

                with signal_tab2:
                    # Divergence analysis
                    bull_divs = results_df[results_df['Divergence'] == 'BULL']
                    bear_divs = results_df[results_df['Divergence'] == 'BEAR']

                    if not bull_divs.empty or not bear_divs.empty:
                        div_col1, div_col2 = st.columns(2)

                        with div_col1:
                            if not bull_divs.empty:
                                st.markdown("**Bullish Divergences** (Price down, Signal up)")
                                div_display = bull_divs[['DisplayName', 'Price', 'Signal', 'Divergence']].copy()
                                div_display.columns = ['Symbol', 'Price', 'Signal', 'Type']
                                st.dataframe(div_display, width="stretch", hide_index=True)
                            else:
                                st.caption("No bullish divergences detected.")

                        with div_col2:
                            if not bear_divs.empty:
                                st.markdown("**Bearish Divergences** (Price up, Signal down)")
                                div_display = bear_divs[['DisplayName', 'Price', 'Signal', 'Divergence']].copy()
                                div_display.columns = ['Symbol', 'Price', 'Signal', 'Type']
                                st.dataframe(div_display, width="stretch", hide_index=True)
                            else:
                                st.caption("No bearish divergences detected.")
                    else:
                        st.info("No divergences detected in current scan.")

                with signal_tab3:
                    # Signal extremes chart
                    st.plotly_chart(create_ranking_chart(results_df, 20), width='stretch', key="ranking_20")

            with tab_structure:
                # Market context and structure
                comps.render_section_header("Market Composition", "Universe statistics, regime breakdown, and performance", icon="grid", accent="info")

                struct_tab1, struct_tab2, struct_tab3 = st.tabs(["Statistics", "Regime Mapping", "Performance"])

                with struct_tab1:
                    # Statistical breakdown
                    stat_col1, stat_col2 = st.columns(2)

                    with stat_col1:
                        st.markdown("**Signal Metrics**")
                        signal_stats = {
                            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Q1 (25%)", "Q3 (75%)"],
                            "Value": [
                                f"{results_df['Signal'].mean():.2f}",
                                f"{results_df['Signal'].median():.2f}",
                                f"{results_df['Signal'].std():.2f}",
                                f"{results_df['Signal'].min():.2f}",
                                f"{results_df['Signal'].max():.2f}",
                                f"{results_df['Signal'].quantile(0.25):.2f}",
                                f"{results_df['Signal'].quantile(0.75):.2f}",
                            ]
                        }
                        st.dataframe(pd.DataFrame(signal_stats), width="stretch", hide_index=True)

                    with stat_col2:
                        st.markdown("**Zone Distribution**")
                        zone_stats = {
                            "Zone": ["Oversold", "Neutral", "Overbought"],
                            "Count": [n_oversold, n_neutral, n_overbought],
                            "Pct": [f"{n_oversold/len(results_df)*100:.1f}%", f"{n_neutral/len(results_df)*100:.1f}%", f"{n_overbought/len(results_df)*100:.1f}%"]
                        }
                        st.dataframe(pd.DataFrame(zone_stats), width="stretch", hide_index=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Trigger Summary**")
                        trigger_stats = {
                            "Trigger": ["Buy", "Sell", "Neutral"],
                            "Count": [n_buys, n_sells, len(results_df) - n_buys - n_sells],
                            "Pct": [f"{n_buys/len(results_df)*100:.1f}%", f"{n_sells/len(results_df)*100:.1f}%", f"{(len(results_df)-n_buys-n_sells)/len(results_df)*100:.1f}%"]
                        }
                        st.dataframe(pd.DataFrame(trigger_stats), width="stretch", hide_index=True)

                with struct_tab2:
                    # Regime breakdown
                    regime_col1, regime_col2 = st.columns(2)

                    with regime_col1:
                        st.markdown("**Bullish Regime Securities**")
                        bull_assets = results_df[results_df['Regime'].str.contains('BULL', na=False)].sort_values('HMM_Bull', ascending=False).head(15)
                        if not bull_assets.empty:
                            bull_display = bull_assets[['DisplayName', 'Regime', 'HMM_Bull', 'HMM_Bear']].copy()
                            bull_display.columns = ['Symbol', 'Regime', 'P(Bull)', 'P(Bear)']
                            st.dataframe(bull_display, width="stretch", hide_index=True, height=350)
                        else:
                            st.info("No securities in bullish regime.")

                    with regime_col2:
                        st.markdown("**Bearish Regime Securities**")
                        bear_assets = results_df[results_df['Regime'].str.contains('BEAR', na=False)].sort_values('HMM_Bear', ascending=False).head(15)
                        if not bear_assets.empty:
                            bear_display = bear_assets[['DisplayName', 'Regime', 'HMM_Bull', 'HMM_Bear']].copy()
                            bear_display.columns = ['Symbol', 'Regime', 'P(Bull)', 'P(Bear)']
                            st.dataframe(bear_display, width="stretch", hide_index=True, height=350)
                        else:
                            st.info("No securities in bearish regime.")

                    # Change points
                    change_points = results_df[results_df['Change_Point'] == True]
                    if not change_points.empty:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Regime Change Points Detected**")
                        cp_display = change_points[['DisplayName', 'Regime', 'Signal', 'Confidence']].copy()
                        cp_display.columns = ['Symbol', 'Regime', 'Signal', 'Confidence']
                        st.dataframe(cp_display, width="stretch", hide_index=True)

                with struct_tab3:
                    # Performance metrics
                    perf_col1, perf_col2 = st.columns(2)

                    with perf_col1:
                        st.markdown("**Top Gainers Today**")
                        gainers = results_df.nlargest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']].copy()
                        gainers.columns = ['Symbol', 'Price', 'Chg %', 'Signal']
                        st.dataframe(gainers, width="stretch", hide_index=True)

                    with perf_col2:
                        st.markdown("**Top Losers Today**")
                        losers = results_df.nsmallest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']].copy()
                        losers.columns = ['Symbol', 'Price', 'Chg %', 'Signal']
                        st.dataframe(losers, width="stretch", hide_index=True)

            with tab_data:
                st.markdown(f"##### Complete Market Scan Results ({len(results_df)} stocks) - {analysis_date_str}")
                
                # Filter options
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                with filter_col1:
                    zone_filter = st.multiselect("Filter by Zone", ["Oversold", "Neutral", "Overbought"], default=["Oversold", "Neutral", "Overbought"])
                with filter_col2:
                    signal_filter = st.multiselect("Filter by Trigger", ["BUY", "SELL", "-"], default=["BUY", "SELL", "-"])
                with filter_col3:
                    sort_by = st.selectbox("Sort by", ["Signal", "Change", "Price", "DisplayName"], index=0)
                
                # Apply filters
                filtered_df = results_df[
                    (results_df['Zone'].isin(zone_filter)) & 
                    (results_df['Trigger'].isin(signal_filter))
                ].sort_values(sort_by, ascending=(sort_by == 'DisplayName'))
                
                display_cols = ['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'Zone', 'Trigger', 'Divergence']
                display_df = filtered_df[display_cols].copy()
                display_df.columns = ['Symbol', 'Price', 'Chg %', 'Signal', 'MSF', 'Zone', 'Trigger', 'Divergence']
                
                st.dataframe(display_df, width="stretch", hide_index=True, height=500)
                
                st.markdown("<br>", unsafe_allow_html=True)
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Report (CSV)",
                    data=csv_data,
                    file_name=f"nirnay_market_{universe_title.replace(' ', '_')}_{analysis_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No data retrieved. Please check your internet connection or try a different universe.")


def run_market_timeseries_mode(length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, start_date, end_date, run_clicked):
    """Market Time Series Analysis: Full NIRNAY (MSF + MMR + Regime) tracking over time"""
    
    # Validate dates
    if start_date is None or end_date is None:
        st.error("Please select both start and end dates.")
        return
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    
    if end_date > datetime.date.today():
        st.error("End date cannot be in the future.")
        return
    
    # Calculate date range
    date_range_days = (end_date - start_date).days

    # Display info (needed for both description and results)
    if spread_universe == "India Indexes" and spread_index == "F&O Stocks":
        universe_title = "F&O Stocks"
    elif spread_universe == "Commodities":
        universe_title = "Commodities"
    elif spread_universe == "Currency":
        universe_title = "Currency"
    elif spread_universe in ("India Indexes", "US Indexes") and spread_index:
        universe_title = spread_index
    else:
        universe_title = spread_universe or "Unknown"

    if not run_clicked:
        # Top spacing
        comps.section_gap()
        comps.section_gap()
        comps.render_section_header(
            f"Time Series Analysis — {universe_title}",
            f"Full NIRNAY (MSF + MMR + Regime) signal distribution over time · {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')} ({date_range_days} days)",
            icon="trending",
            accent="cyan"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Analysis Overview Section
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            comps.render_metric_card("Period", f"{date_range_days}", "Trading Days", "info")
        with col2:
            comps.render_metric_card("Signal Engines", "3", "MSF + MMR + Regime", "cyan")
        with col3:
            comps.render_metric_card("Output Metrics", "10", "Signal + Regime + HMM", "warning")
        with col4:
            comps.render_metric_card("Analysis Mode", "Time Series", "Signal Evolution", "neutral")

        comps.section_gap()
        comps.section_gap()

        # Methodology Section
        comps.render_section_header(
            "Analysis Framework",
            "Rolling window signal tracking with regime state evolution and volatility distribution",
            icon="layers",
            accent="emerald"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #06B6D4; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Market Structure</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    Rolling momentum ROC, efficiency ratio, and microstructure tracking across date range.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(212, 168, 83, 0.08); border: 1px solid rgba(212, 168, 83, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #D4A853; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Macro Regression</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    Macro correlation tracking over time with yield, currency, and commodity dynamics.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #A855F7; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Regime Intelligence</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    HMM state evolution, volatility regime distribution, and change point timeline.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if run_clicked:
        progress_slot = st.empty()

        # Fetch stock list
        theme.progress_bar(progress_slot, 5, f"Fetching {universe_title} list", "Initializing analysis...")

        if spread_universe == "India Indexes" and spread_index == "F&O Stocks":
            stock_list, fetch_msg = get_fno_stock_list()
        elif spread_universe in ("India Indexes", "US Indexes"):
            stock_list, fetch_msg = get_index_stock_list(spread_index)
        elif spread_universe == "Commodities":
            stock_list, fetch_msg = get_commodity_list()
        elif spread_universe == "Currency":
            stock_list, fetch_msg = get_currency_list()
        else:
            stock_list, fetch_msg = None, "Unknown universe"

        if not stock_list:
            st.error(f"Failed to fetch stock list: {fetch_msg}")
            progress_slot.empty()
            return

        st.toast(fetch_msg)
        total_stocks = len(stock_list)

        # Batch download data for entire period
        theme.progress_bar(progress_slot, 10, "Downloading Data", f"{total_stocks} securities historical data")

        data_dict, batch_msg = fetch_batch_data(stock_list, end_date=end_date, days_back=100 + date_range_days)

        if data_dict is None:
            st.error(f"Failed to download data: {batch_msg}")
            progress_slot.empty()
            return

        st.toast(batch_msg)

        # Fetch macro data ONCE for all stocks (VIX, DXY, rates are market-wide)
        theme.progress_bar(progress_slot, 15, "Macro Data", "Fetching global macro factors for MMR...")
        macro_df = fetch_macro_data(days_back=100 + date_range_days)

        # Generate list of trading days to analyze
        theme.progress_bar(progress_slot, 18, "Trading Calendar", "Identifying trading days in range...")

        # Use one of the stocks to identify trading days
        sample_ticker = list(data_dict.keys())[0]
        sample_df = data_dict[sample_ticker]
        sample_df.index = pd.to_datetime(sample_df.index)
        if sample_df.index.tz is not None:
            sample_df.index = sample_df.index.tz_localize(None)

        # Get trading days in range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        trading_days = sample_df.index[(sample_df.index >= start_ts) & (sample_df.index <= end_ts)].tolist()

        if len(trading_days) == 0:
            st.error("No trading days found in the selected date range.")
            progress_slot.empty()
            return

        # Check if requested end date data is available
        actual_last_date = trading_days[-1].date() if trading_days else None
        is_today_included = actual_last_date == datetime.date.today() if actual_last_date else False

        if end_date == datetime.date.today():
            if is_today_included:
                st.toast(f"Live Data Included - {actual_last_date.strftime('%d %b %Y')}")
            else:
                st.toast(f"Data through {actual_last_date.strftime('%d %b %Y')}")
        elif actual_last_date and actual_last_date < end_date:
            st.toast(f"Data through {actual_last_date.strftime('%d %b %Y')}")

        st.toast(f"Found {len(trading_days)} trading days")

        # Process FULL analysis for all stocks (MSF + MMR + Regime)
        theme.progress_bar(progress_slot, 20, "Computing Signals", "Computing MSF + MMR + Regime for all stocks...")
        
        processed_data = {}
        valid_tickers = list(data_dict.keys())
        
        for i, ticker in enumerate(valid_tickers):
            df = data_dict[ticker]
            if df is not None and len(df) > length + 5:
                try:
                    # Merge macro data with stock data for MMR
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.join(macro_df, how='left').ffill()
                    
                    # Run FULL analysis (MSF + MMR + Regime Intelligence)
                    df, _ = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    processed_data[ticker] = df
                except Exception:
                    pass
            
            if (i + 1) % 50 == 0:
                pct = 20 + int(30 * (i + 1) / len(valid_tickers))
                theme.progress_bar(progress_slot, pct, "Computing Signals", f"{i+1}/{len(valid_tickers)} stocks processed")

        theme.progress_bar(progress_slot, 50, "Analyzing Timeline", f"Processing {len(trading_days)} trading days...")

        # Analyze each trading day
        timeseries_results = []

        for day_idx, trading_day in enumerate(trading_days):
            pct = 50 + int(45 * (day_idx + 1) / len(trading_days))
            theme.progress_bar(progress_slot, pct, "Timeline Analysis", f"{day_idx+1}/{len(trading_days)} days")
            
            day_stats = {
                "Date": trading_day.date(),
                "Oversold": 0,
                "Overbought": 0,
                "Neutral": 0,
                "Buy_Signals": 0,
                "Sell_Signals": 0,
                "Total_Analyzed": 0,
                "Avg_Signal": 0,
                "Signal_Sum": 0,
                "Bull_Div": 0,
                "Bear_Div": 0,
                # NEW: Regime Intelligence stats
                "Regime_Bull": 0,
                "Regime_Bear": 0,
                "Regime_Neutral": 0,
                "Regime_Transition": 0,
                "Vol_High": 0,
                "Vol_Low": 0,
                "Change_Points": 0
            }
            
            for ticker, df in processed_data.items():
                try:
                    # Get data for this trading day
                    if trading_day not in df.index:
                        continue
                    
                    row = df.loc[trading_day]
                    
                    day_stats["Total_Analyzed"] += 1
                    day_stats["Signal_Sum"] += row['Unified_Osc']
                    
                    if row['Condition'] == 'Oversold':
                        day_stats["Oversold"] += 1
                    elif row['Condition'] == 'Overbought':
                        day_stats["Overbought"] += 1
                    else:
                        day_stats["Neutral"] += 1
                    
                    if row['Buy_Signal']:
                        day_stats["Buy_Signals"] += 1
                    if row['Sell_Signal']:
                        day_stats["Sell_Signals"] += 1
                    if row['Bullish_Div']:
                        day_stats["Bull_Div"] += 1
                    if row['Bearish_Div']:
                        day_stats["Bear_Div"] += 1
                    
                    # NEW: Regime Intelligence stats
                    regime = row['Regime']
                    if 'BULL' in regime:
                        day_stats["Regime_Bull"] += 1
                    elif 'BEAR' in regime:
                        day_stats["Regime_Bear"] += 1
                    elif regime == 'TRANSITION':
                        day_stats["Regime_Transition"] += 1
                    else:
                        day_stats["Regime_Neutral"] += 1
                    
                    vol_regime = row['Vol_Regime']
                    if vol_regime in ['HIGH', 'EXTREME']:
                        day_stats["Vol_High"] += 1
                    elif vol_regime == 'LOW':
                        day_stats["Vol_Low"] += 1
                    
                    if row['Change_Point']:
                        day_stats["Change_Points"] += 1
                        
                except Exception:
                    pass
            
            if day_stats["Total_Analyzed"] > 0:
                day_stats["Avg_Signal"] = day_stats["Signal_Sum"] / day_stats["Total_Analyzed"]
                day_stats["Oversold_Pct"] = (day_stats["Oversold"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Overbought_Pct"] = (day_stats["Overbought"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Neutral_Pct"] = (day_stats["Neutral"] / day_stats["Total_Analyzed"]) * 100
                # NEW: Regime percentages
                day_stats["Regime_Bull_Pct"] = (day_stats["Regime_Bull"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Regime_Bear_Pct"] = (day_stats["Regime_Bear"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Vol_High_Pct"] = (day_stats["Vol_High"] / day_stats["Total_Analyzed"]) * 100
            else:
                day_stats["Oversold_Pct"] = 0
                day_stats["Overbought_Pct"] = 0
                day_stats["Neutral_Pct"] = 0
                day_stats["Regime_Bull_Pct"] = 0
                day_stats["Regime_Bear_Pct"] = 0
                day_stats["Vol_High_Pct"] = 0
            
            timeseries_results.append(day_stats)
        
        progress_slot.empty()
        
        if not timeseries_results:
            st.warning("No data could be analyzed for the selected period.")
            return
        
        ts_df = pd.DataFrame(timeseries_results)
        ts_df['Date'] = pd.to_datetime(ts_df['Date'])
        ts_df = ts_df.sort_values('Date')
        
        # Show actual analyzed date range
        actual_start = ts_df['Date'].min().strftime('%d %b %Y')
        actual_end = ts_df['Date'].max().strftime('%d %b %Y')
        st.toast(f"Time Series Complete! {len(ts_df)} days")
        
        # Summary metrics
        st.markdown("<br>", unsafe_allow_html=True)
        avg_oversold = ts_df['Oversold_Pct'].mean()
        avg_overbought = ts_df['Overbought_Pct'].mean()
        total_buys = ts_df['Buy_Signals'].sum()
        total_sells = ts_df['Sell_Signals'].sum()
        avg_signal = ts_df['Avg_Signal'].mean()
        regime = "BULLISH" if avg_signal < -1 else "BEARISH" if avg_signal > 1 else "NEUTRAL"
        regime_color = "success" if avg_signal < -1 else "danger" if avg_signal > 1 else "neutral"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            comps.render_metric_card("Avg Oversold", f"{avg_oversold:.1f}%", "Daily Average", "success")
        with c2:
            comps.render_metric_card("Avg Overbought", f"{avg_overbought:.1f}%", "Daily Average", "danger")
        with c3:
            comps.render_metric_card("Total Buys", f"{total_buys:,}", "Over Period", "primary")
        with c4:
            comps.render_metric_card("Total Sells", f"{total_sells:,}", "Over Period", "warning")
        with c5:
            comps.render_metric_card("Period Regime", regime, f"Avg: {avg_signal:.2f}", regime_color)
        with c6:
            comps.render_metric_card("Trading Days", f"{len(ts_df)}", "Analyzed", "info")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Tabs for different views
        tab3, tab1, tab2, tab4 = st.tabs(["Regime Analysis", "Signal Dashboard", "Transaction Dynamics", "Data Terminal"])
        
        with tab1:
            comps.render_section_header("Extreme Signal Trends", "Overbought / Oversold Distribution Over Time", icon="activity", accent="cyan")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Shows the percentage of stocks in each zone daily</p>', unsafe_allow_html=True)
            
            # Stacked area chart for zones
            fig_zones = go.Figure()

            fig_zones.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Oversold_Pct'],
                mode='lines', name='Oversold %',
                fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
                line=dict(color=COLOR_GREEN, width=1.5)
            ))

            fig_zones.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Overbought_Pct'],
                mode='lines', name='Overbought %',
                fill='tozeroy', fillcolor='rgba(251,113,133,0.12)',
                line=dict(color=COLOR_RED, width=1.5)
            ))

            ymax = max(ts_df['Oversold_Pct'].max(), ts_df['Overbought_Pct'].max()) * 1.15
            fig_zones.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
            style_axes(fig_zones, y_title="% of Stocks", y_range=[0, ymax])
            st.plotly_chart(fig_zones, width='stretch', key="ts_etf_zones")
            
            st.markdown("<br>", unsafe_allow_html=True)
            comps.render_section_header("Signal Volume Trends", "Raw Counts Over Time", icon="bar-chart", accent="info")
            
            # Bar chart for raw counts
            fig_counts = go.Figure()

            fig_counts.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Oversold'],
                name='Oversold',
                marker=dict(color='rgba(52,211,153,0.85)')
            ))

            fig_counts.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Overbought'],
                name='Overbought',
                marker=dict(color='rgba(251,113,133,0.85)')
            ))

            fig_counts.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM), barmode='group')
            style_axes(fig_counts, y_title="Stock Count")
            st.plotly_chart(fig_counts, width='stretch', key="ts_etf_counts")
        
        with tab2:
            comps.render_section_header("Transaction Signal Trends", "Buy / Sell Signal Counts Over Time", icon="zap", accent="emerald")
            
            fig_signals = go.Figure()
            
            fig_signals.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Buy_Signals'],
                mode='lines+markers', name='Buy Signals',
                line=dict(color=COLOR_GREEN, width=2),
                marker=dict(size=6, color=COLOR_GREEN)
            ))
            
            fig_signals.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Sell_Signals'],
                mode='lines+markers', name='Sell Signals',
                line=dict(color=COLOR_RED, width=2),
                marker=dict(size=6, color=COLOR_RED)
            ))
            
            fig_signals.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
            style_axes(fig_signals, y_title="Signal Count")
            st.plotly_chart(fig_signals, width='stretch', key="market_signals")
            
            st.markdown("<br>", unsafe_allow_html=True)
            comps.render_section_header("Divergence Persistence", "Divergence Signals Over Time", icon="trending-up", accent="amber")
            
            fig_div = go.Figure()
            
            fig_div.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Bull_Div'],
                name='Bullish Divergence', 
                marker=dict(color=COLOR_GOLD, line=dict(color=COLOR_GOLD, width=1))
            ))
            
            fig_div.add_trace(go.Bar(
                x=ts_df['Date'], y=-ts_df['Bear_Div'],
                name='Bearish Divergence', 
                marker=dict(color=COLOR_CYAN, line=dict(color=COLOR_CYAN, width=1))
            ))
            
            fig_div.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM), barmode='relative')
            style_axes(fig_div, y_title="Divergence Count")
            st.plotly_chart(fig_div, width='stretch', key="ts_div")
        
        with tab3:
            # ORIGINAL: Average Signal Value Over Time
            comps.render_section_header("Aggregate Signal Momentum", "Average Signal Value Over Time", icon="activity", accent="rose")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Negative = Bullish Bias | Positive = Bearish Bias</p>', unsafe_allow_html=True)
            
            fig_avg = go.Figure()
            
            colors = [COLOR_GREEN if v < -2 else COLOR_RED if v > 2 else COLOR_MUTED for v in ts_df['Avg_Signal']]
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'].clip(lower=0),
                fill='tozeroy', fillcolor='rgba(251,113,133,0.05)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'].clip(upper=0),
                fill='tozeroy', fillcolor='rgba(52,211,153,0.05)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'],
                mode='lines+markers', name='Avg Signal',
                line=dict(color=COLOR_GOLD, width=2),
                marker=dict(size=6, color=colors)
            ))
            
            fig_avg.add_hline(y=2, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
            fig_avg.add_hline(y=-2, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
            fig_avg.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
            
            fig_avg.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
            style_axes(fig_avg, y_title="Avg Signal", y_range=[-6, 6])
            st.plotly_chart(fig_avg, width='stretch', key="ts_avg")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # NEW: HMM Regime Distribution Over Time
            comps.render_section_header("HMM State Evolution", "HMM Regime Distribution Over Time", icon="layers", accent="cyan")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Percentage of stocks in each HMM regime daily</p>', unsafe_allow_html=True)
            
            # Regime trend chart
            fig_regime = go.Figure()
            
            fig_regime.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Regime_Bull_Pct'],
                mode='lines', name='Bull Regime %',
                fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
                line=dict(color=COLOR_GREEN, width=2)
            ))
            
            fig_regime.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Regime_Bear_Pct'],
                mode='lines', name='Bear Regime %',
                fill='tozeroy', fillcolor='rgba(251,113,133,0.12)',
                line=dict(color=COLOR_RED, width=2)
            ))
            
            fig_regime.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
            style_axes(fig_regime, y_title="% of Stocks", y_range=[0, 100])
            st.plotly_chart(fig_regime, width='stretch', key="ts_regime")
            
            st.markdown("<br>", unsafe_allow_html=True)
            comps.render_section_header("Volatility Dynamics", "Volatility Regime & Change Points Over Time", icon="shield", accent="amber")
            
            # Volatility regime chart
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Vol_High_Pct'],
                mode='lines+markers', name='High Vol %',
                line=dict(color=COLOR_AMBER, width=2),
                marker=dict(size=5)
            ))
            
            fig_vol.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Change_Points'],
                name='Change Points',
                marker=dict(color=COLOR_PURPLE, opacity=0.7)
            ))
            
            fig_vol.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL))
            style_axes(fig_vol, y_title="Count / %")
            st.plotly_chart(fig_vol, width='stretch', key="ts_vol")
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                comps.render_section_header("State Transition Metrics", "HMM Regime Statistics", icon="bar-chart", accent="emerald")
                avg_bull = ts_df['Regime_Bull_Pct'].mean()
                avg_bear = ts_df['Regime_Bear_Pct'].mean()
                total_changes = ts_df['Change_Points'].sum()
                
                regime_stats = {
                    "Metric": ["Avg Bull Regime %", "Avg Bear Regime %", "Total Change Points", "Avg High Vol %"],
                    "Value": [f"{avg_bull:.1f}%", f"{avg_bear:.1f}%", f"{int(total_changes)}", f"{ts_df['Vol_High_Pct'].mean():.1f}%"]
                }
                st.dataframe(pd.DataFrame(regime_stats), width="stretch", hide_index=True)
            
            with col_r2:
                comps.render_section_header("Distribution Metrics", "Signal Statistics", icon="database", accent="rose")
                signal_stats = {
                    "Metric": ["Mean Signal", "Median Signal", "Min Signal", "Max Signal", "Std Dev"],
                    "Value": [
                        f"{ts_df['Avg_Signal'].mean():.2f}",
                        f"{ts_df['Avg_Signal'].median():.2f}",
                        f"{ts_df['Avg_Signal'].min():.2f}",
                        f"{ts_df['Avg_Signal'].max():.2f}",
                        f"{ts_df['Avg_Signal'].std():.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(signal_stats), width="stretch", hide_index=True)
        
        with tab4:
            st.markdown(f"##### Daily Time Series Data ({len(ts_df)} trading days)")
            
            # Include regime data in display
            display_ts = ts_df[['Date', 'Total_Analyzed', 'Oversold', 'Overbought', 
                               'Buy_Signals', 'Sell_Signals', 'Avg_Signal', 
                               'Regime_Bull', 'Regime_Bear', 'Change_Points']].copy()
            display_ts['Date'] = display_ts['Date'].dt.strftime('%Y-%m-%d')
            display_ts['Avg_Signal'] = display_ts['Avg_Signal'].round(2)
            display_ts.columns = ['Date', 'Stocks', 'Oversold', 'Overbought', 
                                 'Buy Sig', 'Sell Sig', 'Avg Sig', 'Bull Regime', 'Bear Regime', 'Changes']
            
            st.dataframe(display_ts, width="stretch", hide_index=True, height=500)
            
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = ts_df.to_csv(index=False).encode('utf-8')
            actual_start_str = ts_df['Date'].min().strftime('%Y%m%d')
            actual_end_str = ts_df['Date'].max().strftime('%Y%m%d')
            st.download_button(
                label="📥 Download Time Series Data (CSV)",
                data=csv_data,
                file_name=f"nirnay_market_timeseries_{universe_title.replace(' ', '_')}_{actual_start_str}_{actual_end_str}.csv",
                mime="text/csv"
            )


def run_etf_timeseries_mode(length, roc_len, regime_sensitivity, base_weight, start_date, end_date, run_clicked):
    """ETF Time Series Analysis: Track overbought/oversold signals over time for fixed ETF universe"""
    
    # Validate dates
    if start_date is None or end_date is None:
        st.error("Please select both start and end dates.")
        return
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    
    if end_date > datetime.date.today():
        st.error("End date cannot be in the future.")
        return
    
    # Calculate date range
    date_range_days = (end_date - start_date).days

    if not run_clicked:
        # Top spacing
        comps.section_gap()
        comps.section_gap()

        # Display info
        comps.render_section_header(
            "Time Series Analysis — ETF Universe",
            f"Track overbought/oversold signal distribution across {len(SCREENER_SYMBOLS)} ETFs over time · {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')} ({date_range_days} days)",
            icon="trending",
            accent="amber"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Analysis Overview Section
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            comps.render_metric_card("Period", f"{date_range_days}", "Trading Days", "info")
        with col2:
            comps.render_metric_card("Signal Engines", "2", "MSF + MMR", "amber")
        with col3:
            comps.render_metric_card("Output Metrics", "8", "Signal + Regime + Zone", "warning")
        with col4:
            comps.render_metric_card("Universe", f"{len(SCREENER_SYMBOLS)}", "Curated ETFs", "neutral")

        comps.section_gap()
        comps.section_gap()

        # Methodology Section
        comps.render_section_header(
            "Analysis Framework",
            "Rolling window signal tracking with overbought/oversold distribution and macro correlation",
            icon="layers",
            accent="emerald"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #06B6D4; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Market Structure</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    Internal price structure via momentum ROC, efficiency ratio, and microstructure decomposition.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(212, 168, 83, 0.08); border: 1px solid rgba(212, 168, 83, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #D4A853; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Macro Regression</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    Macro correlation tracking bond yields, currencies, and commodity flows for regime shifts.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div style="padding: 1rem; background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px;">
                <div style="font-weight: 600; color: #A855F7; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Regime Intelligence</div>
                <div style="font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.6;">
                    HMM state evolution, volatility regime distribution, and change point timeline.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if run_clicked:
        progress_slot = st.empty()

        # Fetch macro data
        theme.progress_bar(progress_slot, 5, "Macro Data", "Fetching global macro factors...")

        days_back = 100 + date_range_days + (datetime.date.today() - end_date).days
        macro_df = fetch_macro_data(days_back=days_back)

        # Process each ETF
        theme.progress_bar(progress_slot, 10, "Downloading ETFs", f"Processing {len(SCREENER_SYMBOLS)} ETFs...")

        processed_data = {}
        total = len(SCREENER_SYMBOLS)

        for i, symbol in enumerate(SCREENER_SYMBOLS):
            pct = 10 + int(35 * (i + 1) / total)
            theme.progress_bar(progress_slot, pct, f"Processing {get_display_name(symbol)}", f"{i+1}/{total} ETFs")
            
            df = fetch_ticker_data(symbol, macro_df, days_back=days_back)
            
            if df is not None and len(df) > length + 5:
                try:
                    df, _ = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    processed_data[symbol] = df
                except Exception:
                    pass
        
        if not processed_data:
            st.error("Failed to process ETF data.")
            progress_slot.empty()
            return

        # Generate list of trading days
        theme.progress_bar(progress_slot, 45, "Trading Calendar", "Identifying trading days in range...")

        sample_ticker = list(processed_data.keys())[0]
        sample_df = processed_data[sample_ticker]

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        trading_days = sample_df.index[(sample_df.index >= start_ts) & (sample_df.index <= end_ts)].tolist()

        if len(trading_days) == 0:
            st.error("No trading days found in the selected date range.")
            progress_slot.empty()
            return

        # Check if requested end date data is available
        actual_last_date = trading_days[-1].date() if trading_days else None
        is_today_included = actual_last_date == datetime.date.today() if actual_last_date else False

        if end_date == datetime.date.today():
            if is_today_included:
                st.toast(f"Live Data Included - {actual_last_date.strftime('%d %b %Y')}")
            else:
                st.toast(f"Data through {actual_last_date.strftime('%d %b %Y')}")
        elif actual_last_date and actual_last_date < end_date:
            st.toast(f"Data through {actual_last_date.strftime('%d %b %Y')}")

        st.toast(f"Found {len(trading_days)} trading days")

        # Analyze each trading day
        theme.progress_bar(progress_slot, 50, "Timeline Analysis", f"Processing {len(trading_days)} trading days...")

        timeseries_results = []

        for day_idx, trading_day in enumerate(trading_days):
            pct = 50 + int(45 * (day_idx + 1) / len(trading_days))
            theme.progress_bar(progress_slot, pct, "Timeline Analysis", f"{day_idx+1}/{len(trading_days)} days")
            
            day_stats = {
                "Date": trading_day.date(),
                "Oversold": 0,
                "Overbought": 0,
                "Neutral": 0,
                "Buy_Signals": 0,
                "Sell_Signals": 0,
                "Total_Analyzed": 0,
                "Avg_Signal": 0,
                "Signal_Sum": 0,
                "Bull_Div": 0,
                "Bear_Div": 0,
                # Regime Intelligence stats
                "Regime_Bull": 0,
                "Regime_Bear": 0,
                "Regime_Neutral": 0,
                "Regime_Transition": 0,
                "Vol_High": 0,
                "Vol_Low": 0,
                "Change_Points": 0
            }
            
            for symbol, df in processed_data.items():
                try:
                    if trading_day not in df.index:
                        continue
                    
                    row = df.loc[trading_day]
                    
                    day_stats["Total_Analyzed"] += 1
                    day_stats["Signal_Sum"] += row['Unified_Osc']
                    
                    if row['Condition'] == 'Oversold':
                        day_stats["Oversold"] += 1
                    elif row['Condition'] == 'Overbought':
                        day_stats["Overbought"] += 1
                    else:
                        day_stats["Neutral"] += 1
                    
                    if row['Buy_Signal']:
                        day_stats["Buy_Signals"] += 1
                    if row['Sell_Signal']:
                        day_stats["Sell_Signals"] += 1
                    if row['Bullish_Div']:
                        day_stats["Bull_Div"] += 1
                    if row['Bearish_Div']:
                        day_stats["Bear_Div"] += 1
                    
                    # Regime Intelligence stats
                    regime = row['Regime']
                    if 'BULL' in regime:
                        day_stats["Regime_Bull"] += 1
                    elif 'BEAR' in regime:
                        day_stats["Regime_Bear"] += 1
                    elif regime == 'TRANSITION':
                        day_stats["Regime_Transition"] += 1
                    else:
                        day_stats["Regime_Neutral"] += 1
                    
                    vol_regime = row['Vol_Regime']
                    if vol_regime in ['HIGH', 'EXTREME']:
                        day_stats["Vol_High"] += 1
                    elif vol_regime == 'LOW':
                        day_stats["Vol_Low"] += 1
                    
                    if row['Change_Point']:
                        day_stats["Change_Points"] += 1
                        
                except Exception:
                    pass
            
            if day_stats["Total_Analyzed"] > 0:
                day_stats["Avg_Signal"] = day_stats["Signal_Sum"] / day_stats["Total_Analyzed"]
                day_stats["Oversold_Pct"] = (day_stats["Oversold"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Overbought_Pct"] = (day_stats["Overbought"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Neutral_Pct"] = (day_stats["Neutral"] / day_stats["Total_Analyzed"]) * 100
                # Regime percentages
                day_stats["Regime_Bull_Pct"] = (day_stats["Regime_Bull"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Regime_Bear_Pct"] = (day_stats["Regime_Bear"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Vol_High_Pct"] = (day_stats["Vol_High"] / day_stats["Total_Analyzed"]) * 100
            else:
                day_stats["Oversold_Pct"] = 0
                day_stats["Overbought_Pct"] = 0
                day_stats["Neutral_Pct"] = 0
                day_stats["Regime_Bull_Pct"] = 0
                day_stats["Regime_Bear_Pct"] = 0
                day_stats["Vol_High_Pct"] = 0
            
            timeseries_results.append(day_stats)
        
        progress_slot.empty()
        
        if not timeseries_results:
            st.warning("No data could be analyzed for the selected period.")
            return
        
        ts_df = pd.DataFrame(timeseries_results)
        ts_df['Date'] = pd.to_datetime(ts_df['Date'])
        ts_df = ts_df.sort_values('Date')
        
        # Show actual analyzed date range
        actual_start = ts_df['Date'].min().strftime('%d %b %Y')
        actual_end = ts_df['Date'].max().strftime('%d %b %Y')
        st.toast(f"ETF Time Series Complete! {len(ts_df)} days")
        
        # Summary metrics
        st.markdown("<br>", unsafe_allow_html=True)
        avg_oversold = ts_df['Oversold_Pct'].mean()
        avg_overbought = ts_df['Overbought_Pct'].mean()
        total_buys = ts_df['Buy_Signals'].sum()
        total_sells = ts_df['Sell_Signals'].sum()
        avg_signal = ts_df['Avg_Signal'].mean()
        regime = "BULLISH" if avg_signal < -1 else "BEARISH" if avg_signal > 1 else "NEUTRAL"
        regime_color = "success" if avg_signal < -1 else "danger" if avg_signal > 1 else "neutral"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            comps.render_metric_card("Avg Oversold", f"{avg_oversold:.1f}%", "Daily Average", "success")
        with c2:
            comps.render_metric_card("Avg Overbought", f"{avg_overbought:.1f}%", "Daily Average", "danger")
        with c3:
            comps.render_metric_card("Total Buys", f"{total_buys:,}", "Over Period", "primary")
        with c4:
            comps.render_metric_card("Total Sells", f"{total_sells:,}", "Over Period", "warning")
        with c5:
            comps.render_metric_card("Period Regime", regime, f"Avg: {avg_signal:.2f}", regime_color)
        with c6:
            comps.render_metric_card("Trading Days", f"{len(ts_df)}", "Analyzed", "info")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Tabs for different views
        tab3, tab1, tab2, tab4 = st.tabs(["Regime Analysis", "Signal Dashboard", "Transaction Dynamics", "Data Terminal"])
        
        with tab1:
            comps.render_section_header("Extreme Signal Trends", "Overbought / Oversold Distribution Over Time", icon="activity", accent="cyan")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Shows the percentage of ETFs in each zone daily</p>', unsafe_allow_html=True)
            
            fig_zones = go.Figure()
            
            fig_zones.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Oversold_Pct'],
                mode='lines', name='Oversold %',
                fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
                line=dict(color=COLOR_GREEN, width=2)
            ))
            
            fig_zones.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Overbought_Pct'],
                mode='lines', name='Overbought %',
                fill='tozeroy', fillcolor='rgba(251,113,133,0.12)',
                line=dict(color=COLOR_RED, width=2)
            ))
            
            ymax = max(ts_df['Oversold_Pct'].max(), ts_df['Overbought_Pct'].max()) * 1.15
            fig_zones.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
            style_axes(fig_zones, y_title="% of ETFs", y_range=[0, ymax])
            st.plotly_chart(fig_zones, width='stretch', key="market_zones")
            
            st.markdown("<br>", unsafe_allow_html=True)
            comps.render_section_header("Signal Volume Trends", "Raw Counts Over Time", icon="bar-chart", accent="info")
            
            fig_counts = go.Figure()
            
            fig_counts.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Oversold'],
                name='Oversold', 
                marker=dict(color=COLOR_GREEN, line=dict(color=COLOR_GREEN, width=1))
            ))
            
            fig_counts.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Overbought'],
                name='Overbought', 
                marker=dict(color=COLOR_RED, line=dict(color=COLOR_RED, width=1))
            ))
            
            fig_counts.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM), barmode='group')
            style_axes(fig_counts, y_title="ETF Count")
            st.plotly_chart(fig_counts, width='stretch', key="market_counts")
        
        with tab2:
            comps.render_section_header("Transaction Signal Trends", "Buy / Sell Signal Counts Over Time", icon="zap", accent="emerald")
            
            fig_signals = go.Figure()
            
            fig_signals.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Buy_Signals'],
                mode='lines+markers', name='Buy Signals',
                line=dict(color=COLOR_GREEN, width=2),
                marker=dict(size=6, color=COLOR_GREEN)
            ))
            
            fig_signals.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Sell_Signals'],
                mode='lines+markers', name='Sell Signals',
                line=dict(color=COLOR_RED, width=2),
                marker=dict(size=6, color=COLOR_RED)
            ))
            
            fig_signals.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
            style_axes(fig_signals, y_title="Signal Count")
            st.plotly_chart(fig_signals, width='stretch', key="market_signals")
            
            st.markdown("<br>", unsafe_allow_html=True)
            comps.render_section_header("Divergence Persistence", "Divergence Signals Over Time", icon="trending-up", accent="amber")
            
            fig_div = go.Figure()
            
            fig_div.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Bull_Div'],
                name='Bullish Divergence', 
                marker=dict(color=COLOR_GOLD, line=dict(color=COLOR_GOLD, width=1))
            ))
            
            fig_div.add_trace(go.Bar(
                x=ts_df['Date'], y=-ts_df['Bear_Div'],
                name='Bearish Divergence', 
                marker=dict(color=COLOR_CYAN, line=dict(color=COLOR_CYAN, width=1))
            ))
            
            fig_div.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM), barmode='relative')
            style_axes(fig_div, y_title="Divergence Count")
            st.plotly_chart(fig_div, width='stretch', key="ts_div")
        
        with tab3:
            # ORIGINAL: Average Signal Value Over Time
            comps.render_section_header("Aggregate Signal Momentum", "Average Signal Value Over Time", icon="activity", accent="rose")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Negative = Bullish Bias | Positive = Bearish Bias</p>', unsafe_allow_html=True)
            
            fig_avg = go.Figure()
            
            colors = [COLOR_GREEN if v < -2 else COLOR_RED if v > 2 else COLOR_MUTED for v in ts_df['Avg_Signal']]
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'].clip(lower=0),
                fill='tozeroy', fillcolor='rgba(251,113,133,0.05)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'].clip(upper=0),
                fill='tozeroy', fillcolor='rgba(52,211,153,0.05)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'],
                mode='lines+markers', name='Avg Signal',
                line=dict(color=COLOR_GOLD, width=2),
                marker=dict(size=6, color=colors)
            ))
            
            fig_avg.add_hline(y=2, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
            fig_avg.add_hline(y=-2, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
            fig_avg.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
            
            fig_avg.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
            style_axes(fig_avg, y_title="Avg Signal", y_range=[-8, 8])
            st.plotly_chart(fig_avg, width='stretch', key="ts_market_avg")

            st.markdown("<br>", unsafe_allow_html=True)

            # NEW: HMM Regime Distribution Over Time
            comps.render_section_header("HMM Regime Distribution Over Time", "Percentage of ETFs in each HMM regime daily", icon="activity", accent="cyan")
            
            # Regime trend chart
            fig_regime = go.Figure()
            
            fig_regime.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Regime_Bull_Pct'],
                mode='lines', name='Bull Regime %',
                fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
                line=dict(color=COLOR_GREEN, width=2)
            ))
            
            fig_regime.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Regime_Bear_Pct'],
                mode='lines', name='Bear Regime %',
                fill='tozeroy', fillcolor='rgba(251,113,133,0.12)',
                line=dict(color=COLOR_RED, width=2)
            ))
            
            fig_regime.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
            style_axes(fig_regime, y_title="% of ETFs", y_range=[0, 100])
            st.plotly_chart(fig_regime, width='stretch', key="market_regime")
            
            st.markdown("<br>", unsafe_allow_html=True)
            comps.render_section_header("Volatility Dynamics", "Volatility Regime & Change Points Over Time", icon="shield", accent="amber")
            
            # Volatility regime chart
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Vol_High_Pct'],
                mode='lines+markers', name='High Vol %',
                line=dict(color=COLOR_AMBER, width=2),
                marker=dict(size=5)
            ))
            
            fig_vol.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Change_Points'],
                name='Change Points',
                marker=dict(color=COLOR_PURPLE, opacity=0.7)
            ))
            
            fig_vol.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL))
            style_axes(fig_vol, y_title="Count / %")
            st.plotly_chart(fig_vol, width='stretch', key="ts_vol")
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                comps.render_section_header("State Transition Metrics", "HMM Regime Statistics", icon="bar-chart", accent="emerald")
                avg_bull = ts_df['Regime_Bull_Pct'].mean()
                avg_bear = ts_df['Regime_Bear_Pct'].mean()
                total_changes = ts_df['Change_Points'].sum()
                
                regime_stats = {
                    "Metric": ["Avg Bull Regime %", "Avg Bear Regime %", "Total Change Points", "Avg High Vol %"],
                    "Value": [f"{avg_bull:.1f}%", f"{avg_bear:.1f}%", f"{int(total_changes)}", f"{ts_df['Vol_High_Pct'].mean():.1f}%"]
                }
                st.dataframe(pd.DataFrame(regime_stats), width="stretch", hide_index=True)
            
            with col_r2:
                comps.render_section_header("Distribution Metrics", "Signal Statistics", icon="database", accent="rose")
                signal_stats = {
                    "Metric": ["Mean Signal", "Median Signal", "Min Signal", "Max Signal", "Std Dev"],
                    "Value": [
                        f"{ts_df['Avg_Signal'].mean():.2f}",
                        f"{ts_df['Avg_Signal'].median():.2f}",
                        f"{ts_df['Avg_Signal'].min():.2f}",
                        f"{ts_df['Avg_Signal'].max():.2f}",
                        f"{ts_df['Avg_Signal'].std():.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(signal_stats), width="stretch", hide_index=True)
        
        with tab4:
            comps.render_section_header("Analytical Data", f"Daily ETF Time Series ({len(ts_df)} days)", icon="list", accent="cyan")
            
            # Include regime data in display
            display_ts = ts_df[['Date', 'Total_Analyzed', 'Oversold', 'Overbought', 
                               'Buy_Signals', 'Sell_Signals', 'Avg_Signal', 
                               'Regime_Bull', 'Regime_Bear', 'Change_Points']].copy()
            display_ts['Date'] = display_ts['Date'].dt.strftime('%Y-%m-%d')
            display_ts['Avg_Signal'] = display_ts['Avg_Signal'].round(2)
            display_ts.columns = ['Date', 'ETFs', 'Oversold', 'Overbought', 
                                 'Buy Sig', 'Sell Sig', 'Avg Sig', 'Bull Regime', 'Bear Regime', 'Changes']
            
            st.dataframe(display_ts, width="stretch", hide_index=True, height=500)
            
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = ts_df.to_csv(index=False).encode('utf-8')
            actual_start_str = ts_df['Date'].min().strftime('%Y%m%d')
            actual_end_str = ts_df['Date'].max().strftime('%Y%m%d')
            st.download_button(
                label="Download Time Series Data (CSV)",
                data=csv_data,
                file_name=f"nirnay_etf_timeseries_{actual_start_str}_{actual_end_str}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
