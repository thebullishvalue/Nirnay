"""
NIRNAY (निर्णय) - Decisive Market Intelligence

Unified Quantitative Market Intelligence System
A Pragyam Product Family Member

Integrates:
- Signal Generation (MSF + MMR from UMA)
- Regime Intelligence (HMM, Kalman, GARCH, CUSUM from AVASTHA)
- Adaptive Thresholds (Percentile-based, NO fixed values)
- Multi-mode Analysis (Dashboard, Chart, Screener, Regime, Timeseries)

Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Local imports
from nirnay_core import (
    NirnayEngine, NirnayResult, MarketRegime, SignalType,
    VolatilityRegime, run_batch_analysis, MathUtils
)
from data_engine import (
    ETF_UNIVERSE, ETF_NAMES, MACRO_SYMBOLS, INDEX_LIST, UNIVERSE_OPTIONS,
    get_display_name, get_universe_symbols, fetch_macro_data,
    fetch_symbol_data, fetch_batch_data, fetch_symbol_with_macro,
    get_macro_columns
)
from charts import (
    COLORS, create_price_chart, create_oscillator_chart,
    create_signal_gauge, create_regime_gauge, create_hmm_probability_chart,
    create_component_radar, create_heatmap, create_distribution_chart,
    create_regime_distribution_chart, create_time_series_chart, create_ranking_chart
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NIRNAY | निर्णय",
    layout="wide",
    initial_sidebar_state="expanded"
)

VERSION = "v1.0.0 - Unified Intelligence"

# ══════════════════════════════════════════════════════════════════════════════
# PRAGYAM DESIGN SYSTEM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --neutral: #888888;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 95%; padding-left: 2rem; padding-right: 2rem; }
    
    [data-testid="collapsedControl"] {
        display: flex !important; visibility: visible !important; opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important; padding: 10px !important; margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important; position: fixed !important;
        top: 14px !important; left: 14px !important;
        width: 40px !important; height: 40px !important;
        align-items: center !important; justify-content: center !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important;
    }
    
    [data-testid="collapsedControl"] svg { stroke: var(--primary-color) !important; }
    
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color);
        position: relative; overflow: hidden; margin-top: 1rem;
    }
    
    .premium-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; }
    .premium-header .product-badge { display: inline-block; background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    
    .metric-card {
        background-color: var(--bg-card); padding: 1.25rem; border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem; transition: all 0.3s; position: relative; overflow: hidden;
    }
    
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    
    .signal-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; }
    .signal-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .signal-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .signal-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .warning-box { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 0; color: var(--warning-amber); font-size: 0.85rem; }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s; text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); }
    
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border-color), transparent); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER AND SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown(f"""
    <div class='premium-header'>
        <div class='product-badge'>Pragyam Product Family</div>
        <h1>◈ NIRNAY <span style='font-size: 1.2rem; color: var(--text-muted);'>निर्णय</span></h1>
        <div class='tagline'>Unified Signal + Regime Intelligence | {VERSION}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: var(--primary-color); margin: 0;'>◈ NIRNAY</h2>
            <p style='color: var(--text-muted); font-size: 0.8rem; margin: 0.25rem 0;'>निर्णय | Decision Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analysis Mode
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["📊 Dashboard", "📈 Chart Analysis", "🔍 Screener", "🎯 Regime Detection"],
            index=0
        )
        
        st.markdown("---")
        
        # Common parameters
        st.markdown("##### Signal Parameters")
        msf_length = st.slider("MSF Length", 10, 50, 20, help="Lookback period for MSF calculation")
        roc_len = st.slider("ROC Length", 5, 30, 14, help="Rate of change period")
        
        st.markdown("##### Regime Parameters")
        regime_sensitivity = st.slider("Regime Sensitivity", 0.5, 2.0, 1.0, 0.1)
        base_weight = st.slider("MSF Weight", 0.3, 0.8, 0.6, 0.05, help="Base weight for MSF vs MMR")
        
        st.markdown("---")
        
        # Mode-specific options
        if "Screener" in analysis_mode:
            st.markdown("##### Universe Selection")
            universe = st.selectbox("Universe", UNIVERSE_OPTIONS)
            
            index_name = None
            if universe == "Index Constituents":
                index_name = st.selectbox("Index", INDEX_LIST)
            
            return {
                'mode': 'screener',
                'msf_length': msf_length,
                'roc_len': roc_len,
                'regime_sensitivity': regime_sensitivity,
                'base_weight': base_weight,
                'universe': universe,
                'index_name': index_name
            }
        
        elif "Chart" in analysis_mode:
            st.markdown("##### Symbol Selection")
            
            symbol_input = st.selectbox(
                "Select ETF",
                ETF_UNIVERSE,
                format_func=get_display_name
            )
            
            custom_symbol = st.text_input("Or enter symbol", placeholder="RELIANCE.NS")
            symbol = custom_symbol if custom_symbol else symbol_input
            
            include_macro = st.checkbox("Include Macro Analysis (MMR)", value=True)
            
            return {
                'mode': 'chart',
                'symbol': symbol,
                'include_macro': include_macro,
                'msf_length': msf_length,
                'roc_len': roc_len,
                'regime_sensitivity': regime_sensitivity,
                'base_weight': base_weight
            }
        
        elif "Regime" in analysis_mode:
            st.markdown("##### Universe Selection")
            universe = st.selectbox("Universe", UNIVERSE_OPTIONS)
            
            index_name = None
            if universe == "Index Constituents":
                index_name = st.selectbox("Index", INDEX_LIST)
            
            return {
                'mode': 'regime',
                'msf_length': msf_length,
                'roc_len': roc_len,
                'regime_sensitivity': regime_sensitivity,
                'base_weight': base_weight,
                'universe': universe,
                'index_name': index_name
            }
        
        else:  # Dashboard
            return {
                'mode': 'dashboard',
                'msf_length': msf_length,
                'roc_len': roc_len,
                'regime_sensitivity': regime_sensitivity,
                'base_weight': base_weight
            }


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_signal_class(signal: SignalType) -> str:
    if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
        return "buy"
    elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
        return "sell"
    return "neutral"


def display_nirnay_result(result: NirnayResult, df: pd.DataFrame):
    """Display comprehensive NIRNAY analysis result"""
    
    signal_class = get_signal_class(result.signal)
    
    # Top row: Signal, Action, Confidence
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        card_class = "success" if signal_class == "buy" else "danger" if signal_class == "sell" else "neutral"
        st.markdown(f"""
        <div class='metric-card {card_class}'>
            <h4>Signal</h4>
            <h2>{result.signal.value}</h2>
            <div class='sub-metric'>Strength: {result.signal_strength:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Recommended Action</h4>
            <h2 style='font-size: 1.3rem;'>{result.action}</h2>
            <div class='sub-metric'>Position Factor: {result.position_size_factor:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        conf_class = "success" if result.regime.confidence >= 0.7 else "warning" if result.regime.confidence >= 0.5 else "neutral"
        st.markdown(f"""
        <div class='metric-card {conf_class}'>
            <h4>Confidence</h4>
            <h2>{result.regime.confidence:.0%}</h2>
            <div class='sub-metric'>Bayesian estimate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row: Regime info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        regime_class = "success" if "BULL" in result.regime.regime.value else "danger" if "BEAR" in result.regime.regime.value or "CRISIS" in result.regime.regime.value else "neutral"
        st.markdown(f"""
        <div class='metric-card {regime_class}'>
            <h4>Market Regime</h4>
            <h2 style='font-size: 1.2rem;'>{result.regime.regime.value}</h2>
            <div class='sub-metric'>HMM State</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        vol_class = "danger" if result.regime.volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME] else "success" if result.regime.volatility_regime == VolatilityRegime.LOW else "neutral"
        st.markdown(f"""
        <div class='metric-card {vol_class}'>
            <h4>Volatility</h4>
            <h2>{result.regime.volatility_regime.value}</h2>
            <div class='sub-metric'>Multiplier: {result.regime.volatility_multiplier:.2f}x</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card primary'>
            <h4>Persistence</h4>
            <h2>{result.regime.regime_persistence}</h2>
            <div class='sub-metric'>Periods in state</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pct_class = "success" if result.thresholds.signal_percentile >= 0.7 else "danger" if result.thresholds.signal_percentile <= 0.3 else "neutral"
        st.markdown(f"""
        <div class='metric-card {pct_class}'>
            <h4>Signal Percentile</h4>
            <h2>{result.thresholds.signal_percentile:.0%}</h2>
            <div class='sub-metric'>vs History</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warnings
    if result.warnings:
        for warning in result.warnings:
            st.markdown(f"<div class='warning-box'>⚠️ {warning}</div>", unsafe_allow_html=True)
    
    if result.regime.change_point_detected:
        st.markdown("""
        <div class='warning-box' style='background: rgba(239,68,68,0.15); border-color: rgba(239,68,68,0.5);'>
            ⚡ <strong>STRUCTURAL BREAK DETECTED</strong> - Market conditions have fundamentally shifted
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Tabbed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "🎯 Signal Components", "🔮 Regime Analysis", "📊 Macro Drivers"])
    
    with tab1:
        # Price chart
        st.markdown("##### Price Chart")
        st.plotly_chart(create_price_chart(df, result.symbol), use_container_width=True)
        
        # Build oscillator data
        engine = NirnayEngine()
        msf_calc = engine.msf_calc
        msf_series, micro, momentum, flow = msf_calc.calculate(df)
        
        st.markdown("##### Signal Oscillators")
        st.plotly_chart(create_oscillator_chart(df, msf_series, None, msf_series), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### Component Radar")
            components = {
                'MSF': result.components.msf,
                'Momentum': result.components.momentum,
                'Micro': result.components.micro,
                'Flow': result.components.flow
            }
            if result.components.mmr != 0:
                components['MMR'] = result.components.mmr
            st.plotly_chart(create_component_radar(components), use_container_width=True)
        
        with col2:
            st.markdown("##### Signal Gauge")
            st.plotly_chart(create_signal_gauge(result.signal_strength, "Unified Signal"), use_container_width=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("##### Component Details")
        
        cols = st.columns(5)
        comp_data = [
            ("MSF", result.components.msf),
            ("MMR", result.components.mmr),
            ("Momentum", result.components.momentum),
            ("Micro", result.components.micro),
            ("Flow", result.components.flow)
        ]
        
        for i, (name, val) in enumerate(comp_data):
            with cols[i]:
                c_class = "success" if val >= 0.3 else "danger" if val <= -0.3 else "neutral"
                st.markdown(f"""
                <div class='metric-card {c_class}'>
                    <h4>{name}</h4>
                    <h2>{val:+.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### HMM State Probabilities")
            st.plotly_chart(create_hmm_probability_chart(result.regime.hmm_probabilities), use_container_width=True)
        
        with col2:
            st.markdown("##### Regime Strength")
            st.plotly_chart(create_regime_gauge(result.signal_strength, result.regime.confidence), use_container_width=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("##### Adaptive Thresholds (Percentile-Based)")
        st.markdown('<p style="color: var(--text-muted);">Unlike fixed thresholds, these adapt to market conditions</p>', unsafe_allow_html=True)
        
        cols = st.columns(4)
        thresh_data = [
            ("Strong Buy", result.thresholds.strong_buy_threshold, "10th percentile"),
            ("Oversold", result.thresholds.oversold_threshold, "20th percentile"),
            ("Overbought", result.thresholds.overbought_threshold, "80th percentile"),
            ("Strong Sell", result.thresholds.strong_sell_threshold, "90th percentile")
        ]
        
        for i, (name, val, desc) in enumerate(thresh_data):
            with cols[i]:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>{name}</h4>
                    <h2>{val:+.3f}</h2>
                    <div class='sub-metric'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        if result.macro_drivers:
            st.markdown("##### Top Macro Correlations")
            drivers_df = pd.DataFrame(result.macro_drivers)
            st.dataframe(drivers_df, use_container_width=True, hide_index=True)
        else:
            st.info("Macro analysis not available. Enable 'Include Macro Analysis' in sidebar.")


def display_screener_results(results: list):
    """Display screener results"""
    if not results:
        st.warning("No results to display")
        return
    
    df = pd.DataFrame(results)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    buy_count = len(df[df['signal'].str.contains('BUY', na=False)])
    sell_count = len(df[df['signal'].str.contains('SELL', na=False)])
    avg_strength = df['signal_strength'].mean()
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Total Symbols</h4>
            <h2>{total}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card success'>
            <h4>Buy Signals</h4>
            <h2>{buy_count}</h2>
            <div class='sub-metric'>{buy_count/total*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card danger'>
            <h4>Sell Signals</h4>
            <h2>{sell_count}</h2>
            <div class='sub-metric'>{sell_count/total*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_class = "success" if avg_strength > 0.1 else "danger" if avg_strength < -0.1 else "neutral"
        st.markdown(f"""
        <div class='metric-card {avg_class}'>
            <h4>Avg Signal</h4>
            <h2>{avg_strength:+.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🗺️ Heatmap", "📊 Distribution", "📋 Data"])
    
    with tab1:
        st.plotly_chart(create_heatmap(df), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("##### Signal Distribution")
            st.plotly_chart(create_distribution_chart(df), use_container_width=True)
        with col2:
            st.markdown("##### Top & Bottom Rankings")
            st.plotly_chart(create_ranking_chart(df, 8), use_container_width=True)
    
    with tab3:
        # Clean up display
        display_df = df[['symbol', 'signal', 'signal_strength', 'msf', 'regime', 'confidence', 'action']].copy()
        display_df['symbol'] = display_df['symbol'].str.replace('.NS', '')
        display_df = display_df.sort_values('signal_strength', ascending=False)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def run_dashboard_mode(params):
    """Run dashboard mode showing ETF overview"""
    
    st.markdown("##### Market Dashboard - ETF Universe Overview")
    
    if st.button("🔄 REFRESH DATA", use_container_width=True):
        st.session_state.dashboard_data = None
    
    # Fetch and analyze
    with st.spinner("Fetching data and analyzing..."):
        progress = st.progress(0, text="Loading ETF data...")
        
        def update_progress(pct):
            progress.progress(pct, text=f"Processing... {pct*100:.0f}%")
        
        data_dict = fetch_batch_data(ETF_UNIVERSE[:15], days_back=100, progress_callback=update_progress)
        
        if not data_dict:
            progress.empty()
            st.error("Failed to fetch data")
            return
        
        progress.progress(0.7, text="Running NIRNAY analysis...")
        
        results = run_batch_analysis(
            data_dict,
            macro_df=None,
            msf_length=params['msf_length'],
            roc_len=params['roc_len']
        )
        
        progress.empty()
    
    display_screener_results(results)


def run_chart_mode(params):
    """Run chart analysis mode for single symbol"""
    
    symbol = params['symbol']
    
    st.markdown(f"##### Chart Analysis: {get_display_name(symbol)}")
    
    if st.button("🔄 ANALYZE", use_container_width=True):
        st.session_state.chart_result = None
    
    with st.spinner(f"Analyzing {symbol}..."):
        progress = st.progress(0, text="Fetching data...")
        
        # Fetch macro if enabled
        macro_df = None
        macro_cols = []
        if params.get('include_macro'):
            progress.progress(0.2, text="Fetching macro data...")
            macro_df = fetch_macro_data(days_back=100)
            macro_cols = list(macro_df.columns) if macro_df is not None else []
        
        progress.progress(0.4, text="Fetching symbol data...")
        df = fetch_symbol_with_macro(symbol, macro_df, days_back=100)
        
        if df is None or df.empty:
            progress.empty()
            st.error(f"Failed to fetch data for {symbol}")
            return
        
        progress.progress(0.7, text="Running NIRNAY analysis...")
        
        engine = NirnayEngine(
            msf_length=params['msf_length'],
            roc_len=params['roc_len'],
            regime_sensitivity=params['regime_sensitivity'],
            base_weight=params['base_weight']
        )
        
        result = engine.analyze(df, symbol, macro_cols)
        
        progress.empty()
    
    display_nirnay_result(result, df)


def run_screener_mode(params):
    """Run screener mode for universe analysis"""
    
    st.markdown(f"##### Market Screener - {params['universe']}")
    
    if st.button("🔄 RUN SCREENER", use_container_width=True):
        st.session_state.screener_results = None
    
    with st.spinner("Running screener..."):
        progress = st.progress(0, text="Loading universe...")
        
        symbols, msg = get_universe_symbols(params['universe'], params.get('index_name'))
        st.info(msg)
        
        progress.progress(0.2, text="Fetching data...")
        
        def update_progress(pct):
            progress.progress(0.2 + pct * 0.5, text=f"Fetching... {pct*100:.0f}%")
        
        data_dict = fetch_batch_data(symbols[:50], days_back=100, progress_callback=update_progress)
        
        if not data_dict:
            progress.empty()
            st.error("Failed to fetch data")
            return
        
        progress.progress(0.75, text="Running NIRNAY analysis...")
        
        results = run_batch_analysis(
            data_dict,
            macro_df=None,
            msf_length=params['msf_length'],
            roc_len=params['roc_len']
        )
        
        progress.empty()
    
    display_screener_results(results)


def run_regime_mode(params):
    """Run dedicated regime detection mode"""
    
    st.markdown(f"##### Regime Detection - {params['universe']}")
    
    if st.button("🔄 DETECT REGIME", use_container_width=True):
        st.session_state.regime_results = None
    
    with st.spinner("Analyzing market regime..."):
        progress = st.progress(0, text="Loading universe...")
        
        symbols, msg = get_universe_symbols(params['universe'], params.get('index_name'))
        
        progress.progress(0.2, text="Fetching data...")
        data_dict = fetch_batch_data(symbols[:30], days_back=100)
        
        if not data_dict:
            progress.empty()
            st.error("Failed to fetch data")
            return
        
        progress.progress(0.6, text="Analyzing regime...")
        
        results = run_batch_analysis(data_dict, msf_length=params['msf_length'], roc_len=params['roc_len'])
        
        progress.empty()
    
    if not results:
        st.warning("No results")
        return
    
    df = pd.DataFrame(results)
    
    # Aggregate regime stats
    regime_counts = df['regime'].value_counts()
    dominant_regime = regime_counts.index[0] if len(regime_counts) > 0 else "UNKNOWN"
    
    bull_pct = len(df[df['regime'].str.contains('BULL', na=False)]) / len(df) * 100
    bear_pct = len(df[df['regime'].str.contains('BEAR|CRISIS', na=False)]) / len(df) * 100
    avg_confidence = df['confidence'].mean()
    
    # Display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        r_class = "success" if "BULL" in dominant_regime else "danger" if "BEAR" in dominant_regime else "neutral"
        st.markdown(f"""
        <div class='metric-card {r_class}'>
            <h4>Dominant Regime</h4>
            <h2 style='font-size: 1.3rem;'>{dominant_regime}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card success'>
            <h4>Bullish %</h4>
            <h2>{bull_pct:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card danger'>
            <h4>Bearish %</h4>
            <h2>{bear_pct:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        c_class = "success" if avg_confidence >= 0.6 else "warning"
        st.markdown(f"""
        <div class='metric-card {c_class}'>
            <h4>Avg Confidence</h4>
            <h2>{avg_confidence:.0%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### Regime Distribution")
        results_list = df.to_dict('records')
        st.plotly_chart(create_regime_distribution_chart(results_list), use_container_width=True)
    
    with col2:
        st.markdown("##### HMM Bull Probability Distribution")
        fig = create_distribution_chart(df.rename(columns={'regime': 'signal'}))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()
    params = render_sidebar()
    
    if params['mode'] == 'dashboard':
        run_dashboard_mode(params)
    elif params['mode'] == 'chart':
        run_chart_mode(params)
    elif params['mode'] == 'screener':
        run_screener_mode(params)
    elif params['mode'] == 'regime':
        run_regime_mode(params)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"NIRNAY | A Pragyam Product | {VERSION}")


if __name__ == "__main__":
    main()
