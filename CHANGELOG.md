# NIRNAY Changelog

All notable changes to NIRNAY Terminal are documented here.

---

## [7.2.0] — 2026-04-19

### ✨ Design System Refinement
- **Complete chart redesign**: All 27 chart instances refactored to match Nishkarsh "Obsidian Quant Terminal" design language
- **Unified theme system**: Implemented `chart_layout()` and `style_axes()` across all visualizations
- **Consistent color palette**: All charts now use standardized COLOR_GREEN, COLOR_RED, COLOR_GOLD, COLOR_CYAN, COLOR_AMBER, COLOR_PURPLE, COLOR_MUTED
- **Custom spike design**: Added institutional-grade crosshair styling to all time series charts
- **Typography standardization**: JetBrains Mono throughout (headers via Syne display font)
- **Chart height constants**: Introduced UI_CHART_HEIGHT_* constants (SMALL=280px, MEDIUM=340px, LARGE=380px, XLARGE=540px, STACKED=680px)

### 🎛️ UI/UX Improvements
- **Dynamic sidebar button**: Single contextual action button that changes based on mode/analysis type selection
- **Button text**: Removed decorative icons, now reads "RUN ETF SCREENER", "RUN MARKET SCREENER", etc.
- **Conditional display**: Page descriptions now hide when button is pressed, showing only progress bar and results
- **Improved spacing**: Reduced gaps in sidebar, moved descriptions lower on page
- **Progress tracking**: Enhanced progress bar with step-by-step indication (Nishkarsh-style progress logging)

### 📋 Infrastructure
- **Version bumped**: 7.1.0 → 7.2.0
- **Configuration system**: Centralized color and chart dimension constants in `core/config.py`
- **Theme imports**: Added `chart_layout` and `style_axes` to theme.py imports

### 🔧 Technical Details
- **Chart styling**: Eliminated hardcoded colors (#10b981, #ef4444, etc.) — now using config constants
- **Layout unification**: Removed `template='plotly_dark'` pattern; all charts inherit from `chart_layout()`
- **Margin handling**: Fixed duplicate margin parameter conflicts in chart updates
- **Container width**: Updated all `st.plotly_chart()` calls to use `use_container_width=True` with unique keys

---

## [7.1.0] — Previous Release

- Landing page redesign with Pragyam-style card layout
- Sidebar logo and navigation refinements
- Initial institutional terminal aesthetic implementation

---

## Roadmap

### Planned for 7.3.0
- [ ] Terminal logging with Nishkarsh-style step indicators
- [ ] Custom progress bar animations
- [ ] Real-time data streaming for time series modes
- [ ] Advanced filtering and custom universes
- [ ] Export to CSV/Excel with formatting

### Planned for 7.4.0
- [ ] Multi-timeframe analysis (daily / weekly / monthly)
- [ ] Custom signal weighting (MSF vs MMR adjustment)
- [ ] Portfolio analysis mode
- [ ] Signal backtesting framework
- [ ] ML-based anomaly detection

---

## Notes for Contributors

- All new charts must use `fig.update_layout(**chart_layout(...))` + `style_axes(fig, ...)`
- Color codes must be sourced from `core.config` COLOR_* constants
- Button text should be descriptive and icon-free
- Progress updates should follow Nishkarsh terminal logging style

---

## Building & Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Build for production
# (CI/CD pipeline configuration needed)
```

---

**Maintained by @thebullishvalue**
