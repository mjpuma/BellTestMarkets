# Quantum Commodity Coordination Analysis Framework

## 🌟 Overview

This framework analyzes commodity market data for quantum-like coordination patterns using Bell inequality tests. It explores whether commodity price movements exhibit correlations that violate classical bounds, suggesting non-local market coordination similar to quantum entanglement.

The framework is designed for easy experimentation in Spyder or any Python environment, with flexible commodity selection and publication-quality visualizations.

## 📊 Key Features

- **5 Pre-configured Commodity Sets**: Agricultural, Mixed, Energy, Metals, and ETFs
- **Bell Inequality Testing**: CHSH, CH74, and temporal Bell tests
- **Publication-Quality Figures**: Automated generation of analysis plots
- **Flexible Time Resolution**: Daily, hourly, or 5-minute data
- **Easy Configuration**: Simple parameter changes for different analyses

## 🚀 Quick Start

### Installation

```bash
pip install yfinance pandas numpy matplotlib scipy seaborn
```

### Basic Usage

```python
# In SpyderQuantumAnalysis.py, just change these settings:
COMMODITY_SET = 'mixed'              # Choose commodity set
ANALYSIS_START_DATE = '2025-03-01'   # Start date
ANALYSIS_END_DATE = '2025-03-31'     # End date
DATA_INTERVAL = '1h'                 # Data resolution
CORRELATION_THRESHOLD = 0.3          # Correlation threshold

# Run the script - it handles everything else!
```

## 📈 Data Sources

### Primary Source: Yahoo Finance (yfinance)

The framework uses Yahoo Finance API to fetch real-time and historical commodity data:

- **Futures Contracts**: Direct commodity futures (e.g., 'CL=F' for crude oil)
- **ETFs**: Exchange-traded funds tracking commodities (e.g., 'GLD' for gold)

### Why Yahoo Finance?

1. **Free Access**: No API keys required
2. **Comprehensive Coverage**: Most major commodities available
3. **Multiple Resolutions**: From 1-minute to daily data
4. **Reliable Updates**: Real-time during market hours

## 🎯 Commodity Choices

### 1. Agricultural Set
```python
'agricultural': {
    'corn_cme': 'ZC=F',      # CME Corn futures
    'wheat_cme': 'ZW=F',     # CME Wheat futures  
    'soybean_cme': 'ZS=F',   # CME Soybean futures
    'soymeal_cme': 'ZM=F',   # CME Soybean meal futures
    'soyoil_cme': 'ZL=F',    # CME Soybean oil futures
}
```
**Why Good**: Strong weather/supply chain correlations, crop rotation relationships

### 2. Mixed Set
```python
'mixed': {
    'gold_comex': 'GC=F',    # Gold futures
    'crude_oil': 'CL=F',     # Crude Oil futures
    'corn_cme': 'ZC=F',      # Corn futures
    'copper_comex': 'HG=F',  # Copper futures
    'natural_gas': 'NG=F',   # Natural Gas futures
    'wheat_cme': 'ZW=F'      # Wheat futures
}
```
**Why Good**: Tests cross-sector quantum correlations, broader market relationships

### 3. Energy Set
```python
'energy': {
    'crude_oil_wti': 'CL=F',   # WTI Crude futures
    'crude_oil_brent': 'BZ=F', # Brent Crude futures
    'natural_gas': 'NG=F',     # Natural Gas futures
    'gasoline': 'RB=F',        # Gasoline futures
    'heating_oil': 'HO=F'      # Heating Oil futures
}
```
**Why Good**: Tight supply chain connections, geopolitical correlations

### 4. Metals Set
```python
'metals': {
    'gold_comex': 'GC=F',      # Gold futures
    'silver_comex': 'SI=F',    # Silver futures
    'copper_comex': 'HG=F',    # Copper futures
    'platinum_nymex': 'PL=F',  # Platinum futures
    'palladium_nymex': 'PA=F'  # Palladium futures
}
```
**Why Good**: Safe haven correlations, industrial demand relationships

### 5. ETF Set
```python
'etfs': {
    'gold_etf': 'GLD',         # Gold ETF
    'oil_etf': 'USO',          # Oil ETF
    'corn_etf': 'CORN',        # Corn ETF
    'soy_etf': 'SOYB',         # Soybean ETF
    'gas_etf': 'UNG',          # Natural Gas ETF
    'agriculture_etf': 'DBA'   # Agriculture ETF
}
```
**Why Good**: More reliable data availability, retail investor correlations

## ⏰ Resolution Choices

### Available Intervals

- **'1d'**: Daily data - Best for long-term analysis (months/years)
- **'1h'**: Hourly data - Good balance of detail and data availability
- **'5m'**: 5-minute data - High-frequency analysis (limited history)
- **'15m'**: 15-minute data - Alternative high-frequency option
- **'30m'**: 30-minute data - Medium-frequency analysis

### Resolution Considerations

1. **Data Availability**: Shorter intervals have limited historical data
2. **Event Detection**: Higher frequency captures more coordination events
3. **Noise vs Signal**: Very high frequency may introduce noise
4. **Computational Load**: More data points increase processing time

## 📐 Correlation Metrics

### Current Implementation: Pearson Correlation

The framework uses **rolling Pearson correlation** on price returns:

```python
# Calculate returns
df['returns'] = df['Close'].pct_change()

# Rolling correlation between two commodities
window_size = 30  # Adjust based on data frequency
correlations = commodity1['returns'].rolling(window=window_size).corr(commodity2['returns'])
```

### Why Pearson Correlation?

1. **Standard Measure**: Well-understood in finance
2. **Linear Relationships**: Captures direct price co-movements
3. **Range [-1, 1]**: Easy interpretation
4. **Efficient Computation**: Fast for large datasets

### Alternative Correlation Metrics

You can modify the correlation calculation in `detect_coordination_events()`:

#### 1. Spearman Rank Correlation (Non-linear relationships)
```python
from scipy.stats import spearmanr

# Replace the correlation calculation with:
window_data1 = aligned_df1['returns'].rolling(window=window_size)
window_data2 = aligned_df2['returns'].rolling(window=window_size)

correlations = window_data1.apply(
    lambda x: spearmanr(x, window_data2)[0] if len(x) == window_size else np.nan
)
```

#### 2. Kendall's Tau (Ordinal association)
```python
from scipy.stats import kendalltau

# Use for concordance/discordance patterns
correlations = window_data1.apply(
    lambda x: kendalltau(x, window_data2)[0] if len(x) == window_size else np.nan
)
```

#### 3. Mutual Information (Non-linear dependencies)
```python
from sklearn.feature_selection import mutual_info_regression

# Captures any statistical dependency
def calculate_mi(x, y):
    return mutual_info_regression(x.reshape(-1, 1), y)[0]

correlations = window_data1.apply(
    lambda x: calculate_mi(x.values, window_data2.values) if len(x) == window_size else np.nan
)
```

#### 4. Dynamic Time Warping (Time-shifted correlations)
```python
from dtaidistance import dtw

# For detecting lagged relationships
def dtw_correlation(x, y):
    distance = dtw.distance(x.values, y.values)
    return 1 / (1 + distance)  # Convert distance to similarity
```

#### 5. Transfer Entropy (Directional information flow)
```python
# Requires pyinform package
from pyinform import transfer_entropy

# Measures information transfer between commodities
def calculate_te(x, y, k=1):
    return transfer_entropy(x, y, k)
```

### Implementing Custom Metrics

To add a new correlation metric:

1. Modify the `detect_coordination_events()` method:
```python
def detect_coordination_events(self, 
                             time_window_seconds: int = 300,
                             correlation_threshold: float = 0.5,
                             correlation_type: str = 'pearson') -> pd.DataFrame:
    
    # Add correlation type selection
    if correlation_type == 'pearson':
        correlations = aligned_df1['returns'].rolling(window=window_size).corr(aligned_df2['returns'])
    elif correlation_type == 'spearman':
        # Your custom implementation
    # ... etc
```

2. Update threshold values as different metrics have different ranges

3. Consider normalization for metrics not bounded to [-1, 1]

## 🔬 Bell Inequality Testing

### What are Bell Inequalities?

Bell inequalities set bounds on correlations possible in classical (local realistic) systems. Violations suggest quantum-like non-local correlations.

### CHSH Inequality

The main test used: |S| ≤ 2 for classical systems, where:
```
S = E(AB) + E(AB') + E(A'B) - E(A'B')
```

- Classical bound: |S| ≤ 2
- Quantum bound: |S| ≤ 2√2 ≈ 2.828

### Measurement Settings

The framework creates different "measurement bases" from market data:
- **A, B**: Direct price movements
- **A', B'**: Volatility regimes or correlation-based measurements

## 📊 Output Interpretation

### Figure 1: Overview
- **Panel A**: Normalized price evolution
- **Panel B**: Correlation distribution
- **Panel C**: CHSH values for each commodity pair
- **Panel D**: Coordination event timeline
- **Panel E**: Cross-commodity correlation network

### Figure 2: Bell Analysis
- **Panel A**: CHSH values with classical/quantum limits
- **Panel B**: Approach to quantum regime
- **Panel C**: CHSH distribution
- **Panel D**: Statistical significance

### Figure 3: Statistical Analysis
- **Panel A**: Correlation significance testing
- **Panel B**: Temporal distribution of events
- **Panel C**: Power analysis
- **Panel D**: Summary metrics

### Summary Table
CSV file with all key metrics and results

## 🎯 Interpreting Results

### No Violation (CHSH ≤ 2)
- Classical market behavior
- Explainable by common factors
- Normal market efficiency

### Violation (CHSH > 2)
- Quantum-like coordination
- Non-local market correlations
- Potential evidence of:
  - Algorithmic trading synchronization
  - Information propagation anomalies
  - Market microstructure effects

## 🛠️ Customization Tips

### 1. Add New Commodities
```python
self.commodity_sets['custom'] = {
    'symbols': {
        'bitcoin': 'BTC-USD',
        'ethereum': 'ETH-USD',
        # Add more...
    },
    'description': 'Cryptocurrency markets',
    'why_good': 'High volatility, 24/7 trading'
}
```

### 2. Modify Coordination Detection
- Change correlation window size
- Adjust significance thresholds
- Add volatility filters

### 3. Enhance Bell Tests
- Implement additional Bell inequalities
- Try different measurement bases
- Add bootstrap confidence intervals

## 📚 References

1. Bell, J.S. (1964). "On the Einstein Podolsky Rosen Paradox"
2. Clauser, Horne, Shimony, Holt (1969). "Proposed experiment to test local hidden-variable theories"
3. Haven, E. & Khrennikov, A. (2013). "Quantum Social Science"

## 🤝 Contributing

Feel free to:
- Add new commodity sets
- Implement additional correlation metrics
- Enhance visualization options
- Improve Bell test implementations

## 📄 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This framework is for research and educational purposes. Detected "quantum signatures" in financial data should be interpreted carefully and do not guarantee any trading advantage or predictive power.
