# QuantumCommodityFramework.py
# Unified framework for quantum coordination analysis in commodity markets
# Easy commodity selection and clean Spyder integration

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class QuantumCommodityAnalyzer:
    """
    Unified quantum coordination analyzer with flexible commodity selection
    Designed for easy use in Spyder with simple commodity swapping
    """
    
    def __init__(self, commodity_set='agricultural'):
        """Initialize with chosen commodity set"""
        
        # Define all available commodity sets
        self.commodity_sets = {
            'agricultural': {
                'symbols': {
                    'corn_cme': 'ZC=F',      # CME Corn futures
                    'wheat_cme': 'ZW=F',     # CME Wheat futures  
                    'soybean_cme': 'ZS=F',   # CME Soybean futures
                    'soymeal_cme': 'ZM=F',   # CME Soybean meal futures
                    'soyoil_cme': 'ZL=F',    # CME Soybean oil futures
                },
                'description': 'Agricultural commodities (grains & oilseeds)',
                'why_good': 'Strong weather/supply chain correlations, crop rotation relationships'
            },
            
            'mixed': {
                'symbols': {
                    'gold_comex': 'GC=F',    # Gold futures
                    'crude_oil': 'CL=F',     # Crude Oil futures
                    'corn_cme': 'ZC=F',      # Corn futures
                    'copper_comex': 'HG=F',  # Copper futures
                    'natural_gas': 'NG=F',   # Natural Gas futures
                    'wheat_cme': 'ZW=F'      # Wheat futures
                },
                'description': 'Mixed commodities across sectors',
                'why_good': 'Tests cross-sector quantum correlations, broader market relationships'
            },
            
            'energy': {
                'symbols': {
                    'crude_oil_wti': 'CL=F', # WTI Crude futures
                    'crude_oil_brent': 'BZ=F', # Brent Crude futures
                    'natural_gas': 'NG=F',   # Natural Gas futures
                    'gasoline': 'RB=F',      # Gasoline futures
                    'heating_oil': 'HO=F'    # Heating Oil futures
                },
                'description': 'Energy commodities',
                'why_good': 'Tight supply chain connections, geopolitical correlations'
            },
            
            'metals': {
                'symbols': {
                    'gold_comex': 'GC=F',    # Gold futures
                    'silver_comex': 'SI=F',  # Silver futures
                    'copper_comex': 'HG=F',  # Copper futures
                    'platinum_nymex': 'PL=F', # Platinum futures
                    'palladium_nymex': 'PA=F' # Palladium futures
                },
                'description': 'Precious & industrial metals',
                'why_good': 'Safe haven correlations, industrial demand relationships'
            },
            
            'etfs': {
                'symbols': {
                    'gold_etf': 'GLD',       # Gold ETF
                    'oil_etf': 'USO',        # Oil ETF
                    'corn_etf': 'CORN',      # Corn ETF
                    'soy_etf': 'SOYB',       # Soybean ETF
                    'gas_etf': 'UNG',        # Natural Gas ETF
                    'agriculture_etf': 'DBA' # Agriculture ETF
                },
                'description': 'Commodity ETFs (easier data access)',
                'why_good': 'More reliable data availability, retail investor correlations'
            }
        }
        
        # Set the current commodity set
        self.set_commodity_set(commodity_set)
        
        self.data = {}
        self.correlations = {}
        
    def set_commodity_set(self, commodity_set):
        """Change commodity set easily"""
        if commodity_set not in self.commodity_sets:
            print(f"⚠️ Unknown commodity set '{commodity_set}'. Available sets:")
            for name, info in self.commodity_sets.items():
                print(f"   {name}: {info['description']}")
            commodity_set = 'agricultural'  # Default fallback
        
        self.current_set = commodity_set
        self.commodity_config = self.commodity_sets[commodity_set]
        self.commodity_tickers = self.commodity_config['symbols']
        
        print(f"📊 Selected commodity set: {commodity_set.upper()}")
        print(f"📋 Description: {self.commodity_config['description']}")
        print(f"💡 Why good: {self.commodity_config['why_good']}")
        print(f"🎯 Commodities: {list(self.commodity_tickers.keys())}")
        print()
        
    def collect_high_frequency_data(self, 
                                   start_date: str, 
                                   end_date: str,
                                   interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Collect commodity data with current commodity set
        """
        print(f"🌾 Collecting {self.current_set} commodity data...")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Resolution: {interval} intervals")
        
        collected_data = {}
        failed_symbols = []
        
        for name, ticker in self.commodity_tickers.items():
            try:
                print(f"   📊 Fetching {name} ({ticker})...", end=" ")
                
                # Get data
                commodity = yf.Ticker(ticker)
                df = commodity.history(
                    start=start_date, 
                    end=end_date, 
                    interval=interval
                )
                
                if len(df) > 10:  # Need minimum data
                    # Calculate price movements and volatility
                    df['returns'] = df['Close'].pct_change()
                    df['log_returns'] = np.log(df['Close']).diff()
                    df['volatility'] = df['returns'].rolling(window=min(30, len(df)//3)).std()
                    
                    # Binary measurement outcomes for Bell test
                    df['bell_measurement'] = np.where(df['returns'] > 0, 1, -1)
                    
                    # Detect unusual movements
                    window_size = min(50, len(df)//2)
                    df['unusual_movement'] = (
                        np.abs(df['returns']) > 
                        df['returns'].rolling(window=window_size).std() * 2
                    )
                    
                    collected_data[name] = df
                    print(f"✅ {len(df)} data points")
                else:
                    print(f"❌ Insufficient data ({len(df)} points)")
                    failed_symbols.append(ticker)
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                failed_symbols.append(ticker)
        
        if failed_symbols:
            print(f"\n⚠️ Failed to collect: {failed_symbols}")
            print(f"💡 Try different dates, interval, or commodity set")
        
        self.data = collected_data
        print(f"\n✅ Successfully collected: {len(collected_data)} commodities")
        return collected_data
    
    def detect_coordination_events(self, 
                                 time_window_seconds: int = 300,
                                 correlation_threshold: float = 0.5) -> pd.DataFrame:
        """
        Detect simultaneous price movements across commodities
        """
        print(f"🔍 Detecting coordination events...")
        print(f"   Time window: {time_window_seconds} seconds")
        print(f"   Correlation threshold: {correlation_threshold}")
        
        if len(self.data) < 2:
            print("   ❌ Need at least 2 commodities for coordination analysis")
            return pd.DataFrame()
        
        coordination_events = []
        commodity_names = list(self.data.keys())
        
        # Compare each pair of commodities
        for i, commodity1 in enumerate(commodity_names):
            for j, commodity2 in enumerate(commodity_names[i+1:], i+1):
                
                df1 = self.data[commodity1]
                df2 = self.data[commodity2]
                
                # Align timestamps
                aligned_df1, aligned_df2 = self._align_timestamps(df1, df2)
                
                if len(aligned_df1) < 30:
                    continue
                
                # Calculate rolling correlations
                window_size = max(10, min(30, len(aligned_df1) // 5))
                
                # Use returns for correlation (more stable than bell_measurement)
                correlations = aligned_df1['returns'].rolling(
                    window=window_size, min_periods=window_size//2
                ).corr(aligned_df2['returns'])
                
                # Find periods of high correlation
                high_corr_periods = correlations[
                    np.abs(correlations) > correlation_threshold
                ].dropna()
                
                print(f"   📊 {commodity1} ↔ {commodity2}: "
                      f"{len(high_corr_periods)} high-correlation periods")
                
                # Record coordination events
                for timestamp, correlation in high_corr_periods.items():
                    # Safely get values with error handling
                    try:
                        price1 = aligned_df1.loc[timestamp, 'Close'] if timestamp in aligned_df1.index else np.nan
                        price2 = aligned_df2.loc[timestamp, 'Close'] if timestamp in aligned_df2.index else np.nan
                        movement1 = aligned_df1.loc[timestamp, 'bell_measurement'] if timestamp in aligned_df1.index else 0
                        movement2 = aligned_df2.loc[timestamp, 'bell_measurement'] if timestamp in aligned_df2.index else 0
                    except:
                        price1 = price2 = np.nan
                        movement1 = movement2 = 0
                    
                    coordination_events.append({
                        'timestamp': timestamp,
                        'commodity1': commodity1,
                        'commodity2': commodity2,
                        'correlation': correlation,
                        'price1': price1,
                        'price2': price2,
                        'movement1': movement1,
                        'movement2': movement2
                    })
        
        events_df = pd.DataFrame(coordination_events)
        if len(events_df) > 0:
            events_df = events_df.sort_values('timestamp')
            print(f"   🎯 Total coordination events detected: {len(events_df)}")
        else:
            print(f"   ❌ No coordination events found")
        
        return events_df
    
    def test_bell_inequalities(self, 
                             events_df: pd.DataFrame,
                             test_multiple_inequalities: bool = True) -> Dict[str, Any]:
        """
        Test Bell inequalities on commodity coordination data
        """
        print(f"⚛️  Testing Bell inequalities on coordination events...")
        
        if len(events_df) < 4:
            print("   ❌ Insufficient coordination events for Bell test")
            return {}
        
        # Group events by commodity pairs
        commodity_pairs = events_df.groupby(['commodity1', 'commodity2'])
        
        bell_results = {}
        
        for (commodity1, commodity2), pair_events in commodity_pairs:
            print(f"   🧪 Testing {commodity1} ↔ {commodity2}...")
            
            if len(pair_events) < 4:
                print(f"      ⚠️ Only {len(pair_events)} events, skipping")
                continue
            
            # Test CHSH inequality
            chsh_measurements = self._prepare_chsh_measurements(pair_events)
            chsh_value = self._calculate_chsh_value(chsh_measurements)
            
            results = {
                'chsh_value': chsh_value,
                'chsh_violation': chsh_value > 2.0,
                'quantum_bound': chsh_value <= 2.828,
                'events_count': len(pair_events),
                'correlations': chsh_measurements
            }
            
            if test_multiple_inequalities and len(pair_events) >= 8:
                # Additional tests for robust detection
                ch74_value = self._test_ch74_inequality(pair_events)
                results['ch74_value'] = ch74_value
                results['ch74_violation'] = ch74_value > 2.0
                
                temporal_bell = self._test_temporal_bell_inequality(pair_events)
                results['temporal_bell_value'] = temporal_bell
                results['temporal_violation'] = temporal_bell > 2.0
            else:
                results['ch74_value'] = 0
                results['ch74_violation'] = False
                results['temporal_bell_value'] = 0
                results['temporal_violation'] = False
            
            # Overall violation assessment
            violations = sum([
                results.get('chsh_violation', False),
                results.get('ch74_violation', False),
                results.get('temporal_violation', False)
            ])
            
            results['total_violations'] = violations
            results['quantum_signature_strength'] = violations / 3.0
            
            bell_results[f"{commodity1}_{commodity2}"] = results
            
            if violations > 0:
                print(f"      🚨 QUANTUM SIGNATURES: {violations}/3 tests violated!")
                print(f"         CHSH = {chsh_value:.3f}")
            else:
                print(f"      ✅ Classical: CHSH = {chsh_value:.3f}")
        
        return bell_results
    
    def _align_timestamps(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align timestamps between different commodity datasets"""
        # Find common time range
        start_time = max(df1.index.min(), df2.index.min())
        end_time = min(df1.index.max(), df2.index.max())
        
        # Filter to common time range
        df1_aligned = df1[(df1.index >= start_time) & (df1.index <= end_time)]
        df2_aligned = df2[(df2.index >= start_time) & (df2.index <= end_time)]
        
        return df1_aligned, df2_aligned
    
    def _prepare_chsh_measurements(self, events_df: pd.DataFrame) -> Dict[str, float]:
        """
        Prepare measurement settings for CHSH Bell inequality test
        """
        events_df = events_df.copy()
        
        # Create different measurement bases
        events_df['vol_regime1'] = np.where(
            events_df['correlation'] > events_df['correlation'].median(), 1, -1
        )
        events_df['vol_regime2'] = np.where(
            np.abs(events_df['correlation']) > events_df['correlation'].quantile(0.75), 1, -1  
        )
        
        # CHSH measurement settings with error handling
        def safe_corr(x, y):
            try:
                if len(x) != len(y) or len(x) < 2:
                    return 0.0
                corr_val = np.corrcoef(x, y)[0, 1]
                return corr_val if not np.isnan(corr_val) else 0.0
            except:
                return 0.0
        
        measurements = {
            'E_AB': safe_corr(events_df['movement1'], events_df['movement2']),
            'E_AB_prime': safe_corr(events_df['movement1'], events_df['vol_regime2']), 
            'E_A_prime_B': safe_corr(events_df['vol_regime1'], events_df['movement2']),
            'E_A_prime_B_prime': safe_corr(events_df['vol_regime1'], events_df['vol_regime2'])
        }
                
        return measurements
    
    def _calculate_chsh_value(self, measurements: Dict[str, float]) -> float:
        """Calculate CHSH Bell inequality value"""
        E_AB = measurements.get('E_AB', 0)
        E_AB_prime = measurements.get('E_AB_prime', 0)  
        E_A_prime_B = measurements.get('E_A_prime_B', 0)
        E_A_prime_B_prime = measurements.get('E_A_prime_B_prime', 0)
        
        # CHSH inequality: |E(AB) + E(AB') + E(A'B) - E(A'B')| ≤ 2
        chsh_value = abs(E_AB + E_AB_prime + E_A_prime_B - E_A_prime_B_prime)
        
        return chsh_value
    
    def _test_ch74_inequality(self, pair_events: pd.DataFrame) -> float:
        """Test CH74 Bell inequality with different measurement settings"""
        if len(pair_events) < 4:
            return 0.0
        
        try:
            pair_events = pair_events.copy()
            pair_events['volume_direction'] = np.where(
                pair_events['correlation'] > pair_events['correlation'].quantile(0.75), 1, -1
            )
            
            # CH74: |E(AB) - E(AB') + E(A'B) + E(A'B')| ≤ 2
            def safe_corr(x, y):
                try:
                    corr_val = np.corrcoef(x, y)[0, 1]
                    return corr_val if not np.isnan(corr_val) else 0.0
                except:
                    return 0.0
            
            E_AB = safe_corr(pair_events['movement1'], pair_events['movement2'])
            E_AB_prime = safe_corr(pair_events['movement1'], pair_events['volume_direction'])
            E_A_prime_B = safe_corr(pair_events['volume_direction'], pair_events['movement2'])
            E_A_prime_B_prime = safe_corr(pair_events['volume_direction'], pair_events['volume_direction'])
            
            ch74_value = abs(E_AB - E_AB_prime + E_A_prime_B + E_A_prime_B_prime)
            return ch74_value
        except:
            return 0.0
    
    def _test_temporal_bell_inequality(self, pair_events: pd.DataFrame) -> float:
        """Test Bell inequality across time windows"""
        if len(pair_events) < 8:
            return 0.0
        
        try:
            # Split into early and late time windows
            mid_point = len(pair_events) // 2
            early_events = pair_events.iloc[:mid_point]
            late_events = pair_events.iloc[mid_point:]
            
            if len(early_events) < 2 or len(late_events) < 2:
                return 0.0
            
            early_movements1 = early_events['movement1'].values
            late_movements2 = late_events['movement2'].values
            
            # Align lengths
            min_len = min(len(early_movements1), len(late_movements2))
            if min_len < 2:
                return 0.0
            
            early_movements1 = early_movements1[:min_len]
            late_movements2 = late_movements2[:min_len]
            
            temporal_correlation = np.corrcoef(early_movements1, late_movements2)[0, 1]
            
            if np.isnan(temporal_correlation):
                return 0.0
            
            temporal_bell_value = abs(temporal_correlation) * 2.0
            return temporal_bell_value
        except:
            return 0.0
    
    def get_available_commodity_sets(self):
        """Return available commodity sets for easy reference"""
        return list(self.commodity_sets.keys())
    
    def print_commodity_sets(self):
        """Print all available commodity sets"""
        print("🎯 AVAILABLE COMMODITY SETS:")
        print("="*50)
        for name, config in self.commodity_sets.items():
            print(f"{name.upper()}:")
            print(f"   Description: {config['description']}")
            print(f"   Why good: {config['why_good']}")
            print(f"   Commodities: {list(config['symbols'].keys())}")
            print()

# Convenience function for easy imports
def create_analyzer(commodity_set='agricultural'):
    """Create analyzer with specified commodity set"""
    return QuantumCommodityAnalyzer(commodity_set)