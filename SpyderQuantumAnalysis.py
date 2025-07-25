# SpyderQuantumAnalysis.py
# Main analysis script for Spyder - Easy to run and modify
# Just change the settings below and run!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our updated framework
from QuantumCommodityFramework import QuantumCommodityAnalyzer

#============================================================================
# EASY CONFIGURATION - Just change these settings!
#============================================================================

# 1. Choose your commodity set (just change this one line!)
COMMODITY_SET = 'etfs'  # Options: 'agricultural', 'mixed', 'energy', 'metals', 'etfs'

# 2. Choose your analysis period
ANALYSIS_START_DATE = '2025-01-01'  # Start date (use longer period for more data)
ANALYSIS_END_DATE = '2025-07-25'    # End date
PERIOD_NAME = 'Jan_July_2025'       # Name for your analysis files

# 3. Analysis parameters
DATA_INTERVAL = '1d'                # Data interval: '1d' (daily), '1h' (hourly), '5m' (5-min)
CORRELATION_THRESHOLD = 0.3         # Correlation threshold (0.3-0.7, lower = more events)

#============================================================================
# PUBLICATION-QUALITY STYLING
#============================================================================

def setup_publication_style():
    """Set up beautiful plots like Science/Nature journals"""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (14, 10),
        'font.size': 11,
        'font.family': 'Arial',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'axes.edgecolor': 'black',
        'axes.grid': True,
        'grid.alpha': 0.3
    })

#============================================================================
# MAIN ANALYSIS FUNCTION
#============================================================================

def run_quantum_analysis():
    """Run complete quantum analysis with current settings"""
    
    # Setup beautiful plots
    setup_publication_style()
    
    print("🚀 QUANTUM COMMODITY ANALYSIS - SPYDER EDITION")
    print("="*80)
    print(f"📊 Analyzing Period: {PERIOD_NAME}")
    print(f"📅 Date Range: {ANALYSIS_START_DATE} to {ANALYSIS_END_DATE}")
    print(f"🎯 Commodity Set: {COMMODITY_SET.upper()}")
    print(f"⏱️ Data Interval: {DATA_INTERVAL}")
    print(f"🔗 Correlation Threshold: {CORRELATION_THRESHOLD}")
    print()
    
    # Initialize the quantum analyzer with chosen commodity set
    analyzer = QuantumCommodityAnalyzer(commodity_set=COMMODITY_SET)
    
    # Step 1: Collect data
    print("📈 Step 1: Collecting commodity data...")
    data = analyzer.collect_high_frequency_data(
        start_date=ANALYSIS_START_DATE, 
        end_date=ANALYSIS_END_DATE, 
        interval=DATA_INTERVAL
    )
    
    if len(data) < 2:
        print("❌ Not enough data collected. Try:")
        print("   • Different date range (longer period)")
        print("   • Different commodity set (try 'etfs' for better availability)")
        print("   • Different data interval (try '1d' for daily data)")
        return None
    
    print(f"✅ Collected data for {len(data)} commodities")
    total_points = sum(len(df) for df in data.values())
    print(f"📊 Total data points: {total_points:,}")
    
    # Step 2: Detect coordination events
    print("\n🔍 Step 2: Detecting coordination events...")
    events = analyzer.detect_coordination_events(
        correlation_threshold=CORRELATION_THRESHOLD
    )
    
    print(f"🎯 Found {len(events)} coordination events")
    
    if len(events) == 0:
        print("⚠️ No coordination events found. Try:")
        print("   • Lower correlation threshold (0.3 instead of current)")
        print("   • Different commodity set ('mixed' often shows more correlations)")
        print("   • Longer time period")
        
        # Show what data we do have
        print(f"\n📊 Data summary:")
        for name, df in data.items():
            returns_std = df['returns'].std()
            print(f"   {name}: {len(df)} points, volatility: {returns_std:.4f}")
        
        return {
            'data': data,
            'events': pd.DataFrame(),
            'bell_results': {},
            'total_violations': 0,
            'max_chsh': 0,
            'period_name': PERIOD_NAME,
            'commodity_set': COMMODITY_SET
        }
    
    # Step 3: Run Bell inequality tests
    print("\n⚛️ Step 3: Testing Bell inequalities...")
    bell_results = analyzer.test_bell_inequalities(events, test_multiple_inequalities=True)
    
    # Count quantum violations
    total_violations = 0
    max_chsh = 0
    mean_chsh = 0
    chsh_values = []
    
    if bell_results:
        for pair, results in bell_results.items():
            if isinstance(results, dict):
                chsh = results.get('chsh_value', 0)
                violations = results.get('total_violations', 0)
                total_violations += violations
                max_chsh = max(max_chsh, chsh)
                if chsh > 0:
                    chsh_values.append(chsh)
        
        if chsh_values:
            mean_chsh = np.mean(chsh_values)
        
        print(f"📊 Bell test results: {len(bell_results)} commodity pairs tested")
        print(f"📈 Mean CHSH value: {mean_chsh:.3f}")
        print(f"⚡ Maximum CHSH value: {max_chsh:.3f}")
        print(f"🚨 Quantum violations detected: {total_violations}")
        
        if max_chsh > 2.0:
            violation_percent = ((max_chsh - 2.0) / (2.828 - 2.0)) * 100
            print("🎉 BELL INEQUALITY VIOLATION DETECTED!")
            print("⚛️ Evidence for quantum-like coordination!")
            print(f"🔥 {violation_percent:.1f}% toward quantum bound!")
        elif max_chsh > 1.8:
            classical_percent = (max_chsh / 2.0) * 100
            print("⚡ Very strong classical correlations detected!")
            print(f"📊 {classical_percent:.1f}% of classical bound reached!")
        else:
            print("✅ Classical coordination confirmed")
    
    # Step 4: Create beautiful publication-quality figures
    print("\n🎨 Step 4: Creating publication-quality figures...")
    
    # FIGURE 1: Comprehensive Overview
    create_figure_1_overview(data, events, bell_results, PERIOD_NAME, COMMODITY_SET)
    
    # FIGURE 2: Bell Inequality Analysis
    create_figure_2_bell_analysis(bell_results, PERIOD_NAME, COMMODITY_SET)
    
    # FIGURE 3: Statistical Analysis
    if len(events) > 0:
        create_figure_3_statistics(events, bell_results, PERIOD_NAME, COMMODITY_SET)
    
    # Create summary table
    create_summary_table(data, events, bell_results, PERIOD_NAME, COMMODITY_SET)
    
    print("\n🏆 ANALYSIS COMPLETE!")
    print("📁 Check your folder for:")
    print(f"   📊 Figure1_Overview_{COMMODITY_SET}_{PERIOD_NAME}.png")
    print(f"   📈 Figure2_BellAnalysis_{COMMODITY_SET}_{PERIOD_NAME}.png") 
    if len(events) > 0:
        print(f"   📉 Figure3_Statistics_{COMMODITY_SET}_{PERIOD_NAME}.png")
    print(f"   📋 Summary_Table_{COMMODITY_SET}_{PERIOD_NAME}.csv")
    
    # Return results for further analysis
    return {
        'data': data,
        'events': events, 
        'bell_results': bell_results,
        'total_violations': total_violations,
        'max_chsh': max_chsh,
        'mean_chsh': mean_chsh,
        'period_name': PERIOD_NAME,
        'commodity_set': COMMODITY_SET,
        'analyzer': analyzer
    }

#============================================================================
# VISUALIZATION FUNCTIONS (Enhanced for different commodity sets)
#============================================================================

def create_figure_1_overview(data, events, bell_results, period_name, commodity_set):
    """Create Figure 1: Comprehensive overview"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Price evolution
    ax1 = fig.add_subplot(gs[0, :])
    
    # Get colors for different commodities
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    
    for i, (commodity, df) in enumerate(data.items()):
        if len(df) > 1 and i < 6:  # Limit to 6 for clarity
            normalized_price = df['Close'] / df['Close'].iloc[0]
            ax1.plot(df.index, normalized_price, 
                    label=commodity.replace('_', ' ').title(),
                    linewidth=2, alpha=0.8, color=colors[i])
    
    ax1.set_ylabel('Normalized Price', fontweight='bold')
    ax1.set_title(f'A. {commodity_set.title()} Price Evolution - {period_name.replace("_", " ")}', 
                 fontweight='bold', fontsize=14)
    ax1.legend(frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Correlation distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    if len(events) > 0:
        correlations = events['correlation'].values
        n_bins = min(30, len(correlations) // 10 + 5)
        
        ax2.hist(correlations, bins=n_bins, alpha=0.7, color='#2E8B57', 
                edgecolor='black', linewidth=1)
        ax2.axvline(correlations.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {correlations.mean():.3f}')
        ax2.axvline(CORRELATION_THRESHOLD, color='orange', linestyle='--', linewidth=2,
                   label=f'Threshold: {CORRELATION_THRESHOLD}')
        ax2.set_xlabel('Correlation Coefficient', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('B. Correlation Distribution', fontweight='bold')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No Coordination\nEvents Found', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
        ax2.set_title('B. Correlation Distribution', fontweight='bold')
    
    # Panel C: CHSH values
    ax3 = fig.add_subplot(gs[1, 1])
    
    if bell_results:
        chsh_values = [res.get('chsh_value', 0) for res in bell_results.values() 
                      if isinstance(res, dict) and res.get('chsh_value', 0) > 0]
        
        if chsh_values:
            bars = ax3.bar(range(len(chsh_values)), chsh_values, 
                          color='#DAA520', alpha=0.7, edgecolor='black')
            ax3.axhline(y=2.0, color='red', linestyle='--', linewidth=2,
                       label='Classical Limit')
            ax3.axhline(y=2.828, color='purple', linestyle='--', linewidth=2,
                       label='Quantum Limit')
            
            # Highlight violations
            for i, (bar, chsh) in enumerate(zip(bars, chsh_values)):
                if chsh > 2.0:
                    bar.set_color('red')
                    bar.set_alpha(0.9)
                    ax3.text(i, chsh + 0.1, '🚨', ha='center', fontsize=12)
            
            ax3.set_ylabel('CHSH Value', fontweight='bold')
            ax3.set_xlabel('Commodity Pair', fontweight='bold')
            ax3.set_title('C. Bell Inequality Tests', fontweight='bold')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No Valid\nCHSH Values', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12, bbox=dict(boxstyle="round", facecolor='lightblue'))
            ax3.set_title('C. Bell Inequality Tests', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Bell Tests\nNot Performed', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
        ax3.set_title('C. Bell Inequality Tests', fontweight='bold')
    
    # Panel D: Event timeline
    ax4 = fig.add_subplot(gs[1, 2])
    
    if len(events) > 0:
        events_sorted = events.sort_values('timestamp')
        events_sorted['date'] = events_sorted['timestamp'].dt.date
        daily_counts = events_sorted.groupby('date').size()
        
        ax4.plot(daily_counts.index, daily_counts.values, 
                marker='o', linewidth=2, markersize=4, color='#4682B4')
        ax4.fill_between(daily_counts.index, daily_counts.values, alpha=0.3,
                        color='#4682B4')
        ax4.set_ylabel('Daily Events', fontweight='bold')
        ax4.set_title('D. Coordination Timeline', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No Events\nTo Plot', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
        ax4.set_title('D. Coordination Timeline', fontweight='bold')
    
    # Panel E: Correlation network
    ax5 = fig.add_subplot(gs[2, :])
    
    commodities = list(data.keys())
    n_commodities = len(commodities)
    correlation_matrix = np.zeros((n_commodities, n_commodities))
    
    # Calculate correlation matrix
    for i, commodity1 in enumerate(commodities):
        for j, commodity2 in enumerate(commodities):
            if i != j and commodity1 in data and commodity2 in data:
                df1, df2 = data[commodity1], data[commodity2]
                if len(df1) > 10 and len(df2) > 10:
                    try:
                        common_times = set(df1.index) & set(df2.index)
                        if len(common_times) > 10:
                            common_times = sorted(list(common_times))
                            returns1 = df1.loc[common_times, 'returns'].dropna()
                            returns2 = df2.loc[common_times, 'returns'].dropna()
                            
                            if len(returns1) > 5 and len(returns2) > 5:
                                min_len = min(len(returns1), len(returns2))
                                corr = np.corrcoef(returns1[:min_len], returns2[:min_len])[0, 1]
                                if not np.isnan(corr):
                                    correlation_matrix[i, j] = corr
                    except:
                        continue
    
    # Plot correlation matrix
    im = ax5.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    commodity_labels = [c.replace('_', '\n') for c in commodities]
    ax5.set_xticks(range(n_commodities))
    ax5.set_yticks(range(n_commodities))
    ax5.set_xticklabels(commodity_labels, rotation=45, ha='right')
    ax5.set_yticklabels(commodity_labels)
    ax5.set_title(f'E. {commodity_set.title()} Cross-Commodity Correlation Network', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontweight='bold')
    
    plt.suptitle(f'Quantum Coordination Analysis: {commodity_set.title()} - {period_name.replace("_", " ")}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    filename = f'Figure1_Overview_{commodity_set}_{period_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Saved: {filename}")
    plt.show()

def create_figure_2_bell_analysis(bell_results, period_name, commodity_set):
    """Create Figure 2: Bell inequality analysis"""
    
    # Extract CHSH values more robustly
    chsh_values = []
    commodity_pairs = []
    
    if bell_results:
        for pair, results in bell_results.items():
            if isinstance(results, dict) and 'chsh_value' in results:
                chsh_val = results.get('chsh_value', 0)
                if chsh_val > 0:  # Only include non-zero CHSH values
                    chsh_values.append(chsh_val)
                    commodity_pairs.append(pair)
    
    if len(chsh_values) == 0:
        # Create placeholder figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f'No Bell Inequality Results Available\nfor {commodity_set.title()} Commodities\n\nTry:\n• Different commodity set\n• Lower correlation threshold\n• Longer time period', 
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title(f'Bell Analysis: {commodity_set.title()} - {period_name.replace("_", " ")}', 
                 fontsize=14, fontweight='bold')
        
        filename = f'Figure2_BellAnalysis_{commodity_set}_{period_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Saved placeholder: {filename}")
        plt.show()
        return
    
    # Create the full 4-panel figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: CHSH values bar chart
    x_pos = np.arange(len(commodity_pairs))
    colors = ['red' if chsh > 2.0 else 'orange' if chsh > 1.5 else 'steelblue' for chsh in chsh_values]
    bars = ax1.bar(x_pos, chsh_values, color=colors, alpha=0.7, edgecolor='black')
    
    ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=2, 
               label='Classical Limit (CHSH ≤ 2)')
    ax1.axhline(y=2.828, color='purple', linestyle='--', linewidth=2,
               label='Quantum Limit (CHSH ≤ 2√2)')
    
    # Add value labels on bars
    for i, (bar, chsh) in enumerate(zip(bars, chsh_values)):
        if chsh > 2.0:
            ax1.text(i, chsh + 0.05, f'🚨\n{chsh:.3f}', ha='center', 
                    fontweight='bold', color='red', fontsize=9)
        else:
            ax1.text(i, chsh + max(chsh_values)*0.02, f'{chsh:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    ax1.set_ylabel('CHSH Value', fontweight='bold')
    ax1.set_title(f'A. CHSH Bell Inequality Results - {commodity_set.title()}', fontweight='bold')
    ax1.legend()
    
    # Clean up pair names for x-axis
    clean_pairs = [pair.replace('_', ' vs ').replace(' cme', '').replace(' etf', '').replace(' comex', '').replace(' nymex', '') 
                   for pair in commodity_pairs]
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(clean_pairs, rotation=45, ha='right', fontsize=8)
    
    # Panel B: Approach to quantum regime
    max_chsh = max(chsh_values)
    approach_percentage = (max_chsh / 2.0) * 100
    
    ax2.bar(['Current\nMax CHSH', 'Classical\nLimit', 'Quantum\nLimit'], 
            [max_chsh, 2.0, 2.828],
            color=['red' if max_chsh > 2.0 else 'orange' if max_chsh > 1.5 else 'steelblue', 'red', 'purple'],
            alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('CHSH Value', fontweight='bold')
    ax2.set_title(f'B. Approach to Quantum Regime ({approach_percentage:.1f}%)', fontweight='bold')
    
    # Add percentage annotation
    if max_chsh > 2.0:
        violation_pct = ((max_chsh - 2.0) / (2.828 - 2.0)) * 100
        ax2.text(0, max_chsh + 0.1, f'{max_chsh:.3f}\n🚨 {violation_pct:.1f}%\ntoward quantum!', 
                ha='center', fontweight='bold', color='red')
    else:
        ax2.text(0, max_chsh + 0.1, f'{max_chsh:.3f}\n({approach_percentage:.1f}%)', 
                ha='center', fontweight='bold', color='orange')
    
    # Panel C: CHSH distribution histogram
    ax3.hist(chsh_values, bins=min(8, len(chsh_values)), alpha=0.7, 
             color='steelblue', edgecolor='black')
    ax3.axvline(np.mean(chsh_values), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(chsh_values):.3f}')
    ax3.axvline(2.0, color='orange', linestyle='--', linewidth=2, label='Classical Limit')
    
    ax3.set_xlabel('CHSH Value', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('C. CHSH Value Distribution', fontweight='bold')
    ax3.legend()
    
    # Panel D: Statistical analysis
    mean_chsh = np.mean(chsh_values)
    std_chsh = np.std(chsh_values)
    
    # t-test against classical limit
    if len(chsh_values) > 1:
        t_stat, p_value = stats.ttest_1samp(chsh_values, 2.0)
    else:
        t_stat, p_value = 0, 1
    
    # Bar chart comparing observed vs limits
    categories = ['Observed\nMean', 'Classical\nLimit']
    values = [mean_chsh, 2.0]
    colors_d = ['red' if mean_chsh > 2.0 else 'orange' if mean_chsh > 1.5 else 'steelblue', 'red']
    
    bars_d = ax4.bar(categories, values, color=colors_d, alpha=0.7, edgecolor='black')
    
    # Add error bar for observed mean
    if len(chsh_values) > 1:
        ax4.errorbar(0, mean_chsh, yerr=std_chsh, color='black', capsize=5, capthick=2)
    
    ax4.set_ylabel('CHSH Value', fontweight='bold')
    ax4.set_title(f'D. Statistical Comparison\n(t={t_stat:.2f}, p={p_value:.3f})', fontweight='bold')
    
    # Add significance annotation
    if mean_chsh > 2.0:
        ax4.text(0.5, max(values) * 1.1, 
                f'QUANTUM\nVIOLATION!', ha='center', 
                fontweight='bold', color='red')
    elif mean_chsh > 1.5:
        ax4.text(0.5, max(values) * 1.1, 
                f'Strong Classical\nCorrelations!', ha='center', 
                fontweight='bold', color='orange')
    
    plt.suptitle(f'Bell Inequality Analysis: {commodity_set.title()} - {period_name.replace("_", " ")}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f'Figure2_BellAnalysis_{commodity_set}_{period_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Saved: {filename}")
    print(f"📊 Highest CHSH value: {max_chsh:.3f} ({approach_percentage:.1f}% toward quantum limit)")
    plt.show()

def create_figure_3_statistics(events, bell_results, period_name, commodity_set):
    """Create Figure 3: Statistical analysis"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Correlation significance
    correlations = events['correlation'].values
    t_stat, p_value = stats.ttest_1samp(correlations, 0)
    
    n_bins = min(30, len(correlations) // 5 + 5)
    n, bins, patches = ax1.hist(correlations, bins=n_bins, alpha=0.7, 
                               color='#2E8B57', edgecolor='black')
    
    # Color significant correlations
    for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
        if bin_left > 0.7:
            patch.set_facecolor('red')
            patch.set_alpha(0.9)
    
    ax1.axvline(correlations.mean(), color='blue', linestyle='-', linewidth=2,
               label=f'Mean: {correlations.mean():.3f}')
    ax1.axvline(CORRELATION_THRESHOLD, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {CORRELATION_THRESHOLD}')
    
    ax1.set_xlabel('Correlation Coefficient', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title(f'A. Correlation Significance (p={p_value:.2e})', fontweight='bold')
    ax1.legend()
    
    # Panel B: Temporal distribution
    events_sorted = events.sort_values('timestamp')
    events_sorted['hour'] = events_sorted['timestamp'].dt.hour
    hourly_counts = events_sorted.groupby('hour').size()
    
    ax2.bar(hourly_counts.index, hourly_counts.values, 
           color='#DAA520', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Hour of Day', fontweight='bold')
    ax2.set_ylabel('Event Count', fontweight='bold')
    ax2.set_title('B. Temporal Distribution', fontweight='bold')
    
    # Panel C: Effect size analysis
    effect_size = correlations.mean() / correlations.std() if correlations.std() > 0 else 0
    
    # Power analysis
    sample_sizes = np.arange(10, min(1000, len(correlations)*5), 50)
    power_values = []
    
    for n in sample_sizes:
        se = correlations.std() / np.sqrt(n) if correlations.std() > 0 else 0.01
        z_score = correlations.mean() / se if se > 0 else 0
        power = 1 - stats.norm.cdf(1.96 - abs(z_score))
        power_values.append(power)
    
    ax3.plot(sample_sizes, power_values, linewidth=2, color='#4682B4')
    ax3.axhline(y=0.8, color='red', linestyle='--', linewidth=2,
               label='80% Power Threshold')
    ax3.axvline(len(correlations), color='blue', linestyle='-', linewidth=2,
               label=f'Current N: {len(correlations)}')
    
    ax3.set_xlabel('Sample Size', fontweight='bold')
    ax3.set_ylabel('Statistical Power', fontweight='bold')
    ax3.set_title(f'C. Power Analysis (Effect Size: {effect_size:.3f})', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Summary metrics
    metrics = ['Mean\nCorrelation', 'Max\nCorrelation', f'Events\n> {CORRELATION_THRESHOLD}', 'Effect\nSize']
    values = [correlations.mean(), 
             correlations.max(),
             (correlations > CORRELATION_THRESHOLD).sum() / len(correlations),
             effect_size]
    
    bars = ax4.bar(metrics, values, 
                  color=['#2E8B57', '#DAA520', '#4682B4', '#DC143C'], 
                  alpha=0.7, edgecolor='black')
    
    ax4.set_ylabel('Metric Value', fontweight='bold')
    ax4.set_title('D. Summary Metrics', fontweight='bold')
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'Statistical Analysis: {commodity_set.title()} - {period_name.replace("_", " ")}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f'Figure3_Statistics_{commodity_set}_{period_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Saved: {filename}")
    plt.show()

def create_summary_table(data, events, bell_results, period_name, commodity_set):
    """Create enhanced summary table"""
    
    # Calculate summary statistics
    total_points = sum(len(df) for df in data.values())
    n_commodities = len(data)
    n_events = len(events)
    
    if len(events) > 0:
        mean_corr = events['correlation'].mean()
        max_corr = events['correlation'].max()
        high_corr_events = (events['correlation'] > CORRELATION_THRESHOLD).sum()
    else:
        mean_corr = max_corr = high_corr_events = 0
    
    # Bell results
    if bell_results:
        chsh_values = [res.get('chsh_value', 0) for res in bell_results.values() 
                      if isinstance(res, dict) and res.get('chsh_value', 0) > 0]
        max_chsh = max(chsh_values) if chsh_values else 0
        mean_chsh = np.mean(chsh_values) if chsh_values else 0
        violations = sum(1 for res in bell_results.values() 
                        if isinstance(res, dict) and res.get('chsh_violation', False))
    else:
        max_chsh = mean_chsh = violations = 0
    
    # Create summary table
    summary_data = {
        'Metric': [
            'Analysis Period',
            'Commodity Set',
            'Data Interval', 
            'Data Points Collected',
            'Commodities Analyzed',
            'Correlation Threshold',
            'Coordination Events',
            'Mean Correlation',
            'Max Correlation', 
            f'Events > {CORRELATION_THRESHOLD} Correlation',
            'Bell Tests Performed',
            'Mean CHSH Value',
            'Max CHSH Value',
            'Bell Violations',
            'Quantum Signature'
        ],
        'Value': [
            f"{ANALYSIS_START_DATE} to {ANALYSIS_END_DATE}",
            commodity_set.title(),
            DATA_INTERVAL,
            f"{total_points:,}",
            f"{n_commodities}",
            f"{CORRELATION_THRESHOLD}",
            f"{n_events:,}",
            f"{mean_corr:.4f}",
            f"{max_corr:.4f}",
            f"{high_corr_events:,}",
            f"{len(bell_results) if bell_results else 0}",
            f"{mean_chsh:.4f}",
            f"{max_chsh:.4f}",
            f"{violations}",
            "🚨 DETECTED" if violations > 0 or max_chsh > 2.0 else "✗ None"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save table
    filename = f'Summary_Table_{commodity_set}_{period_name}.csv'
    summary_df.to_csv(filename, index=False)
    print(f"📋 Saved: {filename}")
    
    # Display table
    print("\n" + "="*70)
    print(f"SUMMARY TABLE: {commodity_set.title()} - {period_name.replace('_', ' ')}")
    print("="*70)
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']:<25}: {row['Value']}")
    print("="*70)

#============================================================================
# EASY COMMODITY SET SWITCHING FUNCTION
#============================================================================

def show_available_commodity_sets():
    """Show all available commodity sets for easy reference"""
    analyzer = QuantumCommodityAnalyzer()
    analyzer.print_commodity_sets()
    
def quick_test_commodity_set(commodity_set, days_back=30):
    """Quick test of a commodity set with recent data"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    print(f"🧪 Quick test: {commodity_set} commodities")
    print(f"📅 Period: {start_date} to {end_date}")
    
    analyzer = QuantumCommodityAnalyzer(commodity_set)
    data = analyzer.collect_high_frequency_data(start_date, end_date, '1d')
    
    print(f"📊 Result: {len(data)} commodities collected")
    for name, df in data.items():
        print(f"   {name}: {len(df)} data points")
    
    return len(data) >= 2

#============================================================================
# MAIN EXECUTION - Run this in Spyder!
#============================================================================

if __name__ == "__main__":
    
    print("🎯 AVAILABLE COMMODITY SETS:")
    show_available_commodity_sets()
    
    print(f"\n📊 CURRENT CONFIGURATION:")
    print(f"   Commodity Set: {COMMODITY_SET}")
    print(f"   Period: {ANALYSIS_START_DATE} to {ANALYSIS_END_DATE}")
    print(f"   Correlation Threshold: {CORRELATION_THRESHOLD}")
    print(f"   Data Interval: {DATA_INTERVAL}")
    print()
    
    # Run the main analysis
    results = run_quantum_analysis()
    
    if results:
        print(f"\n🎯 FINAL RESULTS SUMMARY:")
        print(f"   Commodity Set: {results['commodity_set']}")
        print(f"   Period: {results['period_name']}")
        print(f"   Coordination Events: {len(results['events']):,}")
        print(f"   Bell Violations: {results['total_violations']}")
        print(f"   Mean CHSH Value: {results.get('mean_chsh', 0):.3f}")
        print(f"   Max CHSH Value: {results['max_chsh']:.3f}")
        
        if results['max_chsh'] > 2.0:
            violation_percent = ((results['max_chsh'] - 2.0) / (2.828 - 2.0)) * 100
            print(f"\n🚨 BREAKTHROUGH: Bell inequality violation detected!")
            print(f"⚛️ {violation_percent:.1f}% toward quantum bound!")
            print(f"🔬 Evidence for quantum-like coordination in {results['commodity_set']} markets!")
        elif results['max_chsh'] > 1.8:
            classical_percent = (results['max_chsh'] / 2.0) * 100
            print(f"\n⚡ Very strong classical correlations detected!")
            print(f"📊 {classical_percent:.1f}% toward classical bound!")
            print(f"🎯 {results['commodity_set'].title()} commodities show significant coordination!")
        else:
            print(f"\n📊 Classical coordination confirmed")
            print(f"✅ Framework successfully validated on {results['commodity_set']} data")
        
        print(f"\n💡 TO TRY DIFFERENT COMMODITIES:")
        print(f"   1. Change COMMODITY_SET = '{COMMODITY_SET}' to another option at the top")
        print(f"   2. Available options: 'agricultural', 'mixed', 'energy', 'metals', 'etfs'")
        print(f"   3. Run the script again!")
        print(f"\n💡 TO GET MORE EVENTS:")
        print(f"   1. Lower CORRELATION_THRESHOLD (try 0.3)")
        print(f"   2. Use longer time period") 
        print(f"   3. Try 'mixed' commodity set (often shows more correlations)")
        
    else:
        print(f"\n❌ Analysis failed with current settings")
        print(f"💡 Try:")
        print(f"   • Different commodity set (try 'etfs' for better data availability)")
        print(f"   • Longer time period (6+ months)")
        print(f"   • Lower correlation threshold (0.3)")

# END OF SCRIPT - Everything above runs automatically in Spyder!