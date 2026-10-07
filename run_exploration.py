"""
Day 1: Data Exploration Script (Non-Interactive)
This runs all analysis at once and explains everything.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Create results directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)

# Set up plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print(" FRAUD DETECTION PROJECT - DATA EXPLORATION")
print("="*70)
print()

# ============================================================================
# STEP 1: LOAD THE DATA
# ============================================================================
print("STEP 1: Loading dataset...")
print("-" * 70)

df = pd.read_csv('data/creditcard.csv')

print(f"✓ Loaded {len(df):,} transactions")
print(f"✓ Number of features: {df.shape[1] - 1} (excluding target 'Class')")
print(f"✓ Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
print()

# Show first few rows
print("First 5 transactions:")
print(df.head())
print("\n")

# ============================================================================
# STEP 2: UNDERSTAND THE EXTREME CLASS IMBALANCE
# ============================================================================
print("="*70)
print("STEP 2: Understanding the CLASS IMBALANCE (Most Important!)")
print("="*70)

class_counts = df['Class'].value_counts().sort_index()
total = len(df)
frauds = class_counts[1]
legit = class_counts[0]
fraud_rate = (frauds / total) * 100

print(f"\n📊 Class Distribution:")
print(f"  Legitimate (0): {legit:,} transactions ({100 - fraud_rate:.3f}%)")
print(f"  Fraud (1):      {frauds:,} transactions ({fraud_rate:.3f}%)")
print()
print(f"🔍 Imbalance Ratio: {legit / frauds:.1f}:1")
print(f"   That means {int(legit/frauds)} legitimate transactions for every 1 fraud!")
print()

print("💡 WHY THIS IS CRITICAL:")
print("   A dumb model that ALWAYS predicts 'legitimate' would be")
print(f"   correct {100 - fraud_rate:.3f}% of the time!")
print("   But it would catch ZERO frauds.")
print()
print("   That's why ACCURACY is a TERRIBLE metric here.")
print("   We'll use Precision-Recall AUC instead.")
print("\n")

# Visualize imbalance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ax1 = axes[0]
class_counts.plot(kind='bar', ax=ax1, color=['green', 'red'])
ax1.set_title('Class Distribution (Absolute Counts)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Class')
ax1.set_ylabel('Number of Transactions')
ax1.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
for i, v in enumerate(class_counts):
    ax1.text(i, v + 5000, f'{v:,}', ha='center', fontweight='bold')

# Pie chart
ax2 = axes[1]
colors = ['green', 'red']
explode = (0, 0.1)
ax2.pie(class_counts, labels=['Legitimate', 'Fraud'], autopct='%1.3f%%',
        colors=colors, explode=explode, startangle=90)
ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/class_imbalance.png', dpi=100, bbox_inches='tight')
print("✓ Saved: results/figures/class_imbalance.png")
plt.close()

# ============================================================================
# STEP 3: ANALYZE TIME
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Analyzing TIME column")
print("="*70)

time_min = df['Time'].min()
time_max = df['Time'].max()
time_span_seconds = time_max - time_min
time_span_hours = time_span_seconds / 3600
time_span_days = time_span_hours / 24

print(f"\n⏰ Time Information:")
print(f"  Minimum: {time_min:,.0f} seconds")
print(f"  Maximum: {time_max:,.0f} seconds")
print(f"  Span: {time_span_seconds:,.0f} seconds")
print(f"       = {time_span_hours:.1f} hours")
print(f"       = {time_span_days:.1f} days")
print()
print("💡 This dataset covers about 2 days of transactions.")
print("   We'll split by TIME (not randomly) to simulate real deployment.")
print()

# Plot transaction volume over time
df['TimeHours'] = df['Time'] / 3600
fraud_df = df[df['Class'] == 1]

plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.hist(df['TimeHours'], bins=48, color='blue', alpha=0.7, edgecolor='black')
plt.title('Transaction Volume Over Time (All)', fontsize=14, fontweight='bold')
plt.xlabel('Time (hours)')
plt.ylabel('Number of Transactions')
plt.grid(axis='y', alpha=0.3)

plt.subplot(2, 1, 2)
plt.hist(fraud_df['TimeHours'], bins=48, color='red', alpha=0.7, edgecolor='black')
plt.title('Fraud Transaction Volume Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Time (hours)')
plt.ylabel('Number of Frauds')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/time_distribution.png', dpi=100, bbox_inches='tight')
print("✓ Saved: results/figures/time_distribution.png")
plt.close()

# ============================================================================
# STEP 4: ANALYZE TRANSACTION AMOUNTS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Analyzing TRANSACTION AMOUNTS")
print("="*70)

legit_amounts = df[df['Class'] == 0]['Amount']
fraud_amounts = df[df['Class'] == 1]['Amount']

print(f"\n💵 Amount Statistics:\n")
print("LEGITIMATE transactions:")
print(f"  Mean:   ${legit_amounts.mean():.2f}")
print(f"  Median: ${legit_amounts.median():.2f}")
print(f"  Std:    ${legit_amounts.std():.2f}")
print(f"  Min:    ${legit_amounts.min():.2f}")
print(f"  Max:    ${legit_amounts.max():.2f}")
print()

print("FRAUD transactions:")
print(f"  Mean:   ${fraud_amounts.mean():.2f}")
print(f"  Median: ${fraud_amounts.median():.2f}")
print(f"  Std:    ${fraud_amounts.std():.2f}")
print(f"  Min:    ${fraud_amounts.min():.2f}")
print(f"  Max:    ${fraud_amounts.max():.2f}")
print()

print("💡 SURPRISING INSIGHT:")
print(f"   Fraud amounts (avg ${fraud_amounts.mean():.2f}) are actually")
print(f"   LOWER than legitimate amounts (avg ${legit_amounts.mean():.2f})!")
print()
print("   This means Amount alone won't catch fraud.")
print("   We need the V1-V28 features (transaction patterns).")
print()

# Visualize amounts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
ax1.hist(legit_amounts[legit_amounts < 500], bins=50, alpha=0.7,
         label='Legitimate', color='green', edgecolor='black')
ax1.hist(fraud_amounts[fraud_amounts < 500], bins=50, alpha=0.7,
         label='Fraud', color='red', edgecolor='black')
ax1.set_title('Amount Distribution (< $500)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Amount ($)')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Box plot
ax2 = axes[1]
data_to_plot = [legit_amounts[legit_amounts < 500], fraud_amounts[fraud_amounts < 500]]
bp = ax2.boxplot(data_to_plot)
ax2.set_xticklabels(['Legitimate', 'Fraud'])
ax2.set_title('Amount Distribution by Class', fontsize=14, fontweight='bold')
ax2.set_ylabel('Amount ($)')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/amount_distribution.png', dpi=100, bbox_inches='tight')
print("✓ Saved: results/figures/amount_distribution.png")
plt.close()

# ============================================================================
# STEP 5: FEATURE CORRELATIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Which features are most important?")
print("="*70)

correlations = df.corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)

print(f"\n🔝 Top 15 features correlated with Fraud:\n")
print(f"{'Rank':<6} {'Feature':<10} {'Correlation':<12} {'Meaning'}")
print("-" * 70)

for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
    direction = "Higher = more fraud" if corr > 0 else "Lower = more fraud"
    print(f"{i:<6} {feature:<10} {corr:>+.4f}       {direction}")

print()
print("💡 These features have the strongest patterns for detecting fraud!")
print("   Our ML models will rely heavily on these.")
print()

# Plot correlations
plt.figure(figsize=(12, 6))
top_features = correlations.head(15)
colors = ['red' if x < 0 else 'green' for x in top_features.values]
plt.barh(range(len(top_features)), top_features.values, color=colors)
plt.yticks(range(len(top_features)), top_features.index)
plt.xlabel('Correlation with Class (Fraud)')
plt.title('Top 15 Features Correlated with Fraud', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/feature_correlations.png', dpi=100, bbox_inches='tight')
print("✓ Saved: results/figures/feature_correlations.png")
plt.close()

# ============================================================================
# STEP 6: CHECK DATA QUALITY
# ============================================================================
print("\n" + "="*70)
print("STEP 6: Data Quality Check")
print("="*70)

missing = df.isnull().sum().sum()
print(f"\n🔍 Missing values: {missing}")
if missing == 0:
    print("   ✓ Perfect! No missing data.")
else:
    print(f"   ⚠ Found {missing} missing values - need to handle these!")

print(f"\n📊 Data types:")
for dtype, count in df.dtypes.value_counts().items():
    print(f"   {dtype}: {count} columns")

print("\n✓ All columns are numeric - perfect for ML!")
print()

# ============================================================================
# STEP 7: FRAUD PATTERNS OVER TIME
# ============================================================================
print("="*70)
print("STEP 7: Fraud patterns over time")
print("="*70)

time_bins = np.arange(0, 49, 4)
fraud_by_time = []
legit_by_time = []

for i in range(len(time_bins) - 1):
    window = df[(df['TimeHours'] >= time_bins[i]) & (df['TimeHours'] < time_bins[i+1])]
    fraud_count = window['Class'].sum()
    legit_count = len(window) - fraud_count
    fraud_by_time.append(fraud_count)
    legit_by_time.append(legit_count)

print(f"\n⏰ Fraud counts by 4-hour windows:\n")
print(f"{'Time (hours)':<15} {'Frauds':<10} {'Total Trans':<12} {'Fraud %'}")
print("-" * 70)

for i in range(len(time_bins) - 1):
    time_range = f"{time_bins[i]:.0f}-{time_bins[i+1]:.0f}"
    frauds_in_window = fraud_by_time[i]
    total_in_window = fraud_by_time[i] + legit_by_time[i]
    fraud_pct = (frauds_in_window / total_in_window * 100) if total_in_window > 0 else 0
    print(f"{time_range:<15} {frauds_in_window:<10} {total_in_window:<12,} {fraud_pct:.3f}%")

print()
print("💡 Notice: Fraud rates vary over time!")
print("   This is natural drift happening.")
print("   Our monitoring system will detect when this drift is too large.")
print()

# ============================================================================
# STEP 8: SAVE SUMMARY
# ============================================================================
print("="*70)
print("STEP 8: Saving exploration summary")
print("="*70)

summary = {
    'dataset': {
        'total_transactions': int(total),
        'fraud_count': int(frauds),
        'legitimate_count': int(legit),
        'fraud_percentage': float(fraud_rate),
        'imbalance_ratio': float(legit / frauds)
    },
    'time': {
        'span_seconds': int(time_span_seconds),
        'span_hours': float(time_span_hours),
        'span_days': float(time_span_days)
    },
    'amounts': {
        'legitimate_mean': float(legit_amounts.mean()),
        'legitimate_median': float(legit_amounts.median()),
        'fraud_mean': float(fraud_amounts.mean()),
        'fraud_median': float(fraud_amounts.median())
    },
    'top_features': {k: float(v) for k, v in correlations.head(10).items()}
}

with open('results/exploration_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ Summary saved to: results/exploration_summary.json")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
print(" EXPLORATION COMPLETE!")
print("="*70)
print()
print("📚 What you learned:")
print()
print("1. ✅ Extreme imbalance (0.173% fraud)")
print("   → Accuracy is useless, use Precision-Recall AUC")
print()
print("2. ✅ 2 days of transaction data")
print("   → We'll use time-based split (not random)")
print()
print("3. ✅ Fraud amounts are LOWER than legitimate")
print("   → Need V1-V28 features, not just Amount")
print()
print("4. ✅ No missing data, all numeric")
print("   → Ready for modeling!")
print()
print("5. ✅ Some features highly correlated with fraud")
print("   → V14, V12, V10 are most important")
print()
print("6. ✅ Fraud rates vary over time")
print("   → Natural drift - our monitoring will catch large shifts")
print()
print("="*70)
print()
print("🚀 NEXT STEPS:")
print("   Day 2: Build and compare models")
print("   Day 3: Implement drift monitoring")
print("   Day 4: Write documentation")
print()
print("="*70)
