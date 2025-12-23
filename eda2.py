"""
Comprehensive Exploratory Data Analysis (EDA) Report
Telco Customer Churn Dataset
All visualizations will be saved as PNG files
Microsoft Color Palette Applied
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ================================
# MICROSOFT COLOR PALETTE
# ================================
MS_BLUE = '#0078D4'
MS_LIGHT_BLUE = '#50E6FF'
MS_GREEN = '#107C10'
MS_ORANGE = '#D83B01'
MS_RED = '#A4262C'
MS_PURPLE = '#5C2D91'
MS_GRAY = '#605E5C'
MS_CHURN_COLORS = [MS_GREEN, MS_RED]  # No Churn, Churn

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ================================
# OUTPUT DIRECTORY
# ================================
output_dir = Path("eda_graphs")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - TELCO CUSTOMER CHURN DATASET")
print("=" * 80)

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Handle TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# ================================
# CHURN DISTRIBUTION
# ================================
plt.figure(figsize=(10, 6))
churn_counts = df['Churn'].value_counts()
plt.pie(
    churn_counts.values,
    labels=churn_counts.index,
    autopct='%1.1f%%',
    colors=MS_CHURN_COLORS,
    startangle=90,
    textprops={'fontsize': 12, 'weight': 'bold'}
)
plt.title("Customer Churn Distribution", fontsize=16, fontweight='bold')
plt.savefig(output_dir / "01_churn_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# CHURN COUNT
# ================================
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='Churn', palette=MS_CHURN_COLORS)
plt.title("Customer Churn Count", fontsize=16, fontweight='bold')
for p in ax.patches:
    ax.annotate(int(p.get_height()),
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontweight='bold')
plt.savefig(output_dir / "02_churn_count.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# GENDER VS CHURN
# ================================
pd.crosstab(df['gender'], df['Churn']).plot(
    kind='bar', color=MS_CHURN_COLORS, edgecolor='black'
)
plt.title("Gender vs Churn", fontsize=16, fontweight='bold')
plt.xticks(rotation=0)
plt.savefig(output_dir / "03_gender_churn.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# SENIOR CITIZEN VS CHURN
# ================================
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
pd.crosstab(df['SeniorCitizen'], df['Churn']).plot(
    kind='bar', color=MS_CHURN_COLORS, edgecolor='black'
)
plt.title("Senior Citizen vs Churn", fontsize=16, fontweight='bold')
plt.xticks(rotation=0)
plt.savefig(output_dir / "04_senior_churn.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# TENURE DISTRIBUTION
# ================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df['tenure'], bins=50, color=MS_BLUE, edgecolor='black', alpha=0.7)
axes[0].set_title("Tenure Distribution")

df.boxplot(column='tenure', by='Churn', ax=axes[1], grid=True)
plt.suptitle("")
plt.savefig(output_dir / "05_tenure.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# MONTHLY CHARGES
# ================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df['MonthlyCharges'], bins=50, color=MS_PURPLE, edgecolor='black', alpha=0.7)
axes[0].set_title("Monthly Charges Distribution")

df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1], grid=True)
plt.suptitle("")
plt.savefig(output_dir / "06_monthly_charges.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# TOTAL CHARGES
# ================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df['TotalCharges'], bins=50, color=MS_ORANGE, edgecolor='black', alpha=0.7)
axes[0].set_title("Total Charges Distribution")

df.boxplot(column='TotalCharges', by='Churn', ax=axes[1], grid=True)
plt.suptitle("")
plt.savefig(output_dir / "07_total_charges.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# CONTRACT TYPE
# ================================
pd.crosstab(df['Contract'], df['Churn']).plot(
    kind='bar', color=MS_CHURN_COLORS, edgecolor='black'
)
plt.title("Contract Type vs Churn", fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.savefig(output_dir / "08_contract_churn.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# PAYMENT METHOD
# ================================
pd.crosstab(df['PaymentMethod'], df['Churn']).plot(
    kind='bar', color=MS_CHURN_COLORS, edgecolor='black'
)
plt.title("Payment Method vs Churn", fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.savefig(output_dir / "09_payment_method_churn.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# INTERNET SERVICE
# ================================
pd.crosstab(df['InternetService'], df['Churn']).plot(
    kind='bar', color=MS_CHURN_COLORS, edgecolor='black'
)
plt.title("Internet Service vs Churn", fontsize=16, fontweight='bold')
plt.savefig(output_dir / "10_internet_service_churn.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# CORRELATION HEATMAP
# ================================
numeric_cols = df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 8))
sns.heatmap(
    numeric_cols.corr(),
    annot=True,
    cmap=sns.light_palette(MS_BLUE, as_cmap=True),
    fmt=".2f",
    square=True
)
plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
plt.savefig(output_dir / "11_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# TENURE VS MONTHLY CHARGES
# ================================
plt.figure(figsize=(12, 8))
plt.scatter(df[df['Churn'] == 'No']['tenure'],
            df[df['Churn'] == 'No']['MonthlyCharges'],
            color=MS_GREEN, alpha=0.5, label='No Churn')

plt.scatter(df[df['Churn'] == 'Yes']['tenure'],
            df[df['Churn'] == 'Yes']['MonthlyCharges'],
            color=MS_RED, alpha=0.5, label='Churn')

plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Tenure vs Monthly Charges by Churn", fontsize=16, fontweight='bold')
plt.legend()
plt.savefig(output_dir / "12_tenure_vs_monthly.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# CHURN RATE BY CONTRACT
# ================================
churn_rate = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
churn_rate.plot(kind='bar', color=MS_RED, edgecolor='black')
plt.title("Churn Rate by Contract Type", fontsize=16, fontweight='bold')
plt.ylabel("Churn Rate (%)")
plt.savefig(output_dir / "13_churn_rate_contract.png", dpi=300, bbox_inches='tight')
plt.close()

print("=" * 80)
print("EDA COMPLETED SUCCESSFULLY")
print(f"Total graphs saved: {len(list(output_dir.glob('*.png')))}")
print("=" * 80)
