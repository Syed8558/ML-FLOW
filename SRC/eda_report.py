"""
Comprehensive Exploratory Data Analysis (EDA) Report
Telco Customer Churn Dataset
All visualizations are saved as PNG files
Consistent color palette applied across ALL graphs
"""

# =============================================================================
# IMPORTS & SETTINGS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL STYLE & COLOR PALETTE (SINGLE SOURCE OF TRUTH)
# =============================================================================
sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

PALETTE = {
    "no_churn": "#2ecc71",     # Green
    "churn": "#e74c3c",        # Red
    "primary": "#3498db",      # Blue
    "secondary": "#9b59b6",    # Purple
    "accent": "#f39c12"        # Orange
}

CHURN_COLORS = [PALETTE["no_churn"], PALETTE["churn"]]
sns.set_palette(CHURN_COLORS)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
output_dir = Path("eda_graphs")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - TELCO CUSTOMER CHURN DATASET")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# =============================================================================
# DATA CLEANING
# =============================================================================
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# =============================================================================
# 1. CHURN DISTRIBUTION (PIE)
# =============================================================================
plt.figure()
churn_counts = df["Churn"].value_counts()
plt.pie(
    churn_counts.values,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    colors=CHURN_COLORS,
    startangle=90
)
plt.title("Customer Churn Distribution", fontweight="bold")
plt.savefig(output_dir / "01_churn_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# 2. CHURN COUNT (BAR)
# =============================================================================
plt.figure()
ax = sns.countplot(data=df, x="Churn", palette=CHURN_COLORS)
for p in ax.patches:
    ax.annotate(int(p.get_height()), 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha="center", va="bottom", fontweight="bold")
plt.title("Customer Churn Count", fontweight="bold")
plt.savefig(output_dir / "02_churn_count.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# 3. GENDER VS CHURN
# =============================================================================
pd.crosstab(df["gender"], df["Churn"]).plot(
    kind="bar", color=CHURN_COLORS, edgecolor="black"
)
plt.title("Gender vs Churn", fontweight="bold")
plt.xticks(rotation=0)
plt.savefig(output_dir / "03_gender_churn.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# 4. SENIOR CITIZEN VS CHURN
# =============================================================================
pd.crosstab(df["SeniorCitizen"], df["Churn"]).plot(
    kind="bar", color=CHURN_COLORS, edgecolor="black"
)
plt.title("Senior Citizen vs Churn", fontweight="bold")
plt.xticks(rotation=0)
plt.savefig(output_dir / "04_senior_citizen_churn.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# 5. PARTNER & DEPENDENTS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pd.crosstab(df["Partner"], df["Churn"]).plot(
    kind="bar", ax=axes[0], color=CHURN_COLORS, edgecolor="black"
)
axes[0].set_title("Partner vs Churn")

pd.crosstab(df["Dependents"], df["Churn"]).plot(
    kind="bar", ax=axes[1], color=CHURN_COLORS, edgecolor="black"
)
axes[1].set_title("Dependents vs Churn")

plt.tight_layout()
plt.savefig(output_dir / "05_partner_dependents.png", dpi=300)
plt.close()

# =============================================================================
# 6. TENURE DISTRIBUTION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df["tenure"], bins=50, color=PALETTE["primary"], edgecolor="black")
axes[0].set_title("Tenure Distribution")

df.boxplot(column="tenure", by="Churn", ax=axes[1], grid=True)
plt.suptitle("")
plt.tight_layout()
plt.savefig(output_dir / "06_tenure_distribution.png", dpi=300)
plt.close()

# =============================================================================
# 7. MONTHLY CHARGES
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df["MonthlyCharges"], bins=50, color=PALETTE["secondary"], edgecolor="black")
axes[0].set_title("Monthly Charges Distribution")

df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[1], grid=True)
plt.suptitle("")
plt.tight_layout()
plt.savefig(output_dir / "07_monthly_charges.png", dpi=300)
plt.close()

# =============================================================================
# 8. TOTAL CHARGES
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df["TotalCharges"].dropna(), bins=50,
             color=PALETTE["accent"], edgecolor="black")
axes[0].set_title("Total Charges Distribution")

df.dropna(subset=["TotalCharges"]).boxplot(
    column="TotalCharges", by="Churn", ax=axes[1], grid=True
)
plt.suptitle("")
plt.tight_layout()
plt.savefig(output_dir / "08_total_charges.png", dpi=300)
plt.close()

# =============================================================================
# 9. CONTRACT TYPE
# =============================================================================
pd.crosstab(df["Contract"], df["Churn"]).plot(
    kind="bar", color=CHURN_COLORS, edgecolor="black"
)
plt.title("Contract Type vs Churn", fontweight="bold")
plt.xticks(rotation=45)
plt.savefig(output_dir / "09_contract_churn.png", dpi=300)
plt.close()

# =============================================================================
# 10. PAYMENT METHOD
# =============================================================================
pd.crosstab(df["PaymentMethod"], df["Churn"]).plot(
    kind="bar", color=CHURN_COLORS, edgecolor="black"
)
plt.title("Payment Method vs Churn", fontweight="bold")
plt.xticks(rotation=45)
plt.savefig(output_dir / "10_payment_method_churn.png", dpi=300)
plt.close()

# =============================================================================
# 11. INTERNET SERVICE
# =============================================================================
pd.crosstab(df["InternetService"], df["Churn"]).plot(
    kind="bar", color=CHURN_COLORS, edgecolor="black"
)
plt.title("Internet Service vs Churn", fontweight="bold")
plt.savefig(output_dir / "11_internet_service_churn.png", dpi=300)
plt.close()

# =============================================================================
# 12. SERVICE FEATURES
# =============================================================================
features = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "PaperlessBilling"
]

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for i, feature in enumerate(features):
    pd.crosstab(df[feature], df["Churn"]).plot(
        kind="bar", ax=axes[i], color=CHURN_COLORS, edgecolor="black"
    )
    axes[i].set_title(feature)
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "12_service_features.png", dpi=300)
plt.close()

# =============================================================================
# 13. CORRELATION HEATMAP
# =============================================================================
numeric_cols = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontweight="bold")
plt.savefig(output_dir / "13_correlation_heatmap.png", dpi=300)
plt.close()

# =============================================================================
# 14. TENURE VS MONTHLY CHARGES
# =============================================================================
plt.figure()
plt.scatter(df[df["Churn"] == "No"]["tenure"],
            df[df["Churn"] == "No"]["MonthlyCharges"],
            color=PALETTE["no_churn"], alpha=0.5, label="No Churn")
plt.scatter(df[df["Churn"] == "Yes"]["tenure"],
            df[df["Churn"] == "Yes"]["MonthlyCharges"],
            color=PALETTE["churn"], alpha=0.5, label="Churn")
plt.legend()
plt.title("Tenure vs Monthly Charges", fontweight="bold")
plt.savefig(output_dir / "14_tenure_vs_monthly_charges.png", dpi=300)
plt.close()

# =============================================================================
# 15. CHURN RATE
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

(df.groupby("Contract")["Churn"]
 .apply(lambda x: (x == "Yes").mean() * 100)
 .plot(kind="bar", ax=axes[0], color=PALETTE["churn"], edgecolor="black"))

(df.groupby("PaymentMethod")["Churn"]
 .apply(lambda x: (x == "Yes").mean() * 100)
 .plot(kind="bar", ax=axes[1], color=PALETTE["churn"], edgecolor="black"))

plt.tight_layout()
plt.savefig(output_dir / "15_churn_rates.png", dpi=300)
plt.close()

# =============================================================================
# COMPLETED
# =============================================================================
print("=" * 80)
print("EDA COMPLETED SUCCESSFULLY")
print("All plots saved in 'eda_graphs/' directory")
print("=" * 80)
