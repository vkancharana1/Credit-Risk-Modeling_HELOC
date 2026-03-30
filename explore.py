import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")

# Load the dataset
df = pd.read_csv("heloc_dataset.csv")

# Display basic info
print("="*50)
print("Shape:", df.shape)
print("="*50)
print("\nColumns:")
print(df.columns.tolist())
print("\nData types and non-null counts:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Target variable distribution
print("\nTarget variable distribution:")
print(df['RiskPerformance'].value_counts())

# Look at special codes -7, -8, -9 (they appear as numbers in the data)
# We'll check the first few columns for unique values
print("\nUnique values in first 3 columns (to see -7, -8, -9):")
for col in df.columns[:3]:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

# Missing values? (currently special codes are not NaN, so we'll see 0 missing)
print("\nMissing values per column (currently no NaNs):")
print(df.isnull().sum())

# Plot target distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['RiskPerformance'])
plt.title('Target Distribution (Bad vs Good)')
plt.savefig('target_dist.png')
plt.show()

# Example: ExternalRiskEstimate vs target
plt.figure(figsize=(8,5))
sns.boxplot(x=df['RiskPerformance'], y=df['ExternalRiskEstimate'])
plt.title('ExternalRiskEstimate by RiskPerformance')
plt.savefig('external_risk_box.png')
plt.show()

# Optional: histogram of ExternalRiskEstimate
plt.figure(figsize=(8,5))
sns.histplot(df['ExternalRiskEstimate'], bins=30, kde=True)
plt.title('Distribution of ExternalRiskEstimate')
plt.savefig('external_risk_hist.png')
plt.show()

print("\nBasic exploration complete. Plots saved as PNG files.")