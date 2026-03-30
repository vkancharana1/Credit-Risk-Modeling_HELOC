import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load raw data
df = pd.read_csv("heloc_dataset.csv")
print("Raw data shape:", df.shape)

# Replace special values with NaN
special_vals = [-7, -8, -9]
df = df.replace(special_vals, np.nan)
print("After replacing special values, missing count per column:")
print(df.isnull().sum())

# Convert target to numeric: Bad -> 1, Good -> 0
df['RiskPerformance'] = df['RiskPerformance'].map({'Bad': 1, 'Good': 0})
print("\nTarget distribution after conversion:")
print(df['RiskPerformance'].value_counts())

# Impute missing values with median (for all numeric columns)
# Note: all features are numeric except target which is already numeric
df = df.fillna(df.median())
print("\nMissing values after imputation:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('RiskPerformance', axis=1)
y = df['RiskPerformance']

# Train/test split with stratification (to keep same proportion of Bad/Good)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training target distribution: {y_train.value_counts().to_dict()}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")

# Save the processed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False, header=True)
y_test.to_csv("y_test.csv", index=False, header=True)
df.to_csv("heloc_cleaned.csv", index=False)

print("\nFiles saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv, heloc_cleaned.csv")