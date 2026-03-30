import pandas as pd
import numpy as np
from pygam import LogisticGAM
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt

# Load data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# Select a few important features (based on GLM p-values and intuition)
features = [
    'ExternalRiskEstimate',
    'NumSatisfactoryTrades',
    'PercentTradesNeverDelq',
    'MSinceMostRecentDelq',
    'MSinceMostRecentInqexcl7days',
    'NumInqLast6M',
    'NetFractionRevolvingBurden',
    'NumRevolvingTradesWBalance'
]

X_train_gam = X_train[features].values
X_test_gam = X_test[features].values

print("Fitting GAM with features:", features)
gam = LogisticGAM().fit(X_train_gam, y_train)

# Predict probabilities on test set
y_pred_proba = gam.predict_proba(X_test_gam)

# Evaluate
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

print("\nGAM Performance on test set:")
print(f"AUC: {auc:.4f}")
print(f"Log-loss: {logloss:.4f}")

# Save predictions for later comparison
pd.Series(y_pred_proba).to_csv("gam_preds.csv", index=False, header=False)

# Plot partial dependence for each feature
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(features):
    XX = gam.generate_X_grid(term=i)
    ax = axes[i]
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.set_title(f"Partial dependence of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Effect on log-odds")
plt.tight_layout()
plt.savefig("gam_partial_dependence.png")
plt.show()

print("Partial dependence plots saved as 'gam_partial_dependence.png'")