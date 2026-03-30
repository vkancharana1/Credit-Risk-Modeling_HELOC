import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

# We'll model ExternalRiskEstimate as the target
y_score = X_train['ExternalRiskEstimate']
# Use all other features as predictors
X_score = X_train.drop('ExternalRiskEstimate', axis=1)

# Add constant
X_score_const = sm.add_constant(X_score)

# Quantiles to model
quantiles = [0.1, 0.5, 0.9]
models_qr = {}

print("Fitting Quantile Regression models...")
for q in quantiles:
    print(f"\nQuantile: {q}")
    mod = sm.QuantReg(y_score, X_score_const)
    res = mod.fit(q=q)
    models_qr[q] = res
    print(res.summary())

# Extract coefficients for comparison
coeffs = pd.DataFrame({f'q{int(q*100)}': res.params for q, res in models_qr.items()}).T
print("\nCoefficients across quantiles:")
print(coeffs)

# Save coefficients for later
coeffs.to_csv("quantile_coefficients.csv")

# Plot coefficients for a few important features
important_features = ['NumInqLast6M', 'NetFractionRevolvingBurden', 'NumRevolvingTradesWBalance']
for feat in important_features:
    if feat in coeffs.columns:
        plt.figure()
        plt.plot([10, 50, 90], coeffs[feat], marker='o')
        plt.xlabel('Quantile')
        plt.ylabel(f'Coefficient for {feat}')
        plt.title(f'Quantile regression coefficients: {feat}')
        plt.savefig(f'quantile_coef_{feat}.png')
        plt.show()

print("\nQuantile regression complete. Coefficients saved to 'quantile_coefficients.csv'")