import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt

# Load the preprocessed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()   # convert 1-col df to Series
y_test = pd.read_csv("y_test.csv").squeeze()

print("Data loaded successfully.")
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Add constant (intercept) to the features
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit logistic regression
print("\nFitting GLM (Logistic Regression)...")
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# Print summary
print("\n" + "="*60)
print("Model Summary")
print("="*60)
print(result.summary())

# Predict probabilities on test set
y_pred_proba = result.predict(X_test_const)

# Evaluate
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

print("\n" + "="*60)
print("Performance on Test Set")
print("="*60)
print(f"AUC: {auc:.4f}")
print(f"Log-loss: {logloss:.4f}")

# Save predictions for later comparison
pd.Series(y_pred_proba).to_csv("glm_preds.csv", index=False, header=False)
print("\nPredictions saved to 'glm_preds.csv'")

# Optionally, plot the ROC curve (optional)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'GLM (AUC = {auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('glm_roc.png')
plt.show()
print("ROC curve saved as 'glm_roc.png'")