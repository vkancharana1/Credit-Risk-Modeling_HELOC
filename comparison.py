import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Load true labels
y_test = pd.read_csv("y_test.csv").squeeze().values

# Load predictions
glm_preds = pd.read_csv("glm_preds.csv", header=None).squeeze().values
gam_preds = pd.read_csv("gam_preds.csv", header=None).squeeze().values
nn_preds = pd.read_csv("nn_preds.csv", header=None).squeeze().values

models = {
    'GLM': glm_preds,
    'GAM': gam_preds,
    'Neural Network': nn_preds
}

print("Model Performance Comparison")
print("="*40)
for name, preds in models.items():
    auc = roc_auc_score(y_test, preds)
    logloss = log_loss(y_test, preds)
    print(f"{name:20} AUC: {auc:.4f}   Log-loss: {logloss:.4f}")

# Calibration curves
plt.figure(figsize=(8,6))
for name, preds in models.items():
    prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)
plt.plot([0,1],[0,1], linestyle='--', label='Perfect')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration curves')
plt.legend()
plt.savefig('calibration_curves.png')
plt.show()