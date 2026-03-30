import pandas as pd
import numpy as np
import shap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze().values
y_test = pd.read_csv("y_test.csv").squeeze().values

# Load the scaler used in training (we need to re-create it)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the trained model
model = tf.keras.models.load_model("nn_model.h5")

# Use a subset of test data for SHAP (to keep runtime reasonable)
background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
explainer = shap.KernelExplainer(model.predict, background)

# Choose a small subset of test data to explain
test_sample = X_test_scaled[:100]
shap_values = explainer.shap_values(test_sample)

# Summary plot
shap.summary_plot(shap_values, test_sample, feature_names=X_train.columns.tolist())
plt.savefig("shap_summary.png", bbox_inches='tight')