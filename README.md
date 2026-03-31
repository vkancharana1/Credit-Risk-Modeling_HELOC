# 📊 HELOC Credit Risk Modeling

This repository contains a complete data science project for predicting serious delinquency (90+ days past due) on Home Equity Lines of Credit (HELOC) using the FICO HELOC dataset. We develop and compare several predictive models: Logistic Regression (GLM), Generalized Additive Models (GAM), Quantile Regression, and a Neural Network. The project emphasizes both predictive performance and interpretability, with business insights derived from each model.

---

## 📁 Table of Contents

- [📊 Dataset](#dataset)
- [📁 Project Structure](#project-structure)
- [⚙️ Requirements](#requirements)
- [🚀 Setup](#setup)
- [🏃‍♂️ Running the Code](#running-the-code)
- [🧠 Models](#models)
- [📈 Results](#results)
- [💡 Interpretation & Business Insights](#interpretation--business-insights)
- [🤝 Contributing](#contributing)
- [📄 License](#license)

---

## 📊 Dataset

We use the **FICO HELOC dataset** from the 2018 Explainable Machine Learning Challenge. It contains 10,459 anonymized credit applications with 23 features and a binary target (`RiskPerformance` = "Bad" or "Good"). The dataset is included in this repository as `heloc_dataset.csv`. You can also download it from the [FICO Community](https://community.fico.com/s/explainable-machine-learning-challenge).

---

## 📁 Project Structure

```
heloc_project/
│
├── heloc_dataset.csv          # Raw dataset
├── requirements.txt           # Python dependencies
│
├── explore.py                 # 🔍 Exploratory data analysis
├── preprocess.py              # 🧹 Data cleaning & splitting
├── glm.py                     # 📈 Logistic Regression (GLM)
├── gam.py                     # 📊 Generalized Additive Model
├── quantile.py                # 📉 Quantile Regression on credit score
├── neuralnet.py               # 🧠 Neural Network (Keras/TensorFlow)
├── comparison.py              # 📊 Model comparison (AUC, log-loss, calibration)
├── interpret.py               # 🔮 SHAP explanation for Neural Network
│
├── outputs/                   # 📁 Generated files (will be created)
│   ├── X_train.csv            # Training features
│   ├── X_test.csv             # Test features
│   ├── y_train.csv            # Training labels
│   ├── y_test.csv             # Test labels
│   ├── glm_preds.csv          # GLM predictions
│   ├── gam_preds.csv          # GAM predictions
│   ├── nn_preds.csv           # Neural Network predictions
│   ├── quantile_coefficients.csv
│   ├── calibration_curves.png
│   ├── gam_partial_dependence.png
│   ├── shap_summary.png
│   └── ... other plots
│
└── README.md                  # 📖 This file
```

---

## ⚙️ Requirements

All required packages are listed in `requirements.txt`. The main libraries are:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `statsmodels`
- `pygam`
- `tensorflow`
- `shap`

---

## 🚀 Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/heloc-credit-risk.git
   cd heloc-credit-risk
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Running the Code

Run the scripts in order:

```bash
python explore.py          # 🔍 Explore the dataset
python preprocess.py       # 🧹 Clean and split data
python glm.py              # 📈 Fit logistic regression
python gam.py              # 📊 Fit GAM
python quantile.py         # 📉 Run quantile regression
python neuralnet.py        # 🧠 Train neural network
python comparison.py       # 📊 Compare all models
python interpret.py        # 🔮 SHAP explanation for NN
```

The scripts will generate output files and plots in the current directory.

---

## 🧠 Models

We employ four complementary approaches:

1. **GLM (Logistic Regression)**: A highly interpretable baseline using `statsmodels`. Coefficients and p‑values identify the most influential features.

2. **GAM (Generalized Additive Model)**: Captures non‑linear relationships while remaining interpretable. Partial dependence plots visualize the effect of each feature on the predicted log‑odds.

3. **Quantile Regression**: Models the credit score (`ExternalRiskEstimate`) at different quantiles (10th, 50th, 90th). This reveals how features affect low‑scoring (high‑risk) applicants differently.

4. **Neural Network**: A feed‑forward network with two hidden layers, dropout for regularization, and sigmoid output. Predictions are explained with SHAP values.

---

## 📈 Results

| Model             | AUC    | Log‑loss |
|-------------------|--------|----------|
| GLM (Logistic)    | 0.7901 | 0.5534   |
| GAM               | 0.7864 | 0.5594   |
| Neural Network    | 0.7900 | 0.5532   |

- ✅ All models achieve an AUC around 0.79, indicating good discriminatory power.
- 🎯 The GLM and neural network have the lowest log‑loss, meaning they are slightly better calibrated.
- 📊 Calibration curves confirm that all models are reasonably well‑calibrated.

---

## 💡 Interpretation & Business Insights

- **🏦 Credit score is the most important predictor**: A 10‑point increase in `ExternalRiskEstimate` reduces default odds by 41%.
- **🔍 Recent credit inquiries signal distress**: Each additional inquiry increases odds of default by 48%. The GAM reveals that the effect flattens after about 4 inquiries.
- **💳 Revolving utilization matters**: Higher revolving burden is associated with higher risk.
- **📉 Quantile regression insight**: Past delinquencies are much more harmful for already low‑score applicants, suggesting risk‑based pricing should be more aggressive in that segment.
- **🔮 Neural network interpretation (SHAP)**: Confirms the same top features as the GLM, increasing trust in the model.

**Business recommendations**:
- 📝 Use the GLM as the primary scorecard for its transparency and regulatory compliance.
- 📊 Incorporate GAM insights (e.g., threshold effects) into rating factors.
- 💰 Leverage quantile regression findings for risk‑based pricing.
- 🧠 Use the neural network as a “second look” tool for borderline applications.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

**👤 Author**: Venkata Sai Prasad Kancharana
**🎓 Course**: Ms in Financial Mathematics 
