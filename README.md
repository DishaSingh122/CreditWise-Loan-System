# CreditWise — Loan Approval Prediction

Lightweight Jupyter workflow for preprocessing, EDA, feature engineering and basic ML models to predict loan approvals using the provided loan dataset.

## Contents
- Notebook: `credit_wise.ipynb` — full pipeline (load → impute → encode → scale → train/evaluate).
- Data: `loan_approval_data.csv` — source dataset (must be next to the notebook).

## Quick start (Windows)
1. Open PowerShell or CMD in the project folder:
   cd "c:\Users\disha\OneDrive\Desktop\PrimeAI\Supervised_Learning\CreditWise_Loan_System"
2. Create / activate environment (optional):
   conda create -n primeai python=3.10 -y
   conda activate primeai
3. Install dependencies:
   pip install pandas numpy scikit-learn seaborn matplotlib jupyterlab
4. Launch Jupyter:
   jupyter lab
5. Open and run `credit_wise.ipynb` from JupyterLab, run cells top-to-bottom.

## What the notebook does (summary)
- Loads `loan_approval_data.csv` and inspects data.
- Imputes missing values: numerical → mean, categorical → mode.
- Encodes categorical features (LabelEncoder + OneHotEncoder).
- Scales numeric features with `StandardScaler`.
- Trains and evaluates: Logistic Regression, KNN, Gaussian Naive Bayes.
- Adds simple feature engineering (squared interactions) and re-evaluates models.
- Prints precision, recall, accuracy, F1 and confusion matrix for each model.

## Key variables (inside notebook)
- df — cleaned DataFrame
- X, y — features and target
- X_train_scaled, X_test_scaled — scaled feature arrays
- log_model, knn_model, nb_model — trained estimators

## Notes & recommendations
- Ensure `loan_approval_data.csv` is present in the same folder as the notebook.
- Use the conda environment reflected in the notebook metadata or install listed packages.
- For production use: convert notebook steps into a reproducible script/module and add train/validation splits with cross-validation and model persistence (joblib).
