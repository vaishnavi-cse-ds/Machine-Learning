"""
Telecom Customer Churn Prediction - CRISP-DM Pipeline
Author : R. Vaishnavi (24501A4448)

Dataset: Telco Customer Churn (IBM sample data, on Kaggle)
         Put Telco-Customer-Churn.csv in the same folder as this file.

Usage  : pip install pandas numpy matplotlib seaborn scikit-learn joblib
         python churn_prediction_pipeline.py [path/to/Telco-Customer-Churn.csv]

Output : plots in the 'figures/' folder and the saved model
         'telco_churn_model.joblib'.
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to files (works without a display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "Telco-Customer-Churn.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=130, bbox_inches="tight")
    plt.close()


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 2. DATA UNDERSTANDING
# ----------------------------------------------------------------------
section("2. DATA UNDERSTANDING")
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.iloc[:5, :8].to_string())
df.info()

# ----------------------------------------------------------------------
# 3. DATA PREPARATION
# ----------------------------------------------------------------------
section("3.1 DATA CLEANING")
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
# TotalCharges is stored as text, so blank spaces hide as non-null
print("Blank TotalCharges values:", (df["TotalCharges"].str.strip() == "").sum())

# Convert TotalCharges to numbers (blanks become NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# customerID is only an identifier
df = df.drop(columns=["customerID"])
# Make SeniorCitizen readable like the other Yes/No columns
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
# Target: Yes -> 1 (customer left), No -> 0 (customer stayed)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
print("Shape after cleaning:", df.shape)
print("Missing values left:", df["TotalCharges"].isnull().sum(),
      "(filled later in the pipeline)")

section("3.2 EXPLORATORY DATA ANALYSIS")
cols = ["Contract", "InternetService", "PaymentMethod"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, cols):
    rate = df.groupby(col)["Churn"].mean() * 100
    print(rate.round(2).to_string(), "\n")
    rate.plot(kind="bar", ax=ax)
    ax.set_title("Churn % by " + col)
    ax.set_xlabel("")
    ax.set_ylabel("Churn %")
    ax.tick_params(axis="x", rotation=20)
save_fig("churn_rate_by_category.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[0])
axes[0].set_title("Tenure (months) vs Churn")
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1])
axes[1].set_title("Monthly Charges vs Churn")
save_fig("tenure_monthly_vs_churn.png")
print(df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean().round(2))

section("3.3 UNDERSTANDING THE TARGET VARIABLE")
print(df["Churn"].value_counts())
print("\nChurn rate:", round(df["Churn"].mean() * 100, 2), "%")
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Churn")
plt.xticks([0, 1], ["Stayed (0)", "Left (1)"])
plt.title("Customer Churn Count")
save_fig("churn_count.png")

section("3.4 - 3.5 FEATURES, TARGET AND COLUMN TYPES")
X = df.drop(columns=["Churn"])
y = df["Churn"]
print("Input Features Shape:", X.shape)
print("Target Shape:", y.shape)

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
print("Categorical Columns:", categorical_cols)
print("\nNumerical Columns:", numerical_cols)

section("3.6 TRAIN/TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42,
    stratify=y,  # keeps the same churn ratio in both sets
)
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Churn rate in training set:", round(y_train.mean() * 100, 2), "%")
print("Churn rate in testing set:", round(y_test.mean() * 100, 2), "%")

section("3.7 PREPROCESSING PIPELINE")
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols),
])
print("Preprocessing pipeline created successfully!")

# ----------------------------------------------------------------------
# 4. MODELLING
# ----------------------------------------------------------------------
section("4. MODELLING")
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1),
}

trained_models = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline
    print(name, "trained successfully!")

# ----------------------------------------------------------------------
# 5. EVALUATION
# ----------------------------------------------------------------------
section("5.2 MODEL COMPARISON")
results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print(results_df.round(3).reset_index(drop=True).to_string())

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print("\nBest Model:", best_model_name)
print("Test ROC-AUC:", round(results_df.iloc[0]["ROC-AUC"], 3))
print("Test Recall:", round(results_df.iloc[0]["Recall"], 3))
print("Test Accuracy:", round(results_df.iloc[0]["Accuracy"], 3))

section("5.3 CONFUSION MATRIX")
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Left"]).plot(cmap="Blues")
plt.title("Confusion Matrix - " + best_model_name)
save_fig("confusion_matrix.png")

section("5.4 ROC CURVES")
plt.figure(figsize=(6, 5))
for name, model in trained_models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
save_fig("roc_curves.png")
print("ROC curve saved.")

section("5.5 CROSS-VALIDATION (5-fold, ROC-AUC)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    cv_results.append({"Model": name,
                       "Mean CV ROC-AUC": scores.mean(),
                       "Standard Deviation": scores.std()})
cv_df = pd.DataFrame(cv_results).sort_values(by="Mean CV ROC-AUC", ascending=False)
print(cv_df.round(3).reset_index(drop=True).to_string())

section("5.6 MOST INFLUENTIAL FEATURES")
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
final_model = best_model.named_steps["model"]
values = (final_model.coef_[0] if hasattr(final_model, "coef_")
          else final_model.feature_importances_)
importance = pd.Series(values, index=feature_names)
print(importance.sort_values(key=abs, ascending=False).head(8).round(3))

# ----------------------------------------------------------------------
# 6. DEPLOYMENT
# ----------------------------------------------------------------------
section("6. DEPLOYMENT")
joblib.dump(best_model, "telco_churn_model.joblib")
print("Model saved successfully!")
loaded_model = joblib.load("telco_churn_model.joblib")
print("Model loaded successfully!")

new_customer = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No",
    "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0, "TotalCharges": 240.0,
}])
prob = loaded_model.predict_proba(new_customer)[0, 1]
print("Churn probability:", round(prob * 100, 2), "%")
print("Prediction:", "Customer is likely to LEAVE" if prob >= 0.5
      else "Customer is likely to STAY")

section("FINAL PROJECT SUMMARY")
print("Project: Telecom Customer Churn Prediction")
print("Total records:", len(df))
print("Features used:", X.shape[1])
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Best model:", best_model_name)
print(results_df[results_df["Model"] == best_model_name]
      .round(3).reset_index(drop=True).to_string())
