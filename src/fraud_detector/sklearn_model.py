import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud_detector.utils import data_preprocessing, load_data

# Load data
df = load_data()

df = data_preprocessing(df)

features = [
    "edge_noise",
    "text_density",
    "grayscale_variance",
    "alpha_channel_density",
    "unique_font_colors",
    "reported_income",
]

X = df[features]
y = df["label"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Train/test split (stratified for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a pipeline for imputation and scaling (only use training data set for fitting)
preprocessing_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)


# model selection pipeline XGBOOST + Logistic regression.

model_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing_pipeline),
        (
            "classifier",
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbose=1),
        ),
    ]
)


# add cross validation and hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__max_depth": [3, 5, 7],
    "classifier__n_estimators": [100, 200, 500, 1000],
    # "classifier__learning_rate": [0.01, 0.1],
}

grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=5, scoring="recall", verbose=2
)


def evaluate_model(model, X, y, dataset_name="Test set"):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    precision, recall, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y, y_pred)

    print(f"{dataset_name} Set Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"F1 Score: {f1:.3f}")


# train model with CV.
grid_search.fit(X_train, y_train)
# Evaluate the best model
best_model = grid_search.best_estimator_

evaluate_model(best_model, X_train, y_train, dataset_name="Train set")
evaluate_model(best_model, X_test, y_test, dataset_name="Test set")


def plot_decision_boundary_sklearn(X, y, pipeline, title, columns):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X[columns[0]], X[columns[1]], c=y, cmap="coolwarm", edgecolors="k", s=50
    )

    # Create a grid to plot the decision boundary
    x_min, x_max = X[columns[0]].min() - 1, X[columns[0]].max() + 1
    y_min, y_max = X[columns[1]].min() - 1, X[columns[1]].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Prepare grid dataframe with all columns, fill others with mean
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=columns)
    for col in X.columns:
        if col not in columns:
            grid[col] = X[col].mean()
    grid = grid[X.columns]  # Ensure column order matches

    Z = pipeline.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.title(title)
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.colorbar(label="Label")


joblib.dump(grid_search, "best_xgboost_model.pkl")

# Example usage:
cols = ["alpha_channel_density", "edge_noise", "grayscale_variance"]
from itertools import combinations

for pair in combinations(cols, 2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_decision_boundary_sklearn(
        X_train, y_train, best_model, "Decision Boundary on Training Set", pair
    )
    plot_decision_boundary_sklearn(
        X_test, y_test, best_model, "Decision Boundary on Test Set", pair
    )
    plt.tight_layout()
    plt.show()

# Save best model in terms of f1 score


# load model from pickle file
loaded_model = joblib.load("best_xgboost_model.pkl")
# Evaluate the loaded model
