
from pathlib import Path
import json

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "placement_dataset.csv"
REPORT_DIR = BASE_DIR / "reports"

TARGET = "PlacementStatus"
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------
print("Loading placement dataset...")
df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

df = df.dropna(subset=[TARGET]).copy()

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str).str.strip()

if y.nunique() != 2:
    raise ValueError(
        f"Expected two placement classes, found: {y.unique().tolist()}"
    )

if "Placed" not in y.unique():
    raise ValueError("The dataset must contain the 'Placed' class.")

print(f"Dataset records: {len(df)}")
print(f"Input features: {X.shape[1]}")
print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. Split data BEFORE cross-validation
# --------------------------------------------------
# The test set is kept separate until model selection
# using cross-validation has been completed.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining records: {len(X_train)}")
print(f"Test records: {len(X_test)}")

# --------------------------------------------------
# 4. Preprocessing inside each model pipeline
# --------------------------------------------------
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

# --------------------------------------------------
# 5. Define the three models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=1,
    ),
}

# Measure precision, recall and F1 specifically for
# students classified as Placed.
from sklearn.metrics import make_scorer

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_placed": make_scorer(
        precision_score,
        pos_label="Placed",
        zero_division=0,
    ),
    "recall_placed": make_scorer(
        recall_score,
        pos_label="Placed",
        zero_division=0,
    ),
    "f1_placed": make_scorer(
        f1_score,
        pos_label="Placed",
        zero_division=0,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# --------------------------------------------------
# 6. Five-fold cross-validation
# --------------------------------------------------
print("\nRunning stratified 5-fold cross-validation...")
cv_results = []
pipelines = {}

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipelines[name] = pipeline

    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean() * 100
        row[f"{metric}_std"] = values.std() * 100

    cv_results.append(row)

# Rank using mean F1-score for the Placed class.
results_df = pd.DataFrame(cv_results)
results_df = results_df.sort_values(
    "f1_placed_mean",
    ascending=False,
).reset_index(drop=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

cv_report_path = REPORT_DIR / "model_comparison_cv.csv"
results_df.to_csv(cv_report_path, index=False)

print("\n========== CROSS-VALIDATION RESULTS ==========")

display_columns = [
    "Model",
    "accuracy_mean",
    "balanced_accuracy_mean",
    "precision_placed_mean",
    "recall_placed_mean",
    "f1_placed_mean",
    "f1_placed_std",
]

print(
    results_df[display_columns]
    .round(2)
    .to_string(index=False)
)

best_name = results_df.iloc[0]["Model"]

print(f"\nSelected by cross-validation F1-score: {best_name}")

# --------------------------------------------------
# 7. Evaluate the selected model ONCE on test data
# --------------------------------------------------
print("\nEvaluating selected model on the held-out test set...")

best_model = pipelines[best_name]
best_model.fit(X_train, y_train)

test_predictions = best_model.predict(X_test)

test_metrics = {
    "selected_model": best_name,
    "test_records": int(len(X_test)),
    "accuracy_percent": round(
        accuracy_score(y_test, test_predictions) * 100, 2
    ),
    "balanced_accuracy_percent": round(
        balanced_accuracy_score(y_test, test_predictions) * 100, 2
    ),
    "precision_placed_percent": round(
        precision_score(
            y_test, test_predictions,
            pos_label="Placed", zero_division=0
        ) * 100, 2
    ),
    "recall_placed_percent": round(
        recall_score(
            y_test, test_predictions,
            pos_label="Placed", zero_division=0
        ) * 100, 2
    ),
    "f1_placed_percent": round(
        f1_score(
            y_test, test_predictions,
            pos_label="Placed", zero_division=0
        ) * 100, 2
    ),
}

print("\n========== HELD-OUT TEST RESULTS ==========")
print(json.dumps(test_metrics, indent=4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(
    y_test,
    test_predictions,
    labels=["NotPlaced", "Placed"],
))

test_report_path = REPORT_DIR / "selected_model_test.json"

with open(test_report_path, "w", encoding="utf-8") as file:
    json.dump(test_metrics, file, indent=4)

print("\nReports saved:")
print(cv_report_path)
print(test_report_path)

print("\nExisting models/placement_model.pkl was NOT modified.")
print("Evaluation completed successfully.")
