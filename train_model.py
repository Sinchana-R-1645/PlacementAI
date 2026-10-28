
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "placement_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "placement_model.pkl"

# 2. Load the student dataset
print("Loading placement dataset...")
df = pd.read_csv(DATA_PATH)

# 3. Define input features and target
target = "PlacementStatus"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' was not found.")

df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target].astype(str).str.strip()

# Remove accidental whitespace and check the target values
print("\nPlacement status counts:")
print(y.value_counts())

if y.nunique() < 2:
    raise ValueError("The dataset must contain at least two placement classes.")

# 4. Identify numerical and categorical columns
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

# 5. Preprocess the input data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 6. Create the machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Train the model
print("\nTraining the Random Forest model...")
model.fit(X_train, y_train)

# 9. Evaluate on unseen test data
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# 10. Save the trained model and feature information
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "classes": model.classes_.tolist()
    },
    MODEL_PATH
)

print(f"\nModel saved successfully to: {MODEL_PATH}")
print("Model training completed!")
