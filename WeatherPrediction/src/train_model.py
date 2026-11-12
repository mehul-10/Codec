
import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_weather.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run preprocessing first."
        )

    return pd.read_csv(DATA_PATH)


def prepare_data(df):

    # Remove rows with missing target
    df = df.dropna(subset=["RainTomorrow"]).copy()

    # Target variable
    X = df.drop(columns=["RainTomorrow"])

    # Explicitly exclude next-day rainfall information if present
    X = X.drop(columns=["RISK_MM"], errors="ignore")

    y = df["RainTomorrow"].map({
        "No": 0,
        "Yes": 1
    })

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return (
        X,
        y,
        numeric_features,
        categorical_features
    )


def build_preprocessor(numeric_features, categorical_features):

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y_test, probabilities)
    }

    print(f"\n========== {name.upper()} RESULTS ==========")

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Rain", "Rain"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return metrics


def train_models():

    print("Loading processed weather dataset...")

    df = load_data()

    print("Dataset shape:", df.shape)

    X, y, numeric_features, categorical_features = prepare_data(df)

    print("\nNumerical features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))

    # Stratified split preserves the rain/no-rain ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, classifier in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test
        )

        results.append(metrics)
        trained_models[name] = pipeline

    # Compare models using F1 score
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Save the best model
    model_path = MODEL_DIR / "best_weather_model.pkl"

    joblib.dump(best_model, model_path, compress=3)

    # Save comparison results
    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    # Save selected model name
    with open(MODEL_DIR / "model_info.json", "w") as file:
        json.dump({
            "best_model": best_model_name,
            "selection_metric": "F1 Score"
        }, file, indent=4)

    print("\n====================================")
    print("BEST MODEL:", best_model_name)
    print("Model saved to:", model_path)
    print("Comparison saved successfully!")
    print("MODEL TRAINING COMPLETED!")
    print("====================================")


if __name__ == "__main__":
    train_models()
