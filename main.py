
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.constants import (
    DATASET_PATH,
    FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE
)
from src.functions import create_models, compare_models_cv
from src.classes import ModelEvaluator


def main():
    # Set up project paths
    project_dir = Path(__file__).resolve().parent
    dataset_path = project_dir / DATASET_PATH

    tables_dir = project_dir / "tables"
    figures_dir = project_dir / "figures"

    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Load the dataset
    data = pd.read_csv(dataset_path)

    # Validate required columns
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep required columns and remove missing values
    data = data[required_columns].dropna()

    X = data[FEATURES]
    y = data[TARGET]

    print(f"Dataset loaded: {data.shape[0]} observations")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Create and evaluate regression models
    models = create_models()

    evaluator = ModelEvaluator(models)
    results = evaluator.evaluate(
        X_train, X_test, y_train, y_test
    )

    print("\nModel Performance Comparison")
    print(results.to_string(index=False))

    # Save test-set results
    results_path = tables_dir / "model_comparison.csv"
    results.to_csv(results_path, index=False)
    print(f"\nTest results saved to: {results_path}")

    # Five-fold cross-validation
    cv_results = compare_models_cv(models, X, y)

    print("\nFive-Fold Cross-Validation Results")
    print(cv_results.to_string(index=False))

    cv_results_path = tables_dir / "cross_validation_results.csv"
    cv_results.to_csv(cv_results_path, index=False)
    print(f"Cross-validation results saved to: {cv_results_path}")

    # Figure 1: Compare model performance metrics
    metrics = ["R2", "MAE", "RMSE"]
    model_names = results["Model"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metrics):
        ax.bar(results["Model"], results[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=30)

    fig.suptitle("Regression Model Performance on the Test Set")
    fig.tight_layout()

    comparison_path = figures_dir / "model_comparison.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Model comparison figure saved to: {comparison_path}")

    # Figure 2: Actual versus predicted growth rates
    best_model_name = results.iloc[0]["Model"]
    best_model = models[best_model_name]

    predictions = best_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_test, predictions, alpha=0.7)

    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Ideal predictions"
    )

    ax.set_xlabel("Actual Specific Growth Rate [d^-1]")
    ax.set_ylabel("Predicted Specific Growth Rate [d^-1]")
    ax.set_title(f"Actual vs. Predicted: {best_model_name}")
    ax.legend()
    fig.tight_layout()

    prediction_path = figures_dir / "actual_vs_predicted.png"
    fig.savefig(prediction_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Prediction figure saved to: {prediction_path}")

    # Figure 3: Individual input variables versus growth rate
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, feature in zip(axes.flat, FEATURES):
        ax.scatter(data[feature], data[TARGET], alpha=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{feature} vs. Specific Growth Rate")

    fig.suptitle("Input Variables and Specific Growth Rate")
    fig.tight_layout()

    relationships_path = figures_dir / "feature_relationships.png"
    fig.savefig(relationships_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Feature relationship figures saved to: {relationships_path}")

    print("\nAll tables and figures have been generated successfully.")


if __name__ == "__main__":
    main()
