
# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold, cross_validate

def create_models():
    """Create regression models for comparison."""

    models = {
        "Polynomial Regression": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LinearRegression()
        ),

        "KNN Regression": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=5,
                weights="distance"
            )
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42
        )
    }

    return models

def compare_models_cv(models, X, y):
    """Compare regression models using five-fold cross-validation."""

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "R2": "r2",
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error"
    }

    results = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        results.append({
            "Model": name,
            "Mean R2": scores["test_R2"].mean(),
            "Mean MAE": -scores["test_MAE"].mean(),
            "Mean RMSE": -scores["test_RMSE"].mean()
        })

    return pd.DataFrame(results).sort_values(
        by="Mean RMSE"
    ).reset_index(drop=True)
