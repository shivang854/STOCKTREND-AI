from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from utils.data_loader import load_featured_data
from utils.metrics import evaluate_model
from utils.save_model import save_model
from utils.plotting import plot_actual_vs_predicted


def train_xgboost():
    # Load data
    X, y, df = load_featured_data()

    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)

    all_results = []

    for train_index, test_index in tscv.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # Train model
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Store fold results
        all_results.append(
            evaluate_model(y_test, predictions)
        )

    # Average metrics across folds
    results = {
        "MAE": sum(r["MAE"] for r in all_results) / len(all_results),
        "MSE": sum(r["MSE"] for r in all_results) / len(all_results),
        "R2 Score": sum(r["R2 Score"] for r in all_results) / len(all_results),
    }

    print("\n========== XGBoost ==========")
    print(f"MAE : {results['MAE']:.4f}")
    print(f"MSE : {results['MSE']:.4f}")
    print(f"R² Score : {results['R2 Score']:.4f}")

    # Save last trained model
    save_model(model, "xgboost")

    # Plot last fold prediction
    plot_actual_vs_predicted(
        y_test.values,
        predictions,
        title="Actual vs Predicted Stock Price (XGBoost)",
        save_path="assets/xgboost_plot.png"
    )

    return results


if __name__ == "__main__":
    train_xgboost()
