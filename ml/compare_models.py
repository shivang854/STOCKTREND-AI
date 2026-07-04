import pandas as pd

from ml.linear_model import train_linear_model
from ml.random_forest import train_random_forest
from ml.xgboost_model import train_xgboost

linear = train_linear_model()
rf = train_random_forest()
xgb = train_xgboost()

results = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "MAE": linear["MAE"],
        "MSE": linear["MSE"],
        "R2 Score": linear["R2 Score"]
    },
    {
        "Model": "Random Forest",
        "MAE": rf["MAE"],
        "MSE": rf["MSE"],
        "R2 Score": rf["R2 Score"]
    },
    {
        "Model": "XGBoost",
        "MAE": xgb["MAE"],
        "MSE": xgb["MSE"],
        "R2 Score": xgb["R2 Score"]
    }
])

results = results.sort_values("R2 Score", ascending=False)

results.to_csv("models/model_comparison.csv", index=False)

print(results)
