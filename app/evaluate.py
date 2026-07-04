import pandas as pd

results = [
    {
        "Model": "Linear Regression",
        "MAE": 2.0894,
        "MSE": 6.7124,
        "R2 Score": 0.9956
    },
    {
        "Model": "Random Forest",
        "MAE": 0.9515,
        "MSE": 1.4546,
        "R2 Score": 0.9991
    }
]

df = pd.DataFrame(results)

print(df)

df.to_csv("models/model_comparison.csv", index=False)

print("\n✅ Model comparison saved successfully!")