import matplotlib.pyplot as plt
import os

def plot_actual_vs_predicted(y_true, y_pred, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:100], label="Actual Price")
    plt.plot(y_pred[:100], label="Predicted Price")
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Price")
    plt.legend()

    os.makedirs("assets", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()