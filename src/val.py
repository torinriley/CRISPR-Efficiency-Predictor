import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

data = pd.read_csv("results/results.csv")

data["True Score"] = pd.to_numeric(data["True Score"], errors='coerce')
data["Predicted Score"] = pd.to_numeric(data["Predicted Score"], errors='coerce')

data = data.dropna(subset=["True Score", "Predicted Score"])

true_scores = data["True Score"]
predicted_scores = data["Predicted Score"]

r_squared = r2_score(true_scores, predicted_scores)
print(f"R-squared (Coefficient of Determination): {r_squared:.4f}")

pearson_corr, p_value = pearsonr(true_scores, predicted_scores)
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"P-value for Pearson Correlation: {p_value:.4e}")

residuals = true_scores - predicted_scores
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)
print(f"Mean Residual: {mean_residual:.4f}")
print(f"Residual Standard Deviation: {std_residual:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(true_scores, predicted_scores, color="blue", label="Predictions", s=50)
plt.plot([min(true_scores), max(true_scores)], [min(true_scores), max(true_scores)],
         color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
plt.title("True vs. Predicted Scores")
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.legend()
plt.grid(True)
plt.savefig("True_vs_Predicted_Scores.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(true_scores, residuals, color="blue", s=50)
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.title("Residual Plot")
plt.xlabel("True Score")
plt.ylabel("Residuals (True - Predicted)")
plt.grid(True)
plt.savefig("Residual_Plot.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=10, color="blue", edgecolor="black", alpha=0.7)
plt.title("Residual Histogram")
plt.xlabel("Residuals (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("Residual_Histogram.png")
plt.show()