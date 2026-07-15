import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def main():
    csv_path = "D:\Excellence_Technology\Salary_Vs_Experience\Salary_Data.csv"
    df = pd.read_csv(csv_path)

    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset shape (rows, columns):", df.shape)

    missing_values = df.isnull().sum()
    print("\nMissing values per column:\n", missing_values)

    if df.isnull().values.any():
        df = df.dropna()
        print("\nMissing values found and removed. New shape:", df.shape)
    else:
        print("\nNo missing values found. No cleaning needed.")

    X = df[["YearsExperience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n----- Model Evaluation on Test Set -----")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.2f}")

    intercept = model.intercept_
    coefficients = model.coef_

    print("\n----- Regression Equation -----")
    equation_terms = " + ".join(
        f"({coef:.2f} * {feature})"
        for coef, feature in zip(coefficients, X.columns)
    )
    print(f"Salary = {intercept:.2f} + {equation_terms}")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, color="royalblue", edgecolor="k", label="Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        label="Perfect Prediction Line",
    )
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    print("\nSaved plot: actual_vs_predicted.png")
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(X_train, y_train, color="green", edgecolor="k", label="Training data")
    plt.scatter(X_test, y_test, color="orange", edgecolor="k", label="Test data")

    X_range = np.linspace(
        X["YearsExperience"].min(),
        X["YearsExperience"].max(),
        100,
    ).reshape(-1, 1)

    X_range_df = pd.DataFrame(X_range, columns=["YearsExperience"])
    y_range_pred = model.predict(X_range_df)

    plt.plot(X_range, y_range_pred, color="red", linewidth=2, label="Regression Line")

    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Salary vs Years of Experience — Regression Line")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_line.png", dpi=150)
    print("Saved plot: regression_line.png")
    plt.close()

    print("\nDone. Review the printed metrics above and the two saved plots.")


if __name__ == "__main__":
    main()