# Salary Prediction using Linear Regression

## 📌 Project Description

This project demonstrates a complete, beginner-friendly supervised
machine learning workflow using **scikit-learn**. It builds a **Simple
Linear Regression** model to predict an employee's **Salary** based on
their **Years of Experience**.

The workflow covers:
1. Loading data from a CSV file
2. Checking and handling missing values
3. Splitting data into training and testing sets
4. Training a Linear Regression model
5. Making predictions on unseen (test) data
6. Evaluating performance using **R² Score** and **RMSE**
7. Displaying the learned regression equation
8. Visualizing results (Actual vs Predicted, and the Regression Line)

## 📂 Project Structure

```
salary_regression/
├── Salary_Data.csv              # Dataset (YearsExperience, Salary)
├── linear_regression_salary.py  # Main script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation (this file)
├── actual_vs_predicted.png      # Output plot (generated after running)
└── regression_line.png          # Output plot (generated after running)
```

## 📊 Dataset

`Salary_Data.csv` contains 30 records with two columns:

| Column           | Description                                  |
|-------------------|-----------------------------------------------|
| `YearsExperience` | Number of years of professional experience (feature / X) |
| `Salary`          | Annual salary in currency units (target / y)  |

> Note: This is a synthetically generated dataset that follows the same
> structure as the well-known "Salary vs Experience" dataset commonly
> used for introductory regression exercises. You can freely replace
> `Salary_Data.csv` with your own dataset — as long as it has a
> continuous target column, the script's structure will still apply
> (just update the `X`/`y` column selection in the script).

## ⚙️ Installation

1. **Clone or download** this project folder.
2. **(Recommended) Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Execution Instructions

Run the main script from inside the project folder:

```bash
python linear_regression_salary.py
```

The script will:
- Print dataset info, missing-value checks, and train/test split sizes to the console
- Print the **R² Score** and **RMSE** on the test set
- Print the learned **regression equation**
- Save two plots as PNG files in the same folder:
  - `actual_vs_predicted.png`
  - `regression_line.png`

## ✅ Sample Output

```
First 5 rows of the dataset:
   YearsExperience    Salary
0              1.2  31796.03
1              1.4  39802.79
2              1.6  37797.45
3              2.3  45683.23
4              2.5  46343.17

Dataset shape (rows, columns): (30, 2)

Missing values per column:
YearsExperience    0
Salary             0
dtype: int64

No missing values found. No cleaning needed.

Training samples: 24
Testing samples:  6

----- Model Evaluation on Test Set -----
R² Score: 0.9680
RMSE:     3933.71

----- Regression Equation -----
Salary = 22696.01 + (9831.43 * YearsExperience)

Saved plot: actual_vs_predicted.png
Saved plot: regression_line.png
```

## 📈 Understanding the Output

**R² Score (Coefficient of Determination)**
- Measures the proportion of variance in the target variable (Salary)
  that is explained by the input feature (Years of Experience).
- Ranges from 0 to 1 (it can be negative for a very poor model).
- An R² of **0.97** means the model explains about **97% of the
  variation** in salary using years of experience alone — indicating a
  strong linear relationship between the two variables.

**RMSE (Root Mean Squared Error)**
- Measures the average magnitude of prediction error, in the *same
  units* as the target variable (here, currency units of salary).
- It penalizes larger errors more heavily than smaller ones (because
  errors are squared before averaging).
- An RMSE of about **3933.71** means, on average, the model's salary
  predictions differ from the actual salary by roughly that amount.
- Lower RMSE = better fit. RMSE should always be interpreted relative
  to the scale of the target variable (e.g., an RMSE of ~3,900 is
  small relative to salaries in the tens of thousands, which supports
  the high R² score above).

**Regression Equation**
- The printed equation (`Salary = intercept + slope * YearsExperience`)
  is the actual formula the model learned. The slope tells you how
  much salary is predicted to increase for each additional year of
  experience, and the intercept is the model's predicted salary at
  zero years of experience.

**Plots**
- *Actual vs Predicted*: Points close to the red dashed diagonal line
  indicate accurate predictions; the closer the points hug the line,
  the better the model.
- *Regression Line*: Shows the straight line the model fits through
  the training/testing data — visually confirming the linear trend
  between experience and salary.

## 🛠️ Requirements

See `requirements.txt`. Tested with Python 3.10+.

