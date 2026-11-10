# %% [markdown]
# # Customer Churn: Logistic Regression vs Random Forest (refreshed version)
#
# SETUP (do once):
#   1. Download "Telco Customer Churn" from Kaggle and extract the zip.
#   2. Put WA_Fn-UseC_-Telco-Customer-Churn.csv in the SAME folder as this file.
#   3. In the VS Code terminal run:
#        python -m pip install pandas scikit-learn matplotlib
#   4. Click the Run button (top right).

# %% Step 1: Imports and file path
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Look for the CSV next to this file (works even if the terminal is in another folder).
# In Jupyter there is no __file__, so we use the current folder instead.
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
CSV_PATH = BASE_DIR / CSV_NAME

if not CSV_PATH.exists():
    raise SystemExit(
        f"\nCSV not found!\nLooked here: {CSV_PATH}\n"
        f"Download '{CSV_NAME}' from Kaggle and put it in: {BASE_DIR}\n"
    )

# %% Step 2: Load the CSV and take a first look
df = pd.read_csv(CSV_PATH)
print("Rows, columns:", df.shape)
print(df.head())
print(df.dtypes)

# %% Step 3: Clean the data with Pandas
# 3a. customerID is just a label, not a clue -> drop it
df = df.drop(columns=["customerID"])

# 3b. TotalCharges is text because a few cells are blank spaces.
#     errors="coerce" turns those blanks into NaN (missing values).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 3c. Find missing values, then drop those few rows
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0])
df = df.dropna()

# 3d. The answer column: Yes -> 1 (customer left), No -> 0 (customer stayed)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 3e. Models only understand numbers, so turn text columns into 0/1 columns
df = pd.get_dummies(df, drop_first=True)

print("\nAfter cleaning, rows and columns:", df.shape)
print("Churn rate (share of customers who left):", round(df["Churn"].mean(), 3))

# %% Step 4: Clues (X) and answer (y), then study set / exam set
X = df.drop(columns=["Churn"])   # clues
y = df["Churn"]                  # answer

# stratify=y keeps the same churn share in both the study set and the exam set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Study rows:", len(X_train), "| Exam rows:", len(X_test))

# %% Step 5: Build the two models (our two "students")
models = {
    # Logistic Regression likes numbers on a similar scale, so we scale first.
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    # Random Forest = many decision trees voting together. No scaling needed.
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# %% Step 6: fit (study), predict (exam), score (marks)
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)            # study
    guesses = model.predict(X_test)        # exam
    predictions[name] = guesses

    results.append({
        "Model": name,
        "Accuracy":  accuracy_score(y_test, guesses),
        "Precision": precision_score(y_test, guesses),
        "Recall":    recall_score(y_test, guesses),
        "F1":        f1_score(y_test, guesses),
    })

comparison = pd.DataFrame(results).set_index("Model").round(3)
print("\n=== MODEL COMPARISON ===")
print(comparison)

# %% Step 7: Confusion matrices as plain numbers
for name, guesses in predictions.items():
    cm = confusion_matrix(y_test, guesses)
    table = pd.DataFrame(
        cm,
        index=["Really stayed", "Really churned"],
        columns=["Said stayed", "Said churned"],
    )
    print(f"\n--- Confusion matrix: {name} ---")
    print(table)
    print(f"Missed churners (costly): {cm[1][0]} | False alarms: {cm[0][1]}")

# %% Step 8: Confusion matrices as pictures
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, guesses) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, guesses)
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()

# How to read each matrix (rows = real answer, columns = model's guess):
#   top-left      = stayed, model said stayed   (correct)
#   top-right     = stayed, model said churned  (false alarm)
#   bottom-left   = churned, model said stayed  (MISSED churner, the costly one)
#   bottom-right  = churned, model said churned (caught churner)

# %% Step 9: The 5-line "which model do I ship?" answer, written from YOUR numbers
lr = comparison.loc["Logistic Regression"]
rf = comparison.loc["Random Forest"]

# Recall is the priority for churn. If recall is nearly tied (within 0.01),
# prefer the simpler, easier-to-explain Logistic Regression.
winner = "Random Forest" if rf["Recall"] > lr["Recall"] + 0.01 else "Logistic Regression"
other = "Logistic Regression" if winner == "Random Forest" else "Random Forest"
w, o = comparison.loc[winner], comparison.loc[other]

print("\n=== WHICH MODEL WOULD I SHIP? ===")
print(f"1. Accuracy is {lr['Accuracy']:.1%} (Logistic Regression) vs {rf['Accuracy']:.1%} (Random Forest), "
      f"a gap of only {abs(lr['Accuracy'] - rf['Accuracy']):.1%}, so accuracy alone can't decide.")
print("2. For churn, recall matters most: every churner we miss is a customer lost without a chance to save them.")
print(f"3. {winner} recall is {w['Recall']:.1%} vs {o['Recall']:.1%} for {other}, and its F1 is {w['F1']:.3f} vs {o['F1']:.3f}.")
print("4. Logistic Regression is also simple, fast, and easy to explain to a manager; Random Forest is harder to explain.")
print(f"5. I would ship {winner}, then try lowering the decision threshold to catch even more churners.")