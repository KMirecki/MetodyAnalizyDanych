import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# DataFrame wczytany z csv
df = pd.read_csv("medical_insurance.csv")
# rows, columns = df.shape
# print(df.head())
# print(f"Number of rows: {rows}, Number of columns: {columns}")
# print(df.info())

# Wybranie kilku najważniejszych kolumn z pełnego DataFrame
df1 = df[["age", "sex", "income", "education", "employment_status", "bmi", "smoker", "alcohol_freq", "risk_score",
          "annual_medical_cost", "is_high_risk"]]
print(df1.info())

# Utworzenie histogramów na podstawie zmiennych numerycznych
num_cols = ["age", "income", "bmi", "risk_score", "annual_medical_cost"]

for col in num_cols:
    plt.figure(figsize=(10, 7))
    sns.histplot(df1[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.show()

# Boxploty
# for col in num_cols:
#     plt.figure(figsize=(10, 7))
#     sns.boxplot(x=df1[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()
