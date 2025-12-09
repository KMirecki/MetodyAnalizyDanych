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
          "annual_medical_cost"]]
print(df1.info())

# Podział danych na zmienne numeryczne i nienumeryczne
num_cols = ["age", "income", "bmi", "risk_score", "annual_medical_cost"]
cat_cols = ["sex", "education", "employment_status", "smoker", "alcohol_freq"]

# W komórkach alcohol_freq None było odczytywane przez pandas jako brak danych - zamiana None na Never
df1["alcohol_freq"] = df1["alcohol_freq"].fillna("Never")
print(df1.info())

# Sprawdzenie czy w datasecie są puste komórki
print(f"Liczba pustych komórek w kolumnie:\n{df1.isnull().sum()}")

# Walidacja danych numerycznych
for col in num_cols:
    print(f"Czy w kolumnie {col} są dane negatywne?: {(df1[col] < 0).any()}")

# Wykrywanie obserwacji odstających metodą 1.5 IQR
for col in num_cols:
    # Obliczenie kwartyli, IQR i progów
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Zliczenie obserwacji odstającyh
    outliers_low = df1[col][df1[col] < lower_bound].count()
    outliers_high = df1[col][df1[col] > upper_bound].count()
    total_outliers = outliers_low + outliers_high

    print(f"\nObserwacje odstające dla kolumny {col}:")
    print(f"Próg dolny i górny: ({lower_bound:.2f}, {upper_bound:.2f})")
    print(f"Obserwacje odstające z dołu: {outliers_low}")
    print(f"Obserwacje odstające z góry: {outliers_high}")
    print(f"Łącznie: {total_outliers} ({total_outliers / len(df1) * 100:.2f}%)")

# Utworzenie histogramów dla zmiennych numerycznych
# for col in num_cols:
#     plt.figure(figsize=(10, 7))
#     sns.histplot(df1[col], kde=True)
#     plt.title(f"Histogram of {col}")
#     plt.show()

# Utworzenie boxplotów dla zmiennych numerycznych
# for col in num_cols:
#     plt.figure(figsize=(10, 7))
#     sns.boxplot(x=df1[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

# Wykresy słupkowe dla zmiennych nienumerycznych
# for col in cat_cols:
#     plt.figure(figsize=(10, 7))
#     sns.countplot(x=df1[col])
#     plt.title(f"Countplot of {col}")
#     plt.show()

# Macierz korelacji zmiennych numerycznych
correlation_matrix = df1[num_cols].corr()
print(correlation_matrix)
# plt.figure(figsize=(10, 7))
# sns.heatmap(correlation_matrix, annot=True)
# plt.yticks(rotation=45)
# plt.title("Macierz korelacji zmiennych numerycznych")
# plt.show()
