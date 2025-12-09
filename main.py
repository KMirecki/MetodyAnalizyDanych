import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# DataFrame wczytany z csv
df = pd.read_csv("medical_insurance.csv")
# rows, columns = df.shape
# print(df.head())
# print(f"Number of rows: {rows}, Number of columns: {columns}")
# print(df.info())

# Wybranie kilku najważniejszych kolumn z pełnego DataFrame
df1 = df[["age", "sex", "income", "education", "employment_status", "bmi", "smoker", "alcohol_freq", "visits_last_year",
          "medication_count", "systolic_bp", "diastolic_bp",
          "risk_score",
          "annual_medical_cost", "had_major_procedure"]]
print(df1.info())

# Podział danych na zmienne numeryczne i kategoryczne
num_cols = ["age", "income", "bmi", "visits_last_year", "medication_count", "systolic_bp", "diastolic_bp", "risk_score",
            "annual_medical_cost",
            "had_major_procedure"]
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
#     plt.figure(figsize=(10, 8))
#     sns.histplot(df1[col], kde=True)
#     plt.title(f"Histogram of {col}")
#     plt.show()

# Utworzenie boxplotów dla zmiennych numerycznych
# for col in num_cols:
#     plt.figure(figsize=(10, 8))
#     sns.boxplot(x=df1[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

# Wykresy słupkowe dla zmiennych kategorycznych
# for col in cat_cols:
#     plt.figure(figsize=(10, 8))
#     sns.countplot(x=df1[col])
#     plt.title(f"Countplot of {col}")
#     plt.show()

# Macierz korelacji zmiennych numerycznych
correlation_matrix = df1[num_cols].corr()
print(correlation_matrix)
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True)
# plt.yticks(rotation=45)
# plt.title("Macierz korelacji zmiennych numerycznych")
# plt.show()

# Analiza wpływu zmiennych kategorycznych na koszt
# for col in cat_cols:
#     plt.figure(figsize=(10, 8))
#     sns.boxplot(x=df1["annual_medical_cost"], y=df1[col])
#     # Ograniczenie dla osi x dla poprawy widoczności boxplotów
#     cost_limit = df1["annual_medical_cost"].quantile(0.95)
#     plt.xlim(0, cost_limit)
#     plt.title(f"Rozkład kosztów ubezpieczenia wg {col}")
#     plt.xticks(rotation=45)
#     plt.show()

# Utworzenie modelu regresji liniowej
df_model = df1.copy()
# Przekształcenie danych kategorycznych do postaci numerycznej - One Hot Encoding
df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
# print(f"Liczba kolumn po One-Hot Encoding: {df_model.shape}")
X = df_model.drop("annual_medical_cost", axis=1)
y = df_model["annual_medical_cost"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Rozmiar zbioru treningowego: {X_train.shape[0]}")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]}")

scaler = StandardScaler()
num_cols.remove("annual_medical_cost")
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Metryki do oceny modelu
r2 = r2_score(y_test, y_pred)
rsme = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Współczynnik R-kwadrat: {r2:.4f}")
print(f"RSME: {rsme:.2f}")

# plt.figure(figsize=[10, 8])
# plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Punkty Danych")
# max_val = max(y_test.max(), y_pred.max())
# min_val = min(y_test.min(), y_pred.min())
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Linia Idealnej Predykcji (y=x)")
# plt.title("Rzeczywiste vs. Przewidywane Koszty Ubezpieczenia")
# plt.xlabel("Rzeczywiste Koszty Ubezpieczenia")
# plt.ylabel("Przewidywane Koszty Ubezpieczenia")
# plt.legend()
# plt.grid(True)
# plt.show()

# Model 2 - Random Forest Regressor
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("RF R2:", r2_score(y_test, rf_pred))
print("RF RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

# Model 3 - Gradient Boosting Regressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)
print("GBR R2:", r2_score(y_test, gbr_pred))
print("GBR RMSE:", np.sqrt(mean_squared_error(y_test, gbr_pred)))
