import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# DataFrame wczytany z csv
df = pd.read_csv("medical_insurance.csv")
# rows, columns = df.shape
# print(df.head())
# print(f"Number of rows: {rows}, Number of columns: {columns}")
print(df.info())

# Wybranie kilku najważniejszych kolumn z pełnego DataFrame
df1 = df[
    ["age", "sex", "education", "smoker", "alcohol_freq", "visits_last_year", "hospitalizations_last_3yrs",
     "days_hospitalized_last_3yrs", "network_tier", "risk_score", "annual_medical_cost", "claims_count",
     "avg_claim_amount", "total_claims_paid", "chronic_count", "hypertension", "is_high_risk", "had_major_procedure"]]
print(df1.info())

# Podział danych na zmienne numeryczne i kategoryczne
num_cols = ["age", "visits_last_year", "hospitalizations_last_3yrs", "days_hospitalized_last_3yrs",
            "risk_score", "annual_medical_cost", "claims_count", "avg_claim_amount", "total_claims_paid",
            "chronic_count", "hypertension", "is_high_risk", "had_major_procedure"]
cat_cols = ["sex", "education", "smoker", "alcohol_freq", "network_tier"]

# W komórkach alcohol_freq None było odczytywane przez pandas jako brak danych - zamiana None na Never
df1["alcohol_freq"] = df1["alcohol_freq"].fillna("Never")
print(df1.info())

# Sprawdzenie czy w datasecie są puste komórki
print(f"Liczba pustych komórek w kolumnie:\n{df1.isnull().sum()}")

# Walidacja danych numerycznych
for col in num_cols:
    print(f"Czy w kolumnie {col} są dane ujemne?: {(df1[col] < 0).any()}")

# Wykrywanie obserwacji odstających metodą 1.5 IQR
# Kolumny, których outliery nie powinny być brane pod uwagę
filter_cols = ["age", "hypertension", "is_high_risk", "had_major_procedure", "days_hospitalized_last_3yrs",
               "hospitalizations_last_3yrs", "visits_last_year", "risk_score", "chronic_count", "claims_count"]
num_cols_filtered = [
    col for col in num_cols if col not in filter_cols
]

print(num_cols_filtered)

for col in num_cols_filtered:
    # Obliczenie kwartyli, IQR i progów
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Zliczenie obserwacji odstających
    outliers_low = df1[col][df1[col] < lower_bound].count()
    outliers_high = df1[col][df1[col] > upper_bound].count()
    total_outliers = outliers_low + outliers_high

    print(f"\nObserwacje odstające dla kolumny {col}:")
    print(f"Próg dolny i górny: ({lower_bound:.2f}, {upper_bound:.2f})")
    print(f"Obserwacje odstające z dołu: {outliers_low}")
    print(f"Obserwacje odstające z góry: {outliers_high}")
    print(f"Łącznie: {total_outliers} ({total_outliers / len(df1) * 100:.2f}%)")

# Wyświetlenie informacji o danych przed modyfikacją outlierów
# for col in num_cols_filtered:
#     print(df1[col].describe())

# Winsoryzacja outlierów
print("Przed winsoryzacją:")
print(df1['annual_medical_cost'].describe())
df_winsorized = df1.copy()

for col in num_cols_filtered:
    upper_limit = df_winsorized[col].quantile(0.99)
    df_winsorized[col] = df_winsorized[col].clip(0, upper_limit)

print("Po winsoryzacji:")
print(df_winsorized['annual_medical_cost'].describe())

# Wyświetlenie informacji o danych po winsoryzacji
# for col in num_cols_filtered:
#     print(df_winsorized[col].describe())

# Utworzenie histogramów dla zmiennych numerycznych
# Przed winsoryzacją
# for col in num_cols_filtered:
#     plt.figure(figsize=(10, 8))
#     sns.histplot(df1[col], kde=True)
#     plt.title(f"Histogram of {col}")
#     plt.show()

# Po winsoryzacji
# for col in num_cols_filtered:
#     plt.figure(figsize=(10, 8))
#     sns.histplot(df_winsorized[col], kde=True)
#     plt.title(f"Histogram of {col}")
#     plt.show()

# Utworzenie boxplotów dla zmiennych numerycznych
# Przed winsoryzacją outlierów
# for col in num_cols_filtered:
#     plt.figure(figsize=(10, 8))
#     sns.boxplot(x=df1[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

# Po winsoryzacji
# for col in num_cols_filtered:
#     plt.figure(figsize=(10, 8))
#     sns.boxplot(x=df_winsorized[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

# Wykresy słupkowe dla zmiennych kategorycznych
# for col in cat_cols:
#     plt.figure(figsize=(10, 8))
#     sns.countplot(x=df1[col])
#     plt.title(f"Countplot of {col}")
#     plt.show()

# Macierz korelacji zmiennych numerycznych
# correlation_matrix = df1[num_cols].corr()
# print(correlation_matrix)
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True)
# plt.yticks(rotation=25)
# plt.xticks(rotation=25)
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
df_model = df_winsorized.copy()
# print(df_model.info())
# Przekształcenie danych kategorycznych do postaci numerycznej - One Hot Encoding
df_model = pd.get_dummies(df_model, columns=cat_cols)
# print(f"Liczba kolumn po One-Hot Encoding: {df_model.shape}")
X = df_model.drop("annual_medical_cost", axis=1)
y = df_model["annual_medical_cost"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Rozmiar zbioru treningowego: {X_train.shape[0]}")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]}")

current_num_cols = [col for col in num_cols_filtered if col in X_train.columns and col != "annual_medical_cost"]
# print(current_num_cols)
scaler = StandardScaler()
X_train[current_num_cols] = scaler.fit_transform(X_train[current_num_cols])
X_test[current_num_cols] = scaler.transform(X_test[current_num_cols])

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Metryki do oceny modelu
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"--- Linear Regression  ---")
print(f"Współczynnik R-kwadrat: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# plt.figure(figsize=[10, 8])
# plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Punkty Danych")
# max_val = max(y_test.max(), y_pred.max())
# min_val = min(y_test.min(), y_pred.min())
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Linia Idealnej Predykcji (y_pred=y_test)")
# plt.title("Rzeczywiste vs. Przewidywane koszty medyczne")
# plt.xlabel("Rzeczywiste koszty medyczne")
# plt.ylabel("Przewidywane koszty medyczne")
# plt.legend()
# plt.grid(True)
# plt.show()

# Dobór hiperparametrów do modelu Random Forest
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
# }
# rf_base = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid,
#                            cv=5, scoring='neg_mean_squared_error',
#                            n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)
# best_rf = grid_search.best_estimator_
# y_pred_rf = best_rf.predict(X_test)
# r2_rf = r2_score(y_test, y_pred_rf)
# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
#
# print(f"Najlepsze parametry: {grid_search.best_params_}")
# print(f"--- Random Forest (Po optymalizacji) ---")
# print(f"Współczynnik R-kwadrat: {r2_rf:.4f}")
# print(f"RMSE: {rmse_rf:.2f}")

# Model Random Forest
# rf_model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_split=10, max_depth=10)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)
# r2_rf = r2_score(y_test, y_pred_rf)
# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
# print(f"--- Random Forest ---")
# print(f"Współczynnik R-kwadrat: {r2_rf:.4f}")
# print(f"RMSE: {rmse_rf:.2f}")

# importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
# importances.nlargest(10).plot(kind='barh')
# plt.title("10 najważniejszych cech wg Random Forest")
# plt.show()

# Dobór parametrów do XGBoost
# param_grid_xgb = {
#     'n_estimators': [200, 300, 400],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.7, 0.8, 0.9],
#     'colsample_bytree': [0.7, 0.8]
# }
#
# xgb_base = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
# grid_xgb = GridSearchCV(
#     estimator=xgb_base,
#     param_grid=param_grid_xgb,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
#     verbose=1
# )
#
# grid_xgb.fit(X_train, y_train)
# best_xgb = grid_xgb.best_estimator_
# y_pred_xgb = best_xgb.predict(X_test)
# r2_xgb = r2_score(y_test, y_pred_xgb)
# rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
#
# print(f"Najlepsze parametry: {grid_xgb.best_params_}")
# print(f"--- XGBoost (Po optymalizacji) ---")
# print(f"Współczynnik R-kwadrat: {r2_xgb:.4f}")
# print(f"RMSE: {rmse_xgb:.2f}")

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.7,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"--- XGBoost ---")
print(f"Współczynnik R-kwadrat: {r2_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.2f}")

# importances = xgb_model.feature_importances_
# feature_names = X_train.columns
# feature_importance_df = pd.DataFrame({'cecha': feature_names, 'waznosc': importances})
# feature_importance_df = feature_importance_df.sort_values(by='waznosc', ascending=False).head(15)
#
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['cecha'], feature_importance_df['waznosc'], color='skyblue')
# plt.xlabel('Ważność (Gain)')
# plt.title('Top 15 zmiennych budujących model kosztów medycznych')
# plt.gca().invert_yaxis()
# plt.show()
