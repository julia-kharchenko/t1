import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# === 1. Завантаження даних ===
train_df = pd.read_csv("PRO2/m43/train.csv")
test_df = pd.read_csv("PRO2/m43/test.csv")

# === Функція для розрахунку віку ===
def calc_age(bdate):
    try:
        d = pd.to_datetime(bdate, format="%d.%m.%Y", errors="coerce")
        if pd.notnull(d):
            return (pd.Timestamp.today() - d).days // 365
    except:
        return None
    return None

# === Обробка тренувальних даних ===
train_df["age"] = train_df["bdate"].apply(calc_age)
train_df["age"] = train_df["age"].fillna(train_df["age"].median())
train_df["followers_count"] = train_df["followers_count"].fillna(0)

# кількість мов
train_df["langs_count"] = train_df["langs"].fillna("").apply(lambda x: len(x.split(",")) if x != "" else 0)

# career_start / career_end у числа
train_df["career_start"] = pd.to_numeric(train_df["career_start"], errors="coerce")
train_df["career_end"]   = pd.to_numeric(train_df["career_end"], errors="coerce")

# тривалість кар'єри
train_df["career_years"] = (train_df["career_end"] - train_df["career_start"]).clip(lower=0)

# === Обробка тестових даних (аналогічно!) ===
test_df["age"] = test_df["bdate"].apply(calc_age)
test_df["age"] = test_df["age"].fillna(train_df["age"].median())
test_df["followers_count"] = test_df["followers_count"].fillna(0)
test_df["langs_count"] = test_df["langs"].fillna("").apply(lambda x: len(x.split(",")) if x != "" else 0)

test_df["career_start"] = pd.to_numeric(test_df["career_start"], errors="coerce")
test_df["career_end"]   = pd.to_numeric(test_df["career_end"], errors="coerce")
test_df["career_years"] = (test_df["career_end"] - test_df["career_start"]).clip(lower=0)

# === Вибір ознак ===
features = [
    "sex", "age", "has_photo", "has_mobile", "followers_count",
    "graduation", "relation", "life_main", "people_main",
    "langs_count", "career_years"
]

X_train = train_df[features].copy()
y_train = train_df["result"]
X_test = test_df[features].copy()

# === Перетворення категоріальних у числові ===
for col in X_train.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# === Заповнення пропусків ===
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# === Масштабування ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Навчання моделі ===
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train_scaled, y_train)

# === Прогноз для тестових даних ===
test_pred = knn.predict(X_test_scaled)

# === Збереження у файл CSV ===
submission = pd.DataFrame({
    "id": test_df["id"],   # якщо у test.csv є колонка id
    "result": test_pred
})
submission.to_csv("PRO2/m43/submission.csv", index=False)

print("Прогноз збережено у файл submission.csv")
