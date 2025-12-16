import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset_preprocessing/Crop_Recommendation_preprocessing.csv")

X = df.drop(columns=["Crop"])
y = df["Crop"]

# =========================
# 2. Split data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. MLflow autolog
# =========================
mlflow.sklearn.autolog()

# =========================
# 4. Training model
# =========================
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # =========================
    # 5. Evaluasi
    # =========================
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
