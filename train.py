import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Starting Wine Quality training script...")

df = pd.read_csv("wine_data.csv")
print("Dataset loaded successfully.")

X = df.drop("class_label", axis=1)
y = df["class_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split into training and testing sets.")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training RandomForest model...")
model.fit(X_train, y_train)
print("Model training complete.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

model_filename = "wine_quality_model.joblib"
joblib.dump(model, model_filename)

print(f"Model saved as {model_filename}. Training script finished.")
