import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("data/water_data.csv")

X = df[['flow', 'temperature', 'turbidity', 'tds', 'ph']]
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print("Training Accuracy:", round(train_acc, 4))
print("Test Accuracy:", round(test_acc, 4))

with open("water_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler
    }, f)

print("\nModel saved as water_model.pkl")
