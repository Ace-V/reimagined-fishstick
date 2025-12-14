import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("../data/Water_data.csv")

# Get unique categories for reference
categories = df['category'].unique()
print("Available categories:", categories)

# Create separate models for each category
models_dict = {}

for category in categories:
    print(f"\n--- Training model for {category} category ---")
    
    # Create binary labels: 1 if water matches this category (safe), 0 otherwise (unsafe)
    df_temp = df.copy()
    df_temp['is_safe'] = (df_temp['category'] == category).astype(int)
    
    X = df_temp[['flow', 'temperature', 'turbidity', 'tds', 'ph']]
    y = df_temp['is_safe']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"Training Accuracy: {round(train_acc, 4)}")
    print(f"Test Accuracy: {round(test_acc, 4)}")
    
    # Store model and scaler for this category
    models_dict[category] = {
        "model": model,
        "scaler": scaler
    }

# Save all models
with open("water_safety_models.pkl", "wb") as f:
    pickle.dump(models_dict, f)

print("\n✅ All category models saved as water_safety_models.pkl")
print(f"Categories included: {list(models_dict.keys())}")