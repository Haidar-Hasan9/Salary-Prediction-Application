import pandas as pd
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import load_data, encode_categorical, split_data

# Load and prepare data
df = load_data()
X, y, mappings = encode_categorical(df)
X_train, X_test, y_train, y_test = split_data(X, y)

#Find better parameters automatically
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [15, 20, 25],
    'min_samples_leaf': [5, 10, 15]
}

grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), 
param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-val R²: {grid_search.best_score_:.4f}")

# Use the best model
model = grid_search.best_estimator_

print("Training Decision Tree Regressor...")
model = DecisionTreeRegressor(
    max_depth=4,            # severely limit depth
    min_samples_split=20,   # require more samples to split
    min_samples_leaf=10,    # minimum samples per leaf
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluation metrics
print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Training MAE: ${mean_absolute_error(y_train, y_pred_train):,.2f}")
print(f"Test MAE: ${mean_absolute_error(y_test, y_pred_test):,.2f}")
print(f"Training RMSE: ${np.sqrt(mean_squared_error(y_train, y_pred_train)):,.2f}")
print(f"Test RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")
print(f"Training R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")

# Save model
joblib.dump(model, "models/decision_tree_v1.pkl")
print("\nModel saved to models/decision_tree_v1.pkl")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': mappings['feature_cols'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)