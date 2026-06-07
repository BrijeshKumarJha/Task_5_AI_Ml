import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

df = pd.read_csv('pokemon.csv')
features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'base_total']
X = df[features]
y = df['is_legendary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

plt.figure(figsize=(16, 8))
plot_tree(dt_model, feature_names=features, class_names=['Normal', 'Legendary'], filled=True, rounded=True, proportion=False)
plt.title("Pokemon Legendary Predictor - Decision Tree")
plt.show()

y_pred_train = dt_model.predict(X_train)
y_pred_test = dt_model.predict(X_test)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

dt_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_overfit.fit(X_train, y_train)
y_pred_train_overfit = dt_overfit.predict(X_train)
y_pred_test_overfit = dt_overfit.predict(X_test)

print("--- 🚨 OVERFITTED TREE PERFORMANCE ---")
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train_overfit):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_test_overfit):.4f}")
print(f"Total Depth of this tree: {dt_overfit.get_depth()} levels!")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_test_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_test_pred_rf)
print("--- 🌳 RANDOM FOREST PERFORMANCE ---")
print(f"Testing Accuracy: {rf_accuracy:.4f}")

cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\n--- 🔄 CROSS-VALIDATION RESULTS (5 Folds) ---")
print(f"Scores for each fold: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}")

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature' : features,
    'Importance' : importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance' ,ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue='Feature', palette='viridis', legend=False)
plt.title('Pokemon Legendary Predictor: Feature Importances', fontsize=16)
plt.xlabel('Importance Score (Total = 1.0)', fontsize=14)
plt.ylabel('Stats', fontsize=14)
plt.show()

print("--- 🏆 TOP 3 MOST IMPORTANT STATS ---")
print(feature_importance_df.head(3))