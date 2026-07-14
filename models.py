import os
import joblib          
import shap            
import mlflow  
import pandas as pd
import matplotlib 
matplotlib.use("Agg") # prevents matplotlib from opening windows when saving
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Starting the program......\n")

# STEP 1: LOADING AND ENCODING
# 1. Load the cleaned data by your first script
INPUT_FILE = "data/cleaned_social_media.csv"
OUTPUT_PLOTS = "plots"  
MODELS_DIR   = "models" 

os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
 
print("  STEP 1 : LOADING AND ENCODING")
 
df = pd.read_csv(INPUT_FILE)
 
# Remove User_ID — it's just an identifier, not a feature
if "User_ID" in df.columns:
    df = df.drop(columns=["User_ID"])
 
# Convert text labels to numbers so ML algorithms can understand them
# LOW=0, MODERATE=1, HIGH=2, SEVERE=3
addiction_mapping = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "SEVERE": 3}
df["Addiction_Level"] = df["Addiction_Level"].map(addiction_mapping)
df = df.dropna(subset=["Addiction_Level"])
df["Addiction_Level"] = df["Addiction_Level"].astype(int)
 
# X = all input features, y = what we want to predict
X = df.drop(columns=["Addiction_Level"])
y = df["Addiction_Level"]
 
# One-hot encoding: convert text columns (Gender, Platform...) to 0/1 columns
# Example: Platform_INSTAGRAM=1, Platform_TIKTOK=0, Platform_YOUTUBE=0...
X_encoded = pd.get_dummies(X, drop_first=True)
 
print(f"Dataset : {X_encoded.shape[0]} lignes, {X_encoded.shape[1]} features")
print(f"Classes : {y.value_counts().sort_index().to_dict()}")
 
print("  STEP 2 : PREPARATION TRAIN / TEST")


# 80% of data to train, 20% to test
# stratify=y ensures each split has the same proportion of LOW/MODERATE/HIGH/SEVERE
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y   # stratify = balances the classes
)

# StandardScaler: puts all numbers on the same scale
# Example: Age (13-60) and Daily_Usage (5-720) → both become values around 0
# IMPORTANT: fit only on train data, then apply same scale to test data
# If we fit on test too, the model "sees" test data during training — that's cheating
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # learn the scale from train, then apply
X_test_scaled  = scaler.transform(X_test)    # apply the SAME scale (don't relearn)
 
print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")
print("Encoding and normalisation finished !")
 
 
print("  STEP 3 : TRAINING THE 5 MODELS")
 
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),  # max_depth prevents overfitting
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"                 : SVC(probability=True),                                  # probability=True for future ROC figs
    "KNN"                 : KNeighborsClassifier(n_neighbors=7),                    # 7 neighbors, more robust than 5
}
 
results      = {}
cv_results   = {}
best_name    = ""
best_acc     = 0
best_y_pred  = None
best_model   = None
 
label_names = ["Low", "Moderate", "High", "Severe"]

mlflow.set_experiment("social-media-addiction")
for name, model in models.items():

    # NEW — each model gets its own MLflow "run" (one row in the dashboard)
    with mlflow.start_run(run_name=name):
    
        # Train
        model.fit(X_train_scaled, y_train)
        y_pred   = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    
        # Cross-validation: train+test 5 times on different slices
        # gives a more reliable score than a single test
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
        cv_results[name] = cv_scores

        # NEW — log everything to MLflow so you can compare runs later
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy_test",  accuracy)
        mlflow.log_metric("cv_mean",        cv_scores.mean())
        mlflow.log_metric("cv_std",         cv_scores.std())
 
        print(f"\n{'='*15} {name} {'='*15}")
        print(f"  Accuracy Test       : {accuracy * 100:.2f}%")
        print(f"  Cross-Val (5-fold)  : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
        print("  Details by class :")
        print(classification_report(
            y_test, y_pred,
            labels=[0, 1, 2, 3],
            target_names=label_names,
            zero_division=0
        ))
    
        if accuracy > best_acc:
            best_acc    = accuracy
            best_name   = name
            best_y_pred = y_pred
            best_model  = model
 
 
print(f"  BEST MODELE: {best_name}  ({best_acc*100:.2f}%)")


print("  ÉTAPE 4 : SAVING THE MODELE (joblib)")

# Right now the model exists only in RAM. When this script ends it disappears.
# joblib.dump() saves it as a .pkl file on disk (like a "photo" of the model).
# Later, the Streamlit app will do joblib.load() to reload it instantly
# without retraining — loading takes 0.1 seconds instead of 30 seconds.
joblib.dump(best_model,               f"{MODELS_DIR}/best_model.pkl")
joblib.dump(scaler,                   f"{MODELS_DIR}/scaler.pkl")
joblib.dump(list(X_encoded.columns),  f"{MODELS_DIR}/feature_names.pkl")
 
print(f"  Modèle sauvegardé    : {MODELS_DIR}/best_model.pkl")
print(f"  Scaler sauvegardé    : {MODELS_DIR}/scaler.pkl")
print(f"  Features sauvegardées: {MODELS_DIR}/feature_names.pkl")
 
print("\n  STEP 5 : GRAPHS")
 
# Chart 1 — accuracy comparison bar chart (5 models)
plt.figure(figsize=(10, 5))
colors = ["#378ADD", "#EF9F27", "#1D9E75", "#D85A30", "#7F77DD"]
bars = plt.bar(results.keys(), [v * 100 for v in results.values()], color=colors)
plt.title("Precision Comparison 5 models (Accuracy)", fontsize=13)
plt.ylabel("Precision (%)")
plt.ylim(0, 110)
for bar, v in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             v * 100 + 1.5, f"{v*100:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/accuracy_comparison.png", dpi=150)
plt.close()
print(f"  SAVED : {OUTPUT_PLOTS}/accuracy_comparison.png")
 
 
# Chart 2 — cross-validation boxplot
plt.figure(figsize=(10, 5))
cv_data  = [cv_results[n] * 100 for n in models.keys()]
plt.boxplot(cv_data, tick_labels=list(models.keys()), patch_artist=True,
            boxprops=dict(facecolor="#B5D4F4", color="#185FA5"),
            medianprops=dict(color="#185FA5", linewidth=2))
plt.title("Cross-Validation 5-fold by model", fontsize=13)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/cross_validation.png", dpi=150)
plt.close()
print(f"  SAVED : {OUTPUT_PLOTS}/cross_validation.png")
 
 
# Chart 3 — confusion matrix of best model
cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title(f"Matrice de Confusion : {best_name}", fontsize=13)
plt.xlabel("Prédictions de la machine")
plt.ylabel("Réalité (Vraies valeurs)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/confusion_matrix.png", dpi=150)
plt.close()
print(f"  SAVED : {OUTPUT_PLOTS}/confusion_matrix.png")
 
 
# Chart 4 — feature importance (Random Forest only)
rf_model   = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X_encoded.columns)
top10       = importances.sort_values(ascending=False).head(10)
 
plt.figure(figsize=(9, 5))
top10.sort_values().plot(kind="barh", color="#1D9E75")
plt.title("Top 10 features importantes (Random Forest)", fontsize=13)
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/feature_importance.png", dpi=150)
plt.close()
print(f"  SAVED : {OUTPUT_PLOTS}/feature_importance.png")


print("  STEP 6 : SHAP — EXPLAINABILITY")

# The model currently says "HIGH addiction" but doesn't say WHY.
# SHAP calculates, for each prediction, how much each feature
# pushed the score UP or DOWN compared to the average prediction.

# We use TreeExplainer because our best model is a tree-based model.
# It's much faster than the general SHAP explainer for trees.
print("  Calcul des valeurs SHAP (peut prendre 30 secondes)...")

# TreeExplainer works specifically for tree-based models (Random Forest, Decision Tree...)
# It's faster and more accurate than the generic SHAP explainer for trees
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_scaled)
# shap_values is a list of 4 arrays — one per class [LOW, MODERATE, HIGH, SEVERE]
# Each array has shape: (n_test_samples, n_features)

# Chart 5 — SHAP bar chart: mean absolute importance per feature across all classes
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=X_encoded.columns.tolist(),
    class_names=label_names,
    plot_type="bar",      # bar = mean |SHAP| per feature, averaged across classes
    show=False
)
plt.title("SHAP — Overall feature importance (all classes)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  SAVED : {OUTPUT_PLOTS}/shap_summary.png")
 
# Chart 6 — SHAP beeswarm for HIGH addiction class (index 2)
# Each dot = one test user. Red = high feature value, Blue = low feature value.
# Dots on the right = pushed prediction toward HIGH
# Dots on the left  = pushed prediction away from HIGH

import numpy as np
if isinstance(shap_values, list):
    # Old SHAP format: list of arrays, one per class
    shap_high = shap_values[2]
else:
    # New SHAP format: 3D array : slice the HIGH class (index 2)
    shap_high = shap_values[:, :, 2]

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_high,
    X_test_scaled,
    feature_names=X_encoded.columns.tolist(),
    plot_type="dot",          # dot = beeswarm, shows individual predictions
    show=False
)
plt.title("SHAP — Impact of features on the HIGH prediction", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/shap_beeswarm_high.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  SAVED : {OUTPUT_PLOTS}/shap_beeswarm_high.png")
 

print("  FINAL SUMMARY")
print(f"  Meilleur modèle  : {best_name}")
print(f"  Accuracy test    : {best_acc * 100:.2f}%")
print(f"\n  generated files :")
print(f"    plots/accuracy_comparison.png")
print(f"    plots/cross_validation.png")
print(f"    plots/confusion_matrix.png")
print(f"    plots/feature_importance.png")
print(f"    plots/shap_summary.png")
print(f"    plots/shap_beeswarm_high.png")
print(f"    models/best_model.pkl")
print(f"    models/scaler.pkl")
print(f"    models/feature_names.pkl")
print(f"\n  MLflow : Type 'mlflow ui' into your terminal to see the dashboard.")


print("Program completed successfully !")