import os
import pandas as pd
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

print("Démarrage du programme...\n")

# ÉTAPE 1 : CHARGEMENT ET ENCODAGE 
# 1. Charger les données nettoyées par ton premier script
INPUT_FILE = "C:/Users/LENOVO/Desktop/data_mining_project/social-media-addiction-prediction/cleaned_social_media.csv"
df = pd.read_csv(INPUT_FILE)
OUTPUT_PLOTS = "C:/Users/LENOVO/Desktop/data_mining_project/social-media-addiction-prediction/plots"  

os.makedirs(OUTPUT_PLOTS, exist_ok=True)
 
print("  ÉTAPE 1 : CHARGEMENT ET ENCODAGE")
 
df = pd.read_csv(INPUT_FILE)
 
# Supprimer la colonne User_ID si elle existe
if "User_ID" in df.columns:
    df = df.drop(columns=["User_ID"])
 
# Encodage ordinal de la cible
addiction_mapping = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "SEVERE": 3}
df["Addiction_Level"] = df["Addiction_Level"].map(addiction_mapping)
df = df.dropna(subset=["Addiction_Level"])
df["Addiction_Level"] = df["Addiction_Level"].astype(int)
 
# Séparation X / y
X = df.drop(columns=["Addiction_Level"])
y = df["Addiction_Level"]
 
# Encodage one-hot des colonnes textuelles
X_encoded = pd.get_dummies(X, drop_first=True)
 
print(f"Dataset : {X_encoded.shape[0]} lignes, {X_encoded.shape[1]} features")
 
 
print("  ÉTAPE 2 : PRÉPARATION TRAIN / TEST")
 
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y   # stratify = équilibre les classes
)
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
 
print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")
print("Encodage et normalisation terminés !")
 
 
print("  ÉTAPE 3 : ENTRAÎNEMENT DES 5 MODÈLES")
 
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),  # max_depth évite le surapprentissage
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"                 : SVC(probability=True),                                  # probability=True pour les courbes ROC futures
    "KNN"                 : KNeighborsClassifier(n_neighbors=7),                    # 7 voisins, plus robuste que 5
}
 
results      = {}
cv_results   = {}
best_name    = ""
best_acc     = 0
best_y_pred  = None
best_model   = None
 
label_names = ["Low", "Moderate", "High", "Severe"]
 
for name, model in models.items():
    # --- Entraînement ---
    model.fit(X_train_scaled, y_train)
    y_pred   = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
 
    # --- Cross-validation (5 folds) ---
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_results[name] = cv_scores
 
    print(f"\n{'='*15} {name} {'='*15}")
    print(f"  Accuracy Test       : {accuracy * 100:.2f}%")
    print(f"  Cross-Val (5-fold)  : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    print("  Détails par classe :")
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
 
 
print(f"  MEILLEUR MODÈLE : {best_name}  ({best_acc*100:.2f}%)")
 
 
print("\n  ÉTAPE 4 : GRAPHIQUES")
 
# --- Graphique 1 : Comparaison Accuracy des 5 modèles ---
plt.figure(figsize=(10, 5))
colors = ["#378ADD", "#EF9F27", "#1D9E75", "#D85A30", "#7F77DD"]
bars = plt.bar(results.keys(), [v * 100 for v in results.values()], color=colors)
plt.title("Comparaison de la précision des 5 modèles (Accuracy)", fontsize=13)
plt.ylabel("Précision (%)")
plt.ylim(0, 110)
for bar, v in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             v * 100 + 1.5, f"{v*100:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/accuracy_comparison.png", dpi=150)
plt.show()
print(f"  Graphique sauvegardé : {OUTPUT_PLOTS}/accuracy_comparison.png")
 
 
# --- Graphique 2 : Cross-Validation (boxplot) ---
plt.figure(figsize=(10, 5))
cv_data  = [cv_results[n] * 100 for n in models.keys()]
plt.boxplot(cv_data, labels=models.keys(), patch_artist=True,
            boxprops=dict(facecolor="#B5D4F4", color="#185FA5"),
            medianprops=dict(color="#185FA5", linewidth=2))
plt.title("Cross-Validation 5-fold par modèle", fontsize=13)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/cross_validation.png", dpi=150)
plt.show()
print(f"  Graphique sauvegardé : {OUTPUT_PLOTS}/cross_validation.png")
 
 
# --- Graphique 3 : Matrice de confusion du meilleur modèle ---
cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title(f"Matrice de Confusion : {best_name}", fontsize=13)
plt.xlabel("Prédictions de la machine")
plt.ylabel("Réalité (Vraies valeurs)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS}/confusion_matrix.png", dpi=150)
plt.show()
print(f"  Graphique sauvegardé : {OUTPUT_PLOTS}/confusion_matrix.png")
 
 
# --- Graphique 4 : Feature Importance (Random Forest uniquement) ---
if "Random Forest" in models:
    rf_model   = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=X_encoded.columns)
    top10       = importances.sort_values(ascending=False).head(10)
 
    plt.figure(figsize=(9, 5))
    top10.sort_values().plot(kind="barh", color="#1D9E75")
    plt.title("Top 10 features importantes (Random Forest)", fontsize=13)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/feature_importance.png", dpi=150)
    plt.show()
    print(f"  Graphique sauvegardé : {OUTPUT_PLOTS}/feature_importance.png")
 
print("\nTous les graphiques sont sauvegardés dans le dossier :", OUTPUT_PLOTS)
print("Programme terminé avec succès !")