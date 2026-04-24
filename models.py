import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("Démarrage du programme...")

# ÉTAPE 1 : CHARGEMENT ET ENCODAGE 
# 1. Charger les données nettoyées par ton premier script
file_path = "C:/Users/LENOVO/Desktop/data_mining_project/social-media-addiction-prediction/cleaned_social_media.csv"
df = pd.read_csv(file_path)

# 2. Supprimer la colonne User_ID 
if "User_ID" in df.columns:
    df = df.drop(columns=["User_ID"])

# 3. Encodage ordinal de la cible : On transforme les mots en niveaux (0 à 3)
addiction_mapping = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "SEVERE": 3}
df["Addiction_Level"] = df["Addiction_Level"].map(addiction_mapping)

# On supprime les lignes où l'addiction n'a pas été reconnue
df = df.dropna(subset=["Addiction_Level"])

# 4. Séparation : X (les indices) et y (ce qu'on cherche à prédire)
X = df.drop(columns=["Addiction_Level"])
y = df["Addiction_Level"]

# 5. Encodage des autres textes (Genre, Plateforme..) en 0 et 1
X_encoded = pd.get_dummies(X, drop_first=True)

# ÉTAPE 2 : PRÉPARATION POUR LE MACHINE LEARNING
# 1. Découpage : 80% pour apprendre (Train) et 20% pour tester (Test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 2. Normalisation : Mettre tous les chiffres sur la même échelle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Encodage et préparation terminés !")
print("Lancement des 5 modèles...\n")

# ÉTAPE 3 : ENTRAÎNEMENT DES 5 MODÈLES
# On prépare nos 5 modeles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

results = {}
best_model_name = ""
best_accuracy = 0
best_y_pred = None

# On fait passer un test à chaque modèle
for name, model in models.items():
    # Le modèle apprend (fit)
    model.fit(X_train_scaled, y_train)
    
    # Le modèle essaie de predir sur les données de test
    y_pred = model.predict(X_test_scaled)
    
    # On calcule sa note (précision)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"-> {name} : {accuracy * 100:.2f}% de bonnes réponses")
    
    # On garde en mémoire le meilleur modèle pour la matrice de confusion
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_y_pred = y_pred

# ÉTAPE 4 : CRÉATION DES GRAPHIQUES 
print(f"\nLe meilleur modèle est {best_model_name} !")
print("Affichage des graphiques...")

# Graphique 1 : Comparaison des 5 modèles (Bar Chart)
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [acc * 100 for acc in results.values()], color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title('Comparaison de la précision des 5 modèles (Accuracy)')
plt.ylabel('Précision (%)')
plt.ylim(0, 100) # L'axe Y va de 0 à 100%
for i, v in enumerate(results.values()):
    plt.text(i, (v * 100) + 1, f"{v * 100:.1f}%", ha='center') # Affiche le % sur les barres
plt.show()

# Graphique 2 : Matrice de confusion du MEILLEUR modèle
cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(6, 5))
# On utilise seaborn pour faire une belle matrice colorée
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Low", "Moderate", "High", "Severe"], 
            yticklabels=["Low", "Moderate", "High", "Severe"])
plt.title(f'Matrice de Confusion : {best_model_name}')
plt.xlabel('Prédictions de la machine')
plt.ylabel('Réalité (Vraies valeurs)')
plt.show()