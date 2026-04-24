import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement des données
file_path = "C:/Users/LENOVO/Desktop/data_mining_project/social-media-addiction-prediction/social_media_addiction.csv"
df = pd.read_csv(file_path)

print("Data Info:")
df.info()
print("\nMissing values avant nettoyage:\n", df.isnull().sum())

# 2. Gestion des valeurs manquantes
# IMPORTANT : On supprime les lignes où la variable cible est manquante
if "Addiction_Level" in df.columns:
    df = df.dropna(subset=["Addiction_Level"])

# Remplir les numériques avec la médiane (plus robuste)
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Remplir les catégorielles avec le mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Gestion des doublons
duplicates_count = df.duplicated().sum()
print(f"\nNumber of duplicates: {duplicates_count}")
if duplicates_count > 0:
    df = df.drop_duplicates()
    print("Doublons supprimés.")

# 4. Standardisation des variables textuelles
# On utilise une boucle et on ajoute .str.strip() pour enlever les espaces
cols_to_clean = ["Gender", "Platform", "Emotional_State_Post_Usage", "Addiction_Level"]
for col in cols_to_clean:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# 5. Gestion des valeurs aberrantes (Outliers)
df.boxplot(column="Daily_Usage_Time_min")
plt.title("Distribution du temps d'utilisation quotidien")
plt.show()

# Filtre logique : entre 0 et 1440 minutes
df = df[(df["Daily_Usage_Time_min"] >= 0) & (df["Daily_Usage_Time_min"] <= 1440)]

# 6. Sauvegarde
output_path = "C:/Users/LENOVO/Desktop/data_mining_project/social-media-addiction-prediction/cleaned_social_media.csv"
df.to_csv(output_path, index=False)

print("\nData cleaning completed successfully!")