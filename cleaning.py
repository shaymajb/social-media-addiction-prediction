import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE  = "C:/Users/LENOVO/Desktop/data_mining_project/social-media-addiction-prediction/social_media_addiction.csv"
OUTPUT_FILE = "C:/Users/LENOVO/Desktop/data_mining_project/social-media-addiction-prediction/cleaned_social_media.csv"

os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)
 
#  ÉTAPE 1 : CHARGEMENT
print("  ÉTAPE 1 : CHARGEMENT DES DONNÉES")
 
df = pd.read_csv(INPUT_FILE)
print(f"Lignes chargées   : {len(df)}")
print(f"Colonnes          : {df.shape[1]}")
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum()[df.isnull().sum() > 0])
 
 
#  ÉTAPE 2 : SUPPRESSION DES LIGNES SANS CIBLE
print("  ÉTAPE 2 : SUPPRESSION DES LIGNES SANS CIBLE")
 
before = len(df)
df = df.dropna(subset=["Addiction_Level"])
print(f"Lignes supprimées (Addiction_Level manquant) : {before - len(df)}")
print(f"Lignes restantes : {len(df)}")
 
 
#  ÉTAPE 3 : STANDARDISATION DES TEXTES
#  (strip espaces + UPPERCASE) avant tout le reste
#  pour que les comparaisons et le mode soient corrects

print("  ÉTAPE 3 : STANDARDISATION DES TEXTES")
 
text_cols = ["Gender", "Platform", "Emotional_State_Post_Usage", "Addiction_Level"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace("NAN", float("nan"))   # remettre les vrais NaN
 
# Normalisation des abréviations Genre : M → MALE, F → FEMALE
gender_map = {"M": "MALE", "F": "FEMALE"}
df["Gender"] = df["Gender"].replace(gender_map)
 
print("  Gender   — valeurs uniques :", sorted(df["Gender"].dropna().unique()))
print("  Platform — valeurs uniques :", sorted(df["Platform"].dropna().unique()))
print("  Addiction— valeurs uniques :", sorted(df["Addiction_Level"].dropna().unique()))
 
 
#  ÉTAPE 4 : VALEURS MANQUANTES
print("  ÉTAPE 4 : REMPLISSAGE DES VALEURS MANQUANTES")
 
# Colonnes numériques : médiane (robuste aux outliers)
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
for col in numeric_cols:
    missing = df[col].isna().sum()
    if missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col:<30} {missing} valeurs → médiane = {median_val:.1f}")
 
# Colonnes textuelles : mode
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    missing = df[col].isna().sum()
    if missing > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  {col:<30} {missing} valeurs → mode = {mode_val}")
 
print(f"\nValeurs manquantes après remplissage : {df.isnull().sum().sum()}")
 
 

#  ÉTAPE 5 : SUPPRESSION DES DOUBLONS
print("  ÉTAPE 5 : SUPPRESSION DES DOUBLONS")
 
# On ignore User_ID pour la détection (deux entrées peuvent avoir des IDs différents
# mais être identiques sur toutes les autres colonnes)
cols_for_dup = [c for c in df.columns if c != "User_ID"]
before = len(df)
df = df.drop_duplicates(subset=cols_for_dup)
print(f"Doublons supprimés : {before - len(df)}")
print(f"Lignes restantes   : {len(df)}")
 
 
#  ÉTAPE 6 : OUTLIERS — Daily_Usage_Time_min
print("  ÉTAPE 6 : GESTION DES OUTLIERS")
 
# Boxplot AVANT filtre
plt.figure(figsize=(7, 4))
df.boxplot(column="Daily_Usage_Time_min")
plt.title("Daily_Usage_Time_min — AVANT nettoyage")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("plots/boxplot_before.png", dpi=150)
plt.close()
 
before = len(df)
df = df[(df["Daily_Usage_Time_min"] >= 0) & (df["Daily_Usage_Time_min"] <= 1440)]
print(f"Outliers supprimés (hors [0, 1440] min) : {before - len(df)}")
print(f"  Min après filtre : {df['Daily_Usage_Time_min'].min()}")
print(f"  Max après filtre : {df['Daily_Usage_Time_min'].max()}")
 
# Boxplot APRÈS filtre
plt.figure(figsize=(7, 4))
df.boxplot(column="Daily_Usage_Time_min")
plt.title("Daily_Usage_Time_min — APRÈS nettoyage")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("plots/boxplot_after.png", dpi=150)
plt.close()
 
print("  Boxplots sauvegardés dans plots/")
 
 
#  ÉTAPE 7 : VÉRIFICATION FINALE
print("  ÉTAPE 7 : VÉRIFICATION FINALE")
 
print(f"  Lignes finales        : {len(df)}")
print(f"  Colonnes              : {df.shape[1]}")
print(f"  Valeurs manquantes    : {df.isnull().sum().sum()}")
print(f"  Doublons restants     : {df.drop_duplicates(subset=cols_for_dup).shape[0] - len(df) + len(df)}")
print(f"\nDistribution Addiction_Level :")
print(df["Addiction_Level"].value_counts())
 
 
#  ÉTAPE 8 : SAUVEGARDE
print("  ÉTAPE 8 : SAUVEGARDE")
 
df.to_csv(OUTPUT_FILE, index=False)
print(f"  Fichier sauvegardé : {OUTPUT_FILE}")
print("\n Data cleaning terminé avec succès !")