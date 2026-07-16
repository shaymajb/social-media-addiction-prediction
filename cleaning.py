import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # prevents matplotlib from trying to open a display window
import matplotlib.pyplot as plt
 
#  CONFIGURATION — paths built automatically from script location
#  Works on any computer, no manual editing needed
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "data", "social_media_addiction.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "cleaned_social_media.csv")
 
os.makedirs(os.path.join(BASE_DIR, "data"),  exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
 
 
#  STEP 1 : LOADING THE DATA

print("  STEP 1 : DATA LOADING")
 
df = pd.read_csv(INPUT_FILE)
print(f"Rows loaded   : {len(df)}")
print(f"Columns       : {df.shape[1]}")
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
 
 
#  STEP 2 : DROP ROWS WITH NO TARGET LABEL
#
#  We cannot train a model on a row that has no answer.
#  These rows are deleted, not filled.
print("\n  STEP 2 : DROP ROWS WITH NO TARGET LABEL")
 
before = len(df)
df = df.dropna(subset=["Addiction_Level"])
print(f"Rows dropped (missing Addiction_Level) : {before - len(df)}")
print(f"Remaining rows : {len(df)}")
 
 
#  STEP 3 : TEXT STANDARDIZATION
#
#  Strip extra spaces + convert to UPPERCASE before anything else.
#  This ensures comparisons and mode calculations are correct.

print("\n  STEP 3 : TEXT STANDARDIZATION")
 
text_cols = ["Gender", "Platform", "Emotional_State_Post_Usage", "Addiction_Level"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace("NAN", float("nan"))  # restore real NaN values
 
# Normalize gender abbreviations: M : MALE, F : FEMALE
gender_map = {"M": "MALE", "F": "FEMALE"}
df["Gender"] = df["Gender"].replace(gender_map)
 
print("  Gender    — unique values :", sorted(df["Gender"].dropna().unique()))
print("  Platform  — unique values :", sorted(df["Platform"].dropna().unique()))
print("  Addiction — unique values :", sorted(df["Addiction_Level"].dropna().unique()))
 
 
#  STEP 4 : FILL MISSING VALUES
#
#  Numeric columns  : median  (more robust than mean against outliers)
#  Text columns     : mode    (most frequent value)
print("\n  STEP 4 : FILL MISSING VALUES")
 
# Numeric columns : median
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
for col in numeric_cols:
    missing = df[col].isna().sum()
    if missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col:<30} {missing} values : median = {median_val:.1f}")
 
# Text columns : mode
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    missing = df[col].isna().sum()
    if missing > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  {col:<30} {missing} values → mode = {mode_val}")
 
print(f"\nMissing values after filling : {df.isnull().sum().sum()}")
 
 
#  STEP 5 : REMOVE DUPLICATE ROWS
#
#  We ignore User_ID for detection — two rows can have different IDs
#  but be identical on all other columns (= a real duplicate).
#  Keeping duplicates would make the model memorize those rows
#  and perform unfairly well on them during testing.
print("\n  STEP 5 : REMOVE DUPLICATE ROWS")
 
cols_for_dup = [c for c in df.columns if c != "User_ID"]
before = len(df)
df = df.drop_duplicates(subset=cols_for_dup)
print(f"Duplicates removed : {before - len(df)}")
print(f"Remaining rows     : {len(df)}")
 
 
#  STEP 6 : OUTLIERS — Daily_Usage_Time_min
#
#  A day only has 1440 minutes.
#  Any value below 0 or above 1440 is physically impossible → removed.
print("\n  STEP 6 : HANDLE OUTLIERS")
 
# Boxplot BEFORE filter — saved to plots/ folder
plt.figure(figsize=(7, 4))
df.boxplot(column="Daily_Usage_Time_min")
plt.title("Daily_Usage_Time_min — BEFORE cleaning")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "boxplot_before.png"), dpi=150)
plt.close()
 
before = len(df)
df = df[(df["Daily_Usage_Time_min"] >= 0) & (df["Daily_Usage_Time_min"] <= 1440)]
print(f"Outliers removed (outside [0, 1440] min) : {before - len(df)}")
print(f"  Min after filter : {df['Daily_Usage_Time_min'].min()}")
print(f"  Max after filter : {df['Daily_Usage_Time_min'].max()}")
 
# Boxplot AFTER filter
plt.figure(figsize=(7, 4))
df.boxplot(column="Daily_Usage_Time_min")
plt.title("Daily_Usage_Time_min — AFTER cleaning")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "boxplot_after.png"), dpi=150)
plt.close()
 
print("  Boxplots saved in plots/")
 
 
#  STEP 7 : FINAL CHECK

print("\n  STEP 7 : FINAL CHECK")
 
print(f"  Final row count       : {len(df)}")
print(f"  Columns               : {df.shape[1]}")
print(f"  Missing values        : {df.isnull().sum().sum()}")
print(f"  Remaining duplicates  : {df.duplicated(subset=cols_for_dup).sum()}")  # BUG FIX: was always printing wrong value
print(f"\nAddiction_Level distribution:")
print(df["Addiction_Level"].value_counts())
 
 
#  STEP 8 : SAVE CLEANED FILE

print("\n  STEP 8 : SAVING")
 
df.to_csv(OUTPUT_FILE, index=False)
print(f"  Cleaned file saved : {OUTPUT_FILE}")
print("\n Data cleaning completed successfully!")