#  Social Media Addiction Prediction
 
> A complete end-to-end Machine Learning project — from raw dirty data to trained classifiers — predicting a user's social media addiction level using behavioral and psychological features.
 
 
---
 
##  Objective
 
Predict whether a social media user is **Low**, **Moderate**, **High**, or **Severely** addicted based on their daily usage patterns, emotional state, and behavioral metrics — using 5 different classification algorithms and comparing their performance.
 
---
 
## used models
 1. Logistic Regression
 2. Decision Tree
 3. Random Forest
 4. SVM
 5. KNN

 
---
 
##  Dataset Features
 
| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | User's age |
| `Gender` | Categorical | Male / Female / Non-Binary |
| `Platform` | Categorical | Instagram, TikTok, YouTube, etc. |
| `Daily_Usage_Time_min` | Numeric | Minutes spent daily on social media |
| `Posts_Per_Day` | Numeric | Number of posts published per day |
| `Likes_Received_Daily` | Numeric | Daily likes received |
| `Comments_Received_Daily` | Numeric | Daily comments received |
| `Messages_Sent_Daily` | Numeric | Daily messages sent |
| `Scroll_Rate_ppm` | Numeric | Scrolling speed (pages per minute) |
| `FOMO_Score` | Numeric | Fear of Missing Out score (1–10) |
| `Mental_Health_Index` | Numeric | Mental health score (20–100, higher = healthier) |
| `Productivity_Loss_Score` | Numeric | Self-reported productivity loss (1–10) |
| `Emotional_State_Post_Usage` | Categorical | Mood after using social media |
| **`Addiction_Level`** | **Target** | **LOW / MODERATE / HIGH / SEVERE** |
 
**Dataset:** 1020 rows · 15 columns · Balanced classes (250 per category)
 
---
 
##  Data Cleaning Pipeline (`cleaning.py`)
 
The raw dataset contains real-world data quality issues that are handled step by step:
 
| Issue | Count | Fix Applied |
|---|---|---|
| Missing numeric values | ~80 per column | Filled with **median** |
| Missing categorical values | ~63 per column | Filled with **mode** |
| Missing target values | 5 rows | **Dropped** (can't train without label) |
| Duplicate rows | 20 | **Removed** with `drop_duplicates()` |
| Outliers (usage > 1440 min or < 0) | 15 rows | **Filtered** with logical range |
| Inconsistent text casing (`male`, `MALE`, `Male`) | Many | Standardized to **UPPERCASE + stripped** |
 
 
---
 
##  Visualizations Generated
 
1. **Accuracy bar chart** — compare all 5 models side by side
2. **Cross-validation boxplot** — shows reliability across 5 folds
3. **Confusion matrix heatmap** — detailed error analysis of the best model
4. **Feature importance chart** — which variables matter most (Random Forest)
---
 
##  How to Run
 
### 1. Clone the repository
```bash
git clone https://github.com/your-username/social-media-addiction-prediction.git
cd social-media-addiction-prediction
```
 
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 3. Run data cleaning first
```bash
python cleaning.py
```
→ Reads `data/social_media_addiction.csv`, outputs `data/cleaned_social_media.csv`
 
### 4. Run model training
```bash
python models.py
```
→ Prints all results in the console and saves 4 charts in `plots/`
 
---
 
##  Requirements
 
```
pandas
matplotlib
seaborn
scikit-learn
```
 
Install with:
```bash
pip install pandas matplotlib seaborn scikit-learn
```
 
---
 
## 👤 Author
 
**Chaima Jbeli** 