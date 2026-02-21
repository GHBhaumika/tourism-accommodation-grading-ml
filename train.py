import pandas as pd
import pickle
import os
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("data/Accommodation.csv")

# Fix incorrect column name from dataset
df = df.rename(columns={'Logitiute': 'Longitude'})

print("\n========== ORIGINAL DATASET PREVIEW ==========")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")
print(df.head())
print("=============================================\n")


# ============================================================
# STEP 2: DROP UNNECESSARY COLUMNS
# ============================================================

df = df.drop(columns=['Name', 'Address', 'PS/MC/UC', 'AGA Division'], errors='ignore')

print("\nColumns after dropping unnecessary ones:")
print(df.columns.tolist())


# ============================================================
# STEP 3: CLEANING
# ============================================================

df = df.drop_duplicates()
df = df.dropna(subset=['Grade'])
df = df.dropna()

print("\n========== DATASET AFTER CLEANING ==========")
print(f"Total Rows after cleaning: {df.shape[0]}")
print(f"Total Columns after cleaning: {df.shape[1]}")
print(df.head())
print("============================================\n")


# ============================================================
# STEP 4: MAP GRADE INTO 3 MAIN LEVELS
# ============================================================

def map_grade(grade):
    grade = str(grade).upper()
    if grade in ['DELUXE', 'SUPERIOR', 'FIVE']:
        return 'High'
    elif grade in ['FOUR', 'STANDARD', 'THREE']:
        return 'Medium'
    else:
        return 'Low'

df['Grade_clean'] = df['Grade'].apply(map_grade)

print("\nNew Grade Distribution:")
print(df['Grade_clean'].value_counts())


# ============================================================
# STEP 5: DEFINE TARGET AND FEATURES
# ============================================================

y = df['Grade_clean']
X = df[['Rooms', 'Latitude', 'Longitude', 'District', 'Type']]


# ============================================================
# STEP 6: ENCODE TARGET VARIABLE
# ============================================================

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

print("\nEncoded Target Classes:")
print(target_encoder.classes_)


# ============================================================
# STEP 7: PREPROCESSING FEATURES
# ============================================================

numeric_features = ['Rooms', 'Latitude', 'Longitude']
categorical_features = ['District', 'Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# ============================================================
# STEP 8: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# ============================================================
# STEP 9: CREATE PIPELINE (Preprocessing + Extra Trees)
# ============================================================

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ExtraTreesClassifier(
        n_estimators=200,
        random_state=42
    ))
])


# ============================================================
# STEP 10: TRAIN MODEL
# ============================================================

model_pipeline.fit(X_train, y_train)

# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

classifier = model_pipeline.named_steps['classifier']
preprocessor = model_pipeline.named_steps['preprocessor']

# Get feature names after encoding
ohe = preprocessor.named_transformers_['cat']
encoded_cat = ohe.get_feature_names_out(['District', 'Type'])

feature_names = ['Rooms', 'Latitude', 'Longitude'] + list(encoded_cat)

importances = classifier.feature_importances_

# Sort features
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importance (Extra Trees)")
plt.bar(range(15), importances[indices][:15])
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=90)
plt.tight_layout()
plt.show()




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_pipeline, X, y, cv=cv)
print("Cross Validation Accuracy:", cv_scores.mean())
print("Cross Validation Accuracy Mean:", cv_scores.mean())
print("Cross Validation Accuracy Std:", cv_scores.std())
# ============================================================
# STEP 11: EVALUATE MODEL
# ============================================================

y_pred = model_pipeline.predict(X_test)

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred,  target_names=target_encoder.classes_))
print("======================================\n")


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ============================================================
# STEP 12: SHAP ANALYSIS
# ============================================================

import numpy as np

classifier = model_pipeline.named_steps['classifier']
X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)

# Convert sparse to dense if needed
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

X_test_transformed = np.array(X_test_transformed, dtype=float)

explainer = shap.TreeExplainer(classifier)

print("\nGenerating SHAP values...")
shap_values = explainer.shap_values(X_test_transformed[:200])

shap.summary_plot(
    shap_values,
    X_test_transformed[:200],
    feature_names=feature_names,
)


# ============================================================
# STEP 13: SAVE MODEL AND ENCODER
# ============================================================

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

with open("models/target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

print("Model and encoders saved successfully!")
