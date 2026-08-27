# %%
# ............................................................................
# IMPORTING LIBRARIES
# ............................................................................
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# %%
# -----------------------------------------------------------------------------
# 1. LOADING THE FEATURE TABLE
# -----------------------------------------------------------------------------
# BEAT and FS are dropped: BEAT is just an index and FS is constant across
# rows, so neither carries predictive information.
df = pd.read_csv("ecg_features.csv")
x = df.drop(["BEAT", "FS", "PVC(1 FOR PVC, 0 FOR NORMAL)"], axis=1)
y = df["PVC(1 FOR PVC, 0 FOR NORMAL)"]

# %%
# -----------------------------------------------------------------------------
# 2. FEATURE ENCODING
# -----------------------------------------------------------------------------
# Convert the boolean COMPENSATORY column to 0/1 for the linear model.
if "COMPENSATORY" in x.columns:
    x["COMPENSATORY"] = x["COMPENSATORY"].astype(int)

# %%
# -----------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
# 80/20 split; random_state=42 for reproducibility.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# %%
# -----------------------------------------------------------------------------
# 4. MODEL TRAINING — LOGISTIC REGRESSION (BASELINE)
# -----------------------------------------------------------------------------
# Logistic Regression as a simple, interpretable baseline for this
# binary classification task.
model = LogisticRegression()
model.fit(x_train, y_train)

print(df.columns.tolist())
print(y.value_counts())

# %%
# -----------------------------------------------------------------------------
# 5. EVALUATION — LOGISTIC REGRESSION
# -----------------------------------------------------------------------------
# Confusion matrix + classification report, since class imbalance
# (few PVCs vs many normal beats) makes plain accuracy misleading.
y_pred = model.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
# -----------------------------------------------------------------------------
# 6. MODEL TRAINING — RANDOM FOREST
# -----------------------------------------------------------------------------
# A Random Forest is tried as a stronger, non-linear alternative to the
# logistic baseline, since it can capture feature interactions the linear
# model can't. class_weight='balanced' compensates for the PVC/normal
# class imbalance by up-weighting the minority (PVC) class during training,
# instead of resampling the data. max_depth and min_samples_split are kept
# conservative to reduce overfitting given the small dataset.
model_2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=4,
    random_state=58,
    class_weight='balanced'
)
model_2.fit(x_train, y_train)

# %%
# -----------------------------------------------------------------------------
# 7. EVALUATION — RANDOM FOREST
# -----------------------------------------------------------------------------
# Same metrics as the baseline, so the two models can be compared directly.
y_pred_2 = model_2.predict(x_test)
print(classification_report(y_test, y_pred_2))
print(confusion_matrix(y_test, y_pred_2))

# %%
# -----------------------------------------------------------------------------
# 8. FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
# Ranks each feature by how much it contributed to the Random Forest's
# splits, giving a sanity check on whether the engineered features
# (RR timing, correlation, compensatory pause) are actually driving the
# model's decisions, and which one matters most.
importances = pd.Series(model_2.feature_importances_, index=x.columns)
print(importances.sort_values(ascending=False))

# %%
