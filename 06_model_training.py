# %%
# ............................................................................
# IMPORTING LIBRARIES
# ............................................................................
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
# 4. MODEL TRAINING
# -----------------------------------------------------------------------------
# Logistic Regression as a simple, interpretable baseline for this
# binary classification task.
model = LogisticRegression()
model.fit(x_train, y_train)

print(df.columns.tolist())
print(y.value_counts())

# %%
# -----------------------------------------------------------------------------
# 5. EVALUATION
# -----------------------------------------------------------------------------
# Confusion matrix + classification report, since class imbalance
# (few PVCs vs many normal beats) makes plain accuracy misleading.
y_pred = model.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
