"""
***************************************************************
*  Compare raw ouput (or the sanitized version) with the ground truth
*
*  This script evaluates the performance of LLaVA's gender classification by:
*  - Extracting the sanitized predictions from the csv
*  - And comparing those with the ground truth
*  - Computing key classification metrics: precision, recall, F1-score, and accuracy.
*  - Displaying and visualizing the confusion matrix to assess prediction patterns.
*
*  This provides a simple performance overview of LLaVA on gender recognition.
***************************************************************
"""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load classified gender file
df = pd.read_csv("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/results/gender/llava/gender_classification.csv") 

# Normalize all text (lowercase + strip)
df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

# Define normalization mapping
def normalize_gender(g):
    if g in ["man", "male", "boy"]:
        return "man"
    elif g in ["woman", "female", "girl"]:
        return "woman"
    else:
        return "undetermined"

#apply normalization to predictions only
df['normalized_pred'] = df['sanitized'].apply(normalize_gender)

#Set ground truth and prediction columns
y_true = df["gender"]
y_pred = df["normalized_pred"]

#Debugging
print("Unique values in y_true (gender):", y_true.unique())
print("Unique values in y_pred (normalized_pred):", y_pred.unique())

#Labels for confusion matrix 
true_labels = ["man", "woman"]
pred_labels = ["man", "woman", "undetermined"]

#classification metrics
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=true_labels, average=None)
accuracy = accuracy_score(y_true, y_pred)

#Build metrics summary
summary_df = pd.DataFrame({
    "precision": precision,
    "recall": recall,
    "f1-score": f1,
    "support": support
}, index=true_labels)

# Add undetermined stats
undetermined_count = (y_pred == "undetermined").sum()
undetermined_precision = undetermined_count / len(y_pred) if undetermined_count > 0 else 0
summary_df.loc["undetermined"] = [undetermined_precision, None, None, undetermined_count]

# Add macro and weighted averages
macro_avg = [precision.mean(), recall.mean(), f1.mean(), None]
weighted_avg = [
    (precision * support).sum() / support.sum(),
    (recall * support).sum() / support.sum(),
    (f1 * support).sum() / support.sum(),
    None
]

summary_df.loc["accuracy"] = [accuracy, accuracy, accuracy, len(y_true)]
summary_df.loc["macro avg"] = macro_avg
summary_df.loc["weighted avg"] = weighted_avg

#Print summary table 
print("\nCLASSIFICATION REPORT")
print(summary_df)

# Confusion matrix 
cm = confusion_matrix(y_true, y_pred, labels=pred_labels)

# Plot Confusion Matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=pred_labels,yticklabels=true_labels)
plt.title("CONFUSION MATRIX (LLAVA - GENDER)")
plt.xlabel("Prediction")
plt.ylabel("Ground truth")
plt.tight_layout()
plt.show()
