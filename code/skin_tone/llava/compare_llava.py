"""
***************************************************************
*  4a. ALTERNATIVE: LLAVA skin tone continues: Merge with truth and Compare
*
*  This script evaluates the performance of LLaVA's skin tone classification on the MST-E dataset by:
*  - Merging ground truth annotations with model predictions.
*  - Sanitizing and extracting predicted MST values from LLaVA's responses.
*  - Filtering out invalid or missing predictions.
*  - Computing key classification metrics: precision, recall, F1-score, and accuracy.
*  - Displaying and visualizing the confusion matrix to assess prediction patterns.
*  - Saving the cleaned, merged results to a new CSV file for further analysis.
*
*  This provides a comprehensive performance overview of LLaVA on skin tone recognition using the MST scale.
***************************************************************
"""

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the original preprocessed CSV and the results CSV
preprocessed_df = pd.read_csv("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/data/skin_tone/preprosessed_initial_MST_full_1510.csv")
results_df = pd.read_csv("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/results/llava/llava_less_strict.csv")

# Step 2: Merge the two dataframes on the image_ID (column in preprocessed_df and image_id in results_df)
merged_df = pd.merge(preprocessed_df, results_df, left_on='image_ID', right_on='image_id')

# Step 3: Filter the data to include only rows where the 'sanitized' column is not empty
merged_df_filtered = merged_df[merged_df['sanitized'].notna()]

# Step 4: Extract the ground truth (MST column, which should be an integer)
merged_df_filtered['ground_truth'] = merged_df_filtered['MST'].astype(int)

# Step 5: Extract the predicted integer from the 'sanitized' column (remove non-integer part)
# Use raw string notation to avoid invalid escape sequence warning
merged_df_filtered['prediction'] = merged_df_filtered['sanitized'].str.extract(r'(\d+)')

# Step 6: Check if there are any missing predictions and handle them
merged_df_filtered['prediction'] = merged_df_filtered['prediction'].astype(float)

# Check if there are any NaN values in the predictions (indicating no valid prediction was extracted)
nan_predictions = merged_df_filtered['prediction'].isna().sum()
print(f"Number of NaN predictions: {nan_predictions}")

# Step 7: Handle missing predictions
# Drop rows with NaN predictions to make sure only valid predictions are used in the analysis
merged_df_filtered = merged_df_filtered.dropna(subset=['prediction'])

# Convert predictions to integers now that there are no NaN values
merged_df_filtered['prediction'] = merged_df_filtered['prediction'].astype(int)

# Step 8: Check distribution of predictions, especially MST6
mst6_predictions = merged_df_filtered[merged_df_filtered['prediction'] == 6]
print(f"Number of predictions for MST6: {mst6_predictions.shape[0]}")

# Step 9: Check all rows where the prediction is MST6 (to see what it's being confused with)
print("Rows where prediction is MST6:")
print(mst6_predictions)

# Step 10: Compute the confusion matrix and accuracy
y_true = merged_df_filtered['ground_truth']
y_pred = merged_df_filtered['prediction']

# Step 11: Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(1, 11))  # MST labels are from 1 to 10

# Step 12: Calculate the accuracy score
accuracy = accuracy_score(y_true, y_pred)

# Step 13: Compute Precision, Recall, F1-Score, and Support
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(1, 11), average=None, zero_division=0)

# Step 14: Create a summary table for Precision, Recall, F1-Score, and Support
summary_df = pd.DataFrame({
    'precision': precision,
    'recall': recall,
    'f1-score': f1,
    'support': support
}, index=range(1, 11))  # MST labels are from 1 to 10

# Add macro and weighted averages
macro_avg = [
    precision.mean(), recall.mean(), f1.mean(), None
]
weighted_avg = [
    (precision * support).sum() / support.sum(),
    (recall * support).sum() / support.sum(),
    (f1 * support).sum() / support.sum(),
    None
]

summary_df.loc['macro avg'] = macro_avg
summary_df.loc['weighted avg'] = weighted_avg

# Add accuracy to the summary table
summary_df.loc['accuracy'] = [accuracy] * 4

# Step 15: Print the summary table
print("Classification Report (Precision, Recall, F1-Score, Support):")
print(summary_df)

# Step 16: Print the confusion matrix
print("\nConfusion Matrix:")
print(cm)

# Step 17: Print the accuracy score
print(f"\nAccuracy: {accuracy:.4f}")

# Step 18: Plot the confusion matrix using Seaborn for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=range(1, 11), yticklabels=range(1, 11))
plt.title('CONFUSION MATRIX (LLAVA - MST)')
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.show()

# Step 16: Save the merged and filtered dataframe to a new CSV for verification and record-keeping
merged_df_filtered.to_csv("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/results/llava/merged_llava_less_strict.csv", index=False)