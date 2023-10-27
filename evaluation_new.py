import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

# Load the predicted labels
predicted_labels = pd.read_csv("predicted_labels_tmp.csv", header=None)
gt = pd.read_csv("data/binary_values.tsv", delimiter='\t', index_col=0)
# Load the true labels (skip the "Argument ID" column)
true_labels_df = pd.read_csv("data/binary_test_labels.tsv", delimiter='\t', index_col=0)
# Ensure that the order of columns in both dataframes is the same
# if not (predicted_labels_df.columns == true_labels_df.columns).all():
#     raise ValueError("The columns of the two files do not match!")

# Compute the F1 score (macro)
all_values = [x for x in true_labels_df['value']]

unique_values = set(all_values)
total_f1 = 0.0

for value in unique_values:
    indexes = [i for i, v in enumerate(all_values) if v == value]
    value_predictions = predicted_labels[0][indexes]
    value_labels = true_labels_df['contains'][indexes]

    f1_for_value = f1_score(value_labels, value_predictions, average='binary', zero_division=0)
    total_f1 += f1_for_value

macro_f1 = total_f1 / len(unique_values)
print(f"F1 Score (Macro): {macro_f1:.4f}")

