import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

# Load the predicted labels
predicted_labels_df_fl = pd.read_csv("predicted_labels_focal_loss_50.csv", header=None)
predicted_labels_df_bce = pd.read_csv("predicted_labels_BCE.csv", header=None)
# Load the true labels (skip the "Argument ID" column)
true_labels_df = pd.read_csv("data/labels-test.tsv", delimiter='\t', index_col='Argument ID')

predicted_labels_df_fl.columns = true_labels_df.columns
predicted_labels_df_fl.index = true_labels_df.index
predicted_labels_df_bce.columns = true_labels_df.columns
predicted_labels_df_bce.index = true_labels_df.index
# Ensure that the order of columns in both dataframes is the same
# if not (predicted_labels_df.columns == true_labels_df.columns).all():
#     raise ValueError("The columns of the two files do not match!")

# Compute the F1 score (macro)
f1_macro = f1_score(true_labels_df.values, predicted_labels_df_fl.values, average='macro')
balanced_acc_fl, balanced_acc_bce = [], []
for i in range(20):
    b_acc_fl = balanced_accuracy_score(true_labels_df.values[:, i], predicted_labels_df_fl.values[:, i])
    b_acc_bce = balanced_accuracy_score(true_labels_df.values[:, i], predicted_labels_df_bce.values[:, i])
    balanced_acc_fl.append(b_acc_fl)
    balanced_acc_bce.append(b_acc_bce)
    # print(f"Balanced Accuracy Score: {b_acc_fl:.4f}")
balanced_acc_df_fl = pd.DataFrame(balanced_acc_fl, columns=['Focal loss'], index=true_labels_df.columns)
balanced_acc_df_bce = pd.DataFrame(balanced_acc_bce, columns=['BCE'], index=true_labels_df.columns)

balance_acc = pd.concat([balanced_acc_df_fl, balanced_acc_df_bce], axis=1)
print(f"F1 Score (Macro): {f1_macro:.4f}")

