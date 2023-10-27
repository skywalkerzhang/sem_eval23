import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from transformers import DebertaV2Tokenizer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import HumanValueDataset
from module import DeBERTaClassifier
import wandb
from utils import set_seeds


def matrix_to_categories(row):
    categories = row.index.tolist()
    active_categories = [categories[idx] for idx, val in enumerate(row) if val == 1]
    return " | ".join(active_categories)


def expand_dataframe(arguments_df, labels_df):
    expanded_data = []
    for idx, row in arguments_df.iterrows():
        premise = row['Premise']
        label_row = labels_df.loc[idx]
        # positive_values = " | ".join([value for value, is_present in label_row.iteritems() if is_present == 1])
        # negative_values = " | ".join([value for value, is_present in label_row.iteritems() if is_present == 0])
        for value, is_present in label_row.iteritems():
            expanded_data.append({
                'text': premise + " </s> " + value,
                'contains': is_present
            })
        # Append the positive and negative values to expanded_data
        # if positive_values:
        #     expanded_data.append({
        #         'text': premise + " </s> " + positive_values,
        #         'contains': 1
        #     })
        # if negative_values:
        #     expanded_data.append({
        #         'text': premise + " </s> " + negative_values,
        #         'contains': 0
        #     })

    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df


if __name__ == '__main__':
    set_seeds(42)


    # Load data
    arguments_train_df = pd.read_csv('data/arguments-training.tsv', delimiter='\t', index_col='Argument ID')
    labels_train_df = pd.read_csv('data/labels-training.tsv', delimiter='\t', index_col='Argument ID')
    arguments_val_df = pd.read_csv('data/arguments-validation.tsv', delimiter='\t', index_col='Argument ID')
    labels_val_df = pd.read_csv('data/labels-validation.tsv', delimiter='\t', index_col='Argument ID')
    arguments_test_df = pd.read_csv('data/arguments-test.tsv', delimiter='\t', index_col='Argument ID')
    labels_test_df = pd.read_csv('data/labels-test.tsv', delimiter='\t', index_col='Argument ID')

    # Convert label matrix to category
    arguments_train_df['label_categories'] = labels_train_df.apply(matrix_to_categories, axis=1)
    arguments_val_df['label_categories'] = labels_val_df.apply(matrix_to_categories, axis=1)
    arguments_test_df['label_categories'] = labels_test_df.apply(matrix_to_categories, axis=1)

    # Combine Conclusion and Label Category
    arguments_train_df['merged_info'] = arguments_train_df['Premise'] + " </s> " + arguments_train_df['Conclusion']
    arguments_val_df['merged_info'] = arguments_val_df['Premise'] + " </s> " + arguments_val_df['Conclusion']
    arguments_test_df['merged_info'] = arguments_test_df['Premise'] + " </s> " + arguments_test_df['Conclusion']

    # Prepare data for training
    texts = arguments_train_df['merged_info'].tolist()
    val_texts = arguments_val_df['merged_info'].tolist()
    test_texts = arguments_test_df['merged_info'].tolist()
    stances = arguments_train_df['Stance'].map({'in favor of': 1, 'against': 0}).tolist()
    val_stances = arguments_val_df['Stance'].map({'in favor of': 1, 'against': 0}).tolist()
    test_stances = arguments_test_df['Stance'].map({'in favor of': 1, 'against': 0}).tolist()

    # expanded_train_df = expand_dataframe(arguments_train_df, labels_train_df)
    # expanded_val_df = expand_dataframe(arguments_val_df, labels_val_df)
    # expanded_test_df = expand_dataframe(arguments_test_df, labels_test_df)
    #
    # # Prepare data for training
    # texts = expanded_train_df['text'].tolist()
    # val_texts = expanded_val_df['text'].tolist()
    # test_texts = expanded_test_df['text'].tolist()
    # labels = expanded_train_df['contains'].tolist()
    # val_labels = expanded_val_df['contains'].tolist()
    # test_labels = expanded_test_df['contains'].tolist()

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    train_dataset = HumanValueDataset(tokenizer, texts, stances)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_dataset = HumanValueDataset(tokenizer, val_texts, val_stances)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_dataset = HumanValueDataset(tokenizer, test_texts, test_stances)
    # train_subset = Subset(train_dataset, indices=range(1))
    # val_subset = Subset(val_dataset, indices=range(1))
    # train_subset_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=2)
    # val_subset_loader = DataLoader(val_subset, batch_size=1, shuffle=True, num_workers=2)
    # test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    wandb.init(project='human value phase 1', entity='skywalkerzhang19')
    wandb_logger = WandbLogger()
    # Initialize and train model
    model = DeBERTaClassifier(num_labels=1, learning_rate=5e-6, warmup_steps=100, epochs=10, extra_training=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1_macro',
        dirpath='./checkpoints_phase1',
        filename='best_model_phase1_premise',
        save_top_k=1,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    wandb.finish()

