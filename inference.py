import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint

from data_module import HumanValueDataset
from module import DeBERTaClassifier
import pandas as pd


if __name__ == '__main__':

    model = DeBERTaClassifier.load_from_checkpoint(r'D:\UTD\Research\semeval2023-baseline\checkpoints_binary\best_model-v2.ckpt')
    model.eval()
    model.to('cuda')
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    arguments_test_df = pd.read_csv('data/arguments-test.tsv', delimiter='\t', index_col='Argument ID')
    labels_test_df = pd.read_csv('data/labels-test.tsv', delimiter='\t', index_col='Argument ID')
    arguments_test_df['combined_text'] = arguments_test_df['Conclusion'] + " </s> " + arguments_test_df['Stance'] + \
                                        " </s> " + arguments_test_df['Premise']
    test_data_df = pd.concat([arguments_test_df, labels_test_df], axis=1)

    test_texts = test_data_df['combined_text'].tolist()
    test_labels = labels_test_df.values
    test_dataset = HumanValueDataset(tokenizer, test_texts, test_labels)
    # test_subset = Subset(test_dataset, indices=range(1))
    # test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
    trainer.test(model, test_loader)

