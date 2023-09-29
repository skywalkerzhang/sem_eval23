from torch.utils.data import DataLoader
from data_module import HumanValueDataset
from transformers import DebertaV2Tokenizer
import pytorch_lightning as pl
from module import DeBERTaClassifier
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Subset
import random
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seeds(42)
    wandb.init(project='human value', entity='skywalkerzhang19')
    wandb_logger = WandbLogger()

    arguments_df = pd.read_csv('data/arguments-training.tsv', delimiter='\t', index_col='Argument ID')
    labels_df = pd.read_csv('data/labels-training.tsv', delimiter='\t', index_col='Argument ID')
    arguments_val_df = pd.read_csv('data/arguments-validation.tsv', delimiter='\t', index_col='Argument ID')
    labels_val_df = pd.read_csv('data/labels-validation.tsv', delimiter='\t', index_col='Argument ID')
    arguments_test_df = pd.read_csv('data/arguments-test.tsv', delimiter='\t', index_col='Argument ID')
    labels_test_df = pd.read_csv('data/labels-test.tsv', delimiter='\t', index_col='Argument ID')

    arguments_df['combined text'] = arguments_df['Conclusion'] + " </s> " + arguments_df['Stance'] + " </s> " + \
                                    arguments_df['Premise']
    data_df = pd.concat([arguments_df, labels_df], axis=1)
    arguments_val_df['combined_text'] = arguments_val_df['Conclusion'] + " </s> " + arguments_val_df['Stance'] + \
                                        " </s> " + arguments_val_df['Premise']
    val_data_df = pd.concat([arguments_val_df, labels_val_df], axis=1)
    arguments_test_df['combined_text'] = arguments_test_df['Conclusion'] + " </s> " + arguments_test_df['Stance'] + \
                                        " </s> " + arguments_test_df['Premise']
    test_data_df = pd.concat([arguments_test_df, labels_test_df], axis=1)

    texts = data_df['combined text'].tolist()
    labels = labels_df.values

    val_texts = val_data_df['combined_text'].tolist()
    val_labels = labels_val_df.values

    test_texts = test_data_df['combined_text'].tolist()
    test_labels = labels_test_df.values

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    train_dataset = HumanValueDataset(tokenizer, texts, labels)
    train_subset = Subset(train_dataset, indices=range(1))

    # train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_dataset = HumanValueDataset(tokenizer, val_texts, val_labels)
    # val_subset = Subset(val_dataset, indices=range(1))
    # val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_dataset = HumanValueDataset(tokenizer, test_texts, test_labels)
    # test_subset = Subset(test_dataset, indices=range(1))
    # test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    # 注意：num_labels要设置为你标签的数量
    model = DeBERTaClassifier(num_labels=labels.shape[1], learning_rate=3e-5, warmup_steps=100, epochs=20)
    pl.seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控验证损失
        dirpath='./checkpoints',  # 保存文件的目录
        filename='best-checkpoint',  # 保存文件的名称
        save_top_k=1,  # 仅保存最好的模型
        mode='min'  # 当监控的验证损失减少时保存模型
    )

    trainer = pl.Trainer(max_epochs=20, accelerator='gpu', logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    wandb.finish()
