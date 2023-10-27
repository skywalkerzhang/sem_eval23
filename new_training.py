import torch
from torch.utils.data import DataLoader
from data_module import HumanValueDataset
from transformers import DebertaV2Tokenizer
import pytorch_lightning as pl
from module import DeBERTaClassifier
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Subset
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import set_seeds
import argparse


def expand_dataframe_binary_with_value_attached(arguments_df, labels_df):
    expanded_data = []
    for idx, row in arguments_df.iterrows():
        premise = row['Premise']
        label_row = labels_df.loc[idx]

        for value, is_present in label_row.iteritems():
            text = premise + " </s> " + value
            expanded_data.append({
                'text': text,
                'contains': is_present,
                'value': value
            })

    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df


# 没改完，需要加如何计算Marco f1的代码，调整模型输出的个数和其他的之类的，最后还要把预测转换为原来的形式
# 把value加在预测里那计算F1的时候怎么办，怎么找到你要的类别
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script for Phase 2")
    parser.add_argument("--pretrained_model_path",
                        default=None,
                        type=str, help="Path to the pretrained model. If not provided, will train from scratch.")
    args = parser.parse_args()
    set_seeds(42)

    arguments_train_df = pd.read_csv('data/arguments-training.tsv', delimiter='\t', index_col='Argument ID')
    labels_train_df = pd.read_csv('data/labels-training.tsv', delimiter='\t', index_col='Argument ID')
    arguments_val_df = pd.read_csv('data/arguments-validation.tsv', delimiter='\t', index_col='Argument ID')
    labels_val_df = pd.read_csv('data/labels-validation.tsv', delimiter='\t', index_col='Argument ID')
    arguments_test_df = pd.read_csv('data/arguments-test.tsv', delimiter='\t', index_col='Argument ID')
    labels_test_df = pd.read_csv('data/labels-test.tsv', delimiter='\t', index_col='Argument ID')

    expanded_train_df = expand_dataframe_binary_with_value_attached(arguments_train_df, labels_train_df)
    expanded_val_df = expand_dataframe_binary_with_value_attached(arguments_val_df, labels_val_df)
    expanded_test_df = expand_dataframe_binary_with_value_attached(arguments_test_df, labels_test_df)

    # Prepare data for training
    texts = expanded_train_df['text'].tolist()
    val_texts = expanded_val_df['text'].tolist()
    test_texts = expanded_test_df['text'].tolist()

    labels = expanded_train_df['contains'].tolist()
    val_labels = expanded_val_df['contains'].tolist()
    test_labels = expanded_test_df['contains'].tolist()

    values = expanded_train_df['value'].tolist()
    val_values = expanded_val_df['value'].tolist()
    test_values = expanded_test_df['value'].tolist()

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    train_dataset = HumanValueDataset(tokenizer, texts, labels, values, binary_training=True)
    # train_subset = Subset(train_dataset, indices=range(16))

    # train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_dataset = HumanValueDataset(tokenizer, val_texts, val_labels, values, binary_training=True)
    # val_subset = Subset(val_dataset, indices=range(16))
    # val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_dataset = HumanValueDataset(tokenizer, test_texts, test_labels, values, binary_training=True)
    # test_subset = Subset(test_dataset, indices=range(16))
    # test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=2)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    if args.pretrained_model_path:
        model = DeBERTaClassifier(binary_training=True, num_labels=1).load_from_checkpoint(
            r'D:\UTD\Research\semeval2023-baseline\checkpoints_binary\best_model.ckpt')
    else:
        # 注意：num_labels要设置为你标签的数量
        model = DeBERTaClassifier(num_labels=1, learning_rate=5e-6, warmup_steps=100, epochs=10, binary_training=True)
    wandb.init(project='human value binary', entity='skywalkerzhang19')
    wandb_logger = WandbLogger()

    checkpoint_callback = ModelCheckpoint(
        monitor='macro_f1',  # 监控验证损失
        dirpath='./checkpoints_binary',  # 保存文件的目录
        filename='best_model',  # 保存文件的名称
        save_top_k=1,  # 仅保存最好的模型
        mode='max'  # 当监控的验证损失减少时保存模型
    )

    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', logger=wandb_logger, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)
    torch.cuda.empty_cache()
    trainer.test(model, test_loader)
    wandb.finish()
