from torch.utils.data import Dataset
import torch


class HumanValueDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, values=None, extra_training=False, binary_training=False):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.extra_training = extra_training
        self.binary_training = binary_training
        self.values = values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        value = self.values[idx]
        label = torch.tensor(self.labels[idx])
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        if not self.binary_training:
            return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), label.float()
        else:
            return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), label.float(), value
