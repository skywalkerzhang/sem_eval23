import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        # Use sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # This computes the probabilities for correct classification
        true_probs = probs * targets + (1 - probs) * (1 - targets)
        log_probs = torch.log(true_probs + 1e-8)  # adding a small epsilon for numerical stability

        # Compute the loss
        loss = -self.alpha * (1 - true_probs) ** self.gamma * log_probs

        if self.reduce:
            return loss.mean()  # use mean() instead of sum() for consistency
        else:
            return loss


class DeBERTaClassifier(pl.LightningModule):
    def __init__(self, num_labels, learning_rate, warmup_steps, epochs,
                 model_path="microsoft/deberta-v3-base"):
        super(DeBERTaClassifier, self).__init__()

        self.model = DebertaV2ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.epochs = epochs
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = FocalLoss(alpha=1, gamma=2)

        for param in self.model.parameters():
            assert param.requires_grad, "Some model parameters do not require gradients!"

    def forward(self, input_ids, attention_mask, labels=None, return_logits=False):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            if return_logits:
                return loss, logits
            return loss
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, logits = self(input_ids, attention_mask, labels, return_logits=True)
        predicted_probs = torch.sigmoid(logits)
        predicted_labels = (predicted_probs > 0.5).long()

        return {"val_loss": loss, "predictions": predicted_labels, "labels": labels}

    def on_validation_batch_end(self, batch, batch_idx, dataloader_idx=0):
        outputs = batch
        if not hasattr(self, "val_outputs"):
            self.val_outputs = []
        self.val_outputs.append(outputs)

    def on_validation_epoch_end(self):
        all_predictions = torch.cat([x["predictions"] for x in self.val_outputs], dim=0).cpu().numpy()
        all_labels = torch.cat([x["labels"] for x in self.val_outputs], dim=0).cpu().numpy()

        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        self.log("val_f1_macro", f1_macro)

        # Clear the outputs list for the next epoch
        self.val_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lambda_func = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_func)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            input_ids, attention_mask, labels = batch
        else:
            input_ids, attention_mask = batch

        logits = self(input_ids, attention_mask, return_logits=True)
        predicted_probs = torch.sigmoid(logits)
        predicted_labels = (predicted_probs > 0.5).long()
        return {"predicted_labels": predicted_labels}

    def on_test_batch_end(self, output_results, batch, batch_idx, dataloader_idx=0):
        # Initialize the list if it doesn't exist yet
        if not hasattr(self, "predicted_labels_list"):
            self.predicted_labels_list = []

        # Append results
        self.predicted_labels_list.append(output_results["predicted_labels"])

    def on_test_epoch_end(self):
        if not hasattr(self, "predicted_labels_list"):
            self.predicted_labels_list = []

        all_predicted_labels = torch.cat(self.predicted_labels_list, dim=0).cpu().numpy()

        # Convert numpy array to pandas DataFrame
        df = pd.DataFrame(all_predicted_labels)

        # Save to CSV
        df.to_csv("predicted_labels_focal_loss.csv", index=False)

        # Optionally, clear the list for future test runs
        del self.predicted_labels_list
