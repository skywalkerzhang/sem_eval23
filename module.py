import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from dice_loss import AdjustedDiceLoss


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


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Sigmoid activation
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        # pos in both predictions and ground-truth
        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice_coeff


class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        true_probs = probs * targets + (1 - probs) * (1 - targets)
        # Calculate the modulating factor for each sample and class based on the prediction probability
        modulating_factor = (1. - true_probs) ** self.gamma

        # Compute Dice coefficient
        intersection = (probs * targets).sum(dim=0)
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum(dim=0) + targets.sum(dim=0) + self.smooth)

        # Compute weighted Dice loss
        weighted_dice_loss = self.alpha * modulating_factor * (1 - dice_coeff)
        weighted_dice_loss = weighted_dice_loss.mean()

        return weighted_dice_loss


class CombinedFocalDiceLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, smooth=1e-6, lambda_coeff=0.5):
        super(CombinedFocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.lambda_coeff = lambda_coeff

    def forward(self, logits, targets, lambda_coeff=0.5):
        self.lambda_coeff = lambda_coeff
        probs = torch.sigmoid(logits)

        # Dice Loss
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        # Focal Loss
        focal_loss_pos = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs + self.smooth)
        focal_loss_neg = -self.alpha * probs ** self.gamma * (1 - targets) * torch.log(1 - probs + self.smooth)
        focal_loss = (focal_loss_pos + focal_loss_neg).mean()

        # Combine
        combined_loss = self.lambda_coeff * focal_loss + (1 - self.lambda_coeff) * dice_loss

        return combined_loss


class AdjustiveDiceLoss(torch.nn.Module):
    def __init__(self, alpha=2, eps=1e-7):
        super(AdjustiveDiceLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        pi1 = probs
        yi1 = targets

        # Compute the numerator of the DSC
        num = 2.0 * ((1.0 - pi1) ** self.alpha) * pi1 * yi1

        # Compute the denominator of the DSC
        den = ((1.0 - pi1) ** self.alpha) * pi1 + yi1

        dsc = num + 1 / den + 1

        return 1.0 - dsc.mean()  # Return the loss


class DeBERTaClassifier(pl.LightningModule):
    def __init__(self, num_labels=1, learning_rate=5e-6, warmup_steps=100, epochs=10,
                 model_path="microsoft/deberta-v3-xsmall", extra_training=False, binary_training=False):
        super(DeBERTaClassifier, self).__init__()
        self.model = DebertaV2ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.epochs = epochs
        self.extra_training = extra_training
        self.loss_fn = CombinedFocalDiceLoss()  # Default loss function
        self.binary_training = binary_training

        # If extra_training is True, set an additional loss function or make other adjustments
        if self.extra_training or self.binary_training:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        for param in self.model.parameters():
            assert param.requires_grad, "Some model parameters do not require gradients!"

    def forward(self, input_ids, attention_mask, labels=None, return_logits=False):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, _ = batch
        logits = self(input_ids=input_ids, attention_mask=attention_mask)

        # Calculate the lambda_coeff based on the current epoch
        lambda_coeff = self.current_epoch / self.trainer.max_epochs

        # Now use this lambda_coeff in your loss calculation
        if self.extra_training or self.binary_training:
            logits = logits.view(-1)
        loss = self.loss_fn(logits, labels)

        self.log('train_loss', loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, values = batch
        logits = self(input_ids, attention_mask, labels, return_logits=True)
        predicted_probs = torch.sigmoid(logits)
        predicted_labels = (predicted_probs > 0.5).long()

        if self.extra_training or self.binary_training:
            logits = logits.view(-1)
        loss = self.loss_fn(logits, labels)

        return {"val_loss": loss, "predictions": predicted_labels.detach().cpu(), "labels": labels.detach().cpu(),
                "values": values}

    def on_validation_batch_end(self, batch, batch_idx, dataloader_idx=0):
        outputs = batch
        if not hasattr(self, "val_outputs"):
            self.val_outputs = []
        self.val_outputs.append(outputs)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.val_outputs]).mean()
        all_predictions = torch.cat([x["predictions"] for x in self.val_outputs], dim=0).cpu().numpy()
        all_labels = torch.cat([x["labels"] for x in self.val_outputs], dim=0).cpu().numpy()
        if self.binary_training:
            all_values = sum([list(x["values"]) for x in self.val_outputs], [])

            unique_values = set(all_values)
            total_f1 = 0.0

            for value in unique_values:
                indexes = [i for i, v in enumerate(all_values) if v == value]
                value_predictions = all_predictions[indexes]
                value_labels = all_labels[indexes]

                f1_for_value = f1_score(value_labels, value_predictions, average='binary', zero_division=0)
                total_f1 += f1_for_value

            macro_f1 = total_f1 / len(unique_values)
        else:
            macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

        # Logging
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('macro_f1', macro_f1, prog_bar=True)
        torch.cuda.empty_cache()

        return {'val_loss': avg_loss, 'macro_f1': macro_f1}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        lambda_func = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_func)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:
            input_ids, attention_mask, labels, _ = batch
        else:
            input_ids, attention_mask, labels = batch

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
        df.to_csv("predicted_labels_tmp.csv", index=False, header=False)

        # Optionally, clear the list for future test runs
        del self.predicted_labels_list
