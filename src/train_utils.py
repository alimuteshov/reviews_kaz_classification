import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

EXPORT_DIR = "export_dir"
EPOCHS = 2


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets.float())


def train_one_epoch(model, training_loader, optimizer, device):
    model.train()
    epoch_loss = 0
    for n, data in enumerate(training_loader, 0):
        optimizer.zero_grad()

        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / (n + 1)


def validation(model, testing_loader):
    model.eval()
    epoch_loss = 0
    val_targets = []
    val_preds = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["ids"]
            mask = data["mask"]
            token_type_ids = data["token_type_ids"]
            targets = data["targets"]
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)

            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            epoch_loss += loss.item()

    val_targets = np.concatenate(val_targets)
    val_preds = np.concatenate(val_preds)
    auc = roc_auc_score(val_targets, val_preds)
    ap = average_precision_score(val_targets, val_preds)
    f1 = f1_score(val_targets, val_preds > 0.5)

    return (
        epoch_loss / len(testing_loader),
        val_targets,
        val_preds,
        auc,
        ap,
        f1,
    )


def train_loop(model, train_loader, val_loader, optimizer, device):
    best_valid_loss = np.inf
    max_epochs_without_improvement = 2
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss, _, _, auc, ap, f1 = validation(model, val_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            if not os.path.exists(EXPORT_DIR):
                os.makedirs(EXPORT_DIR)
            torch.save(model.state_dict(), f"{EXPORT_DIR}/best_model.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement:
            print(
                f"No improvement in {max_epochs_without_improvement} epochs. Early stopping..."
            )
            break

        print(
            {
                "training_ep_loss": train_loss,
                "valid_ep_loss": valid_loss,
                "ROC_AUC": auc,
                "AP": ap,
                "f1": f1,
            }
        )

    return None
