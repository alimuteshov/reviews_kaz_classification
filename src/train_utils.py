import os

import numpy as np
import torch
import mlflow
import git
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

mlflow.set_tracking_uri("http://127.0.0.1:8080")

def get_git_commit_id():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets.float())


def train_one_epoch(model, training_loader, optimizer, device, epoch):
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
        mlflow.log_metric("batch_loss", loss.item(), step=epoch * len(training_loader) + n)

        epoch_loss += loss.item()

    return epoch_loss / (n + 1)


def validation(model, testing_loader, device):
    model.eval()
    epoch_loss = 0
    val_targets = []
    val_preds = []

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            targets = data["targets"].to(device)

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



def train_loop(model, train_loader, val_loader, optimizer, device, cfg):
    best_valid_loss = np.inf
    max_epochs_without_improvement = 2
    epochs_without_improvement = 0

    with mlflow.start_run():
        mlflow.log_param("learning_rate", cfg['training']['learning_rate'])
        mlflow.log_param("epochs", cfg['training']['epochs'])
        mlflow.log_param("git_commit_id", get_git_commit_id())

        for epoch in range(cfg['training']['epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            valid_loss, _, _, auc, ap, f1 = validation(model, val_loader, device)

            mlflow.log_metric("training_loss", train_loss, step=epoch)
            mlflow.log_metric("validation_loss", valid_loss, step=epoch)
            mlflow.log_metric("ROC_AUC", auc, step=epoch)
            mlflow.log_metric("AP", ap, step=epoch)
            mlflow.log_metric("f1_score", f1, step=epoch)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_without_improvement = 0
                if not os.path.exists(cfg['training']['export_dir']):
                    os.makedirs(cfg['training']['export_dir'])
                torch.save(model.state_dict(), f"{cfg['training']['export_dir']}/best_model.pth")
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