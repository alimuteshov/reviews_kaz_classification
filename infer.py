import os
import pandas as pd
import torch
import hydra
from hydra.core.config_store import ConfigStore
from config import Params
from src.data_utils import create_data_loaders
from src.model_utils import RobertaClass
from src.train_utils import validation

device = "cuda" if torch.cuda.is_available() else "cpu"
cs = ConfigStore.instance()
cs.store(name="params", node=Params)
OUTPUT_FILE = "predictions.csv"

def load_model(model_path, model_class, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def infer(model, loader):
    _, predictions, labels, auc, ap, f1 = validation(model, loader, device)
    return predictions, labels, auc, ap, f1

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    test_data = pd.read_csv("data/test_data.csv")
    _, testing_loader = create_data_loaders(None, test_data)

    model_path = os.path.join(cfg['training']['export_dir'], "best_model.pth")
    model = load_model(model_path, RobertaClass, device)

    predictions, labels, auc, ap, f1 = infer(model, testing_loader)

    print({
        "ROC_AUC": auc,
        "AP": ap,
        "f1": f1,
    })

    df_predictions = pd.DataFrame({
        "Predictions": predictions,
        "True_Labels": labels,
    })
    df_predictions.to_csv(os.path.join(cfg['training']['export_dir'], OUTPUT_FILE), index=False)
    print(f"Predictions saved to {os.path.join(cfg['training']['export_dir'], OUTPUT_FILE)}")

if __name__ == "__main__":
    main()

