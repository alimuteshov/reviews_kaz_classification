import os
import pandas as pd
import torch
from src.data_utils import create_data_loaders
from src.model_utils import RobertaClass
from src.train_utils import validation

device = "cuda" if torch.cuda.is_available() else "cpu"
EXPORT_DIR = "export_dir"
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

if __name__ == "__main__":
    test_data = pd.read_csv("data/test_data.csv")
    _, testing_loader = create_data_loaders(None, test_data)

    model_path = os.path.join(EXPORT_DIR, "best_model.pth")
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
    df_predictions.to_csv(os.path.join(EXPORT_DIR, OUTPUT_FILE), index=False)
    print(f"Predictions saved to {os.path.join(EXPORT_DIR, OUTPUT_FILE)}")
