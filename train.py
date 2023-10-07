import pandas as pd
import torch

from src.data_utils import create_data_loaders
from src.model_utils import RobertaClass
from src.train_utils import train_loop

LEARNING_RATE = 3e-5
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")
    training_loader, testing_loader = create_data_loaders(train_data, test_data)

    model = RobertaClass()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    train_loop(model, training_loader, testing_loader, optimizer, device)
