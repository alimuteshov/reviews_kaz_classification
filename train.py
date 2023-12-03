import pandas as pd
import torch
import hydra
from hydra.core.config_store import ConfigStore

from config import Params

from src.data_utils import create_data_loaders
from src.model_utils import RobertaClass
from src.train_utils import train_loop

device = "cuda" if torch.cuda.is_available() else "cpu"

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")
    training_loader, testing_loader = create_data_loaders(train_data, test_data)

    model = RobertaClass()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['learning_rate'])

    train_loop(model, training_loader, testing_loader, optimizer, device, cfg)


if __name__ == "__main__":
    main()
