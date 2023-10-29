import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

tokenizer = AutoTokenizer.from_pretrained("kz-transformers/kaz-roberta-conversational")


class ReviewsData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.combined_text
        self.targets = self.data.target
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=False,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.long),
        }


def collator(items):
    return {
        "ids": torch.permute(pad_sequence([i["ids"] for i in items]), (1, 0)),
        "mask": torch.permute(pad_sequence([i["mask"] for i in items]), (1, 0)),
        "token_type_ids": torch.permute(
            pad_sequence([i["token_type_ids"] for i in items]), (1, 0)
        ),
        "targets": torch.permute(
            pad_sequence([i["targets"].unsqueeze(0) for i in items]), (1, 0)
        ),
    }


def create_data_loaders(train_data=None, test_data=None):
    train_params = {
        "batch_size": TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0,
        "collate_fn": collator,
    }

    test_params = {
        "batch_size": VALID_BATCH_SIZE,
        "shuffle": False,
        "num_workers": 0,
        "collate_fn": collator,
    }

    if train_data is not None:
        training_set = ReviewsData(train_data, tokenizer, MAX_LEN)
        training_loader = DataLoader(training_set, **train_params)
    else:
        training_loader = None

    if test_data is not None:
        testing_set = ReviewsData(test_data, tokenizer, MAX_LEN)
        testing_loader = DataLoader(testing_set, **test_params)
    else:
        testing_loader = None

    return training_loader, testing_loader

