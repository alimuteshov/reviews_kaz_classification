from dataclasses import dataclass


@dataclass
class Training:
    export_dir: str
    epochs: int
    learning_rate: float


@dataclass
class Params:
    training: Training