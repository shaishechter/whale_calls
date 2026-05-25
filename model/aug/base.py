import torch.nn as nn


class BaseAugmentation(nn.Module):
    def __init__(self, on_train: bool, on_val: bool, on_predict: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_train = on_train
        self.on_val = on_val
        self.on_predict = on_predict
