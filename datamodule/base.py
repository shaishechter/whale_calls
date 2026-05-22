import lightning as L


class DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
