import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from .setup_data import *
import pandas as pd

class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.data_dir = './UrbanSound8K/audio'
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        #self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'
            df = pd.read_csv(metadata_file) 
            df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
            df = df[['relative_path', 'classID']]

            myds = SoundDS(df, self.data_dir)
            num_items = len(myds)
            num_train = round(num_items * 0.8)
            num_val = num_items - num_train
            self.trainset, self.valset = random_split(myds, [num_train, num_val])
            #self.trainset = self.instancialize(train=True)
            #self.valset = self.instancialize(train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = SoundDS(self.df, self.data_dir)
            #self.testset = self.instancialize(train=False)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
    # def load_data_module(self):
    #     name = self.dataset
    #     # Change the `snake_case.py` file name to `CamelCase` class name.
    #     # Please always name your model file name as `snake_case.py` and
    #     # class name corresponding `CamelCase`.
    #     camel_name = ''.join([i.capitalize() for i in name.split('_')])
    #     try:
    #         self.data_module = getattr(importlib.import_module(
    #             '.'+name, package=__package__), camel_name)
    #     except:
    #         raise ValueError(
    #             f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    # def instancialize(self, **other_args):
    #     """ Instancialize a model using the corresponding parameters
    #         from self.hparams dictionary. You can also input any args
    #         to overwrite the corresponding value in self.kwargs.
    #     """
    #     class_args = inspect.getargspec(self.data_module.__init__).args[1:]
    #     inkeys = self.kwargs.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = self.kwargs[arg]
    #     args1.update(other_args)
    #     return self.data_module(**args1)