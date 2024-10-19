import os
from pathlib import Path
from typing import Optional, Dict, Tuple
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms, datasets as torch_datasets
from datasets import load_from_disk

import pytorch_lightning as pl

DatasetInformationClass = namedtuple('DatasetInformationClass', ['name', 'number_of_channels', 'image_size',
                                                                 'num_classes', 'mean', 'std'])
DatasetInformationDict: Dict[str, DatasetInformationClass]
DatasetInformationDict = {
    'mnist': DatasetInformationClass(name='mnist', number_of_channels=1, image_size=28, num_classes=10, mean=(0.1307,), std=(0.3081,)),
    'shapes': DatasetInformationClass(name='shapes', number_of_channels=3, image_size=64, num_classes=-1, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
}
DatasetInstanceDict = dict()    # stores instantiated datasets. used in get_pl_datamodule.


def get_directory_path(folder_name: str):
    this_file_path = os.getcwd()
    delimiter = '\\' if '\\' in this_file_path else '/'
    this_file_path = this_file_path.split(delimiter)
    root_path = this_file_path[:this_file_path.index('SemanticConsistency') + 1]
    directory_path = root_path + ['game_parts', folder_name]
    directory_path = delimiter.join(directory_path)
    return directory_path


DATA_PATH = get_directory_path("data")


def is_cached(dataset: str, encoder_name: str):
    path = Path(DATA_PATH) / 'cached_encoded_images' / dataset / encoder_name
    return path.exists()


class MyDataModule(pl.LightningDataModule):
    DatasetName = None
    VAL_SIZE = 1 / 11  # validation : train = 1 : 10
    """
    Each subclass must implement the following methods:
    prepare_data: download the data
    _full_train_data(): return a dataset that will be split into train and validation
    _test_data(): return the test dataset
    """

    def __init__(self, batch_size: int,
                 transform,
                 debug_mode: bool = False,
                 batch_limit: Tuple[int, int, int] = (0, 0, 0),
                 cached_encoder: Optional[str] = None):
        super(MyDataModule, self).__init__()
        self.batch_size = batch_size
        self.debug_mode = debug_mode
        self.transform = transform
        self.batch_limit = batch_limit
        self.cached_encoder = cached_encoder

        self.trainset = None
        self.valset = None
        self.testset = None

        self.collate_fn = self.cached_loader_data_collate_fn if self.cached_encoder is not None else None
        if debug_mode:
            self.dataloader_kwargs = {'drop_last': True}
        else:
            self.dataloader_kwargs = {'drop_last': True,
                                      'num_workers': torch.cuda.device_count()*2}

    def _full_train_data(self):
        raise NotImplementedError()

    def _test_data(self):
        raise NotImplementedError()

    def load_cached_encodings(self, encoder_name: str):
        path = Path(DATA_PATH) / 'cached_encoded_images' / self.DatasetName / encoder_name
        assert path.exists(), f"no cached encodings for {self.DatasetName} with encoder {encoder_name}"
        ds = load_from_disk(path)
        ds = ds.with_transform(self.cache_transform)
        return ds

    class CachedEncodingDataset(Dataset):
        def __init__(self, full_data, cache_data):
            # full_data contains images + labels
            # cache_data contains encoded images + labels
            super().__init__()
            assert len(full_data) == len(cache_data), "cached dataset and full dataset must have the same length"
            self.full_data = full_data
            self.cache_data = cache_data

        def __len__(self):
            return len(self.full_data)

        def __getitem__(self, index):
            img, label1 = self.full_data[index]
            cache_dict = self.cache_data[index]
            assert label1 == cache_dict['labels'], "label mismatch between the datasets!"
            return {'pixel_values': img, 'encoding': cache_dict['encoded_images'], 'label': label1}

    @staticmethod
    def cache_transform(batch):
        batch['encoded_images'] = [torch.tensor(x) for x in batch['encoded_images']]
        return batch

    @staticmethod
    def cached_loader_data_collate_fn(batch):
        pixel_values = []
        encodings = []
        labels = []
        for example in batch:
            pixel_values.append(example["pixel_values"])
            encodings.append(example["encoding"])
            labels.append(example["label"])

        pixel_values = torch.stack(pixel_values)
        encodings = torch.stack(encodings)
        labels = torch.tensor(labels)
        # order is sender_input, labels, receiver_input, aux_input
        return pixel_values, labels, None, {'encodings': encodings}

    def _dataset_limit_batches(self, dataset, batch_limit: int):
        item_limit = batch_limit * self.batch_size
        remainder = len(dataset) - item_limit
        if batch_limit != 0 and remainder > 0:
            dataset, _ = random_split(dataset, [item_limit, remainder], generator=torch.Generator().manual_seed(42))
        return dataset

    def limit_datasets(self):
        assert None not in [self.trainset, self.valset, self.testset],\
            "call `limit_dataests` only after dataset initialization"
        self.trainset = self._dataset_limit_batches(self.trainset, self.batch_limit[0])
        self.valset = self._dataset_limit_batches(self.valset, self.batch_limit[1])
        self.testset = self._dataset_limit_batches(self.testset, self.batch_limit[2])

    def setup(self, stage: Optional[str] = None):
        if all(dataset is not None for dataset in [self.trainset, self.valset, self.testset]):
            return
        full_dataset = self._full_train_data()
        if self.cached_encoder is not None:
            cached_dataset = self.load_cached_encodings(self.cached_encoder)
            full_dataset = self.CachedEncodingDataset(full_dataset, cached_dataset)
        # full_length = len(full_dataset)
        # validation_size = int(self.VAL_SIZE * full_length)
        # indices = torch.randperm(full_length, generator=torch.Generator().manual_seed(42)).tolist()
        # self.trainset = Subset(cached_dataset, indices[:-validation_size])
        # self.valset = Subset(full_dataset, indices[-validation_size:])
        validation_size = int(self.VAL_SIZE * len(full_dataset))
        train_size = len(full_dataset) - validation_size
        self.trainset, self.valset = random_split(full_dataset,
                                                  [train_size, validation_size],
                                                  generator=torch.Generator().manual_seed(42))
        self.testset = self._test_data()
        self.limit_datasets()

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.trainset,
                          batch_size=batch_size,
                          shuffle=not self.debug_mode,
                          collate_fn=self.collate_fn,
                          **self.dataloader_kwargs)

    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.valset,
                          batch_size=batch_size,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          **self.dataloader_kwargs)

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.testset, batch_size=batch_size, shuffle=False, **self.dataloader_kwargs)


class MNIST_DataModule(MyDataModule):
    DatasetName = 'mnist'

    def prepare_data(self):
        # download the data
        if all(dataset is not None for dataset in [self.trainset, self.valset, self.testset]):
            return
        torch_datasets.MNIST(DATA_PATH, train=True, transform=self.transform, download=True)
        torch_datasets.MNIST(DATA_PATH, train=False, transform=self.transform, download=True)

    def _full_train_data(self):
        return torch_datasets.MNIST(DATA_PATH, train=True, transform=self.transform, download=True)

    def _test_data(self):
        return torch_datasets.MNIST(DATA_PATH, train=False, transform=self.transform, download=True)


class Shapes_DataModule(MyDataModule):
    DatasetName = 'shapes'
    N = 10_000
    dataset = None

    class ShapesDataset(Dataset):
        def __init__(self, dataset, generated):
            self.dataset_ = dataset
            self.generated = generated

        def __len__(self):
            return len(self.generated['world_model'])

        def get_attribute_id(self, attribute_name, attribute_value):
            return self.dataset_.vocabularies['pn'][f"Attribute-{attribute_name}-{attribute_value}"]

        def __getitem__(self, idx):
            img = self.generated['world'][idx]
            img = img.transpose(2, 0, 1)
            attributes = self.generated['world_model'][idx]['entities'][0]
            label = [
                self.get_attribute_id('shape', attributes['shape']['name']),
                self.get_attribute_id('color', attributes['color']['name'])
            ]
            return img, torch.tensor(label)

    @staticmethod
    def print_dict_data(d, prefix=""):
        for key, val in d.items():
            if isinstance(val, list):
                print(f"{prefix}{key} -> type: {type(val)}, len: {len(val)}, inner type: {type(val[0])}")
            elif isinstance(val, np.ndarray):
                print(f"{prefix}{key} -> type: {type(val)}, shape: {val.shape}")
            elif isinstance(val, dict):
                print(f"{prefix}{key} -> type: {type(val)}")
                Shapes_DataModule.print_dict_data(val, prefix + "\t")
            else:
                print(f"{prefix}{key} -> type: {type(val)}")

    def prepare_data(self):
        from shapeworld import Dataset as swDataset     # This package does not work on Windows
        # download the data
        if all(dataset is not None for dataset in [self.trainset, self.valset, self.testset]):
            return
        self.dataset = swDataset.create(dtype='agreement', name='existential', entity_counts=[1])

    def _full_train_data(self):
        # generated['world'] -> the images, shape (N, 64, 64, 3)
        # generated['world_model'] -> the attribute, list of size N, each element is a dict
        # generated['world_model'][i]['entities'][0] -> dict with keys:
        # 'id', 'shape', 'color', 'texture', 'rotation', 'center', bounding_box
        generated = self.dataset.generate(n=self.N, mode='train', include_model=True)
        dataset = self.ShapesDataset(self.dataset, generated)
        print(f"{len(dataset)=}")
        return dataset

    def _test_data(self):
        generated = self.dataset.generate(n=128, mode='train', include_model=True)
        dataset = self.ShapesDataset(self.dataset, generated)
        return dataset


def get_pl_datamodule(dataset: str, batch_size: int, batch_limit: tuple = (0,0,0),
                      debug_mode: bool = False,
                      force_instantiation: bool = False,
                      cached_encoder: Optional[str] = None):
    """
    Returns the instantiated data module.
    :param dataset:
    :param batch_size:
    :param batch_limit: how many batches to take from each subset (train, val, test). 0 means all batches. If the
        number of batches exceedes the size of the datasets, it is treated like 0.
    :param debug_mode: the data_module will contain just one batch, to test code and overfitting
    :param force_instantiation: if False, and the dataset has already been instantiated, returns the existing instance.
    :param cached_encoder: if not None, the training set will be loaded from cache, so the encoded vectors will be given
    to Sender instead of the original images. The cache must be created before calling this function, using
    the cache_encoded_images function in scripts.py
    :return: pl DataModule instance
    """
    if dataset in DatasetInstanceDict and not force_instantiation:
        print("DATASET ALREADY INITIALIZED. if you wish to create a new datamodule, set force_instantiation=True.")
        return DatasetInstanceDict[dataset]
    dataset_info = DatasetInformationDict[dataset]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=dataset_info.mean, std=dataset_info.std)
                                    ])

    base_kwargs = dict(batch_size=batch_size, transform=transform,
                       debug_mode=debug_mode, batch_limit=batch_limit,
                       cached_encoder=cached_encoder)
    if dataset == 'mnist':
        data_module = MNIST_DataModule(**base_kwargs)
    elif dataset == 'shapes':
        data_module = Shapes_DataModule(**base_kwargs)
    else:
        raise ValueError(f"illegal dataset {dataset}")
    DatasetInstanceDict[dataset] = data_module
    return data_module


def get_noise_dataloader(dataset: str, batch_size: int, N_batches: int = 16):
    # Return a dataloader with the same shapes as the given dataset, but generates random gaussian noise.
    # The purpose of this is to test generalization to noise as in
    # "How agents see things: On visual representations in an emergent language game"
    did = DatasetInformationDict[dataset]
    # data_shape = (did.number_of_channels, did.image_size, did.image_size)[did.number_of_channels == 1:]
    data_shape = (did.number_of_channels, did.image_size, did.image_size)
    data = torch.normal(0.0, 1.0, size=(batch_size * N_batches, *data_shape))
    y = torch.zeros(data.size(0))
    dataset = TensorDataset(data, y)
    # dataset = MyDataModule.DatasetWrapper(dataset, game_type=game_type)
    loader = DataLoader(dataset, batch_size, shuffle=False)
    return loader


def my_islice(dataloader: DataLoader, limit: int, total_limit: bool = True):
    """
    If total_limit=True, the number of individual data points is limited. Otherwise, the number of batches is limited.
    note: the limit operates on the batch level, so the maximum number of elements returned is actually
    limit + bs - limit % bs     ( for total_limit=True )
    """
    count = 0
    for batch in dataloader:
        yield batch
        count += batch[0].size(0) if total_limit else 1
        if count >= limit:
            return
