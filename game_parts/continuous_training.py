import os
from os.path import join
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from .data import data_utils
from .vision_architectures import initialize_encoder, initialize_decoder, SAVED_WEIGHTS_PATH
from .loss import SupConLoss, get_weighted_loss_function

CONTINUOUS_SAVE_DIR = join(SAVED_WEIGHTS_PATH, 'continuous')
Path(CONTINUOUS_SAVE_DIR).mkdir(parents=True, exist_ok=True)


def get_special_models():
    return {}


def get_existing_models():
    model_names = []
    for ckpt_name in os.listdir(CONTINUOUS_SAVE_DIR):
        if ckpt_name.endswith(".ckpt"):
            model_names.append(ckpt_name)
    return model_names


@dataclass(frozen=True)
class ModelSavePath:
    dataset: str
    dataset_options = set(data_utils.DatasetInformationDict.keys())
    model_type: str
    model_type_options = {'supervised', 'autoencoder', 'contrastive'}
    architecture_type: str
    architecture_type_options = {'conv_deconv', 'conv_pool'}
    ckpt_name: str = None

    def __post_init__(self):
        for attr, options in [(self.dataset, self.dataset_options),
                              (self.model_type, self.model_type_options),
                              (self.architecture_type, self.architecture_type_options)]:
            assert attr is None or attr in options, f"{attr} not in {options}"
        assert self.ckpt_name is None or isinstance(self.ckpt_name, str)
        assert self.ckpt_name != "None"

    @property
    def name_with_suffix(self):
        name = f"{self.model_type}_{self.architecture_type}_{self.dataset}_{self.ckpt_name}.ckpt"
        return name

    @property
    def name(self):
        name = f"{self.model_type}_{self.architecture_type}_{self.dataset}_{self.ckpt_name}"
        return name

    @property
    def full_string(self):
        path = join(CONTINUOUS_SAVE_DIR, self.name_with_suffix)
        return path

    @classmethod
    def from_string(cls, path_str: str):
        # assumes the given str is legal
        path = Path(path_str)
        split = path.with_suffix("").name.split('_')
        if len(split) < 4:
            raise ValueError(f"The given path {path_str} is not a full path with the form <model>_<architecture>_<dataset>_<name>")
        model_type = split[0]
        architecture_type = f'conv_{split[2]}'
        dataset = split[3]
        ckpt_name = '_'.join(split[4:])
        if ckpt_name == 'None':
            ckpt_name = None
        return cls(dataset, model_type, architecture_type, ckpt_name)


class EncoderTrainer(pl.LightningModule):
    def __init__(self, dataset_information: data_utils.DatasetInformationClass, architecture_type):
        super(EncoderTrainer, self).__init__()
        self.encoder = initialize_encoder(dataset_information, architecture_type)
        self.fc = nn.Linear(self.encoder.latent_dim, dataset_information.num_classes)
        self.loss_func = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

    def udf_step(self, batch, batch_idx, state: str):
        X, y = batch
        pred = self(X)
        loss = self.loss_func(pred, y)
        self.log(f"{state}_loss", loss.item())
        return loss

    def training_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='train')

    def validation_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='validation')

    def test_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    @staticmethod
    def get_supervised_encoder(path, entire_network: bool = False):
        model = EncoderTrainer.load_from_checkpoint(path, map_location=torch.device('cpu'))
        if entire_network:
            return model
        encoder = model.encoder
        return encoder


class AutoencoderTrainer(pl.LightningModule):
    def __init__(self, dataset_information: data_utils.DatasetInformationClass, architecture_type, criterion: nn.Module):
        super(AutoencoderTrainer, self).__init__()
        self.encoder = initialize_encoder(dataset_information, architecture_type)
        self.decoder = initialize_decoder(dataset_information, architecture_type)
        self.criterion = criterion
        self.save_hyperparameters(ignore=['criterion'])

    def forward(self, x):
        x = self.encoder(x)  # either a Tensor or a tuple (img_tensor, indices_tensor)
        # x = F.relu(x)
        x = self.decoder(x)
        return x

    def udf_step(self, batch, batch_idx, state: str):
        img, _y = batch
        reconstructed_img = self(img)
        # loss = 1 / img.size(0) * torch.norm(reconstructed_img - img)
        loss, _ = self.criterion(sender_input=img, receiver_output=reconstructed_img,
                                 _message=None, _receiver_input=None, _labels=None)
        # loss = self.criterion(reconstructed_img, img)
        self.log(f"{state}_loss", loss.item())
        return loss

    def training_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='train')

    def validation_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='validation')

    def test_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    @staticmethod
    def get_trained_autoencoder(path, criterion=None, entire_network: bool = True):
        model = AutoencoderTrainer.load_from_checkpoint(path, criterion=criterion, map_location=torch.device('cpu'))
        if entire_network:
            return model, None
        return model.encoder, model.decoder


class ContrastiveEncoderTrainer(pl.LightningModule):
    def __init__(self, dataset_info: data_utils.DatasetInformationClass, architecture_type,
                 proj_dim: int, use_labels: bool = True, device: str = 'cpu'):
        super(ContrastiveEncoderTrainer, self).__init__()
        self.encoder = initialize_encoder(dataset_info, architecture_type)
        self.fc = nn.Linear(self.encoder.latent_dim, proj_dim)
        self.loss_func = SupConLoss(temperature=0.07, contrast_mode='all', device=device)
        self.use_labels = use_labels
        self.save_hyperparameters()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

    def udf_step(self, batch, batch_idx, state: str):
        images, labels = batch
        images = torch.cat([images[0], images[1]], dim=0)
        bsz = labels.size(0)
        features = self(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if self.use_labels:
            loss = self.loss_func(features, labels)
        else:
            loss = self.loss_func(features)
        self.log(f"{state}_loss", loss.item())
        return loss

    def training_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='train')

    def validation_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='validation')

    def test_step(self, batch, batch_idx):
        return self.udf_step(batch, batch_idx, state='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    @staticmethod
    def get_contrastive_encoder(path, entire_network: bool = False):
        """
        if entire_network=False, the encoder module is returned without the linear head that projects to vector during training.
        When using the contrastive encoder as perceptual model in the loss function, we pass entire_network = True.
        """
        model = ContrastiveEncoderTrainer.load_from_checkpoint(path, map_location=torch.device('cpu'))
        if entire_network:
            return model
        encoder = model.encoder
        return encoder


def train_vision_model(model_type: str, architecture_type: str, dataset: str, epochs: int, batch_size: int,
                       save_name: str, verbose: bool,
                       start_checkpoint_name: Optional[str] = None,
                       debug_mode: bool = False, use_wandb_logger: bool = True, use_cuda: bool = True,
                       batch_limit: tuple = (0, 0, 0), **kwargs):
    """
    pre-train the vision module to be later used as initial weights / perceptual model.
    :param model_type: which architecture / model type to train. out of: ['supervised', 'autoencoder', 'contrastive']
    :param architecture_type: type of encoder-decoder arhcitecture, out of: ['conv_deconv', 'conv_pool']
    :param dataset: dataset to train on
    :param epochs:
    :param batch_size:
    :param save_name: version name for the checkpoint file
    :param verbose:
    :param start_checkpoint_name: checkpoint from which to start training. only save name, not full file name.
            note that all parameters will be defined by the checkpoint.
    :param debug_mode: overfits on a single batch
    :param use_wandb_logger: only relevant for autoencoding training with debug_mode = False
    :param use_cuda:
    :param batch_limit: how many batches to take from each subset (see data_utils)
    :param kwargs: if model_type is 'contrastive', kwargs must contain:
        num_augmentations: int, how many random augmentations to perform on each image
        use_labels: bool, whether to use supervised data (e.g., MNIST digits) for contrastive learning
        projection_dim: int
        if model_type is 'autoencoder', kwargs must contain:
        autoencoding_criterion: nn.Module produced by loss.py
    :return:
    """
    save_path = ModelSavePath(dataset, model_type, architecture_type, save_name)
    if os.path.isfile(save_path.full_string):
        print("checkpoint with that name already exists. appending version to the name.")
    accelerator = 'gpu' if torch.cuda.is_available() and use_cuda else 'cpu'
    callbacks = [ModelCheckpoint(dirpath=CONTINUOUS_SAVE_DIR,
                                 filename=save_path.name,
                                 monitor='validation_loss',
                                 verbose=verbose,
                                 mode='min'),
                 # EarlyStopping(monitor='validation_loss', patience=2)
                 ]

    if model_type == 'autoencoder' and not debug_mode and use_wandb_logger:
        logger = WandbLogger(name='continuous_autoencoder')
    else:
        logger = None

    data_module = data_utils.get_pl_datamodule(dataset, batch_size, debug_mode=debug_mode, batch_limit=batch_limit)
    data_module.prepare_data()
    data_module.setup()

    dataset_information = data_utils.DatasetInformationDict[dataset]

    model_cls = {'supervised': EncoderTrainer,
                 'autoencoder': AutoencoderTrainer,
                 'contrastive': ContrastiveEncoderTrainer}[model_type]
    if model_type == 'contrastive':
        model_kwargs = dict(proj_dim=kwargs['projection_dim'],
                            use_labels=kwargs['use_labels'],
                            device='cuda' if accelerator == 'gpu' else accelerator)
    elif model_type == 'autoencoder':
        model_kwargs = dict(criterion=kwargs['autoencoding_criterion'])
    else:
        model_kwargs = {}

    if start_checkpoint_name is not None:
        prev_ckpt_path = ModelSavePath(dataset, model_type, architecture_type, start_checkpoint_name)
        training_network = get_pretrained_encoder(path=prev_ckpt_path,
                                                  criterion=kwargs.get('autoencoding_criterion'),
                                                  entire_network=True)
    else:
        training_network = model_cls(dataset_information, architecture_type, **model_kwargs)

    trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator, callbacks=callbacks, logger=logger)
    trainer.fit(training_network, data_module)
    print("\nTrain:")
    train_dataloader = data_module.train_dataloader()
    trainer.test(training_network, train_dataloader)
    print("\nTest:")
    trainer.test(training_network, data_module)

    if model_type == 'autoencoder':
        input_example, _ = next(iter(data_module.train_dataloader()))
        show_continuous_image_autoencoding(training_network, input_example, save_name=save_name, dataset=dataset)


def show_continuous_image_autoencoding(model: nn.Module, input_example: torch.Tensor, num_plots: int = 1, dataset: str = None, save_name: str = None):
    import matplotlib.pyplot as plt
    with torch.no_grad():
        input_example = input_example.to(model.device)
        reconstructed_img = model(input_example)
    # un-normalize images
    # dataset_info = data_utils.DatasetInformationDict['cifar10']
    dataset_info = model.hparams['dataset_information']
    std = torch.as_tensor(dataset_info.std).view(-1, 1, 1)
    mean = torch.as_tensor(dataset_info.mean).view(-1, 1, 1)
    input_example = input_example.to("cpu")
    reconstructed_img = reconstructed_img.to("cpu")
    input_example.mul_(std).add_(mean)
    reconstructed_img.mul_(std).add_(mean)
    for i in range(min(num_plots, input_example.size(0))):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.suptitle(f"example of continuous reconstruction quality")
        # ax1.imshow(input_example[i].permute(1, 2, 0), cmap='gray')
        ax1.imshow(input_example[i].permute(1, 2, 0))
        ax1.set_title("original")
        # ax2.imshow(reconstructed_img[i].permute(1, 2, 0), cmap='gray')
        ax2.imshow(reconstructed_img[i].permute(1, 2, 0))
        ax2.set_title("reconstructed")
        if save_name is not None:
            assert dataset is not None
            path = Path.cwd() / 'results' / dataset
            path.mkdir(parents=True, exist_ok=True)
            path = path / f'continuous_{save_name}_{i + 1}'
            plt.savefig(path)
        plt.show()


def get_pretrained_encoder(path: str or ModelSavePath, **kwargs):
    """
    path should have the form "{encoder_pretraining_type}_{architecture_type}_{dataset}_{cpt_name}"
    kwargs may contain 'criterion' with a nn.Module generated in loss.py for the AutoencoderTrainer class
    kwargs may contain 'entire_network'=True to return the continuous training object.
    """

    special_models_dict = get_special_models()
    if isinstance(path, str) and path in special_models_dict:
        return special_models_dict[path]()
    if isinstance(path, ModelSavePath) and path.ckpt_name in special_models_dict:
        return special_models_dict[path.ckpt_name]()
    path = path if isinstance(path, ModelSavePath) else ModelSavePath.from_string(path)
    existing_model = get_existing_models()
    if path.name_with_suffix not in existing_model:
        raise ValueError(f"model {path.name_with_suffix} not found in {CONTINUOUS_SAVE_DIR}. "
                         f"To train this continuous model, run: \n "
                         f"pretrain.py --model_type {path.model_type} --architecture_type {path.architecture_type}"
                         f" --dataset {path.dataset} --save_name {path.ckpt_name}\nAnd add any desired arguments.")
    entire_network = kwargs.get('entire_network', False)
    if path.model_type == 'supervised':
        encoder = EncoderTrainer.get_supervised_encoder(path.full_string, entire_network=entire_network)
    elif path.model_type == 'autoencoder':
        encoder, _decoder = AutoencoderTrainer.get_trained_autoencoder(path.full_string,
                                                                       kwargs.get('criterion'),
                                                                       entire_network=entire_network)
    elif path.model_type == 'contrastive':
        encoder = ContrastiveEncoderTrainer.get_contrastive_encoder(path.full_string,
                                                                    entire_network=entire_network)
        print("warning: if you wish to use the contrastive encoder as loss, call the get_contrastive_encoder method "
              "directly. Ignore this warning if you are using the contrastive-pretrained encoder as an agent or "
              "for continuous training.")
    else:
        raise ValueError(f"illegal pretraining type {path.model_type}")
    return encoder


def get_pretrained_decoder(path: str or ModelSavePath):
    """
    similar to `get_pretrained_encoder`.
    """
    path = path if isinstance(path, ModelSavePath) else ModelSavePath.from_string(path)
    assert path.model_type == 'autoencoder', "pretrained decoder is only available via autoencoding"
    _encoder, decoder = AutoencoderTrainer.get_trained_autoencoder(path.full_string)
    return decoder


def make_loss_module(mse_weight: float,
                     perceptual_weight: float, perceptual_ckpt: str,
                     contrastive_weight: float, contrastive_ckpt: str,
                     autoencoder_weight: float, autoencoder_ckpt: str):
    """ loads the checkpoints before calling the method from loss.py """
    if perceptual_weight != 0.0:
        perceptual_model = get_pretrained_encoder(perceptual_ckpt)
    else:
        perceptual_model = None
    if contrastive_weight != 0.0:
        contrastive_model = get_pretrained_encoder(contrastive_ckpt, entire_network=True)
    else:
        contrastive_model = None
    if autoencoder_weight != 0.0:
        path = ModelSavePath.from_string(autoencoder_ckpt).full_string
        autoencoder_model = AutoencoderTrainer.load_from_checkpoint(path, criterion=None,
                                                                    map_location=torch.device('cpu'))
    else:
        autoencoder_model = None
    return get_weighted_loss_function(mse_weight,
                                      perceptual_weight, perceptual_model,
                                      contrastive_weight, contrastive_model,
                                      autoencoder_weight, autoencoder_model)
