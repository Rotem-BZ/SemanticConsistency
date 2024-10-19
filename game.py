import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable
from pathlib import Path
from os.path import join
from pprint import pprint
import json
from functools import cached_property
from dataclasses import dataclass

import torch
import torch.nn as nn

from egg import core
from egg.core import LoggingStrategy
from egg.core.batch import Batch
from egg.core.baselines import Baseline, MeanBaseline

from game_parts import get_sender, get_receiver, vision_architectures, ConvSender
from game_parts.continuous_training import make_loss_module, ModelSavePath, get_pretrained_encoder
from game_parts.data.data_utils import get_pl_datamodule, my_islice, get_noise_dataloader, is_cached
from game_parts.loss import DiscriminationLoss
import callbacks as cb

DISCRETE_SAVE_DIR = join(vision_architectures.SAVED_WEIGHTS_PATH, 'discrete')
Path(DISCRETE_SAVE_DIR).mkdir(exist_ok=True, parents=True)


def get_existing_models():
    # existing_model_paths = Path(DISCRETE_SAVE_DIR).glob("*_final.tar")
    # model_names = [path.name.split('_final.tar')[0] for path in existing_model_paths]
    model_names = []
    for game_type_path in Path(DISCRETE_SAVE_DIR).iterdir():
        if not game_type_path.is_dir():
            continue
        model_names += [path.name for path in game_type_path.iterdir() if path.is_dir()]
    return model_names


class GumbelSoftmaxGame(nn.Module):
    """
    based on: https://github.com/facebookresearch/EGG/blob/main/egg/core/gs_wrappers.py
    """

    def __init__(
            self,
            sender,
            receiver,
            loss,
            nn_modules_dict: dict,
            any_modules_dict: dict,
            make_aux_input_method: Callable,
            length_cost=0.0,
            train_logging_strategy: Optional[LoggingStrategy] = None,
            test_logging_strategy: Optional[LoggingStrategy] = None,
            include_eos_token=True
    ):
        super(GumbelSoftmaxGame, self).__init__()
        assert length_cost == 0.0 or include_eos_token, "can't penalize length without EOS token"
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.nn_modules_dict = nn.ModuleDict(nn_modules_dict)
        self.any_modules_dict = any_modules_dict
        self.make_aux_input = make_aux_input_method
        self.length_cost = length_cost
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )
        self.include_eos_token = include_eos_token

    def forward_with_eos(self, sender_input, labels, receiver_input, aux_input, message, receiver_output):
        loss = 0
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0

        aux_info = {}
        z = 0.0
        for step in range(receiver_output.size(1)):  # if include_eos_token=False, this for will have a single loop
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],  # Wrong? should be message[:, :step+1, ...].
                # Doesn't matter because the loss doesn't depend on the message.
                receiver_input,
                receiver_output[:, step, ...],
                labels,
                aux_input,
            )
            eos_mask = message[:, step, 0]  # always eos == 0

            add_mask = eos_mask * not_eosed_before
            z += add_mask
            loss += step_loss * add_mask + self.length_cost * (1.0 + step) * add_mask
            expected_length += add_mask.detach() * (1.0 + step)

            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += (
                step_loss * not_eosed_before
                + self.length_cost * (step + 1.0) * not_eosed_before
        )
        expected_length += (step + 1) * not_eosed_before

        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)

        aux_info["length"] = expected_length.detach()

        return loss, aux_info

    def forward_no_eos(self, sender_input, labels, receiver_input, aux_input, message, receiver_output):
        loss, aux = self.loss(
            sender_input,
            message,
            receiver_input,
            receiver_output[:, 0, ...],
            labels,
            aux_input,
        )
        length = message.size(1)
        aux["length"] = torch.ones(sender_input.size(0)) * length
        return loss, aux

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        ########################################################
        if aux_input is None:
            aux_input = dict()
        additional_aux_input = (
            self.make_aux_input(self.nn_modules_dict, self.any_modules_dict, sender_input, labels, receiver_input, aux_input)
        )
        aux_input.update(additional_aux_input)
        ########################################################
        message = self.sender(sender_input, aux_input)  # message shape: [B, (max_len + 1 or max_len), V]
        receiver_output = self.receiver(message, None, aux_input)  # shape: [B, (max_len + 1 or 1), data_dim]

        if self.include_eos_token:
            loss, aux_info = self.forward_with_eos(sender_input, labels, receiver_input, aux_input, message,
                                                   receiver_output)
        else:
            loss, aux_info = self.forward_no_eos(sender_input, labels, receiver_input, aux_input, message,
                                                 receiver_output)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=aux_info["length"],
            aux=aux_info,
        )

        return loss.mean(), interaction


class ReinforceGame(nn.Module):
    """
    based on: https://github.com/facebookresearch/EGG/blob/main/egg/core/reinforce_wrappers.py
    """
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        nn_modules_dict: dict,
        any_modules_dict: dict,
        make_aux_input_method: Callable,
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        length_cost: float = 0.0,
        baseline_type: Baseline = MeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        super(ReinforceGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.nn_modules_dict = nn.ModuleDict(nn_modules_dict)
        self.any_modules_dict = any_modules_dict
        self.make_aux_input = make_aux_input_method

        self.mechanics = core.CommunicationRnnReinforce(
            sender_entropy_coeff,
            receiver_entropy_coeff,
            length_cost,
            baseline_type,
            train_logging_strategy,
            test_logging_strategy,
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        ########################################################
        if aux_input is None:
            aux_input = dict()
        additional_aux_input = (
            self.make_aux_input(self.nn_modules_dict, self.any_modules_dict, sender_input, labels, receiver_input,
                                aux_input)
        )
        aux_input.update(additional_aux_input)
        ########################################################
        return self.mechanics(
            self.sender,
            self.receiver,
            self.loss,
            sender_input,
            labels,
            receiver_input,
            aux_input,
        )


class Game(ABC):
    """
    Superclass for a game object, that juggles the different game parts together with a config.
    Each subclass (game type) must implement:
    1. A Config inner class: dataclass that inherits from Game.Config
    2. A variable `game_type`, to be used in get_pl_datamodule
    3. A method _make_loss_module()
    4. Optional: A static method _make_aux_input(), which returns the aux_input object given to the agents.
    5. Optional: A method _make_modules_dict(), which return two dicts to be used by _make_aux_input():
        1. dict of nn modules
        2. dict of anything else (won't be properly registered as nn.Module)
    6. Optional: a method _assert_valid_config()
    7. Optional: a method _additional_callbacks()
    """
    game_type: str = None

    @dataclass(frozen=True, slots=True)
    class Config:
        """
        Base config class. Each game subclasses from this and adds specific hyperparams.
        """
        dataset: str = 'mnist'
        batch_size: int = 64
        batch_limit: Tuple[int, int, int] = (0, 0, 0)
        length_cost: float = 0.0
        verbose: bool = False
        debug_mode: bool = False
        use_cuda: bool = True
        add_validation_callbacks: bool = False
        log_every: int = 10

        discretization_method: str = 'gs'  # 'gs' or 'reinforce' or 'quantize' or 'continuous' or 'english'
        include_eos_token: bool = False
        architecture_type: str = 'conv_deconv'  # or 'conv_pool'
        vocab_size: int = 10
        legal_vocab_subsets: Optional[tuple[int]] = None    # if None, all vocab is legal at all timesteps
        random_sender: bool = False
        num_unique_random_messages: int = None
        max_len: int = 4
        temperature: float = 1.0
        use_temperature_scheduler: bool = False
        temperature_decay: float = 0.9
        temperature_minimum: float = 0.1
        temperature_update_freq: int = 2
        trainable_temperature: bool = False

        # sender kwargs
        encoder_pretraining_type: Optional[str] = 'autoencoder'
        callback_perceptual_model_pretraining_type: Optional[str] = 'autoencoder'
        encoder_cpt_name: str = '1'
        callback_perceptual_model_cpt_name: Optional[str] = None
        callback_perceptual_model_architecture_type: str = 'conv_deconv'
        use_game_encoder_as_perceptual_model: bool = False
        frozen_encoder: bool = True
        use_cached_encoder: bool = True
        sender_lr: float = 1.931e-3
        sender_gen_model: str = 'gru'  # used if discretization method is generative (gs or reinforce).
        sender_RNN_hidden_dim: int = 500
        sender_RNN_emb_size: int = 400

        # receiver kwargs
        receiver_lr: float = 3.82e-4
        receiver_embed_model: str = 'gru'  # 'rnn' or 'gru' or 'lstm'
        receiver_RNN_hidden_dim: int = 30
        receiver_RNN_emb_size: int = 40
        receiver_RNN_num_layers: int = 1

        def get_config_dict(self):
            return {k: eval(f"self.{k}", {"self": self}) for k in self.__slots__}

        def save_to_disk(self, path):
            config_dict = self.get_config_dict()
            with open(Path(path) / "config.json", 'w') as file:
                json.dump(config_dict, file, indent=2)

        @classmethod
        def load_from_disk(cls, path):
            with open(Path(path) / "config.json", 'r') as file:
                config_dict = json.load(file)
            return cls(**config_dict)

    def __init__(self, config: Config):
        assert self.game_type is not None, "every child class of Game must have a static variable `game_type`"
        self.game_config = config
        self._assert_valid_config()
        if config.verbose:
            print("game config:")
            pprint(config)
            print("\n\n")

        # init sender, receiver, loss function and dataset
        self.sender = get_sender(self.game_config, self.game_type)
        self.receiver = get_receiver(self.game_config, self.game_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
        self.loss_func = self._make_loss_module()

        opts = core.init(params=['--random_seed=7'])
        self.game = self.make_game_module()
        if self.game_config.add_validation_callbacks:
            print("setting validation logging to maximal")
            if hasattr(self.game, 'mechanics'):
                self.game.mechanics.test_logging_strategy = LoggingStrategy()
            else:
                self.game.test_logging_strategy = LoggingStrategy()
        self.trainer = None

    def _assert_valid_config(self):
        vocab_subsets = self.game_config.legal_vocab_subsets
        if vocab_subsets is not None:
            assert sum(vocab_subsets) == self.game_config.vocab_size, "sum of vocab subsets must equal vocab size"
            assert self.game_config.discretization_method
        if self.game_config.random_sender:
            assert self.game_config.discretization_method in ['gs', 'reinforce'], "random sender outputs symbol sequences"

    @classmethod
    def get_game_dir_path(cls, save_name):
        return join(DISCRETE_SAVE_DIR, cls.game_type, save_name)

    @abstractmethod
    def _make_loss_module(self) -> nn.Module:
        pass

    @staticmethod
    def _make_aux_input(nn_modules_dict: nn.ModuleDict, any_modules_dict, sender_input, labels, receiver_input, aux_input) -> dict:
        # override this method in child classes if you want additional inputs
        return {}

    def _make_modules_dict(self) -> Tuple[dict[str, nn.Module], dict[str, object]]:
        # override this method in child classes if you want to use modules for _make_inputs()
        return {}, {}

    @cached_property
    def data_module(self):
        config = self.game_config
        sender_encoder = ModelSavePath(config.dataset, config.encoder_pretraining_type,
                                       config.architecture_type, config.encoder_cpt_name).name
        if self.game_config.use_cached_encoder and is_cached(self.game_config.dataset, sender_encoder):
            # print("using cached encoder")
            cached_encoder = sender_encoder
        else:
            cached_encoder = None
        dm = get_pl_datamodule(self.game_config.dataset,
                               self.game_config.batch_size,
                               self.game_config.batch_limit,
                               debug_mode=self.game_config.debug_mode,
                               cached_encoder=cached_encoder)
        dm.prepare_data()
        dm.setup()
        return dm

    @property
    def is_gs(self):
        return self.game_config.discretization_method == 'gs'

    @property
    def num_possible_messages(self):
        if self.game_config.discretization_method in ['quantize', 'continuous']:
            return float('inf')
        legal_subsets = self.game_config.legal_vocab_subsets
        legal_subsets = legal_subsets if legal_subsets is not None else [self.game_config.vocab_size]
        channel_capacity = 0
        for i, subset_size in enumerate(legal_subsets):
            if self.game_config.discretization_method == 'gs' and self.game_config.include_eos_token:
                # \sum_{i=1}^{k}{v^i} = \freq{v^{k+1}-v}{v-1}
                v = subset_size - 1 if i == 0 else subset_size  # ignore EOS token which is id=0
                k = self.game_config.max_len
                channel_capacity += (v ** (k + 1) - v) / (v - 1)
            else:
                channel_capacity += subset_size ** self.game_config.max_len
        return int(channel_capacity)

    def make_game_module(self):
        logging_strategy = LoggingStrategy.minimal()
        nn_modules, any_modules = self._make_modules_dict()
        base_kwargs = dict(sender=self.sender,
                           receiver=self.receiver,
                           loss=self.loss_func,
                           nn_modules_dict=nn_modules,
                           any_modules_dict=any_modules,
                           make_aux_input_method=self._make_aux_input,
                           train_logging_strategy=logging_strategy,
                           test_logging_strategy=logging_strategy
                           )
        discretization_method = self.game_config.discretization_method
        if discretization_method == 'gs':
            game = GumbelSoftmaxGame(**base_kwargs,
                                     length_cost=self.game_config.length_cost,
                                     include_eos_token=self.game_config.include_eos_token)
        elif discretization_method == 'reinforce':
            game = ReinforceGame(**base_kwargs,
                                 sender_entropy_coeff=0.0,
                                 receiver_entropy_coeff=0.0,
                                 length_cost=self.game_config.length_cost)
        elif discretization_method == 'english':
            raise NotImplementedError
        else:
            raise ValueError(f"illegal discretization_method {discretization_method}")
        return game

    @staticmethod
    def logging_strategy(train: str = None, test: str = None):
        """A decorator for logging strategy asignment"""
        assert train in ['minimal', 'maximal', None]
        assert test in ['minimal', 'maximal', None]
        logging_strategy_dict = {'minimal': LoggingStrategy.minimal, 'maximal': LoggingStrategy.maximal}

        def inner(f):
            def wrapper(*args, **kwargs):
                self = args[0]
                game = self.game.mechanics if hasattr(self.game, 'mechanics') else self.game
                # game = args[0].game
                original_train_strategy = game.train_logging_strategy
                original_test_strategy = game.test_logging_strategy
                if train is not None:
                    game.train_logging_strategy = logging_strategy_dict[train]()
                if test is not None:
                    game.test_logging_strategy = logging_strategy_dict[test]()
                out = f(*args, **kwargs)
                game.train_logging_strategy = original_train_strategy
                game.test_logging_strategy = original_test_strategy
                return out

            return wrapper

        return inner

    def make_trainer_module(self, callbacks):
        game = self.game
        optimizer = torch.optim.Adam([
            {'params': game.sender.parameters(), 'lr': self.game_config.sender_lr},
            {'params': game.receiver.parameters(), 'lr': self.game_config.receiver_lr}
        ])
        self.trainer = core.Trainer(self.game, optimizer, self.data_module.train_dataloader(),
                                    validation_data=self.data_module.val_dataloader(),
                                    device=self.device,
                                    callbacks=callbacks)

    def _additional_callbacks(self, kwargs) -> list[cb.MyCallback]:
        callbacks = [
            cb.UniqueMessageCount(**kwargs, num_possible_messages=self.num_possible_messages, is_gs=self.is_gs),
            cb.EmpiricalMessageVarianceCallback(**kwargs, is_gs=self.is_gs),
            cb.EmpiricalMessageVarianceBaselineCallback(**kwargs, is_gs=self.is_gs),
            # cb.PCAByMessagePlotCallback(**kwargs, is_gs=self.is_gs),
            # cb.PCAByLabelPlotCallback(**kwargs),
            cb.ZipfBarPlotCallback(**kwargs, is_gs=self.is_gs),
            cb.MessageLengthCallback(**kwargs),
            cb.LossCallback(**kwargs)
        ]
        if self.game_config.dataset == 'mnist':
            callbacks += [
                cb.MessagePurityCallback(**kwargs, is_gs=self.is_gs),
                cb.MessagePurityBaselineCallback(**kwargs, is_gs=self.is_gs),
            ]
        if self.game_config.dataset == 'shapes':
            callbacks += [
                cb.PosDisCallback(**kwargs, is_gs=self.is_gs),
                cb.BosDisCallback(**kwargs, is_gs=self.is_gs, vocab_size=self.game_config.vocab_size),
                cb.SpeakerPosDisCallback(**kwargs, is_gs=self.is_gs),
                cb.MessagePurityShapesCallback(**kwargs, is_gs=self.is_gs, attribute='shape'),
                cb.MessagePurityShapesBaselineCallback(**kwargs, is_gs=self.is_gs, attribute='shape'),
                cb.MessagePurityShapesCallback(**kwargs, is_gs=self.is_gs, attribute='color'),
                cb.MessagePurityShapesBaselineCallback(**kwargs, is_gs=self.is_gs, attribute='color'),
                cb.MaxAttributeMessagePurityCallback(**kwargs, is_gs=self.is_gs),
                cb.MaxAttributeMessagePurityBaselineCallback(**kwargs, is_gs=self.is_gs)
            ]
        if self.game_config.discretization_method in ['gs', 'reinforce']:
            # Topsim
            callbacks += [
                cb.TopographicSimilarityCallback(**kwargs,
                                                 is_gs=self.is_gs,
                                                 vocab_size=self.game_config.vocab_size,
                                                 message_distance='edit'),
                cb.ReducedTopographicSimilarityCallback(**kwargs,
                                                        is_gs=self.is_gs,
                                                        vocab_size=self.game_config.vocab_size,
                                                        message_distance='edit')
            ]
            # Cluster variance
            if self.game_config.legal_vocab_subsets is not None:
                callbacks += [
                    cb.EmpiricalClusterVarianceCallback(**kwargs,
                                                        is_gs=self.is_gs,
                                                        vocab_subsets=self.game_config.legal_vocab_subsets),
                    cb.EmpiricalClusterVarianceBaselineCallback(**kwargs,
                                                                is_gs=self.is_gs,
                                                                vocab_subsets=self.game_config.legal_vocab_subsets),
                    cb.PCAByClusterPlotCallback(**kwargs,
                                                is_gs=self.is_gs,
                                                vocab_subsets=self.game_config.legal_vocab_subsets)
                ]
        return callbacks

    @logging_strategy(train='minimal')
    def train(self, epochs: int = 10, use_wandb_logger: bool = False,
              save_checkpoint: bool = False, save_name: str = None):
        if save_name is None:
            save_name = f"{self.game_config.dataset}_model"
        this_dir = self.get_game_dir_path(save_name)
        callbacks = [cb.ConsoleLogger(print_train_loss=True)]
        if self.game_config.verbose:
            print("using progress bar logger because verbose=True. warning: may use lots of memory")
            pb_logger = core.ProgressBarLogger(epochs, len(self.data_module.trainset), len(self.data_module.testset),
                                               use_info_table=False)
            callbacks.append(pb_logger)
        if self.is_gs and self.game_config.use_temperature_scheduler:
            temp_logger = core.TemperatureUpdater(self.sender.gen_model,
                                                  self.game_config.temperature_decay,
                                                  self.game_config.temperature_minimum,
                                                  self.game_config.temperature_update_freq)
            callbacks.append(temp_logger)
        if use_wandb_logger:
            wandb_logger = core.callbacks.WandbLogger(opts=dict(),
                                                      project='EmergentComm',
                                                      config=self.game_config.get_config_dict())
            callbacks.append(wandb_logger)
        else:
            wandb_logger = None
        if save_checkpoint:
            checkpoint_callback = core.CheckpointSaver(this_dir,
                                                       prefix='model',
                                                       checkpoint_freq=5,
                                                       max_checkpoints=3)
            callbacks.append(checkpoint_callback)
        if self.game_config.add_validation_callbacks:
            kwargs = dict(wandb_logger=wandb_logger,
                          log_every=self.game_config.log_every,
                          dump_interaction_function=self.dump_interactions,
                          batch_size=self.game_config.batch_size,
                          dataset=self.game_config.dataset)
            additional_callbacks = self._additional_callbacks(kwargs)
            # get pretrained encoder for perceptual metrics
            if self.game_config.use_game_encoder_as_perceptual_model:
                assert self.game_config.frozen_encoder, "perceptual model must be frozen"
                assert not self.game_config.random_sender, "random sender doesn't have a pretrained encoder"
                agent = self.sender if isinstance(self.sender, ConvSender) else self.sender.agent
                perceptual_model = agent.encoder
            elif self.game_config.callback_perceptual_model_cpt_name is not None:
                model_path = ModelSavePath(self.game_config.dataset,
                                           self.game_config.callback_perceptual_model_pretraining_type,
                                           self.game_config.callback_perceptual_model_architecture_type,
                                           self.game_config.callback_perceptual_model_cpt_name)
                perceptual_model = get_pretrained_encoder(model_path)
            else:
                perceptual_model = None
            callbacks.append(cb.CombinedCallbacks(additional_callbacks, input_transform=perceptual_model))
        self.make_trainer_module(callbacks)
        self.trainer.train(n_epochs=epochs)

        # save config for checkpoint loading
        if save_checkpoint:
            self.game_config.save_to_disk(this_dir)

    @logging_strategy(train='minimal', test='maximal')
    def eval(self, subset: str):
        loader = self.get_subset_dataloader(subset)
        loader = my_islice(loader, 500, total_limit=True)  # 500 data points are used in evaluation
        if self.trainer is None:
            self.make_trainer_module([])
        mean_loss, full_interaction = self.trainer.eval(loader)
        for callback in self._additional_callbacks(dict(wandb_logger=None,
                                                        log_every=1,
                                                        dump_interaction_function=self.dump_interactions,
                                                        batch_size=self.game_config.batch_size,
                                                        dataset=self.game_config.dataset)):
            callback.on_validation_end(mean_loss, full_interaction, epoch=0)
        return mean_loss, full_interaction

    def get_subset_dataloader(self, subset, batch_size=None):
        if subset == 'train':
            return self.data_module.train_dataloader(batch_size=batch_size)
        if subset == 'val':
            return self.data_module.val_dataloader(batch_size=batch_size)
        if subset == 'test':
            return self.data_module.test_dataloader(batch_size=batch_size)
        if subset == 'noise':
            return get_noise_dataloader(self.game_config.dataset,
                                        self.game_config.batch_size,
                                        N_batches=500 // self.game_config.batch_size + 1)  # at least 500 samples
        raise ValueError(f"illegal subset {subset}. Choose from: [train, val, test, noise]")

    @logging_strategy(train='maximal', test='maximal')
    def dump_interactions(self, loader, training_mode=False):
        """
        An adjustment of core.dump_interactions:
        https://github.com/facebookresearch/EGG/blob/18d72d86cf9706e7ad82f94719b56accd288e59a/egg/core/interaction.py#L281
        Added funcionality: choosable training_mode (train or eval), TBD
        """
        self.game.to(self.device)
        train_state = self.game.training  # persist so we restore it back
        self.game.train(training_mode)
        full_interaction = None
        mean_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                n_batches += 1
                batch = Batch(*batch).to(self.device)
                loss, interaction = self.game(*batch)
                mean_loss += loss.item()
                interaction = interaction.to("cpu")

                # if self.game_config.include_eos_token:
                #     message_indices = (interaction.message_length - 1).long()
                # else:
                #     message_indices = 0
                # interaction.receiver_output = interaction.receiver_output[torch.arange(interaction.size), message_indices, :]

                if self.is_gs:
                    interaction.message = interaction.message.argmax(
                        dim=-1
                    )  # actual symbols instead of one-hot encoded
                assert interaction.message_length is not None
                if self.game_config.discretization_method in ['gs', 'reinforce']:
                    for i in range(interaction.size):
                        length = interaction.message_length[i].long().item()
                        interaction.message[i, length:] = 0  # 0 is always EOS

                full_interaction = (
                    full_interaction + interaction
                    if full_interaction is not None
                    else interaction
                )

        mean_loss /= n_batches
        self.game.train(mode=train_state)

        return mean_loss, full_interaction

    @classmethod
    def load_from_checkpoint(cls, save_name: str):
        this_dir = cls.get_game_dir_path(save_name)
        weights_path = join(this_dir, "model_final.tar")
        with open(weights_path, 'rb') as file:
            # checkpoint = torch.load(file)
            checkpoint = torch.load(file, map_location=torch.device('cpu'))
            assert isinstance(checkpoint, core.callbacks.Checkpoint)
            # checkpoint: core.callbacks.Checkpoint
        config = cls.Config.load_from_disk(this_dir)
        game_object = cls(config)
        game_object.game.load_state_dict(checkpoint.model_state_dict)
        return game_object


class ImageReconstructionGame(Game):
    game_type = 'Reconstruction'

    @dataclass(frozen=True, slots=True)
    class Config(Game.Config):
        # receiver kwargs
        receiver_type: str = 'MLP'
        decoder_pretraining_type: Optional[str] = 'autoencoder'
        decoder_cpt_name: str = '1'
        frozen_decoder: bool = True
        MLP_num_hidden_layers: int = 0
        MLP_hidden_dim: int = 64
        MLP_activation: str = 'relu'

        # loss kwargs
        loss_mse_weight: float = 1.0
        loss_perceptual_weight: float = 0.0
        loss_perceptual_checkpoint: str = "{encoder_pretraining_type}_{architecture_type}_{dataset}_{cpt_name}"
        loss_contrastive_weight: float = 0.0
        loss_contrastive_checkpoint: str = "{encoder_pretraining_type}_{architecture_type}_{dataset}_{cpt_name}"
        loss_autoencoder_weight: float = 0.0
        loss_autoencoder_checkpoint: str = "{encoder_pretraining_type}_{architecture_type}_{dataset}_{cpt_name}"

        # logging kwargs
        callback_num_distractors: int = 5

    def __init__(self, config: Config):  # this function is only here for type hinting
        super().__init__(config)
        self.game_config = config

    def _assert_valid_config(self):
        super()._assert_valid_config()
        config = self.game_config
        assert config.callback_num_distractors < config.batch_size, "num_distractors must be less than batch_size"

    def _make_loss_module(self):
        config = self.game_config
        loss_module = make_loss_module(config.loss_mse_weight,
                                       config.loss_perceptual_weight,
                                       config.loss_perceptual_checkpoint,
                                       config.loss_contrastive_weight,
                                       config.loss_contrastive_checkpoint,
                                       config.loss_autoencoder_weight,
                                       config.loss_autoencoder_checkpoint)
        return loss_module

    def plot_autoencoding_example(self, subset='test', save_name: str = None, training_mode=False):
        loader = self.get_subset_dataloader(subset, batch_size=min(3, self.game_config.batch_size))
        loader = my_islice(loader, limit=1, total_limit=False)
        _, interaction = self.dump_interactions(loader, training_mode=training_mode)

        figs = cb.plot_autoencoding_from_interaction(interaction, self.is_gs, self.game_config,
                                                     training_mode=training_mode)
        for z, fig in enumerate(figs):
            if save_name is not None:
                this_dir = self.get_game_dir_path(save_name)
                fig.savefig(join(this_dir, f'{save_name}_{subset}_{z + 1}'))
            fig.show()

    def print_disc_accuracy_for_reco(self, subset: str, num_distractors=5, training_mode=False):
        """
        performs the discrimination game by choosing the image with the lowest loss compared to Receiver's output.
        """
        loader = self.get_subset_dataloader(subset, batch_size=num_distractors)
        loader = my_islice(loader, limit=200)
        _, interaction = self.dump_interactions(loader, training_mode=training_mode)

        overall_accuracy = cb.disc_accuracy_for_reco_from_interaction(interaction, num_distractors, self.game_config,
                                                                      self.loss_func)
        print(
            f"Discrimination by reconstruction accuracy on {subset} ({'training' if training_mode else 'inference'} mode): {overall_accuracy:.3f}")
        return overall_accuracy

    def _additional_callbacks(self, kwargs) -> list[cb.MyCallback]:
        base_callbacks = super()._additional_callbacks(kwargs)
        return base_callbacks + [
            cb.AutoencodingPlotCallback(is_gs=self.is_gs, game_config=self.game_config, num_plots=1, **kwargs),
            cb.DiscAccuracyForRecoCallback(num_distractors=self.game_config.callback_num_distractors,
                                           game_config=self.game_config,
                                           loss_func=self.loss_func,
                                           **kwargs)
        ]


class ImageDiscriminationGame(Game):
    """
    The number of distractors is batch_size - 1.
    """

    game_type = 'Discrimination'

    @dataclass(frozen=True, slots=True)
    class Config(Game.Config):
        # receiver kwargs
        receiver_encoder_pretraining_type: Optional[str] = 'autoencoder'
        receiver_encoder_cpt_name: str = '1'
        receiver_frozen_encoder: bool = True
        MLP_num_hidden_layers: int = 0
        MLP_hidden_dim: int = 64
        MLP_activation: str = 'relu'
        num_distractors: int = 5
        # different_label_distractors: bool = False  # avoid using distractors from the same class as the input image

        discrimination_strategy: str = 'vanilla'
        # vanilla - random distractors
        # supervised - different-label distractors
        # classification - one candidate per label, target is not the original image but has its label
        contrastive_loss: bool = False
        # if true, uses contrastive loss (CE in both directions) instead of NLL loss

    def __init__(self, config: Config):  # this function is only here for type hinting
        super().__init__(config)
        self.game_config = config

    def _make_loss_module(self):
        loss_module = DiscriminationLoss(discrimination_strategy=self.game_config.discrimination_strategy)
        return loss_module

    @staticmethod
    def _make_aux_input(nn_modules_dict: nn.ModuleDict, any_modules_dict, sender_input, labels, receiver_input, aux_input) -> dict:
        return {'images': sender_input, 'labels': labels}

    def _assert_valid_config(self):
        super()._assert_valid_config()
        config = self.game_config
        assert config.num_distractors < config.batch_size, "num_distractors must be smaller than batch_size"
        if config.contrastive_loss:
            raise NotImplementedError("contrastive loss is not implemented yet")

    def print_average_accuracy(self, subset: str, num_distractors=None, training_mode=False):
        """
        calculates accuracy over 200 samples. if num_distractors is None, it is set to self.game_config.batch_size
        """
        num_distractors = num_distractors if num_distractors is not None else self.game_config.num_distractors
        bsz = 2 * num_distractors + 1 if self.game_config.distractors_label_strategy == "only_negative" else num_distractors + 1

        loader = self.get_subset_dataloader(subset, batch_size=bsz)
        loader = my_islice(loader, limit=200)
        _, interaction = self.dump_interactions(loader, training_mode=training_mode)
        accuracy = cb.get_accuracy_from_interaction(interaction, num_distractors=num_distractors,
                                                    is_gs=self.is_gs, game_config=self.game_config)
        print(f"Discrimination {subset} accuracy ({num_distractors} distractors): {accuracy:.3f}")
        return accuracy

    def plot_discrimination(self, subset: str, num_plots, num_distractors=None, training_mode=False, save_name=None):
        num_distractors = num_distractors if num_distractors is not None else self.game_config.num_distractors
        if num_distractors > self.game_config.num_distractors:
            raise ValueError(f"more than {self.game_config.num_distractors} distractors are not possible yet")
        bsz = 2 * num_distractors + 1 if self.game_config.distractors_label_strategy == 'only_negative' else num_distractors + 1
        bsz = max(bsz, self.game_config.batch_size)
        loader = self.get_subset_dataloader(subset, batch_size=bsz)
        assert len(loader) >= num_plots, f"not enough available data for {num_plots} plots with batch size {bsz}."
        loader = my_islice(loader, limit=num_plots, total_limit=False)
        _, interaction = self.dump_interactions(loader, training_mode=training_mode)
        figs = cb.get_plot_from_interaction(interaction, is_gs=self.is_gs, game_config=self.game_config,
                                            num_plots=num_plots, num_distractors=num_distractors, training_mode=training_mode)
        for z, fig in enumerate(figs):
            if save_name is not None:
                this_dir = self.get_game_dir_path(save_name)
                fig.savefig(join(this_dir, f'{save_name}_{subset}_{z + 1}'))
            fig.show()

    def _additional_callbacks(self, kwargs) -> list[cb.MyCallback]:
        base_callbacks = super()._additional_callbacks(kwargs)
        return base_callbacks + [
            cb.DiscAccuracyCallback(num_distractors=self.game_config.num_distractors,
                                    is_gs=self.is_gs,
                                    game_config=self.game_config,
                                    **kwargs),
            cb.DiscriminationPlotCallback(is_gs=self.is_gs, game_config=self.game_config, num_plots=1 ,**kwargs)
        ]
