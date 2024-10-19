import time
import json
from pprint import pprint
from numbers import Number
from collections import defaultdict
from typing import Any
import copy

import matplotlib.pyplot as plt
from matplotlib.figure import SubFigure
import torch
import torch.nn as nn
import wandb

from egg import core
from egg.core.language_analysis import TopographicSimilarity, entropy_dict
from egg.core.batch import Batch

from game_parts.data.data_utils import DatasetInformationDict


########################################################################
def noise_like_interaction(interaction: core.Interaction, num_labels: int):
    """
    replaces the sender_input and labels of the given interaction with noise.
    """
    noise = torch.rand_like(interaction.sender_input)
    interaction_parts = copy.deepcopy(vars(interaction))
    interaction_parts['sender_input'] = noise
    if num_labels == -1:
        # sample each attribute uniformly from the set of options
        for attribute_idx in range(interaction.labels.size(1)):
            values = interaction.labels[:, attribute_idx].unique()
            rand_indices = torch.randint(0, len(values), (interaction.size,))
            interaction_parts['labels'][:, attribute_idx] = values[rand_indices]
    elif num_labels is not None:
        print(f"{type(interaction.labels)=}")
        interaction_parts['labels'] = torch.randint(num_labels, size=(interaction.size,))
    return core.Interaction(**interaction_parts)


def randomize_message_content(interaction: core.Interaction):
    """
    keeps the same size of equivalence classes, but assigns inputs to message randomly.
    """
    # message = interaction.message
    # shuffled_interaction = shuffle_interaction(interaction)
    # shuffled_interaction.message = message
    shuffle_perm = torch.randperm(interaction.size)
    interaction_parts = copy.deepcopy(vars(interaction))
    interaction_parts['message'] = interaction_parts['message'][shuffle_perm]
    return core.Interaction(**interaction_parts)


def apply_indices_to_interaction(interaction: core.Interaction, indices: torch.Tensor):
    """
    returns a new interaction (deep copied) with only the indices specified in the given tensor.
    """
    size = interaction.size
    interaction_parts = copy.deepcopy(vars(interaction))
    for key, value in interaction_parts.items():
        if value is None:
            continue
        if isinstance(value, dict):
            new_dict = dict()
            for inner_key, t in value.items():
                assert isinstance(t, torch.Tensor), f"{inner_key} in {key} isn't a tensor"
                assert t.size(0) == size, f"{inner_key} in {key} has size {t.size(0)} instead of {size}"
                new_dict[inner_key] = t[indices]
            interaction_parts[key] = new_dict
        elif isinstance(value, torch.Tensor):
            assert value.size(0) == size, f"{key} has size {value.size(0)} instead of {size}"
            interaction_parts[key] = value[indices]
        else:
            raise ValueError(f"illegal type for {key}: {type(value)}")

    return core.Interaction(**interaction_parts)


def edit_interaction(interaction: core.Interaction, bsz=None, shuffle=False, limit=None):
    if not shuffle and limit is None:
        return interaction
    size = interaction.size
    new_indices = torch.arange(size)
    if shuffle:
        # we shuffle the order of batches within the interaction, keeping the batches themselves unchanged.
        assert bsz is not None
        assert size % bsz == 0
        new_indices = new_indices.view(size // bsz, bsz)
        shuffle_permutation = torch.randperm(size // bsz)
        new_indices = new_indices[shuffle_permutation].flatten()
    if limit is not None:
        assert bsz is None or limit % bsz == 0
        new_indices = new_indices[:limit]

    new_interaction = apply_indices_to_interaction(interaction, new_indices)
    return new_interaction


def shuffle_interaction(interaction: core.Interaction, bsz):
    return edit_interaction(interaction, bsz=bsz, shuffle=True)


def limit_interaction(interaction: core.Interaction, limit: int):
    return edit_interaction(interaction, limit=limit)


def check_valid_interaction(interaction: core.Interaction):
    indices = torch.arange(interaction.size)
    try:
        _ = apply_indices_to_interaction(interaction, indices)
    except AssertionError as e:
        print(e)
        return False
    return True


def seperate_interaction_to_batches(interaction: core.Interaction, batch_size: int):
    assert interaction.size >= batch_size, f"interaction size ({interaction.size}) is smaller than batch size ({batch_size})"
    remainder = interaction.size % batch_size
    if remainder != 0:
        interaction = limit_interaction(interaction, interaction.size - remainder)
    size = interaction.size
    num_batches = size // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        batch_interaction = apply_indices_to_interaction(interaction, torch.arange(batch_start, batch_end))
        yield batch_interaction


def inputs_of_interaction(interaction: core.Interaction) -> Batch:
    return Batch(sender_input=interaction.sender_input,
                 labels=interaction.labels,
                 receiver_input=interaction.receiver_input,
                 aux_input=interaction.aux_input
                 )


def loader_from_interaction(interaction: core.Interaction, batch_size: int):
    generator = seperate_interaction_to_batches(interaction, batch_size)
    for batch in generator:
        yield inputs_of_interaction(batch)


def transform_interaction_inputs(interaction: core.Interaction, transform: nn.Module, batch_size: int):
    device = next(transform.parameters()).device
    transformed_interaction = copy.deepcopy(vars(interaction))
    transformed_input = None
    loader = loader_from_interaction(interaction, batch_size)
    for batch in loader:
        with torch.no_grad():
            transformed_input_ = transform(batch.sender_input.to(device)).to('cpu')
        if transformed_input is None:
            transformed_input = transformed_input_
        else:
            transformed_input = torch.cat((transformed_input, transformed_input_), dim=0)
    transformed_interaction['sender_input'] = transformed_input
    return core.Interaction(**transformed_interaction)


def replace_messages_with_clusters(interaction: core.Interaction, is_gs, vocab_subsets):
    message = interaction.message
    if is_gs and message.ndim == 3:
        message = message.argmax(dim=-1)
    modified_messages = torch.clone(message)
    cumsum = torch.cumsum(torch.tensor(vocab_subsets), 0)
    total = 0
    for upper_bound in cumsum:
        boolean_mask = torch.logical_and(message[:, 0] < upper_bound, message[:, 0] >= total)
        if not boolean_mask.any().item():
            print("WARNING: empty cluster!")
            continue
        example_index = torch.nonzero(boolean_mask)[0].item()
        modified_messages[boolean_mask] = message[example_index]
        total = upper_bound
    modified_interaction = copy.deepcopy(interaction)
    modified_interaction.message = modified_messages
    return modified_interaction
########################################################################


class MyCallback(core.Callback):
    logs_plots: bool = None         # True if the callback returns plots, False if it returns numbers
    only_inference: bool = None     # True if the callback shouldn't be called with train-mode interactions
    transform_inputs: bool = None   # True if the callback should be called with inputs that have been embedded
    # Each callback must:
    # 1. assign boolean values to logs_plots, only_inference and transform_inputs
    # 2. implement method _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
    # that outputs (str, list[plt.Figure]) if logs_plots is True, or (str, Number) otherwise.

    def __init__(self, wandb_logger: core.callbacks.WandbLogger,
                 log_every,
                 dump_interaction_function,
                 batch_size,
                 dataset):
        self.wandb_logger = wandb_logger
        self.log_every = log_every
        self.dump_interactions = dump_interaction_function
        self.batch_size = batch_size
        self.dataset = dataset

        assert self.only_inference or not self.transform_inputs,\
            "transform_inputs is only relevant for inference callbacks"

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        # game_mode is either "train" or "inference"
        # data_mode is either "validation" or "noise"
        raise NotImplementedError

    def dump_train_mode_interaction(self, logs, item_limit: int = None):
        # we need to make sure that the size of the interaction is a multiple of self.batch_size, and
        # not larger than item_limit.
        interaction_size = logs.size if item_limit is None else min(item_limit, logs.size)
        assert interaction_size >= self.batch_size, f"interaction size ({interaction_size}) is smaller than batch size ({self.batch_size})"
        interaction_size = interaction_size - interaction_size % self.batch_size
        logs = limit_interaction(logs, interaction_size)
        loader = loader_from_interaction(logs, self.batch_size)
        return self.dump_interactions(loader, training_mode=True)

    def dump_noise_interaction(self, logs, item_limit: int = None):
        # we need to make sure that the size of the interaction is a multiple of self.batch_size, and
        # not larger than item_limit.
        interaction_size = logs.size if item_limit is None else min(item_limit, logs.size)
        assert interaction_size >= self.batch_size, f"interaction size ({interaction_size}) is smaller than batch size ({self.batch_size})"
        interaction_size = interaction_size - interaction_size % self.batch_size
        logs = limit_interaction(logs, interaction_size)
        logs = noise_like_interaction(logs, num_labels=DatasetInformationDict[self.dataset].num_classes)
        loader = loader_from_interaction(logs, self.batch_size)
        return self.dump_interactions(loader, training_mode=False)

    def dump_noise_and_train_interactions(self, loss, logs: core.Interaction, item_limit=None, only_inference=None):
        # The given interaction is inference-mode over validation data. This function dumps interactions over the
        # same data in train-mode (if not only_inference), and over noise data in inference-mode.
        if only_inference is None:
            only_inference = self.only_inference
        noise_loss, noise_logs = self.dump_noise_interaction(logs, item_limit=item_limit)
        interactions = [dict(loss=loss, logs=logs, game_mode="inference", data_mode="validation"),
                        dict(loss=noise_loss, logs=noise_logs, game_mode="inference", data_mode="noise")]
        if not only_inference:
            train_loss, train_mode_logs = self.dump_train_mode_interaction(logs, item_limit=item_limit)
            interactions.append(dict(loss=train_loss, logs=train_mode_logs, game_mode="train", data_mode="validation"))
        return interactions

    def get_values_to_log(self, interactions: list[dict]) -> dict[str, Any]:
        values_to_log = dict()
        for interaction in interactions:
            if interaction['game_mode'] == "inference" or not self.only_inference:
                name, value = self._calculate_value_to_log(**interaction)
                values_to_log[name] = value
        return values_to_log

    def log_plots(self, values_to_log, epoch):
        values_to_log: dict[str, list[plt.Figure]]
        num_plots = None
        for key, value in values_to_log.items():
            assert isinstance(value, list), f"expected a list of figures for {key}, got {type(value)}"
            assert all(isinstance(fig, plt.Figure) for fig in value), f"expected figures in {key}"
            if num_plots is None:
                num_plots = len(value)
            else:
                assert len(value) == num_plots, f"expected {num_plots} plots for {key}, got {len(value)}"
        for i in range(num_plots):
            dict_to_log = {key: wandb.Image(value[i]) for key, value in values_to_log.items()}
            if self.wandb_logger is not None:
                dict_to_log['epoch'] = epoch
                self.wandb_logger.log_to_wandb(dict_to_log, commit=True)
                for fig_list in values_to_log.values():
                    plt.close(fig_list[i])
            else:
                for fig in values_to_log.values():
                    fig = fig[i]
                    title = fig._suptitle._text + "\n" if fig._suptitle is not None else ""
                    fig.suptitle(title + f"{epoch=}")
                    fig.show()

    def log_numbers(self, values_to_log, epoch, print_class=True):
        values_to_log: dict[str, Number]
        for key, value in values_to_log.items():
            assert isinstance(value, Number), f"expected a Number for {key}, got {type(value)}"
        if self.wandb_logger is not None:
            values_to_log['epoch'] = epoch
            self.wandb_logger.log_to_wandb(values_to_log, commit=True)
        else:
            if print_class:
                print(f"At epoch {epoch}, {type(self).__name__} returned:")
            pprint(values_to_log, sort_dicts=False)

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        # This function isn't used when the callback is combined with others in a CombinedCallbacks object.
        if (epoch - 1) % self.log_every != 0:
            return
        item_limit = self.batch_size if self.logs_plots else None
        logs = shuffle_interaction(logs, bsz=self.batch_size)    # avoids plotting the same inputs every epoch
        interactions = self.dump_noise_and_train_interactions(loss, logs, item_limit=item_limit)
        values_to_log = self.get_values_to_log(interactions)
        if self.logs_plots:
            self.log_plots(values_to_log, epoch)
        else:
            self.log_numbers(values_to_log, epoch)


class CombinedCallbacks(core.Callback):
    """
    this callback combines multiple callbacks. Using this, we can run the dump_train_mode_interaction and
    dump_noise_interaction functions only once per epoch, instead of once per callback.
    """
    def __init__(self, callbacks: list[MyCallback], input_transform: nn.Module = None):
        self.callbacks = callbacks
        self.input_transform = input_transform
        self.log_every = callbacks[0].log_every
        assert all(callback.log_every == self.log_every for callback in callbacks)
        self.batch_size = callbacks[0].batch_size
        assert all(callback.batch_size == self.batch_size for callback in callbacks)
        print("CombinedCallbacks initialized with the following callbacks:")
        pprint([type(callback).__name__ for callback in callbacks])

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        if (epoch - 1) % self.log_every != 0:
            return
        logs = shuffle_interaction(logs, bsz=self.batch_size)  # avoids plotting the same inputs every epoch
        if self.input_transform is None:
            transformed_logs = logs
        else:
            # images are embedded by some pretrained model, so that callbacks are calculated in the latent space
            transformed_logs = transform_interaction_inputs(logs, self.input_transform, self.batch_size)
        all_interactions = self.callbacks[0].dump_noise_and_train_interactions(loss, logs, only_inference=False)

        transformed_interactions = []
        # we replace inputs with the transformed inputs for inference validation interactions. Plotting callbacks
        # still use the original data.
        for interaction in all_interactions:
            logs = interaction['logs']
            if interaction['game_mode'] == "inference" and interaction['data_mode'] == "validation":
                logs = transformed_logs
            transformed_interactions.append(
                dict(loss=interaction['loss'],
                     logs=logs,
                     game_mode=interaction['game_mode'],
                     data_mode=interaction['data_mode']))

        times_dict = dict()
        numbers_to_log = dict()
        for callback in self.callbacks:
            interactions = transformed_interactions if callback.transform_inputs else all_interactions
            t0 = time.perf_counter()
            values_to_log = callback.get_values_to_log(interactions)
            t1 = time.perf_counter()
            times_dict[type(callback).__name__] = t1 - t0
            if callback.logs_plots:
                callback.log_plots(values_to_log, epoch)
            else:
                print(f"{type(callback).__name__} time: {t1 - t0}")
                numbers_to_log.update(values_to_log)
        if numbers_to_log:
            print(f"At epoch {epoch}:")
            self.callbacks[0].log_numbers(numbers_to_log, epoch, print_class=False)
        print("callback times:")
        pprint(times_dict)


def unique_message_count_from_interaction(interaction: core.Interaction, is_gs: bool):
    message = interaction.message
    if is_gs and message.ndim == 3:
        message = message.argmax(dim=-1)
    unique_messages = set(tuple(m.tolist()) for m in message)
    return len(unique_messages)


class UniqueMessageCount(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = False

    def __init__(self, num_possible_messages, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.num_possible_msgs = num_possible_messages
        if self.num_possible_msgs == float("inf"):
            self.num_possible_msgs = "inf"
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        num_unique = unique_message_count_from_interaction(logs, self.is_gs)
        key = f'unique messages over {logs.size} {data_mode} inputs (channel capacity: {self.num_possible_msgs})'
        return key, num_unique


def message_purity_from_interaction(interaction: core.Interaction, is_gs: bool):
    # the purity of a message is the fraction of inputs with the argmax label within the equivalence class. This
    # function computes the (weighted) average purity over all equivalence classes.
    # In other words, this function returns the precetage of inputs which their label is the majority in their
    # equivalence class.
    message = interaction.message
    if is_gs and message.ndim == 3:
        message = message.argmax(dim=-1)
    labels = interaction.labels
    equivalence_class_labels = defaultdict(list)
    for msg, label in zip(message, labels):
        equivalence_class_labels[tuple(msg.tolist())].append(label)
    purity_sum = 0
    for labels in equivalence_class_labels.values():
        label_counts = torch.bincount(torch.tensor(labels))
        purity_sum += label_counts.max().item()
    purity = purity_sum / interaction.size
    return purity


class MessagePurityCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = False

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        purity = message_purity_from_interaction(logs, self.is_gs)
        return f'{data_mode} message purity', purity


class MessagePurityBaselineCallback(MyCallback):
    """
    Message purity is strongly affected by the number and size of equivalence classes. This callback calculates the
    purity of messages with the same size but random content.
    """
    logs_plots = False
    only_inference = True
    transform_inputs = False

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        logs = randomize_message_content(logs)
        purity = message_purity_from_interaction(logs, self.is_gs)
        return f'{data_mode} message purity (random baseline)', purity


class MessagePurityShapesCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = False

    attribute_indices = {'shape': 0, 'color': 1}

    def __init__(self, is_gs, attribute, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs
        self.attribute = attribute
        self.attribute_index = self.attribute_indices[attribute]

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        message = logs.message
        if self.is_gs and message.ndim == 3:
            message = message.argmax(dim=-1)
        labels = logs.labels
        equivalence_class_labels = defaultdict(list)
        for msg, label_tensor in zip(message, labels):
            label = label_tensor.tolist()[self.attribute_index]
            equivalence_class_labels[tuple(msg.tolist())].append(label)
        purity_sum = 0
        for labels in equivalence_class_labels.values():
            label_counts = torch.bincount(torch.tensor(labels))
            purity_sum += label_counts.max().item()
        purity = purity_sum / logs.size
        return f'{data_mode} message purity - attribute {self.attribute}', purity


class MessagePurityShapesBaselineCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = False

    attribute_indices = {'shape': 0, 'color': 1}

    def __init__(self, is_gs, attribute, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs
        self.attribute = attribute
        self.attribute_index = self.attribute_indices[attribute]

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        logs = randomize_message_content(logs)
        _, purity = MessagePurityShapesCallback._calculate_value_to_log(self, loss, logs, game_mode, data_mode)
        return f'{data_mode} message purity - attribute {self.attribute} (random baseline)', purity


class MaxAttributeMessagePurityCallback(MyCallback):
    """
    This callback calculates the purity of each message with respect to the attribute that yields the best purity.
    """
    logs_plots = False
    only_inference = True
    transform_inputs = False

    attribute_indices = {'shape': 0, 'color': 1}

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        message = logs.message
        if self.is_gs and message.ndim == 3:
            message = message.argmax(dim=-1)
        labels = logs.labels
        equivalence_class_labels = defaultdict(lambda: defaultdict(list))
        for msg, label_tensor in zip(message, labels):
            for attribute_idx, label in enumerate(label_tensor.tolist()):
                equivalence_class_labels[tuple(msg.tolist())][attribute_idx].append(label)
        purity_sum = 0
        for attribute_dict in equivalence_class_labels.values():
            label_counts = [torch.bincount(torch.tensor(labels)).max().item() for labels in attribute_dict.values()]
            purity_sum += max(label_counts)
        purity = purity_sum / logs.size
        return f'{data_mode} max-attribute message purity', purity


class MaxAttributeMessagePurityBaselineCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = False

    attribute_indices = {'shape': 0, 'color': 1}

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        logs = randomize_message_content(logs)
        _, purity = MaxAttributeMessagePurityCallback._calculate_value_to_log(self, loss, logs, game_mode, data_mode)
        return f'{data_mode} max-attribute message purity (random baseline)', purity


class PosDisCallback(MyCallback):
    """
    Positional Disentanglement: from "Compositionality and Generalization in Emergent Languages"
    """
    logs_plots = False
    only_inference = True
    transform_inputs = False

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        message = logs.message
        labels = logs.labels
        if self.is_gs and message.ndim == 3:
            message = message.argmax(dim=-1)
        freq_dicts = defaultdict(lambda: defaultdict(int))
        for idx in range(logs.size):
            msg = message[idx].tolist()
            # print(f"{msg=}\nlabels={labels[idx]}")      # msg=(3, 3, 3, 3, 3, 0), labels=tensor([9, 1])
            label_values = labels[idx].tolist()
            for pos, symbol in enumerate(msg):
                freq_dicts[f"pos_{pos}"][symbol] += 1
                for label_idx, label_val in enumerate(label_values):
                    freq_dicts[f"label_{label_idx}"][label_val] += 1
                    freq_dicts[f"pos_{pos}_label_{label_idx}"][(symbol, label_val)] += 1
        posdis = 0.0
        for pos in range(message.size(1)):
            max_mi = None
            second_max_mi = None
            pos_entropy = entropy_dict(freq_dicts[f"pos_{pos}"])
            if pos_entropy == 0:
                continue
            for label_idx in range(labels.size(1)):
                label_entropy = entropy_dict(freq_dicts[f"label_{label_idx}"])
                joint_entropy = entropy_dict(freq_dicts[f"pos_{pos}_label_{label_idx}"])
                mi = pos_entropy + label_entropy - joint_entropy
                if max_mi is None or mi > max_mi:
                    second_max_mi = max_mi
                    max_mi = mi
                elif second_max_mi is None or mi > second_max_mi:
                    second_max_mi = mi
            ratio = (max_mi - second_max_mi) / pos_entropy
            posdis += ratio
        return f'{data_mode} PosDis', posdis


class BosDisCallback(MyCallback):
    """
    Bag-of-symbols Disentanglement: from "Compositionality and Generalization in Emergent Languages"
    """
    logs_plots = False
    only_inference = True
    transform_inputs = False

    def __init__(self, is_gs, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs
        self.vocab_size = vocab_size

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        message = logs.message
        labels = logs.labels
        if self.is_gs and message.ndim == 3:
            message = message.argmax(dim=-1)
        freq_dicts = defaultdict(lambda: defaultdict(int))
        for idx in range(logs.size):
            msg = message[idx]
            symbol_counts = msg.bincount(minlength=self.vocab_size).tolist()
            label_values = labels[idx].tolist()
            for symbol, count in enumerate(symbol_counts):
                freq_dicts[f"symbol_{symbol}"][count] += 1
                for label_idx, label_val in enumerate(label_values):
                    freq_dicts[f"label_{label_idx}"][label_val] += 1
                    freq_dicts[f"symbol_{symbol}_label_{label_idx}"][(count, label_val)] += 1
        bosdis = 0.0
        for symbol in range(self.vocab_size):
            max_mi = None
            second_max_mi = None
            symbol_entropy = entropy_dict(freq_dicts[f"symbol_{symbol}"])
            if symbol_entropy == 0:
                continue
            for label_idx in range(labels.size(1)):
                label_entropy = entropy_dict(freq_dicts[f"label_{label_idx}"])
                joint_entropy = entropy_dict(freq_dicts[f"symbol_{symbol}_label_{label_idx}"])
                mi = symbol_entropy + label_entropy - joint_entropy
                if max_mi is None or mi > max_mi:
                    second_max_mi = max_mi
                    max_mi = mi
                elif second_max_mi is None or mi > second_max_mi:
                    second_max_mi = mi
            ratio = (max_mi - second_max_mi) / symbol_entropy
            bosdis += ratio
        return f'{data_mode} BosDis', bosdis


class SpeakerPosDisCallback(MyCallback):
    """
    Speaker-Positional Disentanglement: from "Visual Referential Games Further the Emergence of Disentangled Representations"
    """
    logs_plots = False
    only_inference = True
    transform_inputs = False

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        message = logs.message
        labels = logs.labels
        if self.is_gs and message.ndim == 3:
            message = message.argmax(dim=-1)
        freq_dicts = defaultdict(lambda: defaultdict(int))
        for idx in range(logs.size):
            msg = message[idx].tolist()
            # print(f"{msg=}\nlabels={labels[idx]}")      # msg=(3, 3, 3, 3, 3, 0), labels=tensor([9, 1])
            label_values = labels[idx].tolist()
            for pos, symbol in enumerate(msg):
                freq_dicts[f"pos_{pos}"][symbol] += 1
                for label_idx, label_val in enumerate(label_values):
                    freq_dicts[f"label_{label_idx}"][label_val] += 1
                    freq_dicts[f"pos_{pos}_label_{label_idx}"][(symbol, label_val)] += 1
        posdis = 0.0
        for label_idx in range(labels.size(1)):
            max_mi = None
            second_max_mi = None
            label_entropy = entropy_dict(freq_dicts[f"label_{label_idx}"])
            if label_entropy == 0:
                continue
            for pos in range(message.size(1)):
                pos_entropy = entropy_dict(freq_dicts[f"pos_{pos}"])
                joint_entropy = entropy_dict(freq_dicts[f"pos_{pos}_label_{label_idx}"])
                mi = pos_entropy + label_entropy - joint_entropy
                if max_mi is None or mi > max_mi:
                    second_max_mi = max_mi
                    max_mi = mi
                elif second_max_mi is None or mi > second_max_mi:
                    second_max_mi = mi
            ratio = (max_mi - second_max_mi) / label_entropy
            posdis += ratio
        return f'{data_mode} Speaker-PosDis', posdis


class PCAByMessagePlotCallback(MyCallback):
    logs_plots = True
    only_inference = True
    transform_inputs = True

    def __init__(self, is_gs, num_samples: int = 100, num_messages: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs
        self.num_samples = num_samples
        self.num_messages = num_messages
        # ^ number of messages to plot from. We take the messages with the largest equivalence classes.

    @staticmethod
    def perform_pca(logs: core.Interaction):
        flattened_images = logs.sender_input.flatten(start_dim=1)
        # standardize the data
        flattened_images = (flattened_images - flattened_images.mean(dim=0)) / flattened_images.std(dim=0)
        # compute the covariance matrix
        # covariance_matrix = torch.matmul(flattened_images.T, flattened_images) / flattened_images.size(0)
        covariance_matrix = torch.cov(flattened_images.T)
        # compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        # sort the eigenvectors by decreasing eigenvalues
        sorted_indices = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # project the data onto the first two principal components
        projected_data = torch.matmul(flattened_images, eigenvectors[:, :2])
        # calculate the explained variance
        explained_variance = eigenvalues[:2].sum() / eigenvalues.sum()
        return projected_data, explained_variance

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        # perform PCA
        projected_data, explained_variance = self.perform_pca(logs)
        # get equivalence classes
        message = logs.message
        if self.is_gs and message.ndim == 3:
            message = message.argmax(dim=-1)
        equivalence_classes = defaultdict(list)
        for i, msg in enumerate(message):
            equivalence_classes[tuple(msg.tolist())].append(projected_data[i])
        # plot the data
        fig, ax = plt.subplots()
        message_sizes = {m: len(images) for m, images in equivalence_classes.items()}
        largest_messages = sorted(message_sizes, key=message_sizes.get, reverse=True)[:self.num_messages]
        num_samples_per_message = self.num_samples // len(largest_messages)
        for m in largest_messages:
            images = equivalence_classes[m]
            images = images[:num_samples_per_message]
            images = torch.stack(images).numpy()
            ax.scatter(images[:, 0], images[:, 1])
        ax.set_title(f'PCA by message ({data_mode} data, explained variance: {explained_variance:.2f})')
        return f'PCA by message ({data_mode} data)', [fig]


class PCAByLabelPlotCallback(MyCallback):
    logs_plots = True
    only_inference = True
    transform_inputs = True

    def __init__(self, num_samples: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        # perform PCA
        projected_data, explained_variance = PCAByMessagePlotCallback.perform_pca(logs)
        # plot the data
        fig, ax = plt.subplots()
        projected_data = projected_data[:self.num_samples]
        labels = logs.labels[:self.num_samples]
        ax.scatter(projected_data[:, 0], projected_data[:, 1], c=labels, cmap='viridis')
        ax.set_title(f'PCA by label ({data_mode} data, explained variance: {explained_variance:.2f})')
        return f'PCA by label ({data_mode} data)', [fig]


class PCAByClusterPlotCallback(MyCallback):
    logs_plots = True
    only_inference = True
    transform_inputs = True

    def __init__(self, is_gs, vocab_subsets: tuple[int], num_samples: int = 100, num_clusters: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.is_gs = is_gs
        self.vocab_subsets = vocab_subsets
        self.num_clusters = num_clusters
        self.perform_pca = PCAByMessagePlotCallback.perform_pca

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        # replace messages with their cluster
        logs = replace_messages_with_clusters(logs, self.is_gs, self.vocab_subsets)
        # perform PCA
        projected_data, explained_variance = PCAByMessagePlotCallback.perform_pca(logs)
        # get equivalence classes
        clusters = logs.message
        if self.is_gs and clusters.ndim == 3:
            clusters = clusters.argmax(dim=-1)
        equivalence_classes = defaultdict(list)
        for i, cluster in enumerate(clusters):
            equivalence_classes[tuple(cluster.tolist())].append(projected_data[i])
        # plot the data
        fig, ax = plt.subplots()
        cluster_sizes = {cluster: len(images) for cluster, images in equivalence_classes.items()}
        largest_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:self.num_clusters]
        num_samples_per_message = self.num_samples // len(largest_clusters)
        for cluster in largest_clusters:
            images = equivalence_classes[cluster]
            images = images[:num_samples_per_message]
            images = torch.stack(images).numpy()
            ax.scatter(images[:, 0], images[:, 1])
        ax.set_title(f'PCA by cluster ({data_mode} data, explained variance: {explained_variance:.2f})')
        return f'PCA by cluster ({data_mode} data)', [fig]


class ZipfBarPlotCallback(MyCallback):
    logs_plots = True
    only_inference = True
    transform_inputs = False

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        message = logs.message
        if self.is_gs and message.ndim == 3:
            message = message.argmax(dim=-1)
        msg_counts = defaultdict(int)
        for i, msg in enumerate(message):
            msg_counts[tuple(msg.tolist())] += 1
        counts = torch.tensor(list(msg_counts.values()))
        counts, indices = counts.sort(descending=True)
        fig, ax = plt.subplots()
        ax.bar(range(len(counts)), counts)
        ax.set_title(f'Zipf plot ({data_mode} data)')
        return f'Zipf plot ({data_mode} data)', [fig]


class MessageLengthCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = False

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        length = logs.message_length.float().mean().item()
        return f'message length on {data_mode} data', length


class LossCallback(MyCallback):
    logs_plots = False
    only_inference = False
    transform_inputs = False

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        return f'{data_mode} loss ({game_mode} messages)', loss


class TopographicSimilarityCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = True

    def __init__(self, is_gs, vocab_size, message_distance: str, num_samples: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs
        self.vocab_size = vocab_size
        self.message_distance = message_distance
        self.num_samples = num_samples  # computing topsim is expensive, so we'll use a subset of the data

    @staticmethod
    def image_criterion(img1, img2):
        mse = ((img1 - img2) ** 2).mean()
        return mse

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        logs = limit_interaction(logs, self.num_samples)
        message = logs.message
        if self.is_gs and message.ndim == 3:
            message = message.argmax(dim=-1)
        if torch.allclose(message, message[0].repeat(message.size(0), 1)):
            print("WARNING: degenerate sender!")
            # full randomness - minimum compositionality
            message = torch.randint(self.vocab_size, size=message.size())
        flattened_images = logs.sender_input.flatten(start_dim=1)
        topsim = TopographicSimilarity.compute_topsim(flattened_images,
                                                      message,
                                                      self.image_criterion,
                                                      self.message_distance)
        return f'topographic similarity on {self.num_samples} {data_mode} inputs', topsim


class ReducedTopographicSimilarityCallback(MyCallback):
    # Equivalence classes are reduced to their mean, and the topographic similarity is computed on these means.
    logs_plots = False
    only_inference = True
    transform_inputs = True

    def __init__(self, is_gs, vocab_size, message_distance: str, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs
        self.vocab_size = vocab_size
        self.message_distance = message_distance

    @staticmethod
    def image_criterion(img1, img2):
        mse = ((img1 - img2) ** 2).mean()
        return mse

    @staticmethod
    def reduce_equivalence_classes(interaction: core.Interaction):
        message = interaction.message
        if message.ndim == 3:
            message = message.argmax(dim=-1)
        flattened_images = interaction.sender_input.flatten(start_dim=1)
        equivalence_classes = defaultdict(list)
        for i, msg in enumerate(message):
            equivalence_classes[tuple(msg.tolist())].append(flattened_images[i])
        mean_images = {m: torch.stack(images).mean(dim=0) for m, images in equivalence_classes.items()}
        messages, mean_images = zip(*mean_images.items())
        message = torch.tensor(messages)
        mean_images = torch.stack(mean_images)
        return message, mean_images

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        message, mean_images = self.reduce_equivalence_classes(logs)
        if message.size(0) == 1:
            print("WARNING: degenerate sender!")
            topsim = 0
        else:
            topsim = TopographicSimilarity.compute_topsim(mean_images,
                                                          message,
                                                          self.image_criterion,
                                                          self.message_distance)
        return f'reduced topographic similarity ({data_mode} data)', topsim


def empirical_message_variance_from_interaction(interaction: core.Interaction, is_gs: bool):
    """ computes the average empirical variance of equivalence classes (weighted average). """
    message = interaction.message
    if is_gs and message.ndim == 3:
        message = message.argmax(dim=-1)
    flattened_images = interaction.sender_input.flatten(start_dim=1)
    equivalence_classes = defaultdict(list)
    for i, msg in enumerate(message):
        equivalence_classes[tuple(msg.tolist())].append(flattened_images[i])
    distance_sum = 0
    for images in equivalence_classes.values():
        images_t = torch.stack(images).unsqueeze(0)
        pairwise_distances = torch.cdist(images_t, images_t, p=2)
        message_distance_sum = pairwise_distances.pow(2).sum() / (images_t.size(1))
        distance_sum += message_distance_sum
    cond_variance = distance_sum / (interaction.size * 2)
    return cond_variance.item()


class EmpiricalMessageVarianceCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = True

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        cond_variance = empirical_message_variance_from_interaction(logs, self.is_gs)
        return f'{data_mode} empirical message variance', cond_variance


class EmpiricalMessageVarianceBaselineCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = True

    def __init__(self, is_gs, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        logs = randomize_message_content(logs)
        cond_variance = empirical_message_variance_from_interaction(logs, self.is_gs)
        return f'{data_mode} empirical message variance (random baseline)', cond_variance


def empirical_cluster_variance_from_interaction(interaction: core.Interaction, is_gs: bool, vocab_subsets: tuple[int]):
    """ computes the average empirical variance of the msesage clusters, by uniting the equivalence classes belonging
    to the same cluster. """
    modified_interaction = replace_messages_with_clusters(interaction, is_gs, vocab_subsets)
    empirical_variance = empirical_message_variance_from_interaction(modified_interaction, is_gs)
    return empirical_variance


class EmpiricalClusterVarianceCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = True

    def __init__(self, is_gs, vocab_subsets, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs
        self.vocab_subsets = vocab_subsets

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        cond_variance = empirical_cluster_variance_from_interaction(logs, self.is_gs, self.vocab_subsets)
        return f'{data_mode} empirical cluster variance', cond_variance


class EmpiricalClusterVarianceBaselineCallback(MyCallback):
    logs_plots = False
    only_inference = True
    transform_inputs = True

    def __init__(self, is_gs, vocab_subsets, **kwargs):
        super().__init__(**kwargs)
        self.is_gs = is_gs
        self.vocab_subsets = vocab_subsets

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        logs = randomize_message_content(logs)
        cond_variance = empirical_cluster_variance_from_interaction(logs, self.is_gs, self.vocab_subsets)
        return f'{data_mode} empirical cluster variance (random baseline)', cond_variance


class ConsoleLogger(core.Callback):
    def __init__(self, print_train_loss=False, as_json=False):
        self.print_train_loss = print_train_loss
        self.as_json = as_json

    def aggregate_print(self, loss: float, logs: core.Interaction, mode: str, epoch: int):
        # length = logs.aux['length'].mean().item() if 'length' in logs.aux else None
        dump = dict(loss=loss, length=logs.aux['length'].mean().item())
        # aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        # dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, "train", epoch)


def plot_autoencoding_from_interaction(interaction: core.Interaction, is_gs: bool, game_config, *,
                                       training_mode: bool, noise_data: bool = False, num_figs: int = None):
    figs = []
    interaction_size = interaction.sender_input.size(0)
    num_figs = interaction_size if num_figs is None else min(num_figs, interaction_size)
    for z in range(num_figs):
        src = interaction.sender_input[z]
        message = interaction.message[z]
        if is_gs:
            message_length = int(interaction.aux['length'][z])
            pred_idx = message_length - 1 if game_config.include_eos_token else 0
            dst = interaction.receiver_output[z][pred_idx]
            if len(message.shape) == 2:
                # message represented by one-hot encoding instead of actual symbols
                # this happens when used as a callback, because dump_interactions is never called
                message = message.argmax(dim=-1)
        else:
            dst = interaction.receiver_output[z]
        dst = dst.view_as(src)
        # we'll plot two images side-by-side: the original (left) and the reconstruction
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        fig.suptitle(f"Channel message {message.tolist()}{'*' if training_mode else ''}")
        ax1.imshow(src.permute(1, 2, 0), cmap='gray')
        ax1.set_title(f"original ({'noise' if noise_data else game_config.dataset})")
        ax2.imshow(dst.permute(1, 2, 0), cmap='gray')
        ax2.set_title("reconstructed")
        figs.append(fig)
    return figs


class AutoencodingPlotCallback(MyCallback):
    logs_plots = True
    only_inference = False
    transform_inputs = False

    def __init__(self, is_gs, game_config, num_plots=1, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = dict(is_gs=is_gs, game_config=game_config)
        self.num_plots = num_plots

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        figs = plot_autoencoding_from_interaction(logs, num_figs=self.num_plots, **self.kwargs,
                                                  training_mode=game_mode == "train",
                                                  noise_data=data_mode == "noise")
        name = f"{data_mode} autoencoding ({game_mode} mode)"
        return name, figs


def disc_accuracy_for_reco_from_interaction(interaction: core.Interaction, num_distractors, game_config, loss_func):
    bsz = num_distractors + 1
    receiver_output = interaction.receiver_output
    if game_config.discretization_method == 'gs' and len(receiver_output.shape) == 3:
        message_indices = (interaction.message_length - 1).long() if game_config.include_eos_token else 0
        receiver_output = receiver_output[torch.arange(interaction.size), message_indices, :]
    assert bsz <= interaction.size, "interaction size must be greater than the number of distractors"
    limit = receiver_output.size(0) // bsz * bsz  # truncate to a multiple of bsz
    receiver_output = receiver_output[:limit]
    receiver_output = receiver_output.view(-1, bsz, *receiver_output.shape[1:])
    # original_images = interaction.aux_input['images']
    original_images = interaction.sender_input[:limit]
    original_images = original_images.view(-1, bsz, *original_images.shape[1:])

    try:
        device = next(loss_func.parameters()).device
    except StopIteration:
        device = 'cpu'

    @torch.no_grad()
    def loss_func_wrapper(t, pred):
        return loss_func(t, None, None, pred, None)[0].item()

    accuracies = []
    for original_img_batch, output_batch in zip(original_images, receiver_output):
        num_correct = 0
        for i in range(bsz):
            losses = [loss_func_wrapper(img.unsqueeze(0).to(device), output_batch[i].to(device)) for img in
                      original_img_batch]
            # t_position = sorted(losses).index(losses[i])
            best_index, min_loss = min(enumerate(losses), key=lambda tup: tup[1])
            if best_index == i:
                num_correct += 1
        accuracies.append(num_correct / bsz)

    overall_accuracy = sum(accuracies) / len(accuracies)
    return overall_accuracy


class DiscAccuracyForRecoCallback(MyCallback):
    logs_plots = False
    only_inference = False
    transform_inputs = False

    def __init__(self, num_distractors, game_config, loss_func, num_samples: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.num_distractors = num_distractors
        self.kwargs = dict(num_distractors=num_distractors,
                           game_config=game_config,
                           loss_func=loss_func)
        self.num_samples = num_samples

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        logs = limit_interaction(logs, self.num_samples)
        accuracy = disc_accuracy_for_reco_from_interaction(logs, **self.kwargs)
        name = f"{data_mode} discrimination accuracy ({self.num_distractors} distractors), {game_mode} mode"
        return name, accuracy


def get_accuracy_from_interaction(interaction: core.Interaction, is_gs, game_config, num_distractors=None):
    num_distractors = num_distractors if num_distractors is not None else game_config.num_distractors
    receiver_output = interaction.receiver_output
    if is_gs and len(receiver_output.shape) == 3:
        message_indices = (interaction.message_length - 1).long() if game_config.include_eos_token else 0
        receiver_output = receiver_output[torch.arange(interaction.size), message_indices, :]
    assert receiver_output.size(
        -1) >= num_distractors + 1, "requested a number of distractors larger than batch size"
    receiver_output = receiver_output[:, :num_distractors + 1]
    preds = receiver_output.argmax(dim=-1)
    correct = (preds == 0)
    accuracy = correct.float().mean().item()
    return accuracy


class DiscAccuracyCallback(MyCallback):
    logs_plots = False
    only_inference = False
    transform_inputs = False

    def __init__(self, num_distractors, is_gs, game_config, **kwargs):
        super().__init__(**kwargs)
        self.num_distractors = num_distractors
        self.kwargs = dict(num_distractors=num_distractors,
                           is_gs=is_gs,
                           game_config=game_config)

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        accuracy = get_accuracy_from_interaction(logs, **self.kwargs)
        name = f"{data_mode} discrimination accuracy ({self.num_distractors} distractors), {game_mode} mode"
        return name, accuracy


def get_plot_from_interaction(interaction: core.Interaction, is_gs, game_config, num_plots, num_distractors, *,
                              training_mode=False):
    # this function makes a single plot for each batch in the given interaction. Thus, it expects the interaction to
    # contain exactly `num_plots` batches.
    bsz = game_config.batch_size
    assert interaction.size % bsz == 0
    # take only num_plots batches
    interaction = limit_interaction(interaction, num_plots * bsz)
    receiver_output = interaction.receiver_output
    message = interaction.message
    if is_gs and len(receiver_output.shape) == 3:
        message_indices = (interaction.message_length - 1).long() if game_config.include_eos_token else 0
        receiver_output = receiver_output[torch.arange(interaction.size), message_indices, :]
        if len(message.shape) == 3:
            message = message.argmax(dim=-1)

    assert receiver_output.size(-1) >= num_distractors + 1, "requested a number of distractors larger than the one used in training"

    # take only the desired number of distractors
    receiver_output = receiver_output[:, :num_distractors + 1]
    candidate_indices = interaction.aux_input['candidate_indices'][:, :num_distractors + 1]
    original_images = interaction.sender_input

    original_images = original_images.view(num_plots, bsz, *original_images.shape[1:])
    receiver_output = receiver_output.view(num_plots, bsz, num_distractors + 1)
    candidate_indices = candidate_indices.view(num_plots, bsz, num_distractors + 1)
    message = message.view(num_plots, bsz, -1)

    figs = []
    for image_batch, output_batch, indices_batch, msg in zip(original_images,
                                                             receiver_output,
                                                             candidate_indices,
                                                             message):
        # we plot the discrimination of the first image in each batch
        output_to_plot = output_batch[0]
        candidate_images = image_batch[indices_batch[0]]     # first image is the target.
        original_image = image_batch[0]     # same as target unless discrimination strategy is "classification"
        msg_to_plot = msg[0]

        fig: plt.Figure = plt.figure(constrained_layout=True)
        subfig1, subfig2 = fig.subfigures(nrows=1, ncols=2)
        subfig1: SubFigure
        subfig2: SubFigure
        fig.suptitle(f"Channel message {msg_to_plot.tolist()}{'*' if training_mode else ''}")
        ax1 = subfig1.subplots()
        # ax1.imshow(distractor_images[0].permute(1, 2, 0), cmap='gray')
        ax1.imshow(original_image.permute(1, 2, 0), cmap='gray')
        ax1.set_title(f"input image ({game_config.dataset})")
        ax1.set_xticks([])
        ax1.set_yticks([])
        subfig2.suptitle("candidates")
        # ax2 is a subplot with each distractor image and its corresponding score from output_to_plot
        axes2 = subfig2.subplots(nrows=num_distractors + 1)
        for ax, image, score in zip(axes2, candidate_images, output_to_plot):
            ax.imshow(image.permute(1, 2, 0), cmap='gray')
            ax.set_title(f"score: {score:.3f}")
            ax.set_xticks([])
            ax.set_yticks([])
        figs.append(fig)

    return figs


class DiscriminationPlotCallback(MyCallback):
    logs_plots = True
    only_inference = False
    transform_inputs = False

    def __init__(self, is_gs, game_config, num_plots, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = dict(num_distractors=min(3, game_config.num_distractors),
                           is_gs=is_gs,
                           game_config=game_config,
                           num_plots=num_plots)

    def _calculate_value_to_log(self, loss, logs: core.Interaction, game_mode: str, data_mode: str):
        figs = get_plot_from_interaction(logs, **self.kwargs, training_mode=game_mode == "train")
        name = f"{data_mode} discrimination plot, {game_mode} mode"
        return name, figs
