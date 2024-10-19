# Choose the script you which to run in main.
import platform
from pathlib import Path
from pprint import pprint

from game import ImageReconstructionGame, ImageDiscriminationGame, DISCRETE_SAVE_DIR
from game_parts.sender import get_sender
from game_parts.continuous_training import CONTINUOUS_SAVE_DIR, ModelSavePath, get_pretrained_encoder, get_special_models
from game_parts.data import data_utils

import torch
from torch.utils.data import DataLoader
from datasets import Dataset


# # if training on cpu, this prevents a "too many open files" error
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


def interactive_load_game(communication_type: str = None, game_type: str = None, save_name: str = None):
    # choose a specific checkpoint to load. missing arguments will be chosen interactively.

    def choose_from_options(text: str, options: list[str]):
        for i, option in enumerate(options):
            text += f"\n({i+1})\t{option}"
        print(text)
        while True:
            try:
                choice = int(input())
                if choice not in range(1, len(options) + 1):
                    raise ValueError
                choice = options[choice - 1]
                print(f"you chose {choice}")
                return choice
            except ValueError:
                print("invalid input, try again")

    if communication_type is None:
        communication_type = choose_from_options("choose communication type:",
                                                 options=['discrete', 'continuous'])
    if communication_type == 'discrete':
        game_class_dict = {'Reconstruction': ImageReconstructionGame,
                           'Discrimination': ImageDiscriminationGame}
        if game_type is None:
            game_type = choose_from_options("choose game type:", options=list(game_class_dict.keys()))
        path = Path(DISCRETE_SAVE_DIR) / game_type
        saved_names = [game.name for game in path.iterdir()]
        if not saved_names:
            raise ValueError(f"no saved models found for {game_type}. use pretrain.py to create one.")
        if save_name is None:
            save_name = choose_from_options("choose saved checkpoint:", options=saved_names)
        game = game_class_dict[game_type].load_from_checkpoint(save_name)
        return game, save_name
    elif communication_type == 'continuous':
        special_models = get_special_models().keys()
        if save_name is None:
            saved_names = list(special_models) + [game.name for game in Path(CONTINUOUS_SAVE_DIR).iterdir()]
            save_name = choose_from_options("choose saved checkpoint:", options=saved_names)
        save_name = save_name if save_name in special_models else ModelSavePath.from_string(save_name)
        model = get_pretrained_encoder(save_name)
        return model, save_name
    else:
        raise ValueError(f"invalid communication type {communication_type}")


@torch.no_grad()
def cache_encoded_images():
    save_dir = Path(data_utils.DATA_PATH) / 'cached_encoded_images'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    save_dir.mkdir(exist_ok=True)
    model, save_name = interactive_load_game(communication_type='continuous')
    model.to(device)
    model.eval()
    if isinstance(save_name, ModelSavePath):
        dataset_name = save_name.dataset
        save_name = save_name.name
    else:
        dataset_name = save_name.split('_')[0]

    datamodule = data_utils.get_pl_datamodule(dataset_name, batch_size=1, batch_limit=(0, 0, 0))
    datamodule.prepare_data()
    data = datamodule._full_train_data()
    loader = DataLoader(data, batch_size=64, shuffle=False)

    def parallel_data_generator():
        for X, y in loader:
            X = X.to(device)
            encoded_images = model(X).cpu()
            for pixels, encoding, label in zip(X, encoded_images, y):
                yield dict(encoded_images=encoding, labels=label)

    ds = Dataset.from_generator(parallel_data_generator)
    print(ds)
    (save_dir / dataset_name).mkdir(exist_ok=True)
    ds.save_to_disk(save_dir / dataset_name / save_name)


def unique_messages_of_random_sender():
    model, save_name = interactive_load_game(communication_type='discrete')
    print(f"{type(model.sender)=}")
    model.sender = get_sender(model.game_config, model.game_type)
    for subset in ['val', 'noise']:
        print(f"\n{subset=}")
        model.print_num_unique_messages(subset)


def main():
    cache_encoded_images()


if __name__ == '__main__':
    main()
