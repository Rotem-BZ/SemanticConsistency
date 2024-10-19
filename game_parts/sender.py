from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_training import get_pretrained_encoder, ModelSavePath
from .vision_architectures import initialize_encoder, LATENT_DIM_DICT
from .message_generation import RNNGenerativeModel, GenerativeAgentWrapper


class ConvSender(nn.Module):
    """
    This encoder receives an image and outputs a vector, which can be used as an initial hidden state for some RNN which
    will generate the discrete message.
    The vision module that encodes the image can be frozen or learnable, and the linear layer is learnable.
    """

    def __init__(self, out_dim: int, encoder_path: ModelSavePath, frozen_encoder: bool, cached_encoder: bool):
        super(ConvSender, self).__init__()
        if cached_encoder and not frozen_encoder:
            raise ValueError("cached encoder must be frozen")
        if encoder_path.model_type is None:
            # make new encoder
            assert not frozen_encoder, "frozen encoder is not trained!"
            self.encoder = initialize_encoder(encoder_path.dataset, encoder_path.architecture_type)
        else:
            # load pretrained encoder
            self.encoder = get_pretrained_encoder(encoder_path)

        self.frozen_encoder = frozen_encoder
        self.cached_encoder = cached_encoder
        self.fc = nn.Linear(LATENT_DIM_DICT[encoder_path.dataset], out_dim)

        if self.frozen_encoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

    def forward(self, img: torch.Tensor, aux_input=None):
        if aux_input is not None and 'encodings' in aux_input:
            encodings = aux_input['encodings']
        else:
            encodings = self.encoder(img)
        encodings = self.fc(encodings)
        return encodings

    # future idea: add `to` and `train` methods, so that the encoder is kept on cpu and eval mode if cached.


class RandomSender(nn.Module):
    """
    Outputs random full-length messages consistently (the same input will always produce the same message).
    Note that if num_unique_messages is not None, all the unique messages are stored in the object, so a large value
    will consume a lot of memory.
    """
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 discretization_method: str,
                 legal_vocab_subsets,
                 use_eos_token: bool,
                 num_unique_messages: Optional[int] = None):
        super(RandomSender, self).__init__()
        print("Using a RandomSender!")
        self.vocab_size = vocab_size
        # effective vocab size is vocab_size - 1 if EOS token is used
        self.max_len = max_len
        self.disc_method = discretization_method
        self.legal_vocab_subsets = legal_vocab_subsets
        self.subset_indices = None
        self.use_eos_token = use_eos_token

        self.vocab_subset_probabilities = None
        if legal_vocab_subsets is not None:
            self.vocab_subset_probabilities = torch.tensor(legal_vocab_subsets).float()
            self.vocab_subset_probabilities /= self.vocab_subset_probabilities.sum()
            start_index = 1 if use_eos_token else 0
            self.subset_indices = [start_index] + torch.cumsum(torch.tensor(legal_vocab_subsets), dim=0).tolist()
        else:
            start_index = 1 if use_eos_token else 0
            self.subset_indices = [start_index, vocab_size]

        self.num_unique_messages = num_unique_messages
        self.unique_messages = None
        if num_unique_messages is not None:
            if legal_vocab_subsets is None:
                num_unique_messages_per_subset = [num_unique_messages]
            else:
                message_counts = torch.multinomial(self.vocab_subset_probabilities,
                                                   num_samples=num_unique_messages,
                                                   replacement=True)
                num_unique_messages_per_subset = torch.bincount(message_counts, minlength=len(legal_vocab_subsets))
                assert num_unique_messages_per_subset.sum() == num_unique_messages  # sanity check
                # recalculate the subset probabilities based on the actual number of unique messages
                self.vocab_subset_probabilities = num_unique_messages_per_subset.float() / num_unique_messages
                num_unique_messages_per_subset = num_unique_messages_per_subset.int().tolist()
            self.unique_messages = []
            for num_unique_messages, low, high in zip(num_unique_messages_per_subset,
                                                      self.subset_indices[:-1],
                                                      self.subset_indices[1:]):
                messages = self._generate_messages(low, high, max_len, num_unique_messages)
                self.unique_messages.append(messages)

    @staticmethod
    def _generate_messages(low, high, length, num_messages):
        # if num_messages is 0, this will return an empty list
        assert num_messages <= (high - low) ** length, "too many messages to generate"
        messages = []
        for _ in range(num_messages):
            found = False
            while not found:
                message = torch.randint(low, high, (length,)).tolist()
                if message not in messages:
                    messages.append(message)
                    found = True
        return messages

    def forward(self, img: torch.Tensor, aux_input=None):
        # we use the sum of each input as seed for the random generator
        device = img.device
        generators = [torch.Generator().manual_seed(int(x.sum().item() * 1000)) for x in img]

        messages = []
        for i, gen in enumerate(generators):
            # 1. sample a subset
            if self.legal_vocab_subsets is not None:
                subset = torch.multinomial(self.vocab_subset_probabilities, 1, generator=gen).item()
            else:
                subset = 0
            # 2. sample a message
            if self.unique_messages is None:
                low = self.subset_indices[subset]
                high = self.subset_indices[subset + 1]
                messages = [torch.randint(low, high, (self.max_len,), generator=gen).tolist() for gen in
                            generators]
            else:
                potential_messages = self.unique_messages[subset]
                choice = torch.randint(len(potential_messages), size=(1,), generator=gen).item()
                messages.append(potential_messages[choice])

        if self.disc_method == 'reinforce':
            raise NotImplementedError("RandomSender doesn't support reinforce yet")
        elif self.disc_method == 'gs':
            messages = torch.tensor(messages, device=device)
            if self.use_eos_token:
                eos = torch.zeros_like(messages[:, :1])
                messages = torch.cat([messages, eos], dim=1)
            return F.one_hot(messages, num_classes=self.vocab_size).float()
        else:
            raise ValueError(f"illegal discretization_method {self.disc_method}")


def get_sender(game_config, game_type):
    disc_method = game_config.discretization_method.lower()
    if game_config.random_sender:
        return RandomSender(vocab_size=game_config.vocab_size,
                            max_len=game_config.max_len,
                            discretization_method=disc_method,
                            legal_vocab_subsets=game_config.legal_vocab_subsets,
                            use_eos_token=game_config.include_eos_token,
                            num_unique_messages=game_config.num_unique_random_messages)
    gen_model_base_kwargs = dict(discretization_method=disc_method,
                                 vocab_size=game_config.vocab_size,
                                 max_len=game_config.max_len,
                                 temperature=game_config.temperature,
                                 trainable_temperature=game_config.trainable_temperature,
                                 include_eos_token=game_config.include_eos_token,
                                 legal_vocab_subsets=game_config.legal_vocab_subsets,)
    gen_model_name = game_config.sender_gen_model
    if gen_model_name in ['rnn', 'gru', 'lstm']:
        out_dim = game_config.sender_RNN_hidden_dim
        gen_model = RNNGenerativeModel(**gen_model_base_kwargs,
                                       hidden_size=out_dim,
                                       embed_dim=game_config.sender_RNN_emb_size,
                                       cell=gen_model_name)
    else:
        raise ValueError(f"illegal gen model {gen_model_name}")
    encoder_path = ModelSavePath(game_config.dataset, game_config.encoder_pretraining_type,
                                 game_config.architecture_type, game_config.encoder_cpt_name)
    agent = ConvSender(out_dim=out_dim, encoder_path=encoder_path, frozen_encoder=game_config.frozen_encoder,
                       cached_encoder=game_config.use_cached_encoder)
    if game_config.verbose:
        print(f"\n\nsender encoder:\n{agent.encoder}\n\n")
        print(f"\n\nsender generative model:\n{gen_model}\n\n")
    # add the generative model to the agent
    agent = GenerativeAgentWrapper(agent, gen_model)
    return agent
