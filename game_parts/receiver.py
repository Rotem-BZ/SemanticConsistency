"""
The Receiver agent consists of two parts:
1. A message embedding model, which embeds the message into a vector or sequence of vectors.
2. A Receiver agent, which receives the message embedding and optionally other inputs, and outputs a prediction.

This file implements the agents, and provides the method `get_receiver` which returns the complete Receiver.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_training import get_pretrained_decoder, get_pretrained_encoder, ModelSavePath
from .vision_architectures import initialize_decoder, initialize_encoder, LATENT_DIM_DICT
from .data.data_utils import DatasetInformationClass, DatasetInformationDict
from .message_embedding import RnnMessageEmbedder


class MLP_Receiver(nn.Module):
    """
    generate image via FC layers. Serves as baseline.
    """

    activations_dict = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }

    def __init__(self, input_size: int, dataset_info: DatasetInformationClass,
                 num_hidden_layers=0, hidden_dim=64, activation='relu'):
        super(MLP_Receiver, self).__init__()
        self.numel_out = dataset_info.number_of_channels * dataset_info.image_size ** 2
        self.input_size = input_size
        self.net = self.get_MLP(self.input_size, self.numel_out, num_hidden_layers, hidden_dim, activation)

    @staticmethod
    def get_MLP(in_dim: int, out_dim: int, num_hidden_layers: int, hidden_dim: int, activation: str):
        activation_func = MLP_Receiver.activations_dict[activation]
        dims = [in_dim] + [hidden_dim] * num_hidden_layers + [out_dim]
        linear_layers = [nn.Linear(dim1, dim2) for dim1, dim2 in zip(dims[:-1], dims[1:])]
        activations = [activation_func() for _ in range(len(linear_layers) - 1)]
        net = nn.Sequential(*sum(zip(linear_layers[:-1], activations), start=()), linear_layers[-1])
        return net

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        return self.net(channel_input)


class DeconvReceiver(nn.Module):
    """
    Receiver which generates image by deconvolution
    """

    def __init__(self, input_dim: int, decoder_path: ModelSavePath, frozen_decoder: bool):
        super(DeconvReceiver, self).__init__()
        if decoder_path.model_type is None:
            assert not frozen_decoder, "frozen decoder is not trained!"
            self.decoder = initialize_decoder(decoder_path.dataset, decoder_path.architecture_type)
        else:
            self.decoder = get_pretrained_decoder(decoder_path)
        self.frozen_decoder = frozen_decoder
        self.fc = nn.Linear(input_dim, LATENT_DIM_DICT[decoder_path.dataset])
        if self.frozen_decoder:
            self.decoder.requires_grad_(False)
        self.flatten = nn.Flatten()

    def forward(self, x, receiver_input=None, aux_input=None):
        x = self.fc(x)
        x = x.view(*x.shape, 1, 1)  # view the given 1D vector as BxCx1x1 image
        x = self.decoder(x)
        x = self.flatten(x)
        return x


class DiscReceiver(nn.Module):
    """
    Receiver which performs inner product between message representation (made by the message embedder) and
    image representation (encoded here via vision encoder + MLP).
    """

    def __init__(self, out_dim: int,
                 encoder_path: ModelSavePath,
                 frozen_encoder: bool,
                 num_distractors: int,
                 discrimination_strategy: str,
                 mlp_num_layers: int,
                 mlp_hidden_dim: int,
                 mlp_activation: str):
        super(DiscReceiver, self).__init__()
        if encoder_path.model_type is None:
            # make new encoder
            assert not frozen_encoder, "frozen encoder is not trained!"
            self.encoder = initialize_encoder(encoder_path.dataset, encoder_path.architecture_type)
        else:
            # load pretrained encoder
            self.encoder = get_pretrained_encoder(encoder_path)

        self.frozen_encoder = frozen_encoder
        self.num_distractors = num_distractors
        self.discrimination_strategy = discrimination_strategy
        assert discrimination_strategy in ['vanilla', 'supervised', 'contrastive', 'classification']

        if self.frozen_encoder:
            self.encoder.requires_grad_(False)

        self.mlp = MLP_Receiver.get_MLP(LATENT_DIM_DICT[encoder_path.dataset],
                                        out_dim,
                                        mlp_num_layers,
                                        mlp_hidden_dim,
                                        mlp_activation)
        # self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, embedded_messages, receiver_input=None, aux_input=None):
        # aux_input contains the original images and their labels. Note that those aren't shuffled, meaning the first
        # image in aux_input matches the first message in embedded_messages. This is ok because the processing of the
        # images and messages is order-agnostic.
        embedded_images = aux_input['images']
        embedded_images = self.encoder(embedded_images)
        embedded_images = self.mlp(embedded_images)
        embedded_images = F.normalize(embedded_images, dim=1, eps=1e-8)
        embedded_messages = F.normalize(embedded_messages, dim=1, eps=1e-8)

        N = self.num_distractors
        B = embedded_messages.size(0)  # batch size

        distractor_indices = None   # BxN, indices of distractors in the batch
        target_indices = None       # B, indices of the target image in the batch

        if self.discrimination_strategy in ['supervised', 'classification']:
            # this avoids chossing positive distractors if possible.
            labels = aux_input['labels']
            diff = torch.ne(labels.unsqueeze(0), labels.unsqueeze(1))  # True <-> can be used as a distractor
            if not torch.any(diff):
                print(f"all inputs in a batch of size {len(labels)} have the same label {labels[0].item()}. "
                      f"Using same-label distractors.")
                diff = ~torch.eye(len(labels), dtype=torch.bool)
            row_idx, col_idx = torch.nonzero(diff, as_tuple=True)
            nonzero_counts = torch.bincount(row_idx)
            min_count = nonzero_counts.min().item()  # smallest numer of potential distractors
            if min_count < N:
                print(f"Not enough distractors in batch. Using {min_count} instead of {N}.")
                N = min_count
            counts_cumsum = torch.cumsum(nonzero_counts, dim=0)
            distractor_indices = torch.stack([col_idx[i - N:i] for i in counts_cumsum])
            if self.discrimination_strategy == 'supervised':
                target_indices = torch.arange(B, device=distractor_indices.device)
            else:
                # any candidate that has the same label as the gt
                same_label = ~diff
                same_label[torch.arange(B), torch.arange(B)] = False
                success, target_indices = torch.max(same_label, dim=1)
                if not success.all().item():
                    print(f"{success.sum()} labels are unique in the batch. Using gt as candidate.")
                    target_indices[~success] = torch.arange(B, device=distractor_indices.device)[~success]
        elif self.discrimination_strategy == 'vanilla':
            # the vanilla strategy is to use the first N images as distractors
            assert N < B, "batch size must be larger than the number of distractors (will be fixed in the future)"
            indices = torch.arange(N).repeat(B, 1)  # indices of distractrs, taken from this batch (first N)
            indices[torch.arange(N), torch.arange(N)] = N  # make sure the gt image isn't used as a distractor
            distractor_indices = indices
            target_indices = torch.arange(B, device=distractor_indices.device)
        else:
            raise ValueError(f"illegal discrimination strategy {self.discrimination_strategy}")

        # add gt as first column
        candidate_indices = torch.cat((target_indices.unsqueeze(1), distractor_indices), dim=1)
        assert aux_input is not None
        aux_input['candidate_indices'] = candidate_indices
        similarity = torch.bmm(embedded_messages.unsqueeze(1), embedded_images[candidate_indices].transpose(1, 2))
        return similarity.squeeze(1)


def get_receiver(game_config, game_type):
    # 1. get the input dimension to the agent
    disc_method = game_config.discretization_method.lower()
    if disc_method in ['gs', 'reinforce']:
        if game_config.receiver_embed_model in ['rnn', 'gru', 'lstm']:
            input_dim = game_config.receiver_RNN_hidden_dim
        else:
            raise ValueError(f"illegal embed model {game_config.embed_model}")
    else:
        raise ValueError(f"illegal discretization method {disc_method}")

    # 2. Create the agent
    if game_type == 'Reconstruction':
        if game_config.receiver_type == 'MLP':
            agent = MLP_Receiver(input_size=input_dim,
                                 dataset_info=DatasetInformationDict[game_config.dataset],
                                 num_hidden_layers=game_config.MLP_num_hidden_layers,
                                 hidden_dim=game_config.MLP_hidden_dim,
                                 activation=game_config.MLP_activation)
        elif game_config.receiver_type == 'deconv':
            decoder_path = ModelSavePath(game_config.dataset, game_config.decoder_pretraining_type,
                                         game_config.architecture_type, game_config.decoder_cpt_name)
            agent = DeconvReceiver(input_dim=input_dim, decoder_path=decoder_path,
                                   frozen_decoder=game_config.frozen_decoder)
            if game_config.verbose:
                print(f"\n\nreceiver decoder:\n{agent.decoder}\n\n")
        else:
            raise ValueError(f"illegal receiver type {game_config.receiver_type} for the Reconstruction game")
    elif game_type == 'Discrimination':
        encoder_path = ModelSavePath(game_config.dataset, game_config.receiver_encoder_pretraining_type,
                                     game_config.architecture_type, game_config.receiver_encoder_cpt_name)
        agent = DiscReceiver(out_dim=input_dim,
                             encoder_path=encoder_path,
                             frozen_encoder=game_config.receiver_frozen_encoder,
                             num_distractors=game_config.num_distractors,
                             discrimination_strategy=game_config.discrimination_strategy,
                             mlp_num_layers=game_config.MLP_num_hidden_layers,
                             mlp_hidden_dim=game_config.MLP_hidden_dim,
                             mlp_activation=game_config.MLP_activation)
    else:
        raise ValueError(f"illegal game type {game_config.game_type}")

    # 3. Create the message embedder
    if game_config.receiver_embed_model.lower() in ['rnn', 'gru', 'lstm']:
        agent = RnnMessageEmbedder(agent=agent,
                                   discretization_method=disc_method,
                                   vocab_size=game_config.vocab_size,
                                   hidden_size=game_config.receiver_RNN_hidden_dim,
                                   embed_dim=game_config.receiver_RNN_emb_size,
                                   max_len=game_config.max_len,
                                   include_eos_token=game_config.include_eos_token,
                                   reduce_hs=True,
                                   cell=game_config.receiver_embed_model,
                                   num_layers=game_config.receiver_RNN_num_layers)
    else:
        raise ValueError(f"illegal embed model {game_config.embed_model}")
    if game_config.verbose:
        print(f"\n\nmessage embedder:\n{agent}\n\n")
    return agent
