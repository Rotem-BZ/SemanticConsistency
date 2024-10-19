# This file implements the discrete message generators, so that in train mode the message is
# differentiable via backward.

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from egg.core.gs_wrappers import gumbel_softmax_sample


class GenerativeModel(nn.Module):
    """
    A generative model should implement the method next_token_probs(encoder_vec, cached_states, token_embedding).
    This method receives:
    1. encoder_vec: a vector / sequence of vectors of the encoded input
    2. cached_states: Any cached information outputted from the previous call in the current generation loop.
    For example, could contain the previous hidden states.
    3. The embedding of the lastly generated token, for which there is no cached information yet.
    And outputs:
    1. The new hidden state
    2. Cached information to be given to the next call (unless this is the last call in the generation loop).

    The sampling is implemented in this base class. In training mode, the generated embeddings will
    be differentiable (via the chosen discretization method). In inference mode, the sampling will always be one-hot
    encoded.
    """
    def __init__(self,
                 discretization_method: str,
                 vocab_size: int,
                 hidden_size: int,
                 embed_dim: int,
                 max_len: int,
                 temperature: float,
                 trainable_temperature=False,
                 include_eos_token=True,
                 # whether to allow generation of EOS. This increases training time because
                 # Receiver will generate several times.
                 legal_vocab_subsets=None
                 ):
        super().__init__()
        assert discretization_method in ['gs', 'reinforce'], "only GS and Reinforce use a generative model"
        assert discretization_method == 'gs' or legal_vocab_subsets is None, "vocab_subsets is only relevant to GS"
        self.discretization_method = discretization_method
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.max_len = max_len

        if not trainable_temperature:   # only relevant if discretization_method is "gs"
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.include_eos_token = include_eos_token
        self.legal_vocab_subsets = legal_vocab_subsets

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        if discretization_method == 'gs':
            self.embedding = nn.Linear(vocab_size, embed_dim)   # tokens are linear combinations during training
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.reset_parameters()

    def next_token_probs(self, encoder_vec, cache, token_embedding) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError("instantiated the abstract class GenerativeModel!")

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def get_legal_subset_mask(self, logits):
        # only relevant to GS
        if self.legal_vocab_subsets is None:
            return None
        first_token = logits.argmax(dim=1)
        cumsum = torch.cumsum(torch.tensor(self.legal_vocab_subsets, device=first_token.device), 0)
        chosen_subset_idx = (first_token.unsqueeze(1) < cumsum.unsqueeze(0)).long().argmax(dim=1)
        mask_patterns = torch.ones(len(self.legal_vocab_subsets), self.vocab_size, device=first_token.device)
        for row, col in enumerate(cumsum):
            mask_patterns[row, col:] = 0
            mask_patterns[row+1:, :col] = 0
        # if we use EOS, it is always legal
        if self.include_eos_token:
            mask_patterns[:, 0] = 1
        mask = mask_patterns[chosen_subset_idx]
        return mask

    @staticmethod
    def filter_illegal_vocab(logits, mask):
        # we use the straight-through estimator to allow flow of gradients
        if mask is None:
            return logits
        return logits + mask.log()

    def generate_GS(self, encoder_vec):
        cache = None
        legal_subset_mask = None
        e_t = torch.stack([self.sos_embedding] * encoder_vec.size(0))
        sequence = []

        for step in range(self.max_len):
            # call the model
            h_t, cache = self.next_token_probs(encoder_vec, cache, e_t)

            # sample token with GS
            step_logits = self.hidden_to_output(h_t)
            step_logits = self.filter_illegal_vocab(step_logits, legal_subset_mask)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, straight_through=False
            )
            if step == 0:
                legal_subset_mask = self.get_legal_subset_mask(x)

            # update cache, sequence and new embedding
            sequence.append(x)
            e_t = self.embedding(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        if self.include_eos_token:
            eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
            eos[:, 0, 0] = 1
            sequence = torch.cat([sequence, eos], dim=1)

        return sequence

    def generate_reinforce(self, encoder_vec):
        cache = None
        e_t = torch.stack([self.sos_embedding] * encoder_vec.size(0))
        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            # call the model
            h_t, cache = self.next_token_probs(encoder_vec, cache, e_t)

            # sample token
            step_logits = self.hidden_to_output(h_t)
            step_logits = torch.log_softmax(step_logits, dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            # update cache, sequence and new embedding
            e_t = self.embedding(x)
            sequence.append(x)
            # future idea: check if EOS is sampled and break

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

    def forward(self, encoder_vec):
        if self.discretization_method == 'gs':
            return self.generate_GS(encoder_vec)
        elif self.discretization_method == 'reinforce':
            return self.generate_reinforce(encoder_vec)
        else:
            raise ValueError(f"illegal discretization method {self.discretization_method} for a generative model")


class RNNGenerativeModel(GenerativeModel):
    def __init__(self, cell: str = "gru", **kwargs):
        super().__init__(**kwargs)
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

    def next_token_probs(self, encoder_vec, cache, token_embedding) -> tuple[torch.Tensor, Any]:
        """ The cached state is all the previous HS, and also prev_c if the cell is LSTM. The encoder vec is
         used as the first hidden state"""
        if cache is None:
            # first generation - hidden state is the input from encoder
            cache = []
            prev_hidden = encoder_vec
            prev_c = torch.zeros_like(encoder_vec)  # only for LSTM
        else:
            # not the first generation - use cache for the previous hidden state
            if isinstance(cache[-1], tuple):
                prev_hidden, prev_c = cache[-1]
            else:
                prev_hidden = cache[-1]
                prev_c = None

        if isinstance(self.cell, nn.LSTMCell):
            new_hidden, new_c = self.cell(token_embedding, (prev_hidden, prev_c))
            cache.append((new_hidden, new_c))
        else:
            new_hidden = self.cell(token_embedding, prev_hidden)
            cache.append(new_hidden)
        return new_hidden, cache


class GenerativeAgentWrapper(nn.Module):
    """
    combine an encoder agent (which yields a vector or sequence of vectors) with a message generator (RNN).
    """
    def __init__(self, agent, gen_model):
        super().__init__()
        self.agent = agent
        self.gen_model = gen_model

    def forward(self, x, aux_input=None):
        encoder_vec = self.agent(x, aux_input)
        generated_sequence = self.gen_model(encoder_vec)
        return generated_sequence
