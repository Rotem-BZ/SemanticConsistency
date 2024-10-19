# This file implements the message-reading modules, i.e., message embedders.

import torch
import torch.nn as nn


class MessageEmbedder(nn.Module):
    """
    A message embedder should implement the methods:

    1. embed_full_message(embedded_message, lengths) - output a single sequence of embeddings, one for each token.

    2. embed_every_length(embedded_message) - output a list of sequences of embeddings, one sequence
    for each possible message length. This can usually be implemented more efficiently than calling embed_full_message
    several times.

    The agent receives a single embedding in each call if reduce_hs=True, or the entire sequence if reduce_hs=False.

    Note 1: Multiple calls to the agent will only be performed when using GS with include_eos_token=True. Otherwise,
    only one call will be made, corresponding to the full message.
    Note 2: The initial embedding is implemented in this base class.
    """
    def __init__(self,
                 agent,
                 discretization_method: str,
                 vocab_size: int,
                 embed_dim: int,
                 max_len: int,
                 include_eos_token=True,
                 reduce_hs=True    # Single embedding (True) or embedding for each token (False)
                 ):
        super().__init__()
        assert discretization_method in ['gs', 'reinforce'], "only GS and Reinforce use a trainable message embed model"
        self.agent = agent
        self.discretization_method = discretization_method
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.include_eos_token = include_eos_token
        self.reduce_hs = reduce_hs

        if discretization_method == 'reinforce' and not reduce_hs:
            raise NotImplementedError(
                "Reinforce with reduce_hs=False requires batching different lengths, and is not implemented yet"
            )

        if discretization_method == 'gs':
            self.embedding = nn.Linear(vocab_size, embed_dim)   # tokens are linear combinations during training
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

    def embed_full_message(self, embedded_message, lengths):
        raise NotImplementedError("instantiated the abstract class MessageEmbedder!")

    def embed_every_length(self, embedded_message):
        raise NotImplementedError("instantiated the abstract class MessageEmbedder!")

    def _embed_every_length_parallel(self, embedded_message):
        """
        Under the assumptions that token representations aren't effected by increasing message
        length (the representation "doesn't look rightwards"), this implements the function embed_every_length.
        """
        batch_size, seq_len, _ = embedded_message.size()
        lengths = torch.ones(batch_size, dtype=torch.int) * seq_len
        full_message_embedding = self.embed_full_message(embedded_message, lengths)  # [batch_size, max_len, hidden_dim]
        # assert full_message_embedding.size(1) == self.max_len + 1
        # assert full_message_embedding.size(1) == seq_len
        if self.reduce_hs:
            # the input to the agent is the last token embedding
            agent_inputs = [full_message_embedding[:, i, :] for i in range(seq_len)]
        else:
            # the input to the agent contains all token embeddings
            agent_inputs = [full_message_embedding[:, :i+1, :] for i in range(seq_len)]
        return agent_inputs

    def _embed_every_length_sequential(self, embedded_message):
        """
        Implements the function embed_every_length by calling embed_full_message multiple times.
        """
        batch_size, seq_len, _ = embedded_message.size()
        agent_inputs = []
        for i in range(1, seq_len + 1):
            limited_emb = embedded_message[:, :i, :]
            lengths = torch.ones(batch_size, dtype=torch.int) * i
            encoding = self.embed_full_message(limited_emb, lengths)  # [batch_size, i, hidden_dim]
            if self.reduce_hs:
                # the input to the agent is the last token embedding
                agent_input = encoding[:, -1, :]
            else:
                # the input to the agent contains all token embeddings
                agent_input = encoding
            agent_inputs.append(agent_input)
        return agent_inputs

    def forward_reinforce(self, message, lengths, receiver_input=None, aux_input=None):
        emb = self.embedding(message)
        encoding = self.embed_full_message(emb, lengths)
        encoding = encoding[torch.arange(message.size(0)), lengths - 1, :]  # [batch_size, hidden_dim]
        output = self.agent(encoding, receiver_input, aux_input)
        logits = torch.zeros(output.size(0)).to(output.device)
        entropy = logits
        return output, logits, entropy

    def forward_gs(self, message, receiver_input=None, aux_input=None):
        emb = self.embedding(message)   # [batch_size, max_len, hidden_dim]

        outputs = []
        if self.include_eos_token:
            # multiple calls to the agent - one for each possible message length
            encodings = self.embed_every_length(emb)
            for encoding in encodings:
                # future idea: consider combining the encodings into a single tensor and calling the agent once
                output = self.agent(encoding, receiver_input, aux_input)
                outputs.append(output)
        else:
            # single call to the agent, with the entire message
            lengths = torch.ones(message.size(0), dtype=torch.int).to(message.device) * self.max_len
            # The actual message length may be different in inference mode, if include_eos_token=True.
            # Effectively, this means the agent is called with sequences that have probability 0 of being the message.
            # The loss is still calculated correctly since it takes into account this probability.
            encoding = self.embed_full_message(emb, lengths)
            if self.reduce_hs:
                # the input to the agent is the last token embedding
                encoding = encoding[:, -1, :]
            output = self.agent(encoding, receiver_input, aux_input)
            outputs.append(output)
        outputs = torch.stack(outputs).permute(1, 0, 2)
        return outputs

    def forward(self, message, receiver_input=None, aux_input=None, lengths=None):
        if self.discretization_method == 'gs':
            return self.forward_gs(message, receiver_input, aux_input)
        elif self.discretization_method == 'reinforce':
            assert lengths is not None, "lengths must be provided for Reinforce"
            return self.forward_reinforce(message, lengths, receiver_input, aux_input)
        else:
            raise ValueError(f"Unknown discretization method: {self.discretization_method}")


class RnnMessageEmbedder(MessageEmbedder):
    def __init__(self,
                 agent,
                 discretization_method: str,
                 vocab_size: int,
                 hidden_size: int,
                 embed_dim: int,
                 max_len: int,
                 include_eos_token=True,
                 reduce_hs=True,
                 cell: str = "rnn",
                 num_layers: int = 1,
                 ):
        super().__init__(agent, discretization_method, vocab_size, embed_dim, max_len, include_eos_token, reduce_hs)
        cell = cell.lower()
        cell_types = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")
        self.cell = cell_types[cell](
            input_size=embed_dim,
            batch_first=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def embed_full_message(self, embedded_message, lengths):
        """
        Feeds the sequence into an RNN cell and returns the hidden state of the last layer at each timestep.
        """
        # lengths = self.find_lengths(embedded_message)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_message, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.cell(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        return rnn_out

    def embed_every_length(self, embedded_message):
        # RNNs are sequential, so additional tokens don't affect previous hidden states.
        return self._embed_every_length_parallel(embedded_message)