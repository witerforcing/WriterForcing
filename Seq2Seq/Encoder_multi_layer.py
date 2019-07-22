from io import open
import unicodedata
import string
import re
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Encoder(nn.Module):
    def __init__(self,
                 input_vocab_size,
                 emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 dropout,
                 pad_idx,
                 device,
                 embedding,
                 num_layers=1,
                 bidirectional=False):

        super(Encoder, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.num_directions = 2 if bidirectional is True else 1
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = embedding
        self.rnn_layer = nn.GRU(emb_dim, enc_hid_dim, num_layers=self.num_layers, bidirectional=self.bidirectional)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src shape (src_sent_len, batch_size)
        src_len = (src != self.pad_idx).sum(dim=0)

        # embedded tokens of shape (src_sent_len, batch_size, emb_dim)
        embedded = self.dropout(self.embedding(src))

        sorted_indices = sorted(range(len(src_len)), key=lambda i: src_len[i], reverse=True)
        sorted_lengths = [src_len[i] for i in sorted_indices]
        unsort_indices = sorted(range(len(sorted_indices)), key=lambda i: sorted_indices[i])
        embedded = embedded[:, sorted_indices]

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths)

        # Pass packed data through RNN layer, and unpack
        # Shape of output (seq_len, batch_size, num_directions * hidden_size)
        # Shape of hidden unit (num_dir*num_layer, batch_size, hidden_size)
        outputs, hidden = self.rnn_layer(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # Shape of output (seq_len, batch_size, num_directions * hidden_size)
        outputs = outputs[:, unsort_indices, :]
        hidden = hidden[:, unsort_indices, :]

        max_len = src.shape[0]

        #shape (batch, src_length)
        mask = torch.arange(max_len).expand(len(src_len), max_len).to(self.device) < src_len.unsqueeze(1)

        return outputs, hidden, mask