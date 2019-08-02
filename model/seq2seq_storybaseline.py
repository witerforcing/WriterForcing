import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
from io import open
import unicodedata
import string
import re
import random
from torch import optim
from Seq2Seq.Encoder_multi_layer import Encoder
from Seq2Seq.AttentionDecoder_key_multi_layer import Decoder


class Story_model(nn.Module):
    def __init__(self, args, output_field, vocab=None, embed=None):
        super(Story_model, self).__init__()

        self.vocabsize = args.vocabsize
        self.batchsize = args.batchsize
        self.wordembsize = args.wordembsize
        self.numunits = args.numunits
        self.num_layers = args.numlayers
        self.is_train = args.istrain
        self.learning_rate = args.learningrate
        self.learning_rate_decay_factor = args.learning_rate_decay_factor
        self.max_gradient_norm = args.norm
        self.num_samples = args.numsamples
        self.max_length = args.maxlength
        self.use_lstm = args.uselstm
        self.num_directions = 2 if args.bidirectional else 1
        self.device = args.device
        self.vocab = vocab
        self.teacher_forcing_ratio = args.teacherforcingratio
        self.input_vocab_size = args.input_vocab_size
        self.output_vocab_size = args.output_vocab_size
        self.output_field = output_field
        self.pad_idx = self.output_field.vocab.stoi['<pad>']
        self.auto_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.dropout = args.dropout
        self.adaptive_softmax = args.adaptivesoftmax

        self.keyword_attention = args.keyword_attention
        self.lambda_inference = args.lambda_inference
        self.itf_loss = args.itf_loss

        #         self.auto_criterion = nn.CrossEntropyLoss()
        # todo initialize embedding from pretrained vectors
        self.embed = embed
        if embed is None:
            self.embedding = nn.Embedding(self.input_vocab_size, self.wordembsize)
        else:
            print("initializing embeddings from pretrained")
            self.embedding = nn.Embedding.from_pretrained(self.output_field.vocab.vectors)
        #         self.emb_fixed = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embed))
        #         print(self.embedding_matrix_decoder.shape)

        # self.encoder1 = Encoder(self.numunits, self.wordembsize, self.numunits, self.input_vocab_size, self.embed,
        #                         self.device, self.pad_idx, num_layers=self.num_layers).to(self.device).to(self.device)

        # self.encoder1 = Encoder(args.input_vocab_size, self.wordembsize, self.numunits , self.numunits , self.dropout,self.pad_idx, embed=self.output_field.vocab.vectors).to(self.device)
        #
        # self.attn = Attention(self.numunits, self.numunits * self.num_directions).to(self.device)
        #
        # self.decoder = Decoder(args.output_vocab_size, self.wordembsize, self.numunits , self.numunits , self.dropout, self.attn, embed=self.output_field.vocab.vectors).to(self.device)

        self.encoder1 = Encoder(args.input_vocab_size, self.wordembsize, self.numunits, self.numunits, self.dropout,
                                self.pad_idx, self.device, self.embedding, num_layers = self.num_layers).to(self.device)

        self.decoder = Decoder(args.output_vocab_size, self.wordembsize, self.numunits , self.numunits , self.dropout, self.device, self.pad_idx, self.embedding, num_layers = self.num_layers,  att_type = 'bilinear', adaptive_softmax=self.adaptive_softmax).to(self.device)

    def forward(self, src1, keyword_scores1, trg, teacher_forcing_ratio = 1.0):
        eval = not self.is_train
        batch_size = src1.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden, att_mask = self.encoder1(src1)
        # first input to the decoder is the <sos> tokens

        # hidden = torch.zeros_like(hidden).to(self.device)

        outputs, decoder_attentions, generated_sequence, last_hidden_states, decoder_hidden_states, att_mask,loss, summed_attentions, total_coverage_loss =\
         self.decoder(max_len, batch_size, encoder_outputs, keyword_scores1, hidden, trg, trg_vocab_size, teacher_forcing_ratio = teacher_forcing_ratio, att_mask = att_mask, keyword_attention = self.keyword_attention, lambda_inference = self.lambda_inference, itf_loss = self.itf_loss)

        '''output = trg[0, :]
        decoder_hidden_states = torch.zeros(max_len, batch_size, self.decoder.dec_hid_dim).to(self.device)
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs, trg)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
            decoder_hidden_states[t] = hidden

        decoder_hidden_states = decoder_hidden_states.permute(1, 0, 2)
        target_lengths = (trg != self.pad_idx).sum(dim=0) - 1
        masks = torch.arange(max_len).expand(len(target_lengths), max_len).to(self.device) < target_lengths.unsqueeze(1)
        masks = masks.unsqueeze(2)
        masks = masks.type(torch.cuda.FloatTensor)
        decoder_hidden_states[:, :, :] *= masks
        target_lengths = target_lengths.type(torch.cuda.LongTensor)
        last_hidden_states = torch.index_select(decoder_hidden_states, 1, target_lengths)'''

        return outputs, decoder_hidden_states, generated_sequence, loss, loss, summed_attentions, total_coverage_loss
