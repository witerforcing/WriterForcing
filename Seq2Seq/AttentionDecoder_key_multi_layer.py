import torch.nn as nn
import torch
import torch.nn.functional as F
import random


class Decoder(nn.Module):

    def __init__(self,
                 output_dim,
                 emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 dropout,
                 device,
                 pad_idx,
                 embedding,
                 att_type='concat',
                 num_layers=1,
                 adaptive_softmax = None):

        super(Decoder, self).__init__()

        #The num directions in decoder is always 1....

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device
        self.att_type = att_type
        self.num_layers = num_layers
        if self.att_type == 'concat':
            self.attn = nn.Linear((enc_hid_dim ) + dec_hid_dim, dec_hid_dim)
        elif self.att_type == 'bilinear':
            self.attn = nn.Linear((enc_hid_dim) , dec_hid_dim)
        self.context_linear = nn.Linear(enc_hid_dim*2, enc_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        self.pad_idx = pad_idx


        self.embedding = embedding

        self.attn_linear = nn.Linear((enc_hid_dim * 1) + emb_dim, dec_hid_dim)
        self.rnn_layer = nn.GRU((enc_hid_dim * 1), dec_hid_dim, num_layers=self.num_layers)
        self.out = nn.Linear((enc_hid_dim * 1) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        vocab_size = output_dim
        self.softmax_layer = nn.AdaptiveLogSoftmaxWithLoss((enc_hid_dim * 1) + dec_hid_dim + emb_dim, vocab_size,
                                                           cutoffs=[round(vocab_size / 15), 3 * round(vocab_size / 15)])
        self.adaptive_softmax = adaptive_softmax

    def get_context_vector(self, encoder_outputs, alphas):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        weighted = torch.bmm(alphas, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)

        return weighted

    def get_alphas(self, encoder_outputs, hidden_decoder_vector, att_mask= None):

        """
        :param encoder_outputs: (seq_len, batch_size, num_hidden_encoder)
        :param hidden_decoder_vector: (batch_size, hidden_dim_dec)
        :return:
        """
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # (batch_size, source_length, hidden_dim_enc)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        if self.att_type == 'concat':

            hidden_decoder_vector = hidden_decoder_vector.unsqueeze(1).repeat(1, src_len, 1)
            energy = torch.tanh(self.attn(torch.cat((hidden_decoder_vector, encoder_outputs), dim=2)))
            # energy = [batch size, src sent len, dec hid dim]
            energy = energy.permute(0, 2, 1)
            # energy = [batch size, dec hid dim, src sent len]
            # v = [dec hid dim]
            v = self.v.repeat(batch_size, 1).unsqueeze(1)
            # v = [batch size, 1, dec hid dim]
            attention = torch.bmm(v, energy).squeeze(1)

        elif self.att_type == 'bilinear':
            hidden_decoder_vector_bi = hidden_decoder_vector.unsqueeze(2)
            energy_bl = self.attn(encoder_outputs)
            attention = energy_bl.bmm(hidden_decoder_vector_bi).squeeze(2)

        if att_mask is not None:
            attention.data.masked_fill_(att_mask^1, float('-inf'))
        
        return F.softmax(attention, dim=1)

    def unroll(self, input, encoder_outputs, keyword_outputs, hidden,target, att_mask = None, keyword_attention = 0):

        """
                :param input:
                :param encoder_outputs: (seq_len, batch_size, num_hidden_encoder)
                :param hidden: shape of hidden (num_layers*num_dir, batch_size, hidden_dim_dec)
                :return:
                """

        # shape (1, batch_size)
        input = input.unsqueeze(0)

        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))

        # a = (batch size, src len)
        #print("shape of hidden in uroll is: ", hidden.shape)
        hidden_for_alphas = hidden[self.num_layers-1:self.num_layers, :, :].squeeze(0)
        alphas = self.get_alphas(encoder_outputs, hidden_for_alphas, att_mask)

        # a = [batch size, 1, src len]
        alphas = alphas.unsqueeze(1)
        keyword_alphas = keyword_outputs.permute(1,0).unsqueeze(1)

        if keyword_attention == 'alpha_add':
            alphas = alphas.add(keyword_alphas)#alphas#keyword_outputs#alphas.add(keyword_outputs)
            alphas = F.normalize(alphas, p=1, dim=2)

        context_vector = self.get_context_vector(encoder_outputs, alphas)

        if keyword_attention == 'context_add':
            context_vector_keywords = self.get_context_vector(encoder_outputs, keyword_alphas)
            context_cat = torch.cat((context_vector, context_vector_keywords), dim = 2)
            context_vector = self.context_linear(context_cat)

        rnn_input = torch.cat((embedded, context_vector), dim=2)
        rnn_input2 = self.attn_linear(rnn_input)

        #ouput shape (sent_len, batch_size, dec hid dim * n directions)
        #hidden shape (num_layers*num_dir, batch_size, dec_hidden_dim)
        #Note, but since this is a decoder, num_dir would be 1 only.
        output, hidden = self.rnn_layer(rnn_input2, hidden.contiguous())

        #assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context_vector = context_vector.squeeze(0)

        concatenated_vector = torch.cat((output, context_vector, embedded), dim=1)
        if self.adaptive_softmax:
            output, loss = self.softmax_layer(concatenated_vector, target)
        else:
            loss = None
            output = self.out(concatenated_vector)

        return output, hidden, alphas.squeeze(1), loss, concatenated_vector

    def forward(self,
                max_len,
                batch_size,
                encoder_outputs,
                keyword_outputs,
                hidden_encoder,
                trg,
                trg_vocab_size,
                teacher_forcing_ratio=1,
                att_mask=None,
                keyword_attention = 0,
                lambda_inference = 1.0,
                itf_loss = True):

        """

        :param max_len:
        :param batch_size:
        :param encoder_outputs: (seq_len, batch_size, num_hidden_encoder)
        :param hidden_encoder: (num_layer*num_dir, batch_size, num_hidden_encoder)
        :param trg:
        :param trg_vocab_size:
        :param teacher_forcing_ratio:
        :param att_mask:
        :return:
        """

        hidden = hidden_encoder
        #print("input hidden dimension: ", hidden.shape)
        if self.adaptive_softmax:
            outputs = torch.zeros(max_len, batch_size).to(self.device)
            outputs = outputs.type(torch.cuda.LongTensor)
        else:
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        decoder_attentions = []
        generated_sequence = []
        generated_count_dict = list( {} for i in range(batch_size) ) 

        output = trg[0, :]
        decoder_attentions = torch.zeros(max_len, batch_size, encoder_outputs.shape[0]).to(self.device)
        decoder_hidden_states = torch.zeros(max_len, batch_size, self.dec_hid_dim).to(self.device)
        timed_hidden_states = torch.zeros(max_len, self.num_layers, batch_size, self.dec_hid_dim).to(self.device)
        total_loss = torch.tensor(0.0, device=self.device, ) if not self.adaptive_softmax else 0
        loss_per_time_Step = []
        for t in range(1, max_len):
            output_logits, hidden, alphas, loss, concatenated_vector = self.unroll(output, encoder_outputs, keyword_outputs, hidden,trg[t], att_mask=att_mask, keyword_attention = keyword_attention)
            if self.adaptive_softmax:
                total_loss += loss
                outputs[t] = self.softmax_layer.predict(concatenated_vector).type(torch.cuda.LongTensor)
                top1 = outputs[t]
            else:
                outputs[t] = output_logits

                ## heuristics to avoid repetition of words
                if teacher_forcing_ratio == 0 and itf_loss:
                    output_logits = output_logits.clone()
                    output_logits = F.softmax(output_logits, dim = 1)
                    for i in range(batch_size):
                        ## save the probability fo the generated words till now
                        for generated_word in generated_count_dict[i].keys():
                            output_logits[i,generated_word] = output_logits[i,generated_word]/(generated_count_dict[i].get(generated_word, 1))**lambda_inference
                        ## reduce the probability for the last generated word - heuristics for reducing repetitions occuring due to ITF loss                       
                        output_logits[i,output[i]] = output_logits[i,output[i]]/10000

                top1 = output_logits.max(1)[1]

                ## reduce the probability fo the generated words till now, heuristics #2 for reducing repetitions occuring due to ITF loss
                if teacher_forcing_ratio == 0 and itf_loss:
                    for i in range(batch_size):
                        g_word_index = top1[i].item()
                        existing_count = generated_count_dict[i].get(g_word_index, 1)
                        generated_count_dict[i][g_word_index] = existing_count + 1

                converage_vector = decoder_attentions.sum(dim = 0)
                coverage_loss = torch.min(converage_vector, alphas).sum()

            loss_per_time_Step.append(coverage_loss)
            
            decoder_attentions[t] = alphas
            teacher_force = random.random() < teacher_forcing_ratio
            output = (trg[t] if teacher_force else top1)

            generated_sequence.append(output)

            #Since hidden is of size....(num_layer*num_dir, batch_size, hidden_dim_size)
            #but for now, our code is a bit coupled and only works when num_dir=1 for the
            #encoder...this would give the shape of hidden as:
            # (num_layer, batch_size, hidden_dim_size)
            # Also, for calculating the attentions, we are just using the hidden state from the
            # last layer...

            batch_size = hidden.shape[1]
            #(num_layer*num_dir, batch, hidden_dim_size)
            timed_hidden_states[t] = hidden

            #(batch, num_dir, hidden_dim_size), taking last layer output
            hidden_to_store = hidden[self.num_layers-1:self.num_layers, :, :].permute(1, 0, 2).contiguous()

            #(batch_size, hidden_dim_size*num_dir)
            hidden_to_store = hidden_to_store.contiguous().view(batch_size, -1)
            #print("shape of hidden to store", hidden_to_store.shape)

            decoder_hidden_states[t] = hidden_to_store

#         decoder_hidden_states = decoder_hidden_states.permute(1, 0, 2)
        target_lengths = (trg != self.pad_idx).sum(dim=0)
        masks = torch.arange(max_len).expand(len(target_lengths), max_len).to(self.device) < target_lengths.unsqueeze(1)
        masks_forward = torch.arange(max_len).expand(len(target_lengths), max_len).to(self.device) < target_lengths.unsqueeze(1)
        #batch_size * seq_len * 1
        masks = masks.permute(1, 0).unsqueeze(2)
        masks = masks.type(torch.cuda.FloatTensor)
        decoder_hidden_states[:, :, :] *= masks
        target_lengths = target_lengths.type(torch.cuda.LongTensor)
#         last_hidden_states = decoder_hidden_states[torch.arange(batch_size),target_lengths-1, :]
        
        # (num_layers, batch_size, hidden_dec_dim)
        timed_hidden_states = timed_hidden_states.permute(2, 0, 1, 3)
        # batch_size * num_layers * hidden_dec_dim
        last_hidden_states = timed_hidden_states[torch.arange(batch_size), target_lengths-1, :, :]
        last_hidden_states = last_hidden_states.permute(1, 0, 2)
        # return outputs, decoder_attentions, generated_sequence, decoder_hidden_states, last_hidden_states, masks_forward, total_loss/max_len
        summed_attentions = torch.sum(decoder_attentions, dim = 0)
        loss_per_time_Step = torch.tensor([loss_per_time_Step], device=self.device).sum()/(batch_size * max_len)
        return outputs, decoder_attentions, generated_sequence, decoder_hidden_states, last_hidden_states, masks_forward, total_loss/max_len, summed_attentions, loss_per_time_Step


