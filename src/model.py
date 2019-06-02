import random, sys, pdb

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from gensim import models
import numpy as np

from src.components.encoder import EncoderRNN
from src.components.decoder import DecoderRNN
from src.components.attention import LuongAttnDecoderRNN
from src.components.beamsearch import Hypothesis

from submodopt.submodopt import SubmodularOpt

class s2s(nn.Module):
    def __init__(self, config, voc, device, logger, EOS_tag='EOS', SOS_tag='SOS'):
        super(s2s, self).__init__()

        self.config         = config
        self.device         = device
        self.voc            = voc
        self.EOS_tag        = EOS_tag
        self.SOS_tag        = SOS_tag
        self.SOS_token      = self.voc.w2id['SOS']
        self.EOS_token      = self.voc.w2id['EOS']
        self.logger         = logger

        if self.config.use_word2vec:
            self.embedding  = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze=not self.config.train_word2vec)
        else:
            self.embedding  = nn.Embedding(self.voc.nwords, self.config.emb_size)
            # Based on AWD-LSTM
            nn.init.uniform_(self.embedding.weight, -1*self.config.init_range, self.config.init_range)

        # Fill params for encoder and decoder
        self.logger.info('Building Encoder RNN..')
        self.encoder        = EncoderRNN(self.config.hidden_size,
                                         self.embedding,
                                         self.config.cell_type,
                                         self.config.depth,
                                         self.config.s2sdprate,
                                         self.config.bidirectional).to(device)
        self.logger.info('Encoder RNN built')

        self.logger.info('Building Decoder RNN..')
        if self.config.use_attn:
            self.decoder    = LuongAttnDecoderRNN(self.config.attn_type,
                                                  self.embedding,
                                                  self.config.cell_type,
                                                  self.config.hidden_size,
                                                  self.voc.nwords,
                                                  self.config.depth,
                                                  self.config.s2sdprate).to(device)
        else:
            self.decoder    = DecoderRNN(self.embedding,
                                         self.config.cell_type,
                                         self.config.hidden_size,
                                         self.voc.nwords,
                                         self.config.depth,
                                         self.config.s2sdprate).to(device)
        self.logger.info('Decoder RNN built')

        self._optim()

        # Specify criterion
        self.criterion = nn.NLLLoss()

    def _form_embeddings(self, file_path):
        weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
        weight_req  = torch.randn(self.voc.nwords, self.config.emb_size)
        for key, value in self.voc.id2w.items():
            if value in weights_all:
                weight_req[key] = torch.FloatTensor(weights_all[value])

        return weight_req

    def _optim(self):
        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        if self.config.opt     == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.config.lr)
        elif self.config.opt   == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.config.lr)
        elif self.config.opt   == 'asgd':
            self.optimizer = optim.ASGD(self.params, lr=self.config.lr)
        else:
            self.optimizer = optim.SGD(self.params, lr=self.config.lr)

    def trainer(self, src_tens, src_len, tgt_tens, tgt_len):
        self.optimizer.zero_grad()
        encoder_outputs, encoder_hidden = self.encoder(src_tens, src_len)

        self.loss       = 0

        decoder_input   = torch.tensor([self.SOS_token for i in range(src_tens.size(1))], device=self.device)
        if self.config.cell_type == 'lstm':
            decoder_hidden = []
            decoder_hidden.append(encoder_hidden[0][:self.decoder.nlayers])
            decoder_hidden.append(encoder_hidden[1][:self.decoder.nlayers])
            decoder_hidden = (decoder_hidden[0], decoder_hidden[1])
        else:
            decoder_hidden  = encoder_hidden[:self.decoder.nlayers]

        use_teacher_forcing = True if random.random() < self.config.tfr else False
        target_length = max(tgt_len)

        if use_teacher_forcing:
            for di in range(target_length):
                if self.config.use_attn:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                self.loss += self.criterion(decoder_output, tgt_tens[di])
                decoder_input = tgt_tens[di]
        else:
            for di in range(target_length):
                if self.config.use_attn:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                topv, topi = decoder_output.topk(1)
                self.loss += self.criterion(decoder_output, tgt_tens[di])
                decoder_input=topi.squeeze().detach()

        self.loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)
        self.optimizer.step()

        return self.loss.item()/target_length

    # Do this carefully when checking - PER SENTENCE EVALUATION
    def greedy_decode(self, src_tens, src_len, tgt_tens=None, tgt_len=None, validation=False):
        with torch.no_grad():
            encoder_outputs = torch.zeros(self.config.max_length, self.encoder.hidden_size)
            encoder_outputs, encoder_hidden = self.encoder(src_tens, src_len)

            loss = 0.0
            decoder_input       = torch.tensor([self.SOS_token for i in range(src_tens.size(1))], device=self.device)

            if self.config.cell_type == 'lstm':
                decoder_hidden = []
                decoder_hidden.append(encoder_hidden[0][:self.decoder.nlayers])
                decoder_hidden.append(encoder_hidden[1][:self.decoder.nlayers])
                decoder_hidden = (decoder_hidden[0], decoder_hidden[1])
            else:
                decoder_hidden  = encoder_hidden[:self.decoder.nlayers]

            # Change from here. Do this after training process is working well
            decoded_words       = [[] for i in range(src_tens.size(1))]
            decoder_attentions  = []

            if validation:
                target_length = max(tgt_len)
            else:
                target_length = self.config.max_length

            for di in range(target_length):
                if self.config.use_attn:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions.append(decoder_attention.data)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                if validation:
                    loss += self.criterion(decoder_output, tgt_tens[di])
                topv, topi  = decoder_output.data.topk(1)
                for i in range(src_tens.size(1)):
                    if topi[i].item() == self.EOS_token:
                        continue
                    decoded_words[i].append(self.voc.id2w[topi[i].item()])
                decoder_input = topi.squeeze().detach()

            if validation:
                if self.config.use_attn:
                    return loss/target_length, decoded_words, decoder_attentions[:di + 1]
                else:
                    return loss/target_length, decoded_words, None
            else:
                return decoded_words



    def beam_decode(self, src_sents, src_tens, src_lens,  beam_width):
        with torch.no_grad():
            batch_size = src_tens.size(1)
            batch_beam_sents = []
            get_top = 20

            src_sents = [' '.join(sent) for sent in src_sents]

            for b in range(batch_size):
                source_sent = src_sents[b]

                src_ten = torch.cat([src_tens[:, b].unsqueeze(1).contiguous() for _ in range(beam_width)], dim=1)
                src_len = [src_lens[b] for _ in range(beam_width)]
                encoder_output = torch.zeros(self.config.max_length, self.encoder.hidden_size)
                encoder_output, encoder_hidden = self.encoder(src_ten, src_len)

                decoder_input               = torch.tensor([[self.SOS_token] for _ in range(beam_width)], device=self.device)
                if self.config.cell_type == 'lstm':
                    decoder_hidden  = []
                    decoder_hidden.append(encoder_hidden[0][:self.decoder.nlayers])
                    decoder_hidden.append(encoder_hidden[1][:self.decoder.nlayers])
                    decoder_hidden  = (decoder_hidden[0], decoder_hidden[1])
                    beam_decoder_hidden  = [(decoder_hidden[0][:, i, :].unsqueeze(1).contiguous(), decoder_hidden[1][:, i, :].unsqueeze(1).contiguous()) for i in range(beam_width)]
                else:
                    decoder_hidden  = encoder_hidden[:self.decoder.nlayers]
                    beam_decoder_hidden = torch.cat([decoder_hidden[:, i, :].unsqueeze(1).unsqueeze(0).contiguous() for i in range(beam_width)], dim=0)

                hyps = [Hypothesis(tokens=[decoder_input[i].item()],
                                   log_probs=[0.0],
                                   state=beam_decoder_hidden[i])
                        for i in range(beam_width)]
                results = []
                steps = 0

                while steps < 2 * self.config.max_length and len(results) < beam_width:
                    latest_tokens = torch.tensor([[h.latest_token] for h in hyps], device=self.device)
                    states = [h.state for h in hyps]

                    decoder_output = []
                    new_states = []
                    topvs = []
                    topis = []

                    for i in range(len(hyps)):
                        dec_out, new_state,  _ = self.decoder(latest_tokens[i], states[i], encoder_output[:, i, :].unsqueeze(1).contiguous())
                        topv, topi = dec_out.topk(get_top)
                        decoder_output.append(dec_out)
                        new_states.append(new_state)
                        topvs.append(topv)
                        topis.append(topi)

                    all_hyps = []
                    num_orig_hyps = 1 if steps == 0 else len(hyps)
                    for i in range(num_orig_hyps):
                        h, new_state = hyps[i], new_states[i]
                        for j in range(get_top):
                            new_hyp = h.extend(token = topis[i][0, j].item(),
                                               log_prob = topvs[i][0,j].item(),
                                               state = new_state)
                            all_hyps.append(new_hyp)

                    hyps = []
                    for h in self._sort_hyps(all_hyps):
                        if h.latest_token == self.EOS_token:
                            if steps >= 5:
                                results.append(h)
                        else:
                            hyps.append(h)
                        if len(hyps) == beam_width or len(results) == beam_width:
                            break

                    steps += 1

                if len(results) < beam_width:
                    results = hyps

                hyps_sorted = self._sort_hyps(results)

                beam_sent = []
                for m in range(len(hyps_sorted)):
                    beam_sent.append([self.voc.id2w[to] for to in hyps_sorted[m].tokens[1:-1]])

                for k in range(len(beam_sent)):
                    beam_sent[k] = ' '.join(beam_sent[k])


                batch_beam_sents.append(beam_sent)

        return batch_beam_sents


    def beam_decode_sub(self, src_sents, src_tens, src_lens,    beam_width, method = 'dpp', slam=0.5, sparam =[1.0,1.0,1.0,1.0] , outer_width=50, inner_width=20):
        with torch.no_grad():
            batch_size = src_tens.size(1)
            batch_beam_sents = []
            inner_width = beam_width
            get_top = beam_width
            beam_width= outer_width
            src_sents = [' '.join(sent) for sent in src_sents]

            for b in range(batch_size):
                source_sent = src_sents[b]

                src_ten = torch.cat([src_tens[:, b].unsqueeze(1).contiguous() for _ in range(beam_width)], dim=1)
                src_len = [src_lens[b] for _ in range(beam_width)]
                encoder_output = torch.zeros(self.config.max_length, self.encoder.hidden_size)
                encoder_output, encoder_hidden = self.encoder(src_ten, src_len)

                decoder_input               = torch.tensor([[self.SOS_token] for _ in range(beam_width)], device=self.device)
                if self.config.cell_type == 'lstm':
                    decoder_hidden  = []
                    decoder_hidden.append(encoder_hidden[0][:self.decoder.nlayers])
                    decoder_hidden.append(encoder_hidden[1][:self.decoder.nlayers])
                    decoder_hidden  = (decoder_hidden[0], decoder_hidden[1])
                    beam_decoder_hidden  = [(decoder_hidden[0][:, i, :].unsqueeze(1).contiguous(), decoder_hidden[1][:, i, :].unsqueeze(1).contiguous()) for i in range(beam_width)]
                else:
                    decoder_hidden  = encoder_hidden[:self.decoder.nlayers]
                    beam_decoder_hidden = torch.cat([decoder_hidden[:, i, :].unsqueeze(1).unsqueeze(0).contiguous() for i in range(beam_width)], dim=0)

                hyps = [Hypothesis(tokens=[decoder_input[i].item()],
                                   log_probs=[0.0],
                                   state=beam_decoder_hidden[i])
                        for i in range(beam_width)]
                results = []
                steps = 0

                while steps < 2*self.config.max_length and len(results) < beam_width:
                    latest_tokens = torch.tensor([[h.latest_token] for h in hyps], device=self.device)
                    states = [h.state for h in hyps]

                    decoder_output = []
                    new_states = []
                    topvs = []
                    topis = []
                    for i in range(len(hyps)):
                        dec_out, new_state,  _ = self.decoder(latest_tokens[i], states[i], encoder_output[:, i, :].unsqueeze(1).contiguous())
                        topv, topi = dec_out.topk(get_top)
                        decoder_output.append(dec_out)
                        new_states.append(new_state)
                        topvs.append(topv)
                        topis.append(topi)

                    all_hyps = []
                    num_orig_hyps = 1 if steps == 0 else len(hyps)
                    for i in range(num_orig_hyps):
                        h, new_state = hyps[i], new_states[i]
                        for j in range(get_top):
                            new_hyp = h.extend(token = topis[i][0, j].item(),
                                               log_prob = topvs[i][0,j].item(),
                                               state = new_state)
                            all_hyps.append(new_hyp)

                    hyps = []
                    for h in self._sort_hyps(all_hyps):
                        if h.latest_token == self.EOS_token:
                            if steps >= 5:
                                results.append(h)
                        else:
                            hyps.append(h)
                        if len(hyps) == beam_width or len(results) == beam_width:
                            break

                    steps += 1

                if len(results) <  beam_width:
                    results = hyps

                hyps_sorted = self._sort_hyps(results)

                beam_sent = []
                for m in range(len(hyps_sorted)):
                    beam_sent.append([self.voc.id2w[to] for to in hyps_sorted[m].tokens[1:-1]])

                for k in range(len(beam_sent)):
                    beam_sent[k] = ' '.join(beam_sent[k])

                final_seqs = np.array(beam_sent)
                if method == 'submod':
                    try:
                        subopt= SubmodularOpt(final_seqs, source_sent)
                        subopt.initialize_function(slam, a1=sparam[0], a2=sparam[1], b1=sparam[2], b2= sparam[3])
                        selec_sents= subopt.maximize_func(inner_width)
                        final_seqs = list(selec_sents)
                    except Exception as e:
                        print('Error in Submod: {}'.format(e))
                        final_seqs = list(final_seqs[:inner_width])
                else:
                    selected_ids = np.random.choice(inner_width, outer_width, replace=False)
                    final_seqs = list(final_seqs[selected_ids])

                beam_sents = final_seqs

                batch_beam_sents.append(beam_sents)
        return batch_beam_sents


    def beam_decode_sub_timestep(self, src_sents, src_tens, src_lens,  beam_width, method = 'dpp', slam=0.5, outer_width=50, inner_width=20):
        with torch.no_grad():
            batch_size = src_tens.size(1)
            batch_beam_sents = []
            inner_width = beam_width
            get_top = beam_width
            beam_width= outer_width
            src_sents = [' '.join(sent) for sent in src_sents]


            for b in range(batch_size):
                source_sent = src_sents[b]

                src_ten = torch.cat([src_tens[:, b].unsqueeze(1).contiguous() for _ in range(beam_width)], dim=1)
                src_len = [src_lens[b] for _ in range(beam_width)]
                encoder_output = torch.zeros(self.config.max_length, self.encoder.hidden_size)
                encoder_output, encoder_hidden = self.encoder(src_ten, src_len)

                decoder_input               = torch.tensor([[self.SOS_token] for _ in range(beam_width)], device=self.device)
                if self.config.cell_type == 'lstm':
                    decoder_hidden  = []
                    decoder_hidden.append(encoder_hidden[0][:self.decoder.nlayers])
                    decoder_hidden.append(encoder_hidden[1][:self.decoder.nlayers])
                    decoder_hidden  = (decoder_hidden[0], decoder_hidden[1])
                    beam_decoder_hidden  = [(decoder_hidden[0][:, i, :].unsqueeze(1).contiguous(), decoder_hidden[1][:, i, :].unsqueeze(1).contiguous()) for i in range(beam_width)]
                else:
                    decoder_hidden  = encoder_hidden[:self.decoder.nlayers]
                    beam_decoder_hidden = torch.cat([decoder_hidden[:, i, :].unsqueeze(1).unsqueeze(0).contiguous() for i in range(beam_width)], dim=0)

                hyps = [Hypothesis(tokens=[decoder_input[i].item()],
                                   log_probs=[0.0],
                                   state=beam_decoder_hidden[i])
                        for i in range(beam_width)]
                results = []
                steps = 0

                while steps < 2* self.config.max_length and len(results) < beam_width:
                    latest_tokens = torch.tensor([[h.latest_token] for h in hyps], device=self.device)
                    states = [h.state for h in hyps]

                    decoder_output = []
                    new_states = []
                    topvs = []
                    topis = []
                    for i in range(len(hyps)):
                        dec_out, new_state,  _ = self.decoder(latest_tokens[i], states[i], encoder_output[:, i, :].unsqueeze(1).contiguous())
                        topv, topi = dec_out.topk(get_top)
                        decoder_output.append(dec_out)
                        new_states.append(new_state)
                        topvs.append(topv)
                        topis.append(topi)

                    all_hyps = []
                    num_orig_hyps = 1 if steps == 0 else len(hyps)
                    for i in range(num_orig_hyps):
                        h, new_state = hyps[i], new_states[i]
                        for j in range(get_top):
                            new_hyp = h.extend(token = topis[i][0, j].item(),
                                               log_prob = topvs[i][0,j].item(),
                                               state = new_state)
                            all_hyps.append(new_hyp)

                    all_hyps_final = []
                    sents_hyps = []
                    for h in self._sort_hyps(all_hyps)[:outer_width]:
                        sents_hyps.append(' '.join([self.voc.id2w[to] for to in h.tokens[1:]]))
                        all_hyps_final.append(h)

                    final_hyps_seqs = np.array(sents_hyps)

                    if method == 'submod':
                        try:
                            subopt= SubmodularOpt(final_seqs, source_sent)
                            subopt.initialize_function(slam)
                            selec_sents= subopt.maximize_func(inner_width)
                            final_seqs = list(selec_sents)
                        except:
                            print('Error in Submod')
                            final_seqs = list(final_seqs[:2*inner_width])
                    else:
                        selected_ids = np.random.choice(outer_width, inner_width, replace=False)
                        final_seqs = list(final_seqs[selected_ids])
                        all_hyps_final = all_hyps_final[selected_ids]



                    hyps = []
                    for h in all_hyps_final:
                        if h.latest_token == self.EOS_token:
                            if steps >= 5:
                                results.append(h)
                        else:
                            hyps.append(h)
                        if len(hyps) == beam_width or len(results) == beam_width:
                            break


                    steps += 1

                if len(results) <  beam_width:
                    results = hyps

                hyps_sorted = self._sort_hyps(results)

                beam_sent = []
                for m in range(len(hyps_sorted)):
                    beam_sent.append([self.voc.id2w[to] for to in hyps_sorted[m].tokens[1:-1]])

                for k in range(len(beam_sent)):
                    beam_sent[k] = ' '.join(beam_sent[k])

                batch_beam_sents.append(beam_sents)

            return batch_beam_sents

    def _sort_hyps(self, hyps):
        return sorted(hyps, key=lambda h:h.avg_log_prob, reverse=True)


    # The following code is for diverse beam search decoding scheme.
    def hamming(self, s1, s2):
        s1_split = s1.split()
        s2_split = s2.split()
        if len(s1_split) != len(s2_split):
            return 0

        dist = 0
        for i in range(len(s1_split)):
            if s1_split[i] != s2_split[i]:
                dist +=1

            return dist/len(s1_split)

    def dissimilarity(self, hyps, prev_hyps, type_dis='hamming'):
        m = 0
        seq = ' '.join([str(t) for t in hyps.tokens])

        for ph in prev_hyps:
            s = ' '.join([str(t) for t in ph.tokens])
            if type_dis == 'hamming':
                m += self.hamming(s, seq)
            return m/len(prev_hyps)

    def div_beam_decode(self, src_tens, src_lens, beam_width=20, groups=5, lam=0.2):
        assert beam_width % groups == 0

        bprime = int(beam_width/groups)
        with torch.no_grad():
            batch_size = src_tens.size(1)
            batch_beam_sents = []
            get_top = beam_width

            for b in range(batch_size):
                src_ten = torch.cat([src_tens[:, b].unsqueeze(1).contiguous() for _ in range(beam_width)], dim=1)
                src_len = [src_lens[b] for _ in range(beam_width)]
                encoder_output = torch.zeros(self.config.max_length, self.encoder.hidden_size)
                encoder_output, encoder_hidden = self.encoder(src_ten, src_len)

                decoder_input               = torch.tensor([[self.SOS_token] for _ in range(beam_width)], device=self.device)
                if self.config.cell_type == 'lstm':
                    decoder_hidden  = []
                    decoder_hidden.append(encoder_hidden[0][:self.decoder.nlayers])
                    decoder_hidden.append(encoder_hidden[1][:self.decoder.nlayers])
                    decoder_hidden  = (decoder_hidden[0], decoder_hidden[1])
                    beam_decoder_hidden  = [(decoder_hidden[0][:, i, :].unsqueeze(1).contiguous(), decoder_hidden[1][:, i, :].unsqueeze(1).contiguous()) for i in range(beam_width)]
                else:
                    decoder_hidden  = encoder_hidden[:self.decoder.nlayers]
                    beam_decoder_hidden = torch.cat([decoder_hidden[:, i, :].unsqueeze(1).unsqueeze(0).contiguous() for i in range(beam_width)], dim=0)

                    hyps = [Hypothesis(tokens=[decoder_input[i].item()],
                                           log_probs=[0.0],
                                           state=beam_decoder_hidden[i])
                            for i in range(beam_width)]
                    results = []
                    steps = 0

                    while steps < self.config.max_length*2 and len(results) < beam_width:
                        latest_tokens = torch.tensor([[h.latest_token] for h in hyps], device=self.device)
                        states = [h.state for h in hyps]

                        decoder_output = []
                        new_states = []
                        topvs = []
                        topis = []
                        for i in range(len(hyps)):
                            dec_out, new_state,  _ = self.decoder(latest_tokens[i], states[i], encoder_output[:, i, :].unsqueeze(1).contiguous())
                            topv, topi = dec_out.topk(get_top)
                            decoder_output.append(dec_out)
                            new_states.append(new_state)
                            topvs.append(topv)
                            topis.append(topi)

                            all_hyps = []
                            num_orig_hyps = 1 if steps == 0 else len(hyps)
                            for i in range(num_orig_hyps):
                                h, new_state = hyps[i], new_states[i]
                                for j in range(get_top):
                                    new_hyp = h.extend(token = topis[i][0, j].item(),
                                                       log_prob = topvs[i][0,j].item(),
                                                       state = new_state)
                                    all_hyps.append(new_hyp)

                                    sorted_hyps = self._sort_hyps(all_hyps)

                                    group1 = sorted_hyps[:bprime]
                                    group_rem = sorted_hyps[bprime:]

                                    gp1 = []
                                    for seqg in group1:
                                        if seqg.latest_token == self.EOS_token:
                                            if steps>=5:
                                                results.append(seqg)
                                            else:
                                                gp1.append(seqg)

                                    if len(results) == beam_width:
                                        break

                                    bcount = len(gp1)
                                    b_t = 0
                                    for i in range(len(group_rem)):
                                        if bcount == bprime:
                                            break
                                        if group_rem[i].latest_token == self.EOS_token:
                                            if steps>=5:
                                                results.append(group_rem[i])
                                            else:
                                                gp1.append(group_rem[i])
                                                bcount += 1
                                            b_t +=1

                                    if len(results) == beam_width:
                                        break

                                    group_rem = group_rem[b_t:]
                                    sorted_hyps = gp1 + group_rem

                                    m = bprime
                                    k = 1
                                    prev_gps = gp1

                                    flag = 0
                                    for gb in range(1, groups):
                                        for m in range(len(sorted_hyps)-gb*bprime):
                                            group_rem[m].log_probs[-1] = group_rem[m].log_probs[-1] + lam*self.dissimilarity(group_rem[m], prev_gps)

                                            group_rem = self._sort_hyps(group_rem)

                                            selec_group = group_rem[:bprime]
                                            group_rem = group_rem[bprime:]

                                            gp1 = []
                                            for seqg in selec_group:
                                                if seqg.latest_token == self.EOS_token:
                                                    if steps>=5:
                                                        results.append(seqg)
                                                    else:
                                                        gp1.append(seqg)

                                            if len(results) == beam_width:
                                                flag = 1
                                                break

                                            bcount = len(gp1)
                                            b_t = 0
                                            for i in range(len(group_rem)):
                                                if bcount == bprime:
                                                    break
                                                if group_rem[i].latest_token == self.EOS_token:
                                                    if steps>=5:
                                                        results.append(group_rem[i])
                                                    else:
                                                        gp1.append(group_rem[i])
                                                        bcount += 1
                                                    b_t +=1

                                            if len(results) == beam_width:
                                                flag = 1
                                                break

                                            group_rem = group_rem[b_t:]
                                            sorted_hyps = gp1 + group_rem
                                            prev_gps.extend(gp1)

                                    steps += 1
                                    if flag == 1:
                                        break

                                    hyps = prev_gps

                            if len(results) < beam_width:
                                results = hyps

                            hyps_sorted = self._sort_hyps(results)

                            beam_sent = []
                            for m in range(len(hyps_sorted)):
                                beam_sent.append([self.voc.id2w[to] for to in hyps_sorted[m].tokens[1:-1]])

                            batch_beam_sents.append(beam_sent)

        return batch_beam_sents
