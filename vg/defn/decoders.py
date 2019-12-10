import random

import torch
import torch.nn as nn
import onion.util as util
import torch.nn.functional as F
import numpy as np


class BilinearAttention(nn.Module):
    """
    Soft alignment between two sequences, aka attention. Based on
    https://arxiv.org/abs/1508.04025v5.
    """

    def __init__(self, size_in):
        super(BilinearAttention, self).__init__()
        util.autoassign(locals())
        self.W = nn.Linear(self.size_in, self.size_in)

    def forward(self, g, h):
        alpha = F.softmax(g.bmm(self.W(h).permute(0, 2, 1)), dim=1)  # FIXME is dim=1 correct?
        # Alignment based on bi-linear scores between g (source) and h (target)
        context = alpha.permute(0, 2, 1).bmm(g)
        # print(context.size(), h.size())
        return context


class SimpleDecoder(nn.Module):
    """Simple decoder."""
    def __init__(self, size_feature, size, size_embed=64, depth=1):
        super(SimpleDecoder, self).__init__()
        util.autoassign(locals())
        self.Embed = nn.Embedding(self.size_feature, self.size_embed)  # Why not share embeddings with encoder?
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1,
                                                      self.size))
        self.RNN = nn.GRU(self.size_embed, self.size, self.depth,
                          batch_first=True)

    def forward(self, prev, rep):
        # FIXME: previous symbol or previous h?
        R = rep.expand(self.depth, rep.size(0), rep.size(1))
        out, last = self.RNN(self.Embed(prev), R)
        return out, last


# Let's look at this post
# https://stackoverflow.com/questions/50571991/implementing-luong-attention-in-pytorch

# Original Matlab code by Luong is here: https://github.com/lmthang/nmt.hybrid

class DecoderWithAttn(nn.Module):
    def __init__(self, hidden_size, output_size, size_embed=64, depth=1):
        super(DecoderWithAttn, self).__init__()
        util.autoassign(locals())
        self.Decoder = SimpleDecoder(self.output_size, self.hidden_size,
                                     size_embed=self.size_embed,
                                     depth=self.depth)
        self.BAttn = BilinearAttention(self.hidden_size)
        self.Proj = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, states, rep, prev):
        # Encoder returns the hidden state per time step, states, and a global
        # representation, rep
        # Decoder decodes conditioned on rep, and on the symbol decoded at
        # previous time step, prev
        # FIXME: previous symbol or previous h?
        h, _last = self.Decoder(prev, rep)
        # Bilinear attention generates a weighted sum of the source hidden
        # states (context) for each time
        # step of the target
        context = self.BAttn(states, h)
        # The context is concatenated with target hidden states
        h_context = torch.cat((context, h), dim=2)
        # The target symbols are generated conditioned on the concatenation of
        # target states and context
        # FIXME: linear projection?
        pred = self.Proj(h_context)
        #print(pred.size())
        return pred


# Conditioned on source and on target at t-1
class CondDecoder(nn.Module):
    def __init__(self, size_feature, size, depth=1):
        super(CondDecoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1,
                                                      self.size))
        self.RNN = nn.GRU(self.size_feature, self.size, self.depth,
                          batch_first=True)
        self.Proj = nn.Linear(self.size, self.size_feature)

    def forward(self, rep, prev):
        rep = rep.expand(self.depth, rep.size(0), rep.size(1))
        out, last = self.RNN(prev, rep)
        pred = self.Proj(out)
        return pred


# Conditioned only on source
class UncondDecoder(nn.Module):
    def __init__(self, size_feature, size, depth=1):
        super(UncondDecoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1,
                                                      self.size))
        self.RNN = nn.GRU(self.size, self.size, self.depth, batch_first=True)
        self.Proj = nn.Linear(self.size, self.size_feature)

    def forward(self, rep, target):
        R = rep.unsqueeze(1).expand(-1, target.size(1), -1).cuda()
        H0 = self.h0.expand(self.depth, target.size(0), self.size).cuda()
        out, last = self.RNN(R, H0)
        pred = self.Proj(out)
        return pred


class BahdanauAttn(nn.Module):
    '''
    Attention mechanism following Bahdanau et al. (2015)
    [https://arxiv.org/abs/1409.0473]
    '''
    def __init__(self, hidden_size, use_cuda=True, bidirectional_enc=False):
        super(BahdanauAttn, self).__init__()

        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        if bidirectional_enc:
            self.U_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        else:
            self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        dims = encoder_outputs.shape[:2]

        # Create variable to store attention energies
        attn_energies = torch.zeros(dims)  # B x S
        if self.use_cuda:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        hidden = hidden.permute(1, 0, 2)  # B x S x N
        attn_energies = self.W_a(hidden) + self.U_a(encoder_outputs)
        attn_energies = torch.tanh(attn_energies)
        attn_energies = self.v_a(attn_energies)

        # Normalize energies to weights in range 0 to 1,
        return F.softmax(attn_energies, dim=1)


class BahdanauAttnDecoderRNN(nn.Module):
    '''
    Attention decoder, following Bahdanau et al. (2015)
    [https://arxiv.org/abs/1409.0473]
    Borrowed from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
    '''
    def __init__(self, hidden_size, output_size, mapper, max_output_length,
                 depth=1, dropout_p=0.0, teacher_forcing_ratio=1.0,
                 use_cuda=True, bidirectional_enc=False):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.dropout_p = dropout_p
        self.max_output_length = max_output_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_cuda = use_cuda
        self.bidirectional_enc = bidirectional_enc
        self.mapper = mapper

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = BahdanauAttn(hidden_size, use_cuda=use_cuda,
                                 bidirectional_enc=bidirectional_enc)
        if bidirectional_enc:
            mult = 3
        else:
            mult = 2
        self.gru = nn.GRU(hidden_size * mult, hidden_size, depth,
                          dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size * mult, output_size)
        self.sos = torch.LongTensor([[mapper.BEG_ID]])
        self.h0 = torch.zeros([1, 1, hidden_size])
        if use_cuda:
            self.sos = self.sos.cuda()
            self.h0 = self.h0.cuda()

    def forward(self, input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time
        # step, but will use all encoder outputs

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(input)
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden, encoder_outputs)
        context = attn_weights * encoder_outputs
        context = context.sum(dim=1)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context[:, None, :]), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = self.out(torch.cat((output, context[:, None, :]), 2))
        output = F.log_softmax(output, dim=2)

        # Return final output, hidden state, and attention weights (for
        # visualization)
        return output, hidden, attn_weights

    def decode(self, encoder_outputs, input_seq=None):
        # TODO: stop when predicting <eos>
        # Prepare variables
        batch_size = encoder_outputs.shape[0]
        input = self.sos.expand(batch_size, -1)
        hidden = self.h0.expand(-1, batch_size, -1).contiguous()
        logits = torch.empty([batch_size, 0, self.output_size])
        input_size = encoder_outputs.shape[1]
        attn_weights = torch.empty([batch_size, input_size, 0])
        if self.use_cuda:
            logits = logits.cuda()

        if input_seq is not None:
            target_length = input_seq.shape[1]
        else:
            target_length = self.max_output_length
        for di in range(target_length):
            output, hidden, att = self.forward(input, hidden, encoder_outputs)
            logits = torch.cat((logits, output), 1)
            attn_weights = torch.cat((attn_weights, att.detach().cpu()), 2)
            # Select next input
            use_teacher_forcing = False
            if input_seq is not None:
                use_teacher_forcing = random.random() < self.teacher_forcing_ratio
            if use_teacher_forcing:
                # Teacher forcing: Use the ground-truth target as the next
                # input
                input = input_seq[:, di].view(-1, 1)
            else:
                # Without teacher forcing: use network's own prediction as
                # the next input
                input = output.argmax(dim=2)
        return logits, attn_weights

    def beam_search(self, encoder_outputs, beam_size):
        # Prepare variables
        batch_size = encoder_outputs.shape[0]
        preds = np.empty([batch_size, self.max_output_length])
        # Loop over sequences
        for i_seq, eo in enumerate(encoder_outputs):
            eo = eo.unsqueeze(0)
            inputs = self.sos
            hiddens = self.h0
            hyps = np.empty([1, 0], dtype=int)
            scores = np.ones(1)
            ended_hyps = np.empty([0, 1], dtype=int)
            ended_scores = np.empty([0])
            ended_hidden = torch.ones([1, 0, self.hidden_size])
            if self.use_cuda:
                ended_hidden = ended_hidden.cuda()
            # Loop over time steps
            for di in range(self.max_output_length):
                logits, ht, _ = self.forward(inputs, hiddens, eo)
                new_hyps = ended_hyps
                new_scores = ended_scores
                new_hidden = ended_hidden
                # Loop over hypotheses
                for idx_h, h in enumerate(hyps):
                    best_scores, best_ids = torch.topk(logits[idx_h], beam_size)
                    tmp_scores = scores[idx_h] + best_scores.squeeze().cpu()
                    new_scores = np.hstack((new_scores, tmp_scores))
                    tmp_hyps = np.hstack([
                        np.repeat(h[np.newaxis, :], beam_size, axis=0),
                        best_ids.view(beam_size, 1).cpu()])
                    new_hyps = np.vstack((new_hyps, tmp_hyps))
                    tmp_hidden = ht[:, idx_h].repeat([beam_size, 1]).unsqueeze(0)
                    new_hidden = torch.cat((new_hidden, tmp_hidden), dim=1)
                    # Keep only <beam_size> best examples
                    new_order = np.argsort(-new_scores)
                    new_scores = new_scores[new_order][:beam_size]
                    new_hyps = new_hyps[new_order][:beam_size]
                    new_hidden = new_hidden[:, new_order][:, :beam_size]
                # Filter ended sequences
                ended = ((new_hyps[:, -1] == self.mapper.END_ID) |
                         (new_hyps[:, -1] == self.mapper.PAD_ID)).nonzero()[0]
                num_ended = len(ended)
                if num_ended == beam_size:
                    break
                ended_hyps = np.hstack((
                    new_hyps[ended],
                    np.repeat([[self.mapper.PAD_ID]], num_ended, axis=0)))
                ended_scores = new_scores[ended]
                # hidden vectors of ended sequences won't be used
                # but we need to reserve the space for the indices to match
                ended_hidden = torch.ones([1, num_ended, self.hidden_size])
                mask = np.ones(beam_size, dtype=bool)
                mask[ended] = 0
                hyps = new_hyps[mask]
                scores = new_scores[mask]
                indices = mask.nonzero()[0]
                hiddens = new_hidden[:, indices]
                # Select next input
                inputs = torch.unsqueeze(torch.from_numpy(hyps[:, -1]), 1)
                if self.use_cuda:
                    ended_hidden = ended_hidden.cuda()
                    inputs = inputs.cuda()
                # Duplicate encoder's output
                eo = eo[0].unsqueeze(0).repeat([beam_size - num_ended, 1, 1])
            preds[i_seq, :len(new_hyps[0])] = new_hyps[0]
        return preds
