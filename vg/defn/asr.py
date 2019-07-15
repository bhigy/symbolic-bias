import os
import time

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from vg.scorer import testing
from vg.defn.encoders import SpeechEncoderBottom, SpeechEncoderTop
from vg.defn.decoders import BahdanauAttnDecoderRNN
from collections import Counter
import sys
import json


def step(task, *args):
    loss = task.train_cost(*args)
    task.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(task.parameters(), task.config['max_norm'])
    return loss


class SpeechTranscriber(nn.Module):
    def __init__(self, speech_encoder, config):
        super(SpeechTranscriber, self).__init__()
        self.config = config
        self.SpeechEncoderBottom = speech_encoder
        self.SpeechEncoderTop = SpeechEncoderTop(**config['SpeechEncoderTop'])
        #self.TextDecoder = DecoderWithAttn(**config['TextDecoder'])
        self.TextDecoder = BahdanauAttnDecoderRNN(**config['TextDecoder'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])
        self.mapper = config['mapper']

    def forward(self, speech, seq_len, target=None):
        states, rep = self.SpeechEncoderTop.states(
            self.SpeechEncoderBottom(speech, seq_len))
        logits, attn_weights = self.TextDecoder.decode(states, target)
        return logits

    def predict(self, audio, audio_len):
        pred = []
        with testing(self):
            logits = self.forward(audio, audio_len)
            ids = logits.argmax(dim=2)
            for i_seq in range(ids.shape[0]):
                seq = ids[i_seq]
                i_eos = (seq == self.mapper.END_ID).nonzero()
                i_last = i_eos[0] if i_eos.shape[0] > 0 else seq.shape[0]
                chars = [self.mapper.ids.from_id(id.item()) for id in seq[:i_last]]
                pred.append(''.join(chars))
        return pred

    def cost(self, speech, target, seq_len):
        target_logits = self.forward(speech, seq_len, target)

        # Masking padding
        nb_tokens = self.mapper.ids.max
        # * flatten vectors
        target = target.view(-1)
        target_logits = target_logits.view(-1, nb_tokens)
        # * compute and apply mask
        mask = (target != self.mapper.BEG_ID)
        target = target[mask]
        target_logits = target_logits[mask, :]

        cost = F.cross_entropy(target_logits, target)
        return cost

    def args(self, item):
        return (item['audio'], item['target_t'].astype('int64'),
                item['nb_frames'])

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.SpeechEncoderBottom = SpeechEncoderBottom(**config['SpeechEncoderBottom'])
        self.SpeechTranscriber = SpeechTranscriber(self.SpeechEncoderBottom,
                                                   config['SpeechTranscriber'])

    def predict(self, audio, audio_len):
        return self.SpeechTranscriber.predict(audio, audio_len)


def valid_loss(net, task, data):
    result = []
    with testing(net):
        for item in data.iter_valid_batches():
            args = task.args(item)
            args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args]
            result.append(task.test_cost(*args).data.cpu().numpy())
    return result


def experiment(net, data, run_config):
    net.cuda()
    net.train()
    scorer = run_config['Scorer']
    last_epoch = 0
    result_fpath = "result.json"
    model_fpath_tmpl = "model.{}.pkl"
    if run_config['save_path'] is not None:
        result_fpath = os.path.join(run_config['save_path'], result_fpath)
        model_fpath_tmpl = os.path.join(run_config['save_path'],
                                        model_fpath_tmpl)

    for _, task in run_config['tasks']:
        task.optimizer.zero_grad()

    with open(result_fpath, "w") as out:
        t = time.time()
        for epoch in range(last_epoch+1, run_config['epochs'] + 1):
            cost = Counter()

            # FIXME: avoid end of epoch with small batch
            for _j, item in enumerate(data.iter_train_batches(reshuffle=True)):
                j = _j + 1
                spk = item['speaker'][0] if len(set(item['speaker'])) == 1 else 'MIXED'
                args = task.args(item)
                args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args]

                loss = task.cost(*args)

                task.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(task.parameters(),
                                         task.config['max_norm'])
                task.optimizer.step()
                cost += Counter({'cost': loss.data.item(), 'N': 1})
                print(epoch, j, j*data.batch_size, spk, "train",
                      "".join([str(cost['cost']/cost['N'])]))

                if j % run_config['validate_period'] == 0:
                    loss = valid_loss(net, task, data)
                    print(epoch, j, 0, "VALID", "valid",
                          "".join([str(np.mean(loss))]))

                sys.stdout.flush()
            torch.save(net, model_fpath_tmpl.format(epoch))

            t2 = time.time()
            print("Elapsed time: {:3f}".format(t2 - t))
            t = t2
            with testing(net):
                result = dict(epoch=epoch,
                              cer=scorer.cer(net))
                out.write(json.dumps(result))
                out.write("\n")
                out.flush()
            t2 = time.time()
            print("Elapsed time: {:3f}".format(t2 - t))
            t = t2

    torch.save(net, "model.pkl")
