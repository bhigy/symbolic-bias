import os
import time

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from vg.scorer import testing
from vg.defn.encoders import SpeechEncoderBottom, SpeechEncoderBottomVGG
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
        self.TextDecoder = BahdanauAttnDecoderRNN(**config['TextDecoder'])
        # TODO: choose optimizer
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-8)
        #self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])
        self.mapper = config['mapper']

    def forward(self, speech, seq_len, target=None):
        out = self.SpeechEncoderBottom(speech, seq_len)
        logits, attn_weights = self.TextDecoder.decode(out, target)
        return logits, attn_weights

    def predict(self, audio, audio_len):
        with testing(self):
            logits, _ = self.forward(audio, audio_len)
        return self.logits2pred(logits.detach().cpu())

    def logits2pred(self, logits):
        pred = []
        ids = logits.argmax(dim=2)
        for i_seq in range(ids.shape[0]):
            seq = ids[i_seq]
            i_eos = (seq == self.mapper.END_ID).nonzero()
            i_last = i_eos[0] if i_eos.shape[0] > 0 else seq.shape[0]
            chars = [self.mapper.ids.from_id(id.item()) for id in seq[:i_last]]
            pred.append(''.join(chars))
        return pred

    def cost(self, speech, target, seq_len):
        logits, _ = self.forward(speech, seq_len, target)

        # Masking padding
        nb_tokens = self.mapper.ids.max
        # * flatten vectors
        target = target.view(-1)
        logits = logits.view(-1, nb_tokens)
        # * compute and apply mask
        mask = (target != self.mapper.PAD_ID)
        target = target[mask]
        logits = logits[mask, :]

        cost = F.cross_entropy(logits, target)
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
        self.SpeechEncoderBottom = SpeechEncoderBottom(
            **config['SpeechEncoderBottom'])
        self.SpeechTranscriber = SpeechTranscriber(
            self.SpeechEncoderBottom, config['SpeechTranscriber'])

    def predict(self, audio, audio_len):
        return self.SpeechTranscriber.predict(audio, audio_len)


class NetVGG(nn.Module):
    def __init__(self, config):
        super(NetVGG, self).__init__()
        self.SpeechEncoderBottom = SpeechEncoderBottomVGG(
            **config['SpeechEncoderBottomVGG'])
        self.SpeechTranscriber = SpeechTranscriber(
            self.SpeechEncoderBottom, config['SpeechTranscriber'])

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
    model_fpath = "model.pkl"
    if run_config['debug']:
        wdump_fpath = "weights.csv"
    if run_config['save_path'] is not None:
        result_fpath = os.path.join(run_config['save_path'], result_fpath)
        model_fpath_tmpl = os.path.join(run_config['save_path'],
                                        model_fpath_tmpl)
        model_fpath = os.path.join(run_config['save_path'], model_fpath)
        if run_config['debug']:
            wdump_fpath = os.path.join(run_config['save_path'], wdump_fpath)
            wdump = open(wdump_fpath, "w")

    for _, task in run_config['tasks']:
        task.optimizer.zero_grad()

    with open(result_fpath, "w") as out:
        if run_config['debug']:
            t = time.time()
        best_wer = None
        for epoch in range(last_epoch+1, run_config['epochs'] + 1):
            # FIXME: avoid end of epoch with small batch?
            for _j, item in enumerate(data.iter_train_batches(reshuffle=True)):
                j = _j + 1
                for name, task in run_config['tasks']:
                    spkr = item['speaker']
                    spkr = spkr[0] if len(set(spkr)) == 1 else 'MIXED'
                    args = task.args(item)
                    args = [torch.from_numpy(x).cuda() for x in args]

                    loss = task.cost(*args)

                    task.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(task.parameters(),
                                             task.config['max_norm'])
                    task.optimizer.step()

                    print(epoch, j, j*data.batch_size, spkr, "train",
                          str(loss.item()))

                    if j % run_config['validate_period'] == 0:
                        loss = valid_loss(net, task, data)
                        print(epoch, j, 0, "VALID", "valid", str(np.mean(loss)))
                        # Dump weights for debugging
                        if run_config['debug']:
                            weights = [str(p.view(-1)[0].item()) for p in task.parameters()]
                            wdump.write(",".join(weights))
                            wdump.write("\n")
                            wdump.flush()

                    sys.stdout.flush()
            torch.save(net, model_fpath_tmpl.format(epoch))

            if run_config['debug']:
                t2 = time.time()
                print("Elapsed time: {:3f}".format(t2 - t))
                t = t2
            with testing(net):
                scorer.set_net(net)
                result = dict(epoch=epoch,
                              cer=scorer.cer(),
                              wer=scorer.wer())
                cer = result['cer']['CER']
                wer = result['wer']['WER']
                print(epoch, j, 0, "CER", "valid", cer, "WER", "valid", wer)
                out.write(json.dumps(result))
                out.write("\n")
                out.flush()
                # Save best model
                if best_wer is None or wer < best_wer:
                    torch.save(net.state_dict(), model_fpath)
                    best_wer = wer
                else:
                    net.load_state_dict(torch.load(model_fpath))
                    if 'epsilon_decay' in run_config.keys():
                        for p in net.SpeechTranscriber.optimizer.param_groups:
                            p["eps"] *= run_config['epsilon_decay']
                            print('Epsilon decay - new value: ', p["eps"])
            if run_config['debug']:
                t2 = time.time()
                print("Elapsed time: {:3f}".format(t2 - t))
                t = t2

    if run_config['debug']:
        wdump.close()
