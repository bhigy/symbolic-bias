# Pytorch version of imaginet.audiovis_rhn
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from vg.scorer import testing
from vg.defn.encoders import SpeechEncoderBottom, SpeechEncoderTop
from vg.defn.decoders import DecoderWithAttn
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
        self.TextDecoder = DecoderWithAttn(**config['TextDecoder'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])
        self.mapper = config['mapper']
        self.sos = torch.LongTensor([[self.mapper.BEG_ID]]).cuda()

    def cost(self, speech, target, target_prev):
        states, rep = self.SpeechEncoderTop.states(self.SpeechEncoderBottom(speech))
        target_logits = self.TextDecoder(states, rep, target_prev)

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
                item['target_prev_t'].astype('int64'))

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.SpeechEncoderBottom = SpeechEncoderBottom(**config['SpeechEncoderBottom'])
        self.SpeechTranscriber = SpeechTranscriber(self.SpeechEncoderBottom,
                                                   config['SpeechTranscriber'])
        self.max_output_length = config['max_output_length']

    def predict(self, audio):
        pred = []
        with testing(self):
            states, rep = self.SpeechTranscriber.SpeechEncoderTop.states(
                self.SpeechTranscriber.SpeechEncoderBottom(audio))
            out = self.SpeechTranscriber.sos
            while out.item() != self.SpeechTranscriber.mapper.END_ID and \
                  len(pred) < self.max_output_length:
                out = self.SpeechTranscriber.TextDecoder(states, rep, out)
                imax = out[0, 0].argmax()
                pred.append(self.SpeechTranscriber.mapper.ids.from_id(imax.item()))
                out = imax.view(1, 1)
        return ''.join(pred)


def valid_loss(net, task, data):
    result = []
    with testing(net): #net.eval()
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

    for _, task in run_config['tasks']:
        task.optimizer.zero_grad()

    with open("result.json", "w") as out:
        for epoch in range(last_epoch+1, run_config['epochs'] + 1):
            cost = Counter()

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
            torch.save(net, "model.{}.pkl".format(epoch))

            with testing(net):
                result = dict(epoch=epoch,
                              cer=scorer.cer(net))
                out.write(json.dumps(result))
                out.write("\n")
                out.flush()

    torch.save(net, "model.pkl")
