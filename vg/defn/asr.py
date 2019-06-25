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

    def cost(self, speech, target, target_prev):
        states, rep = self.SpeechEncoderTop.states(self.SpeechEncoderBottom(speech))
        target_logits = self.TextDecoder(states, rep, target_prev)
        cost = F.cross_entropy(
            target_logits.view(target_logits.size(0) * target_logits.size(1), -1),
            target.view(target.size(0)*target.size(1)))
        return cost

    def args(self, item):
        return (item['audio'], item['target_t'].astype('int64'),
                item['target_prev_t'].astype('int64'))

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)

    # FIXME: implement
    def predict(self, audio):
        raise NotImplemented
        with testing(self):
            rep = self.SpeechImage.SpeechEncoderTop(self.SpeechImage.SpeechEncoderBottom(audio))
        return rep


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.SpeechEncoderBottom = SpeechEncoderBottom(**config['SpeechEncoderBottom'])
        self.SpeechTranscriber = SpeechTranscriber(self.SpeechEncoderBottom,
                                                   config['SpeechTranscriber'])


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
                import pdb; pdb.set_trace()
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

            # FIXME: replace evaluation
            with testing(net):
                result = dict(epoch=epoch,
                              cer=scorer.cer(net))
                out.write(json.dumps(result))
                out.write("\n")
                out.flush()

    torch.save(net, "model.pkl")
