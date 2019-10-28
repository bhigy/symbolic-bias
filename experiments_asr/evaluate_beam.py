import argparse
import json
import numpy
import os
import random
import torch

import vg.simple_data as sd
import vg.flickr8k_provider as dp_f
import vg.defn.asr as D
import vg.scorer

seed = 103
random.seed(seed)
numpy.random.seed(seed)

batch_size = 16
epochs = 25
limit = None

# Parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', metavar='path', help='Model\'s path', nargs='+')
parser.add_argument('-t', help='Test mode', dest='testmode',
                    action='store_true', default=False)
args = parser.parse_args()

# Setup test mode
if args.testmode:
    epochs = 1
    limit = 100

prov_flickr = dp_f.getDataProvider('flickr8k', root='..', audio_kind='mfcc')
data_flickr = sd.SimpleData(prov_flickr, tokenize=sd.characters, min_df=1,
                            scale=False, batch_size=batch_size, shuffle=True,
                            limit=limit, limit_val=limit)


def get_audio(sent):
    return sent['audio']


scorer = vg.scorer.ScorerASR(
    prov_flickr,
    dict(split='val',
         tokenize=get_audio,
         batch_size=batch_size,
         limit=limit,
         decode_sentences=vg.scorer.decode_sentences_beam))

for path in args.path:
    net = torch.load(path)

    with vg.scorer.testing(net):
        scorer.set_net(net)
        result = dict(model=path,
                      cer=scorer.cer(),
                      wer=scorer.wer()['WER'])
        print(json.dumps(result))
