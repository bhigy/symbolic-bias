import argparse
import numpy
import random
seed = 103
random.seed(seed)
numpy.random.seed(seed)

import vg.simple_data as sd
import vg.flickr8k_provider as dp_f

import vg.defn.asr as D
import vg.scorer

import time

batch_size = 16
epochs = 25
limit = None

# Parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-t', help='Test mode', dest='testmode',
                    action='store_true', default=False)
args = parser.parse_args()

# Setup test mode
if args.testmode:
    epochs = 1
    limit = 100

prov_flickr = dp_f.getDataProvider('flickr8k', root='../..', audio_kind='mfcc')
data_flickr = sd.SimpleData(prov_flickr, tokenize=sd.characters, min_df=1,
                            scale=False, batch_size=batch_size, shuffle=True,
                            limit=limit)

model_config = dict(
    SpeechEncoderBottom=dict(
        size=1024,
        depth=2,
        size_vocab=13,
        filter_length=6,
        filter_size=64,
        stride=2),
    SpeechTranscriber=dict(
        SpeechEncoderTop=dict(
            size=1024,
            size_input=1024,
            depth=0,
            size_attn=128),
        TextDecoder=dict(
            size=1024,
            size_target_vocab=data_flickr.mapper.ids.max,
            size_embed=64,
            depth=1),
        lr=0.0002,
        max_norm=2.0,
        mapper=data_flickr.mapper),
    max_output_length=76) # max length for annotations is 38


def audio(sent):
    return sent['audio']


net = D.Net(model_config)
net.batcher = None
net.mapper = None

scorer = vg.scorer.ScorerASR(prov_flickr,
                             dict(split='val',
                                  tokenize=audio,
                                  batch_size=batch_size))
run_config = dict(epochs=epochs,
                  validate_period=400,
                  tasks=[('SpeechTranscriber', net.SpeechTranscriber)],
                  Scorer=scorer)
D.experiment(net=net,
             data=data_flickr,
             run_config=run_config)
