import argparse
import numpy
import os
import random
seed = 103
random.seed(seed)
numpy.random.seed(seed)

import vg.simple_data as sd
import vg.flickr8k_provider as dp_f

import vg.defn.asr as D
import vg.scorer

import time

batch_size = 8
epochs = 25
limit = None
save_path = None
validate_period=400

# Parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-d', help='Debug mode', dest='debugmode',
                    action='store_true', default=False)
parser.add_argument('-t', help='Test mode', dest='testmode',
                    action='store_true', default=False)
args = parser.parse_args()

# Setup test mode
if args.testmode:
    epochs = 1
    limit = 50
    validate_period = (limit * 5) // (batch_size * 2)
    save_path = "tmp"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

prov_flickr = dp_f.getDataProvider('flickr8k', root='../..', audio_kind='mfcc')
data_flickr = sd.SimpleData(prov_flickr, tokenize=sd.characters, min_df=1,
                            scale=False, batch_size=batch_size, shuffle=True,
                            limit=limit, limit_val=limit)

model_config = dict(
    SpeechEncoderBottom=dict(
        size=320,
        depth=2,
        size_vocab=13,
        nb_conv_layer=1,
        filter_length=6,
        filter_size=[64],
        stride=2,
        bidirectional=True),
    SpeechTranscriber=dict(
        TextDecoder=dict(
            hidden_size=320,
            output_size=data_flickr.mapper.ids.max,
            mapper=data_flickr.mapper,
            depth=1,
            max_output_length=400,
            bidirectional_enc=True), # max length for annotations is 199
        beam_size=10,
        lr=0.0002,
        max_norm=2.0,
        mapper=data_flickr.mapper))


def audio(sent):
    return sent['audio']


net = D.Net(model_config)
net.batcher = None
net.mapper = None

scorer = vg.scorer.ScorerASR(
    prov_flickr,
    dict(split='val',
         tokenize=audio,
         batch_size=64,
         limit=limit,
         decode_sentences=vg.scorer.decode_sentences_beam))
run_config = dict(epochs=epochs,
                  validate_period=validate_period,
                  tasks=[('SpeechTranscriber', net.SpeechTranscriber)],
                  Scorer=scorer,
                  save_path=save_path,
                  debug=args.debugmode,
                  epsilon_decay=0.01)
D.experiment(net=net,
             data=data_flickr,
             run_config=run_config)
