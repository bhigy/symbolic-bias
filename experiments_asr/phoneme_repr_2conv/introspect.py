import argparse
import numpy
import pickle
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
                            limit=limit, limit_val=limit)


model_config = dict(
    SpeechEncoderBottom=dict(
        size=1024,
        depth=4,
        size_vocab=39,
        nb_conv_layer=2,
        filter_length=6,
        filter_size=[64, 64],
        stride=2,
        bidirectional=True),
    SpeechTranscriber=dict(
        TextDecoder=dict(
            hidden_size=1024,
            output_size=data_flickr.mapper.ids.max,
            mapper=data_flickr.mapper,
            depth=1,
            max_output_length=400, # max length for annotations is 199
            bidirectional_enc=True),
        lr=0.0002,
        max_norm=2.0,
        mapper=data_flickr.mapper))


def audio(sent):
    return sent['audio']


scorer = vg.scorer.ScorerASR(prov_flickr,
                             dict(split='val',
                                  tokenize=audio,
                                  batch_size=batch_size,
                                  limit=limit))


net_random = D.Net(model_config).cuda()
with vg.scorer.testing(net_random):
    metadata, activations = scorer.introspect(net_random)
    print(pickle.dump(metadata, open('global_input.pkl', 'wb')))
    print(pickle.dump(activations, open('global_random.pkl', 'wb')))
net_trained = torch.load('model.pkl').cuda()
with vg.scorer.testing(net_trained):
    _, activations = scorer.introspect(net_trained)
    print(pickle.dump(activations, open('global_trained.pkl', 'wb')))
