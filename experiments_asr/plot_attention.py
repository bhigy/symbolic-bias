import argparse
import math
import numpy
import os
import random
import torch

import vg.simple_data as sd
import vg.flickr8k_provider as dp_f
import vg.scorer
from plotting import plot_attention

seed = 103
random.seed(seed)
numpy.random.seed(seed)

batch_size = 16
epochs = 25
limit = None
save_path = None

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
    save_path = "tmp"

prov_flickr = dp_f.getDataProvider('flickr8k', root='..', audio_kind='mfcc')
data_flickr = sd.SimpleData(prov_flickr, tokenize=sd.characters, min_df=1,
                            scale=False, batch_size=batch_size, shuffle=True,
                            limit=limit, limit_val=limit)


def get_audio(sent):
    return sent['audio']


def load_sentences(split, limit):
    sentences = []
    cntr = 0
    for image in prov_flickr.iterImages(split=split):
        for sent in image['sentences']:
            sentences.append(sent)
        cntr += 1
        if cntr >= limit:
            break
    return sentences


def transcribe(speech_transcriber, sentences):
    sent_audio = [get_audio(s) for s in sentences]
    sent_len = [sd.shape[0] for sd in sent_audio]
    v_audio = torch.from_numpy(
        sd.vector_padder(sent_audio, pad_end=True)).cuda()
    v_audio_len = torch.from_numpy(numpy.array(sent_len)).cuda()
    trn = speech_transcriber.predict(v_audio, v_audio_len)
    return trn


def get_attn_weights(speech_transcriber, sentences):
    sent_audio = [get_audio(s) for s in sentences]
    sent_len = [sd.shape[0] for sd in sent_audio]
    v_audio = torch.from_numpy(
        sd.vector_padder(sent_audio, pad_end=True)).cuda()
    v_audio_len = torch.from_numpy(numpy.array(sent_len)).cuda()
    attn_weights = speech_transcriber.get_attn_weights(v_audio, v_audio_len)
    import pdb; pdb.set_trace()
    return attn_weights.numpy()


scorer = vg.scorer.ScorerASR(prov_flickr,
                             dict(split='val',
                                  tokenize=get_audio,
                                  batch_size=batch_size,
                                  limit=limit))

for path in args.path:
    net = torch.load(path)

    for sentences_tr in data_flickr.iter_train_batches():
        break
    audio = sentences_tr['audio']
    target = sentences_tr['target_t']
    audio_len = [sd.shape[0] for sd in audio]
    v_audio = torch.from_numpy(numpy.array(audio)).cuda()
    v_target = torch.LongTensor(numpy.array(target)).cuda()
    v_audio_len = torch.from_numpy(numpy.array(audio_len)).cuda()
    net.SpeechTranscriber.cost(v_audio, v_target, v_audio_len)

    sentences_tr = load_sentences('train', 1)
    hyp_tr = transcribe(net.SpeechTranscriber, sentences_tr)
    attn_weights = get_attn_weights(net.SpeechTranscriber, sentences_tr)
    for i_sent, sent in enumerate(sentences_tr):
        # Plotting attention
        len_hyp = len(hyp_tr[i_sent])
        len_audio = math.ceil(sent['audio'].shape[0] / 4)
        weights = attn_weights[i_sent, :len_audio, :len_hyp].T
        fig, ax = plot_attention(weights, hyp_tr[i_sent])
        path = os.path.join('figs/tr_' + str(i_sent) + '.png')
        fig.savefig(path, bbox_inches='tight')

    sentences_dt = load_sentences('val', 1)
    hyp_dt = transcribe(net.SpeechTranscriber, sentences_dt)
    attn_weights = get_attn_weights(net.SpeechTranscriber, sentences_dt)
    for i_sent, sent in enumerate(sentences_dt):
        # Plotting attention
        len_hyp = len(hyp_dt[i_sent])
        len_audio = math.ceil(sent['audio'].shape[0] / 4)
        weights = attn_weights[i_sent, :len_audio, :len_hyp].T
        fig, ax = plot_attention(weights, hyp_dt[i_sent])
        path = os.path.join('figs/dt_' + str(i_sent) + '.png')
        fig.savefig(path, bbox_inches='tight')
