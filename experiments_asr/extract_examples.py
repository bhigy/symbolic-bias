import argparse
import numpy
import os
import random
import sys
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
save_path = None

# Parse command line parameters
parser = argparse.ArgumentParser()
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
            hidden_size=1024,
            output_size=data_flickr.mapper.ids.max,
            sos_id=data_flickr.mapper.BEG_ID,
            depth=1,
            max_output_length=400),  # max length for annotations is 199
        lr=0.0002,
        max_norm=2.0,
        mapper=data_flickr.mapper))


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
    v_audio = torch.autograd.Variable(torch.from_numpy(
        sd.vector_padder(sent_audio, pad_end=True))).cuda()
    v_audio_len = torch.autograd.Variable(torch.from_numpy(
        numpy.array(sent_len))).cuda()
    logits = speech_transcriber.forward(v_audio, v_audio_len)
    trn = speech_transcriber.logits2pred(logits)
    return trn


net = D.Net(model_config)
net.batcher = None
net.mapper = None

scorer = vg.scorer.ScorerASR(prov_flickr,
                             dict(split='val',
                                  tokenize=get_audio,
                                  batch_size=batch_size,
                                  limit=limit))

net.cuda()
last_epoch = 0
model_fpath_tmpl = "model.{}.pkl"
if save_path is not None:
    model_fpath_tmpl = os.path.join(save_path, model_fpath_tmpl)

for epoch in range(last_epoch + 1, epochs + 1):
    print("EPOCH", epoch)
    net = torch.load(model_fpath_tmpl.format(epoch))

    for sentences_tr in data_flickr.iter_train_batches():
        break
    audio = sentences_tr['audio']
    target = sentences_tr['target_t']
    audio_len = [sd.shape[0] for sd in audio]
    v_audio = torch.from_numpy(numpy.array(audio)).cuda()
    v_target = torch.LongTensor(numpy.array(target)).cuda()
    v_audio_len = torch.from_numpy(numpy.array(audio_len)).cuda()
    net.SpeechTranscriber.cost(v_audio, v_target, v_audio_len)

    print("TRAINING")
    sentences_tr = load_sentences('train', 1)
    hyp_tr = transcribe(net.SpeechTranscriber, sentences_tr)
    for i_sent, sent in enumerate(sentences_tr):
        ref = sent['raw']
        print("-" * 80)
        print("ref{}: {}".format(i_sent, ref))
        print("hyp{}: {}".format(i_sent, hyp_tr[i_sent]))
    print("-" * 80)
    print("")

    print("VALIDATION")
    sentences_dt = load_sentences('val', 1)
    hyp_dt = transcribe(net.SpeechTranscriber, sentences_dt)
    for i_sent, sent in enumerate(sentences_dt):
        tgt = sent['raw']
        print("-" * 80)
        print("t{}: {}".format(i_sent, tgt))
        print("p{}: {}".format(i_sent, hyp_dt[i_sent]))
    print("-" * 80)
    print("")
    sys.stdout.flush()
