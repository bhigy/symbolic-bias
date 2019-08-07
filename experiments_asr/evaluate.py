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
result_fpath = "result.json"
model_fpath_tmpl = "model.{}.pkl"
if save_path is not None:
    result_fpath = os.path.join(save_path, result_fpath)
    model_fpath_tmpl = os.path.join(save_path, model_fpath_tmpl)

with open(result_fpath, "w") as out:
    for epoch in range(last_epoch + 1, epochs + 1):
        print("EPOCH", epoch)
        net = torch.load(model_fpath_tmpl.format(epoch))

        with vg.scorer.testing(net):
            scorer.set_net(net)
            result = dict(epoch=epoch,
                          cer=scorer.cer(),
                          wer=scorer.wer()['WER'])
            print(epoch, "CER", "valid", result['cer'], "WER",
                  "valid", result['wer'])
            out.write(json.dumps(result))
            out.write("\n")
            out.flush()
