import numpy
from sklearn.metrics.pairwise import cosine_similarity
from vg.evaluate import ranking, paraphrase_ranking
import scipy
from vg.simple_data import vector_padder
import torch
import onion.util as util
import contextlib
import argparse
import logging
import vg.bundle as bundle
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from vg.simple_data import characters
from vg.wer import editDistance, getStepList, alignedPrint
import json
import Levenshtein as L


def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()

    p = commands.add_parser('score')
    p.set_defaults(func=score)
    p.add_argument('model',  nargs='+', help='Model file(s)')
    p.add_argument('--text', action='store_true')
    p.add_argument('--dataset', default='flickr8k')
    p.add_argument('--split', default='val')
    p.add_argument('--root', default='.')
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--output', default='result.json')
    args = parser.parse_args()
    args.func(args)


def score(args):
    if args.dataset == 'coco':
        import vg.vendrov_provider as dp
    elif args.dataset == 'places':
        import vg.places_provider as dp
    elif args.dataset == 'flickr8k':
        import vg.flickr8k_provider as dp
    logging.info('Loading data')
    prov = dp.getDataProvider(args.dataset, root=args.root, audio_kind='mfcc')
    tokenize = characters if args.text else lambda x: x['audio']
    config = dict(split=args.split, tokenize=tokenize,
                  batch_size=args.batch_size)
    if args.text:
        config['encode_sentences'] = encode_texts
    scorer = Scorer(prov, config)
    output = []
    for path in args.model:
        task = load(path)
        task.eval().cuda()
        rsa = scorer.rsa_image(task)
        para = scorer.retrieval_para(task)
        result = dict(path=path, rsa=rsa, para=para)
        if not args.text:
            result['speaker_id'] = scorer.speaker_id(task)
        output.append(result)
    json.dump(output, open(args.output, 'w'), indent=2)


def load(path):
    try:
        model = bundle.load(path)
        return model.task
    except:
        task = torch.load(path)
        return task


@contextlib.contextmanager
def testing(net):
    with torch.no_grad():
        net.eval()
        yield net
        net.train()


def stringsim(a, b):
    """Levenshtein edit distance normalized by length of longer string."""
    return 1 - L.distance(a, b) / max(len(a), len(b))


class Scorer:
    # FIXME this should just take a encoder function, not net
    def __init__(self, prov, config, net=None):
        self.prov = prov
        self.config = config
        self.sentences = []
        self.rsa_image_data = []
        self.images = []
        self.encode_sentences = config.get('encode_sentences',
                                           encode_sentences)
        self.encode_images = config.get('encode_images', encode_images)
        for image in prov.iterImages(split=config['split']):
            self.images.append(image)
            for sent in image['sentences']:
                self.rsa_image_data.append(image['feat'])
                self.sentences.append(sent)
        self.sentence_data = [config['tokenize'](s) for s in self.sentences]
        self.sim_images = cosine_similarity(self.rsa_image_data)
        self.correct_para = numpy.array(
            [[self.sentences[i]['imgid'] == self.sentences[j]['imgid']
              for j in range(len(self.sentences))]
             for i in range(len(self.sentences))])
        self.correct_img = numpy.array(
            [[self.sentences[i]['imgid'] == self.images[j]['imgid']
              for j in range(len(self.images))]
             for i in range(len(self.sentences))])
        # Precompute string similarity
        S = len(self.sentences)
        self.string_sim = numpy.zeros(shape=(S, S))
        for i in range(S):
            for j in range(S):
                self.string_sim[i, j] = stringsim(self.sentences[i]['raw'],
                                                  self.sentences[j]['raw'])

        self.net = net
        if self.net is not None:
            self.pred = self.encode_sentences(
                self.net, self.sentence_data,
                batch_size=self.config['batch_size'])
        self.speakers = Counter(s['speaker'] for s in self.sentences)

    def speaker_id(self, net=None):
        if net is None:
            pred = self.pred
            net = self.net
        else:
            with testing(net):
                pred = self.encode_sentences(
                    net, self.sentence_data,
                    batch_size=self.config['batch_size'])
        X = pred
        if hasattr(net, 'mapper'):
            # FIXME do something reasonable here
            Z = numpy.array([numpy.zeros((1)) for audio in self.sentence_data])
        else:
            Z = numpy.array([audio.mean(axis=0) for audio in self.sentence_data])
        y = LabelEncoder().fit_transform([s['speaker'] for s in self.sentences])
        C = [10**p for p in range(2, 3)]
        X, X_test, Z, Z_test, y, y_test = train_test_split(X, Z, y, random_state=42)
        scores = dict(rep=[], mfcc=[])
        counts = numpy.array(list(self.speakers.values()))
        maj = counts.max()/counts.sum()
        for c in C:
            model_rep = LogisticRegression(C=c)
            model_rep.fit(X, y)
            acc_rep = model_rep.score(X_test, y_test)
            model_mfcc = LogisticRegression(C=c)
            model_mfcc.fit(Z, y)
            acc_mfcc = model_mfcc.score(Z_test, y_test)
            logging.info("speaker_id acc {} {} {} {}".format(c, acc_rep, acc_mfcc, rer(acc_mfcc, acc_rep)))
            scores['rep'].append(acc_rep)
            scores['mfcc'].append(acc_mfcc)
        return dict(maj=maj, rep=max(scores['rep']), mfcc=max(scores['mfcc']))

    def rsa_image(self, net=None, within=False):
        # Full RSA
        if net is None:
            pred = self.pred
        else:
            with testing(net):
                pred = self.encode_sentences(
                    net, self.sentence_data,
                    batch_size=self.config['batch_size'])
        if hasattr(net, 'mapper') and net.mapper is not None:
            # FIXME do something reasonable here
            #print("This is a text net")
            mfcc = numpy.array([numpy.zeros((1)) for audio in self.sentence_data])
        else:
            #print("This is an audio net")
            mfcc = numpy.array([audio.mean(axis=0) for audio in self.sentence_data])
        sim_mfcc = cosine_similarity(mfcc)
        sim_pred = cosine_similarity(pred)

        img_rep = scipy.stats.pearsonr(triu(self.sim_images),
                                       triu(sim_pred))[0]
        img_mfcc = scipy.stats.pearsonr(triu(self.sim_images),
                                        triu(sim_mfcc))[0]
        rep_mfcc = scipy.stats.pearsonr(triu(sim_pred), triu(sim_mfcc))[0]
        result = dict(img_rep=float(img_rep), img_mfcc=float(img_mfcc),
                      rep_mfcc=float(rep_mfcc))  # make json happy
        if within:
            result['within'] = within_rsa
        return result

    def rsa_string(self, net=None):
        # Full RSA
        if net is None:
            pred = self.pred
        else:
            with testing(net):
                pred = self.encode_sentences(
                    net, self.sentence_data,
                    batch_size=self.config['batch_size'])
        if hasattr(net, 'mapper') and net.mapper is not None:
            # FIXME do something reasonable here
            #print("This is a text net")
            mfcc = numpy.array([numpy.zeros((1)) for audio in self.sentence_data])
        else:
            #print("This is an audio net")
            mfcc = numpy.array([audio.mean(axis=0) for audio in self.sentence_data])
        sim_mfcc = cosine_similarity(mfcc)
        sim_pred = cosine_similarity(pred)

        string_rep = scipy.stats.pearsonr(triu(self.string_sim),
                                          triu(sim_pred))[0]
        string_mfcc = scipy.stats.pearsonr(triu(self.string_sim),
                                           triu(sim_mfcc))[0]
        rep_mfcc = scipy.stats.pearsonr(triu(sim_pred), triu(sim_mfcc))[0]
        result = dict(string_rep=float(string_rep),
                      string_mfcc=float(string_mfcc),
                      rep_mfcc=float(rep_mfcc))  # make json happy
        return result

    def retrieval(self, net=None):
        img_fs = self.encode_images(net, [s['feat'] for s in self.images])
        if net is None:
            pred = self.pred
        else:
            with testing(net):
                pred = self.encode_sentences(
                    net, self.sentence_data,
                    batch_size=self.config['batch_size'])

        result = {}
        ret = ranking(img_fs, pred, self.correct_img, ns=(1, 5, 10),
                      exclude_self=False)
        result['recall@1'] = numpy.mean(ret['recall'][1])
        result['recall@5'] = numpy.mean(ret['recall'][5])
        result['recall@10'] = numpy.mean(ret['recall'][10])
        result['medr'] = numpy.median(ret['ranks'])
        return result

    def retrieval_para(self, net=None):
        if net is None:
            pred = self.pred
        else:
            with testing(net):
                pred = self.encode_sentences(
                    net, self.sentence_data,
                    batch_size=self.config['batch_size'])

        result = {}
        ret = paraphrase_ranking(pred, self.correct_para, ns=(1, 5, 10))
        result['recall@1'] = numpy.mean(ret['recall'][1])
        result['recall@5'] = numpy.mean(ret['recall'][5])
        result['recall@10'] = numpy.mean(ret['recall'][10])
        result['medr'] = numpy.median(ret['ranks'])
        return result


class ScorerASR:
    # FIXME this should just take a encoder function, not net
    def __init__(self, prov, config, net=None):
        self.prov = prov
        self.config = config
        self.sentences = []
        self.limit = config.get('limit', None)
        self.decode_sentences = config.get('decode_sentences',
                                           decode_sentences)
        for idx, image in enumerate(prov.iterImages(split=config['split'])):
            if self.limit is not None and idx >= self.limit:
                break
            for sent in image['sentences']:
                self.sentences.append(sent)
        self.sentence_data = [config['tokenize'](s) for s in self.sentences]

        self.net = net
        if self.net is not None:
            self.pred = self.decode_sentences(self.net, self.sentence_data)

    def set_net(self, net):
        self.net = net
        self.pred = self.decode_sentences(self.net, self.sentence_data)

    @staticmethod
    def nbeditops(s1, s2):
        d = 0
        i = 0
        s = 0
        for op in L.editops(s1, s2):
            if op[0] == 'delete':
                d += 1
            elif op[0] == 'insert':
                i += 1
            elif op[0] == 'replace':
                s += 1
        return d, i, s

    def cer(self, net=None):
        if net is None:
            pred = self.pred
        else:
            with testing(net):
                pred = self.decode_sentences(net, self.sentence_data)
        delete = 0
        insert = 0
        substitute = 0
        nbchar = 0
        for sent, p in zip(self.sentences, pred):
            ref = sent['raw']
            d, i, s = self.nbeditops(ref, p)
            delete += d
            insert += i
            substitute += s
            nbchar += len(ref)
        return (delete + insert + substitute) / nbchar

    def wer(self, net=None):
        if net is None:
            pred = self.pred
        else:
            with testing(net):
                pred = self.decode_sentences(net, self.sentence_data)
        delete = 0
        insert = 0
        substitute = 0
        correct = 0
        nbwords = 0
        for sent, p in zip(self.sentences, pred):
            ref = sent['raw']
            results = self.wer_sent(ref, p)
            delete += results['Del']
            insert += results['Ins']
            substitute += results['Sub']
            correct += results['Cor']
            nbwords += len(ref.split())
        wer =  (delete + insert + substitute) / nbwords
        return {'WER':wer, 'Cor':correct, 'Sub':substitute, 'Ins':insert,
                'Del':delete}

    def wer2(self, net=None):
        if net is None:
            pred = self.pred
        else:
            with testing(net):
                pred = self.decode_sentences(net, self.sentence_data)
        delete = 0
        insert = 0
        substitute = 0
        correct = 0
        nbwords = 0
        for sent, p in zip(self.sentences, pred):
            ref = sent['raw']
            results = self.wer_sent2(ref, p)
            delete += results['Del']
            insert += results['Ins']
            substitute += results['Sub']
            correct += results['Cor']
            nbwords += len(ref.split())
        wer =  (delete + insert + substitute) / nbwords
        return {'WER':wer, 'Cor':correct, 'Sub':substitute, 'Ins':insert,
                'Del':delete}

    @staticmethod
    def wer_sent(ref, hyp, debug=False):
        '''
        Computes the word error rate between reference and hypothesis
        Modified from SpacePineapple
        (https://progfruits.blogspot.com/2014/02/word-error-rate-wer-and-word.html)
        '''
        SUB_PENALTY = 100
        INS_PENALTY = 75
        DEL_PENALTY = 75

        r = ref.split()
        h = hyp.split()
        #costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r)+1):
            costs[i][0] = DEL_PENALTY*i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    costs[i][j] = costs[i-1][j-1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                    insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                    deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        numSub = 0
        numDel = 0
        numIns = 0
        numCor = 0
        if debug:
            print("OP\tREF\tHYP")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i-=1
                j-=1
                if debug:
                    lines.append("OK\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_SUB:
                numSub +=1
                i-=1
                j-=1
                if debug:
                    lines.append("SUB\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_INS:
                numIns += 1
                j-=1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                numDel += 1
                i-=1
                if debug:
                    lines.append("DEL\t" + r[i]+"\t"+"****")
        if debug:
            lines = reversed(lines)
            for line in lines:
                print(line)
            print("#cor " + str(numCor))
            print("#sub " + str(numSub))
            print("#del " + str(numDel))
            print("#ins " + str(numIns))
        #return (numSub + numDel + numIns) / (float) (len(r))
        wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
        return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns,
                'Del':numDel}

    @staticmethod
    def wer_sent2(ref, hyp ,debug=False):
        """
        This is a function that calculate the word error rate in ASR.
        You can use it like this: wer("what is it".split(), "what is".split())
        """
        ref = ref.split()
        hyp = hyp.split()

        # build the matrix
        d = editDistance(ref, hyp)

        # find out the manipulation steps
        list = getStepList(ref, hyp, d)

        wer = float(d[len(ref)][len(hyp)]) / len(r) * 100
        results = {'WER': wer, 'Cor': None, 'Sub': 0, 'Ins': 0, 'Del': 0}
        if debug:
            # print the result in aligned way
            wer = str("%.2f" % wer) + "%"
            alignedPrint(list, r, h, wer)

        for step in list:
            if step == 's':
                results['Sub'] += 1
            elif step == 'i':
                results['Ins'] += 1
            elif step == 'd':
                results['Del'] += 1
        nbwords = len(ref)
        results['Cor'] = nbwords - results['Sub'] - results['Del']
        return results


def rer(hi, lo):
    return ((1-lo)-(1-hi))/(1-lo)


def triu(x):
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones = numpy.ones_like(x)
    return x[numpy.triu(ones, k=1) == 1]


def RSA(M, N):
    return round(scipy.stats.pearsonr(triu(M), triu(N))[0], 3)


def encode_sentences(task, audios, batch_size=128):
    return numpy.vstack([task.predict(
        torch.autograd.Variable(torch.from_numpy(
            vector_padder(batch))).cuda()).data.cpu().numpy()
        for batch in util.grouper(audios, batch_size)])


def decode_sentences(task, audio, batch_size=128):
    pred = []
    for batch in util.grouper(audio, batch_size):
        audio_len = [a.shape[0] for a in batch]
        v_audio = torch.autograd.Variable(torch.from_numpy(
            vector_padder(batch, pad_end=True))).cuda()
        v_audio_len = torch.autograd.Variable(torch.from_numpy(
            numpy.array(audio_len))).cuda()
        pred.extend(task.predict(v_audio, v_audio_len))
    return pred


def encode_texts(task, texts, batch_size=128):
    return numpy.vstack([task.predict(
        torch.autograd.Variable(torch.from_numpy(
            task.batcher.batch_inp(task.mapper.transform(
                batch)).astype('int64'))).cuda()).data.cpu().numpy()
        for batch in util.grouper(texts, batch_size)])


def encode_images(task, imgs, batch_size=128):
    """Project imgs to the joint space using model.
    """
    return numpy.vstack([task.encode_images(
        torch.autograd.Variable(torch.from_numpy(
            numpy.vstack(batch))).cuda()).data.cpu().numpy()
        for batch in util.grouper(imgs, batch_size)])


def encode_sentences_SpeechText(task, audios, batch_size=128):
    def predict(x):
        return task.SpeechText.SpeechEncoderTop(
            task.SpeechText.SpeechEncoderBottom(x))
    return numpy.vstack([predict(
        torch.autograd.Variable(torch.from_numpy(
            vector_padder(batch))).cuda()).data.cpu().numpy()
        for batch in util.grouper(audios, batch_size)])


if __name__ == '__main__':
    main()
