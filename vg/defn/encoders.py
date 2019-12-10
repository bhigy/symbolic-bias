import math
import torch
import torch.nn as nn
from onion import attention, conv
import onion.util as util
import torch.nn.functional as F
import vg.defn.introspect


def l2normalize(x):
    return F.normalize(x, p=2, dim=1)


# LEGACY
class TextEncoder(nn.Module):
    def __init__(self, size_feature, size, size_embed=64, depth=1,
                 size_attn=512, dropout_p=0.0):
        super(TextEncoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed = nn.Embedding(self.size_feature, self.size_embed)
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN = nn.GRU(self.size_embed, self.size, self.depth,
                          batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return l2normalize(self.Attn(out))


class TextEncoderBottom(nn.Module):
    def __init__(self, size_feature, size, size_embed=64, depth=1,
                 dropout_p=0.0):
        super(TextEncoderBottom, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed = nn.Embedding(self.size_feature, self.size_embed)
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN = nn.GRU(self.size_embed, self.size, self.depth,
                          batch_first=True)

    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return out


class TextEncoderTop(nn.Module):
    def __init__(self, size_feature, size, depth=1, size_attn=512, dropout_p=0.0):
        super(TextEncoderTop, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_feature, self.size, self.depth,
                              batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, _last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return l2normalize(self.Attn(out))


class SpeechEncoder(nn.Module):
    def __init__(self, size_vocab, size, depth=1, filter_length=6,
                 filter_size=64, stride=2, size_attn=512, dropout_p=0.0):
        super(SpeechEncoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length,
                                       self.filter_size, stride=self.stride)
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN = nn.GRU(self.filter_size, self.size, self.depth,
                          batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def forward(self, input):
        h0 = self.h0.expand(self.depth, input.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Conv(input)), h0)
        return l2normalize(self.Attn(out))


class GRUStack(nn.Module):
    """
    GRU stack with separate GRU modules so that full intermediate states
    can be accessed.
    """
    def __init__(self, size_in, size, depth):
        super(GRUStack, self).__init__()
        self.bottom = nn.GRU(size_in, size, 1, batch_first=True)
        self.layers = nn.ModuleList([nn.GRU(size, size, 1, batch_first=True) for i in range(depth-1)])

    def forward(self, x):
        hidden = []
        output, _ = self.bottom(x)
        hidden.append(output)
        for rnn in self.layers:
            output, _ = rnn(hidden[-1])
            hidden.append(output)
        return torch.stack(hidden)


class SpeechEncoderBottom(nn.Module):
    def __init__(self, size_vocab, size, nb_conv_layer=1, depth=1,
                 filter_length=6, filter_size=[64], stride=2, dropout_p=0.0,
                 relu=False, maxpool=False, bidirectional=False):
        super(SpeechEncoderBottom, self).__init__()
        util.autoassign(locals())
        layers = []
        size_in = self.size_vocab
        for i_conv in range(0, self.nb_conv_layer):
            layers.append(conv.Convolution1D(size_in,
                                             self.filter_length,
                                             self.filter_size[i_conv],
                                             stride=self.stride,
                                             maxpool=self.maxpool,
                                             padding=0))
            if self.relu:
                layers.append(nn.ReLU(True))
            size_in = self.filter_size[i_conv]
        self.Conv = nn.Sequential(*layers)
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.filter_size[self.nb_conv_layer - 1],
                              self.size, self.depth, batch_first=True,
                              bidirectional=self.bidirectional)

    def forward(self, x, x_len):
        out = self.Conv(x)
        if self.depth > 0:
            out, _ = self.RNN(self.Dropout(out))
        return out

    def introspect(self, input, l):
        if not hasattr(self, 'IntrospectRNN'):
            self.IntrospectRNN = vg.defn.introspect.IntrospectGRU(self.RNN)
        result = {}

        # Computing convolutional activations
        i_module = 0
        for i_conv in range(0, self.nb_conv_layer):
            conv = self.Conv._modules[str(i_module)]
            input = conv(input)
            i_module += 1
            if self.relu:
                input = self.Conv._modules[str(i_module)](input)
                i_module += 1
            l = inout(conv.Conv, l, self.maxpool)
            activations = [input[i, :l[i], :].cpu().numpy() for i in range(len(input))]
            result['conv'.format(i_conv)] = activations

        # Computing full stack of RNN
        out = self.Dropout(input)
        rnn = self.IntrospectRNN.introspect(out)
        for layer in range(self.RNN.num_layers):
            activations = [rnn[i, layer, :l[i], :].cpu().numpy() for i in range(len(rnn))]
            result['rnn{}'.format(layer)] = activations

        return result


def inout(Conv, L, maxpool):
    """Mapping from size of input to the size of the output of a 1D
    convolutional layer.
    https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    """
    pad    = 0
    ksize  = Conv.kernel_size[0]
    stride = Conv.stride[0] * 2 if maxpool else Conv.stride[0]
    return ((L.float() + 2*pad - 1*(ksize-1) - 1) / stride + 1).floor().long()


class SpeechEncoderBottomVGG(nn.Module):
    def __init__(self, size, depth=1, filter_length=3, init_filter_size=64,
                 stride=1, padding=1, dropout_p=0.0, relu=True,
                 bidirectional=False):
        super(SpeechEncoderBottomVGG, self).__init__()
        util.autoassign(locals())
        layers = []
        layers.append(conv.Convolution2D(1,
                                         self.filter_length,
                                         self.init_filter_size,
                                         stride=self.stride,
                                         padding=self.padding,
                                         relu=self.relu))
        layers.append(conv.Convolution2D(self.init_filter_size,
                                         self.filter_length,
                                         self.init_filter_size,
                                         stride=self.stride,
                                         padding=self.padding,
                                         relu=self.relu,
                                         maxpool=True))
        layers.append(conv.Convolution2D(self.init_filter_size,
                                         self.filter_length,
                                         self.init_filter_size * 2,
                                         stride=self.stride,
                                         padding=self.padding,
                                         relu=self.relu))
        layers.append(conv.Convolution2D(self.init_filter_size * 2,
                                         self.filter_length,
                                         self.init_filter_size * 2,
                                         stride=self.stride,
                                         padding=self.padding,
                                         relu=self.relu,
                                         maxpool=True))
        self.Conv = nn.Sequential(*layers)
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            # TODO: LSTM/GRU?
            #self.RNN = nn.GRU(self.init_filter_size * 2,
            #                  self.size, self.depth, batch_first=True,
            #                  bidirectional=self.bidirectional)
            self.RNN = nn.LSTM(self.init_filter_size * 2 * 4,
                               self.size, self.depth, batch_first=True,
                               bidirectional=self.bidirectional)

    def forward(self, x, x_len):
        out = self.Conv(x.unsqueeze(-1))
        out = out.contiguous()
        out = out.view(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
        if self.depth > 0:
            out, last = self.RNN(self.Dropout(out))
        return out


class SpeechEncoderBottomStack(nn.Module):
    def __init__(self, size_vocab, size, depth=1, filter_length=6,
                 filter_size=64, stride=2, dropout_p=0.0):
        super(SpeechEncoderBottomStack, self).__init__()
        util.autoassign(locals())
        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length,
                                       self.filter_size, stride=self.stride)
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = GRUStack(self.filter_size, self.size, self.depth)

    def forward(self, x):
        return self.states(x)[-1]

    def states(self, x):
        if self.depth > 0:
            out = self.RNN(self.Dropout(self.Conv(x)))
        else:
            out = self.Conv(x)
        return out


class SpeechEncoderBottomNoConv(nn.Module):
    def __init__(self, size_vocab, size, depth=1, dropout_p=0.0):
        super(SpeechEncoderBottomNoConv, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1,
                                                          self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_vocab, self.size, self.depth,
                              batch_first=True)

    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return out


class SpeechEncoderBottomBidi(nn.Module):
    def __init__(self, size_vocab, size, depth=1, dropout_p=0.0):
        super(SpeechEncoderBottomBidi, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_vocab, self.size, self.depth,
                              batch_first=True, bidirectional=True)
            self.Down = nn.Linear(self.size * 2, self.size)

    def forward(self, x):
        if self.depth > 0:
            out, last = self.RNN(self.Dropout(x))
            out = self.Down(out)
        else:
            out = x
        return out


class SpeechEncoderTop(nn.Module):
    def __init__(self, size_input, size, depth=1, size_attn=512, dropout_p=0.0):
        super(SpeechEncoderTop, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1,
                                                          self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_input, self.size, self.depth,
                              batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, _last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return l2normalize(self.Attn(out))

    def states(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, _last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return out, l2normalize(self.Attn(out))


class SpeechEncoderTopStack(nn.Module):
    def __init__(self, size_input, size, depth=1, size_attn=512, dropout_p=0.0):
        super(SpeechEncoderTopStack, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = GRUStack(self.size_input, self.size, self.depth)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def states(self, x):
        if self.depth > 0:
            out = self.RNN(self.Dropout(x))
        else:
            out = x
        return out

    def forward(self, x):
        out = self.states(x)
        return l2normalize(self.Attn(out[-1]))


class SpeechEncoderTopBidi(nn.Module):
    def __init__(self, size_input, size, depth=1, size_attn=512, dropout_p=0.0):
        super(SpeechEncoderTopBidi, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_input, self.size, self.depth,
                              batch_first=True, bidirectional=True)
            self.Down = nn.Linear(self.size * 2, self.size)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def forward(self, x):
        if self.depth > 0:
            out, _last = self.RNN(self.Dropout(x))
            out = self.Down(out)
        else:
            out = x
        return l2normalize(self.Attn(out))

    def states(self, x):
        if self.depth > 0:
            out, _last = self.RNN(self.Dropout(x))
            out = self.Down(out)
        else:
            out = x
        return out, l2normalize(self.Attn(out))


class ImageEncoder(nn.Module):

    def __init__(self, size, size_target):
        super(ImageEncoder, self).__init__()
        self.Encoder = util.make_linear(size_target, size)

    def forward(self, img):
        return l2normalize(self.Encoder(img))
