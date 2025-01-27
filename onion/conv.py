import torch.nn as nn
import onion.util as util
import onion.init as init

class Convolution1D(nn.Module):
    """A one-dimensional convolutional layer.
    """
    def __init__(self, size_in, length, size, stride=1, padding=None,
                 maxpool=False):
        super(Convolution1D, self).__init__()
        util.autoassign(locals())
        padding = padding if padding is not None else self.length
        self.Conv = nn.Conv1d(self.size_in, self.size, self.length,
                              stride=self.stride, padding=padding, bias=False)
        # use Glorot uniform initialization
        self.Conv.weight.data = init.glorot_uniform((self.size, self.size_in,
                                                     self.length, 1)).squeeze()
        #FIXME what is the correct padding???
        if self.maxpool:
            self.Maxpool = nn.MaxPool1d(2, 2, ceil_mode=True)

    def forward(self, signal):
        # signal's shape is (B, T, C) where B=batch size, T=timesteps, C=channels
        out = self.Conv(signal.permute(0, 2, 1))
        if self.maxpool:
            out = self.Maxpool(out)
        return out.permute(0, 2, 1)


class Convolution2D(nn.Module):
    """A one-dimensional convolutional layer.
    """
    def __init__(self, size_in, length, size, stride=1, padding=None,
                 maxpool=False, relu=False):
        super(Convolution2D, self).__init__()
        util.autoassign(locals())
        padding = padding if padding is not None else self.length
        #self.Conv = nn.Conv2d(self.size_in, self.size, self.length,
        #                      stride=self.stride, padding=padding, bias=False)
        self.Conv = nn.Conv2d(self.size_in, self.size, self.length,
                              stride=self.stride, padding=padding)
        # use Glorot uniform initialization
        # TODO: decide which initialization to use
        #self.Conv.weight.data = init.glorot_uniform((self.size, self.size_in,
        #                                             self.length, self.length))
        if self.relu:
            self.Relu = nn.ReLU(True)
        #FIXME what is the correct padding???
        if self.maxpool:
            self.Maxpool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, signal):
        # signal's shape is (B, T, F, C)
        # where B=batch size, T=timesteps, F=filter, C=channels
        out = self.Conv(signal.permute(0, 3, 1, 2))
        if self.relu:
            out = self.Relu(out)
        if self.maxpool:
            out = self.Maxpool(out)
        return out.permute(0, 3, 2, 1)


class ConvTranspose1D(nn.Module):
    """A one-dimensional convolutional layer.
    """
    def __init__(self, size_in, length, size, stride=1, padding=None):
        super(ConvTranspose1D, self).__init__()
        util.autoassign(locals())
        padding = padding if padding is not None else self.length
        self.ConvT = nn.ConvTranspose1d(self.size_in, self.size, self.length,
                                        stride=self.stride, padding=padding,
                                        bias=False)
        # use Glorot uniform initialization
        self.ConvT.weight.data = init.glorot_uniform((self.size, self.size_in,
                                                      self.length, 1)).squeeze()
        #FIXME what is the correct padding???

    def forward(self, signal):
        # signal's shape is (B, T, C) where B=batch size, T=timesteps, C=channels
        out = self.ConvT(signal.permute(0, 2, 1))
        return out.permute(0, 2, 1)


