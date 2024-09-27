import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=0.1):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


class ConvAE(nn.Module):
    def __init__(self, channels=3):
        super(ConvAE, self).__init__()

        d = 16

        # Encoder
        self.encoder_conv1 = nn.Conv2d(channels, d, 4, 2, 1)
        self.encoder_conv1_bn = nn.BatchNorm2d(d)
        self.encoder_conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.encoder_conv2_bn = nn.BatchNorm2d(d * 2)
        self.encoder_conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.encoder_conv3_bn = nn.BatchNorm2d(d * 4)

        self.encoder_conv4 = nn.Conv2d(d * 4, d * 4, 4, 2, 1)
        self.encoder_conv4_bn = nn.BatchNorm2d(d * 4)
        self.encoder_conv5 = nn.Conv2d(d * 4, d * 4, 4, 2, 1)
        self.encoder_conv5_bn = nn.BatchNorm2d(d * 4)
        self.encoder_conv6 = nn.Conv2d(d * 4, d * 4, 4, 2, 1)
        self.encoder_conv6_bn = nn.BatchNorm2d(d * 4)

        # Decoder
        self.decoder_deconv6 = nn.ConvTranspose2d(d * 4, d * 4, 4, 2, 1)
        self.decoder_deconv6_bn = nn.BatchNorm2d(d * 4)
        self.decoder_deconv5 = nn.ConvTranspose2d(d * 4, d * 4, 4, 2, 1)
        self.decoder_deconv5_bn = nn.BatchNorm2d(d * 4)
        self.decoder_deconv4 = nn.ConvTranspose2d(d * 4, d * 4, 4, 2, 1)
        self.decoder_deconv4_bn = nn.BatchNorm2d(d * 4)

        self.decoder_deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.decoder_deconv3_bn = nn.BatchNorm2d(d * 2)
        self.decoder_deconv2 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.decoder_deconv2_bn = nn.BatchNorm2d(d)
        self.decoder_deconv1 = nn.ConvTranspose2d(d, channels, 4, 2, 1)

    def encode(self, x):
        x = F.leaky_relu(
            self.encoder_conv1_bn(self.encoder_conv1(x)), negative_slope=0.2
        )
        x = F.leaky_relu(
            self.encoder_conv2_bn(self.encoder_conv2(x)), negative_slope=0.2
        )
        x = F.leaky_relu(
            self.encoder_conv3_bn(self.encoder_conv3(x)), negative_slope=0.2
        )

        x = F.leaky_relu(
            self.encoder_conv4_bn(self.encoder_conv4(x)), negative_slope=0.2
        )
        x = F.leaky_relu(
            self.encoder_conv5_bn(self.encoder_conv5(x)), negative_slope=0.2
        )
        x = F.leaky_relu(
            self.encoder_conv6_bn(self.encoder_conv6(x)), negative_slope=0.2
        )

        return x

    def decode(self, x):

        x = F.leaky_relu(
            self.decoder_deconv6_bn(self.decoder_deconv6(x)), negative_slope=0.2
        )
        x = F.leaky_relu(
            self.decoder_deconv5_bn(self.decoder_deconv5(x)), negative_slope=0.2
        )
        x = F.leaky_relu(
            self.decoder_deconv4_bn(self.decoder_deconv4(x)), negative_slope=0.2
        )

        x = F.leaky_relu(
            self.decoder_deconv3_bn(self.decoder_deconv3(x)), negative_slope=0.2
        )
        x = F.leaky_relu(
            self.decoder_deconv2_bn(self.decoder_deconv2(x)), negative_slope=0.2
        )
        x = torch.sigmoid(self.decoder_deconv1(x))
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        width_diff = x.size(-1) - decoded.size(-1)
        height_diff = x.size(-2) - decoded.size(-2)
        if height_diff>0 and width_diff>0:
            decoded = torch.nn.functional.pad(decoded, [int(width_diff/2),int(width_diff/2),int(height_diff/2),int(height_diff/2)], mode="replicate")
        elif height_diff<0 and width_diff<0:
            decoded=decoded[:,:,:height_diff,:width_diff]
        return encoded, decoded
        
