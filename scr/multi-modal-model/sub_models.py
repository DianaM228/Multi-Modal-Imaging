"""MultiSurv sub-models."""

from bisect import bisect_left
import torch
import torch.nn as nn
from torchvision import models
from embrace_net import EmbraceNet
from attention import Attention
from utils_baseline import check_parameters_between_two_models, count_parameters
import torch.nn.init as init


def freeze_layers(model, up_to_layer=None):
    if up_to_layer is not None:
        # Freeze all layers
        for i, param in model.named_parameters():
            param.requires_grad = False

        # Release all layers after chosen layer
        frozen_layers = []
        for name, child in model.named_children():
            if up_to_layer in frozen_layers:
                for params in child.parameters():
                    params.requires_grad = True
            else:
                frozen_layers.append(name)
    else:
        for i, param in model.named_parameters():
            param.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResNet(nn.Module):
    def __init__(self, freeze_up_to=None):
        super(ResNet, self).__init__()
        """self.model = models.resnext50_32x4d(pretrained=True)        
        freeze_layers(self.model, up_to_layer="layer3")"""
        self.model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        self.freeze_up_to = freeze_up_to
        freeze_layers(self.model, self.freeze_up_to)
        self.n_features = self.model.fc.in_features
        # Remove classifier (last layer)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x):
        out = self.model(x)

        return out


class FC(nn.Module):
    "Fully-connected model to generate final output."

    def __init__(
        self,
        in_features,
        out_features,
        n_layers,
        dropout,
        batchnorm=False,
        scaling_factor=4,
    ):
        super(FC, self).__init__()
        if n_layers == 1:
            layers = self._make_layer(in_features, out_features, dropout, batchnorm)
        elif n_layers > 1:
            n_neurons = self._pick_n_neurons(in_features)
            if n_neurons < out_features:
                n_neurons = out_features

            if n_layers == 2:
                layers = self._make_layer(
                    in_features, n_neurons, dropout, batchnorm=True
                )
                layers += self._make_layer(n_neurons, out_features, dropout, batchnorm)
            else:
                for layer in range(n_layers):
                    last_layer_i = range(n_layers)[-1]

                    if layer == 0:
                        n_neurons *= scaling_factor
                        layers = self._make_layer(
                            in_features, n_neurons, dropout, batchnorm=True
                        )
                    elif layer < last_layer_i:
                        n_in = n_neurons
                        n_neurons = self._pick_n_neurons(n_in)
                        if n_neurons < out_features:
                            n_neurons = out_features
                        layers += self._make_layer(
                            n_in, n_neurons, dropout, batchnorm=True
                        )
                    else:
                        layers += self._make_layer(
                            n_neurons, out_features, dropout, batchnorm
                        )
        else:
            raise ValueError('"n_layers" must be positive.')

        self.fc = nn.Sequential(*layers)

    def _make_layer(self, in_features, out_features, dropout, batchnorm):
        layer = nn.ModuleList()
        if dropout:
            layer.append(nn.Dropout(p=dropout))
        layer.append(nn.Linear(in_features, out_features))
        layer.append(nn.ReLU(inplace=True))
        if batchnorm:
            layer.append(nn.BatchNorm1d(out_features))

        return layer

    def _pick_n_neurons(self, n_features):
        # Pick number of features from list immediately below n input
        n_neurons = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        idx = bisect_left(n_neurons, n_features)

        return n_neurons[0 if idx == 0 else idx - 1]

    def forward(self, x):
        return self.fc(x)


class ChannelNormalizationLayer(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ChannelNormalizationLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, input):
        mean = torch.mean(input, dim=1, keepdim=True)
        var = torch.var(input, dim=1, unbiased=False, keepdim=True)

        normalized_input = (input - mean) / torch.sqrt(var + self.eps)

        output = self.gamma * normalized_input + self.beta

        return output


class FC_fixed(nn.Module):
    def __init__(self, dropout,k_init):
        super(FC_fixed, self).__init__()
        self.dropout = dropout
        self.k_init = k_init
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            # ChannelNormalizationLayer(256),
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            # ChannelNormalizationLayer(128),
        )
        if self.k_init:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):                
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(x)
    


class WsiNet(nn.Module):
    "WSI patch feature extractor and aggregator."

    def __init__(self, dropout,k_init, output_vector_size, freeze_up_t=None):
        super(WsiNet, self).__init__()
        self.feature_extractor = ResNet(freeze_up_t)
        self.num_image_features = self.feature_extractor.n_features
        # Multiview WSI patch aggregation
        self.fc = FC(self.num_image_features, output_vector_size, 1, dropout)

    def forward(self, x):
        view_pool = []

        # Extract features from each patch
        for v in x:  
            v = self.feature_extractor(v)  
            v = v.view(v.size(0), self.num_image_features)  

            view_pool.append(v)  

        # Aggregate features from all patches
        patch_features = torch.stack(view_pool).max(dim=1)[0]  

        out = self.fc(patch_features)  

        return out


class Fusion(nn.Module):
    "Multimodal data aggregator."

    def __init__(self, method, feature_size, device):
        super(Fusion, self).__init__()
        self.method = method
        methods = ["cat", "max", "sum", "prod", "embrace", "attention"]

        if self.method not in methods:
            raise ValueError('"method" must be one of ', methods)

        if self.method == "embrace":
            if device is None:
                raise ValueError('"device" is required if "method" is "embrace"')

            self.embrace = EmbraceNet(device=device)

        if self.method == "attention":
            if not feature_size:
                raise ValueError(
                    '"feature_size" is required if "method" is "attention"'
                )
            self.attention = Attention(size=feature_size)

    def forward(self, x):
        if self.method == "attention":
            out = self.attention(x)
        if self.method == "cat":
            out = torch.cat([m for m in x], dim=1)
        if self.method == "max":
            out = x.max(dim=0)[0]
        if self.method == "sum":
            out = x.sum(dim=0)
        if self.method == "prod":
            out = x.prod(dim=0)
        if self.method == "embrace":
            out = self.embrace(x)

        return out
