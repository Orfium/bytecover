from typing import Tuple

import nnAudio.features.cqt as nnAudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={str(self.eps)})"


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self, in_channels: int, out_channels: int, last: bool = False, downsample=None, stride=1, bias: bool = True
    ):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        if not last:
            # Apply Instance normalization in first half channels (ratio=0.5)
            self.ibn = IBN(out_channels, ratio=0.5)
        else:
            self.ibn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x.clone()

        x = self.conv1(x)
        x = self.ibn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = residual + x
        out = self.relu(out)

        return out


class Resnet50(nn.Module):
    def __init__(
        self,
        ResBlock: Bottleneck,
        emb_dim: int = 2048,
        num_channels: int = 1,
        num_classes: int = 8858,
        sr: int = 22050,
        hop_lenght: int = 512,
        n_bins=84,
        bins_per_octave=12,
        window="hann",
        compress_ratio: int = 20,
        tempo_factors: Tuple[float, float] = None,
    ) -> None:

        super(Resnet50, self).__init__()
        self.in_channels = 64

        self.cqt = nnAudio.CQT2010v2(
            sr=sr,
            hop_length=hop_lenght,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            window=window,
            output_format="Complex",
            verbose=False,
        )
        self.compress = nn.AvgPool2d((1, compress_ratio))
        self.time_strech = T.TimeStretch(n_freq=n_bins)
        self.tempo_factors = tempo_factors

        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, blocks=3, planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, blocks=4, planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, blocks=6, planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, blocks=3, planes=512, stride=1, last=True)

        self.gem_pool = GeM()

        self.bn_fc = nn.BatchNorm1d(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.fc.weight)

    def _make_layer(self, ResBlock: Bottleneck, blocks: int, planes: int, stride: int = 1, last: bool = False):
        downsample = None
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )
        layers = []
        layers.append(
            ResBlock(in_channels=self.in_channels, out_channels=planes, stride=stride, downsample=downsample, last=last)
        )
        self.in_channels = planes * ResBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResBlock(in_channels=self.in_channels, out_channels=planes, last=last))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        x = self.cqt(x)
        # Time-strech requires complex tensors, that's why cqt function returns complex
        x = torch.view_as_complex(x)
        if self.tempo_factors is not None:
            rate = abs(self.tempo_factors[1] - self.tempo_factors[0]) * torch.rand(1).item() + min(self.tempo_factors)
            strech = (
                abs(1 - rate) > 7e-2
            )  # if the strech ratio is too close to 1 (i.e. 0.93 < ratio < 1.07), skip time streching
            if self.training and strech:
                x = self.time_strech(x, rate)
        # Compress the magnitude of the CQT
        x = self.compress(torch.abs(x))

        # Unsqueeze to simulate 1-channel image
        x = self.conv1(x.unsqueeze(1))
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        f_t = self.gem_pool(x)
        f_t = torch.flatten(f_t, start_dim=1)

        f_c = self.bn_fc(f_t)
        cls = self.fc(f_c)

        return dict(f_t=f_t, f_c=f_c, cls=cls)
