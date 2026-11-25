"""Standard 2D U-Net (Ronneberger et al., 2015) written from scratch; random initialisation, no pretrained weights.

Widths 32/64/128/256/512, four max-pool stages, each block = two (3x3 conv -> BatchNorm -> ReLU), transposed-conv
upsampling with skip concatenation, 1x1 head producing one logit channel at 256x256. 7,763,041 parameters at base 32.
"""

import torch
import torch.nn as nn

EXPECTED_PARAMETER_COUNT = 7_763_041


def double_convolution_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    # Attribute names (downs, ups, decs, head) are kept from the original study so checkpoints stay compatible.
    def __init__(self, in_channels=3, base_channels=32, depth=4):
        super().__init__()
        widths = [base_channels * 2**i for i in range(depth + 1)]
        self.downs = nn.ModuleList(
            [double_convolution_block(in_channels, widths[0])]
            + [double_convolution_block(widths[i], widths[i + 1]) for i in range(depth)]
        )
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(depth))]
        )
        self.decs = nn.ModuleList([double_convolution_block(widths[i] * 2, widths[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, encoder_block in enumerate(self.downs):
            x = encoder_block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for upsample, decoder_block in zip(self.ups, self.decs, strict=True):
            x = decoder_block(torch.cat([upsample(x), skips.pop()], 1))
        return self.head(x)  # logits, same height and width as the input


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
