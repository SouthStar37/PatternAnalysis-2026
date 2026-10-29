"""
modules.py - Model components, loss and metric. Pure PyTorch (no NumPy, as the spec requires).

References:
    Isensee et al. 2018, "Brain Tumor Segmentation and Radiomics Survival Prediction:
        Contribution to the BRATS 2017 Challenge" (the Improved U-Net).
    Ronneberger et al. 2015, "U-Net: Convolutional Networks for Biomedical Image Segmentation".

U-Net layout:
    ENCODER     repeated "process features, then halve the resolution"
    BOTTLENECK  lowest resolution, most channels
    DECODER     repeated "double the resolution, join the matching encoder output
                (skip connection), process features"
    OUTPUT      1x1 conv -> one score map per class (logits)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2D Improved U-Net 
# -	Use HipMRI 2D slices. Ref: [2]. Target Dice ≥ 0.75 on prostate label on test set

# ---- building blocks  -----

# ---- the network -----

class ConvBlock(nn.Module):
    """Tutorial's conv_block(): two 3x3 convs with ReLU, optional dropout, optional 2x2 max-pool.

    Returns (next_layer, skip_connection) like the tutorial:
        skip_connection = output before pooling (sent across to the decoder)
        next_layer      = pooled output (sent down to the next encoder level)
    """

    def __init__(self, in_channels, n_filters, dropout_prob=0.0, max_pooling=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, 3, padding=1)    # padding=1 = Keras padding='same'
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2) if max_pooling else nn.Identity()
        for conv in (self.conv1, self.conv2):                          # = kernel_initializer='he_normal'
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")

    def forward(self, x):
        conv = self.relu(self.conv1(x))
        conv = self.relu(self.conv2(conv))
        conv = self.dropout(conv)
        skip_connection = conv
        next_layer = self.pool(conv)
        return next_layer, skip_connection


class UpsamplingBlock(nn.Module):
    """Tutorial's upsampling_block(): transposed conv up -> concatenate the skip -> two 3x3 convs with ReLU."""

    def __init__(self, in_channels, skip_channels, n_filters):
        super().__init__()
        # Keras Conv2DTranspose(n_filters, 3, strides=2, padding='same') doubles H and W;
        # in PyTorch that needs padding=1 and output_padding=1
        self.up = nn.ConvTranspose2d(in_channels, n_filters, 3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(n_filters + skip_channels, n_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        for conv in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")

    def forward(self, expansive_input, contractive_input):
        up = self.up(expansive_input)
        merge = torch.cat([up, contractive_input], dim=1)              # = concatenate(axis=3) in Keras
        conv = self.relu(self.conv1(merge))
        return self.relu(self.conv2(conv))


class UNet2D(nn.Module):
    """Tutorial's unet_model(): 4 encoder blocks + bottleneck, 4 upsampling blocks, 1x1 output conv."""

    def __init__(self, in_channels=1, n_classes=6, n_filters=32):
        super().__init__()
        self.c1 = ConvBlock(in_channels, n_filters)
        self.c2 = ConvBlock(n_filters, n_filters * 2)
        self.c3 = ConvBlock(n_filters * 2, n_filters * 4)
        self.c4 = ConvBlock(n_filters * 4, n_filters * 8, dropout_prob=0.3)
        self.c5 = ConvBlock(n_filters * 8, n_filters * 16, dropout_prob=0.3, max_pooling=False)   # bottleneck
        self.u6 = UpsamplingBlock(n_filters * 16, n_filters * 8, n_filters * 8)
        self.u7 = UpsamplingBlock(n_filters * 8, n_filters * 4, n_filters * 4)
        self.u8 = UpsamplingBlock(n_filters * 4, n_filters * 2, n_filters * 2)
        self.u9 = UpsamplingBlock(n_filters * 2, n_filters, n_filters)
        self.output_conv = nn.Conv2d(n_filters, n_classes, 1)

    def forward(self, x):
        x1, s1 = self.c1(x)          # tutorial: c1 = conv_block(inputs, n_filters)
        x2, s2 = self.c2(x1)
        x3, s3 = self.c3(x2)
        x4, s4 = self.c4(x3)
        x5, _ = self.c5(x4)          # bottleneck, no pooling
        u = self.u6(x5, s4)          # tutorial: u6 = upsampling_block(c5[0], c4[1], n_filters*8)
        u = self.u7(u, s3)
        u = self.u8(u, s2)
        u = self.u9(u, s1)
        return self.output_conv(u)   # logits; the tutorial's softmax is done inside the loss instead


# 3D Improved U-Net
# -	Use downsampled HipMRI 3D data. Ref: [2]. Volumetric residual units and deep supervision; target Dice ≥ 0.7 on test set.

# Same blocks with 3D layers: Conv3d, InstanceNorm3d, Dropout3d, trilinear upsampling.

# class ImprovedUNet3D(...)

# Loss and metric (shared by 2D and 3D; write them to work for any number of dimensions)

# class DiceCELoss(n_classes, dice_weight=1, ce_weight=1)
#   - cross-entropy + (1 - mean soft Dice of the foreground classes present in the batch)
#   - one-hot the labels here with F.one_hot(...).movedim(-1, 1)
#   - why Dice: the prostate is ~0.1-0.4% of pixels; cross-entropy alone lets the model ignore it

# class SegMetrics
#   - update(pred_labels, true_labels): accumulate per-class confusion counts (torch.bincount)
#   - dice(), iou(): per-class tensors pooled over the whole validation/test set

# build_model(name, dims) and count_parameters(model): small helpers used by train/predict
