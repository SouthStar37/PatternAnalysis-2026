# modules.py
# Reference: Isensee et al. 2018 (Improved U-Net), Ronneberger et al. 2015 (U-Net).

# U-Net 

#   ENCODER (contracting path)  : repeated "process features, then halve the resolution"
#   BOTTLENECK                  : lowest resolution, most channels
#   DECODER (expanding path)    : repeated "double the resolution, join the matching encoder
#                                 output (skip connection), process features"
#   OUTPUT HEAD                 : 1x1 conv -> one score map per class (logits)


# 2D Improved U-Net 
# -	Use HipMRI 2D slices. Ref: [2]. Target Dice ≥ 0.75 on prostate label on test set

# ---- building blocks  -----

# ConvNormAct(cin, cout, k=3, stride=1)
#   - Conv2d -> InstanceNorm2d -> LeakyReLU(0.01); used by the decoder modules

# ContextModule(channels, dropout=0.3) [encoder block]
#   - pre-activation residual unit:
#     InstanceNorm -> LeakyReLU -> 3x3 conv -> Dropout -> InstanceNorm -> LeakyReLU -> 3x3 conv
#   - output = input + block(input)   (the "residual" connection)

# downsampling: plain 3x3 conv with stride 2  [replaces max-pooling; doubles channels]

# UpsamplingModule(cin, cout) [decoder: going up]
#   - Upsample x2 -> 3x3 conv (halves channels) -> norm -> act

# LocalisationModule(cin, cout) [decoder: after the skip concatenation]
#   - 3x3 conv (keeps channels) -> 1x1 conv (halves channels)

# segmentation layers [deep supervision]
#   - 1x1 conv to n_classes on three decoder levels; upsample the lower one and add,
#     level by level, into the final output

# ---- the network -----

# class ImprovedUNet2D(in_channels=1, n_classes=6, base_filters=16, depth=5, dropout=0.3)
#   __init__: channels per level = 16, 32, 64, 128, 256
#   forward(x):
#     1. encoder: for each level: (stride-2) conv -> ContextModule; keep each output as a skip
#     2. decoder: for each level up: UpsamplingModule -> concat skip -> LocalisationModule
#     3. deep supervision: sum the upsampled segmentation layers -> logits (B, 6, H, W)



# 3D Improved U-Net
# -	Use downsampled HipMRI 3D data. Ref: [2]. Volumetric residual units and deep supervision; target Dice ≥ 0.7 on test set.

# Same blocks with 3D layers: Conv3d, InstanceNorm3d, Dropout3d, trilinear upsampling.
# Two ways to write it:
#   a) copy the 2D classes and swap the layer types (simplest first)
#   b) refactor ONE ImprovedUNet(dims=2|3) that picks layer types from a lookup table
#      -> a good commit later: "refactor: dimension-generic Improved U-Net"
# class ImprovedUNet3D(...)
#   ✓ shape check with (1, 1, 128, 128, 128); input sizes must be divisible by 16
#   ✓ GPU memory test: one forward + backward at batch 1 and 2 (peak VRAM -> criterion A5)

# ============================================================================
# Loss and metric (shared by 2D and 3D; write them to work for any number of dimensions)
# ============================================================================
# class DiceCELoss(n_classes, dice_weight=1, ce_weight=1)
#   - cross-entropy + (1 - mean soft Dice of the foreground classes present in the batch)
#   - one-hot the labels here with F.one_hot(...).movedim(-1, 1)
#   - why Dice: the prostate is ~0.1-0.4% of pixels; cross-entropy alone lets the model ignore it
#   ✓ loss is a single number and .backward() works
# class SegMetrics
#   - update(pred_labels, true_labels): accumulate per-class confusion counts (torch.bincount)
#   - dice(), iou(): per-class tensors pooled over the whole validation/test set
#   ✓ metrics of a label map against itself = 1.0 for every class
# build_model(name, dims) and count_parameters(model): small helpers used by train/predict
