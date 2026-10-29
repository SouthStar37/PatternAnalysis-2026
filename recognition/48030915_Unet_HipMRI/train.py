
# train.py
# Training, validation, testing and saving. Imports the model from modules.py and the data
# from dataset.py (spec rule). Plots losses and metrics during training.

# Run: python train.py --dim 2d   |   python train.py --dim 3d


# SHARED 

# get_args()  (argparse)
#   --dim 2d|3d, --model improved|unet, --epochs, --batch-size, --lr, --base-filters,
#   --dropout, --dice-weight, --ce-weight, --patience, --no-augment, --out, --seed,
#   --max-scans / --max-volumes (small subset for quick tests), --resume (3D)
#   - defaults differ per dim: 2D batch 16 / 60 epochs; 3D batch 2 / 150 epochs

# set_seed(seed): random, numpy, torch: so runs are reproducible 

# train_one_epoch(model, loader, loss_fn, optimiser, scaler, device, augment_fn) -> mean loss
#   - for each batch: to GPU -> augment -> forward -> loss -> backward -> optimiser step
#   - mixed precision (torch.autocast + GradScaler): faster, about half the memory

# validate(model, loader, loss_fn) -> val loss, per-class Dice
#   - model.eval(), torch.no_grad(), argmax -> SegMetrics

# plot_curves(history, path)
#   - left: train vs validation loss; right: validation Dice per class (prostate in bold,
#     dashed line at the target). Save every epoch so you can look while it trains.

# main()
#   1. data: build datasets with the dataset.py functions; check_no_leakage; print summaries
#   2. model, loss (DiceCELoss), optimiser (AdamW), learning-rate schedule (cosine)
#   3. for each epoch: train_one_epoch -> validate -> scheduler step
#        - save last.pt every epoch; save best.pt when validation PROSTATE Dice improves
#          (store the settings inside the checkpoint, e.g. dim, model, base_filters, crop)
#        - log a row to history.csv: losses, Dice per class, lr, epoch time, peak VRAM
#        - plot_curves
#        - early stopping: stop after `patience` epochs without improvement
#   4. TEST: reload best.pt, run on the test split, print + save per-class Dice and IoU


# 2D Improved U-Net

#   - datasets: HipMRISlices("train"/"validate"/"test") from the provided folders
#   - DataLoader with the prostate oversampling sampler; batch 16
#   - augment_2d; target: test prostate Dice >= 0.75


# 3D Improved U-Net

#   - split_volumes(match2d) -> crop centre from TRAINING labels -> HipMRIVolumes
#     (train=True for random shifted crops; train=False for validation/test)
#   - batch 2 (memory); validation batch 1
#   - augment_3d; target: test prostate Dice >= 0.70
#   - --resume from last.pt (save optimiser/scheduler/history in it) for Rangpur time limits
#   - store crop centre, crop size and the split's volume IDs in the checkpoint for predict.py



