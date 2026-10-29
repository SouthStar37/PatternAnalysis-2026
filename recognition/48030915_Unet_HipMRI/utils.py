# utils.py 
# Things shared by train.py and predict.py that are not part of the model.


# metrics: dice_iou, surface_distances (HD95, ASD), band_masks, slice_profile, leakage_ml
# figures: overlay(ax, image, label), failure_figure, band_boxplot, profile_plot, plot_curves
# io: write_csv(rows, path), save_json(dict, path)
# misc: set_seed(seed), get_device() (cuda / cpu), Timer for inference timing
