# 2D vs 3D Improved U-Net for Prostate Segmentation on HipMRI: Does Volumetric Context Fix Boundary Failures?

## Table of Contents

1. [Overview](#1-overview)
2. [Feasibility Review](#2-feasibility-review)
3. [Repository Structure and Usage](#3-repository-structure-and-usage)
4. [Dependencies and Reproducibility](#4-dependencies-and-reproducibility)
5. [Data and Pre-processing](#5-data-and-pre-processing)
6. [Train / Validation / Test Splits](#6-train--validation--test-splits)
7. [Models and Training](#7-models-and-training)
8. [Hyperparameter Tuning (3D)](#8-hyperparameter-tuning-3d)
9. [Results: 2D Baseline vs 3D](#9-results-2d-baseline-vs-3d)
10. [Resource Profiling](#10-resource-profiling)
11. [Error Autopsy](#11-error-autopsy)
12. [Discussion: Answering the Research Dilemma](#12-discussion-answering-the-research-dilemma)
13. [Recommendation to the Project Manager](#13-recommendation-to-the-project-manager)
14. [Future Work](#14-future-work)
15. [References](#15-references)
16. [Artificial Intelligence Usage Disclosure](#16-artificial-intelligence-usage-disclosure)

## 1. Overview

### 1.1 The Problem

In prostate radiotherapy, the prostate, bladder and rectum must be contoured on every planning MRI. Automatic segmentation can draft these contours, but its usual quality measure, a single global Sørensen–Dice score, can hide the failures that matter clinically: a model can score well on the wide central slices while failing at the **base** (top, against the bladder) and **apex** (bottom) of the prostate, or leaking across the millimetre-thin boundary into the **rectal wall** or **bladder**. Leakage into these organs risks radiation proctitis or cystitis; under-contouring risks missing tumour. This project asks whether **3D volumetric context** fixes these boundary failures compared with a 2D slice-by-slice model, or whether it only smooths the easy central slices while costing several times the GPU memory and training time ([Section 12](#12-discussion-answering-the-research-dilemma)).

### 1.2 The Algorithm

Both models are the **Improved U-Net** of Isensee et al. [2]: a U-shaped encoder–decoder in which every encoder level has a residual _context module_ (two convolutions with dropout, added back to their input), the resolution is halved by stride-2 convolutions, the decoder doubles it again with _upsampling modules_ and refines the joined skip connections with _localisation modules_, and _deep supervision_ adds segmentation maps from three decoder levels. Instance normalisation and leaky ReLU are used throughout. The **2D baseline** segments individual coronal slices (HipMRI 2D data); the **3D model** segments whole prostate-centred volumes (HipMRI 3D data). Both are built from the same building blocks, which only swap 2D for 3D layers, so the comparison changes nothing but the spatial dimension (architecture details in [Section 7.1](#71-architecture)).

### 1.3 How It Works

Both models go through the same four steps, so any difference in the results comes from 2D versus 3D context alone.

1. **Split the data by patient.** Each patient's scans all go into one set (training, validation or test), so the models are never tested on a patient they have seen. The 3D split copies the provided 2D split, so both models are tested on the **same patients and scans** (see [Section 6](#6-train--validation--test-splits)).
2. **Prepare the images.** Each scan's intensities are standardised (z-scored) so that scans from different sessions look alike. 2D slices are cropped to 256 × 128; 3D volumes are cropped to a 128 × 128 × 128 box around the prostate, positioned using the training labels only. During training, images are randomly rotated, scaled and brightened so the network learns to handle variation (see [Section 5](#5-data-and-pre-processing)).
3. **Train the network.** The loss combines Dice, which rewards overlap and so stops the network ignoring the tiny prostate (under 0.5% of the image), with cross-entropy, which checks every voxel. Training stops once the prostate score on the validation patients has not improved for a set number of epochs, and the best checkpoint is kept (see [Section 7](#7-models-and-training); the 3D settings were tuned in [Section 8](#8-hyperparameter-tuning-3d)).
4. **Evaluate each test scan as a whole.** The 2D model's slice predictions are stacked back into one block per scan, and both models are then scored per scan with the same code. Both contain the whole prostate (the true prostate volume is identical in the 2D and 3D evaluations of every test scan). Each metric answers one question (results in [Section 9](#9-results-2d-baseline-vs-3d)):
   - **Dice** (0–1): how well does the predicted prostate overlap the true one overall?
   - **Apex / middle / base Dice:** the same, measured separately in the bottom 20%, middle and top 20% of the prostate, which shows _where_ the errors are.
   - **HD95** (mm): how far does the predicted outline stray from the true outline? (The 95th percentile of the boundary distances, so a few outlier points don't dominate.)
   - **Leakage** (ml): how much prostate was predicted inside the rectum or bladder, the errors that would harm healthy organs in radiotherapy?
   - **Pieces:** is the predicted prostate one connected shape, like the real one, or broken into fragments?

<p align="center">
  <img src="figures/3d_best_case.png" width="90%" alt="3D Improved U-Net on a test scan">
  <br><em><b>Figure 1.</b> Example output of the 3D model on a test scan (W041, coronal slice): input, ground truth, prediction, and errors (red = missed prostate, blue = false prostate; dashed lines = edges of the apex and base bands). </em>
</p>

### 1.4 Key Results

**Table 1.** Key test results, mean ± sd over 3 seeds; both models evaluated on the same 9 scans from 3 patients. Bold marks the better model.

|                                       | 2D Improved U-Net (baseline)    | 3D Improved U-Net                   |
| ------------------------------------- | ------------------------------- | ----------------------------------- |
| Prostate Dice                         | 0.862 ± 0.008 (target ≥ 0.75 ✓) | **0.876 ± 0.005** (target ≥ 0.70 ✓) |
| Apex / middle / base Dice             | 0.735 / 0.895 / 0.772           | 0.712 / **0.917** / 0.786           |
| HD95 median                           | 4.02 ± 0.59 mm                  | **3.20 ± 0.14 mm**                  |
| Prostate leaked into rectum / bladder | 0.08 / **0.27** ml              | **0.02** / 0.42 ml                  |
| Separate prostate pieces (truth = 1)  | 1.33                            | **1.00**                            |
| Peak training VRAM                    | **0.68 GB**                     | 3.87 GB (5.7×)                      |

**In short:** 3D context gives a consistent but modest gain (3D beat 2D in all three seed pairs), removes fragmented predictions and greatly reduces rectal leakage, but it does **not** fix the apex, which remains the main failure mode of both models (see [Section 12](#12-discussion-answering-the-research-dilemma)).

## 2. Feasibility Review

**Proposal:** first build a 2D Improved U-Net on the HipMRI 2D slices (Normal difficulty, prostate Dice ≥ 0.75), then a 3D Improved U-Net on the downsampled 3D volumes (Hard difficulty, prostate Dice ≥ 0.70), with the 2D model as its baseline. **The question:** does 3D context remove the prostate base/apex and rectum/bladder boundary errors, or does it only smooth easy central slices at several times the GPU cost?

### 2.1 User Need, Scope and Acceptance Criteria

**User:** a radiotherapy planner or MR physicist who contours the prostate target volume. **Purpose:** an annotation assistant that drafts prostate, bladder and rectum contours for the specialist to review and correct. **Scope:** an offline research prototype.

**Table 2.** Acceptance criteria and how each was tested.

| Acceptance criterion                                                                               | Test                                                                                                  |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 2D Improved U-Net prostate Dice ≥ 0.75                                                             | Per-class Dice and IoU on the held-out test split                                                     |
| 3D Improved U-Net prostate Dice ≥ 0.70                                                             | Per-class Dice and IoU on the held-out test patients                                                  |
| Boundary dilemma: 3D only counts as a gain if it improves the base and apex, not just overall Dice | Prostate Dice in the top and bottom 20% of each prostate vs the central slices, for both models; HD95 |
| Less leakage into the rectum and bladder                                                           | Predicted-prostate volume inside the true rectum / bladder, per scan                                  |
| Resource: 3D training fits one comp3710 GPU                                                        | Peak VRAM and inference time per scan for both models; the 3D : 2D ratio                              |

### 2.2 Model Choice and Course Concepts

**Table 3.** The two planned stages.

|              | Stage 1: 2D Improved U-Net                                              | Stage 2: 3D Improved U-Net                                      |
| ------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------- |
| Difficulty   | Normal (baseline and fallback)                                          | Hard (main model)                                               |
| Data         | `keras_slices_data` 2D slices, provided train / validate / test folders | `semantic_MRs` + `semantic_labels_only`, patient-level split    |
| Input        | 256 × 128 slices, z-scored per scan                                     | Fixed 128 × 128 × 128 crop around the pelvis, z-scored per scan |
| Convolutions | 3 × 3                                                                   | 3 × 3 × 3 (volumetric residual units)                           |

Both follow the Improved U-Net: residual context modules, stride-2 downsampling, instance normalisation, leaky ReLU, localisation modules and deep supervision, trained with Dice + cross-entropy loss.

- **Encoder–decoder with skip connections** keeps the fine detail needed for millimetre-scale boundaries.
- **Residual units and deep supervision** help gradients reach early layers in a deep network trained on few patients.
- **Dice + cross-entropy loss** stops the model ignoring the prostate, which is under 2% of the scan.
- **Augmentation and early stopping** (small rotations, scaling, intensity jitter) limit overfitting.

The 2D and 3D models share loss, metrics and test patients, so differences come from spatial context alone.

### 2.3 Preliminary Evidence: Initial Data Audit

**Table 4.** Initial data audit of the 2D slices and 3D volumes.

| Check                          | 2D slices                                                             | 3D volumes                                                                                    |
| ------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Files                          | 11,460 train / 660 validate / 540 test slices (211 scans × 60 slices) | 211 volumes from 38 patients (1–8 scans each)                                                 |
| Shape and spacing              | 256 × 128 (one scan's 60 slices are 256 × 144)                        | 256 × 256 × 128 (one is × 144); about 1.41–1.88 mm in-plane × 1.56 mm slices; orientation LPS |
| Labels                         | 0–5 in every split (0 = background … 5 = prostate)                    | 0–5; prostate 0.10%, rectum 0.14%, bladder 0.57%, bone 3.4%, body 35.5% of voxels             |
| Prostate share                 | 0.43% / 0.50% / 0.34% of pixels; 36–39% of slices contain prostate    | 0.023–0.311% of voxels (median 0.086%); 8.4–99.4 ml (median 31 ml)                            |
| Patients shared between splits | None: the split is already by case (31 / 4 / 3 patients)              | Split by patient ID                                                                           |

### 2.4 Risks, Budget and Fallback

**Table 5.** Risks identified before training, and the planned mitigation for each.

| Risk                                                       | Mitigation                                                                                                      |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 3D model runs out of GPU memory                            | 128³ pelvic crop, batch size 1–2, mixed precision, fewer base filters                                           |
| Prostate ignored because it is ~0.1% of voxels             | Dice + cross-entropy loss; oversample slices and crops that contain prostate                                    |
| Overfitting: only 31 training patients                     | Augmentation; early stopping on validation Dice                                                                 |
| 2D case IDs do not map to 3D patients                      | Check week counts per case against each 3D patient; if they don't match, use an independent patient-level split |
| Test set of 3 patients is too small to separate the models | Per-scan and per-patient results with their spread; report differences inside the spread as inconclusive        |

**Compute budget:** about 25 GPU-hours on the comp3710 partition (5 for 2D, 15 for 3D, 5 for evaluation and reruns), refined after timing the first epochs. **Fallback:** if the 3D model cannot fit in memory or reach Dice 0.70, submit the 2D Improved U-Net (Normal).

## 3. Repository Structure and Usage

### 3.1 Files

### 3.2 Quick Start

### 3.3 Running on Rangpur

### 3.4 Outputs

### 3.5 Testing

## 4. Dependencies and Reproducibility

### 4.1 Dependencies

### 4.2 Hardware

### 4.3 Reproducibility

## 5. Data and Pre-processing

### 5.1 Dataset

### 5.2 Data Audit

### 5.3 2D Pre-processing

### 5.4 3D Pre-processing

### 5.5 Data Augmentation

## 6. Train / Validation / Test Splits

### 6.1 2D Split

### 6.2 3D Split Matched to 2D

### 6.3 Verifying the Patient Match

### 6.4 Split Sizes and Limitations

## 7. Models and Training

### 7.1 Architecture

### 7.2 Choice of Baseline

### 7.3 Loss Function

### 7.4 Training Settings

### 7.5 Training Curves

## 8. Hyperparameter Tuning (3D)

## 9. Results: 2D Baseline vs 3D

### 9.1 Overall Results

### 9.2 Per-class Dice and IoU

### 9.3 Paired Per-scan Comparison

### 9.4 Per-patient Results

### 9.5 Where Along the Prostate Each Model Fails

## 10. Resource Profiling

## 11. Error Autopsy

### 11.1 Case A: 3D Over-segmentation Past the Apex (B040)

### 11.2 Case B: 2D Apex Miss (W041)

### 11.3 Case C: Large Prostate Under-segmentation (K042)

### 11.4 Case D: Fragmented 2D Predictions

### 11.5 Case E: Rectal Interface Leakage

## 12. Discussion: Answering the Research Dilemma

### 12.1 Where 3D Context Helps

### 12.2 Where It Breaks Down

### 12.3 Why: Coronal Slices and Missing Front-to-Back Context

### 12.4 Limitations

## 13. Recommendation to the Project Manager

## 14. Future Work

## 15. References

## 16. Artificial Intelligence Usage Disclosure
