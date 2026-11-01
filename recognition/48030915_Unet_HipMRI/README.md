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

### 1.2 The Algorithm

### 1.3 How It Works

### 1.4 Key Results

## 2. Feasibility Review

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
