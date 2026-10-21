# PVT-FDSC-Net — Transformer-Based Polyp Segmentation

PVT-FDSC-Net is a deep learning research project for medical image segmentation, focused on polyp segmentation in colonoscopy images. The architecture integrates a Pyramid Vision Transformer (PVTv2-B2) backbone, dual decoder branches, and a Frequency Decomposed Selective Concatenation (FDSC) module for combining complementary feature representations in the frequency domain.

## Features

- **Transformer-Based Backbone:** Use PVTv2-B2 to extract image features.
- **Dual Decoder Architecture:** Combine global semantic aggregation and local structural extraction.
- **Frequency-Domain Feature Fusion:** Implement FDSC using Fourier transforms to combine feature representations.
- **Model Training and Evaluation:** Run experiments on five polyp-segmentation datasets.
- **Performance Evaluation:** Reported experimental results include a mean Dice coefficient of **0.9068**, accuracy of **0.9721**, and mean Intersection over Union (mIoU) of **0.8487**, with approximately **48–50 FPS** inference throughput.

## Tech Stack

**Programming Language:** Python

**Deep Learning:** PyTorch

**Model Architecture:** PVTv2-B2, dual decoders, Frequency Decomposed Selective Concatenation (FDSC)

**Image Processing and Analysis:** OpenCV, NumPy, Matplotlib

**GPU Computing:** CUDA

## Research Status

The associated research manuscript, *“Pyramid Vision Transformer Integrated with Frequency Decomposed Selective Concatenation for Polyp Segmentation,”* has been submitted and is currently under review.
