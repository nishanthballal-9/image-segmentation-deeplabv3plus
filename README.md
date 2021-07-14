# image-segmentation-deeplabv3plus
Image Segmentation is the task of clustering parts of an image together which belong to the same object class. In this project the goal is to segment various ROI from whole slide images of digitized H&amp;E stained biopsies.

Tensorflow implementation of DeepLabV3Plus for Image Segmentation on Biomedical images.

## Dependencies

Tensorflow 2.3.0

## Problem and Dataset

To perform semantic segmentation on images of H&E stained biopsies to cluster three categories.

Dataset: Consists of Whole Slide Images (WSI)

Train: 5 WSI
Validation: 2 WSI

Whole Slide Images can be very large and have as much as 50k x 50k pixels. It becomes important to take all the parts of the image for training and validation.

## Solution

1) Generate patches for training and validation. Use the script generate_patches.py. Patches can be generated with or without overlap, but training data patches generated with overallping strides perform better.
2) Use Rotation, Flipping and Color Augmentation with a condition in the training pipeline.
3) Training DeepLabV3Plus modeland validate on validation patch (with or without overlap). Metrics to validation Dice Score and mean IoU score can be used.

## Installation

1) Clone this repo

2) Install the required python packages

```
pip install tensorflow-gpu==2.3.0
pip install opencv-python
pip install -r image-segmentationutilities/requirements.txt
```

## Model

ResNet50 used as backbone for DeepLabV3Plus model. To make changes in terms of backbone, make changes to DeeplabV3Plus() function in model.py.

Other good options of backbone:

1) Xception
2) ResNet101
3) Inception-ResNet-v2

## Training

```
python deeplab_v3_plus.py
```

Hyperparameters:
1) Learning Rate for optimizer
2) Patience for Early Stopping 
3) Patience and Factor for ReduceLR
4) Training Steps per Epoch
5) Batch Size
6) Patch Size

Change them according to your requirements and input patch size.

## Generating Whole Slide Image Output

Image Stitching done on validation images using stitching.py

```
python stitching.py
```

Note: Tested on validation images without overlap.

## Evaluation

Evaluate overall and class wise dice scores using mean_dice_evaluation.ipynb

Note: Evaluation can be done image wise with the above notebook.

## Sample Result

Original Mask:
![d0cf594c5106fb84e894c0b12013f367_mask](https://github.com/nishanthballal-9/image-segmentation-deeplabv3plus/blob/main/Valid_Mask/d0cf594c5106fb84e894c0b12013f367_mask.png)

Predicted Mask:
![d0cf594c5106fb84e894c0b12013f367](https://github.com/nishanthballal-9/image-segmentation-deeplabv3plus/blob/main/Val_Results_reconstructed/d0cf594c5106fb84e894c0b12013f367.png)