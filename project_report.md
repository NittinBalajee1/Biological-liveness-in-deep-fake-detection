# Project Summary

## Title

Deepfake Detection Using Frequency Domain Analysis

## Objective

The goal of this project is to classify images as `real` or `fake` by detecting frequency-domain anomalies that often appear in synthetic or manipulated media.

## Core Idea

Deepfake generators and image editing pipelines can leave artifacts that are not always obvious in the spatial domain but become more visible in the frequency domain. By applying the Fast Fourier Transform (FFT), we can inspect spectral energy patterns and use them for classification.

## Methodology

1. Load image or video data from benchmark datasets.
2. Extract frames from videos when required.
3. Detect and crop faces using MTCNN.
4. Resize face crops to a fixed size.
5. Build a spatial-domain RGB face representation.
6. Build two frequency-domain representations using FFT and DCT.
7. Train a dual-branch model that fuses spatial features with frequency features for binary classification.
8. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.

## Tools and Libraries

- Python
- NumPy
- OpenCV
- scikit-learn
- Matplotlib
- Seaborn
- PyYAML
- Joblib

## Files Included

- `src/prepare_dataset.py`: frame extraction and face-crop preparation
- `src/face_processing.py`: MTCNN-based face detection helpers
- `src/frequency_maps.py`: spatial preprocessing plus FFT and DCT map generation
- `src/models_cnn.py`: dual-branch spatial-frequency architecture
- `src/train_cnn.py`: dual-branch training with threshold tuning
- `src/evaluate_cnn.py`: upgraded testing and metrics generation
- `src/predict_media.py`: prediction for a single image or video
- `configs/frequency_cnn.yaml`: CNN training configuration
- `DATASETS.md`: recommended benchmark datasets and usage guidance

## Expected Outcome

The trained model learns to combine spatial inconsistencies with spectral irregularities associated with deepfakes. The project now provides a stronger academic pipeline built around spatial-frequency fusion, and it can be extended further with attention modules or cross-dataset evaluation on larger benchmarks.
