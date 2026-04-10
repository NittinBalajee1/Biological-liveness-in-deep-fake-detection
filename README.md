# Deepfake Detection Using Frequency Domain Analysis

This project now includes three pipelines:

- `Classical baseline`: handcrafted FFT features + scikit-learn
- `Frequency CNN`: face crop -> FFT/DCT map -> CNN -> real/fake prediction
- `Biological liveness`: ROI tracking -> green-channel pulse signal -> bandpass + FFT -> authenticated/synthetic

The frequency CNN is the main artifact detector. The liveness module is a complementary defense that checks for a plausible biological pulse in a face video.

## Recommended Datasets

For a proper experiment, use these datasets:

- `FaceForensics++ (c23)`: strong benchmark for training
- `Celeb-DF v2`: harder external benchmark for testing generalization
- `DFDC`: large-scale dataset that helps reduce overfitting

See `DATASETS.md` for dataset guidance and official links.

## Recommended Training Strategy

1. Train on `FaceForensics++ c23`
2. Validate on a held-out split from the same source
3. Test on `Celeb-DF v2` or `DFDC`

This is much better than training and testing on the same easy synthetic subset.

## Project Structure

```text
configs/
data/
models/
outputs/
src/
  config.py
  dataset.py
  evaluate.py
  evaluate_cnn.py
  face_processing.py
  features.py
  frequency_maps.py
  models_cnn.py
  predict.py
  predict_media.py
  prepare_dataset.py
  torch_data.py
  train.py
  train_cnn.py
  utils.py
```

## Pipeline

### A. Dataset Preparation

If your dataset contains raw images or videos in this format:

```text
raw_dataset/
  real/
  fake/
```

prepare cropped-face splits:

```bash
python -m src.prepare_dataset ^
  --input-dir raw_dataset ^
  --output-dir data/frequency ^
  --source-type videos ^
  --image-size 224 ^
  --frames-per-video 12 ^
  --frame-stride 10
```

For already extracted face images:

```bash
python -m src.prepare_dataset ^
  --input-dir raw_dataset ^
  --output-dir data/frequency ^
  --source-type images ^
  --image-size 224
```

For FaceForensics++ extracted frames:

```bash
python -m src.prepare_faceforensics ^
  --ffpp-root path\to\FaceForensics++ ^
  --output-dir data\ffpp_frequency ^
  --compression c23 ^
  --max-frames-per-clip 20
```

For a partial FaceForensics++ download, use a fallback split over available videos:

```bash
python -m src.prepare_faceforensics ^
  --ffpp-root datasets\ffpp_small ^
  --output-dir data\ffpp_frequency ^
  --compression c23 ^
  --max-frames-per-clip 8 ^
  --frame-stride 12 ^
  --fallback-random-split
```

### B. Train Dual-Branch Spatial + Frequency Model

```bash
python -m src.train_cnn --config configs/frequency_cnn.yaml
```

### C. Evaluate

```bash
python -m src.evaluate_cnn --config configs/frequency_cnn.yaml
```

### D. Predict On One Image

```bash
python -m src.predict_media ^
  --config configs/frequency_cnn.yaml ^
  --media path/to/image.jpg ^
  --media-type image
```

### E. Predict On One Video

```bash
python -m src.predict_media ^
  --config configs/frequency_cnn.yaml ^
  --media path/to/video.mp4 ^
  --media-type video
```

### F. Generate Frequency Diagrams For Presentation

```bash
python -m src.visualize_frequency ^
  --config configs/frequency_cnn.yaml ^
  --image path/to/image.jpg ^
  --output outputs/visualizations/frequency_panel.png
```

This creates a presentation-ready panel showing:

- detected face crop
- grayscale input
- FFT heatmap
- DCT spectrum with visible intensity scale

### G. Run Biological Liveness Detection

```bash
python -m streamlit run src\liveness_app.py
```

Or launch with:

```bash
.\run_liveness_app.bat
```

This module:

- tracks forehead and cheek skin regions using MediaPipe Face Mesh
- extracts a green-channel temporal signal
- removes low-frequency drift
- bandpass filters the signal to the human pulse range
- applies FFT to find a physiological heartbeat peak
- returns `AUTHENTICATED` or `SYNTHETIC / INCONCLUSIVE`

## Architecture

The upgraded detector uses:

- spatial branch: pretrained `ResNet18` on RGB face crops
- frequency branch: CNN on stacked `FFT` and `DCT` maps
- attention fusion: learns how much weight to give spatial vs frequency evidence before classification

The liveness detector uses:

- facial ROI tracking from MediaPipe Face Mesh
- green-channel photoplethysmography signal extraction
- temporal detrending and bandpass filtering
- FFT peak detection in the 0.7 Hz to 2.5 Hz pulse band

## Frequency Representation

The CNN can use either:

- `fft`: log magnitude spectrum from the 2D Fast Fourier Transform
- `dct`: 2D Discrete Cosine Transform

Both are supported from config.

## Outputs

- best model checkpoint: `models/dual_branch_best.pt`
- final metrics: `outputs/dual_branch_metrics.json`
- confusion matrix: `outputs/dual_branch_confusion_matrix.png`
- training history: `outputs/dual_branch_history.json`
- frequency diagrams: `outputs/visualizations/*.png`

## Notes

- `MTCNN` face detection is used for preprocessing and optional video inference.
- If a dataset already provides cropped faces, skip face extraction and train directly on those crops.
- Cross-dataset testing is strongly recommended because same-dataset accuracy can look high even when generalization is weak.
