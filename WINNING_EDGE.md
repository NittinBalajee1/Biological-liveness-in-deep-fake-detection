# Winning Edge Plan

If another team is also doing frequency-domain deepfake detection, your edge should come from proof, polish, and scope.

## What Makes This Project Stronger

- Dual-branch model using both spatial and frequency evidence
- FFT and DCT used together instead of only one transform
- Attention fusion that learns how much to trust spatial vs frequency cues
- Presentation-ready visual outputs showing face crop, FFT heatmap, and DCT spectrum
- Video support with frame aggregation

## What Will Impress Faculty

### 1. Ablation Study

Show a table with:

- Spatial only
- FFT only
- DCT only
- Spatial + FFT
- Spatial + FFT + DCT
- Spatial + FFT + DCT + Attention

This proves your design choices are deliberate.

### 2. Cross-Dataset Evaluation

Do not rely only on one benchmark.

- Train on FaceForensics++ c23
- Test on UADFV or Celeb-DF if available

This gives you a much stronger claim than same-dataset accuracy.

### 3. Explainability

Show:

- the input face crop
- grayscale image
- FFT heatmap
- DCT spectrum
- model confidence
- attention weights for spatial vs frequency branches

### 4. Failure Analysis

Include examples where the model gets confused:

- extreme compression
- motion blur
- side pose
- low-resolution face

This makes the project look research-minded, not just demo-minded.

### 5. Demo

Bring a live demo:

- image upload
- video upload
- live prediction
- visual analysis panel

## Best Story For Viva

Say this clearly:

"Most student projects stop at one transform or one classifier. Our system combines spatial evidence, FFT features, and DCT features using an attention fusion mechanism, then explains its decision visually."

That is the line that creates separation.
