# Proper Datasets For This Project

## Best Choice

If your goal is a strong academic project instead of a toy demo, use this setup:

1. `FaceForensics++ c23` for training
2. `Celeb-DF v2` for external testing
3. `DFDC` if you want more scale

## Why These Datasets

### FaceForensics++

- One of the most widely used deepfake detection benchmarks
- Good for controlled training and ablation studies
- The `c23` compressed version is commonly used in the literature because it is more realistic than raw-only evaluation
- Official access is still distributed through the request form linked from the FaceForensics++ repository README, rather than a stable public direct download

### Celeb-DF v2

- Considerably harder than easier benchmark splits
- Better for checking whether your model really generalizes
- Useful as an external test set after training on FaceForensics++

### DFDC

- Large and diverse
- Helpful for reducing overfitting
- Good if you want more robust large-scale training

## Practical Recommendation

Use one of these two setups:

### Setup A: Balanced academic setup

- Train: `FaceForensics++ c23`
- Validate: held-out FaceForensics++ split
- Test: `Celeb-DF v2`

### Setup B: Larger-scale setup

- Train: `DFDC`
- Validate: held-out DFDC split
- Test: `Celeb-DF v2`

## Official / Primary Sources

- FaceForensics++ repository and dataset instructions: <https://github.com/ondyari/FaceForensics>
- Celeb-DF project page: <https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html>
- DeepfakeBench benchmark repository: <https://github.com/SCLBD/DeepfakeBench>
- DFDC Kaggle challenge page: <https://www.kaggle.com/c/deepfake-detection-challenge>

## Important Notes

- Some datasets require request forms or acceptance of terms before download.
- If you use DeepfakeBench preprocessing outputs, you can skip most raw-video preprocessing because the faces are already cropped.
- A model that only performs well on the same dataset it was trained on is often overfit.
- FaceForensics++ is the better next step if you want a more credible accuracy number than UADFV.
