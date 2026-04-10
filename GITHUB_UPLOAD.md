# GitHub Upload Guide

This project is ready to upload as source code. Do not upload local datasets, trained checkpoints, virtual environments, or generated outputs.

## What Is Included

- `src/` - all Python source code
- `templates/` - Flask web UI template
- `configs/` - training and liveness configuration files
- `requirements.txt` - Python dependencies
- `README.md` - project overview and run commands
- `DATASETS.md` - dataset notes
- `project_report.md` - report draft
- `WINNING_EDGE.md` - presentation strategy
- `run_*.bat` - Windows launch scripts
- `download_ffpp.py` - official FaceForensics++ downloader script you were given

## What Is Ignored

These are intentionally excluded by `.gitignore`:

- `.venv/`
- `data/`
- `datasets/`
- `raw_dataset/`
- `outputs/`
- `models/`
- `FaceForensics-master/`
- video files
- trained model checkpoints
- zip archives

This keeps the repository small and legal to share.

## Create The Git Repository

Run these commands from the project folder:

```powershell
cd "E:\Deepfake analysis"
git init
git add .
git status
git commit -m "Initial deepfake and liveness detection project"
```

## Push To GitHub

Create an empty GitHub repository first. Then run:

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

Replace:

```text
YOUR_USERNAME
YOUR_REPO_NAME
```

with your actual GitHub details.

## If Git Says Files Are Too Large

Check what is being staged:

```powershell
git status
```

If a dataset/model/output file appears, remove it from staging:

```powershell
git rm --cached path\to\large_file
```

Then commit again.

## How Others Run The Project

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.liveness_v2_web
```

Open:

```text
http://127.0.0.1:5050
```

## Notes For Evaluators

Datasets and checkpoints are not included due to size and access restrictions. The repository contains the full implementation, configuration, and web demo code. Users can prepare datasets using the provided scripts and run the models locally.
