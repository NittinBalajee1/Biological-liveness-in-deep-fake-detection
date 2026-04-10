@echo off
python -m src.make_sample_dataset --output-dir data
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
echo Demo run completed.
