@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe train.py --data-dir library --output-dir checkpoints --config configs\default.yaml