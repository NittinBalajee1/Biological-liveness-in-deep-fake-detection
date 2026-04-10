@echo off
call .venv\Scripts\activate.bat
python -m streamlit run src\liveness_app.py
