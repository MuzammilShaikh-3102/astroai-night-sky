@echo off
cd /d %~dp0
where conda >nul 2>&1
if %errorlevel%==0 (
  echo Activating existing conda environment if present...
  call conda activate astroai 2>nul
) else (
  echo Conda was not found on PATH. Activate your Python environment manually.
)
python -m streamlit run app.py --server.address 0.0.0.0
pause
