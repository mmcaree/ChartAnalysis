@echo off
cd /d "%~dp0"
echo Starting Chart Analysis UI...
start "Server Chart UI" powershell -NoExit -Command ".\chartEnv\Scripts\Activate.ps1; streamlit run ui\streamlit_app.py"