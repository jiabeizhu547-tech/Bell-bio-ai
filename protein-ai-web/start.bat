@echo off
chcp 65001 >nul
echo ================================================
echo   Protein AI - Local Server
echo ================================================
echo.
echo  1. Starting server...
echo  2. Open browser to http://localhost:8765
echo  3. First prediction takes ~60s (loading models)
echo.
echo  Press Ctrl+C to stop
echo ================================================
echo.

set HF_ENDPOINT=https://hf-mirror.com
set PYTHONIOENCODING=utf-8

"D:\code_test\.venv\Scripts\python.exe" "D:\code_test\protein-ai-web\server.py"

pause
