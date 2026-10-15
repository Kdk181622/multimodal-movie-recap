@echo off
title Movie Recap v2 - re-run from the summary stage (about 60-75 minutes)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_XET=1
set LOG=outputs\experiments\run_v2.log
set ST=outputs\experiments\run_v2_status.txt
echo START %date% %time% > %ST%
echo.
echo  Multimodal Movie Recap - final v2 run
echo  Steps: summary (LLM) - peaks - recap render - evaluation - tests - notebook
echo  Leave this window open. Progress is written to %ST%
echo.
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
  echo Starting Ollama...
  start "" /min ollama serve
  timeout /t 10 >nul
)
curl -s http://localhost:11434/api/tags > outputs\experiments\ollama_tags.json 2>&1
if errorlevel 1 (
  echo OLLAMA_DOWN >> %ST%
  echo Ollama is not reachable - aborting so the summary does not fall back to TextRank.
  pause
  exit /b 1
)
findstr /c:"qwen2.5:7b" outputs\experiments\ollama_tags.json >nul
if errorlevel 1 (
  echo MODEL_MISSING >> %ST%
  echo The model qwen2.5:7b is not installed. Run:  ollama pull qwen2.5:7b   then start this file again.
  pause
  exit /b 1
)
rem Pick the Python that has the project libraries (the first run used Python 3.10, which lacks them)
set PY=
set CHECK=import sacrebleu, faster_whisper, sentence_transformers, cv2, pytest, nbconvert
py -3.10 -c "%CHECK%" >nul 2>&1 && set "PY=py -3.10"
if not defined PY py -3.14 -c "%CHECK%" >nul 2>&1 && set "PY=py -3.14"
if not defined PY py -3.13 -c "%CHECK%" >nul 2>&1 && set "PY=py -3.13"
if not defined PY py -3.12 -c "%CHECK%" >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3 -c "%CHECK%" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "%CHECK%" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo PYTHON_ENV_MISSING >> %ST%
  echo No Python with the project libraries was found. Run:  py -3.14 -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Using %PY%
%PY% --version >> %ST% 2>&1
echo [1/3] Pipeline (summary onward) ... about 60-70 min
echo PIPELINE_RUNNING %time% >> %ST%
%PY% run_pipeline.py --fresh summary.json peaks.json > %LOG% 2>&1
echo PIPELINE_EXIT %errorlevel% %time% >> %ST%
copy /y outputs\detour_1945\summary.json outputs\experiments\summary_qwen2.5-7b_v2.json >nul
copy /y outputs\detour_1945\evaluation.json outputs\experiments\evaluation_qwen2.5-7b_v2.json >nul
echo [2/3] Unit tests ...
%PY% -m pytest -q tests > outputs\experiments\pytest_v2.log 2>&1
echo PYTEST_EXIT %errorlevel% %time% >> %ST%
echo [3/3] Build and execute the notebook ... about 15 min
%PY% notebooks\build_notebook.py >> %LOG% 2>&1
%PY% -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\Movie_Recap_Walkthrough.ipynb > outputs\experiments\notebook_v2.log 2>&1
echo NOTEBOOK_EXIT %errorlevel% %time% >> %ST%
echo END %date% %time% >> %ST%
echo.
echo  Done. Tell Claude "run finished". This window closes in 60 seconds.
timeout /t 60
