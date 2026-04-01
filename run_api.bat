@echo off
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Virtual environment not found at venv\Scripts\python.exe
  echo Create it first or update this launcher.
  pause
  exit /b 1
)

if not exist "venv\Scripts\uvicorn.exe" (
  echo Installing FastAPI and Uvicorn...
  call "venv\Scripts\pip.exe" install fastapi uvicorn
  if errorlevel 1 (
    echo Failed to install FastAPI/Uvicorn.
    pause
    exit /b 1
  )
)

echo Starting API server in a new window...
start "Fake News API" cmd /k "cd /d %~dp0 && venv\Scripts\python.exe -m uvicorn api:app --reload"

echo Waiting for server startup...
timeout /t 4 /nobreak >nul

start "" http://127.0.0.1:8000/
