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

echo Preparing API startup...
echo Opening the browser automatically when the API is ready...
start "" powershell -NoProfile -Command ^
  "$deadline=(Get-Date).AddSeconds(120);" ^
  "while((Get-Date) -lt $deadline){" ^
  "  try {" ^
  "    $response=Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 3;" ^
  "    if($response.StatusCode -eq 200){ Start-Process 'http://127.0.0.1:8000/'; exit 0 }" ^
  "  } catch {}" ^
  "  Start-Sleep -Seconds 2" ^
  "}" ^
  "Write-Host 'API did not become ready within 120 seconds. Check this terminal for errors.'"

echo Starting API server in this window...
"%~dp0venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
