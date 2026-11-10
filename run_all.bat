@echo off
title Vietnamese Fake News Detection System (All-in-One)
echo ======================================================
echo   DANG KHOI CHAY HE THONG (MOI THU TRONG 1 CUA SO)
echo ======================================================
echo.

:: 0. Kiem tra va cai dat moi truong
echo [+] Dang kiem tra Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python chua duoc cai dat. Vui long cai dat Python truoc.
    pause
    exit /b
)

echo [+] Dang cai dat/cap nhat thu vien Python...
call python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Co loi khi cai dat thu vien Python.
    pause
)

echo [+] Dang kiem tra Node.js/NPM...
call npm --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js/NPM chua duoc cai dat. Vui long cai dat truoc khi chay Frontend.
    pause
    exit /b
)

echo [+] Dang cai dat/cap nhat thu vien Frontend (npm install)...
cd frontend
call npm install
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Co loi khi cai dat thu vien Frontend.
    cd ..
    pause
) else (
    cd ..
)

echo [+] Dang kiem tra FastAPI...
python -c "import fastapi" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] FastAPI chua duoc cai dat. Dang thu cai dat lai...
    call python -m pip install fastapi uvicorn
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Khong the tu dong cai dat FastAPI. Vui long kiem tra ket noi mang.
        pause
    )
)

echo [+] Dang kiem tra Vite...
if not exist "frontend\node_modules\.bin\vite.cmd" (
    echo [ERROR] Vite chua duoc cai dat. Dang thu cai dat lai...
    cd frontend && call npm install vite && cd ..
)

:: 1. Khoi chay cac Backend trong nen (background)
echo [+] Dang khoi chay Backend Du doan (FastAPI: 8000)...
start /b python main.py

echo [+] Dang khoi chay Backend Huan luyen (FastAPI: 8001)...
start /b python training_api.py

:: 2. Doi mot chut de cac Backend san sang, sau do mo trinh duyet
echo [+] Dang chuan bi mo trinh duyet...
timeout /t 7 /nobreak > nul
start "" "http://localhost:5173"

:: 3. Khoi chay Frontend trong cua so nay (foreground)
echo [+] Dang khoi chay Frontend (React + Vite)...
echo.
echo LUU Y: Nhan Ctrl+C de dung ca hai dich vu.
echo.
cd frontend
call npm run dev

:: Neu nguoi dung thoat khoi npm run dev, script se dung lai o day
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Frontend bi dung dot ngot.
    pause
)
cd ..
pause