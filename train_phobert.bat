@echo off
setlocal
title Quy trinh Training PhoBERT - Fake News Detection
echo ======================================================
echo    QUY TRINH HUAN LUYEN MO HINH PHOBERT (DEEP LEARNING)
echo ======================================================
echo.

:: 1. Kiem tra moi truong
echo [BUOC 1] Kiem tra moi truong Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python chua duoc cai dat.
    pause
    exit /b
)

echo [+] Dang kiem tra thu vien (Transformers, Torch)...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt --quiet
) else (
    echo [!] Khong tim thay requirements.txt, bo qua buoc cai dat tu dong.
)
echo.

:: 2. Kiem tra du lieu dau vao
echo [BUOC 2] Kiem tra cau truc thu muc du lieu...
if not exist "data\train" (
    if not exist "data\fake_news.csv" (
        echo [ERROR] Khong tim thay thu muc data\train HOAC file data\fake_news.csv
        pause
        exit /b
    ) else (
        echo [+] Dang tien hanh chia du lieu vao cac folder...
        python src\split_data.py
    )
) else (
    echo [+] Da tim thay cac thu muc du lieu: train, val, test.
)
echo.

:: 3. Tien hanh Training
echo [BUOC 3] Bat dau qua trinh Training PhoBERT...
echo (Luu y: Qua trinh nay rat nang, co the mat 15-30 phut neu khong co GPU)
echo.
python src\train_phobert.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Co loi xay ra trong qua trinh Training PhoBERT.
    pause
    exit /b
)
echo.

:: 4. Kiem tra ket qua sau Training
echo [BUOC 4] Kiem tra file mo hinh PhoBERT...
if exist "models\phobert_model" (
    echo [+] Da tim thay model tai: models\phobert_model
) else (
    echo [!] Khong tim thay thu muc phobert_model.
)
echo.

echo [BUOC 5] Ket qua danh gia PhoBERT (Metrics):
if exist "src\show_metrics.py" (
    python src\show_metrics.py
)
echo.

echo ======================================================
echo    HOAN TAT QUA TRINH TRAINING PHOBERT
echo ======================================================
pause
endlocal