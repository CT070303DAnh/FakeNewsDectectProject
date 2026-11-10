@echo off
setlocal
title Quy trinh Training Vietnamese-SBERT (Keepitreal) - Fake News Detection
echo ======================================================
echo    QUY TRINH HUAN LUYEN MO HINH VIETNAMESE-SBERT
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
)
echo.

:: 2. Kiem tra du lieu dau vao
echo [BUOC 2] Kiem tra cau truc thu muc du lieu...
if not exist "data\train" (
    if exist "data\fake_news.csv" (
        echo [+] Dang tien hanh chia du lieu vao cac folder...
        python src\split_data.py
    ) else (
        echo [ERROR] Khong tim thay du lieu. Vui long kiem tra data\train HOAC data\fake_news.csv
        pause
        exit /b
    )
)
echo.

:: 3. Tien hanh Training
echo [BUOC 3] Bat dau qua trinh Training SBERT (Keepitreal)...
echo (Luu y: Qua trinh nay nang, co the mat nhieu thoi gian neu khong co GPU)
echo.
python src\train_sbert.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Co loi xay ra trong qua trinh Training SBERT.
    pause
    exit /b
)
echo.

:: 4. Kiem tra ket qua sau Training
echo [BUOC 4] Kiem tra file mo hinh SBERT...
if exist "models\sbert_model" (
    echo [+] Da tim thay model tai: models\sbert_model
) else (
    echo [!] Khong tim thay thu muc sbert_model.
)
echo.

echo [BUOC 5] Ket qua danh gia (Metrics):
if exist "src\show_metrics.py" (
    python src\show_metrics.py
)
echo.

echo ======================================================
echo    HOAN TAT QUA TRINH TRAINING VIETNAMESE-SBERT
echo ======================================================
pause
endlocal
