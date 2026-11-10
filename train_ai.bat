@echo off
setlocal
title Quy trinh Training AI - Fake News Detection
echo ======================================================
echo    QUY TRINH DUA DU LIEU VAO TRAINING MO HINH AI
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

echo [+] Dang kiem tra thu vien...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt --quiet
) else (
    echo [!] Khong tim thay requirements.txt, bo qua buoc cai dat tu dong.
)
echo.

:: 2. Kiem tra du lieu dau vao (KIEM TRA THU MUC)
echo [BUOC 2] Kiem tra cau truc thu muc du lieu...
if not exist "data\train" (
    if not exist "data\fake_news.csv" (
        echo [ERROR] Khong tim thay thu muc data\train HOAC file data\fake_news.csv
        echo Vui long kiem tra lai du lieu trong thu muc data.
        pause
        exit /b
    ) else (
        echo [+] Tim thay file nguon, dang tien hanh chia du lieu vao cac folder...
        python src\split_data.py
    )
) else (
    echo [+] Da tim thay cac thu muc du lieu: train, val, test.
)
echo.

:: 3. Tien hanh Training
echo [BUOC 3] Bat dau qua trinh Training cac mo hinh...
echo (Luu y: Qua trinh nay co the mat vai phut)
echo.
python src\train.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Co loi xay ra trong qua trinh Training Python.
    pause
    exit /b
)
echo.

:: 4. Kiem tra ket qua sau Training
echo [BUOC 4] Kiem tra cac file mo hinh da duoc tao ra...
if exist "models" (
    dir models\*.pkl /b
    if exist "models\doc2vec.model" echo doc2vec.model
)
echo.

echo [BUOC 5] Ket qua danh gia mo hinh (Metrics):
if exist "src\show_metrics.py" (
    python src\show_metrics.py
)
echo.

echo ======================================================
echo    HOAN TAT QUA TRINH TRAINING AI
echo ======================================================
pause
endlocal