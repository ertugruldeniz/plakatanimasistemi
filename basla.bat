@echo off
echo Plaka Tanima Sistemi baslatiliyor...
echo.

REM Python kurulu mu kontrol et
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python kurulu degil! Lutfen Python'u yukleyin.
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Uygulamayı başlat
python basla.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Bir hata olustu.
    pause
) 