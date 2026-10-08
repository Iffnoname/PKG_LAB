@echo off
rem
chcp 65001 >nul
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error
python -m unittest discover -s tests -t . || goto :error
pyinstaller --noconfirm --clean --onefile --windowed --name Lab2ImageInfo ^
  --exclude-module tkinter --exclude-module numpy ^
  --exclude-module PySide6.QtNetwork --exclude-module PySide6.QtQml --exclude-module PySide6.QtQuick ^
  --exclude-module PySide6.QtPdf --exclude-module PySide6.QtSvg --exclude-module PySide6.QtOpenGL ^
  main.py || goto :error
echo.
echo Готово: dist\Lab2ImageInfo.exe
exit /b 0
:error
echo Сборка не удалась.
exit /b 1
