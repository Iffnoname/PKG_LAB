@echo off
rem
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --name Lab3_Variant4 ^
  --add-data "images;images" ^
  --add-data "app;app" ^
  --collect-submodules numba --collect-submodules llvmlite ^
  main.py
echo.
echo Готово: dist\Lab3_Variant4\Lab3_Variant4.exe  (папку dist\Lab3_Variant4 копировать целиком)
pause
