@echo off
REM Genera dist\XOR_PerceptronMulticapa.exe (un solo archivo, sin consola, con sus librerias adentro).
REM Requiere una vez: pip install -r requirements.txt pyinstaller
cd /d "%~dp0"
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name XOR_PerceptronMulticapa ^
    --paths src ^
    --workpath build --specpath build --distpath dist ^
    --exclude-module scipy --exclude-module pandas --exclude-module IPython ^
    --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module PySide6 ^
    --exclude-module pytest --exclude-module sphinx --exclude-module pygame ^
    src\main.py
