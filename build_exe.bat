@echo off
rem 배포용 exe 빌드 스크립트 (PyInstaller 필요: pip install pyinstaller)
rem 결과물: dist\ContourReader.exe  (단일 파일, 콘솔 창 없음)
rem --collect-data customtkinter : CustomTkinter 의 테마/폰트 리소스를 함께 포함
rem --exclude-module            : 사용하지 않는 큰 모듈을 빼서 exe 크기와 시작 시간을 줄임
cd /d "%~dp0"
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name ContourReader ^
  --collect-data customtkinter ^
  --exclude-module matplotlib --exclude-module scipy --exclude-module pandas ^
  --exclude-module IPython --exclude-module pytest --exclude-module unittest ^
  --exclude-module setuptools --exclude-module pkg_resources --exclude-module numpy.tests ^
  main.py
if errorlevel 1 (
  echo 빌드 실패
  exit /b 1
)
echo.
echo 빌드 완료: %~dp0dist\ContourReader.exe
