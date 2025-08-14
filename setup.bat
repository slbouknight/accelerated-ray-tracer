@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM === Config ===
set "BUILD_DIR=build"
set "BUILD_TYPE=Debug"
if not "%~1"=="" set "BUILD_TYPE=%~1"  REM usage: setup.bat Release

REM === Clean ===
echo [1/5] Cleaning "%BUILD_DIR%"...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"

REM === Check CMake ===
where cmake >nul 2>&1
if errorlevel 1 (
  echo [ERROR] CMake not found in PATH.
  exit /b 1
)

REM === Configure ===
echo [2/5] Configuring with CMake...
cmake -S . -B "%BUILD_DIR%"
if errorlevel 1 goto :err

REM === Build ===
echo [3/5] Building (%BUILD_TYPE%)...
cmake --build "%BUILD_DIR%" --config %BUILD_TYPE%
if errorlevel 1 goto :err

REM === Ask for output filename ===
set "OUTNAME="
set /p OUTNAME=Enter output file name (without extension): 
if "%OUTNAME%"=="" set "OUTNAME=output"

set "EXT=!OUTNAME:~-4!"
if /I "!EXT!"==".ppm" (
  set "OUTFILE=%OUTNAME%"
) else (
  set "OUTFILE=%OUTNAME%.ppm"
)

REM === Find the executable (try common paths, then scan) ===
echo [4/5] Locating rayTracer.exe...
set "EXE=%CD%\%BUILD_DIR%\bin\%BUILD_TYPE%\rayTracer.exe"
if not exist "%EXE%" set "EXE=%CD%\%BUILD_DIR%\%BUILD_TYPE%\rayTracer.exe"
if not exist "%EXE%" set "EXE=%CD%\%BUILD_DIR%\rayTracer.exe"

if not exist "%EXE%" (
  set "EXE="
  for /r "%BUILD_DIR%" %%F in (rayTracer.exe) do (
    set "EXE=%%~fF"
  )
)

if not defined EXE (
  echo [ERROR] rayTracer.exe not found under "%BUILD_DIR%".
  exit /b 1
)

REM === Run ===
echo [5/5] Running: "%EXE%" ^> "%OUTFILE%"
"%EXE%" > "%OUTFILE%"
if errorlevel 1 goto :err

echo.
echo Done. Wrote "%OUTFILE%".
exit /b 0

:err
echo.
echo [ERROR] A step failed. See messages above.
exit /b 1
