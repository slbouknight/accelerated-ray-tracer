@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM === Config ===
REM Release by default. A Debug CUDA build compiles device code with -G, which
REM disables almost all device-side optimisation -- it is for debugging, never
REM for timing. This script used to default to Debug.
set "BUILD_DIR=build"
set "BUILD_TYPE=Release"
if not "%~1"=="" set "BUILD_TYPE=%~1"

echo [1/4] Configuring (%BUILD_TYPE%)...
where cmake >nul 2>&1
if errorlevel 1 (
  echo [ERROR] CMake not found in PATH.
  exit /b 1
)
cmake -S . -B "%BUILD_DIR%"
if errorlevel 1 goto :err

echo [2/4] Building...
cmake --build "%BUILD_DIR%" --config %BUILD_TYPE% --parallel
if errorlevel 1 goto :err

echo [3/4] Running tests...
ctest --test-dir "%BUILD_DIR%" -C %BUILD_TYPE% --output-on-failure
if errorlevel 1 echo [WARN] Tests failed. Continuing to the render anyway.

REM === Locate the executable ===
set "EXE=%CD%\%BUILD_DIR%\bin\%BUILD_TYPE%\rayTracer.exe"
if not exist "%EXE%" set "EXE=%CD%\%BUILD_DIR%\bin\rayTracer.exe"
if not exist "%EXE%" (
  set "EXE="
  for /r "%BUILD_DIR%" %%F in (rayTracer.exe) do set "EXE=%%~fF"
)
if not defined EXE (
  echo [ERROR] rayTracer.exe not found under "%BUILD_DIR%".
  exit /b 1
)

REM === Ask for scene and output name ===
echo.
"%EXE%" --list
echo.
set "SCENE="
set /p SCENE=Scene [original]:
if "%SCENE%"=="" set "SCENE=original"

set "OUTNAME="
set /p OUTNAME=Output file name (without extension) [output]:
if "%OUTNAME%"=="" set "OUTNAME=output"

set "EXT=!OUTNAME:~-4!"
if /I "!EXT!"==".ppm" ( set "OUTFILE=%OUTNAME%" ) else ( set "OUTFILE=%OUTNAME%.ppm" )

echo [4/4] Rendering scene "%SCENE%" to "%OUTFILE%"...
"%EXE%" --scene %SCENE% --out "%CD%\%OUTFILE%"
if errorlevel 1 goto :err

echo.
echo Done. Wrote "%OUTFILE%".
echo Convert to PNG with:  python tools\ppm_to_png.py "%OUTFILE%"
exit /b 0

:err
echo.
echo [ERROR] A step failed. See messages above.
exit /b 1
