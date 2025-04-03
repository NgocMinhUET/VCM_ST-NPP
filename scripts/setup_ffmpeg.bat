@echo off
setlocal enabledelayedexpansion

echo Checking for FFmpeg installation...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo FFmpeg is already installed.
    ffmpeg -version
    goto :EOF
)

echo FFmpeg not found. Downloading FFmpeg...

:: Create temp directory for download
set TEMP_DIR=%TEMP%\ffmpeg_install
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

:: Set FFmpeg download URL
set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
set FFMPEG_ZIP=%TEMP_DIR%\ffmpeg.zip

:: Download FFmpeg
echo Downloading FFmpeg from %FFMPEG_URL%...
powershell -Command "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%'"
if %ERRORLEVEL% NEQ 0 (
    echo Failed to download FFmpeg.
    echo Please download manually from https://ffmpeg.org/download.html
    goto :EOF
)

:: Extract the ZIP file
echo Extracting FFmpeg...
powershell -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%TEMP_DIR%' -Force"
if %ERRORLEVEL% NEQ 0 (
    echo Failed to extract FFmpeg.
    goto :EOF
)

:: Move the FFmpeg folder to the project directory
set PROJECT_DIR=%~dp0..
set FFMPEG_DIR=%PROJECT_DIR%\ffmpeg
echo Moving FFmpeg to %FFMPEG_DIR%...

:: Find the directory name (it might vary based on the release)
for /d %%i in (%TEMP_DIR%\ffmpeg-*) do (
    set EXTRACTED_DIR=%%i
)

:: Create FFmpeg directory if it doesn't exist
if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"

:: Copy FFmpeg bin directory
xcopy "!EXTRACTED_DIR!\bin" "%FFMPEG_DIR%\bin\" /E /I /Y
if %ERRORLEVEL% NEQ 0 (
    echo Failed to copy FFmpeg files.
    goto :EOF
)

:: Add FFmpeg to the user's PATH temporarily
set PATH=%PATH%;%FFMPEG_DIR%\bin

:: Verify installation
echo Testing FFmpeg installation...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo FFmpeg has been installed successfully.
    ffmpeg -version
) else (
    echo FFmpeg installation may have failed.
    echo Please add %FFMPEG_DIR%\bin to your PATH manually.
)

echo.
echo To use FFmpeg in current session, run:
echo set PATH=%%PATH%%;%FFMPEG_DIR%\bin
echo.
echo To add FFmpeg to PATH permanently, add the following directory to your system PATH:
echo %FFMPEG_DIR%\bin

:: Clean up
echo Cleaning up temporary files...
rd /s /q "%TEMP_DIR%" 2>nul

endlocal 