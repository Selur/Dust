@echo off
REM DustPort build script
REM Requires: MSVC x64 Native Tools Command Prompt (VS 2019 or newer)
REM           AviSynth+ FilterSDK installed
REM
REM Run this from the directory containing DustPort.cpp.

setlocal

REM Default AviSynth+ FilterSDK path. Override with ASP_SDK environment variable.
if "%ASP_SDK%"=="" set ASP_SDK=C:\Program Files (x86)\AviSynth+\FilterSDK

if not exist "%ASP_SDK%\include\avisynth.h" (
    echo ERROR: avisynth.h not found at "%ASP_SDK%\include"
    echo Set ASP_SDK environment variable to your FilterSDK location, or install
    echo AviSynth+ to the default path.
    exit /b 1
)

echo Building DustPort.dll ...
cl /nologo /LD /EHsc /O2 /arch:AVX2 ^
   /I "%ASP_SDK%\include" ^
   DustPort.cpp ^
   /link /OUT:DustPort.dll

if errorlevel 1 (
    echo.
    echo BUILD FAILED.
    exit /b 1
)

REM Clean up intermediate files
del /q DustPort.obj DustPort.exp DustPort.lib 2>nul

echo.
echo BUILD SUCCEEDED.
echo Output: DustPort.dll
