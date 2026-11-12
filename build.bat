@echo off
rem Local MSVC build with Meson: build\Dust.dll for AviSynth+ and VapourSynth (x64, static runtime).
rem Uses meson and ninja from PATH, else the ones of F:\Python4Claude (pip install meson ninja).
setlocal
cd /d "%~dp0"
where meson >/dev/null 2>/dev/null || set "PATH=F:\Python4Claude\Lib\site-packages\bin;%PATH%"
where ninja >/dev/null 2>/dev/null || set "PATH=F:\Python4Claude\Lib\site-packages\bin;%PATH%"
if not exist build\build.ninja (
  meson setup build --vsenv -Db_vscrt=mt || exit /b 1
)
meson compile -C build || exit /b 1
