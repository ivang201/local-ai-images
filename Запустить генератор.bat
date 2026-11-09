@echo off
chcp 65001 >nul
title Генератор картинок FLUX Klein
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "flux\app\app.py"
) else (
  python "flux\app\app.py"
)
if errorlevel 1 pause
