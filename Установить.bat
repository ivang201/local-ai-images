@echo off
chcp 65001 >nul
title Установка: генерация картинок и удаление фона
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
