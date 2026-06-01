@echo off
chcp 65001 >nul
set PYTHONUTF8=1
D:\Software\miniconda3\envs\quant_env\python.exe D:\quantitative\main.py %*
pause
