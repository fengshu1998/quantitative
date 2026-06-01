@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set NO_PROXY=*.eastmoney.com,*.eastmoney.com,*.akshare.com,localhost,127.*
set no_proxy=*.eastmoney.com,*.eastmoney.com,*.akshare.com,localhost,127.*
D:\Software\miniconda3\envs\quant_env\python.exe D:\quantitative\main.py %*
pause
