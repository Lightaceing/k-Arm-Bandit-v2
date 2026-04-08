@echo off
for /L %%i in (1, 1, 5) do (
    echo Iteration number %%i
    python main.py
)
