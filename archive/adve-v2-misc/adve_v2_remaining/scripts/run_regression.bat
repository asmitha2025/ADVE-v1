@echo off
set PYTHONPATH=.
..\venv\Scripts\python.exe scripts/run_regression.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] REGRESSION SUITE FAILED!
    exit /b %ERRORLEVEL%
)
echo [PASSED] ALL REGRESSION TESTS PASSED!
