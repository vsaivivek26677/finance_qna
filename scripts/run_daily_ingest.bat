@echo off
REM Wrapper for Windows Task Scheduler. Point a daily trigger at this file.
REM Task Scheduler runs with no shell profile loaded, so paths are absolute here.

cd /d "C:\Users\vsaiv\Downloads\finance_qna"
"C:\Users\vsaiv\AppData\Local\Programs\Python\Python312\python.exe" scripts\daily_ingest.py >> scripts\logs\task_scheduler.log 2>&1
