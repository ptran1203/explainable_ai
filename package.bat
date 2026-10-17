@echo off
REM Packages the assignment submission.
REM   1. Rebuilds submission\25C11016_25C11003\source.zip from repe_source\ (no .git, no data).
REM   2. If Report.docx and Slides.pptx are already in that folder, zips the whole
REM      folder into submission\25C11016_25C11003.zip (the final file to hand in).
cd /d "%~dp0"
python package.py
pause
