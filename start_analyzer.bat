@echo off
echo Starting Excel Data Analyzer...
echo.
echo This will open your default web browser with the Excel Data Analyzer application.
echo You can upload Excel or CSV files and get comprehensive data analysis.
echo.
echo Press Ctrl+C to stop the application.
echo.

"D:/COMPLETED PROJECTS/reports/New folder/.venv/Scripts/python.exe" -m streamlit run excel_analyzer.py

pause
