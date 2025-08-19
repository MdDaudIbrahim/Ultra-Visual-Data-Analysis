@echo off
title Excel Data Analyzer Launcher
color 0B

:MENU
cls
echo =========================================================
echo           EXCEL DATA ANALYZER LAUNCHER
echo =========================================================
echo.
echo Choose your version:
echo.
echo  [1] Basic Excel Analyzer
echo      - Simple data visualization
echo      - Basic statistics
echo      - Quick analysis
echo.
echo  [2] Advanced Excel Analyzer
echo      - AI-powered insights
echo      - Advanced statistical analysis
echo      - Professional data quality assessment
echo      - Comprehensive reporting
echo.
echo  [3] Ultra-Visual Analyzer (🌟 NEW!)
echo      - Stunning visual dashboard
echo      - 20+ interactive chart types
echo      - 3D visualizations and special plots
echo      - Enhanced UI with gradients
echo      - Visual analytics gallery
echo.
echo  [4] View User Guide
echo      - Complete documentation
echo      - Best practices
echo      - Troubleshooting tips
echo.
echo  [5] Exit
echo.
echo =========================================================

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto BASIC
if "%choice%"=="2" goto ADVANCED
if "%choice%"=="3" goto ULTRAVISUAL
if "%choice%"=="4" goto GUIDE
if "%choice%"=="5" goto EXIT

echo Invalid choice. Please try again.
pause
goto MENU

:BASIC
echo.
echo Starting Basic Excel Analyzer...
echo.
& "D:/COMPLETED PROJECTS/reports/New folder/.venv/Scripts/python.exe" -m streamlit run excel_analyzer.py
goto END

:ADVANCED
echo.
echo Starting Advanced Excel Analyzer...
echo.
& "D:/COMPLETED PROJECTS/reports/New folder/.venv/Scripts/python.exe" -m streamlit run advanced_excel_analyzer.py --server.port 8502
goto END

:ULTRAVISUAL
echo.
echo Starting Ultra-Visual Excel Analyzer...
echo.
& "D:/COMPLETED PROJECTS/reports/New folder/.venv/Scripts/python.exe" -m streamlit run ultra_visual_analyzer.py --server.port 8503
goto END

:GUIDE
echo.
echo Opening User Guide...
echo.
start USER_GUIDE.md
goto MENU

:EXIT
echo.
echo Thank you for using Excel Data Analyzer!
echo.
timeout /t 2 >nul
exit

:END
echo.
echo Application has been closed.
pause
goto MENU
