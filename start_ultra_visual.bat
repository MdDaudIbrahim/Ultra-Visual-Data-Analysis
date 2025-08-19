@echo off
echo =========================================================
echo    ULTRA-VISUAL EXCEL DATA ANALYZER
echo =========================================================
echo.
echo Starting the Ultra-Visual Excel Data Analyzer...
echo This enhanced version provides:
echo.
echo  🌟 Enhanced Visual Dashboard with gradient metrics
echo  🗃️ Advanced Interactive Data Explorer
echo  🎨 Comprehensive Visual Analytics Gallery:
echo     - Distribution Gallery (Histograms, Box plots, Violin plots)
echo     - Relationship Charts (Scatter, Correlation, Density)
echo     - Category Analysis (Bar charts, Pie charts, Grouped analysis)
echo     - Time Series (Line charts, Area charts, Moving averages)
echo     - Advanced Visuals (3D plots, Parallel coordinates)
echo     - Special Charts (Radar charts, Surface plots)
echo  📈 Enhanced Statistical Dashboard
echo  🧠 AI-Powered Visual Insights
echo  💾 Complete Export Center
echo.
echo The application will open in your browser with stunning visuals!
echo Upload your Excel/CSV files to see beautiful interactive charts.
echo.
echo Press Ctrl+C to stop the application.
echo =========================================================
echo.

& "D:/COMPLETED PROJECTS/reports/New folder/.venv/Scripts/python.exe" -m streamlit run ultra_visual_analyzer.py --server.port 8503

pause
