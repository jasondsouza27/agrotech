@echo off
echo ========================================
echo   AgroGuardian System
echo ========================================
echo.
echo Starting Backend Server...
start "AgroGuardian Backend" cmd /k "python prediction_server.py"
timeout /t 3 /nobreak >nul
echo.
echo ========================================
echo   Backend Running on http://localhost:5000
echo ========================================
echo.
echo To start the React frontend:
echo   cd frontend
echo   npm install
echo   npm run dev
echo.
echo Then open http://localhost:5173 in your browser
echo.
pause
