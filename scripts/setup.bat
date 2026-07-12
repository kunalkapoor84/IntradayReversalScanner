@echo off
REM Setup script for Intraday Reversal Scanner

echo ============================================
echo Intraday Reversal Scanner - Setup
echo ============================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3.12+ is required. Install from https://www.python.org/downloads/
    exit /b 1
)

REM Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate and install
echo Installing dependencies...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create .env if not exists
if not exist .env (
    echo Creating .env template...
    (
        echo # Dhan API Credentials
        echo DHAN_CLIENT_ID=your_client_id_here
        echo DHAN_ACCESS_TOKEN=your_access_token_here
        echo.
        echo # Telegram (optional)
        echo TELEGRAM_BOT_TOKEN=
        echo TELEGRAM_CHAT_ID=
        echo.
        echo # Discord (optional)
        echo DISCORD_WEBHOOK_URL=
        echo.
        echo # Slack (optional)
        echo SLACK_WEBHOOK_URL=
        echo.
        echo # Email (optional)
        echo SMTP_PASSWORD=
    ) > .env
    echo Created .env file - please edit with your credentials
)

REM Create logs directory
if not exist logs mkdir logs

echo.
echo Setup complete!
echo.
echo To run live scanner:
echo   run.bat
echo.
echo To scan a single ticker:
echo   python run.py --mode single --ticker TATAMOTORS
echo.
echo To run backtest:
echo   python run.py --mode backtest --years 5