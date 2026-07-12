#!/bin/bash
# Setup script for Linux (Oracle Cloud / Google Cloud)

set -e

echo "=== Intraday Reversal Scanner Setup ==="

# Check Python
python3 --version || { echo "Python 3.12+ required"; exit 1; }

# Create venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create .env
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Dhan API Credentials
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here

# Telegram (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Discord (optional)
DISCORD_WEBHOOK_URL=

# Slack (optional)
SLACK_WEBHOOK_URL=

# Email (optional)
SMTP_PASSWORD=
EOF
    echo "Created .env - edit with your credentials"
fi

mkdir -p logs backtest_results

echo ""
echo "Setup complete!"
echo "Run: source venv/bin/activate && python run.py --mode live"