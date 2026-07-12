# Intraday Reversal Scanner

Production-grade NSE intraday reversal and pullback scanner using Dhan API. Detects institutional accumulation/distribution patterns across all NSE cash and F&O stocks in real-time.

## Architecture

```
src/
├── data/           # Dhan API client, market data, universe builder
├── indicators/     # Technical indicators (EMA, VWAP, ATR, RSI, MACD, ADX, etc.)
├── signals/        # Candlestick patterns, SMC, trend filter, relative strength
├── scanners/       # 7 scanner types + scanner manager
├── ranking/        # Confidence scoring, AI ranking engine
├── risk/           # Risk management, position sizing, targets
├── alerts/         # Telegram, Discord, Slack, desktop notifications
├── dashboard/      # Rich-based live terminal dashboard
├── backtest/       # Complete backtesting engine
├── catalysts/      # Catalyst detection, market context
├── utils/          # Utilities
├── config.py       # Configuration loader
└── main.py         # Orchestrator
```

## Scanner Types

1. **Bullish Pullback** - Strong rally, pullback to support, resume uptrend
2. **Bearish Pullback** - Strong selloff, pullback rally, sell into strength
3. **Exhaustion Reversal** - Panic selloff climax, institutional reversal
4. **Trend Reversal** - EMA crossover, VWAP reclaim, structure change
5. **VWAP Reversal** - Price stretched 2+ ATR from VWAP, reversal
6. **Failed Breakdown** - Price breaks support, immediately recovers
7. **Failed Breakout** - Price breaks resistance, immediately rejects

## Setup

### Windows
```bash
scripts\setup.bat
scripts\run_scanner.bat
```

### Linux / Cloud
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
python run.py --mode live
```

### Docker
```bash
docker-compose up --build -d
```

### Environment Variables
Copy `.env.example` to `.env` and fill in your Dhan API credentials.

## Usage

### Live Scanner
```bash
python run.py --mode live
```

### Single Ticker Scan
```bash
python run.py --mode single --ticker TATAMOTORS
```

### Backtest
```bash
python run.py --mode backtest --years 5
```

## Features

- **Multi-timeframe**: 1m, 3m, 5m, 15m simultaneously
- **7 Scanner Types**: Pullbacks, reversals, failed breakouts
- **Smart Money Concepts**: Liquidity sweeps, order blocks, FVG
- **Candlestick Patterns**: 15+ patterns auto-detected
- **Volume Analysis**: Climax, absorption, stopping volume, effort vs result
- **Institutional Filters**: Delivery %, OI, relative strength, sector trend
- **AI Confidence Score**: 0-100 with weighted components
- **Risk Management**: ATR stop, trailing, position sizing, R/R targets
- **Alerts**: Telegram, Discord, Slack, desktop, sound
- **Live Dashboard**: Rich terminal UI with real-time updates
- **Backtesting**: 5-year engine with full metrics
- **Catalyst Detection**: Earnings, news, corporate actions
- **Market Context**: Nifty trend, sector, VIX, A/D ratio

## Market Filters

- Min volume: 200,000 shares
- Min price: ₹100
- Min market cap: ₹5,000 Crore
- No circuit/operator stocks
- Top 500 by liquidity

## Indicators

EMA (9,20,50,200), VWAP, ATR, RSI, MACD, ADX, Supertrend, CCI, Stochastic, OBV, CMF, Volume Profile

## Deployment

### Oracle Cloud / Google Cloud (24x7)
```bash
# Option 1: Direct deployment
chmod +x scripts/deploy.sh
./scripts/deploy.sh <vm-ip> <ssh-key>

# Option 2: Docker
docker build -t scanner .
docker run -d --restart unless-stopped --env-file .env scanner
```

## License

Proprietary - For authorized use only