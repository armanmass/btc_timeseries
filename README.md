# BTC Analysis

A Bitcoin price analysis and visualization tool that fetches historical price data and generates technical analysis charts.

## Features

- Fetches historical BTC/USD price data from Kraken exchange
- Generates candlestick charts with technical indicators (SMA20, SMA50)
- Creates returns distribution analysis
- Supports hourly timeframe data

## Setup

Install dependencies:
```bash
pip install -e ".[dev]"  # Install with development tools
```

## Usage

Run the main script:
```bash
python btcproject.py
```

This will:
1. Download historical BTC/USD price data
2. Generate a candlestick chart with moving averages
3. Create a returns distribution plot
4. Save the data and charts to disk 