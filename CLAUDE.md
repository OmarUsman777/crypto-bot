# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file PENGU/USDT **paper-trading** bot (`paper_bot.py`) — simulation only. It never places real orders and uses no exchange API keys; it only *reads* public OHLCV data from Binance via `ccxt`. All "balance" and "P/L" are simulated in memory and on disk. Keep it that way unless explicitly asked to add live trading.

## Commands

```bash
pip install -r requirements.txt        # install deps
cp .env.example .env                    # then fill in TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
python paper_bot.py                     # run (reads config from env / .env)
```

There is no test suite, linter, or build step. `step()` is intentionally factored out as one strategy iteration so it can be driven in isolation if you add tests.

## Architecture

Two threads share one global `state` dict guarded by `state_lock`:

1. **`trading_loop()`** (main thread) — calls `step()` every `POLL_SECONDS`, with a backoff/error counter that only alerts Telegram after 3 consecutive failures.
2. **`command_listener()`** (daemon thread) — long-polls Telegram `getUpdates` for owner `/commands` (`/status`, `/pause`, `/resume`, `/help`). Only messages from `TELEGRAM_CHAT_ID` are honored.

Strategy in `step()`:
- **Entry signals** come from `fetch_market_signals()`, which reads the **last *closed* candle** (`df.iloc[-2]`) for RSI and volume — never the half-formed current candle — but uses the live candle's close (`df.iloc[-1]`) as the freshest price for entries/exits. This distinction matters: don't "simplify" it to a single row.
- **Buy** when `rsi < RSI_OVERSOLD` AND `volume > avg_volume * VOLUME_SPIKE_MULT`. Goes all-in with the full simulated USDT balance.
- **Exit** on profit target, stop-loss (if `STOP_LOSS_PCT > 0`), or max-hold timeout (if `MAX_HOLD_MINUTES > 0`).
- Balance **compounds**: on sell, proceeds become the new `usdt` balance (no reset to `START_CAPITAL`).

State persistence: the whole `state` dict is JSON-dumped to `STATE_FILE` (default `state.json`) on every mutation and restored on startup via `load_state()`, so an open position survives a restart. Every `state` write happens inside `with state_lock:` followed by `save_state()` — preserve this pattern.

Telegram is optional: if `TELEGRAM_TOKEN`/`TELEGRAM_CHAT_ID` are unset, the bot runs console-only and `notify()` just logs. Every Telegram call is wrapped so an outage never kills trading.

## Configuration

All tunables are environment variables read at startup (see `.env.example` and the CONFIG block at the top of `paper_bot.py`) — no config file, no constants to edit. Helpers `_f()`/`_i()` coerce env values with safe fallbacks.

## Deployment

Targets Railway (`Procfile`: `worker: python paper_bot.py`). Logs go to stdout (Railway log viewer) plus a local `bot.log`. On fatal startup problems (pair not on Binance, can't reach Binance) the bot sleeps 30s before exiting to avoid a tight crash-restart loop. To persist `state.json` across Railway deploys, attach a Volume and point `STATE_FILE` at it (e.g. `/data/state.json`).
