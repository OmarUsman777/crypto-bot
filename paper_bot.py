"""
PENGU/USDT paper-trading bot (simulation only — no API keys, no real orders).

Improvements over the original:
  - Reads the last CLOSED candle for RSI/volume signals (not the half-formed one)
  - Stop-loss + optional max-hold timeout so capital never gets stuck forever
  - Balance now compounds correctly (no hardcoded $50 reset)
  - State persisted to disk -> survives restarts mid-position
  - All config via environment variables (Railway-friendly, no secrets in code)
  - Proper logging to stdout (Railway logs) + a local file
  - Telegram notifications (buy / sell / errors / startup) + live commands
  - Verifies the trading pair exists on Binance before starting
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd
import pandas_ta as ta
import requests

# ---------------------------------------------------------------------------
# CONFIG  — every value can be overridden by an environment variable
# ---------------------------------------------------------------------------
def _f(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)

def _i(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)

SYMBOL            = os.getenv("SYMBOL", "PENGU/USDT")
TIMEFRAME         = os.getenv("TIMEFRAME", "5m")
LOOKBACK_PERIODS  = _i("LOOKBACK_PERIODS", 30)      # volume baseline window
RSI_LENGTH        = _i("RSI_LENGTH", 14)
RSI_OVERSOLD      = _f("RSI_OVERSOLD", 30)
VOLUME_SPIKE_MULT = _f("VOLUME_SPIKE_MULT", 1.5)    # vol must be > 1.5x average
PROFIT_TARGET_PCT = _f("PROFIT_TARGET_PCT", 1.0)    # take profit at +1%
STOP_LOSS_PCT     = _f("STOP_LOSS_PCT", 2.0)        # cut loss at -2% (0 = off)
MAX_HOLD_MINUTES  = _i("MAX_HOLD_MINUTES", 0)       # force-exit after N min (0 = off)
START_CAPITAL     = _f("START_CAPITAL", 50.0)
POLL_SECONDS      = _i("POLL_SECONDS", 30)
STATE_FILE        = os.getenv("STATE_FILE", "state.json")

TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TG_API            = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ---------------------------------------------------------------------------
# LOGGING  — stdout shows up in Railway's log viewer; file is a local backup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")],
)
log = logging.getLogger("paperbot")

# ---------------------------------------------------------------------------
# SHARED STATE  (guarded by a lock because the Telegram thread reads it too)
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
state = {
    "is_holding": False,
    "buy_price": 0.0,
    "tokens": 0.0,
    "usdt": START_CAPITAL,
    "capital_at_entry": 0.0,
    "buy_ts": 0.0,
    "trades_completed": 0,
    "total_profit": 0.0,
    "paused": False,
}

def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        log.warning(f"Could not save state: {e}")

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state.update(json.load(f))
            log.info("Restored previous state from disk.")
        except Exception as e:
            log.warning(f"Could not load state: {e}")

# ---------------------------------------------------------------------------
# TELEGRAM  (every call is wrapped so a Telegram outage never kills trading)
# ---------------------------------------------------------------------------
def notify(text):
    log.info(text.replace("\n", " | "))
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        requests.post(
            f"{TG_API}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")

def status_text():
    with state_lock:
        s = dict(state)
    if s["is_holding"]:
        held = (time.time() - s["buy_ts"]) / 60 if s["buy_ts"] else 0
        return (f"📊 <b>HOLDING {SYMBOL}</b>\n"
                f"Entry: ${s['buy_price']:.6f}\n"
                f"Tokens: {s['tokens']:.2f}\n"
                f"Held: {held:.0f} min\n"
                f"Trades done: {s['trades_completed']}\n"
                f"Total P/L: ${s['total_profit']:.2f}")
    return (f"📊 <b>SCANNING {SYMBOL}</b>\n"
            f"Balance: ${s['usdt']:.2f}\n"
            f"Trades done: {s['trades_completed']}\n"
            f"Total P/L: ${s['total_profit']:.2f}\n"
            f"Paused: {s['paused']}")

def handle_command(text):
    if text in ("/status", "/balance", "/trades"):
        notify(status_text())
    elif text == "/pause":
        with state_lock:
            state["paused"] = True
            save_state()
        notify("⏸️ Trading <b>paused</b>. Open positions are still monitored. Send /resume to scan again.")
    elif text == "/resume":
        with state_lock:
            state["paused"] = False
            save_state()
        notify("▶️ Trading <b>resumed</b>.")
    elif text in ("/help", "/start"):
        notify("🤖 <b>Paper Trading Bot</b>\n"
               "/status — current position & P/L\n"
               "/pause — stop opening new trades\n"
               "/resume — resume scanning\n"
               "/help — this message")
    # anything else is ignored silently

def command_listener():
    """Background thread: long-polls Telegram for /commands from the owner."""
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        log.info("Telegram not configured — running in console-only mode.")
        return
    offset = None
    while True:
        try:
            params = {"timeout": 50}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(f"{TG_API}/getUpdates", params=params, timeout=60)
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat_id = str((msg.get("chat") or {}).get("id", ""))
                text = (msg.get("text") or "").strip().lower()
                if chat_id == TELEGRAM_CHAT_ID:   # only the owner can command it
                    handle_command(text)
        except Exception as e:
            log.warning(f"Telegram listener hiccup: {e}")
            time.sleep(5)

# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------
exchange = ccxt.binance({"enableRateLimit": True})

def fetch_market_signals():
    """Returns entry signals from the last CLOSED candle, plus the freshest price."""
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["rsi"] = ta.rsi(df["close"], length=RSI_LENGTH)
    df["volume_avg"] = df["volume"].rolling(window=LOOKBACK_PERIODS).mean()

    closed = df.iloc[-2]   # last fully-closed candle -> trustworthy RSI & volume
    live   = df.iloc[-1]   # current forming candle  -> freshest price for exits
    return {
        "price": float(live["close"]),        # current price, used for entries/exits
        "rsi": float(closed["rsi"]),
        "volume": float(closed["volume"]),
        "avg_volume": float(closed["volume_avg"]),
    }

# ---------------------------------------------------------------------------
# ONE ITERATION OF STRATEGY  (separated out so it can be tested in isolation)
# ---------------------------------------------------------------------------
def step():
    m = fetch_market_signals()
    target_mult = 1 + PROFIT_TARGET_PCT / 100
    stop_mult   = 1 - STOP_LOSS_PCT / 100

    with state_lock:
        holding = state["is_holding"]
        paused = state["paused"]

    if not holding:
        log.info(f"Scan | px ${m['price']:.6f} | rsi {m['rsi']:.1f} | "
                 f"vol {m['volume']:.0f}/{m['avg_volume']:.0f}"
                 + (" | PAUSED" if paused else ""))
        if paused:
            return
        if m["rsi"] < RSI_OVERSOLD and m["volume"] > m["avg_volume"] * VOLUME_SPIKE_MULT:
            with state_lock:
                spent = state["usdt"]
                state["buy_price"] = m["price"]
                state["capital_at_entry"] = spent
                state["tokens"] = spent / m["price"]
                state["usdt"] = 0.0
                state["is_holding"] = True
                state["buy_ts"] = time.time()
                save_state()
            msg = (f"🟢 <b>BUY</b> {SYMBOL}\n"
                   f"Price: ${m['price']:.6f}\n"
                   f"Spent: ${spent:.2f}\n"
                   f"Target: ${m['price'] * target_mult:.6f} (+{PROFIT_TARGET_PCT:.1f}%)")
            if STOP_LOSS_PCT > 0:
                msg += f"\nStop: ${m['price'] * stop_mult:.6f} (-{STOP_LOSS_PCT:.1f}%)"
            notify(msg)
        return

    # ---- HOLDING: monitor for target / stop-loss / max-hold ----
    with state_lock:
        bp = state["buy_price"]
        toks = state["tokens"]
        cap = state["capital_at_entry"]
        bts = state["buy_ts"]
    target = bp * target_mult
    stop   = bp * stop_mult
    held_min = (time.time() - bts) / 60 if bts else 0

    log.info(f"Hold | cur ${m['price']:.6f} | tgt ${target:.6f}"
             + (f" | stop ${stop:.6f}" if STOP_LOSS_PCT > 0 else "")
             + f" | {held_min:.0f}m")

    reason = None
    if m["price"] >= target:
        reason = "🎯 TARGET HIT"
    elif STOP_LOSS_PCT > 0 and m["price"] <= stop:
        reason = "🛑 STOP-LOSS"
    elif MAX_HOLD_MINUTES > 0 and held_min >= MAX_HOLD_MINUTES:
        reason = "⏱️ MAX-HOLD TIMEOUT"

    if reason:
        proceeds = toks * m["price"]
        pl = proceeds - cap
        with state_lock:
            state["usdt"] = proceeds          # balance compounds forward now
            state["tokens"] = 0.0
            state["is_holding"] = False
            state["buy_ts"] = 0.0
            state["trades_completed"] += 1
            state["total_profit"] += pl
            tot = state["total_profit"]
            tc = state["trades_completed"]
            save_state()
        sign = "+" if pl >= 0 else ""
        notify(f"🔴 <b>SELL — {reason}</b> {SYMBOL}\n"
               f"Price: ${m['price']:.6f}\n"
               f"Trade P/L: {sign}${pl:.2f}\n"
               f"New balance: ${proceeds:.2f}\n"
               f"Total P/L: ${tot:.2f} over {tc} trades")

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def trading_loop():
    err_count = 0
    while True:
        try:
            step()
            err_count = 0
            time.sleep(POLL_SECONDS)
        except Exception as e:
            err_count += 1
            log.warning(f"Data stream hiccup ({err_count}): {e}")
            if err_count == 3:               # only ping Telegram on a real outage
                notify(f"⚠️ Bot error: {e}")
            time.sleep(10)

def main():
    load_state()

    # Verify the pair actually trades on Binance spot before doing anything.
    try:
        exchange.load_markets()
        if SYMBOL not in exchange.markets:
            notify(f"❌ {SYMBOL} is not a Binance spot pair. Fix the SYMBOL and redeploy.")
            log.error(f"{SYMBOL} not found on Binance. Exiting.")
            time.sleep(30)   # avoid a tight crash-restart loop on Railway
            return
    except Exception as e:
        log.error(f"Could not reach Binance to verify {SYMBOL}: {e}")
        notify(f"❌ Could not reach Binance at startup: {e}")
        time.sleep(30)
        return

    notify(f"🚀 <b>Paper bot started</b> — {SYMBOL}\n"
           f"Capital: ${state['usdt']:.2f} | TF {TIMEFRAME}\n"
           f"Target +{PROFIT_TARGET_PCT:.1f}% / Stop -{STOP_LOSS_PCT:.1f}%\n"
           f"Send /help for commands")

    threading.Thread(target=command_listener, daemon=True).start()
    trading_loop()

if __name__ == "__main__":
    main()
