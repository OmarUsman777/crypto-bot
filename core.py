"""
core.py — Shared engine for the virtual trading bot.
Handles: Binance data fetching, indicator calculation,
virtual wallet, trade logging (console + CSV).
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import csv
import os
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

# ─────────────────────────────────────────────
# CONFIG — edit these to change bot behaviour
# ─────────────────────────────────────────────
COINS = [
    "SOL/USDT", "BNB/USDT", "SUI/USDT", "ETH/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "ADA/USDT",
    "XRP/USDT", "DOT/USDT", "PENGU/USDT"
]

STARTING_BALANCE   = 135.0   # Virtual wallet in USDT
TRADE_SIZE         = 120.0   # Capital per trade in USDT
TAKE_PROFIT_PCT    = 0.005   # 0.5% TP
STOP_LOSS_PCT      = 0.004   # 0.4% SL
TRAIL_TRIGGER_PCT  = 0.003   # 0.3% — trailing stop activates here
TRAIL_STOP_PCT     = 0.003   # trail locks in 0.3% profit floor
FEE_PCT            = 0.001   # 0.1% per side (Binance taker)
MAX_POSITIONS      = 1       # single position only

LOG_FILE = "trades_log.csv"

# ─────────────────────────────────────────────
# EXCHANGE SETUP
# ─────────────────────────────────────────────
exchange = ccxt.binance({
    "enableRateLimit": True,
})

# ─────────────────────────────────────────────
# VIRTUAL WALLET
# ─────────────────────────────────────────────
class VirtualWallet:
    def __init__(self, balance=STARTING_BALANCE):
        self.balance        = balance
        self.starting       = balance
        self.positions      = []     # list of open position dicts
        self.trades         = []
        self._init_log()

    def _init_log(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "strategy", "coin", "side",
                    "entry_price", "exit_price", "size_usdt",
                    "pnl_usdt", "pnl_pct", "result", "exit_reason", "balance_after"
                ])

    def has_position(self):
        return len(self.positions) > 0

    def position_count(self):
        return len(self.positions)

    def has_position_for(self, coin):
        return any(p["coin"] == coin for p in self.positions)

    def can_open(self):
        return (len(self.positions) < MAX_POSITIONS and
                self.balance >= TRADE_SIZE)

    def open_trade(self, coin, price, strategy):
        if not self.can_open():
            if len(self.positions) >= MAX_POSITIONS:
                print(Fore.YELLOW + f"[WALLET] Max {MAX_POSITIONS} positions reached — skipping {coin}")
            else:
                print(Fore.RED + f"[WALLET] Insufficient balance ${self.balance:.2f} — skipping {coin}")
            return False
        if self.has_position_for(coin):
            print(Fore.YELLOW + f"[WALLET] Already holding {coin} — skipping")
            return False

        fee = TRADE_SIZE * FEE_PCT
        self.balance -= (TRADE_SIZE + fee)
        pos = {
            "coin":            coin,
            "entry":           price,
            "size":            TRADE_SIZE,
            "strategy":        strategy,
            "opened_at":       datetime.now(),
            "tp":              price * (1 + TAKE_PROFIT_PCT),
            "sl":              price * (1 - STOP_LOSS_PCT),
            "trail_active":    False,   # trailing stop not yet activated
            "trail_sl":        None,    # dynamic trailing SL price
            "peak_price":      price,   # highest price seen since entry
        }
        self.positions.append(pos)

        print(Fore.CYAN + f"\n{'─'*55}")
        print(Fore.GREEN + f"  ▶  BUY  {coin}  @  ${price:.4f}")
        print(f"     Strategy : {strategy}")
        print(f"     Size     : ${TRADE_SIZE:.2f}")
        print(f"     TP       : ${pos['tp']:.4f}  (+{TAKE_PROFIT_PCT*100:.1f}%)")
        print(f"     SL       : ${pos['sl']:.4f}  (-{STOP_LOSS_PCT*100:.1f}%)")
        print(f"     Trail    : activates at +{TRAIL_TRIGGER_PCT*100:.1f}%, locks ≥+{TRAIL_STOP_PCT*100:.1f}%")
        print(f"     Slots    : {len(self.positions)}/{MAX_POSITIONS}  |  Balance: ${self.balance:.2f}")
        print(Fore.CYAN + f"{'─'*55}\n")
        return True

    def _close_position(self, pos, exit_price, reason):
        fee     = pos["size"] * FEE_PCT
        gross   = pos["size"] * (exit_price / pos["entry"])
        net     = gross - fee
        pnl     = net - pos["size"]
        pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
        self.balance += net
        result = "WIN" if pnl > 0 else "LOSS"
        color  = Fore.GREEN if pnl > 0 else Fore.RED

        print(color + f"\n{'═'*55}")
        print(color + f"  {'✔' if pnl>0 else '✘'}  CLOSE  {pos['coin']}  @  ${exit_price:.4f}  [{reason}]")
        print(f"     Entry    : ${pos['entry']:.4f}")
        print(f"     PnL      : ${pnl:+.3f}  ({pnl_pct:+.2f}%)")
        print(f"     Result   : {result}")
        print(f"     Slots    : {len(self.positions)-1}/{MAX_POSITIONS}  |  Balance: ${self.balance:.2f}")
        print(color + f"{'═'*55}\n")

        record = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy":     pos["strategy"],
            "coin":         pos["coin"],
            "side":         "BUY",
            "entry_price":  round(pos["entry"], 6),
            "exit_price":   round(exit_price, 6),
            "size_usdt":    pos["size"],
            "pnl_usdt":     round(pnl, 4),
            "pnl_pct":      round(pnl_pct, 3),
            "result":       result,
            "exit_reason":  reason,
            "balance_after": round(self.balance, 4),
        }
        self.trades.append(record)
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(record.values())

        self.positions.remove(pos)
        return pnl

    def check_exits(self, coin, current_price):
        """Check all exit conditions for a specific coin position."""
        pos = next((p for p in self.positions if p["coin"] == coin), None)
        if not pos:
            return

        # Update peak price
        if current_price > pos["peak_price"]:
            pos["peak_price"] = current_price

        elapsed = (datetime.now() - pos["opened_at"]).total_seconds() / 60

        # ── Trailing stop logic ──
        profit_pct = (current_price - pos["entry"]) / pos["entry"]

        if not pos["trail_active"] and profit_pct >= TRAIL_TRIGGER_PCT:
            # Activate trailing stop — set SL to lock in TRAIL_STOP_PCT profit
            pos["trail_active"] = True
            pos["trail_sl"]     = pos["entry"] * (1 + TRAIL_STOP_PCT)
            print(Fore.YELLOW +
                f"  [TRAIL] {coin} trailing stop ACTIVATED — locked at"
                f" ${pos['trail_sl']:.4f} (+{TRAIL_STOP_PCT*100:.1f}%)")

        if pos["trail_active"]:
            # Move trail SL up if price keeps rising (always lock in TRAIL_STOP_PCT below peak)
            new_trail_sl = pos["peak_price"] * (1 - TRAIL_STOP_PCT)
            if new_trail_sl > pos["trail_sl"]:
                pos["trail_sl"] = new_trail_sl

            # Exit if price drops below trailing SL
            if current_price <= pos["trail_sl"]:
                self._close_position(pos, current_price, "TRAILING STOP")
                return

        # ── Standard exits ──
        if elapsed >= 90:
            self._close_position(pos, current_price, "TIMEOUT (90min)")
        elif current_price >= pos["tp"]:
            self._close_position(pos, current_price, "TAKE PROFIT")
        elif current_price <= pos["sl"]:
            self._close_position(pos, current_price, "STOP LOSS")

    def check_all_exits(self, price_map):
        """Call with dict of {coin: price} to check all open positions."""
        for pos in list(self.positions):
            price = price_map.get(pos["coin"])
            if price:
                self.check_exits(pos["coin"], price)

    def summary(self):
        total_trades = len(self.trades)
        wins   = sum(1 for t in self.trades if t["result"] == "WIN")
        losses = total_trades - wins
        total_pnl = sum(t["pnl_usdt"] for t in self.trades)
        win_rate  = (wins / total_trades * 100) if total_trades else 0
        print(Fore.YELLOW + f"\n{'━'*55}")
        print(Fore.YELLOW + "  SESSION SUMMARY")
        print(f"  Trades      : {total_trades}  (W:{wins}  L:{losses})")
        print(f"  Win Rate    : {win_rate:.1f}%")
        print(f"  Total PnL   : ${total_pnl:+.3f}")
        print(f"  Balance     : ${self.balance:.2f}  (started ${self.starting:.2f})")
        print(Fore.YELLOW + f"{'━'*55}\n")


# ─────────────────────────────────────────────
# DATA FETCHING & INDICATORS
# ─────────────────────────────────────────────
def fetch_ohlcv(symbol, timeframe="5m", limit=100):
    """Fetch OHLCV candles and return as DataFrame with indicators."""
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df
    except Exception as e:
        print(Fore.RED + f"[DATA] Error fetching {symbol} {timeframe}: {e}")
        return None


def add_indicators(df):
    """Add RSI, EMA50, EMA200, MACD, Bollinger Bands, VWAP to a DataFrame."""
    df = df.copy()
    df["rsi"]    = ta.rsi(df["close"], length=14)
    df["ema50"]  = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd_hist"] = macd["MACDh_12_26_9"]

    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        # Column name varies by pandas-ta version — find it dynamically
        lower_col  = [c for c in bb.columns if c.startswith("BBL")][0]
        middle_col = [c for c in bb.columns if c.startswith("BBM")][0]
        upper_col  = [c for c in bb.columns if c.startswith("BBU")][0]
        df["bb_lower"]  = bb[lower_col]
        df["bb_middle"] = bb[middle_col]
        df["bb_upper"]  = bb[upper_col]

    # Average volume (20 periods)
    df["vol_avg"] = df["volume"].rolling(20).mean()

    return df


def get_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker["last"]
    except Exception as e:
        print(Fore.RED + f"[PRICE] Error fetching {symbol}: {e}")
        return None