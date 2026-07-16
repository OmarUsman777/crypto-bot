"""
core.py — Shared engine for the virtual trading bot.
Handles: Binance data fetching, indicator calculation,
virtual wallet, trade logging (console + CSV).
Also contains the Grid Bot engine (GridEngine class).
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
FEE_PCT            = 0.001   # 0.1% per side (Binance taker)
MAX_POSITIONS      = 1       # single position only

# ── Per-mode TP/SL ──
MOMENTUM_TP        = 0.006   # 0.6% TP for momentum mode
MOMENTUM_SL        = 0.003   # 0.3% SL for momentum mode
REVERSION_TP       = 0.005   # 0.5% TP for mean reversion mode
REVERSION_SL       = 0.004   # 0.4% SL for mean reversion mode

# ── Trailing stop (shared) ──
TRAIL_TRIGGER_PCT  = 0.003   # 0.3% — trailing stop activates here
TRAIL_STOP_PCT     = 0.003   # trail locks in 0.3% profit floor

# Use these as defaults (overridden per trade via open_trade)
TAKE_PROFIT_PCT    = REVERSION_TP
STOP_LOSS_PCT      = REVERSION_SL

LOG_FILE = "trades_log.csv"

# ─────────────────────────────────────────────
# GRID BOT CONFIG
# ─────────────────────────────────────────────
GRID_LEVELS        = 8        # Total grid lines (4 buy + 4 sell)
GRID_SPACING_PCT   = 0.004    # 0.4% between each level (covers 0.2% fees + 0.2% profit)
CAPITAL_PER_LEVEL  = 15.0     # USDT per grid level ($15 x 8 = $120 total)
STOP_LOSS_LEVELS   = 2        # Stop loss if price drops N levels below grid bottom
GRID_RESET_MINS    = 60       # Rebuild grid every 60 minutes
GRID_LOG_FILE      = "grid_trades_log.csv"

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

    def open_trade(self, coin, price, strategy, tp_pct=None, sl_pct=None):
        if not self.can_open():
            if len(self.positions) >= MAX_POSITIONS:
                print(Fore.YELLOW + f"[WALLET] Max {MAX_POSITIONS} positions reached — skipping {coin}")
            else:
                print(Fore.RED + f"[WALLET] Insufficient balance ${self.balance:.2f} — skipping {coin}")
            return False
        if self.has_position_for(coin):
            print(Fore.YELLOW + f"[WALLET] Already holding {coin} — skipping")
            return False

        tp_pct = tp_pct if tp_pct is not None else TAKE_PROFIT_PCT
        sl_pct = sl_pct if sl_pct is not None else STOP_LOSS_PCT

        fee = TRADE_SIZE * FEE_PCT
        self.balance -= (TRADE_SIZE + fee)
        pos = {
            "coin":            coin,
            "entry":           price,
            "size":            TRADE_SIZE,
            "strategy":        strategy,
            "opened_at":       datetime.now(),
            "tp":              price * (1 + tp_pct),
            "sl":              price * (1 - sl_pct),
            "tp_pct":          tp_pct,
            "sl_pct":          sl_pct,
            "trail_active":    False,
            "trail_sl":        None,
            "peak_price":      price,
        }
        self.positions.append(pos)

        print(Fore.CYAN + f"\n{'─'*55}")
        print(Fore.GREEN + f"  ▶  BUY  {coin}  @  ${price:.4f}")
        print(f"     Strategy : {strategy}")
        print(f"     Size     : ${TRADE_SIZE:.2f}")
        print(f"     TP       : ${pos['tp']:.4f}  (+{tp_pct*100:.1f}%)")
        print(f"     SL       : ${pos['sl']:.4f}  (-{sl_pct*100:.1f}%)")
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


# ─────────────────────────────────────────────
# GRID BOT HELPERS
# ─────────────────────────────────────────────
def calc_volatility(symbol):
    """Returns 24h price range % as volatility measure."""
    try:
        raw = exchange.fetch_ohlcv(symbol, "1h", limit=24)
        df  = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        if len(df) < 6:
            return 0
        high = df["high"].max()
        low  = df["low"].min()
        mid  = (high + low) / 2
        return (high - low) / mid if mid > 0 else 0
    except:
        return 0


def calc_volume_usdt(symbol):
    """Returns average hourly USDT volume."""
    try:
        raw = exchange.fetch_ohlcv(symbol, "1h", limit=6)
        df  = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        if len(df) < 3:
            return 0
        return (df["close"] * df["volume"]).mean()
    except:
        return 0


def select_best_grid_coin():
    """
    Scans all COINS and returns (best_coin, current_price).
    Ranks by volatility (price movement = more grid cycles)
    confirmed by volume (liquidity for fills).
    """
    print(Fore.YELLOW + "\n  Scanning coins for best grid opportunity...")
    scores = []
    for coin in COINS:
        vol_pct = calc_volatility(coin)
        volume  = calc_volume_usdt(coin)
        price   = get_current_price(coin)
        if not price or vol_pct == 0:
            continue
        score = vol_pct * 100 + (min(volume, 1_000_000) / 1_000_000 * 0.5)
        scores.append((coin, score, vol_pct, volume, price))
        print(Fore.WHITE +
            f"    {coin:<14}  volatility={vol_pct*100:.2f}%"
            f"  vol=${volume:,.0f}  score={score:.3f}")

    if not scores:
        return None, None
    scores.sort(key=lambda x: x[1], reverse=True)
    best = scores[0]
    print(Fore.GREEN +
        f"\n  ★ Best grid coin: {best[0]}"
        f"  (vol={best[2]*100:.2f}%, score={best[1]:.3f})")
    return best[0], best[4]


# ─────────────────────────────────────────────
# GRID ENGINE
# ─────────────────────────────────────────────
class GridEngine:
    """
    Virtual grid trading engine.
    Places buy/sell levels at fixed intervals around current price.
    Profits from price oscillation — no direction prediction needed.

    Math:
      Grid spacing  : 0.4% between levels
      Round-trip fee: 0.2% (0.1% buy + 0.1% sell)
      Net per cycle : ~0.2% profit per completed buy→sell pair
    """

    def __init__(self, coin, center_price, balance):
        self.coin        = coin
        self.center      = center_price
        self.balance     = balance
        self.starting    = balance
        self.created_at  = datetime.now()
        self.total_profit = 0.0
        self.cycles      = 0
        self.trades      = []

        self.levels = self._build_levels(center_price)
        self.stop_loss_price = (
            self.levels[0]["price"] * (1 - GRID_SPACING_PCT * STOP_LOSS_LEVELS)
        )
        self._init_log()

    def _build_levels(self, center):
        levels = []
        n = GRID_LEVELS // 2
        for i in range(-n, n + 1):
            if i == 0:
                continue
            price   = center * (1 + i * GRID_SPACING_PCT)
            lv_type = "buy" if i < 0 else "sell"
            levels.append({
                "index":     i,
                "price":     price,
                "type":      lv_type,
                "state":     "watch_buy" if i < 0 else "watch_sell",
                "coin_held": 0.0,
            })
        levels.sort(key=lambda x: x["price"])
        return levels

    def _init_log(self):
        if not os.path.exists(GRID_LOG_FILE):
            with open(GRID_LOG_FILE, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "coin", "action", "price",
                    "level_index", "usdt_amount", "profit_usdt",
                    "total_profit", "balance"
                ])

    def _log_trade(self, action, price, level_index, usdt, profit=0):
        self.total_profit += profit
        self.balance      += profit
        record = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.coin, action, round(price, 6),
            level_index, round(usdt, 4),
            round(profit, 4), round(self.total_profit, 4),
            round(self.balance, 4)
        ]
        with open(GRID_LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow(record)
        self.trades.append(record)

    def check_price(self, current_price):
        """
        Core grid logic: detect level crossings and simulate fills.
        Returns list of events: ('BUY'|'SELL'|'STOP_LOSS', price, level, [profit])
        """
        events = []

        if current_price <= self.stop_loss_price:
            events.append(("STOP_LOSS", current_price, 0))
            return events

        for lv in self.levels:
            # Price drops to buy level → simulate buy fill
            if lv["state"] == "watch_buy" and current_price <= lv["price"]:
                fee           = CAPITAL_PER_LEVEL * FEE_PCT
                coins_got     = (CAPITAL_PER_LEVEL - fee) / lv["price"]
                lv["coin_held"] = coins_got
                lv["state"]     = "watch_sell"
                self.balance   -= CAPITAL_PER_LEVEL
                events.append(("BUY", lv["price"], lv["index"]))
                self._log_trade("BUY", lv["price"], lv["index"], CAPITAL_PER_LEVEL)

            # Price rises to sell level AND we're holding coins from a buy below
            elif (lv["state"] == "watch_sell" and
                  lv["coin_held"] > 0 and
                  current_price >= lv["price"]):
                sell_value  = lv["coin_held"] * lv["price"]
                fee         = sell_value * FEE_PCT
                received    = sell_value - fee
                profit      = received - CAPITAL_PER_LEVEL
                lv["coin_held"] = 0.0
                # Re-arm the buy level one step below this sell level
                buy_lv = next(
                    (b for b in self.levels if b["index"] == lv["index"] - 1), None
                )
                if buy_lv:
                    buy_lv["state"] = "watch_buy"
                self.cycles += 1
                events.append(("SELL", lv["price"], lv["index"], profit))
                self._log_trade("SELL", lv["price"], lv["index"], received, profit)

        return events

    def needs_reset(self, current_price):
        """True if price has moved completely outside the grid range."""
        prices = [lv["price"] for lv in self.levels]
        return (current_price > max(prices) * 1.002 or
                current_price < min(prices) * 0.998)

    def time_for_reset(self):
        elapsed = (datetime.now() - self.created_at).total_seconds() / 60
        return elapsed >= GRID_RESET_MINS