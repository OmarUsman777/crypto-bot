"""
strategy_scoring.py — Multi-Coin Scoring Engine
================================================
Scores every coin each minute using:
  5m RSI < 40               → +2 pts  (oversold macro)
  5m EMA50 > EMA200         → +2 pts  (uptrend filter)
  1m RSI crossover SMA(RSI) → +3 pts  (entry trigger)
  Volume spike > 1.3x avg   → +2 pts  (confirmation)
  MACD histogram positive   → +2 pts  (momentum)
  Near lower Bollinger Band → +1 pt   (mean reversion)

Trades only if score ≥ 7.
One open position at a time.
"""

import time
from datetime import datetime
from colorama import Fore, Style, init

from core import (
    COINS, MAX_POSITIONS, VirtualWallet, fetch_ohlcv, add_indicators,
    get_current_price
)

init(autoreset=True)

STRATEGY_NAME   = "MultiCoin Scoring"
SCAN_INTERVAL   = 60        # seconds between scans
MIN_SCORE       = 7         # minimum score to enter trade
RSI_SMA_PERIOD  = 14        # SMA of RSI for crossover detection

# ─────────────────────────────────────────────
# SCORING LOGIC
# ─────────────────────────────────────────────
def score_coin(symbol):
    """
    Returns (score, detail_dict) for a coin.
    Fetches both 5m and 1m candles.
    """
    score   = 0
    detail  = {}

    # ── 5-minute data ──
    # Need 250 candles minimum: EMA200 needs 200+ candles to warm up
    df5 = fetch_ohlcv(symbol, "5m", limit=250)
    if df5 is None or len(df5) < 210:
        return 0, {}
    df5 = add_indicators(df5)

    last5 = df5.iloc[-1]

    import pandas as pd

    # Uptrend filter: EMA50 > EMA200 — HARD REQUIREMENT, not just points
    ema50  = last5.get("ema50")
    ema200 = last5.get("ema200")
    if pd.notna(ema50) and pd.notna(ema200) and ema50 > ema200:
        score += 2
        detail["trend_filter"] = f"+2 (EMA50={ema50:.4f} > EMA200={ema200:.4f} ✔)"
    elif not pd.notna(ema50) or not pd.notna(ema200):
        detail["trend_filter"] = "FAIL (EMA not ready — need more candles)"
        return 0, detail
    else:
        detail["trend_filter"] = f"FAIL (EMA50={ema50:.4f} < EMA200={ema200:.4f} — downtrend)"
        return 0, detail

    # Oversold on 5m
    rsi5 = last5.get("rsi")
    if pd.notna(rsi5) and rsi5 < 40:
        score += 2
        detail["rsi_5m"] = f"+2 (RSI={rsi5:.1f} < 40)"
    else:
        rsi5_str = f"{rsi5:.1f}" if pd.notna(rsi5) else "n/a"
        detail["rsi_5m"] = f"0 (RSI={rsi5_str})"

    # ── 1-minute data ──
    df1 = fetch_ohlcv(symbol, "1m", limit=100)
    if df1 is None or len(df1) < 20:
        return score, detail
    df1 = add_indicators(df1)

    # RSI SMA for crossover detection
    df1["rsi_sma"] = df1["rsi"].rolling(RSI_SMA_PERIOD).mean()

    last1 = df1.iloc[-1]
    prev1 = df1.iloc[-2]

    # 1m RSI crossover above its SMA
    rsi1      = last1.get("rsi");    rsi1_sma  = last1.get("rsi_sma")
    rsi1_prev = prev1.get("rsi");   rsi1_sma_prev = prev1.get("rsi_sma")
    rsi_cross = (
        pd.notna(rsi1) and pd.notna(rsi1_sma) and
        pd.notna(rsi1_prev) and pd.notna(rsi1_sma_prev) and
        rsi1_prev < rsi1_sma_prev and rsi1 > rsi1_sma
    )
    if rsi_cross:
        score += 3
        detail["rsi_cross"] = f"+3 (1m RSI={rsi1:.1f} crossed above SMA={rsi1_sma:.1f})"
    else:
        rsi_str = f"{rsi1:.1f}" if pd.notna(rsi1) else "n/a"
        detail["rsi_cross"] = f"0 (no crossover, RSI={rsi_str})"

    # Volume spike
    vol  = last1.get("volume")
    vavg = last1.get("vol_avg")
    if pd.notna(vol) and pd.notna(vavg) and vol > 1.3 * vavg:
        score += 2
        detail["volume"] = f"+2 (vol spike {vol:.0f} > {1.3*vavg:.0f})"
    else:
        detail["volume"] = "0 (no volume spike)"

    # MACD histogram positive
    macd_hist = last1.get("macd_hist")
    if pd.notna(macd_hist) and macd_hist > 0:
        score += 2
        detail["macd"] = f"+2 (MACD hist={macd_hist:.4f})"
    else:
        detail["macd"] = "0 (MACD not positive)"

    # Near lower Bollinger Band (within 0.5% of it)
    bb_lower = last1.get("bb_lower")
    if pd.notna(bb_lower):
        dist = (last1["close"] - bb_lower) / last1["close"]
        if dist < 0.005:
            score += 1
            detail["bband"] = f"+1 (near lower BB, dist={dist*100:.2f}%)"
        else:
            detail["bband"] = f"0 (dist from BB={dist*100:.2f}%)"

    return score, detail


def print_scan_header(scan_num):
    ts = datetime.now().strftime("%H:%M:%S")
    print(Fore.YELLOW + f"\n{'━'*55}")
    print(Fore.YELLOW + f"  SCAN #{scan_num}  |  {ts}  |  {STRATEGY_NAME}")
    print(Fore.YELLOW + f"{'━'*55}")


def print_coin_score(symbol, score, detail):
    color = Fore.GREEN if score >= MIN_SCORE else (
            Fore.YELLOW if score >= 4 else Fore.WHITE)
    bar = "█" * score + "░" * (10 - score)
    trend = detail.get("trend_filter", "")
    trend_tag = f"  ← {trend}" if score == 0 and trend else ""
    print(color + f"  {symbol:<12}  [{bar}]  Score: {score}/10{trend_tag}")
    if score >= MIN_SCORE:
        for k, v in detail.items():
            print(f"    {k:<14}: {v}")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
PRICE_CHECK_INTERVAL = 5    # seconds between price checks for open positions

def monitor_positions(wallet, duration_seconds):
    """
    Monitors open positions every 5 seconds for the given duration.
    Exits immediately on TP / trailing stop / SL / timeout.
    Returns when duration is up or all positions are closed.
    """
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        if not wallet.has_position():
            remaining = end_time - time.time()
            if remaining > 0:
                time.sleep(min(PRICE_CHECK_INTERVAL, remaining))
            break

        price_map = {}
        for pos in list(wallet.positions):
            price = get_current_price(pos["coin"])
            if price:
                price_map[pos["coin"]] = price

        if price_map:
            # Print a compact tick line
            tick_parts = []
            for pos in wallet.positions:
                p = price_map.get(pos["coin"])
                if p:
                    pnl_pct = (p - pos["entry"]) / pos["entry"] * 100
                    trail = " ▲TRAIL" if pos["trail_active"] else ""
                    tick_parts.append(
                        f"{pos['coin']} ${p:.4f} ({pnl_pct:+.2f}%){trail}"
                    )
            if tick_parts:
                ts = datetime.now().strftime("%H:%M:%S")
                print(Fore.WHITE + f"  [{ts}] " + "  |  ".join(tick_parts))

            wallet.check_all_exits(price_map)

        remaining = end_time - time.time()
        if remaining > 0:
            time.sleep(min(PRICE_CHECK_INTERVAL, remaining))


def run():
    wallet   = VirtualWallet()
    scan_num = 0

    print(Fore.CYAN + f"""
╔══════════════════════════════════════════════╗
║   MULTI-COIN SCORING ENGINE  (Virtual $135)  ║
║   Strategy : {STRATEGY_NAME:<31}║
║   Min Score: {MIN_SCORE}/10  |  Coins: {len(COINS)}  |  Max Slots: 3      ║
║   Exit check: every {PRICE_CHECK_INTERVAL}s  |  Signal scan: every 60s   ║
╚══════════════════════════════════════════════╝
""")

    try:
        while True:
            scan_num += 1
            scan_start = time.time()
            print_scan_header(scan_num)

            # ── Display all open positions ──
            for pos in wallet.positions:
                price   = get_current_price(pos["coin"])
                elapsed = (datetime.now() - pos["opened_at"]).total_seconds() / 60
                trail   = f"  TRAIL@${pos['trail_sl']:.4f}" if pos["trail_active"] else ""
                if price:
                    pnl_pct = (price - pos["entry"]) / pos["entry"] * 100
                    print(Fore.CYAN +
                        f"  [HOLDING] {pos['coin']} @ ${pos['entry']:.4f}"
                        f"  |  Now: ${price:.4f} ({pnl_pct:+.2f}%)"
                        f"  |  TP: ${pos['tp']:.4f}  SL: ${pos['sl']:.4f}"
                        f"{trail}  |  {elapsed:.0f}min/90min")

            # ── Score all coins ──
            scored = []
            for coin in COINS:
                score, detail = score_coin(coin)
                print_coin_score(coin, score, detail)
                if score >= MIN_SCORE:
                    scored.append((coin, score, detail))

            scored.sort(key=lambda x: x[1], reverse=True)

            # ── Enter trades for qualifying coins ──
            for coin, score, detail in scored:
                if not wallet.can_open():
                    break
                if wallet.has_position_for(coin):
                    continue
                print(Fore.GREEN +
                    f"\n  ★ SIGNAL: {coin} scored {score}/10 — entering trade")
                price = get_current_price(coin)
                if price:
                    wallet.open_trade(coin, price, STRATEGY_NAME)

            if not scored:
                print(Fore.WHITE + f"\n  No signals ≥ {MIN_SCORE}. Waiting...")

            # ── How long did the scan take? ──
            scan_elapsed = time.time() - scan_start
            monitor_time = max(0, SCAN_INTERVAL - scan_elapsed)

            print(Fore.YELLOW +
                f"\n  Balance: ${wallet.balance:.2f}"
                f"  |  Positions: {wallet.position_count()}/{MAX_POSITIONS}"
                f"  |  Monitoring exits for {monitor_time:.0f}s (every {PRICE_CHECK_INTERVAL}s)...\n")

            # ── Real-time price monitoring until next scan ──
            monitor_positions(wallet, monitor_time)

    except KeyboardInterrupt:
        print(Fore.RED + "\n\n  [STOPPED] Bot stopped by user.")
        wallet.summary()


if __name__ == "__main__":
    run()