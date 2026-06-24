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
MIN_SCORE       = 9         # need 9/10 to trade
RSI_SMA_PERIOD  = 14        # SMA of RSI for crossover detection

# ─────────────────────────────────────────────
# SCORING LOGIC
# ─────────────────────────────────────────────
def score_coin(symbol):
    """
    Sideways scalping — max 10 pts, need 9 to trade.

    Hard gates (instant disqualify, 0 points):
      - Price NOT within 0.5% of lower BB  → not at range bottom
      - 1m RSI >= 40                        → not oversold on entry TF

    Scored conditions:
      +3  1m RSI crosses above its SMA      (momentum turning up)
      +2  Volume > 1.5x average             (buyers stepping in)
      +2  MACD histogram flips positive     (momentum confirmation)
      +2  5m RSI < 45                       (macro still in pullback)
      +1  BB bandwidth > 0.3%               (enough range to hit 0.5% TP)
    """
    import pandas as pd

    score  = 0
    detail = {}

    # ── 1-minute data (primary timeframe) ──
    df1 = fetch_ohlcv(symbol, "1m", limit=100)
    if df1 is None or len(df1) < 20:
        return 0, {}
    df1 = add_indicators(df1)
    df1["rsi_sma"] = df1["rsi"].rolling(RSI_SMA_PERIOD).mean()

    last1 = df1.iloc[-1]
    prev1 = df1.iloc[-2]

    # ── HARD GATE 1: Price must be within 0.5% of lower Bollinger Band ──
    bb_lower  = last1.get("bb_lower")
    bb_upper  = last1.get("bb_upper")
    bb_middle = last1.get("bb_middle")
    close     = last1["close"]

    if not pd.notna(bb_lower) or not pd.notna(bb_upper) or bb_lower <= 0:
        detail["bb_gate"] = "FAIL (BB not ready)"
        return 0, detail

    dist_from_lower = (close - bb_lower) / close
    if dist_from_lower > 0.005:
        detail["bb_gate"] = f"FAIL (price {dist_from_lower*100:.2f}% above lower BB — not at range bottom)"
        return 0, detail
    detail["bb_gate"] = f"PASS (price {dist_from_lower*100:.2f}% from lower BB ✔)"

    # ── HARD GATE 2: 1m RSI must be below 40 ──
    rsi1 = last1.get("rsi")
    if not pd.notna(rsi1):
        detail["rsi_gate"] = "FAIL (RSI not ready)"
        return 0, detail
    if rsi1 >= 40:
        detail["rsi_gate"] = f"FAIL (1m RSI={rsi1:.1f} ≥ 40 — not oversold)"
        return 0, detail
    detail["rsi_gate"] = f"PASS (1m RSI={rsi1:.1f} < 40 ✔)"

    # Both gates passed — now score

    # ── +3: 1m RSI crosses above its SMA ──
    rsi1_sma      = last1.get("rsi_sma")
    rsi1_prev     = prev1.get("rsi")
    rsi1_sma_prev = prev1.get("rsi_sma")

    rsi_cross = (
        pd.notna(rsi1) and pd.notna(rsi1_sma) and
        pd.notna(rsi1_prev) and pd.notna(rsi1_sma_prev) and
        rsi1_prev < rsi1_sma_prev and
        rsi1 > rsi1_sma
    )
    if rsi_cross:
        score += 3
        detail["rsi_cross"] = f"+3 (1m RSI={rsi1:.1f} crossed above SMA={rsi1_sma:.1f} ✔)"
    else:
        rsi_str  = f"{rsi1:.1f}"      if pd.notna(rsi1)      else "n/a"
        sma_str  = f"{rsi1_sma:.1f}"  if pd.notna(rsi1_sma)  else "n/a"
        detail["rsi_cross"] = f"0 (no crossover — RSI={rsi_str}, SMA={sma_str})"

    # ── +2: Volume > 1.5x average ──
    vol  = last1.get("volume")
    vavg = last1.get("vol_avg")
    if pd.notna(vol) and pd.notna(vavg) and vavg > 0 and vol > 1.5 * vavg:
        score += 2
        detail["volume"] = f"+2 (vol={vol:.0f} > 1.5x avg={vavg:.0f} ✔)"
    else:
        ratio = f"{vol/vavg:.2f}x" if (pd.notna(vol) and pd.notna(vavg) and vavg > 0) else "n/a"
        detail["volume"] = f"0 (vol={ratio} — need >1.5x)"

    # ── +2: MACD histogram flips from negative to positive ──
    macd_now  = last1.get("macd_hist")
    macd_prev = prev1.get("macd_hist")
    if pd.notna(macd_now) and pd.notna(macd_prev) and macd_prev < 0 and macd_now > 0:
        score += 2
        detail["macd"] = f"+2 (MACD flipped: {macd_prev:.5f} → {macd_now:.5f} ✔)"
    else:
        now_str  = f"{macd_now:.5f}"  if pd.notna(macd_now)  else "n/a"
        prev_str = f"{macd_prev:.5f}" if pd.notna(macd_prev) else "n/a"
        detail["macd"] = f"0 (no flip — prev={prev_str} now={now_str})"

    # ── +2: 5m RSI < 45 (macro pullback) ──
    df5 = fetch_ohlcv(symbol, "5m", limit=50)
    if df5 is not None and len(df5) >= 20:
        df5 = add_indicators(df5)
        rsi5 = df5.iloc[-1].get("rsi")
        if pd.notna(rsi5) and rsi5 < 45:
            score += 2
            detail["rsi_5m"] = f"+2 (5m RSI={rsi5:.1f} < 45 — macro pullback ✔)"
        else:
            rsi5_str = f"{rsi5:.1f}" if pd.notna(rsi5) else "n/a"
            detail["rsi_5m"] = f"0 (5m RSI={rsi5_str} ≥ 45)"
    else:
        detail["rsi_5m"] = "0 (5m data not ready)"

    # ── +1: BB bandwidth > 0.3% (enough volatility to hit TP) ──
    if pd.notna(bb_upper) and pd.notna(bb_middle) and bb_middle > 0:
        bandwidth = (bb_upper - bb_lower) / bb_middle
        if bandwidth > 0.003:
            score += 1
            detail["bb_width"] = f"+1 (BB bandwidth={bandwidth*100:.2f}% > 0.3% ✔)"
        else:
            detail["bb_width"] = f"0 (BB bandwidth={bandwidth*100:.2f}% — too tight, <0.3%)"
    else:
        detail["bb_width"] = "0 (BB width not ready)"

    return score, detail


def print_scan_header(scan_num):
    ts = datetime.now().strftime("%H:%M:%S")
    print(Fore.YELLOW + f"\n{'━'*55}")
    print(Fore.YELLOW + f"  SCAN #{scan_num}  |  {ts}  |  {STRATEGY_NAME}")
    print(Fore.YELLOW + f"{'━'*55}")


def print_coin_score(symbol, score, detail):
    color = Fore.GREEN if score >= MIN_SCORE else (
            Fore.YELLOW if score >= 6 else Fore.WHITE)
    filled = min(score, 10)
    bar = "█" * filled + "░" * (10 - filled)
    # Show first FAIL gate reason inline
    fail_tag = ""
    for k, v in detail.items():
        if "FAIL" in str(v):
            fail_tag = f"  ← {v}"
            break
    print(color + f"  {symbol:<12}  [{bar}]  Score: {score}/10{fail_tag}")
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
║   Min Score : {MIN_SCORE}/10  |  Coins: {len(COINS)}  |  1 Position     ║
║   TP: 0.5%  SL: 0.4%  Trail: 0.3%           ║
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
                print(Fore.WHITE + f"\n  No signals ≥ {MIN_SCORE}/10. Waiting for setup...")

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