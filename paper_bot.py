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
MIN_SCORE       = 9         # strict threshold — max possible is 11
RSI_SMA_PERIOD  = 14        # SMA of RSI for crossover detection

# ─────────────────────────────────────────────
# SCORING LOGIC
# ─────────────────────────────────────────────
def score_coin(symbol):
    """
    Strict scoring — max 11 pts, need 9 to trade.

    Hard gates (instant disqualify, 0 points):
      - EMA50 < EMA200 on 5m  → downtrend, skip
      - 5m RSI not in 20–35   → not genuinely oversold (or in freefall)

    Scored conditions:
      +3  5m RSI < 35                          (genuinely oversold macro)
      +3  1m RSI crosses SMA from below 35     (real recovery, not noise)
      +2  Volume > 2.0x average                (strong conviction)
      +2  MACD histogram flips positive        (momentum just turning)
      +1  Price within 0.3% of lower BB        (tight mean reversion)
    """
    import pandas as pd

    score  = 0
    detail = {}

    # ── 5-minute data (250 candles for EMA200 warmup) ──
    df5 = fetch_ohlcv(symbol, "5m", limit=250)
    if df5 is None or len(df5) < 210:
        return 0, {}
    df5 = add_indicators(df5)
    last5 = df5.iloc[-1]

    # ── HARD GATE 1: Trend filter — EMA50 > EMA200 ──
    ema50  = last5.get("ema50")
    ema200 = last5.get("ema200")
    if not pd.notna(ema50) or not pd.notna(ema200):
        detail["trend"] = "FAIL (EMA not ready)"
        return 0, detail
    if ema50 <= ema200:
        detail["trend"] = f"FAIL (EMA50={ema50:.4f} ≤ EMA200={ema200:.4f} — downtrend)"
        return 0, detail
    detail["trend"] = f"PASS (EMA50={ema50:.4f} > EMA200={ema200:.4f} ✔)"

    # ── HARD GATE 2: 5m RSI must be in 20–35 zone ──
    rsi5 = last5.get("rsi")
    if not pd.notna(rsi5):
        detail["rsi_5m_gate"] = "FAIL (RSI not ready)"
        return 0, detail
    if rsi5 < 20:
        detail["rsi_5m_gate"] = f"FAIL (RSI={rsi5:.1f} < 20 — freefall, avoid)"
        return 0, detail
    if rsi5 >= 35:
        detail["rsi_5m_gate"] = f"FAIL (RSI={rsi5:.1f} ≥ 35 — not oversold)"
        return 0, detail

    # Passed both hard gates — now score
    # ── +3: 5m RSI < 35 (already confirmed above) ──
    score += 3
    detail["rsi_5m"] = f"+3 (RSI={rsi5:.1f} in 20–35 zone ✔)"

    # ── 1-minute data ──
    df1 = fetch_ohlcv(symbol, "1m", limit=100)
    if df1 is None or len(df1) < 20:
        return score, detail
    df1 = add_indicators(df1)
    df1["rsi_sma"] = df1["rsi"].rolling(RSI_SMA_PERIOD).mean()

    last1 = df1.iloc[-1]
    prev1 = df1.iloc[-2]

    rsi1          = last1.get("rsi")
    rsi1_sma      = last1.get("rsi_sma")
    rsi1_prev     = prev1.get("rsi")
    rsi1_sma_prev = prev1.get("rsi_sma")

    # ── +3: 1m RSI crosses SMA from below AND prev RSI was below 35 ──
    rsi_cross = (
        pd.notna(rsi1) and pd.notna(rsi1_sma) and
        pd.notna(rsi1_prev) and pd.notna(rsi1_sma_prev) and
        rsi1_prev < rsi1_sma_prev and   # was below SMA
        rsi1 > rsi1_sma and             # now above SMA
        rsi1_prev < 35                  # was genuinely oversold before crossing
    )
    if rsi_cross:
        score += 3
        detail["rsi_cross"] = f"+3 (1m RSI={rsi1:.1f} crossed SMA={rsi1_sma:.1f} from oversold ✔)"
    else:
        rsi_str = f"{rsi1:.1f}" if pd.notna(rsi1) else "n/a"
        prev_str = f"{rsi1_prev:.1f}" if pd.notna(rsi1_prev) else "n/a"
        detail["rsi_cross"] = f"0 (no valid crossover — RSI prev={prev_str} now={rsi_str})"

    # ── +2: Volume > 2.0x average ──
    vol  = last1.get("volume")
    vavg = last1.get("vol_avg")
    if pd.notna(vol) and pd.notna(vavg) and vavg > 0 and vol > 2.0 * vavg:
        score += 2
        detail["volume"] = f"+2 (vol={vol:.0f} > 2x avg={vavg:.0f} ✔)"
    else:
        ratio = f"{vol/vavg:.2f}x" if (pd.notna(vol) and pd.notna(vavg) and vavg > 0) else "n/a"
        detail["volume"] = f"0 (vol={ratio} avg — need >2.0x)"

    # ── +2: MACD histogram flips from negative to positive ──
    macd_now  = last1.get("macd_hist")
    macd_prev = df1.iloc[-2].get("macd_hist")
    if (pd.notna(macd_now) and pd.notna(macd_prev) and
            macd_prev < 0 and macd_now > 0):
        score += 2
        detail["macd"] = f"+2 (MACD flipped +ve: {macd_prev:.5f} → {macd_now:.5f} ✔)"
    else:
        now_str  = f"{macd_now:.5f}"  if pd.notna(macd_now)  else "n/a"
        prev_str = f"{macd_prev:.5f}" if pd.notna(macd_prev) else "n/a"
        detail["macd"] = f"0 (no flip — prev={prev_str} now={now_str})"

    # ── +1: Price within 0.3% of lower Bollinger Band ──
    bb_lower = last1.get("bb_lower")
    if pd.notna(bb_lower) and bb_lower > 0:
        dist = (last1["close"] - bb_lower) / last1["close"]
        if dist < 0.003:
            score += 1
            detail["bband"] = f"+1 (dist from lower BB={dist*100:.2f}% < 0.3% ✔)"
        else:
            detail["bband"] = f"0 (dist from lower BB={dist*100:.2f}% — need <0.3%)"
    else:
        detail["bband"] = "0 (BB not ready)"

    return score, detail


def print_scan_header(scan_num):
    ts = datetime.now().strftime("%H:%M:%S")
    print(Fore.YELLOW + f"\n{'━'*55}")
    print(Fore.YELLOW + f"  SCAN #{scan_num}  |  {ts}  |  {STRATEGY_NAME}")
    print(Fore.YELLOW + f"{'━'*55}")


def print_coin_score(symbol, score, detail):
    color = Fore.GREEN if score >= MIN_SCORE else (
            Fore.YELLOW if score >= 6 else Fore.WHITE)
    filled = min(score, 11)
    bar = "█" * filled + "░" * (11 - filled)
    trend = detail.get("trend", detail.get("trend_filter", ""))
    is_fail = "FAIL" in trend or score == 0
    trend_tag = f"  ← {trend}" if is_fail and trend else ""
    print(color + f"  {symbol:<12}  [{bar}]  Score: {score}/11{trend_tag}")
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
║   Min Score : {MIN_SCORE}/11  |  Coins: {len(COINS)}  |  1 Position     ║
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
                print(Fore.WHITE + f"\n  No signals ≥ {MIN_SCORE}/11. Waiting for perfect setup...")

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