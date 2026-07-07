"""
strategy_scoring.py — Unified Dual-Mode Scoring Bot
====================================================
Detects market regime every scan using BTC indicators,
then applies the appropriate strategy:

  UPTREND   → Momentum mode  (buy breakouts, TP 0.6%, SL 0.3%)
  SIDEWAYS  → Mean reversion (buy dips,      TP 0.5%, SL 0.4%)
  DOWNTREND → Skip entirely  (protect capital)

One position at a time, $120 per trade.
Exit checks every 5 seconds. Signal scan every 60 seconds.
"""

import time
from datetime import datetime
from colorama import Fore, init

from core import (
    COINS, MAX_POSITIONS, MOMENTUM_TP, MOMENTUM_SL,
    REVERSION_TP, REVERSION_SL,
    VirtualWallet, fetch_ohlcv, add_indicators, get_current_price
)

init(autoreset=True)

SCAN_INTERVAL        = 60
PRICE_CHECK_INTERVAL = 5
RSI_SMA_PERIOD       = 14
MIN_SCORE            = 8    # 8/10 for both modes


# ─────────────────────────────────────────────
# MARKET REGIME DETECTOR
# ─────────────────────────────────────────────
def detect_regime():
    """
    Classifies current market as UPTREND, SIDEWAYS, or DOWNTREND
    using BTC 5m EMA20/EMA50 and RSI.
    Returns (regime, detail_str)
    """
    import pandas as pd

    df = fetch_ohlcv("BTC/USDT", "5m", limit=60)
    if df is None or len(df) < 55:
        return "SIDEWAYS", "BTC data unavailable — defaulting to sideways"

    df = add_indicators(df)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    last = df.iloc[-1]
    ema20  = last.get("ema20")
    ema50  = last.get("ema50")
    rsi    = last.get("rsi")

    if not pd.notna(ema20) or not pd.notna(ema50) or not pd.notna(rsi):
        return "SIDEWAYS", "BTC indicators not ready — defaulting to sideways"

    if ema20 > ema50 and rsi > 52:
        detail = f"BTC EMA20={ema20:.0f} > EMA50={ema50:.0f}, RSI={rsi:.1f} > 52"
        return "UPTREND", detail
    elif ema20 < ema50 and rsi < 48:
        detail = f"BTC EMA20={ema20:.0f} < EMA50={ema50:.0f}, RSI={rsi:.1f} < 48"
        return "DOWNTREND", detail
    else:
        detail = f"BTC EMA20={ema20:.0f} ≈ EMA50={ema50:.0f}, RSI={rsi:.1f} — ranging"
        return "SIDEWAYS", detail


# ─────────────────────────────────────────────
# MOMENTUM SCORING (Uptrend mode)
# ─────────────────────────────────────────────
def score_momentum(symbol):
    """
    Buys breakouts during uptrend.
    Max 10 pts, need 8.

    Hard gates:
      - 1m RSI must be 55–75  (breaking out, not overbought)
      - 5m RSI must be > 50   (macro confirmation)

    Scored:
      +3  1m RSI crossed above 55 from below
      +3  Volume > 2.0x average
      +2  MACD histogram positive and rising
      +2  Price above middle BB
    """
    import pandas as pd

    score  = 0
    detail = {}

    df1 = fetch_ohlcv(symbol, "1m", limit=100)
    if df1 is None or len(df1) < 20:
        return 0, {}
    df1 = add_indicators(df1)

    last1 = df1.iloc[-1]
    prev1 = df1.iloc[-2]
    rsi1  = last1.get("rsi")
    rsi1_prev = prev1.get("rsi")

    # ── HARD GATE 1: 1m RSI in 55–75 zone ──
    if not pd.notna(rsi1):
        detail["rsi_gate"] = "FAIL (RSI not ready)"
        return 0, detail
    if rsi1 < 55 or rsi1 > 75:
        detail["rsi_gate"] = f"FAIL (1m RSI={rsi1:.1f} — need 55–75)"
        return 0, detail
    detail["rsi_gate"] = f"PASS (1m RSI={rsi1:.1f} in breakout zone ✔)"

    # ── HARD GATE 2: 5m RSI > 50 ──
    df5 = fetch_ohlcv(symbol, "5m", limit=50)
    if df5 is None or len(df5) < 20:
        detail["rsi5_gate"] = "FAIL (5m data not ready)"
        return 0, detail
    df5 = add_indicators(df5)
    rsi5 = df5.iloc[-1].get("rsi")
    if not pd.notna(rsi5) or rsi5 <= 50:
        rsi5_str = f"{rsi5:.1f}" if pd.notna(rsi5) else "n/a"
        detail["rsi5_gate"] = f"FAIL (5m RSI={rsi5_str} ≤ 50 — no macro confirmation)"
        return 0, detail
    detail["rsi5_gate"] = f"PASS (5m RSI={rsi5:.1f} > 50 ✔)"

    # ── +3: 1m RSI crossed above 55 from below ──
    if pd.notna(rsi1_prev) and rsi1_prev < 55 and rsi1 >= 55:
        score += 3
        detail["rsi_cross"] = f"+3 (1m RSI crossed above 55: {rsi1_prev:.1f}→{rsi1:.1f} ✔)"
    else:
        prev_str = f"{rsi1_prev:.1f}" if pd.notna(rsi1_prev) else "n/a"
        detail["rsi_cross"] = f"0 (no 55 crossover — prev={prev_str} now={rsi1:.1f})"

    # ── +3: Volume > 2.0x average ──
    vol  = last1.get("volume")
    vavg = last1.get("vol_avg")
    if pd.notna(vol) and pd.notna(vavg) and vavg > 0 and vol > 2.0 * vavg:
        score += 3
        detail["volume"] = f"+3 (vol={vol:.0f} > 2x avg={vavg:.0f} ✔)"
    else:
        ratio = f"{vol/vavg:.2f}x" if (pd.notna(vol) and pd.notna(vavg) and vavg > 0) else "n/a"
        detail["volume"] = f"0 (vol={ratio} — need >2.0x)"

    # ── +2: MACD histogram positive and rising ──
    macd_now  = last1.get("macd_hist")
    macd_prev = prev1.get("macd_hist")
    if pd.notna(macd_now) and pd.notna(macd_prev) and macd_now > 0 and macd_now > macd_prev:
        score += 2
        detail["macd"] = f"+2 (MACD rising: {macd_prev:.5f}→{macd_now:.5f} ✔)"
    else:
        now_str = f"{macd_now:.5f}" if pd.notna(macd_now) else "n/a"
        detail["macd"] = f"0 (MACD not positive/rising — {now_str})"

    # ── +2: Price above middle BB ──
    bb_middle = last1.get("bb_middle")
    close     = last1["close"]
    if pd.notna(bb_middle) and close > bb_middle:
        score += 2
        detail["bb_pos"] = f"+2 (price ${close:.4f} > mid BB ${bb_middle:.4f} ✔)"
    else:
        mid_str = f"{bb_middle:.4f}" if pd.notna(bb_middle) else "n/a"
        detail["bb_pos"] = f"0 (price below mid BB={mid_str})"

    return score, detail


# ─────────────────────────────────────────────
# MEAN REVERSION SCORING (Sideways mode)
# ─────────────────────────────────────────────
def score_reversion(symbol):
    """
    Buys dips near lower BB during sideways market.
    Max 10 pts, need 8.

    Hard gates:
      - Price within 1.5% of lower BB
      - 1m RSI < 45

    Scored:
      +3  1m RSI crosses above its SMA
      +2  Volume > 1.3x average
      +2  MACD histogram > 0
      +2  5m RSI < 50
      +1  BB bandwidth > 0.3%
    """
    import pandas as pd

    score  = 0
    detail = {}

    df1 = fetch_ohlcv(symbol, "1m", limit=100)
    if df1 is None or len(df1) < 20:
        return 0, {}
    df1 = add_indicators(df1)
    df1["rsi_sma"] = df1["rsi"].rolling(RSI_SMA_PERIOD).mean()

    last1 = df1.iloc[-1]
    prev1 = df1.iloc[-2]
    close = last1["close"]

    bb_lower  = last1.get("bb_lower")
    bb_upper  = last1.get("bb_upper")
    bb_middle = last1.get("bb_middle")

    # ── HARD GATE 1: Price within 1.5% of lower BB ──
    if not pd.notna(bb_lower) or bb_lower <= 0:
        detail["bb_gate"] = "FAIL (BB not ready)"
        return 0, detail
    dist = (close - bb_lower) / close
    if dist > 0.015:
        detail["bb_gate"] = f"FAIL (price {dist*100:.2f}% above lower BB — need <1.5%)"
        return 0, detail
    detail["bb_gate"] = f"PASS ({dist*100:.2f}% from lower BB ✔)"

    # ── HARD GATE 2: 1m RSI < 45 ──
    rsi1 = last1.get("rsi")
    if not pd.notna(rsi1):
        detail["rsi_gate"] = "FAIL (RSI not ready)"
        return 0, detail
    if rsi1 >= 45:
        detail["rsi_gate"] = f"FAIL (1m RSI={rsi1:.1f} ≥ 45)"
        return 0, detail
    detail["rsi_gate"] = f"PASS (1m RSI={rsi1:.1f} < 45 ✔)"

    # ── +3: 1m RSI crosses above its SMA ──
    rsi1_sma      = last1.get("rsi_sma")
    rsi1_prev     = prev1.get("rsi")
    rsi1_sma_prev = prev1.get("rsi_sma")
    rsi_cross = (
        pd.notna(rsi1) and pd.notna(rsi1_sma) and
        pd.notna(rsi1_prev) and pd.notna(rsi1_sma_prev) and
        rsi1_prev < rsi1_sma_prev and rsi1 > rsi1_sma
    )
    if rsi_cross:
        score += 3
        detail["rsi_cross"] = f"+3 (1m RSI={rsi1:.1f} crossed SMA={rsi1_sma:.1f} ✔)"
    else:
        rsi_str = f"{rsi1:.1f}"     if pd.notna(rsi1)     else "n/a"
        sma_str = f"{rsi1_sma:.1f}" if pd.notna(rsi1_sma) else "n/a"
        detail["rsi_cross"] = f"0 (no crossover — RSI={rsi_str} SMA={sma_str})"

    # ── +2: Volume > 1.3x average ──
    vol  = last1.get("volume")
    vavg = last1.get("vol_avg")
    if pd.notna(vol) and pd.notna(vavg) and vavg > 0 and vol > 1.3 * vavg:
        score += 2
        detail["volume"] = f"+2 (vol={vol:.0f} > 1.3x avg={vavg:.0f} ✔)"
    else:
        ratio = f"{vol/vavg:.2f}x" if (pd.notna(vol) and pd.notna(vavg) and vavg > 0) else "n/a"
        detail["volume"] = f"0 (vol={ratio} — need >1.3x)"

    # ── +2: MACD histogram > 0 ──
    macd_now = last1.get("macd_hist")
    if pd.notna(macd_now) and macd_now > 0:
        score += 2
        detail["macd"] = f"+2 (MACD={macd_now:.5f} > 0 ✔)"
    else:
        now_str = f"{macd_now:.5f}" if pd.notna(macd_now) else "n/a"
        detail["macd"] = f"0 (MACD={now_str} — need >0)"

    # ── +2: 5m RSI < 50 ──
    df5 = fetch_ohlcv(symbol, "5m", limit=50)
    if df5 is not None and len(df5) >= 20:
        df5 = add_indicators(df5)
        rsi5 = df5.iloc[-1].get("rsi")
        if pd.notna(rsi5) and rsi5 < 50:
            score += 2
            detail["rsi_5m"] = f"+2 (5m RSI={rsi5:.1f} < 50 ✔)"
        else:
            rsi5_str = f"{rsi5:.1f}" if pd.notna(rsi5) else "n/a"
            detail["rsi_5m"] = f"0 (5m RSI={rsi5_str} ≥ 50)"
    else:
        detail["rsi_5m"] = "0 (5m data not ready)"

    # ── +1: BB bandwidth > 0.3% ──
    if pd.notna(bb_upper) and pd.notna(bb_middle) and bb_middle > 0:
        bw = (bb_upper - bb_lower) / bb_middle
        if bw > 0.003:
            score += 1
            detail["bb_width"] = f"+1 (bandwidth={bw*100:.2f}% > 0.3% ✔)"
        else:
            detail["bb_width"] = f"0 (bandwidth={bw*100:.2f}% — too tight)"
    else:
        detail["bb_width"] = "0 (BB width not ready)"

    return score, detail


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def print_scan_header(scan_num, regime, regime_detail):
    ts = datetime.now().strftime("%H:%M:%S")
    regime_colors = {
        "UPTREND":   Fore.GREEN,
        "SIDEWAYS":  Fore.YELLOW,
        "DOWNTREND": Fore.RED,
    }
    mode_labels = {
        "UPTREND":   "MOMENTUM  — buying breakouts",
        "SIDEWAYS":  "REVERSION — buying dips",
        "DOWNTREND": "SKIP      — protecting capital",
    }
    color = regime_colors.get(regime, Fore.WHITE)
    print(Fore.YELLOW + f"\n{'━'*60}")
    print(Fore.YELLOW + f"  SCAN #{scan_num}  |  {ts}")
    print(color + f"  Regime : {regime}  ({regime_detail})")
    print(color + f"  Mode   : {mode_labels[regime]}")
    print(Fore.YELLOW + f"{'━'*60}")


def print_coin_score(symbol, score, detail, regime):
    color = Fore.GREEN if score >= MIN_SCORE else (
            Fore.YELLOW if score >= 5 else Fore.WHITE)
    bar = "█" * min(score, 10) + "░" * (10 - min(score, 10))
    fail_tag = ""
    for v in detail.values():
        if "FAIL" in str(v):
            fail_tag = f"  ← {v}"
            break
    mode_tag = "[M]" if regime == "UPTREND" else "[R]"
    print(color + f"  {mode_tag} {symbol:<12}  [{bar}]  Score: {score}/10{fail_tag}")
    if score >= MIN_SCORE:
        for k, v in detail.items():
            print(f"      {k:<14}: {v}")


# ─────────────────────────────────────────────
# REAL-TIME EXIT MONITOR
# ─────────────────────────────────────────────
def monitor_positions(wallet, duration_seconds):
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
            tick_parts = []
            for pos in wallet.positions:
                p = price_map.get(pos["coin"])
                if p:
                    pnl_pct = (p - pos["entry"]) / pos["entry"] * 100
                    trail   = " ▲TRAIL" if pos["trail_active"] else ""
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


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run():
    wallet   = VirtualWallet()
    scan_num = 0

    print(Fore.CYAN + f"""
╔══════════════════════════════════════════════════════╗
║   DUAL-MODE SCORING BOT  (Virtual $135)              ║
║   UPTREND  → Momentum  (TP 0.6% / SL 0.3%)          ║
║   SIDEWAYS → Reversion (TP 0.5% / SL 0.4%)          ║
║   DOWNTREND→ Skip (capital protection)               ║
║   Min Score: {MIN_SCORE}/10  |  Exit check: every {PRICE_CHECK_INTERVAL}s          ║
╚══════════════════════════════════════════════════════╝
""")

    try:
        while True:
            scan_num += 1
            scan_start = time.time()

            # ── Detect market regime ──
            regime, regime_detail = detect_regime()
            print_scan_header(scan_num, regime, regime_detail)

            # ── Display open positions ──
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

            # ── Skip entirely if downtrend ──
            if regime == "DOWNTREND":
                print(Fore.RED + "\n  ⛔ Downtrend — no trades. Capital protected.")
            else:
                # ── Score all coins in the appropriate mode ──
                scored = []
                for coin in COINS:
                    if regime == "UPTREND":
                        score, detail = score_momentum(coin)
                    else:
                        score, detail = score_reversion(coin)
                    print_coin_score(coin, score, detail, regime)
                    if score >= MIN_SCORE:
                        scored.append((coin, score, detail))

                scored.sort(key=lambda x: x[1], reverse=True)

                # ── Enter best signal ──
                if scored and wallet.can_open():
                    coin, score, detail = scored[0]
                    print(Fore.GREEN +
                        f"\n  ★ SIGNAL: {coin} scored {score}/10"
                        f" [{regime} mode] — entering trade")
                    price = get_current_price(coin)
                    if price:
                        if regime == "UPTREND":
                            wallet.open_trade(coin, price,
                                f"Momentum ({regime})",
                                tp_pct=MOMENTUM_TP, sl_pct=MOMENTUM_SL)
                        else:
                            wallet.open_trade(coin, price,
                                f"Reversion ({regime})",
                                tp_pct=REVERSION_TP, sl_pct=REVERSION_SL)
                elif not scored:
                    print(Fore.WHITE +
                        f"\n  No signals ≥ {MIN_SCORE}/10 in {regime} mode. Waiting...")

            scan_elapsed = time.time() - scan_start
            monitor_time = max(0, SCAN_INTERVAL - scan_elapsed)

            print(Fore.YELLOW +
                f"\n  Balance: ${wallet.balance:.2f}"
                f"  |  Positions: {wallet.position_count()}/{MAX_POSITIONS}"
                f"  |  Monitoring for {monitor_time:.0f}s...\n")

            monitor_positions(wallet, monitor_time)

    except KeyboardInterrupt:
        print(Fore.RED + "\n\n  [STOPPED] Bot stopped by user.")
        wallet.summary()


if __name__ == "__main__":
    run()