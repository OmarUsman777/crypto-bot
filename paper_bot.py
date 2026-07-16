"""
strategy_grid.py — Grid Trading Bot Runner
==========================================
Imports GridEngine and helpers from core.py.
Run this file to start the grid bot:

    python strategy_grid.py

How it works:
  1. Scans all COINS and picks the most volatile + liquid one
  2. Builds a grid of 8 levels (4 buys below, 4 sells above price)
  3. Every 30s: checks if price crossed any level
     - Price drops to buy level  → virtual buy fill
     - Price rises to sell level → virtual sell fill + lock profit
  4. Resets grid every 60 min or when price exits the range entirely
  5. Stop loss if price drops 2 levels below grid bottom
"""

import time
from datetime import datetime
from colorama import Fore, init

from core import (
    COINS, STARTING_BALANCE, FEE_PCT,
    GRID_LEVELS, GRID_SPACING_PCT, CAPITAL_PER_LEVEL,
    STOP_LOSS_LEVELS, GRID_RESET_MINS, GRID_LOG_FILE,
    get_current_price, select_best_grid_coin, GridEngine
)

init(autoreset=True)

CHECK_INTERVAL   = 30     # seconds between price checks
PROFIT_PER_CYCLE = GRID_SPACING_PCT - (FEE_PCT * 2)


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def print_header(scan_num, coin):
    ts = datetime.now().strftime("%H:%M:%S")
    print(Fore.YELLOW + f"\n{'━'*65}")
    print(Fore.YELLOW + f"  SCAN #{scan_num}  |  {ts}  |  GRID BOT  |  {coin}")
    print(Fore.YELLOW + f"{'━'*65}")


def print_grid_state(grid, current_price):
    print(Fore.CYAN + f"\n  {'LVL':<6}  {'PRICE':>10}  {'TYPE':<5}  {'STATE':<14}  HOLDING")
    print(Fore.CYAN + f"  {'─'*58}")

    for lv in reversed(grid.levels):
        dist      = (current_price - lv["price"]) / lv["price"] * 100
        is_near   = abs(dist) < GRID_SPACING_PCT * 100 * 0.5

        if lv["coin_held"] > 0:
            color = Fore.GREEN
        elif lv["type"] == "buy" and lv["state"] == "watch_buy":
            color = Fore.CYAN
        elif is_near:
            color = Fore.YELLOW
        else:
            color = Fore.WHITE

        holding_str = f"{lv['coin_held']:.6f}" if lv["coin_held"] > 0 else "—"
        arrow       = "  ◄ NOW" if is_near else f"  ({dist:+.2f}%)"
        print(color +
            f"  [{lv['index']:+d}]    "
            f"  ${lv['price']:>10.4f}"
            f"  {lv['type']:<5}"
            f"  {lv['state']:<14}"
            f"  {holding_str}{arrow}")

    print()
    print(Fore.RED   + f"  Stop Loss : ${grid.stop_loss_price:.4f}")
    print(Fore.WHITE + f"  Current   : ${current_price:.4f}  "
          f"(grid center: ${grid.center:.4f})\n")


def print_event(event, coin):
    action = event[0]
    price  = event[1]
    level  = event[2]
    if action == "BUY":
        print(Fore.GREEN +
            f"\n  ▶  BUY   {coin}  @  ${price:.4f}  "
            f"[Level {level:+d}]  — ${CAPITAL_PER_LEVEL:.2f} deployed")
    elif action == "SELL":
        profit = event[3]
        color  = Fore.GREEN if profit > 0 else Fore.RED
        print(color +
            f"\n  ◀  SELL  {coin}  @  ${price:.4f}  "
            f"[Level {level:+d}]  — Profit: ${profit:+.4f}")
    elif action == "STOP_LOSS":
        print(Fore.RED + f"\n  ⛔ STOP LOSS triggered  @  ${price:.4f}")


def print_tick(coin, price, grid):
    ts        = datetime.now().strftime("%H:%M:%S")
    elapsed   = (datetime.now() - grid.created_at).total_seconds() / 60
    pnl_color = Fore.GREEN if grid.total_profit >= 0 else Fore.RED
    active_buys  = sum(1 for lv in grid.levels if lv["state"] == "watch_buy")
    held_sells   = sum(1 for lv in grid.levels if lv["coin_held"] > 0)
    print(Fore.WHITE +
        f"  [{ts}]  {coin}  ${price:.4f}"
        f"  |  Cycles: {grid.cycles}"
        f"  |  {pnl_color}PnL: ${grid.total_profit:+.4f}{Fore.RESET}"
        f"  |  Buys: {active_buys}  Sells pending: {held_sells}"
        f"  |  Grid: {elapsed:.0f}/{GRID_RESET_MINS}min"
        f"  |  Bal: ${grid.balance:.2f}")


def print_session_summary(grid):
    wins   = sum(1 for t in grid.trades if t[2] == "SELL" and float(t[6]) > 0)
    losses = sum(1 for t in grid.trades if t[2] == "SELL" and float(t[6]) <= 0)
    print(Fore.YELLOW + f"\n{'━'*65}")
    print(Fore.YELLOW + "  GRID SESSION SUMMARY")
    print(f"  Coin          : {grid.coin}")
    print(f"  Grid cycles   : {grid.cycles}  (W:{wins} L:{losses})")
    print(f"  Total trades  : {len(grid.trades)}")
    pnl_color = Fore.GREEN if grid.total_profit >= 0 else Fore.RED
    print(pnl_color + f"  Total PnL     : ${grid.total_profit:+.4f}")
    print(f"  Final balance : ${grid.balance:.2f}  (started ${grid.starting:.2f})")
    print(f"  Log file      : {GRID_LOG_FILE}")
    print(Fore.YELLOW + f"{'━'*65}\n")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run():
    balance  = STARTING_BALANCE
    scan_num = 0
    grid     = None
    coin     = None

    print(Fore.CYAN + f"""
╔═══════════════════════════════════════════════════════════════╗
║         VIRTUAL GRID TRADING BOT  (${STARTING_BALANCE:.0f} wallet)             ║
║                                                               ║
║  Grid spacing  : {GRID_SPACING_PCT*100:.1f}%  ({GRID_LEVELS} levels × ${CAPITAL_PER_LEVEL:.0f}/level = ${CAPITAL_PER_LEVEL*GRID_LEVELS:.0f} total) ║
║  Net per cycle : ~{PROFIT_PER_CYCLE*100:.2f}% after fees                      ║
║  Check interval: every {CHECK_INTERVAL}s                               ║
║  Grid reset    : every {GRID_RESET_MINS}min or when price exits range  ║
║  Stop loss     : {STOP_LOSS_LEVELS} levels below grid bottom                  ║
╚═══════════════════════════════════════════════════════════════╝
""")

    try:
        while True:
            scan_num += 1

            # ── Build or rebuild grid when needed ──
            need_new_grid = grid is None or grid.time_for_reset()

            if need_new_grid:
                if grid is not None:
                    print_session_summary(grid)
                    balance = grid.balance

                print(Fore.YELLOW + f"\n{'━'*65}")
                print(Fore.YELLOW +
                    f"  BUILDING NEW GRID  |  {datetime.now().strftime('%H:%M:%S')}")
                print(Fore.YELLOW + f"{'━'*65}")

                coin, price = select_best_grid_coin()
                if coin is None:
                    print(Fore.RED + "  No coin data available. Retrying in 60s...")
                    time.sleep(60)
                    continue

                grid = GridEngine(coin, price, balance)

                prices = [lv["price"] for lv in grid.levels]
                print(Fore.CYAN + f"\n  Grid built for {coin} @ ${price:.4f}")
                print(f"  Range   : ${min(prices):.4f}  ↔  ${max(prices):.4f}")
                print(f"  Levels  : {GRID_LEVELS}  ({GRID_LEVELS//2} buys below, {GRID_LEVELS//2} sells above)")
                print(f"  Spacing : {GRID_SPACING_PCT*100:.1f}% per level")
                print(f"  Capital : ${CAPITAL_PER_LEVEL:.0f}/level  (${CAPITAL_PER_LEVEL*GRID_LEVELS:.0f} deployed)")
                print(f"  SL      : ${grid.stop_loss_price:.4f}")

            # ── Print scan header ──
            print_header(scan_num, coin)

            # ── Fetch price ──
            current_price = get_current_price(coin)
            if current_price is None:
                print(Fore.RED + "  Price fetch failed — retrying next cycle...")
                time.sleep(CHECK_INTERVAL)
                continue

            # ── Rebuild if price exits grid range ──
            if grid.needs_reset(current_price):
                print(Fore.YELLOW +
                    f"  ⚠  Price ${current_price:.4f} outside grid range — rebuilding...")
                print_session_summary(grid)
                balance = grid.balance
                grid    = GridEngine(coin, current_price, balance)
                prices  = [lv["price"] for lv in grid.levels]
                print(Fore.CYAN +
                    f"  New grid: ${min(prices):.4f} ↔ ${max(prices):.4f}  "
                    f"@ ${current_price:.4f}")

            # ── Run grid logic ──
            events = grid.check_price(current_price)

            stop_triggered = False
            for event in events:
                print_event(event, coin)
                if event[0] == "STOP_LOSS":
                    stop_triggered = True

            if stop_triggered:
                print(Fore.RED + "  Rebuilding grid after stop loss...")
                print_session_summary(grid)
                balance = grid.balance
                grid    = None
                time.sleep(5)
                continue

            # ── Print grid visual ──
            print_grid_state(grid, current_price)

            # ── Print live tick ──
            print_tick(coin, current_price, grid)

            print(Fore.YELLOW +
                f"\n  Balance: ${grid.balance:.2f}"
                f"  |  Next check in {CHECK_INTERVAL}s...\n")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print(Fore.RED + "\n\n  [STOPPED] Bot stopped by user.")
        if grid:
            print_session_summary(grid)


if __name__ == "__main__":
    run()