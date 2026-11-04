"""XAUUSD liquidity-sweep alert -> Telegram.

Rules (short side; the long side is the exact mirror):
  1. 4H, 2H and 1H trends are ALL down. If the three disagree, no alert.
       - 2H and 1H: "down" after a candle closes below the last confirmed swing low, until a candle
         closes above the last confirmed swing high (up is the mirror).
       - 4H (TREND_4H_MODE = "engulf"): a bullish 4H candle that is consumed (a close below its open)
         by the next 1-2 candles flips the trend to down; a bearish one consumed upward flips it to up.
         The trend holds until the opposite event.
  2. On 5m, take the most recent confirmed swing high that is lower than the swing high before it (the LH).
  3. A candle trades above the LH (the sweep), then price closes back below the LH:
       - wick sweep: the sweep candle itself closes below the LH, or
       - engulf sweep: within 3 candles a candle closes below the LOW of the candle that made the sweep high
         (and below the LH).
  4. The sweep must poke at least MIN_SWEEP_DEPTH beyond the level (filters tiny 20-cent pokes).
  5. Alert fires when that closing candle completes.
A swing high/low = highest/lowest of 2 candles on each side.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ---------------- settings you can tune ----------------
SYMBOL = "XAU/USD"
SWING_N = 2             # candles on each side for a swing high/low
ENGULF_WINDOW = 3       # max candles after the sweep starts to close back
REQUIRE_TRUE_LH = False # True = swept level must be a real lower high / higher low (not equal or flat levels)
CHECK_LAST = 3          # re-check last N closed 5m candles, so a delayed run doesn't miss one
STALE_MINUTES = 30      # ignore data older than this (market closed / feed stuck)
MIN_SWEEP_DEPTH = 0.50  # dollars the sweep must poke beyond the swept level
HTF = {"4H": 240, "2H": 120, "1H": 60}  # higher timeframes (minutes) that must all agree
TREND_4H_MODE = "engulf"  # "engulf" = 4H trend from candle engulfing; "structure" = swing-break trend
ENGULF_4H_WINDOW = 2    # a 4H candle counts as consumed if one of the next N 4H candles closes beyond its open
HTF_OFFSET_HOURS = 0    # shift higher-timeframe candle boundaries if they don't match your chart
FETCH_5M = 4000         # 5m candles fetched; higher timeframes are built from these (1 data call per run)
STATE_FILE = "state.json"
# --------------------------------------------------------

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
API_KEY = os.environ.get("TWELVEDATA_KEY", "")
DISPLAY_TZ = ZoneInfo(os.environ.get("DISPLAY_TZ", "UTC"))


def send_telegram(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text},
        timeout=30,
    )
    r.raise_for_status()


def fetch(interval, size):
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": SYMBOL, "interval": interval, "outputsize": size,
                "timezone": "UTC", "apikey": API_KEY},
        timeout=30,
    )
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error: {data}")
    df = pd.DataFrame(data["values"])
    df["time"] = pd.to_datetime(df["datetime"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)[["time", "open", "high", "low", "close"]]


def closed_only(df, minutes, now_utc):
    """Drop the candle that is still forming."""
    return df[df["time"] + timedelta(minutes=minutes) <= now_utc].reset_index(drop=True)


def mirror(df):
    """Flip prices upside down so the long setup can reuse the short logic."""
    m = df.copy()
    m["open"], m["close"] = -df["open"], -df["close"]
    m["high"], m["low"] = -df["low"], -df["high"]
    return m


def resample(df5, minutes, last_end):
    """Build higher-timeframe candles from 5m candles; keep only fully closed ones."""
    r = (df5.set_index("time")
         .resample(f"{minutes}min", offset=pd.Timedelta(hours=HTF_OFFSET_HOURS))
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
         .dropna()
         .reset_index())
    return r[r["time"] + timedelta(minutes=minutes) <= last_end].reset_index(drop=True)


def swings(df, kind):
    """Indices of confirmed swing highs ('high') or swing lows ('low')."""
    v = df[kind].values
    out = []
    for i in range(SWING_N, len(v) - SWING_N):
        left, right = v[i - SWING_N:i], v[i + 1:i + 1 + SWING_N]
        if kind == "high" and v[i] > left.max() and v[i] >= right.max():
            out.append(i)
        if kind == "low" and v[i] < left.min() and v[i] <= right.min():
            out.append(i)
    return out


def trend_of(d):
    """Structure-break trend (handles strong impulses straight away).
    UP once a candle closes above the last confirmed swing high; DOWN once one closes below the
    last confirmed swing low; it stays that way until the opposite break. None until the first break."""
    hi, lo, cl = d["high"].values, d["low"].values, d["close"].values
    known_high = {i + SWING_N: hi[i] for i in swings(d, "high")}   # a swing is known SWING_N candles later
    known_low = {i + SWING_N: lo[i] for i in swings(d, "low")}
    trend, last_high, last_low = None, None, None
    for j in range(len(d)):
        if j in known_high:
            last_high = known_high[j]
        if j in known_low:
            last_low = known_low[j]
        if last_high is not None and cl[j] > last_high:
            trend = "up"
        elif last_low is not None and cl[j] < last_low:
            trend = "down"
    return trend


def engulf_series(d):
    """Trend after each candle, from candle engulfing. A bullish candle is consumed when one of the next
    ENGULF_4H_WINDOW candles closes below its open -> trend becomes down. A bearish candle consumed by a
    close above its open -> trend becomes up. Holds until the opposite event."""
    o, c = d["open"].values, d["close"].values
    n = len(d)
    events = {}                                   # candle index where the flip is confirmed -> direction
    for k in range(n - 1):
        if c[k] == o[k]:
            continue
        bull = c[k] > o[k]
        for j in range(k + 1, min(k + 1 + ENGULF_4H_WINDOW, n)):
            if bull and c[j] < o[k]:
                events[j] = "down"
                break
            if not bull and c[j] > o[k]:
                events[j] = "up"
                break
    out, trend = [], None
    for j in range(n):
        trend = events.get(j, trend)
        out.append(trend)
    return out


def engulf_trend(d):
    series = engulf_series(d)
    return series[-1] if series else None


def compute_trends(part, end):
    trends = {}
    for name, minutes in HTF.items():
        d = resample(part, minutes, end)
        use_engulf = name == "4H" and TREND_4H_MODE == "engulf"
        trends[name] = engulf_trend(d) if use_engulf else trend_of(d)
    return trends


def find_short_sweep(sub):
    """Does the LAST candle of `sub` complete a sweep of a recent lower high?
    Returns dict(level, extreme, kind) or None. Uses only candles up to the last one."""
    t = len(sub) - 1
    sw = swings(sub, "high")
    for pos in range(len(sw) - 1, max(len(sw) - 6, -1), -1):   # latest 5 swing highs
        L = sw[pos]
        P = sub["high"][L]
        if REQUIRE_TRUE_LH:
            if pos == 0 or not P < sub["high"][sw[pos - 1]]:
                continue
        after = sub.iloc[L + 1:]
        beyond = after[after["high"] > P]
        if beyond.empty:
            continue
        s = beyond.index[0]                       # first candle that traded above the level
        if t - s > ENGULF_WINDOW:
            continue
        seg = sub.iloc[s:t]                       # candles s .. t-1
        if (seg["close"] < P).any() or sub["close"][t] >= P:
            continue                              # already closed back (alerted before) or not yet
        extreme = sub["high"].iloc[s:t + 1].max()
        if extreme - P < MIN_SWEEP_DEPTH:
            continue                              # poke too small to count as a real sweep
        if t == s:
            return {"level": P, "extreme": extreme, "kind": "wick sweep"}
        k = seg["high"].idxmax()                  # candle that made the sweep high
        if sub["close"][t] < sub["low"][k]:
            return {"level": P, "extreme": extreme, "kind": "engulf sweep"}
    return None


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"sent": []}


def save_state(state):
    state["sent"] = state["sent"][-50:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def backtest(days=3):
    """Prints every sweep found in the last `days` days, with the trends at that moment and whether
    it would have alerted. Sends nothing to Telegram. Use it to compare against setups you spotted."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    df5 = closed_only(fetch("5min", FETCH_5M), 5, now)
    mir = mirror(df5)
    n = len(df5)
    start = max(n - int(days * 288), 300)
    found = 0
    for t in range(start, n):
        end = df5["time"][t] + timedelta(minutes=5)
        for direction, src in (("down", df5), ("up", mir)):
            lo = max(0, t - 299)
            res = find_short_sweep(src.iloc[lo:t + 1].reset_index(drop=True))
            if not res:
                continue
            sign = 1 if direction == "down" else -1
            part = df5.iloc[:t + 1]
            trends = compute_trends(part, end)
            ok = len(set(trends.values())) == 1 and trends["4H"] == direction
            tag = "ALERT" if ok else "blocked (trends)"
            side = "SHORT" if direction == "down" else "LONG"
            found += 1
            print(f"{df5['time'][t]} UTC  {side:5} {res['kind']:12} level {sign * res['level']:.2f} "
                  f"depth {res['extreme'] - res['level']:.2f}  trends {trends}  -> {tag}")
    d4 = resample(df5, 240, df5["time"].iloc[-1] + timedelta(minutes=5))
    ser = engulf_series(d4)
    flips = sum(1 for a, b in zip(ser, ser[1:]) if b is not None and a is not None and a != b)
    print(f"\n4H candles (UTC, compare with your chart) and engulf trend; {flips} flips in {len(d4)} candles:")
    for i in range(max(0, len(d4) - 15), len(d4)):
        r = d4.iloc[i]
        print(f"  {r['time']}  open {r['open']:.2f}  close {r['close']:.2f}  trend {ser[i]}")
    print(f"done: {found} sweeps in the last {days} days (depth filter {MIN_SWEEP_DEPTH}, "
          f"true-LH rule {REQUIRE_TRUE_LH})")


def market_closed(now):
    wd, hr = now.weekday(), now.hour
    return wd == 5 or (wd == 4 and hr >= 22) or (wd == 6 and hr < 22)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        send_telegram("Test alert: your gold bot is connected.")
        print("test message sent")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest()
        return

    now = datetime.now(timezone.utc)
    now_naive = now.replace(tzinfo=None)
    if market_closed(now):
        print("market closed")
        return

    df5 = closed_only(fetch("5min", FETCH_5M), 5, now_naive)
    if df5.empty or now_naive - df5["time"].iloc[-1] > timedelta(minutes=STALE_MINUTES):
        print("5m data is stale, skipping")
        return

    last_end = df5["time"].iloc[-1] + timedelta(minutes=5)
    trends = compute_trends(df5, last_end)
    print("Trends:", trends)
    values = set(trends.values())
    if len(values) != 1 or None in values:
        print("timeframes disagree or no clear trend, no alert")
        return
    trend = values.pop()

    work = df5 if trend == "down" else mirror(df5)
    state = load_state()
    n = len(df5)
    changed = False
    for t in range(max(n - CHECK_LAST, 0), n):
        res = find_short_sweep(work.iloc[:t + 1].reset_index(drop=True))
        if not res:
            continue
        key = f"{trend}|{df5['time'][t]}"
        if key in state["sent"]:
            continue
        sign = 1 if trend == "down" else -1
        level, extreme = sign * res["level"], sign * res["extreme"]
        when = df5["time"][t].replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ)
        side = "SHORT setup (lower high swept)" if trend == "down" else "LONG setup (higher low swept)"
        msg = (f"XAUUSD {side}\n"
               f"Type: {res['kind']}\n"
               f"Swept level: {level:.2f}\n"
               f"Sweep extreme (stop reference): {extreme:.2f}\n"
               f"Trigger candle closed: {when:%d %b %H:%M} {when.tzname()}\n"
               f"Trend: 4H {trends['4H']}, 2H {trends['2H']}, 1H {trends['1H']}. Go check the chart.")
        send_telegram(msg)
        state["sent"].append(key)
        changed = True
        print("alert sent:", key)
    if changed:
        save_state(state)


if __name__ == "__main__":
    main()
