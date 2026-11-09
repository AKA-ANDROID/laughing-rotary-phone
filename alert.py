"""XAUUSD liquidity-sweep alert -> Telegram.

Rules (short side; the long side is the exact mirror):
  1. 4H and 2H trends are BOTH down. If they disagree, no alert.
       - 2H: "down" after a candle closes below the last confirmed swing low, until a candle
         closes above the last confirmed swing high (up is the mirror).
       - 4H (TREND_4H_MODE = "engulf"): a bullish 4H candle that is consumed (a close below its open)
         by the next 1-2 candles flips the trend to down; a bearish one consumed upward flips it to up.
         The trend holds until the opposite event.
  2. On the sweep timeframe (SWEEP_TF, 15m by default), take the most recent confirmed swing high (the level).
  3. A candle trades above the LH (the sweep), then price closes back below the LH:
       - wick sweep: the sweep candle itself closes below the LH, or
       - engulf sweep: within 3 candles a candle closes below the LOW of the candle that made the sweep high
         (and below the LH).
  4. The sweep must poke at least MIN_SWEEP_DEPTH[SWEEP_TF] beyond the level (filters tiny pokes).
  5. Follow-through (REQUIRE_FOLLOW_THROUGH): a wick sweep only alerts once a later candle, within 3 candles,
     closes beyond the far end of the sweep candle (it is "engulfed"). An engulf sweep already includes this.
  6. The alert fires when that confirming candle closes.
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
SWEEP_TF = 15           # minutes: candle size used to find sweeps (5 or 15)
CHECK_MINUTES = 30      # re-check sweep candles closed in the last N minutes, so a delayed run doesn't miss one
STALE_MINUTES = 30      # ignore data older than this (market closed / feed stuck)
REQUIRE_FOLLOW_THROUGH = True  # wick sweeps must be engulfed by a later candle (within ENGULF_WINDOW) before alerting
MIN_SWEEP_DEPTH = {5: 0.50, 15: 1.00}  # dollars the sweep must poke beyond the swept level, per timeframe
HTF = {"4H": 240, "2H": 120}  # higher timeframes (minutes) that must agree
TREND_4H_MODE = "engulf"  # "engulf" = 4H trend from candle engulfing; "structure" = swing-break trend
ENGULF_4H_WINDOW = 2    # a 4H candle counts as consumed if one of the next N 4H candles closes beyond its open
HTF_OFFSET_HOURS = "auto"  # "auto" = candles start at 17:00 New York time (like TradingView forex/gold); or a number of hours
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


def htf_offset():
    """Where higher-timeframe candles start. TradingView forex/gold days start at 17:00 New York time,
    which is 22:00 UTC in winter and 21:00 UTC in summer."""
    if HTF_OFFSET_HOURS != "auto":
        return pd.Timedelta(hours=HTF_OFFSET_HOURS)
    ny = datetime.now(ZoneInfo("America/New_York")).utcoffset().total_seconds() / 3600   # -5 winter, -4 summer
    return pd.Timedelta(hours=(17 - ny) % 4)


def resample(df5, minutes, last_end, offset=None):
    """Build higher-timeframe candles from 5m candles; keep only fully closed ones."""
    r = (df5.set_index("time")
         .resample(f"{minutes}min", offset=htf_offset() if offset is None else offset)
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


def find_short_sweep(sub, min_depth=None, trace=None, sign=1):
    """Does the LAST candle of `sub` complete a sweep of a recent swing high?
    Returns dict(level, extreme, kind) or None. Uses only candles up to the last one.
    If `trace` is a list, a line explaining each rejection is appended to it (used by explain mode)."""
    if min_depth is None:
        min_depth = MIN_SWEEP_DEPTH[SWEEP_TF]
    say = trace.append if trace is not None else (lambda _x: None)
    T = lambda i: f"{sub['time'][i]:%H:%M}"
    t = len(sub) - 1
    sw = swings(sub, "high")
    if not sw:
        say("no confirmed swing level found")
    for pos in range(len(sw) - 1, max(len(sw) - 6, -1), -1):   # latest 5 swing highs
        L = sw[pos]
        P = sub["high"][L]
        tag = f"level {sign * P:.2f} (swing at {T(L)} UTC)"
        if REQUIRE_TRUE_LH:
            if pos == 0 or not P < sub["high"][sw[pos - 1]]:
                say(f"{tag}: skipped, not a lower high / higher low")
                continue
        after = sub.iloc[L + 1:]
        beyond = after[after["high"] > P]
        if beyond.empty:
            say(f"{tag}: not pierced yet")
            continue
        s_ = beyond.index[0]                      # first candle that traded beyond the level
        if t - s_ > ENGULF_WINDOW:
            say(f"{tag}: first pierced at {T(s_)} UTC, more than {ENGULF_WINDOW} candles ago")
            continue
        seg = sub.iloc[s_:t]                      # candles s_ .. t-1
        if (seg["close"] < P).any():
            say(f"{tag}: pierced at {T(s_)} UTC and already closed back inside earlier")
            continue
        if sub["close"][t] >= P:
            say(f"{tag}: pierced at {T(s_)} UTC, this candle has not closed back inside yet")
            continue
        extreme = sub["high"].iloc[s_:t + 1].max()
        if extreme - P < min_depth:
            say(f"{tag}: pierced at {T(s_)} UTC but depth {extreme - P:.2f} is below the minimum {min_depth}")
            continue
        if t == s_:
            return {"level": P, "extreme": extreme, "kind": "wick sweep", "far": sub["low"][t]}
        k = seg["high"].idxmax()                  # candle that made the sweep extreme
        if sub["close"][t] < sub["low"][k]:
            return {"level": P, "extreme": extreme, "kind": "engulf sweep", "far": sub["low"][k]}
        say(f"{tag}: pierced at {T(s_)} UTC, waiting for a close beyond the sweep candle's far end")
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


def sweep_candles(df5, tf):
    """Candles used to look for sweeps (closed candles only)."""
    if tf == 5:
        return df5
    last_end = df5["time"].iloc[-1] + timedelta(minutes=5)
    return resample(df5, tf, last_end, offset=pd.Timedelta(0))


def scan(df5, tf, days):
    """Find every sweep on `tf`-minute candles over the last `days` days (shorts and longs)."""
    dfs = sweep_candles(df5, tf)
    mir = mirror(dfs)
    n = len(dfs)
    start = max(n - int(days * 1440 / tf), 100)
    out = []
    for t in range(start, n):
        end = dfs["time"][t] + timedelta(minutes=tf)
        for direction, src in (("down", dfs), ("up", mir)):
            lo = max(0, t - 299)
            res = find_short_sweep(src.iloc[lo:t + 1].reset_index(drop=True), MIN_SWEEP_DEPTH[tf])
            if not res:
                continue
            part = df5[df5["time"] + timedelta(minutes=5) <= end]
            trends = compute_trends(part, end)
            trend_ok = len(set(trends.values())) == 1 and trends["4H"] == direction
            sign = 1 if direction == "down" else -1
            conf = "yes (engulf sweep)"
            if res["kind"] == "wick sweep":
                conf = f"NO follow-through in next {ENGULF_WINDOW}"
                for j in range(1, ENGULF_WINDOW + 1):
                    if t + j >= n:
                        conf = "too recent to know"
                        break
                    if src["close"].iloc[t + j] < res["far"]:      # closes beyond the sweep candle's far end
                        conf = f"yes, engulfed after {j} candle(s)"
                        break
            out.append({"time": dfs["time"][t], "side": "SHORT" if direction == "down" else "LONG",
                        "kind": res["kind"], "level": sign * res["level"],
                        "depth": res["extreme"] - res["level"], "trends": trends, "trend_ok": trend_ok,
                        "ok": trend_ok and (conf.startswith("yes") or not REQUIRE_FOLLOW_THROUGH),
                        "confirm": conf})
    return out


def backtest(days=5):
    """Prints every sweep found in the last `days` days with the trends at that moment and whether it
    would have alerted, then compares alerts per day for 5m and 15m sweeps. Sends nothing to Telegram."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    df5 = closed_only(drop_closed_hours(fetch("5min", FETCH_5M)), 5, now)
    other = 5 if SWEEP_TF != 5 else 15
    results = {tf: scan(df5, tf, days) for tf in (SWEEP_TF, other)}

    print(f"=== {SWEEP_TF}m sweeps, last {days} trading days (time = candle OPEN; it closes {SWEEP_TF} min later) ===")
    for r in results[SWEEP_TF]:
        if r["ok"]:
            tag = "ALERT"
        elif not r["trend_ok"]:
            tag = "blocked (trends)"
        elif r["confirm"].startswith("too recent"):
            tag = "waiting for follow-through"
        else:
            tag = "no alert (no follow-through)"
        utc = r["time"].replace(tzinfo=timezone.utc)
        loc = utc.astimezone(DISPLAY_TZ)
        print(f"{utc:%a %d %b %H:%M} UTC = {loc:%a %d %b %H:%M} {loc.tzname()}  {r['side']:5} {r['kind']:12} "
              f"level {r['level']:.2f} depth {r['depth']:.2f}  trends {r['trends']}  -> {tag}  | follow-through: {r['confirm']}")

    passed = [r for r in results[SWEEP_TF] if r["trend_ok"]]
    alerts = [r for r in passed if r["ok"]]
    print(f"\n{SWEEP_TF}m: {len(passed)} sweeps passed the trend gate; {len(alerts)} of them had follow-through "
          f"(sweep candle engulfed within {ENGULF_WINDOW} candles) and would alert")
    print(f"\n=== alerts per day ({DISPLAY_TZ.key} dates), by sweep timeframe ===")
    for tf, recs in results.items():
        per_day = {}
        for r in recs:
            if r["ok"]:
                day = r["time"].replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ).date()
                per_day[day] = per_day.get(day, 0) + 1
        days_txt = ", ".join(f"{d} {c}" for d, c in sorted(per_day.items())) or "none"
        print(f"{tf}m (min depth {MIN_SWEEP_DEPTH[tf]}): {sum(per_day.values())} alerts; per day: {days_txt}")

    d4 = resample(df5, 240, df5["time"].iloc[-1] + timedelta(minutes=5))
    ser = engulf_series(d4)
    flips = sum(1 for a, b in zip(ser, ser[1:]) if b is not None and a is not None and a != b)
    print(f"\n4H candles (UTC, compare with your chart) and engulf trend; {flips} flips in {len(d4)} candles:")
    for i in range(max(0, len(d4) - 15), len(d4)):
        r = d4.iloc[i]
        print(f"  {r['time']}  open {r['open']:.2f}  close {r['close']:.2f}  trend {ser[i]}")
    print(f"done (true-LH rule {REQUIRE_TRUE_LH})")


def explain(start_txt, end_txt):
    """For each candle on the sweep timeframe between two UTC times, print its prices and why a long or a
    short sweep was or was not found. Sends nothing to Telegram."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    df5 = closed_only(drop_closed_hours(fetch("5min", FETCH_5M)), 5, now)
    dfs = sweep_candles(df5, SWEEP_TF)
    mir = mirror(dfs)
    t0, t1 = pd.Timestamp(start_txt), pd.Timestamp(end_txt)
    print(f"=== explain: {SWEEP_TF}m candles from {t0} to {t1} UTC (min depth {MIN_SWEEP_DEPTH[SWEEP_TF]}) ===")
    for t in range(len(dfs)):
        ts = dfs["time"][t]
        if ts < t0 or ts > t1:
            continue
        utc = ts.replace(tzinfo=timezone.utc)
        loc = utc.astimezone(DISPLAY_TZ)
        r = dfs.iloc[t]
        end = ts + timedelta(minutes=SWEEP_TF)
        trends = compute_trends(df5[df5["time"] + timedelta(minutes=5) <= end], end)
        print(f"\n{utc:%a %d %b %H:%M} UTC = {loc:%H:%M} {loc.tzname()}  open {r['open']:.2f} high {r['high']:.2f} "
              f"low {r['low']:.2f} close {r['close']:.2f}  trends {trends}")
        for name, src, sign in (("LONG ", mir, -1), ("SHORT", dfs, 1)):
            lo = max(0, t - 299)
            trace = []
            res = find_short_sweep(src.iloc[lo:t + 1].reset_index(drop=True), MIN_SWEEP_DEPTH[SWEEP_TF], trace, sign)
            if res:
                print(f"  {name}: SWEEP FOUND ({res['kind']}) level {sign * res['level']:.2f} depth {res['extreme'] - res['level']:.2f}")
            else:
                for line in trace[:3]:
                    print(f"  {name}: {line}")
    print("\nend of explain")


def market_closed(now_utc):
    """Gold is closed from Friday 17:00 to Sunday 18:00 New York time (22:00 -> 23:00 UTC in winter)."""
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    wd, hr = ny.weekday(), ny.hour
    return wd == 5 or (wd == 4 and hr >= 17) or (wd == 6 and hr < 18)


def drop_closed_hours(df):
    """Remove weekend candles. The feed fills the weekend with flat dust candles that would otherwise
    create fake 4H candles and flip the engulf trend."""
    t = df["time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    wd, hr = t.dt.dayofweek, t.dt.hour
    closed = (wd == 5) | ((wd == 4) & (hr >= 17)) | ((wd == 6) & (hr < 18))
    return df[~closed].reset_index(drop=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        send_telegram("Test alert: your gold bot is connected.")
        print("test message sent")
        return

    if os.environ.get("EXPLAIN_FROM"):
        explain(os.environ["EXPLAIN_FROM"], os.environ.get("EXPLAIN_TO") or os.environ["EXPLAIN_FROM"])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest()
        return

    now = datetime.now(timezone.utc)
    now_naive = now.replace(tzinfo=None)
    if market_closed(now):
        print("market closed")
        return

    df5 = closed_only(drop_closed_hours(fetch("5min", FETCH_5M)), 5, now_naive)
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

    dfs = sweep_candles(df5, SWEEP_TF)
    work = dfs if trend == "down" else mirror(dfs)
    state = load_state()
    n = len(dfs)
    n_check = max(1, CHECK_MINUTES // SWEEP_TF)
    changed = False

    def sweep_at(j):
        return find_short_sweep(work.iloc[max(0, j - 299):j + 1].reset_index(drop=True))

    def first_confirm(j, far, upto):
        """First candle after j (within the window, up to `upto`) closing beyond the sweep candle's far end."""
        for c in range(j + 1, min(j + ENGULF_WINDOW, upto) + 1):
            if work["close"].iloc[c] < far:
                return c
        return None

    hits = []                                     # (sweep candle, result, confirming candle)
    for t in range(max(n - n_check, 0), n):
        res = sweep_at(t)
        if res and (res["kind"] == "engulf sweep" or not REQUIRE_FOLLOW_THROUGH):
            hits.append((t, res, t))
        if REQUIRE_FOLLOW_THROUGH:
            for j in range(max(t - ENGULF_WINDOW, 0), t):
                rj = sweep_at(j)
                if rj and rj["kind"] == "wick sweep" and first_confirm(j, rj["far"], t) == t:
                    hits.append((j, rj, t))

    for j, res, c in hits:
        key = f"{trend}|{SWEEP_TF}|{dfs['time'][j]}"
        if key in state["sent"]:
            continue
        sign = 1 if trend == "down" else -1
        level, extreme = sign * res["level"], sign * res["extreme"]
        local = lambda i: (dfs["time"][i] + timedelta(minutes=SWEEP_TF)).replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ)
        side = "SHORT setup (high swept)" if trend == "down" else "LONG setup (low swept)"
        extra = "" if c == j else f" (engulfed after {c - j} candle{'s' if c - j > 1 else ''})"
        lines = [f"XAUUSD {side}",
                 f"Type: {SWEEP_TF}m {res['kind']}{extra}",
                 f"Swept level: {level:.2f}",
                 f"Sweep extreme (stop reference): {extreme:.2f}",
                 f"Sweep candle closed: {local(j):%d %b %H:%M} {local(j).tzname()}"]
        if c != j:
            lines.append(f"Confirmed: {local(c):%d %b %H:%M} {local(c).tzname()}")
        lines.append(f"Trend: 4H {trends['4H']}, 2H {trends['2H']}. Go check the chart.")
        send_telegram("\n".join(lines))
        state["sent"].append(key)
        changed = True
        print("alert sent:", key)
    if changed:
        save_state(state)


if __name__ == "__main__":
    main()
