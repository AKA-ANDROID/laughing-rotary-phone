"""
NSE F&O end-of-day scanner.

Criteria:
  1. Last 4H candle of the day is a BULLISH engulfing of the previous 4H candle
     (current body fully covers the previous body).
  2. Daily trend is either:
       - UPTREND: last two swing highs rising AND last two swing lows rising
       - BREAKOUT: was making lower highs, latest close is above the most
         recent swing high (downtrend structure broken)
  3. NSE stocks only (Yahoo tickers end in .NS).

Sends the result to Telegram. Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
from environment variables.
"""
import os
import sys
import datetime as dt
import pandas as pd
import requests
import yfinance as yf

SWING_N = 2  # swing = highest/lowest of N candles on each side


# ---------- universe ----------
def load_symbols():
    path = os.path.join(os.path.dirname(__file__), "symbols.txt")
    with open(path) as f:
        syms = [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]
    return sorted(set(syms))


def to_yahoo(sym):
    return sym + ".NS"


# ---------- 4H candles ----------
def build_4h(df1h):
    """NSE 4H candles like TradingView: 9:15-13:15 and 13:15-15:30."""
    df = df1h.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    df.index = idx
    df["day"] = df.index.date
    df["half"] = (df.index.hour >= 13).astype(int)
    g = df.groupby(["day", "half"])
    out = g.agg(Open=("Open", "first"), High=("High", "max"),
                Low=("Low", "min"), Close=("Close", "last"))
    return out


def bullish_engulf(prev, cur):
    prev_bear = prev.Close < prev.Open
    cur_bull = cur.Close > cur.Open
    return bool(prev_bear and cur_bull
                and cur.Open <= prev.Close and cur.Close >= prev.Open)


# ---------- daily trend ----------
def swings(df, n=SWING_N):
    highs, lows = [], []
    H, L = df["High"].values, df["Low"].values
    for i in range(n, len(df) - n):
        if H[i] == max(H[i - n:i + n + 1]):
            highs.append((i, H[i]))
        if L[i] == min(L[i - n:i + n + 1]):
            lows.append((i, L[i]))
    return highs, lows


def classify_trend(daily):
    daily = daily.dropna(subset=["High", "Low", "Close"])
    if len(daily) < 30:
        return None, None
    highs, lows = swings(daily)
    if len(highs) < 2 or len(lows) < 2:
        return None, None
    h1, h2 = highs[-2][1], highs[-1][1]
    l1, l2 = lows[-2][1], lows[-1][1]
    close = float(daily["Close"].iloc[-1])
    if h2 > h1 and l2 > l1:
        return "UPTREND", f"HH/HL, last swing high {h2:.2f}"
    if h2 < h1 and close > h2:
        return "BREAKOUT", f"closed above lower-high {h2:.2f}"
    return None, None


# ---------- main ----------
def pick(data, ticker, multi):
    if multi:
        if ticker not in data.columns.get_level_values(0):
            return None
        return data[ticker]
    return data


def scan():
    symbols = load_symbols()
    tickers = [to_yahoo(s) for s in symbols]

    h1 = yf.download(tickers, period="30d", interval="1h", group_by="ticker",
                     auto_adjust=False, progress=False, threads=True)
    d1 = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                     auto_adjust=False, progress=False, threads=True)
    multi = len(tickers) > 1

    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date()
    hits, no_data = [], []

    for sym, tk in zip(symbols, tickers):
        try:
            a = pick(h1, tk, multi)
            b = pick(d1, tk, multi)
            if a is None or b is None:
                no_data.append(sym)
                continue
            c4 = build_4h(a)
            if len(c4) < 2:
                no_data.append(sym)
                continue
            last_day = c4.index[-1][0]
            if last_day != today:
                continue  # holiday / stale data
            cur, prev = c4.iloc[-1], c4.iloc[-2]
            if not bullish_engulf(prev, cur):
                continue
            trend, note = classify_trend(b)
            if trend:
                hits.append((sym, trend, float(cur.Close), note))
        except Exception as e:  # one bad ticker must not kill the scan
            print(f"{sym}: {e}", file=sys.stderr)
            no_data.append(sym)
    return hits, no_data, today


def format_message(hits, no_data, today):
    if not hits:
        msg = f"Scan {today}: no stocks passed."
    else:
        up = [h for h in hits if h[1] == "UPTREND"]
        bo = [h for h in hits if h[1] == "BREAKOUT"]
        lines = [f"NSE scan {today} | 4H bullish engulf + trend"]
        if up:
            lines.append("\nUPTREND")
            lines += [f"{s}  {c:.2f}  ({n})" for s, _, c, n in up]
        if bo:
            lines.append("\nDOWNTREND BROKEN")
            lines += [f"{s}  {c:.2f}  ({n})" for s, _, c, n in bo]
        msg = "\n".join(lines)
    if no_data:
        msg += f"\n\nNo data for {len(no_data)} symbols: " + ", ".join(no_data[:15])
    return msg


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(text)
        print("\n(Telegram secrets missing, printed instead)")
        return
    for i in range(0, len(text), 4000):
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": text[i:i + 4000]}, timeout=30)
        r.raise_for_status()


if __name__ == "__main__":
    hits, no_data, today = scan()
    send_telegram(format_message(hits, no_data, today))
