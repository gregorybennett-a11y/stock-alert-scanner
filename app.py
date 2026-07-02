"""
Stock Alert Scanner — v1
Two-layer engine:
  Layer 1 (trigger):  volume z-score, ATR-relative price shock, gap open, Bollinger breakout
  Layer 2 (confirm):  RSI, 50/200-day MA context, MACD cross
Alerts appear in the in-app feed only. Not investment advice.
"""

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Alert Scanner", page_icon="📡", layout="wide")

# ----------------------------- universe ------------------------------------


@st.cache_data(ttl=24 * 3600)
def get_sp500_tickers() -> list[str]:
    """S&P 500 universe scraped from Wikipedia. Falls back to a core list."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        ticks = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(ticks) > 400:
            return ticks
    except Exception:
        pass
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "PG", "MA", "HD", "AVGO",
        "CVX", "MRK", "ABBV", "PEP", "COST", "KO", "ADBE", "WMT", "CRM",
        "BAC", "NFLX", "AMD", "TMO", "DIS", "ORCL", "CSCO", "INTC", "QCOM",
    ]


# ----------------------------- indicators ----------------------------------


def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def atr(df: pd.DataFrame, period: int = 14) -> float:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def macd_cross(close: pd.Series) -> str:
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    diff = macd - signal
    if len(diff) < 2 or diff.isna().iloc[-1] or diff.isna().iloc[-2]:
        return "none"
    if diff.iloc[-2] <= 0 < diff.iloc[-1]:
        return "bullish"
    if diff.iloc[-2] >= 0 > diff.iloc[-1]:
        return "bearish"
    return "none"


# ----------------------------- engine --------------------------------------


def analyze(ticker: str, df: pd.DataFrame, cfg: dict) -> dict | None:
    """Layer 1 triggers -> Layer 2 confirmation -> scored alert (or None)."""
    df = df.dropna()
    if len(df) < 60:
        return None

    close, vol = df["Close"], df["Volume"]
    last = float(close.iloc[-1])
    ret_pct = float(close.pct_change().iloc[-1] * 100)

    # ---- Layer 1: triggers ----
    triggers = []

    v_mean = float(vol.iloc[-21:-1].mean())
    v_std = float(vol.iloc[-21:-1].std())
    vol_z = (float(vol.iloc[-1]) - v_mean) / v_std if v_std > 0 else 0.0
    if vol_z >= cfg["vol_z"]:
        triggers.append(f"volume {vol_z:.1f}σ above normal")

    a = atr(df)
    atr_pct = a / last * 100 if last else 0.0
    day_move = abs(float(close.iloc[-1] - close.iloc[-2]))
    if a > 0 and day_move > cfg["atr_mult"] * a:
        triggers.append(f"{ret_pct:+.1f}% move vs typical {atr_pct:.1f}% (ATR)")

    gap_pct = float((df["Open"].iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
    if abs(gap_pct) >= cfg["gap_pct"]:
        triggers.append(f"gapped {gap_pct:+.1f}% at open")

    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    upper = float(ma20.iloc[-1] + 2 * sd20.iloc[-1])
    lower = float(ma20.iloc[-1] - 2 * sd20.iloc[-1])
    if last > upper:
        triggers.append("closed above upper Bollinger band")
    elif last < lower:
        triggers.append("closed below lower Bollinger band")

    if not triggers:
        return None

    # ---- Layer 2: confirmation & scoring ----
    score = 2 * len(triggers)
    notes = []
    direction = "bullish" if ret_pct >= 0 else "bearish"

    r = rsi(close)
    if not np.isnan(r):
        if r <= 30:
            score += 2
            notes.append(f"RSI {r:.0f} — oversold, bounce candidate")
        elif r >= 70:
            score += 1
            notes.append(f"RSI {r:.0f} — overbought, may be extended")
        else:
            notes.append(f"RSI {r:.0f} — room to run")
            score += 1

    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    if not np.isnan(ma200):
        trend = "above" if last > ma200 else "below"
        notes.append(f"price {trend} 200-day MA")
        if (direction == "bullish" and trend == "above") or (
            direction == "bearish" and trend == "below"
        ):
            score += 2  # move agrees with long-term trend
    if last > ma50:
        notes.append("above 50-day MA")

    mc = macd_cross(close)
    if mc != "none":
        notes.append(f"MACD {mc} cross")
        if mc == direction:
            score += 2

    return {
        "Ticker": ticker,
        "Price": round(last, 2),
        "Day %": round(ret_pct, 2),
        "Direction": direction,
        "Score": min(score, 10),
        "Triggers": "; ".join(triggers),
        "Confirmation": "; ".join(notes),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def run_scan(tickers: tuple[str, ...], cfg_key: tuple) -> pd.DataFrame:
    cfg = dict(zip(("vol_z", "atr_mult", "gap_pct"), cfg_key))
    data = yf.download(
        list(tickers),
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    rows = []
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            result = analyze(t, df, cfg)
            if result:
                rows.append(result)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )


# ----------------------------- UI ------------------------------------------

st.title("📡 Stock Alert Scanner")
st.caption(
    "Detects unusual moves (Layer 1), confirms with technical indicators (Layer 2), "
    "and scores them 1–10. Informational only — **not investment advice**."
)

with st.sidebar:
    st.header("Settings")
    universe = st.radio(
        "Universe", ["S&P 500", "My watchlist"], help="Watchlist = your own tickers"
    )
    watchlist_txt = st.text_area(
        "Watchlist tickers (comma-separated)",
        "AAPL, NVDA, TSLA, AMD, PLTR",
        disabled=(universe == "S&P 500"),
    )
    min_score = st.slider("Minimum alert score", 1, 10, 5)
    direction_filter = st.selectbox("Direction", ["Both", "Bullish only", "Bearish only"])

    st.subheader("Trigger sensitivity")
    vol_z = st.slider("Volume z-score ≥", 1.0, 5.0, 3.0, 0.5)
    atr_mult = st.slider("Move vs ATR ≥ ×", 1.0, 5.0, 2.5, 0.5)
    gap_pct = st.slider("Gap open ≥ %", 1.0, 10.0, 3.0, 0.5)

    st.divider()
    st.caption("🔕 Notifications: in-app feed only (email/push opt-in comes later).")

if universe == "S&P 500":
    tickers = tuple(get_sp500_tickers())
else:
    tickers = tuple(
        t.strip().upper().replace(".", "-")
        for t in watchlist_txt.split(",")
        if t.strip()
    )

col1, col2 = st.columns([1, 3])
with col1:
    scan_clicked = st.button("🔍 Run scan", type="primary", use_container_width=True)
with col2:
    st.caption(
        f"{len(tickers)} tickers · data: Yahoo Finance (daily) · "
        f"results cached 30 min · {dt.date.today():%b %d, %Y}"
    )

if scan_clicked or "last_scan" in st.session_state:
    if scan_clicked:
        with st.spinner(f"Scanning {len(tickers)} tickers…"):
            st.session_state["last_scan"] = run_scan(
                tickers, (vol_z, atr_mult, gap_pct)
            )
    results = st.session_state["last_scan"]

    if results.empty:
        st.info("No unusual activity found with current settings. Try lowering the trigger sensitivity.")
    else:
        filtered = results[results["Score"] >= min_score]
        if direction_filter != "Both":
            want = "bullish" if direction_filter == "Bullish only" else "bearish"
            filtered = filtered[filtered["Direction"] == want]

        st.subheader(f"🚨 Alert feed — {len(filtered)} alert(s)")
        for _, row in filtered.iterrows():
            emoji = "🟢" if row["Direction"] == "bullish" else "🔴"
            with st.expander(
                f"{emoji} **{row['Ticker']}** · ${row['Price']} · {row['Day %']:+.1f}% · score {row['Score']}/10",
                expanded=(row["Score"] >= 8),
            ):
                st.markdown(f"**Triggers:** {row['Triggers']}")
                st.markdown(f"**Confirmation:** {row['Confirmation']}")

        with st.expander("Full results table"):
            st.dataframe(filtered, use_container_width=True, hide_index=True)
else:
    st.info("Press **Run scan** to check the market.")
