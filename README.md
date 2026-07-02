# Stock Alert Scanner

Streamlit app that scans stocks for unusual activity and scores potential setups.

**Two-layer engine**

1. **Triggers** — volume z-score ≥ 3σ, price move > 2.5× ATR, gap opens ≥ 3%, Bollinger breakouts
2. **Confirmation** — RSI, 50/200-day moving-average context, MACD crossovers → score 1–10

Alerts appear in the in-app feed only. No emails.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push to GitHub → share.streamlit.io → New app → pick this repo, `app.py`.

---

*Informational tool only — not investment advice.*
