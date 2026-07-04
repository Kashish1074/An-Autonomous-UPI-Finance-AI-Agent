# 💸 PaisaPilot — Your Autonomous UPI Finance Agent

**Track:** Agents for Business
**Event:** AI Agents Intensive — Vibe Coding Capstone (Google x Kaggle)

PaisaPilot is a tool-using AI agent that helps Indian UPI users understand
their spending, catches transactions that match common UPI scam patterns,
and answers natural-language money questions by actually querying the
user's transaction data — not by guessing.

## Why this exists

India processes billions of UPI transactions a month across GPay, PhonePe,
Paytm, and bank apps, but there's no single place users can see spend
patterns or get warned about fraud. UPI-based scams (fake "test transfers",
task/job scams, QR code tricks) are extremely common and most victims only
realize something is wrong after the money is gone. This project came out
of dealing with a real UPI task-scam attempt firsthand.

## What makes it an *agent*, not just a chatbot

PaisaPilot uses Gemini's function calling in a real loop:

1. The user asks a question in plain English/Hindi-English mix
   (e.g. "How much did I spend on food last month?")
2. Gemini decides which tool(s) to call: `get_total_spend`,
   `get_category_breakdown`, `flag_anomalies`, or
   `get_savings_recommendation`
3. The app executes that tool against the actual pandas dataframe of
   transactions
4. The result is returned to Gemini, which grounds its final answer in the
   real numbers — it never fabricates a total

The anomaly scanner itself is rule-based and runs independently of the
chat, catching two patterns:
- Spend far above a user's own category average
- Two payments to the same payee within 10 minutes (the classic
  "small test transfer, then a large one" UPI scam pattern)

## Run it locally

```bash
git clone <this-repo>
cd paisapilot
pip install -r requirements.txt
export GEMINI_API_KEY=AIza...      # free key from aistudio.google.com/apikey, or paste it in the sidebar
streamlit run app.py
```

The app ships with a synthetic 6-month demo dataset
(`sample_upi_transactions.csv`, 165 transactions, with a scam pattern and
an outlier deliberately injected) so judges can try it with zero setup —
just run and open the "Ask PaisaPilot" tab.

To use it with your own data, export any UPI/bank statement to CSV with
columns: `date, payee, category, amount` (negative = spend, positive =
income) and upload it from the sidebar.

## Tech stack

- **Agent brain:** Gemini (Google Gen AI SDK, free-tier function calling)
- **App:** Streamlit
- **Data:** pandas
- **Charts:** Plotly

## Files

- `app.py` — the full agent + dashboard app
- `sample_upi_transactions.csv` — synthetic demo data
- `requirements.txt`

## Limitations & future work

- Anomaly detection is heuristic, not a trained fraud model — a natural
  next step is training on labeled scam reports (e.g. from
  cybercrime.gov.in patterns) for higher precision
- Currently single-user/session; a production version would need secure
  auth and encrypted storage of financial data
- CSV-only ingestion for the hackathon; a real version would parse UPI
  SMS notifications or connect via Account Aggregator (AA) APIs under
  RBI's consent framework
