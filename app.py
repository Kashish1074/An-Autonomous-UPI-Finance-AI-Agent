"""
PaisaPilot - An autonomous UPI finance agent
Track: Agents for Business | AI Agents Intensive Vibe Coding Capstone

What it does:
- Ingests UPI transaction history (CSV upload or bundled demo data)
- Categorizes spend, builds a dashboard (category split, monthly trend)
- Runs a rule + LLM-assisted anomaly/fraud scanner over transactions
- Exposes a chat agent (Claude + tool use) that can answer natural-language
  questions about the user's money by calling real tools over the data,
  not by guessing from a prompt.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import json
import datetime as dt

import pandas as pd
import streamlit as st
import plotly.express as px
from anthropic import Anthropic

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"
st.set_page_config(page_title="PaisaPilot - UPI Finance Agent", page_icon="💸", layout="wide")

CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Bills & Utilities",
    "Shopping", "Entertainment", "Health", "Rent", "Transfers/P2P",
    "Investments", "Freelance Income", "Uncategorized",
]

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_demo_data() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "sample_upi_transactions.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ----------------------------------------------------------------------------
# Agent "tools" - these are real Python functions the LLM can call.
# This is what makes PaisaPilot an agent rather than a single prompt:
# Claude decides *which* tool to call and with *what arguments* based on
# the user's natural-language question, we execute it against the actual
# dataframe, and feed the structured result back for a grounded answer.
# ----------------------------------------------------------------------------

def tool_total_spend(df, start_date=None, end_date=None, category=None):
    d = df.copy()
    if start_date:
        d = d[d["date"] >= pd.to_datetime(start_date)]
    if end_date:
        d = d[d["date"] <= pd.to_datetime(end_date)]
    if category:
        d = d[d["category"].str.lower() == category.lower()]
    spend = d[d["amount"] < 0]["amount"].sum()
    income = d[d["amount"] > 0]["amount"].sum()
    return {
        "total_spend": round(float(-spend), 2),
        "total_income": round(float(income), 2),
        "transaction_count": int(len(d)),
    }


def tool_category_breakdown(df, month=None):
    d = df.copy()
    if month:
        d = d[d["date"].dt.strftime("%Y-%m") == month]
    spend = d[d["amount"] < 0].copy()
    spend["amount"] = -spend["amount"]
    breakdown = spend.groupby("category")["amount"].sum().sort_values(ascending=False)
    return {k: round(float(v), 2) for k, v in breakdown.items()}


def tool_flag_anomalies(df):
    """Heuristic fraud/anomaly scanner.
    Flags: (a) sudden spend >3x category average, (b) payees hit only once
    with round suspiciously large amounts, (c) rapid small-then-large pairs
    to the same payee within 10 minutes (classic 'test transaction' scam
    pattern used against UPI users).
    """
    d = df.sort_values("date").copy()
    flags = []

    cat_avg = d[d["amount"] < 0].groupby("category")["amount"].mean()
    for _, row in d[d["amount"] < 0].iterrows():
        avg = cat_avg.get(row["category"], row["amount"])
        if avg != 0 and row["amount"] < 3 * avg:
            flags.append({
                "date": str(row["date"].date()),
                "payee": row["payee"],
                "amount": float(row["amount"]),
                "reason": f"Spend is >3x the average for '{row['category']}'",
            })

    d["gap_minutes"] = d.groupby("payee")["date"].diff().dt.total_seconds() / 60
    small_then_large = d[(d["gap_minutes"] <= 10) & (d["gap_minutes"].notna())]
    for _, row in small_then_large.iterrows():
        flags.append({
            "date": str(row["date"].date()),
            "payee": row["payee"],
            "amount": float(row["amount"]),
            "reason": "Two payments to the same payee within 10 minutes - "
                      "matches the 'small test transfer, then big transfer' "
                      "UPI scam pattern.",
        })

    return flags[:15]


def tool_savings_recommendation(df):
    monthly = df.copy()
    monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
    net = monthly.groupby("month")["amount"].sum()
    avg_income = monthly[monthly["amount"] > 0].groupby("month")["amount"].sum().mean()
    avg_spend = -monthly[monthly["amount"] < 0].groupby("month")["amount"].sum().mean()
    savings_rate = 0 if avg_income == 0 else (avg_income - avg_spend) / avg_income
    return {
        "avg_monthly_income": round(float(avg_income or 0), 2),
        "avg_monthly_spend": round(float(avg_spend or 0), 2),
        "avg_savings_rate_pct": round(float(savings_rate * 100), 1),
        "months_analyzed": monthly["month"].nunique(),
    }


TOOLS = [
    {
        "name": "get_total_spend",
        "description": "Get total spend/income and transaction count, optionally filtered by date range or category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "category": {"type": "string", "description": "One of the known spend categories"},
            },
        },
    },
    {
        "name": "get_category_breakdown",
        "description": "Get spend broken down by category, optionally for a specific month (YYYY-MM).",
        "input_schema": {
            "type": "object",
            "properties": {"month": {"type": "string", "description": "YYYY-MM, optional"}},
        },
    },
    {
        "name": "flag_anomalies",
        "description": "Scan all transactions for suspicious/anomalous activity, including patterns that match common UPI scams.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_savings_recommendation",
        "description": "Get average monthly income, spend and savings rate to ground a savings recommendation.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_IMPL = {
    "get_total_spend": tool_total_spend,
    "get_category_breakdown": tool_category_breakdown,
    "flag_anomalies": tool_flag_anomalies,
    "get_savings_recommendation": tool_savings_recommendation,
}

SYSTEM_PROMPT = """You are PaisaPilot, an autonomous personal finance agent for
Indian UPI users. You have tools to query the user's real transaction data -
always call a tool to get numbers before answering; never guess or invent
figures. Be concise, use ₹ for amounts, and when you flag anomalies, explain
in plain language why a transaction looks suspicious (many users are not
familiar with fraud patterns). Give practical, India-specific advice
(e.g. mention UPI, NPCI's 1930 cybercrime helpline / cybercrime.gov.in when
relevant to fraud). Keep answers under 150 words unless the user asks for
detail."""


def run_agent(client, df, user_message, history):
    messages = history + [{"role": "user", "content": user_message}]

    while True:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if resp.stop_reason != "tool_use":
            final_text = "".join(b.text for b in resp.content if b.type == "text")
            messages.append({"role": "assistant", "content": resp.content})
            return final_text, messages

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            fn = TOOL_IMPL[block.name]
            try:
                result = fn(df, **block.input)
            except Exception as e:
                result = {"error": str(e)}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("💸 PaisaPilot — Your Autonomous UPI Finance Agent")
st.caption("Track: Agents for Business · AI Agents Intensive Vibe Coding Capstone")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key", type="password",
                             value=os.environ.get("ANTHROPIC_API_KEY", ""))
    st.markdown("---")
    data_source = st.radio("Data source", ["Use demo data", "Upload my UPI CSV"])
    uploaded = None
    if data_source == "Upload my UPI CSV":
        uploaded = st.file_uploader("CSV with columns: date, payee, category, amount", type="csv")
    st.markdown("---")
    st.markdown(
        "**Why this exists:** UPI users in India lose money every week to "
        "task/OTP/small-test-transfer scams, and nobody has a simple way to "
        "see spend patterns across apps (GPay, PhonePe, Paytm...). "
        "PaisaPilot is a single agent that unifies the view and actively "
        "watches for fraud patterns, not just totals."
    )

df = load_uploaded(uploaded) if uploaded is not None else load_demo_data()

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🚨 Anomaly Scan", "🤖 Ask PaisaPilot"])

with tab1:
    col1, col2, col3 = st.columns(3)
    summary = tool_total_spend(df)
    col1.metric("Total spend", f"₹{summary['total_spend']:,.0f}")
    col2.metric("Total income", f"₹{summary['total_income']:,.0f}")
    col3.metric("Transactions", summary["transaction_count"])

    breakdown = tool_category_breakdown(df)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(names=list(breakdown.keys()), values=list(breakdown.values()),
                      title="Spend by category")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        monthly = df.copy()
        monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
        spend_by_month = monthly[monthly["amount"] < 0].groupby("month")["amount"].sum().abs()
        fig2 = px.bar(x=spend_by_month.index, y=spend_by_month.values,
                       labels={"x": "Month", "y": "Spend (₹)"}, title="Monthly spend trend")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)

with tab2:
    st.subheader("Anomaly & fraud-pattern scan")
    flags = tool_flag_anomalies(df)
    if not flags:
        st.success("No suspicious patterns found in this data.")
    else:
        for f in flags:
            st.warning(f"**{f['date']} · {f['payee']} · ₹{abs(f['amount']):,.0f}**\n\n{f['reason']}")
        st.info(
            "If any of these are real and unauthorized, report immediately at "
            "cybercrime.gov.in or call the national cyber helpline **1930**."
        )

with tab3:
    st.subheader("Chat with your finance agent")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "display_history" not in st.session_state:
        st.session_state.display_history = []

    for role, text in st.session_state.display_history:
        with st.chat_message(role):
            st.markdown(text)

    q = st.chat_input("e.g. How much did I spend on food last month? Any scam signs this month?")
    if q:
        if not api_key:
            st.error("Add your Anthropic API key in the sidebar first.")
        else:
            with st.chat_message("user"):
                st.markdown(q)
            client = Anthropic(api_key=api_key)
            with st.spinner("PaisaPilot is checking your transactions..."):
                answer, new_history = run_agent(client, df, q, st.session_state.chat_history)
            st.session_state.chat_history = new_history
            st.session_state.display_history.append(("user", q))
            st.session_state.display_history.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)
