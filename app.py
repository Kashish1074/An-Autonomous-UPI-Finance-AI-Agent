"""
PaisaPilot - An autonomous UPI finance agent
Track: Agents for Business | AI Agents Intensive Vibe Coding Capstone

What it does:
- Ingests UPI transaction history (CSV upload or bundled demo data)
- Categorizes spend, builds a dashboard (category split, monthly trend)
- Runs a rule + LLM-assisted anomaly/fraud scanner over transactions
- Exposes a chat agent (Gemini + function calling) that can answer
  natural-language questions about the user's money by calling real tools
  over the data, not by guessing from a prompt.

Run:
    export GEMINI_API_KEY=AIza...      # free key from aistudio.google.com/apikey
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import datetime as dt

import pandas as pd
import streamlit as st
import plotly.express as px
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"  # free-tier eligible, function-calling capable
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
# Gemini decides *which* tool to call and with *what arguments* based on
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


SYSTEM_PROMPT = """You are PaisaPilot, an autonomous personal finance agent for
Indian UPI users. You have tools to query the user's real transaction data -
always call a tool to get numbers before answering; never guess or invent
figures. Be concise, use ₹ for amounts, and when you flag anomalies, explain
in plain language why a transaction looks suspicious (many users are not
familiar with fraud patterns). Give practical, India-specific advice
(e.g. mention UPI, NPCI's 1930 cybercrime helpline / cybercrime.gov.in when
relevant to fraud). Keep answers under 150 words unless the user asks for
detail."""


def make_gemini_tools(df):
    """Build the tool set Gemini can call, closed over the live dataframe.

    The Gen AI SDK supports "automatic function calling": pass real Python
    functions (with type hints + docstrings) as tools, and the SDK inspects
    their signature to build the schema, decides when to call them, executes
    them, and feeds the result back to the model — the same plan -> act ->
    observe -> respond loop as manual tool use, just handled by the SDK.
    Each wrapper below takes no `df` argument (the model shouldn't supply
    that) and instead closes over the currently loaded dataframe.
    """

    def get_total_spend(start_date: str = "", end_date: str = "", category: str = "") -> dict:
        """Get total spend/income and transaction count, optionally filtered by date range or category.

        Args:
            start_date: Filter start date in YYYY-MM-DD format. Leave empty for no filter.
            end_date: Filter end date in YYYY-MM-DD format. Leave empty for no filter.
            category: One of the known spend categories. Leave empty for no filter.
        """
        return tool_total_spend(df, start_date or None, end_date or None, category or None)

    def get_category_breakdown(month: str = "") -> dict:
        """Get spend broken down by category, optionally for one month.

        Args:
            month: Month in YYYY-MM format. Leave empty to cover all months.
        """
        return tool_category_breakdown(df, month or None)

    def flag_anomalies() -> list:
        """Scan all transactions for suspicious/anomalous activity, including
        patterns that match common UPI scams (e.g. rapid small-then-large
        transfers to the same payee, or spend far above a category average).
        """
        return tool_flag_anomalies(df)

    def get_savings_recommendation() -> dict:
        """Get average monthly income, spend and savings rate to ground a savings recommendation."""
        return tool_savings_recommendation(df)

    return [get_total_spend, get_category_breakdown, flag_anomalies, get_savings_recommendation]


def run_agent(client, df, user_message, chat):
    """Send a message through the Gemini agent.

    `chat` is a persistent `client.chats.create(...)` session (created once
    per Streamlit session in `st.session_state`) so conversational memory
    and tool-call context carry across turns without us re-serializing
    history by hand.
    """
    resp = chat.send_message(user_message)
    return resp.text, chat


# ----------------------------------------------------------------------------
# Theme — "Passbook Ledger"
# Grounded in the Indian bank-passbook / accounts-ledger aesthetic: kraft
# paper, indigo fountain-pen ink for entries, a red ledger margin rule down
# the page, dot-matrix monospace for figures, and rubber-stamp badges for
# flagged (fraud-pattern) entries — the same visual language as a physical
# passbook a bank teller stamps, applied to an autonomous agent instead.
# ----------------------------------------------------------------------------
PAPER = "#EDE3C8"
PAPER_ALT = "#E3D6AE"
PAPER_DARK = "#DCCE9F"
INK = "#1D2A5E"
INK_LIGHT = "#3B4E96"
STAMP_RED = "#A8362A"
STAMP_GREEN = "#2E6B4E"
RULE = "#B9A876"
TEXT = "#241F14"
TEXT_MUTED = "#5B4F36"

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background-color: {PAPER};
    background-image:
        repeating-linear-gradient(#EDE3C8 0px, #EDE3C8 27px, {RULE}55 28px),
        linear-gradient(90deg, transparent 0px, transparent 78px, {STAMP_RED}55 79px, {STAMP_RED}55 80px, transparent 81px);
    background-attachment: local, fixed;
    color: {TEXT};
}}

section[data-testid="stSidebar"] {{
    background-color: {PAPER_DARK};
    border-right: 2px solid {RULE};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

/* Passbook cover header */
.passbook-cover {{
    background: {INK};
    color: {PAPER};
    border-radius: 4px;
    padding: 28px 36px 24px 96px;
    position: relative;
    margin-bottom: 18px;
    box-shadow: 0 4px 0 {RULE}, 0 4px 14px rgba(0,0,0,0.25);
}}
.passbook-cover::before {{
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0; width: 68px;
    background-image: radial-gradient({PAPER} 3px, transparent 4px);
    background-size: 100% 22px;
    background-position: 34px 12px;
    border-right: 2px dashed {PAPER}66;
}}
.passbook-eyebrow {{
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #E8B84B;
    margin-bottom: 6px;
}}
.passbook-title {{
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 42px;
    letter-spacing: 1px;
    margin: 0;
}}
.passbook-sub {{
    font-size: 15px;
    color: {PAPER}CC;
    margin-top: 6px;
    max-width: 640px;
}}

/* Ledger stat cards */
.ledger-row {{ display: flex; gap: 14px; margin-bottom: 6px; }}
.ledger-card {{
    flex: 1;
    background: {PAPER_DARK};
    border: 1px solid {RULE};
    border-left: 4px solid {INK};
    border-radius: 3px;
    padding: 14px 18px;
}}
.ledger-card.credit {{ border-left-color: {STAMP_GREEN}; }}
.ledger-card.debit {{ border-left-color: {STAMP_RED}; }}
.ledger-label {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {TEXT_MUTED};
}}
.ledger-value {{
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: {INK};
    margin-top: 2px;
}}

/* Section labels look like ledger column headers */
.ledger-heading {{
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    border-bottom: 1px solid {RULE};
    padding-bottom: 6px;
    margin: 22px 0 10px 0;
}}

/* Rubber-stamp anomaly badges */
.stamp-card {{
    display: flex;
    align-items: center;
    gap: 16px;
    background: {PAPER_DARK};
    border: 1px solid {RULE};
    border-radius: 3px;
    padding: 12px 16px;
    margin-bottom: 10px;
}}
.stamp-badge {{
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1px;
    color: {STAMP_RED};
    border: 2px solid {STAMP_RED};
    border-radius: 50%;
    width: 74px; height: 74px;
    min-width: 74px;
    display: flex; align-items: center; justify-content: center;
    text-align: center;
    transform: rotate(-8deg);
    line-height: 1.1;
    opacity: 0.85;
}}
.stamp-body b {{ font-family: 'Space Mono', monospace; color: {INK}; }}
.stamp-body span {{ color: {TEXT_MUTED}; font-size: 13px; }}

.clear-card {{
    background: {PAPER_DARK};
    border: 1px solid {STAMP_GREEN};
    border-left: 4px solid {STAMP_GREEN};
    border-radius: 3px;
    padding: 14px 18px;
    color: {TEXT};
}}

.helpline-note {{
    font-family: 'Space Mono', monospace;
    font-size: 12.5px;
    background: {INK};
    color: {PAPER};
    border-radius: 3px;
    padding: 12px 16px;
    margin-top: 8px;
}}

/* Tabs styled like passbook page dividers */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.5px;
    background-color: {PAPER_DARK};
    border-radius: 4px 4px 0 0;
    padding: 8px 18px;
    color: {TEXT_MUTED};
}}
.stTabs [aria-selected="true"] {{
    background-color: {INK} !important;
    color: {PAPER} !important;
}}

/* Buttons / inputs */
.stButton>button, .stDownloadButton>button {{
    font-family: 'Space Mono', monospace;
    background-color: {INK};
    color: {PAPER};
    border: none;
    border-radius: 3px;
}}
div[data-testid="stChatInput"] textarea {{
    font-family: 'Inter', sans-serif;
}}

/* Dataframe container */
div[data-testid="stDataFrame"] {{
    border: 1px solid {RULE};
    border-radius: 3px;
}}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=PAPER_DARK,
        plot_bgcolor=PAPER_DARK,
        font=dict(family="Space Mono, monospace", color=TEXT, size=12),
        colorway=[INK, STAMP_RED, STAMP_GREEN, INK_LIGHT, "#8A6D3B", "#6E4B3A",
                  "#4B6E5E", "#B98B3E", "#7A4B6E", "#3E6E8A"],
        title_font=dict(family="Space Mono, monospace", size=14, color=TEXT),
        legend=dict(font=dict(size=11)),
    )
)

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="passbook-cover">
        <div class="passbook-eyebrow">Account Book · Autonomous Agent Edition</div>
        <p class="passbook-title">💸 PaisaPilot</p>
        <p class="passbook-sub">Your UPI transactions, read and watched by an agent that
        answers questions with real numbers and stamps entries that look like fraud —
        Track: Agents for Business · AI Agents Intensive Vibe Coding Capstone</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='ledger-heading'>Teller Setup</div>", unsafe_allow_html=True)

    # The app owner's key (set via Streamlit Cloud "Secrets", never shown to
    # the browser) powers the demo by default. The sidebar field is only for
    # a visitor who wants to use their own key instead — it is intentionally
    # left blank, never pre-filled with a secret, so the secret is never sent
    # to the client.
    server_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    user_key = st.text_input(
        "Gemini API key (optional — leave blank to use the demo's key)",
        type="password",
        value="",
    )
    api_key = user_key or server_key

    if server_key and not user_key:
        st.caption("✓ Using the demo's built-in key.")
    else:
        st.caption(
            "Free — no credit card. Get your own key at "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
        )
    st.markdown("<div class='ledger-heading'>Data Source</div>", unsafe_allow_html=True)
    data_source = st.radio("Data source", ["Use demo data", "Upload my UPI CSV"], label_visibility="collapsed")
    uploaded = None
    if data_source == "Upload my UPI CSV":
        uploaded = st.file_uploader("CSV with columns: date, payee, category, amount", type="csv")
    st.markdown("<div class='ledger-heading'>Why This Ledger Exists</div>", unsafe_allow_html=True)
    st.markdown(
        "UPI users in India lose money every week to task/OTP/small-test-transfer "
        "scams, and nobody has a simple way to see spend patterns across apps "
        "(GPay, PhonePe, Paytm...). PaisaPilot is a single agent that unifies "
        "the view and actively watches for fraud patterns, not just totals."
    )

df = load_uploaded(uploaded) if uploaded is not None else load_demo_data()

tab1, tab2, tab3 = st.tabs(["📊 Ledger Summary", "🔏 Stamped Entries (Anomalies)", "🤖 Ask The Teller"])

with tab1:
    summary = tool_total_spend(df)
    st.markdown(
        f"""
        <div class="ledger-row">
            <div class="ledger-card debit">
                <div class="ledger-label">Total Debit</div>
                <div class="ledger-value">₹{summary['total_spend']:,.0f}</div>
            </div>
            <div class="ledger-card credit">
                <div class="ledger-label">Total Credit</div>
                <div class="ledger-value">₹{summary['total_income']:,.0f}</div>
            </div>
            <div class="ledger-card">
                <div class="ledger-label">Entries</div>
                <div class="ledger-value">{summary['transaction_count']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    breakdown = tool_category_breakdown(df)
    st.markdown("<div class='ledger-heading'>Spend Split &amp; Monthly Trend</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(names=list(breakdown.keys()), values=list(breakdown.values()))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        fig.update_traces(marker=dict(line=dict(color=PAPER, width=1.5)))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        monthly = df.copy()
        monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
        spend_by_month = monthly[monthly["amount"] < 0].groupby("month")["amount"].sum().abs()
        fig2 = px.bar(x=spend_by_month.index, y=spend_by_month.values,
                       labels={"x": "Month", "y": "Spend (₹)"})
        fig2.update_traces(marker_color=INK)
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='ledger-heading'>Raw Ledger Entries</div>", unsafe_allow_html=True)
    st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)

with tab2:
    st.markdown("<div class='ledger-heading'>Fraud-Pattern &amp; Anomaly Scan</div>", unsafe_allow_html=True)
    flags = tool_flag_anomalies(df)
    if not flags:
        st.markdown(
            "<div class='clear-card'><b>✓ CLEAR</b> — No suspicious patterns found in this data.</div>",
            unsafe_allow_html=True,
        )
    else:
        for f in flags:
            st.markdown(
                f"""
                <div class="stamp-card">
                    <div class="stamp-badge">⚠<br/>FLAGGED</div>
                    <div class="stamp-body">
                        <b>{f['date']} · {f['payee']} · ₹{abs(f['amount']):,.0f}</b><br/>
                        <span>{f['reason']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div class='helpline-note'>⚠ If any of these are real and unauthorized, "
            "report immediately at cybercrime.gov.in or call the national cyber "
            "helpline <b>1930</b>.</div>",
            unsafe_allow_html=True,
        )

with tab3:
    st.markdown("<div class='ledger-heading'>Chat With Your Finance Agent</div>", unsafe_allow_html=True)
    if "display_history" not in st.session_state:
        st.session_state.display_history = []
    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = None
        st.session_state.gemini_chat_signature = None

    for role, text in st.session_state.display_history:
        with st.chat_message(role):
            st.markdown(text)

    q = st.chat_input("e.g. How much did I spend on food last month? Any scam signs this month?")
    if q:
        if not api_key:
            st.error("Add your Gemini API key in the sidebar first (it's free — see the link above).")
        else:
            with st.chat_message("user"):
                st.markdown(q)

            # Recreate the chat session if the API key or loaded data changed,
            # so the tools always close over the current dataframe.
            signature = (api_key, id(df))
            if st.session_state.gemini_chat_signature != signature:
                client = genai.Client(api_key=api_key)
                st.session_state.gemini_chat = client.chats.create(
                    model=GEMINI_MODEL,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=make_gemini_tools(df),
                    ),
                )
                st.session_state.gemini_chat_signature = signature

            with st.spinner("PaisaPilot is checking your transactions..."):
                try:
                    answer, _ = run_agent(None, df, q, st.session_state.gemini_chat)
                except Exception as e:
                    answer = f"Something went wrong talking to Gemini: {e}"

            st.session_state.display_history.append(("user", q))
            st.session_state.display_history.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)