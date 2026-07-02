# Title
PaisaPilot: An Autonomous UPI Finance Agent for Everyday India

# Subtitle
A tool-using AI agent that turns scattered UPI transactions into clear
answers, savings guidance, and real-time scam alerts — built after a
personal brush with a UPI task scam.

# Track
Agents for Business

---

## 1. The Problem

India runs on UPI. Billions of transactions move every month across GPay,
PhonePe, Paytm, and bank apps — but that money is scattered across five
different apps with five different dashboards, none of which talk to each
other. Two very ordinary problems fall out of this:

1. **Nobody can answer a simple question about their own money.** "How
   much did I actually spend on food last month?" requires manually
   opening every app and adding it up by hand.
2. **UPI fraud is common and fast-moving.** Scammers increasingly use a
   "small test transfer, then a big one" pattern, or "task scams" that ask
   a victim to make a payment before releasing a promised reward. I ran
   into an attempted version of this myself while looking for freelance
   work, and had to figure out cybercrime reporting channels (1930 /
   cybercrime.gov.in) after the fact — there was no tool that would have
   flagged it *at the moment it happened*.

PaisaPilot is a direct response to both problems: one agent that unifies
the view across a user's UPI history, answers questions about it in plain
language, and actively watches for the transaction patterns that precede
common Indian UPI scams.

## 2. What I Built

PaisaPilot is a Streamlit app with three surfaces built on a single shared
transaction dataset:

- **Dashboard** — total spend/income, category breakdown (pie), and
  month-over-month spend trend (bar chart), computed directly from the
  data.
- **Anomaly Scan** — a rule-based scanner that flags (a) any spend more
  than 3x a user's own historical average for that category, and (b) two
  payments to the *same* payee within a 10-minute window, which matches
  the "small test transfer → large transfer" scam pattern common on UPI.
  Flagged items link straight to the official reporting channel
  (cybercrime.gov.in, helpline 1930).
- **Ask PaisaPilot** — a chat interface where the user asks questions in
  natural language and gets grounded, data-backed answers.

The third surface is the actual "agent" part of the project, and it's
worth being specific about why.

### Why this is an agent, not a chatbot with a system prompt

A simple approach would be to dump the whole transaction CSV into a
prompt and ask Claude to "act like a finance assistant." That does not
scale (a real user's history won't fit in context indefinitely), and it
invites hallucinated numbers, which is the worst possible failure mode
for a finance tool.

Instead, PaisaPilot gives Claude four real tools via the Anthropic API's
tool-use (function calling) interface:

- `get_total_spend(start_date, end_date, category)`
- `get_category_breakdown(month)`
- `flag_anomalies()`
- `get_savings_recommendation()`

When the user asks a question, the model doesn't answer directly — it
decides which tool(s) it needs, the app executes that tool against the
live pandas dataframe, the *actual computed result* (a JSON object with
real numbers) is returned to the model, and only then does Claude
compose a final answer, always grounded in what the tool returned. This
is a genuine plan → act → observe → respond loop, run inside a `while`
loop in `run_agent()` that continues calling tools until the model has
enough information to answer in natural language.

For example, asking "Am I overspending on shopping this month, and is
anything suspicious?" causes the model to call *both*
`get_category_breakdown` and `flag_anomalies` in the same turn, combine
the two structured results, and produce one coherent answer — something
a single static prompt cannot reliably do.

### Fraud detection design

The anomaly scanner is intentionally simple and explainable rather than a
black-box model, because trust matters most exactly where money is
involved. It runs two checks:

1. **Category-relative outliers**: for every spend, it compares the
   amount to the user's own historical average for that category. A
   sudden ₹22,000 "Shopping" transaction against a personal average of
   ~₹1,500 gets flagged, with the reasoning shown in plain language.
2. **Rapid same-payee pairs**: it looks for two transfers to the same UPI
   ID within 10 minutes. This directly mirrors a real scam pattern where
   a fraudster asks a victim to "test" a small transfer (often to build
   trust or verify the UPI ID works) before requesting a much larger one
   under time pressure.

Both checks were validated against a synthetic 165-transaction, 6-month
dataset (`sample_upi_transactions.csv`) with a scam pattern and an
outlier deliberately injected — the scanner catches both, with zero false
positives on the surrounding 163 normal transactions.

## 3. Architecture

```
User (Streamlit UI)
   │
   ├── Dashboard / Anomaly tabs ──▶ pure pandas functions (no LLM call,
   │                                deterministic, instant)
   │
   └── Ask PaisaPilot tab
          │
          ▼
     Claude (claude-sonnet-4-6) with 4 registered tools
          │  (tool_use)
          ▼
     Python executes the matching tool against the live dataframe
          │  (tool_result, structured JSON)
          ▼
     Claude composes the final natural-language answer
          │
          ▼
     Streamlit chat UI
```

Keeping the dashboard and anomaly scan **deterministic and LLM-free** was
a deliberate choice: for numbers a user is going to trust and act on, the
agent should reserve the LLM for *interpretation and conversation*, not
for arithmetic. The LLM is only in the loop where natural language
understanding is genuinely needed — deciding what the user is asking for
and explaining findings — never for computing the underlying figures.

## 4. Tech Stack

- **Agent / reasoning layer:** Claude (Anthropic Messages API, tool use)
- **App framework:** Streamlit (single-file app, no backend server needed)
- **Data:** pandas
- **Visualization:** Plotly
- **Data ingestion:** CSV upload (bank/UPI statement export) or bundled
  synthetic demo data

The whole stack was chosen so the project needs zero infrastructure: no
database, no auth server, no hosting beyond a single free Streamlit Cloud
deployment.

## 5. Track Fit — Agents for Business

PaisaPilot targets a very concrete business/financial use case: personal
finance management and fraud prevention for India's ~400M+ UPI users. It
does three things a business-facing finance agent should do — aggregate
fragmented transaction data, surface actionable financial insight
(savings rate, category overspend), and reduce financial risk (fraud
pattern detection) — using an agent architecture (tool use, grounded
responses) rather than a static report generator.

## 6. Results / Demo Walkthrough

Using the bundled demo dataset (165 transactions across 6 months,
including a Freelance Income stream, rent, groceries, and one injected
scam attempt):

- The dashboard correctly totals ~₹1.15L in spend against ~₹95K in
  income/transfers over the period and renders an accurate category split
- The anomaly scanner flags exactly the two injected anomalies (the
  ₹22,000 Shopping outlier and the ₹1 → ₹4,999 rapid-pair scam pattern)
  out of 165 transactions — no false positives
- In chat, asking "How much did I spend on food and groceries combined
  last month?" correctly triggers two tool calls and returns a summed,
  accurate figure with a one-line breakdown
- Asking "Did anything suspicious happen with my money recently?" calls
  `flag_anomalies` and explains the scam pattern in plain language,
  including the 1930/cybercrime.gov.in reporting guidance

## 7. Limitations and Future Work

- The fraud scanner is heuristic, not a trained model — a real fraud
  classifier trained on labeled scam reports would catch a broader range
  of patterns and reduce false negatives over time
- Current version is single-session/single-user with CSV upload; a
  production version would integrate with the RBI's Account Aggregator
  framework for consented, automatic transaction sync instead of manual
  export
- No persistent storage between sessions yet — a natural next step given
  more time
- Category labels currently come from the input data; an LLM-based
  auto-categorizer for raw, uncategorized UPI SMS text would remove the
  need for users to categorize transactions themselves before uploading

## 8. Closing

PaisaPilot started from a real, slightly unnerving personal experience
with a UPI scam attempt, and turned into a small demonstration of what
"vibe coding" an agent actually looks like end-to-end: a real tool-use
loop grounded in real data, kept simple enough that a judge can clone the
repo and have it running in under two minutes.

---
*(Word count: ~1,180 — well under the 2,500-word limit.)*
