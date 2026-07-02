# PaisaPilot — 5-Minute Video Script

Target length: 4:30–5:00. Record screen + voiceover, upload unlisted/public
to YouTube, attach the link in the Kaggle Writeup Media Gallery.

**0:00–0:40 — Hook + Problem**
- "A while back I nearly got caught in a UPI task scam — someone asked me
  to make a small 'test' payment before a bigger 'reward.' It made me
  realize there's no tool that watches for this in the moment."
- "UPI users in India have money spread across GPay, PhonePe, Paytm — no
  single place to see spend, and no fraud alerts. So I built PaisaPilot."

**0:40–1:10 — What it is, one line**
- "PaisaPilot is an AI agent — not a chatbot, an agent that actually calls
  tools over your real transaction data — that gives you a spend
  dashboard, a fraud scanner, and a chat you can ask money questions to."

**1:10–2:30 — Live demo: Dashboard + Anomaly Scan**
- Open the app (screen share), show the Dashboard tab: total spend/income,
  category pie chart, monthly trend
- Switch to Anomaly Scan tab: show the two flagged transactions
  (₹22,000 outlier + the rapid ₹1 → ₹4,999 pair) and explain out loud
  why each looks suspicious, mention the 1930/cybercrime.gov.in guidance
  shown in the app

**2:30–3:45 — Live demo: Ask PaisaPilot (the agent loop)**
- Type: "How much did I spend on food and groceries last month?"
  → show the answer coming back with real numbers
- Type: "Did anything suspicious happen with my money recently?"
  → show it correctly explaining the scam pattern
- Briefly explain on screen (or with a simple diagram) what's happening
  under the hood: "Claude doesn't guess — it calls a real Python function
  against my transaction data, gets the actual numbers back, then
  answers. That tool-use loop is what makes this an agent."

**3:45–4:20 — Architecture + tech stack (quick)**
- Show the architecture diagram from the README/writeup
- "Claude for reasoning and tool use, Streamlit for the UI, pandas for
  the data — no backend server, so anyone can clone this and run it in
  under two minutes."

**4:20–4:50 — Track fit + close**
- "This is my submission for the Agents for Business track — it's a real
  agent solving a real financial problem I ran into myself."
- Show GitHub repo URL / live demo URL on screen
- "Thanks for watching — code and setup instructions are linked below."

**4:50–5:00 — End card**
- GitHub repo link + (if deployed) live Streamlit URL on screen for 3+
  seconds so viewers can screenshot it
