# Website Chatbot & Lead-Qualification Agent

An agentic chatbot for an agency website that answers client FAQs via
**retrieval-augmented generation (RAG)** and autonomously asks qualifying
questions (project type, budget, timeline) to assess lead readiness — built
with **LangChain**, **FAISS**, and the **Gemini API**.

## Why this project

Most agency website chatbots either (a) just answer FAQs, or (b) just run a
rigid lead-capture form. This agent does both, and — this is the "agentic"
part — decides, turn by turn, which one the conversation calls for, instead
of following a fixed script.

## How the routing works

Every incoming message is classified by a small Gemini prompt into one of:

- **`faq`** → the `faq_tool` retrieves relevant chunks from a FAISS index
  built over sample knowledge-base documents (`/data`: services, FAQ,
  pricing) and generates a grounded answer.
- **`qualify`** → the `qualify_tool` asks the next unanswered qualifying
  question (project type → budget → timeline) and, once all three are
  known, produces a lead-readiness summary.
- **`other`** → a general fallback response (greetings, small talk).

This tool-based routing is implemented in `agent.py`'s `ChatAgent` class.

## Project structure

```
website-chatbot-lead-agent/
├── app.py                # Streamlit chat UI
├── agent.py               # Agentic routing + both tools (the core logic)
├── knowledge_base.py      # FAISS index + retrieval over /data
├── data/                  # Sample "client website" content
│   ├── services.txt
│   ├── faq.txt
│   └── pricing.txt
├── requirements.txt
├── .env.example            # Template for your API key — copy to .env
└── .gitignore
```

## Setup

1. Clone this repo and move into the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
4. Copy `.env.example` to `.env` and paste in your key:
   ```bash
   cp .env.example .env
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Customizing the knowledge base

Replace the files in `/data` with your own services/FAQ/pricing content —
`knowledge_base.py` automatically loads every `.txt` file in that folder,
so no code changes are needed to swap in a different "client."

## Tech stack

Python · Streamlit · LangChain · FAISS · Gemini API (chat + embeddings) ·
Retrieval-Augmented Generation · Agentic tool routing

## Evaluation & Results

Run the evaluation yourself with `python eval_metrics.py` (needs `GEMINI_API_KEY` in `.env`).

**Setup**
- Knowledge base: 3 documents (services, FAQ, pricing), split into 10 chunks (chunk size 400, overlap 50)
- Embeddings: Gemini `embedding-001` | Vector store: FAISS | Retrieval: top-k similarity search
- Retrieval test set: 15 hand-labelled questions. A chunk counts as relevant if it contains the phrase that answers the question
- Routing test set: 20 hand-labelled messages (6 `faq`, 7 `qualify`, 7 `other`)

**Retrieval metrics**

| Metric | Score |
|---|---|
| MRR | 0.867 |
| Precision@1 | 0.733 |
| Recall@1 | 0.700 |
| Precision@3 | 0.356 |
| Recall@3 | 1.000 |

**Routing and latency**

| Metric | Result |
|---|---|
| Routing accuracy (20 messages) | 95% (19/20) |
| Avg routing latency | 5.50 s |
| Avg retrieval latency | 0.48 s |
| End-to-end FAQ answer latency | 10.43 s |

**Observations**
- Recall@3 of 1.0 means the correct chunk was always in the top 3. Precision@3 is low because most questions have only one relevant chunk.
- The one routing miss: "I'm interested in your branding services, how do we start?" was routed to `faq` instead of `qualify`.
- Routing is evaluated on single messages. Multi-turn behaviour (for example, a short answer like "Around 2 lakhs" right after a budget question) is not covered, because the router receives no conversation history.
- Latency is dominated by the Gemini calls (routing and answer generation), not by FAISS retrieval.
- The `google.generativeai` package is deprecated; migrating to `google.genai` is a planned improvement.

## Notes

This is a portfolio/learning project. The lead-scoring logic here is
intentionally simple (collect 3 fields → summarize); a production version
would weight answers, push captured leads into a CRM, and handle
multi-turn corrections more robustly.