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
<<<<<<< HEAD

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
| MRR | _TBD_ |
| Precision@1 | _TBD_ |
| Recall@1 | _TBD_ |
| Precision@3 | _TBD_ |
| Recall@3 | _TBD_ |

**Routing and latency**

| Metric | Result |
|---|---|
| Routing accuracy (20 messages) | _TBD_ |
| Avg routing latency | _TBD_ |
| Avg retrieval latency | _TBD_ |
| End-to-end FAQ answer latency | _TBD_ |

**Known limitation**

The router classifies each message without conversation history, so short answers to qualifying questions (for example "Around 2 lakhs") can be misrouted. See the MISS lines in the evaluation output.

## Notes

This is a portfolio/learning project. The lead-scoring logic here is
intentionally simple (collect 3 fields → summarize); a production version
would weight answers, push captured leads into a CRM, and handle
multi-turn corrections more robustly.
=======
>>>>>>> 53940ce2add5064a07e44be5653f60d1a29c97c3
