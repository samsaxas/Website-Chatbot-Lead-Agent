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

## Notes

This is a portfolio/learning project. The lead-scoring logic here is
intentionally simple (collect 3 fields → summarize); a production version
would weight answers, push captured leads into a CRM, and handle
multi-turn corrections more robustly.
