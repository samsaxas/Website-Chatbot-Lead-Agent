"""
agent.py
--------
The agentic core of the chatbot.

WHAT "AGENTIC" MEANS HERE:
Instead of always doing the same thing (e.g. always retrieving from the
knowledge base), the agent looks at each incoming message and decides,
turn by turn, which of two TOOLS to use:

    1. `faq_tool`      -> answers factual questions using FAISS retrieval
                          grounded generation (RAG) over the knowledge base
                          in /data (services, FAQ, pricing).

    2. `qualify_tool`  -> advances a lead-qualification flow, asking about
                          budget, timeline, and project type one at a time,
                          and produces a final lead-readiness summary once
                          all three are known.

This mirrors real agency chatbot behaviour: answer whatever the visitor
asks, but also work toward qualifying them as a sales lead in the
background -- without forcing a rigid, scripted question order.

The ROUTING decision itself is made by a small classification prompt sent
to Gemini (see `_route_message`), which is the "tool selection" step of
the agent loop.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

from knowledge_base import build_vectorstore, retrieve

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Create a .env file (see .env.example) "
        "and set GEMINI_API_KEY=your_key_here."
    )

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.5-flash-lite")


def _generate(prompt: str) -> str:
    """Helper to generate content with fallback/retry on rate limits."""
    import time
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
            raise e
    return ""

# The three pieces of information the "qualify" tool tries to collect,
# in the order it asks for them.
QUALIFICATION_FIELDS = ["project_type", "budget", "timeline"]

QUALIFICATION_QUESTIONS = {
    "project_type": "What kind of project are you looking for help with — branding, a website, digital marketing, or something else?",
    "budget": "Do you have a rough budget range in mind for this project?",
    "timeline": "What's your ideal timeline — are you hoping to start immediately, or planning for later?",
}


class ChatAgent:
    """
    Holds conversation state for a single visitor session and exposes
    `handle_message()` as the main agent entry point.
    """

    def __init__(self):
        # Loading the FAISS index once per session keeps the demo simple.
        # In production you'd build this once at server startup and share
        # it across sessions instead of rebuilding it per user.
        self.vectorstore = build_vectorstore()
        self.qualification_answers = {field: None for field in QUALIFICATION_FIELDS}
        self.current_field = None
        self.history = []  # list of (role, message) tuples, for context

    # -----------------------------------------------------------------
    # ROUTING: the "agentic" decision step
    # -----------------------------------------------------------------
    def _route_message(self, user_message: str) -> str:
        """
        Asks Gemini to classify the visitor's message into one of:
          - "faq"     : a factual question about services/pricing/process
          - "qualify" : the visitor is engaging generally / showing interest,
                        so it's a good moment to ask a qualifying question
          - "other"   : greetings, small talk, anything else

        This single classification call is the "custom tool routing" logic:
        it decides which tool (`_faq_tool` or `_qualify_tool`) handles the
        turn.
        """
        routing_prompt = f"""Classify the visitor's message into exactly one
label: faq, qualify, or other.

- "faq": the message asks a factual question about services, pricing,
  process, or timelines.
- "qualify": the message expresses general interest in working together,
  or answers a qualifying question that was just asked.
- "other": greetings, thanks, or anything unrelated.

Message: "{user_message}"

Respond with only one word: faq, qualify, or other."""

        response_text = _generate(routing_prompt)
        label = response_text.strip().lower()
        if label not in {"faq", "qualify", "other"}:
            label = "other"
        return label

    # -----------------------------------------------------------------
    # TOOL 1: FAQ retrieval (RAG)
    # -----------------------------------------------------------------
    def _faq_tool(self, user_message: str) -> str:
        context = retrieve(self.vectorstore, user_message, k=3)

        answer_prompt = f"""You are a helpful assistant for a branding and
digital agency's website. Answer the visitor's question using ONLY the
context below. If the context doesn't contain the answer, say you'll have
a team member follow up with details.

Context:
{context}

Visitor question: {user_message}

Answer in 2-3 concise sentences."""

        return _generate(answer_prompt)

    # -----------------------------------------------------------------
    # TOOL 2: Lead qualification
    # -----------------------------------------------------------------
    def _qualify_tool(self, user_message: str) -> str:
        # If we were mid-question, capture the answer against whichever field we asked about.
        if self.current_field and self.qualification_answers[self.current_field] is None:
            self.qualification_answers[self.current_field] = user_message

        next_field = self._next_unanswered_field()

        if next_field:
            self.current_field = next_field
            question = QUALIFICATION_QUESTIONS[next_field]
            self.history.append(("agent_question", next_field))
            return question

        # All three fields are collected -> produce a lead summary/score.
        self.current_field = None
        return self._score_lead()

    def _next_unanswered_field(self):
        for field in QUALIFICATION_FIELDS:
            if self.qualification_answers[field] is None:
                return field
        return None

    def _score_lead(self) -> str:
        """
        Very simple, transparent scoring: a lead is "qualified" once we
        have a project type, budget, and timeline. A real implementation
        might weight these (e.g. "immediate" timeline scores higher) or
        push the captured fields into a CRM.
        """
        answers = self.qualification_answers
        summary = (
            f"Thanks! Here's what I've got:\n"
            f"- Project type: {answers['project_type']}\n"
            f"- Budget: {answers['budget']}\n"
            f"- Timeline: {answers['timeline']}\n\n"
            f"You're all set — I'll pass this along to our team, and "
            f"someone will follow up with next steps."
        )
        return summary

    # -----------------------------------------------------------------
    # MAIN ENTRY POINT
    # -----------------------------------------------------------------
    def handle_message(self, user_message: str) -> str:
        self.history.append(("user", user_message))

        label = self._route_message(user_message)

        if label == "faq":
            reply = self._faq_tool(user_message)
        elif label == "qualify":
            reply = self._qualify_tool(user_message)
        else:
            reply = (
                "Hi! I can answer questions about our services, pricing, "
                "or process — or if you're interested in working together, "
                "just let me know and I can get a few quick details."
            )

        self.history.append(("agent", reply))
        return reply
