"""
eval_metrics.py  --  put this in the project root (next to agent.py) and run:
    python eval_metrics.py

Needs GEMINI_API_KEY in .env. Prints retrieval metrics (MRR, Precision@k,
Recall@k), routing accuracy, and average latency. Relevance is defined by a
phrase that must appear in a chunk, so the labels don't break if you change
chunk size.
"""
import time
from knowledge_base import build_vectorstore

K_VALUES = [1, 3]

# (question, phrase that a relevant chunk must contain)
RETRIEVAL_SET = [
    ("How long does a typical website project take?", "3-6 weeks"),
    ("Do you work with startups?", "early-stage"),
    ("What is your typical process?", "discovery call"),
    ("Do you offer support after launch?", "monthly retainer packages"),
    ("Can I get branding and web design in one project?", "brand identity work"),
    ("How are branding packages priced?", "Branding packages start"),
    ("How are website projects priced?", "quoted per project"),
    ("Is ad spend included in the marketing fee?", "ad spend itself is separate"),
    ("How long does a branding project take?", "2-4 weeks"),
    ("Do you make explainer videos?", "explainer videos"),
    ("Do you do product photography?", "product photography"),
    ("What does your digital marketing include?", "performance marketing"),
    ("How many revision rounds do I get?", "1-2 rounds of revisions"),
    ("What does a brand identity include?", "color palettes"),
    ("How do I get an exact quote?", "member of our team will follow up"),
]

# (message, expected route)
ROUTING_SET = [
    ("How long does a website project take?", "faq"),
    ("What do you charge for branding?", "faq"),
    ("Do you offer retainers?", "faq"),
    ("What services do you provide?", "faq"),
    ("Do you do animation work?", "faq"),
    ("What is your process?", "faq"),
    ("I'd like to work with you on a new project", "qualify"),
    ("We want to build a website for our company", "qualify"),
    ("I'm interested in your branding services, how do we start?", "qualify"),
    ("Can we get started?", "qualify"),
    ("A website", "qualify"),
    ("Around 2 lakhs", "qualify"),
    ("Within the next month", "qualify"),
    ("Hi", "other"),
    ("Thanks!", "other"),
    ("Good morning", "other"),
    ("What's the weather today?", "other"),
    ("Tell me a joke", "other"),
    ("Bye", "other"),
    ("Ok", "other"),
]


def retrieval_eval(vs):
    # all chunks in the index, so we can find every relevant chunk per question
    all_docs = list(vs.docstore._dict.values())
    max_k = max(K_VALUES)
    mrr, prec, rec, lat = 0.0, {k: 0.0 for k in K_VALUES}, {k: 0.0 for k in K_VALUES}, []
    for q, phrase in RETRIEVAL_SET:
        relevant = {d.page_content for d in all_docs if phrase in d.page_content}
        assert relevant, f"No chunk contains phrase: {phrase!r}"
        t = time.time()
        ranked = [d.page_content for d in vs.similarity_search(q, k=max_k)]
        lat.append(time.time() - t)
        rr = next((1 / (i + 1) for i, c in enumerate(ranked) if c in relevant), 0)
        mrr += rr
        for k in K_VALUES:
            hits = sum(1 for c in ranked[:k] if c in relevant)
            prec[k] += hits / k
            rec[k] += hits / len(relevant)
    n = len(RETRIEVAL_SET)
    print(f"Chunks indexed: {len(all_docs)} | Retrieval questions: {n}")
    print(f"MRR: {mrr / n:.3f}")
    for k in K_VALUES:
        print(f"Precision@{k}: {prec[k] / n:.3f}   Recall@{k}: {rec[k] / n:.3f}")
    print(f"Avg retrieval latency: {sum(lat) / n:.2f}s")


def routing_eval():
    from agent import ChatAgent
    agent = ChatAgent()
    correct, lat, wrong = 0, [], []
    for msg, expected in ROUTING_SET:
        t = time.time()
        got = agent._route_message(msg)
        lat.append(time.time() - t)
        if got == expected:
            correct += 1
        else:
            wrong.append((msg, expected, got))
    n = len(ROUTING_SET)
    print(f"\nRouting accuracy: {correct}/{n} = {correct / n:.1%}")
    print(f"Avg routing latency: {sum(lat) / n:.2f}s")
    for m, e, g in wrong:
        print(f"  MISS: {m!r} expected={e} got={g}")

    # end-to-end FAQ latency
    agent2 = ChatAgent()
    t = time.time()
    agent2.handle_message("How long does a website project take?")
    print(f"End-to-end FAQ answer latency: {time.time() - t:.2f}s")


if __name__ == "__main__":
    retrieval_eval(build_vectorstore())
    routing_eval()
