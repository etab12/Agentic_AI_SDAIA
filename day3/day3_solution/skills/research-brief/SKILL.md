
name: prioritize-papers
description: Rank a list of papers by relevance to LLM agents and recency. Use when asked to prioritize, rank, or triage papers.
---

# Prioritizing papers

1. Read `papers.json` from the project root.
2. For each paper, call `recency_score(year)`.
3. Assign a relatedness score 0.0-1.0 by judging the title and abstract:
   - 1.0 — agent architecture, tool use, planning, multi-agent systems
   - 0.6 — LLM capability work that agents depend on (reasoning, long context)
   - 0.3 — general LLM work with no agent framing
   - 0.0 — unrelated
4. priority = 0.7 * relatedness + 0.3 * recency
5. Output a table sorted by priority: title, year, relatedness, recency, priority.
6. State your reasoning for any relatedness score above 0.6 in one line.
