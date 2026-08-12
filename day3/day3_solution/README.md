# Day 3 — Research Agent 

A deep agent that ranks research papers by relevance to LLM-agent work and by recency.

## Files

| File | Purpose |
|---|---|
| `research_agent.py` | Agent definition and entry point |
| `papers.json` | 10 papers (2017–2023) with title, authors, year, venue, abstract |
| `skills/research-brief/SKILL.md` | Ranking rubric and output format |
| `ranked_papers.md` | Agent output

## Design

**One tool.** `recency_score(year)` returns 0.0–1.0, losing 0.15 per year of age.
Recency is arithmetic, so it belongs in code.

**Relatedness is not a tool.** Judging how central a paper is to agent research
requires reading the abstract — keyword matching would rank a paper that says
"agent" once above one that is actually about agent architecture. That judgment
lives in the skill's rubric instead. Here i used Claude sonnet 5 as the brain 

**Weights live in markdown, not Python.** `priority = 0.7 * relatedness + 0.3 * recency`
is set in `SKILL.md`, so tuning the rubric means editing text, not code.

## Safety

`FilesystemBackend(virtual_mode=True)` confines all file access to this folder —
the agent sees it as `/`. There is no shell or execute tool. 

## Run

```bash
python day3_solution/research_agent.py
```

## Notes

Attention Is All You Need is included deliberately as a control: foundational,
oldest in the set, near-zero relatedness. If it ranks highly, the weights are wrong.
