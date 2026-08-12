# Day 3 — paper research agent
 
  
# imports 
 
import os
from datetime import datetime, timezone
from pathlib import Path
  
from dotenv import load_dotenv
 

load_dotenv()

 
USE_FAKE = os.getenv("USE_FAKE", "0") == "1"
 
# This file lives in day3_solution/, and papers.json + skills/ sit beside it.
PROJECT_ROOT = Path(__file__).resolve().parent
 
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
 
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
 
 
# ---------- tool ----------
 
def recency_score(year: int) -> float:
    """Score a paper's recency from 0.0 to 1.0.
 
    The current year scores 1.0, each year older loses 0.15, floored at 0.0.
    Call this for every paper before ranking.
    """
    current = datetime.now(timezone.utc).year
    return max(0.0, 1.0 - 0.15 * (current - year))
 
 
# ---------- system prompt ----------
 
SYSTEM_PROMPT = (
    "You are a research agent for the AAASEC course. Your job is to help "
    "the user decide which papers are worth reading first.\n\n"
    "Papers are listed in /papers.json. Read it before answering any question "
    "about the literature — never rely on memory for titles, years, or authors.\n\n"
    "Call recency_score for every paper rather than estimating its age yourself.\n\n"
    "Relatedness is your judgment, not a tool. Score it from the title and "
    "abstract only, and say in one line why whenever you score above 0.6.\n\n"
    "When a skill matches the request, follow it exactly. Never invent a paper, "
    "a year, or a score you did not compute."
)
 
 
# ---------- the fake ----------
 
class FakeAgent:
    """Deterministic stand-in so the WHOLE pipeline (FastAPI, Docker,
    compose, curl) can be exercised with zero API keys. Same interface
    as the real agent: .ainvoke({'messages': [...]}) -> {'messages': [...]}
    """
 
    class _Msg:
        def __init__(self, content):
            self.content = content
 
    async def ainvoke(self, payload, config=None):
        user = payload["messages"][-1]
        text = user["content"] if isinstance(user, dict) else user.content
        reply = (
            f"[FAKE AGENT] I received: '{text[:120]}'. "
            f"With real keys I would read /papers.json, score each paper, and "
            f"follow the prioritize-papers skill. "
            f"recency_score(2022) would give {recency_score(2022):.2f}."
        )
        return {"messages": payload["messages"] + [self._Msg(reply)]}
 
 
# ---------- the boundary ----------
 
def build_agent():
    """Return SOMETHING with .ainvoke({'messages': [...]}).
 
    api.py depends on this signature — never on what's behind it.
    """
    if USE_FAKE:
        return FakeAgent()
 
    '''llm = ChatOpenAI(
        temperature=0,
        model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1",
    )'''

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        max_tokens=1024,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
 
    # FilesystemBackend so the agent's ls/read/write/edit tools operate on
    # this folder, and so skills/ can be read from disk. virtual_mode confines
    # paths under root_dir. No shell, no execute tool — that is Day 4.
    backend = FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=True)
 
    return create_deep_agent(
        model=llm,
        tools=[recency_score],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        skills=["/skills/"],  # virtual path under root_dir
    )
 
 
# ---------- smoke test ----------
 
if __name__ == "__main__":
    import asyncio
 
    agent = build_agent()
    result = asyncio.run(
        agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Which papers should I read first if I'm building "
                            "an LLM agent? Rank them."
                        ),
                    }
                ]
            }
        )
    )
    output = result["messages"][-1].content
    print(output)
    Path("ranked_papers.md").write_text(output, encoding="utf-8")
 
