# ============================================================
# DAY 2 LAB — SKELETON: Build a Multi-Agent Research Team
# ============================================================

#
#   Day 1 (single agent)              Day 2 (multi-agent)
#   ─────────────────────             ─────────────────────────────
#   nodes = Python functions          nodes = LLM agents w/ personas
#   routing = your if/else            routing = supervisor LLM decides
#   one prompt for everything         one system prompt PER agent
#   tools available everywhere        tools SCOPED (only researcher
#                                       can search the web)
#   loop = quality-score retry        loop = critic sends draft back
#                                       to writer for revision
#
# What does NOT change: State + Nodes + Edges. A multi-agent system
# is STILL just a StateGraph. If you can build Day 1, you can build
# this — the new ideas are personas, the supervisor, and guardrails.
#
# The system you're building (the SUPERVISOR pattern):
#
#              ┌──────────── supervisor ─────────────┐
#              │       (LLM decides who's next)      │
#     ┌────────┼───────────┬───────────┬─────────────┤
#     ↓        ↓           ↓           ↓             ↓
#  researcher  analyst    writer     critic       FINISH
#     │        │           │           │             ↓
#     └────────┴───────────┴───────────┘            END
#          (every worker reports back to the supervisor)
#
# Recommended reading BEFORE you start (~25 min):
#   1. Multi-agent concepts (architectures, supervisor pattern):
#      https://docs.langchain.com/oss/python/langgraph/multi-agent
#   2. Refresh: conditional branching + loops (you need both again):
#      https://docs.langchain.com/oss/python/langgraph/use-graph-api#conditional-branching
#   3. Structured output (the supervisor's decision is structured!):
#      https://docs.langchain.com/oss/python/langchain/structured-output
#
# Setup: same as Day 1 — `uv sync`, keys in .env, or USE_FAKE=1.
# ============================================================

import os
import operator
from datetime import datetime
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

# STEP 0 
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.vectorstores import InMemoryVectorStore


load_dotenv()

USE_FAKE = os.getenv("USE_FAKE", "0") == "1"

MAX_REVISIONS = 2      # cap on writer↔critic loops
MAX_TURNS = 12         # cap on total supervisor decisions


# ============================================================
# STEP 1 — SHARED STATE: the team's "blackboard"
# ============================================================
# Day 1's state was a data PIPELINE (each field filled once, in
# order). Day 2's state is a BLACKBOARD: every agent reads all of
# it and writes only its own section; the supervisor reads it to
# decide who goes next.
#
# Define a TypedDict with:
#   task (str)
#   research_notes  <- List[str], APPEND-ONLY (which reducer? Day 1!)
#   analysis (str), draft (str), critique (str)
#   revision_count (int), turn_count (int)
#   next_agent (str)   <- the supervisor writes its decision HERE
#   execution_logs     <- append-only, same as Day 1
#
# ASK YOURSELF: why must research_notes append but draft overwrite?
# What would happen to the revision loop if draft used operator.add?

class TeamState(TypedDict):
    task: str
    research_notes: Annotated[List[str], operator.add]
    analysis: str
    draft: str
    critique: str
    revision_count: int
    turn_count: int
    next_agent: str
    execution_logs: Annotated[List[str], operator.add]


# ============================================================
# STEP 2 — STRUCTURED ROUTING DECISION
# ============================================================
# Day 1: structured output produced a quality SCORE.
# Day 2: structured output produces a ROUTING DECISION — this is
# the trick that turns an LLM into a supervisor. Literal[...] means
# the model CANNOT invent an agent that doesn't exist.
#
# WHERE TO LOOK: structured-output docs (same page as Day 1).

class RouterDecision(BaseModel):
    """The supervisor's choice of who acts next."""
    next_agent: Literal["researcher", "analyst", "writer", "critic", "FINISH"]
    reason: str = Field(description="One sentence explaining the choice")


# ============================================================
# STEP 3 — ONE LLM, FOUR PERSONAS (+ tools scoped per agent)
# ============================================================
# A multi-agent "team" doesn't need four models — it needs four
# SYSTEM PROMPTS. (In production you might also vary the model per
# agent: cheap model for the critic, big one for the writer.)
#
# TODO:
# 1. Write a PERSONAS dict: role -> system prompt, for
#    "researcher", "analyst", "writer", "critic".
#    Each persona must say what the agent DOES and what it MUST NOT
#    do (e.g. the researcher never analyzes). Boundaries between
#    agents live in the prompts — write them sharp.
# 2. Create llm (ChatOpenAI + OpenRouter, exactly like Day 1) and
#    search_tool (TavilySearch(max_results=4)).
# 3. supervisor_llm = llm.with_structured_output(RouterDecision)
# 4. Helper: run_persona(role, user_content) → invoke llm with
#    [SystemMessage(PERSONAS[role]), HumanMessage(user_content)]
#    and return response.content.
#
# TOOL SCOPING: only the researcher node may call search_tool.
# That's a deliberate design decision, not a limitation — ask
# yourself what could go wrong if the critic could search.


# Identify the 4 personas
PERSONAS = {
    "researcher": (
        "You are an expert researcher. Your role in the team is to gather facts and sources on the topic. "
        "You do NOT analyze, judge, or draw conclusions — you only report what you found, "
        "with sources. Hand raw findings to the next agent."
    ),
    "analyst": (
        "You are an analyst. Your task is to identify patterns,"
        "trade-offs, and implications in the research findings. You do NOT gather new facts and you do NOT "
        "critique the work — only interpret what you were given."
    ),
    "critic": (
        "You are a critic. You review the analysis for weak reasoning, unsupported "
        "claims, and missing considerations. You do NOT add new research or rewrite "
        "the work — only identify problems, specifically and concretely."
    ),
    "writer": (
        "You are a writer. You synthesize research, analysis, and critique into a "
        "clear final answer for the user. You do NOT introduce new claims — only "
        "present what the other agents produced."
    ),
}

SUPERVISOR_PROMPT = (
    "You are the SUPERVISOR of a research team (researcher, analyst, writer, critic). "
    "Given the team's progress, pick who acts next. Standard order: researcher -> analyst -> writer -> critic. "
    "If the critique starts with REVISE and revisions < max, send the writer. "
    "If the critique is APPROVED or revisions are maxed out, FINISH."
)

if USE_FAKE:
    class FakeWorker:
        def __init__(self):
            self.critic_calls = 0

        def invoke_persona(self, role, _messages):
            if role == "researcher":
                return (
                    "- Fact A: enterprises adopt supervisor-pattern agent teams (src: example.com/a)\n"
                    "- Fact B: tool scoping reduces agent error rates (src: example.com/b)\n"
                    "- Fact C: revision loops improve output quality (src: example.com/c)"
                )
            if role == "analyst":
                return (
                    "The pattern across sources: coordination, not raw model power, drives multi-agent value. "
                    "Scoped tools and review loops are the recurring levers; unstructured agent swarms underperform supervised teams."
                )
            if role == "writer":
                return (
                    "HEADLINE: Supervised agent teams beat solo agents.\n"
                    "Findings: (1) supervisor routing enables specialization; (2) tool scoping cuts errors; (3) critic loops raise quality.\n"
                    "Recommendation: pilot a supervisor-pattern team on one workflow."
                )
            if role == "critic":
                self.critic_calls += 1
                if self.critic_calls == 1:
                    return "REVISE: cite the sources from the research notes; quantify finding (2)."
                return "APPROVED"
            return ""

    fake_worker = FakeWorker()

    def run_persona(role: str, user_content: str) -> str:
        return fake_worker.invoke_persona(role, user_content)

    def supervisor_decide(state: TeamState) -> RouterDecision:
        if not state["research_notes"]:
            return RouterDecision(next_agent="researcher", reason="No research yet.")
        if not state["analysis"]:
            return RouterDecision(next_agent="analyst", reason="Research done, needs analysis.")
        if not state["draft"]:
            return RouterDecision(next_agent="writer", reason="Analysis done, needs a draft.")
        if not state["critique"]:
            return RouterDecision(next_agent="critic", reason="Draft ready for review.")
        if state["critique"].startswith("REVISE") and state["revision_count"] < MAX_REVISIONS:
            return RouterDecision(next_agent="writer", reason="Critic requested changes.")
        return RouterDecision(next_agent="FINISH", reason="Draft approved (or revision cap hit).")

else:
    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )

    search_tool = TavilySearch(max_results=4)
    supervisor_llm = llm.with_structured_output(RouterDecision)

    def run_persona(role: str, user_content: str) -> str:
        response = llm.invoke([
            SystemMessage(content=PERSONAS[role]),
            HumanMessage(content=user_content),
        ])
        return response.content

    def supervisor_decide(state: TeamState) -> RouterDecision:
        status = (
            f"Task: {state['task']}\n"
            f"Research notes: {'YES (' + str(len(state['research_notes'])) + ')' if state['research_notes'] else 'none'}\n"
            f"Analysis: {'YES' if state['analysis'] else 'none'}\n"
            f"Draft: {'YES' if state['draft'] else 'none'}\n"
            f"Critique: {state['critique'] or 'none'}\n"
            f"Revisions so far: {state['revision_count']} (max {MAX_REVISIONS})\n"
        )
        return supervisor_llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=status),
        ])


# ============================================================
# STEP 4 — THE SUPERVISOR NODE (the piece Day 1 didn't have)
# ============================================================
# The supervisor node must:
# 1. Increment turn_count.
# 2. Build a STATUS SUMMARY of the blackboard (which sections are
#    filled? what does the critique say? how many revisions?).
#    Don't dump the full text of everything — the supervisor needs
#    STATUS, not content. (Why? Think tokens and attention.)
# 3. Ask supervisor_llm for a RouterDecision.
# 4. GUARDRAILS — never trust an LLM to terminate a loop:
#      a) if turn_count > MAX_TURNS → force FINISH
#      b) if the LLM picks writer/critic but revision_count >=
#         MAX_REVISIONS and a draft exists → force FINISH
#    This is Day 1's iteration cap wearing a new hat. Same lesson:
#    the LLM proposes, YOUR CODE disposes.
# 5. Return {"next_agent": ..., "turn_count": ..., "execution_logs": [...]}
#
# WHERE TO LOOK: multi-agent docs → "Supervisor" section.




def supervisor_node(state: TeamState):
    turn_count = state["turn_count"] + 1

    status = (
        f"Task: {state['task']}\n"
        f"Research notes: {'YES (' + str(len(state['research_notes'])) + ')' if state['research_notes'] else 'none'}\n"
        f"Analysis: {'YES' if state['analysis'] else 'none'}\n"
        f"Draft: {'YES' if state['draft'] else 'none'}\n"
        f"Critique: {state['critique'] or 'none'}\n"
        f"Turn: {turn_count}/{MAX_TURNS}\n"
        f"Revision_count: {state['revision_count']}/{MAX_REVISIONS}\n"
    )

    if turn_count > MAX_TURNS:
        decision = RouterDecision(next_agent="FINISH", reason=f"forced: turn cap {MAX_TURNS} exceeded")
    else:
        decision = supervisor_decide(state)
        if (
            decision.next_agent in ("writer", "critic")
            and state["revision_count"] >= MAX_REVISIONS
            and state.get("draft")
        ):
            decision = RouterDecision(next_agent="FINISH", reason=f"forced: revision cap {MAX_REVISIONS} reached")

    return {
        "next_agent": decision.next_agent,
        "turn_count": turn_count,
        "execution_logs": [f"[turn {turn_count}] supervisor → {decision.next_agent} ({decision.reason})"],
    }


# ============================================================
# STEP 5 — WORKER AGENT NODES
# ============================================================
# Each worker: read the blackboard → act in persona → return a
# PARTIAL update with ONLY its own section (Day 1 rule, unchanged).

def researcher_node(state: TeamState):
    """Search the web (ONLY this agent may), condense to notes."""
    if USE_FAKE:
        notes = run_persona("researcher", state["task"])
        source_count = 1
    else:
        results = search_tool.invoke({"query": state["task"]})["results"]
        source_count = len(results)
        raw = "\n\n".join(
            f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}"
            for r in results
        )
        notes = run_persona(
            "researcher",
            f"Task: {state['task']}\n\nSearch results:\n{raw}",
        )

    # LIST — research_notes is append-only, so round 2 joins round 1
    return {
        "research_notes": [notes],
        "execution_logs": [f"researcher: condensed {source_count} sources into notes"],
    }
 
 
def analyst_node(state: TeamState):
    """Turn raw notes into analysis."""
    notes = "\n\n---\n\n".join(state["research_notes"])
 
    analysis = run_persona(
        "analyst",
        f"Task: {state['task']}\n\nResearch notes:\n{notes}",
    )
 
    return {
        "analysis": analysis,
        "execution_logs": ["analyst: produced analysis from research notes"],
    }
 
 
def writer_node(state: TeamState):
    """Write the draft — or REVISE it if a critique is present."""
    critique = state.get("critique", "")
    revising = bool(critique) and critique.startswith("REVISE")
 
    if revising:
        user_content = (
            f"Task: {state['task']}\n\n"
            f"Analysis:\n{state['analysis']}\n\n"
            f"Your previous draft:\n{state['draft']}\n\n"
            f"Critique to address:\n{critique}\n\n"
            f"Revise the draft so that every issue raised is fixed."
        )
    else:
        user_content = (
            f"Task: {state['task']}\n\n"
            f"Analysis:\n{state['analysis']}\n\n"
            f"Write the draft."
        )
 
    draft = run_persona("writer", user_content)
    revision_count = state["revision_count"] + (1 if revising else 0)
 
    return {
        "draft": draft,
        # Reset: a critique is a work order, not a record. Once addressed it
        # describes a draft that no longer exists. Leaving it set would make
        # the writer loop on a ghost and the supervisor route on stale data.
        # Empty critique now means "nobody has reviewed the current draft".
        "critique": "",
        "revision_count": revision_count,
        "execution_logs": [
            f"writer: {'revised' if revising else 'wrote'} draft "
            f"(revision {revision_count})"
        ],
    }
 
 
def critic_node(state: TeamState):
    """Review the draft against the research notes."""
    notes = "\n\n---\n\n".join(state["research_notes"])
 
    critique = run_persona(
        "critic",
        f"Task: {state['task']}\n\n"
        f"Research notes:\n{notes}\n\n"
        f"Draft to review:\n{state['draft']}",
    )
 
    return {
        "critique": critique,
        "execution_logs": [f"critic: {critique[:70]}"],
    }
 


# ============================================================
# STEP 6 — ROUTING FUNCTION + WIRE THE GRAPH
# ============================================================
# The conditional-edge function is now TRIVIAL — it just reads the
# supervisor's decision:
#
#     def route_from_supervisor(state) -> str:
#         return state["next_agent"]
#
# Compare with Day 1, where all decision logic lived inside
# quality_router. The intelligence MOVED from the edge into a node.
#
# Wiring checklist:
# 1. add all five nodes
# 2. START → supervisor
# 3. add_conditional_edges("supervisor", route_from_supervisor,
#        {"researcher": "researcher", "analyst": "analyst",
#         "writer": "writer", "critic": "critic", "FINISH": END})
# 4. EVERY worker gets an edge BACK to supervisor — the
#    hub-and-spoke shape that defines the supervisor pattern.
#    (A for-loop over the four worker names is idiomatic.)

# TODO: route_from_supervisor + graph wiring


def route_from_supervisor(state: TeamState) -> str:
    return state["next_agent"]


builder = StateGraph(TeamState)

# 1. all five nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("analyst", analyst_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)

# 2. START → supervisor
builder.add_edge(START, "supervisor")

# 3. supervisor fans out
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "critic": "critic",
        "FINISH": END,
    },
)

# 4. every worker returns to the hub
for worker in ["researcher", "analyst", "writer", "critic"]:
    builder.add_edge(worker, "supervisor")

app = builder.compile()
# ============================================================
# STEP 7 — COMPILE, VISUALIZE, RUN
# ============================================================
# Same as Day 1: compile with InMemorySaver, print the Mermaid
# diagram (it should look like a STAR, not Day 1's chain), stream
# with stream_mode="values" and a thread_id, print the final draft.
#
# EXPERIMENT 1: set MAX_REVISIONS = 0. What happens to quality?
# EXPERIMENT 2: delete guardrail (a) and make the critic always
#   say REVISE. Watch the turn cap save you — then delete guardrail
#   (b) too and meet your old friend GraphRecursionError.
# EXPERIMENT 3: swap the analyst's persona for a terrible one
#   ("you are vague and generic"). How far does the damage spread
#   through the team? This is why persona boundaries matter.

if __name__ == "__main__":
    initial_state = {
        "task": "Should our company adopt multi-agent AI systems in 2026?",
        "research_notes": [],
        "analysis": "",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "turn_count": 0,
        "next_agent": "",
        "execution_logs": [],
    }
    # TODO: compile, visualize, stream, print final draft + stats
    print(app.get_graph().draw_mermaid())

    # run 
    config = {"configurable": {"thread_id": "day2-run-1"}}

    final = None
    for chunk in app.stream(initial_state, config, stream_mode="values"):
        final = chunk
        if chunk.get("execution_logs"):
            print(chunk["execution_logs"][-1])

    # print results 
    print("\n" + "=" * 60)
    print("FINAL DRAFT")
    print("=" * 60)
    print(final["draft"])

    print("\n" + "=" * 60)
    print("STATS")
    print("=" * 60)
    print(f"turns:           {final['turn_count']}")
    print(f"revisions:       {final['revision_count']}")
    print(f"research rounds: {len(final['research_notes'])}")
    print(f"log entries:     {len(final['execution_logs'])}")
# Additional experiments

# ============================================================
# SELF-CHECK before you look at the solution
# ============================================================
# [ ] I can explain the supervisor pattern in one sentence
# [ ] My routing function reads state — the DECISION was made in a node
# [ ] research_notes appends; draft overwrites; I know why each
# [ ] The writer RESETS critique — I can explain what breaks if not
#     (hint: what does the supervisor see on the turn after a revision?)
# [ ] Only researcher_node touches search_tool
# [ ] My supervisor has BOTH guardrails, and I triggered EXPERIMENT 2
# [ ] My Mermaid diagram is a star: supervisor in the middle
# [ ] I can name one task where Day 1's single agent is the BETTER
#     design (multi-agent is not free: more calls, more latency,
#     more places to break — coordination must earn its cost)

