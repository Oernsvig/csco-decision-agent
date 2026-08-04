"""Shared invoke/extract/instrument plumbing for the arm modules.

Two execution shapes recur across the four arm modules with identical
mechanics and only the prompt strings (and, for the ReAct arms, the tool
list) differing:

  - single-turn (static.py): one LLM call over a fixed
    system+human message pair, then the shared extractor.
  - ReAct agent (vector.py, lexical.py): a LangGraph ReAct loop over
    arm-specific tools, then the shared extractor over the concatenated
    AI-turn trace.

This module holds only that plumbing — no prompt text, no tool
definitions, no arm-specific logging wording. Each arm module still builds
its own system/human prompt strings and its own tools, and still owns its
own log messages around the call.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from csco.arms.extractor import extract_decision
from csco.arms.llm import get_llm
from csco.models import ScenarioDecision


def turn_input_tokens(msg) -> int:
    """Extract one model turn's input-token count from its usage_metadata (0 if absent)."""
    um = getattr(msg, "usage_metadata", None)
    try:
        return int((um or {}).get("input_tokens", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return 0


def run_single_turn(
    system_prompt: str,
    human_prompt: str,
    *,
    result=None,
    policy_tokens: int | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> ScenarioDecision:
    """One reasoning call + shared extraction. Used by static.py.

    on_reasoning, if given, is called with the raw reasoning text right after
    it's captured (before extraction) — lets a caller emit its own arm-specific
    debug log without duplicating the invoke/extract/instrument sequence.
    """
    if result is not None and policy_tokens is not None:
        result.policy_tokens = policy_tokens

    reasoning_response = get_llm().invoke(
        [SystemMessage(system_prompt), HumanMessage(human_prompt)]
    )
    agent_answer = reasoning_response.content

    if result is not None:
        result.llm_calls_reasoning += 1
        result.agent_answer_raw = agent_answer if isinstance(agent_answer, str) else ""
        result.add_usage(getattr(reasoning_response, "usage_metadata", None))
        result.reasoning_turn_inputs.append(turn_input_tokens(reasoning_response))

    if on_reasoning is not None:
        on_reasoning(agent_answer)

    decision: ScenarioDecision = extract_decision(agent_answer, result=result)

    if result is not None:
        result.llm_calls_extraction += 1

    return decision


def run_react_agent(
    llm,
    tools: list,
    system_prompt: str,
    scenario_text: str,
    *,
    result=None,
) -> ScenarioDecision:
    """ReAct loop + shared extraction. Used by vector.py / lexical.py.

    Each arm supplies its own tools (search_playbooks, or find_playcard+open);
    this only drives the agent, captures the full trace, and extracts.
    """
    agent = create_react_agent(llm, tools=tools)
    agent_output = agent.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(scenario_text)]}
    )
    ai_messages = [
        m for m in agent_output["messages"]
        if hasattr(m, "tool_calls") or (hasattr(m, "type") and m.type == "ai")
    ]
    # Concatenate all AI-turn content, not just the final message — after a long
    # tool-call chain the model's last message is often a terse summary that can
    # omit the fired provision's identifier even when an earlier reasoning step
    # named it explicitly; the extractor needs the full trace to catch it.
    agent_answer = "\n\n".join(
        m.content for m in ai_messages if isinstance(m.content, str) and m.content.strip()
    )

    if result is not None:
        result.llm_calls_reasoning = len(ai_messages)
        result.agent_answer_raw = agent_answer if isinstance(agent_answer, str) else ""
        # Accumulate token usage across every model turn in the ReAct trace
        # (reasoning + tool-call turns). tool_calls_total is counted from the
        # tool_calls attached to AI messages.
        tool_calls = 0
        for m in agent_output["messages"]:
            result.add_usage(getattr(m, "usage_metadata", None))
            tool_calls += len(getattr(m, "tool_calls", []) or [])
        result.tool_calls_total = tool_calls
        # Per-model-turn input sizes: the ReAct prefix grows monotonically, so the
        # largest (final) turn holds the full unique context. Within-run uncached
        # input = that maximum; the rest is re-sent prefix (within-run cache).
        result.reasoning_turn_inputs = [t for m in ai_messages if (t := turn_input_tokens(m)) > 0]

    decision: ScenarioDecision = extract_decision(agent_answer, result=result)

    if result is not None:
        result.llm_calls_extraction = 1

    return decision
