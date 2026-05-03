"""
ProcDNA Intelligence — Chat Engine
=====================================
Manages the full Claude conversation loop including tool use.

Flow per user message:
  1. Append user message to history
  2. Call Claude with system prompt + history + tool schemas
  3. If Claude calls a tool → dispatch to tools.py → append result → loop back
  4. When Claude returns final text → return to user

Session state is stored in _sessions (shared with main.py).
Conversation history is stored separately in _chat_sessions.
"""

import os
import time
import json
import httpx
from typing import List, Dict, Any, Optional

from tool_schemas import TOOL_SCHEMAS
from tools import dispatch

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama3-70b-8192"  # Fast, capable, free tier available
MAX_TOKENS        = 2048
MAX_TOOL_LOOPS    = 8     # prevent infinite tool call loops

# Conversation histories: chat_session_id → list of messages
_chat_sessions: Dict[str, List[Dict]] = {}

SYSTEM_PROMPT = """You are IC Intelligence — the ProcDNA enterprise incentive compensation planning assistant.

You help IC planning teams design, score, and validate quota and goal-setting plans. You have access to tools that query live session data from the user's uploaded sales file and generated scenarios.

## Your personality
- Concise and direct. No filler phrases.
- Think like a senior IC consultant — you understand the data and offer insight, not just facts.
- When you get data back from a tool, don't just repeat it. Interpret it. Flag what matters.

## How to use tools
- Call a tool whenever the user asks for data, analysis, or a calculation you cannot answer from memory.
- You may call multiple tools in sequence if the answer requires it.
- After getting tool results, synthesise them into a clear, helpful response.
- If a tool returns an error, tell the user plainly and suggest what to do.

## What you know without tools
- IC plan design principles, pay mix, quota mechanics, accelerator design
- What each metric means (Proportionality, Attainability, Consistency, etc.)
- Forecasting model characteristics (Holt-Winters, ARIMA, etc.)
- Industry norms for attainment bands, pay mix, plan complexity

## What requires a tool
- Any specific number from the user's session (goals, scores, territory data)
- Comparisons between scenarios
- Guardrail checks
- Downloads and exports
- What-if recalculations

## Format
- Use markdown tables when showing territory or scenario data — they render in the UI.
- Keep prose responses to 3-5 sentences unless the user asks for detail.
- Always end with a relevant follow-up question or next step suggestion.

## Session context
The user's active session ID will be provided in the conversation context.
Always use the provided session_id when calling tools — never guess or make one up.
If no session exists yet, tell the user to run Goal Setting first.
"""


def _call_groq(messages: List[Dict]) -> Dict:
    """Make one call to the Groq API (OpenAI-compatible). Returns normalised response dict."""
    # Convert system prompt + messages to OpenAI format
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": (
            m["content"] if isinstance(m["content"], str)
            else " ".join(
                b.get("text","") or b.get("content","")
                for b in (m["content"] if isinstance(m["content"], list) else [])
                if isinstance(b, dict)
            )
        )}
        for m in messages
        if m.get("role") in ("user","assistant")
    ]

    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":       GROQ_MODEL,
            "messages":    groq_messages,
            "max_tokens":  MAX_TOKENS,
            "temperature": 0.3,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    # Normalise to Anthropic-like shape so rest of code works
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
    }


def _extract_text(response: Dict) -> str:
    """Pull the text content blocks from a Claude response."""
    texts = [
        block["text"]
        for block in response.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(texts).strip()


def _extract_tool_uses(response: Dict) -> List[Dict]:
    """Pull tool_use blocks from a Claude response."""
    return [
        block
        for block in response.get("content", [])
        if block.get("type") == "tool_use"
    ]


def chat(
    user_message: str,
    chat_session_id: str,
    goal_session_id: Optional[str],
    sessions: dict,
) -> Dict[str, Any]:
    """
    Process one user message through the full Claude + tool loop.

    Args:
        user_message:     The user's text input
        chat_session_id:  ID for this conversation thread
        goal_session_id:  Active goal-setting session ID (may be None)
        sessions:         The _sessions dict from main.py

    Returns:
        {
            "reply":         str   — Claude's final text response
            "tools_called":  list  — names of tools that were called
            "tool_results":  list  — results from each tool call
        }
    """
    if not GROQ_API_KEY:
        from rule_engine import rule_based_response
        return rule_based_response(user_message, goal_session_id or "", sessions)

    # Initialise conversation history for this chat session
    if chat_session_id not in _chat_sessions:
        _chat_sessions[chat_session_id] = []

    history = _chat_sessions[chat_session_id]

    # Inject session context into the user message
    context_note = ""
    if goal_session_id and goal_session_id in sessions:
        sess = sessions[goal_session_id]
        context_note = (
            f"\n\n[System context — do not repeat to user: "
            f"Active session_id={goal_session_id}, "
            f"best_scenario={sess.get('best_id','—')}, "
            f"preset={sess.get('preset','fairness')}, "
            f"territory_count={len(sess.get('territories',[]))}, "
            f"national_forecast={sess.get('nf','—')}]"
        )
    elif not goal_session_id:
        context_note = (
            "\n\n[System context: No active goal-setting session. "
            "If the user asks for session data, tell them to run Goal Setting first.]"
        )

    # Append user message to history
    history.append({
        "role":    "user",
        "content": user_message + context_note,
    })

    tools_called  = []
    tool_results  = []
    status_labels = ["Thinking..."]   # tracks state progression for frontend
    loop_count    = 0
    start_time    = time.time()

    while loop_count < MAX_TOOL_LOOPS:
        loop_count += 1

        # Call Claude
        response = _call_groq(history)
        stop_reason = response.get("stop_reason")

        # Append Claude's full response to history
        history.append({
            "role":    "assistant",
            "content": response.get("content", []),
        })

        # If Claude is done — return the text
        if stop_reason == "end_turn":
            break

        # Tool use not supported via Groq — handled by rule engine
        # Groq always returns end_turn
        if stop_reason == "tool_use":
            tool_uses = _extract_tool_uses(response)
            tool_result_blocks = []

            for tu in tool_uses:
                tool_name  = tu["name"]
                tool_input = tu.get("input", {})
                tool_id    = tu["id"]

                tools_called.append(tool_name)
                if "Analysing your data..." not in status_labels:
                    status_labels.append("Analysing your data...")

                # Inject session_id if the tool needs it and it wasn't provided
                if "session_id" in {
                    p for s in TOOL_SCHEMAS
                    if s["name"] == tool_name
                    for p in s.get("input_schema", {}).get("properties", {})
                }:
                    if "session_id" not in tool_input and goal_session_id:
                        tool_input["session_id"] = goal_session_id

                # Dispatch to the tool function
                result = dispatch(tool_name, tool_input, sessions)
                tool_results.append({"tool": tool_name, "result": result})

                tool_result_blocks.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_id,
                    "content":     json.dumps(result, default=str),
                })

            # Append tool results to history and loop back to Claude
            history.append({
                "role":    "user",
                "content": tool_result_blocks,
            })
            continue

        # Any other stop reason — break
        break

    # Status: generating final response
    if len(tools_called) > 0 and "Generating your plan..." not in status_labels:
        status_labels.append("Generating your plan...")

    # Enforce minimum 1.2 second delay so the UI feels premium
    elapsed = time.time() - start_time
    if elapsed < 1.2:
        time.sleep(1.2 - elapsed)

    # Extract final text reply
    final_text = ""
    for block in history[-1].get("content", []) if isinstance(history[-1].get("content"), list) else []:
        if isinstance(block, dict) and block.get("type") == "text":
            final_text += block["text"]
    if not final_text:
        final_text = _extract_text(response)

    # Trim history to last 30 messages to avoid token bloat
    if len(history) > 30:
        _chat_sessions[chat_session_id] = history[-30:]

    return {
        "reply":         final_text.strip(),
        "tools_called":  tools_called,
        "tool_results":  tool_results,
        "status_labels": status_labels,
        "elapsed_ms":    round((time.time() - start_time) * 1000),
    }


def get_history(chat_session_id: str) -> List[Dict]:
    """Return conversation history for a chat session (text only, no tool blocks)."""
    history = _chat_sessions.get(chat_session_id, [])
    readable = []
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, str):
            readable.append({"role": msg["role"], "text": content})
        elif isinstance(content, list):
            texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                readable.append({"role": msg["role"], "text": "\n".join(texts)})
    return readable


def clear_history(chat_session_id: str) -> None:
    """Clear conversation history for a chat session."""
    if chat_session_id in _chat_sessions:
        del _chat_sessions[chat_session_id]
