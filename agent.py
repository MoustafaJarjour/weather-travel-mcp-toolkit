"""
OpenAI Agents SDK Host & MCP Client (`agent.py`)
Course: EECE 503P/798S - Agentic Systems (C5: Agent Tools)

Connects to the FastMCP server (`server.py`) over JSON-RPC 2.0 `stdio` transport
using `MCPServerStdio` from the OpenAI Agents SDK (`openai-agents`).

Demonstrates full Host <-> Client <-> Server round-trip execution across:
  1. `get_weather_forecast`    (Data Tool — read-only live weather forecast lookup)
  2. `save_travel_itinerary`   (Action Tool — single-purpose JSON persistence)
  3. `plan_weather_smart_trip` (Orchestration Tool — sequential Data + Rules + Action workflow)
  4. Predictable failure cases (unavailable city resource & invalid input constraints)

Usage:
  python agent.py                    # Run full 5-query round-trip demonstration
  python agent.py --interactive      # Launch interactive multi-turn CLI session
  python agent.py --query "..."      # Execute a single custom query
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from collections import deque
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

from agents import (
    Agent,
    Model,
    ModelResponse,
    RunContextWrapper,
    RunHooks,
    Runner,
    Tool,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio
from offline_fallback import LocalDeterministicAgentModel

# Ensure clean UTF-8 output (`°C`, `→`, `✓`) across Windows PowerShell / CMD
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(override=True)

SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"

AGENT_INSTRUCTIONS = """You are the Weather & Travel Planning Assistant, an AI agent connected to a FastMCP server.
Route every user request to the single most appropriate MCP tool:

1. `get_weather_forecast` (Data Tool):
   Use ONLY for read-only weather forecast inquiries when the user does NOT ask to plan or save an itinerary.

2. `save_travel_itinerary` (Action Tool):
   Use ONLY when the user explicitly supplies their own packing items or checklist details to save directly.
   Pass the user's items as given (even if empty, so server validation can run) without inventing extra items.

3. `plan_weather_smart_trip` (Orchestration Tool):
   Use whenever the user asks to plan/prepare a trip, generate a weather-adapted packing checklist and save it,
   or handle end-to-end trip preparation for a destination.

Always pass user-requested parameters (such as city name or day counts) directly to the tool so the MCP server's
validation rules are authoritative. When the MCP server returns a validation or resource error, clearly explain
the error to the user and suggest how to correct the request.
"""


# ---------------------------------------------------------------------------
# Full Round-Trip Lifecycle Logger (Host -> MCP Client -> FastMCP Server -> Host)
# ---------------------------------------------------------------------------

def _pretty_preview_mcp_payload(raw_result: Any, max_lines: int = 46) -> list[str]:
    """Unwraps MCP TextContent payloads and formats indented JSON lines while preserving head and tail fields."""
    parsed: Any = raw_result
    if isinstance(raw_result, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw_result)
                break
            except (json.JSONDecodeError, ValueError, SyntaxError, TypeError):
                continue

    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]

    text_body = parsed.get("text", raw_result) if isinstance(parsed, dict) else raw_result
    try:
        json_obj = json.loads(str(text_body))
        formatted = json.dumps(json_obj, indent=2, ensure_ascii=False).splitlines()
        if len(formatted) > max_lines:
            head_count = 26
            tail_count = 18
            omitted = len(formatted) - head_count - tail_count
            return (
                formatted[:head_count]
                + [f"  ... ({omitted} intermediate lines omitted for display) ..."]
                + formatted[-tail_count:]
            )
        return formatted
    except json.JSONDecodeError:
        return [str(text_body)]


class MCPRoundTripHooks(RunHooks):
    """Logs every stage of the Agent <-> MCP Client <-> FastMCP Server round trip, including JSON args."""

    def __init__(self) -> None:
        self._pending_tool_args: deque[tuple[str, str]] = deque()

    async def on_agent_start(self, context: RunContextWrapper[Any], agent: Agent[Any]) -> None:
        print(f"  [1. MCP Host] Agent '{agent.name}' analyzing user prompt...", flush=True)

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        """Captures exact JSON tool arguments emitted by the LLM prior to MCP client dispatch."""
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                t_name = getattr(item, "name", "unknown_tool")
                t_args = getattr(item, "arguments", "{}")
                self._pending_tool_args.append((t_name, t_args))

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
    ) -> None:
        args_str = "{}"
        if self._pending_tool_args:
            _, args_str = self._pending_tool_args.popleft()
        print(f"  [2. Agent -> MCP Client -> Server] JSON-RPC tools/call: `{tool.name}`")
        print(f"      Arguments : {args_str}", flush=True)

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
        result: str,
    ) -> None:
        print(f"  [3. FastMCP Server -> Client -> Agent] Response from `{tool.name}`:")
        for line in _pretty_preview_mcp_payload(result):
            print(f"      {line}")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# MCP Host / Client Orchestration
# ---------------------------------------------------------------------------

def format_mcp_tool_error(context: RunContextWrapper[Any], error: Exception) -> str:
    """Formats predictable MCP server errors cleanly into the Agent's observation context."""
    return f"MCP Tool Execution Error: {error}"


def resolve_agent_model(force_offline: bool = False) -> str | Model:
    """Uses OpenAI's hosted LLM when `OPENAI_API_KEY` is configured, else deterministic local model."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not force_offline and api_key and not api_key.startswith("your_"):
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    set_tracing_disabled(True)
    return LocalDeterministicAgentModel()


def _print_startup_banner(
    server_name: str,
    model: str | Model,
    discovered_tools: list[Any],
    interactive: bool = False,
) -> None:
    """Prints MCP Host/Client connection telemetry, active model mode, and discovered tool schemas."""
    model_label = (
        f"{model} (Live OpenAI API)"
        if isinstance(model, str)
        else "LocalDeterministicAgentModel (Offline Deterministic Fallback)"
    )
    print("=" * 80)
    header = "MCP HOST & CLIENT INITIALIZED (OpenAI Agents SDK + FastMCP over JSON-RPC stdio)"
    if interactive:
        header += " [INTERACTIVE]"
    print(header)
    print(f"  Connected MCP Server : {server_name}")
    print(f"  Active Agent Model   : {model_label}")
    print(f"  Discovered MCP Tools ({len(discovered_tools)}):")
    for t in discovered_tools:
        ro_hint = getattr(t.annotations, "read_only_hint", None) if t.annotations else None
        out_props = list((getattr(t, "output_schema", None) or {}).get("properties", {}).keys())
        print(f"    • {t.name:<24} | readOnlyHint={str(ro_hint):<5} | outputSchema={len(out_props)} typed fields")
    print("=" * 80, flush=True)


async def run_agent_with_mcp(queries: list[str], force_offline: bool = False) -> list[str]:
    """
    Spawns `server.py` over `stdio` via `MCPServerStdio`, introspects exposed MCP tools
    and schema metadata, and runs each user query end-to-end through `Runner.run`.
    """
    mcp_server = MCPServerStdio(
        name="WeatherTravelFastMCPServer",
        params={
            "command": sys.executable,
            "args": [str(SERVER_SCRIPT)],
            "env": os.environ.copy(),
        },
        client_session_timeout_seconds=30.0,
        failure_error_function=format_mcp_tool_error,
    )

    model = resolve_agent_model(force_offline=force_offline)

    outputs: list[str] = []
    async with mcp_server as server:
        discovered_tools = await server.list_tools()
        _print_startup_banner(server.name, model, discovered_tools, interactive=False)

        agent = Agent(
            name="WeatherTravelAssistantAgent",
            instructions=AGENT_INSTRUCTIONS,
            model=model,
            mcp_servers=[server],
        )

        for idx, query in enumerate(queries, start=1):
            hooks = MCPRoundTripHooks()
            print(f"\n[Query #{idx}] User: {query}")
            print("-" * 80, flush=True)
            result = await Runner.run(agent, input=query, hooks=hooks)
            print("\n  [4. MCP Host / Agent Final Response]:")
            for line in str(result.final_output).splitlines():
                print(f"    {line}")
            print("-" * 80, flush=True)
            outputs.append(str(result.final_output))

    return outputs


async def interactive_loop(force_offline: bool = False) -> None:
    """Runs an interactive multi-turn CLI session connected to the FastMCP server."""
    mcp_server = MCPServerStdio(
        name="WeatherTravelFastMCPServer",
        params={
            "command": sys.executable,
            "args": [str(SERVER_SCRIPT)],
            "env": os.environ.copy(),
        },
        client_session_timeout_seconds=30.0,
        failure_error_function=format_mcp_tool_error,
    )

    model = resolve_agent_model(force_offline=force_offline)
    async with mcp_server as server:
        discovered_tools = await server.list_tools()
        _print_startup_banner(server.name, model, discovered_tools, interactive=True)
        print("Type a travel/weather request, or 'exit' / 'quit' to close the MCP session.", flush=True)

        agent = Agent(
            name="WeatherTravelAssistantAgent",
            instructions=AGENT_INSTRUCTIONS,
            model=model,
            mcp_servers=[server],
        )

        turn = 0
        while True:
            try:
                print("\nYou > ", end="", flush=True)
                raw_line = await asyncio.to_thread(sys.stdin.readline)
                if not raw_line:
                    print("\nExiting interactive session.", flush=True)
                    break
                user_input = raw_line.strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive session.", flush=True)
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Closing MCP stdio session. Goodbye!", flush=True)
                break

            turn += 1
            print("-" * 80, flush=True)
            hooks = MCPRoundTripHooks()
            result = await Runner.run(agent, input=user_input, hooks=hooks)
            print("\n  [4. MCP Host / Agent Final Response]:")
            for line in str(result.final_output).splitlines():
                print(f"    {line}")
            print("-" * 80, flush=True)


DEFAULT_DEMO_QUERIES = [
    # 1. Data Tool (`get_weather_forecast`)
    "What is the 3-day weather forecast in Beirut in celsius?",
    # 2. Action Tool (`save_travel_itinerary`)
    "Please save a 4-day travel itinerary for Paris with high priority: Passport, Trench coat, Umbrella, Walking shoes",
    # 3. Orchestration Tool (`plan_weather_smart_trip`)
    "Plan a smart 3-day trip to Reykjavik in celsius and save my weather-adapted packing itinerary.",
    # 4. Predictable Failure Case #1 (Unavailable Resource in Data Tool)
    "What is the 3-day weather forecast in Atlantis_UnknownCity?",
    # 5. Predictable Failure Case #2 (Invalid Input Constraint in Action Tool)
    "Save a 3-day travel itinerary for Tokyo with empty packing items: []",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI Agents SDK Host/Client for the Weather & Travel FastMCP Server."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single custom query through the agent.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive multi-turn CLI session.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the local deterministic Agents SDK model instead of calling the OpenAI API.",
    )
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_loop(force_offline=args.offline))
    elif args.query:
        asyncio.run(run_agent_with_mcp([args.query], force_offline=args.offline))
    else:
        asyncio.run(run_agent_with_mcp(DEFAULT_DEMO_QUERIES, force_offline=args.offline))


if __name__ == "__main__":
    main()
