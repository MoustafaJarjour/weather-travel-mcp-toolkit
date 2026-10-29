"""
Offline Deterministic Fallback Model (`offline_fallback.py`)
Course: EECE 503P/798S - Agentic Systems (C5: Agent Tools)

Provides `LocalDeterministicAgentModel`, an implementation of `agents.Model`
from the OpenAI Agents SDK. Used automatically ONLY when `OPENAI_API_KEY` is not
configured (or when `--offline` is passed), allowing unit tests and offline checks
to execute the genuine `Runner.run()` + `MCPServerStdio` control loop without
external API calls.
"""

from __future__ import annotations

import ast
from collections.abc import AsyncIterator
import json
import re
from typing import Any
import uuid

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agents import (
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Tool,
    Usage,
)
from agents.items import TResponseInputItem


class LocalDeterministicAgentModel(Model):
    """Local `agents.Model` adapter for deterministic offline unit tests."""

    def _extract_last_user_text(self, input_data: str | list[TResponseInputItem]) -> str:
        if isinstance(input_data, str):
            return input_data
        for item in reversed(input_data):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content
        return str(input_data)

    def _extract_tool_output(self, input_data: str | list[TResponseInputItem]) -> dict[str, Any] | None:
        if not isinstance(input_data, list):
            return None
        for item in reversed(input_data):
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                return item
        return None

    def _route_query_to_tool_call(self, query: str) -> tuple[str, dict[str, Any]]:
        q_lower = query.lower()

        days_match = re.search(r"(-?\d+)\s*(?:-|\s)?\s*day", q_lower)
        days = int(days_match.group(1)) if days_match else 3
        units = "fahrenheit" if "fahrenheit" in q_lower else "celsius"

        known_cities = [
            "beirut", "london", "paris", "tokyo",
            "reykjavik", "dubai", "new york", "cairo",
        ]
        detected_city = "Beirut"
        city_match = re.search(r"(?:in|for|to)\s+([A-Za-z0-9_]+(?:\s+[A-Z][a-zA-Z0-9_]+)?)", query)
        for c in known_cities:
            if c in q_lower:
                detected_city = c.title()
                break
        else:
            if city_match:
                detected_city = city_match.group(1).strip().rstrip("?.!,")

        if "empty city" in q_lower or 'city=""' in q_lower or "city ''" in q_lower:
            detected_city = ""

        # 1. Orchestration Tool (`plan_weather_smart_trip`)
        if any(kw in q_lower for kw in ["plan", "prepare", "smart trip", "orchestrate", "what should i pack for"]):
            notes = "Business conference trip" if "conference" in q_lower else ""
            return (
                "plan_weather_smart_trip",
                {
                    "city": detected_city,
                    "days": days,
                    "units": units,
                    "traveler_notes": notes,
                },
            )

        # 2. Action Tool (`save_travel_itinerary`)
        if any(kw in q_lower for kw in ["save", "record", "store", "checklist", "itinerary:"]):
            priority = "medium"
            for p in ("high", "low", "medium", "urgent"):
                if f"{p} priority" in q_lower or f"priority {p}" in q_lower:
                    priority = p
                    break

            if "empty packing" in q_lower or "no items" in q_lower or "empty checklist" in q_lower:
                items: list[str] = []
            else:
                colon_idx = query.find(":")
                if colon_idx != -1:
                    raw_part = query[colon_idx + 1 :]
                    items = [
                        i.strip().strip(".")
                        for i in re.split(r",|\band\b", raw_part)
                        if i.strip().strip(".")
                    ]
                else:
                    items = ["Passport", "Charger", "Walking shoes"]

            return (
                "save_travel_itinerary",
                {
                    "city": detected_city,
                    "trip_days": days,
                    "packing_items": items,
                    "priority": priority,
                    "notes": "User-submitted custom packing checklist",
                },
            )

        # 3. Data Tool (`get_weather_forecast`)
        return (
            "get_weather_forecast",
            {
                "city": detected_city,
                "days": days,
                "units": units,
            },
        )

    def _synthesize_final_answer(self, raw_output: Any) -> str:
        parsed: Any = raw_output
        if isinstance(raw_output, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(raw_output)
                    break
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue

        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]

        if isinstance(parsed, dict) and "text" in parsed:
            inner_text = str(parsed["text"])
            try:
                parsed = json.loads(inner_text)
            except json.JSONDecodeError:
                raw_output = inner_text
                parsed = None

        if not isinstance(parsed, dict):
            return (
                f"The MCP Server reported a validation or resource error:\n"
                f"  -> {raw_output}\n"
                f"Please check the input parameters (city spelling, days 1-7, or non-empty packing items)."
            )

        if "daily_forecasts" in parsed and "orchestration_status" not in parsed:
            unit_sym = parsed.get("unit_symbol", "°C")
            lines = [
                f"Weather Forecast for {parsed['city']}, {parsed['country']} "
                f"({parsed['days_requested']}-Day Outlook, source: {parsed['source']}):",
                f"Summary: {parsed['summary_condition']}",
            ]
            for day in parsed["daily_forecasts"]:
                lines.append(
                    f"  - {day['date']}: {day['condition']} | High: {day['temp_high']}{unit_sym}, "
                    f"Low: {day['temp_low']}{unit_sym} | Precip: {day['precipitation_chance_pct']}% | "
                    f"Wind: {day['wind_speed_kmh']} km/h"
                )
            return "\n".join(lines)

        if parsed.get("status") == "saved" and "orchestration_status" not in parsed:
            items_str = ", ".join(parsed["packing_items"])
            return (
                f"Travel Itinerary Saved Successfully!\n"
                f"  - Trip ID: {parsed['trip_id']}\n"
                f"  - Destination: {parsed['city']} ({parsed['trip_days']} days, Priority: {parsed['priority'].upper()})\n"
                f"  - Packing Items ({len(parsed['packing_items'])}): {items_str}"
            )

        if parsed.get("orchestration_status") == "completed":
            saved = parsed["saved_itinerary"]
            wx = parsed["weather_overview"]
            steps = "\n".join(f"    {s}" for s in parsed["steps_executed"])
            items = "\n".join(f"    * {item}" for item in parsed["recommended_packing_items"])
            return (
                f"End-to-End Weather-Smart Trip Plan Completed for {parsed['destination']}!\n"
                f"  Orchestration Trace:\n{steps}\n"
                f"  Weather Summary: {wx['summary']}\n"
                f"  Travel Advisory: {parsed['weather_advisory']}\n"
                f"  Recommended Packing Checklist ({len(parsed['recommended_packing_items'])} items):\n{items}\n"
                f"  Saved Record: Trip ID {saved['trip_id']} (Priority: {saved['priority'].upper()})."
            )

        return json.dumps(parsed, indent=2)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> ModelResponse:
        tool_output_item = self._extract_tool_output(input)
        if tool_output_item is not None:
            raw_output = str(tool_output_item.get("output", ""))
            final_text = self._synthesize_final_answer(raw_output)
            msg = ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex[:10]}",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=final_text, annotations=[])],
            )
            return ModelResponse(
                output=[msg],
                usage=Usage(requests=1, input_tokens=120, output_tokens=160, total_tokens=280),
                response_id=f"resp_{uuid.uuid4().hex[:10]}",
            )

        user_query = self._extract_last_user_text(input)
        tool_name, tool_args = self._route_query_to_tool_call(user_query)
        call_id = f"call_{uuid.uuid4().hex[:10]}"
        tool_call = ResponseFunctionToolCall(
            id=f"fc_{uuid.uuid4().hex[:10]}",
            call_id=call_id,
            type="function_call",
            name=tool_name,
            arguments=json.dumps(tool_args),
            status="completed",
        )
        return ModelResponse(
            output=[tool_call],
            usage=Usage(requests=1, input_tokens=95, output_tokens=45, total_tokens=140),
            response_id=f"resp_{uuid.uuid4().hex[:10]}",
        )

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError("Streaming not required for standard Runner.run execution.")
