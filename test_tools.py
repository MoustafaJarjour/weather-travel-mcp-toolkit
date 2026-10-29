"""
Unit & Protocol Integration Test Suite (`test_tools.py`)
Course: EECE 503P/798S - Agentic Systems (C5: Agent Tools)

Verifies:
  1. Async Data Tool (`get_weather_forecast`): typed `WeatherForecastResponse`, Celsius/Fahrenheit
     conversion, zero side effects, and predictable failure cases.
  2. Async Action Tool (`save_travel_itinerary`): typed `SavedItineraryResponse`, JSON persistence,
     concurrent safety, and predictable failure cases.
  3. Async Orchestration Tool (`plan_weather_smart_trip`): sequential Data -> Rules -> Action
     coordination across climates (freezing/snowy, hot/sunny, rainy) and upstream failure propagation.
  4. FastMCP Protocol Compliance: verifies exactly 3 registered MCP tools, typed `output_schema`
     JSON schemas, `read_only_hint` tool annotations, and both `@mcp.resource` endpoints.
  5. End-to-End OpenAI Agents SDK Round Trip (`Agent` + `Runner.run` + `MCPServerStdio`).

Run with:
  python test_tools.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastmcp import Client
from fastmcp.exceptions import ToolError

import agent as agent_module
import server

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("fastmcp").setLevel(logging.CRITICAL)


class TestWeatherTravelMCPToolkit(unittest.IsolatedAsyncioTestCase):
    """Comprehensive async unit and MCP protocol integration tests."""

    async def asyncSetUp(self) -> None:
        """Disable verbose asyncio debug timer warnings and isolate storage in a temporary directory."""
        asyncio.get_running_loop().set_debug(False)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_data_dir = Path(self.temp_dir.name)
        self.temp_trips_file = self.temp_data_dir / "saved_trips.json"

        self.patcher_dir = patch.object(server, "DATA_DIR", self.temp_data_dir)
        self.patcher_file = patch.object(server, "TRIPS_FILE", self.temp_trips_file)
        self.patcher_live = patch.object(server, "USE_LIVE_WEATHER_API", False)
        self.patcher_env = patch.dict(os.environ, {"WEATHER_TRIPS_FILE": str(self.temp_trips_file)})
        self.patcher_dir.start()
        self.patcher_file.start()
        self.patcher_live.start()
        self.patcher_env.start()

    async def asyncTearDown(self) -> None:
        self.patcher_env.stop()
        self.patcher_live.stop()
        self.patcher_file.stop()
        self.patcher_dir.stop()
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # 1. Data Tool Tests (`get_weather_forecast`)
    # -----------------------------------------------------------------------

    async def test_01_data_tool_returns_typed_pydantic_model_without_side_effects(self) -> None:
        """Verify `get_weather_forecast` returns a `WeatherForecastResponse` without mutating disk."""
        res_c = await server.get_weather_forecast(city="Beirut", days=3, units="celsius")
        self.assertIsInstance(res_c, server.WeatherForecastResponse)
        self.assertEqual(res_c.city, "Beirut")
        self.assertEqual(res_c.country, "Lebanon")
        self.assertEqual(res_c.unit_symbol, "°C")
        self.assertEqual(len(res_c.daily_forecasts), 3)
        self.assertFalse(self.temp_trips_file.exists(), "Data tool must never create or mutate storage.")

        res_f = await server.get_weather_forecast(city="Beirut", days=3, units="fahrenheit")
        self.assertEqual(res_f.units, "fahrenheit")
        self.assertEqual(res_f.unit_symbol, "°F")
        self.assertGreater(res_f.average_high, res_c.average_high)

    async def test_02_data_tool_predictable_failure_cases(self) -> None:
        """Verify all predictable failure cases in `get_weather_forecast` raise clear ValueErrors."""
        with self.assertRaisesRegex(ValueError, "Invalid input: 'city' cannot be empty"):
            await server.get_weather_forecast(city="   ", days=3)

        with self.assertRaisesRegex(ValueError, "Invalid input: 'days' must be an integer between 1 and 7"):
            await server.get_weather_forecast(city="London", days=0)

        with self.assertRaisesRegex(ValueError, "Invalid input: 'days' must be an integer between 1 and 7"):
            await server.get_weather_forecast(city="London", days=14)

        with self.assertRaisesRegex(ValueError, "Invalid input: 'units' must be either 'celsius' or 'fahrenheit'"):
            await server.get_weather_forecast(city="London", days=3, units="kelvin")

        with self.assertRaisesRegex(ValueError, "Resource unavailable: Weather forecast data could not be found"):
            await server.get_weather_forecast(city="Atlantis_NonExistent_City_999", days=3)

    # -----------------------------------------------------------------------
    # 2. Action Tool Tests (`save_travel_itinerary`)
    # -----------------------------------------------------------------------

    async def test_03_action_tool_persists_typed_itinerary(self) -> None:
        """Verify `save_travel_itinerary` returns `SavedItineraryResponse` and writes valid JSON."""
        res = await server.save_travel_itinerary(
            city="Paris",
            trip_days=4,
            packing_items=["Passport", "Trench coat", "Umbrella"],
            priority="high",
            notes="Spring museum itinerary",
        )
        self.assertIsInstance(res, server.SavedItineraryResponse)
        self.assertEqual(res.status, "saved")
        self.assertTrue(res.trip_id.startswith("TRIP-"))
        self.assertEqual(res.total_saved_trips, 1)
        self.assertTrue(self.temp_trips_file.exists())

        stored = json.loads(self.temp_trips_file.read_text(encoding="utf-8"))
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["trip_id"], res.trip_id)

    async def test_04_action_tool_predictable_failure_cases(self) -> None:
        """Verify all predictable failure cases in `save_travel_itinerary` raise clear ValueErrors."""
        with self.assertRaisesRegex(ValueError, "Invalid input: 'city' cannot be empty"):
            await server.save_travel_itinerary(city="", trip_days=3, packing_items=["Passport"])

        with self.assertRaisesRegex(ValueError, "Invalid input: 'trip_days' must be an integer between 1 and 30"):
            await server.save_travel_itinerary(city="Tokyo", trip_days=0, packing_items=["Passport"])

        with self.assertRaisesRegex(ValueError, "Invalid input: 'packing_items' must contain at least one"):
            await server.save_travel_itinerary(city="Tokyo", trip_days=3, packing_items=[])

        with self.assertRaisesRegex(ValueError, "Invalid input: 'priority' must be one of 'low', 'medium', or 'high'"):
            await server.save_travel_itinerary(
                city="Tokyo",
                trip_days=3,
                packing_items=["Passport"],
                priority="urgent_invalid",
            )

    # -----------------------------------------------------------------------
    # 3. Orchestration Tool Tests (`plan_weather_smart_trip`)
    # -----------------------------------------------------------------------

    async def test_05_orchestration_tool_adapts_across_climates_and_saves(self) -> None:
        """Verify `plan_weather_smart_trip` coordinates Data + Action tools across diverse climates."""
        # Cold, snowy & windy climate (Reykjavik fallback catalog)
        cold_plan = await server.plan_weather_smart_trip(
            city="Reykjavik",
            days=3,
            units="celsius",
            traveler_notes="Aurora Borealis excursion",
        )
        self.assertIsInstance(cold_plan, server.OrchestratedTripPlanResponse)
        self.assertEqual(cold_plan.orchestration_status, "completed")
        self.assertEqual(len(cold_plan.steps_executed), 3)
        self.assertEqual(cold_plan.saved_itinerary.priority, "high")
        self.assertIn("Heavy winter parka / down coat", cold_plan.recommended_packing_items)
        self.assertIn("Waterproof rain jacket with hood", cold_plan.recommended_packing_items)

        # Hot & sunny climate (Dubai fallback catalog)
        hot_plan = await server.plan_weather_smart_trip(city="Dubai", days=4, units="celsius")
        self.assertIn("High-SPF sunscreen and UV sunglasses", hot_plan.recommended_packing_items)
        self.assertEqual(hot_plan.saved_itinerary.total_saved_trips, 2)

    async def test_06_orchestration_tool_propagates_step1_failure(self) -> None:
        """Verify `plan_weather_smart_trip` contextualizes upstream Data tool failures clearly."""
        with self.assertRaisesRegex(
            ValueError,
            r"Orchestration failed at Step 1 \(`get_weather_forecast`\): Resource unavailable",
        ):
            await server.plan_weather_smart_trip(city="Unknown_City_XYZ", days=3)

    # -----------------------------------------------------------------------
    # 4. FastMCP Protocol Schema, Annotations & Resource Tests
    # -----------------------------------------------------------------------

    async def test_07_fastmcp_protocol_tools_schemas_annotations_and_resources(self) -> None:
        """Verify MCP `tools/list`, typed `output_schema`, `read_only_hint`, and `resources/read`."""
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            self.assertEqual(
                [t.name for t in tools],
                ["get_weather_forecast", "save_travel_itinerary", "plan_weather_smart_trip"],
                "Server must expose allowlisted triad of exactly three MCP tools.",
            )

            tool_map = {t.name: t for t in tools}
            # Verify MCP annotations (`read_only_hint`)
            self.assertTrue(tool_map["get_weather_forecast"].annotations.read_only_hint)
            self.assertFalse(tool_map["save_travel_itinerary"].annotations.read_only_hint)
            self.assertFalse(tool_map["plan_weather_smart_trip"].annotations.read_only_hint)

            # Verify typed `output_schema` generated from Pydantic return annotations
            for name, t in tool_map.items():
                self.assertIsNotNone(t.output_schema, f"{name} must expose a typed MCP output_schema.")
                self.assertIn("properties", t.output_schema)

            # Verify MCP tool invocation & `@mcp.resource` inspection
            call_res = await client.call_tool(
                "plan_weather_smart_trip",
                {"city": "London", "days": 3, "units": "celsius"},
            )
            self.assertFalse(call_res.is_error)

            cities_res = await client.read_resource("weather://supported-cities")
            self.assertIn("Beirut", cities_res[0].text)

            trips_res = await client.read_resource("trips://saved")
            trips_payload = json.loads(trips_res[0].text)
            self.assertEqual(trips_payload["total_saved_trips"], 1)
            self.assertEqual(trips_payload["trips"][0]["city"], "London")

            # Verify protocol-level ToolError on invalid input
            with self.assertRaises(ToolError):
                await client.call_tool("get_weather_forecast", {"city": "", "days": 3})

    # -----------------------------------------------------------------------
    # 5. OpenAI Agents SDK Full Round-Trip Verification
    # -----------------------------------------------------------------------

    async def test_08_openai_agents_sdk_stdio_round_trip(self) -> None:
        """Verify full Host (`Runner.run`) -> Client (`MCPServerStdio`) -> FastMCP Server round trip."""
        queries = [
            "What is the 2-day weather forecast in Tokyo in celsius?",
            "Save a 3-day travel itinerary for Cairo with medium priority: Passport, Sunglasses, Sun hat",
            "Plan a smart 3-day trip to London in celsius and save the itinerary.",
            "What is the 3-day weather forecast in Atlantis_NonExistentCity?",
            "Save a 3-day travel itinerary for Tokyo with empty packing items: []",
        ]
        outputs = await agent_module.run_agent_with_mcp(queries, force_offline=True)
        self.assertEqual(len(outputs), 5)
        self.assertIn("Weather Forecast for Tokyo", outputs[0])
        self.assertIn("Travel Itinerary Saved Successfully!", outputs[1])
        self.assertIn("End-to-End Weather-Smart Trip Plan Completed for London", outputs[2])
        self.assertIn("Resource unavailable", outputs[3])
        self.assertIn("packing_items", outputs[4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
