"""
FastMCP Server: Weather & Travel Planning Assistant Toolkit (`server.py`)
Course: EECE 503P/798S - Agentic Systems (C5: Agent Tools)

Implements an asynchronous MCP server exposing:
  1. Data Tool (`get_weather_forecast`):
     Pure read-only async lookup of multi-day meteorological forecasts via the
     Open-Meteo Geocoding & Weather Forecast REST APIs (with deterministic offline fallback).
  2. Action Tool (`save_travel_itinerary`):
     Single-purpose state-mutating tool that validates and persists a structured
     packing checklist and travel itinerary to `data/saved_trips.json`.
  3. Orchestration Tool (`plan_weather_smart_trip`):
     Coordinates `get_weather_forecast` and `save_travel_itinerary` in sequence to
     derive weather-adapted packing essentials and advisories and save the resulting trip.

Also exposes two read-only MCP Resources via `@mcp.resource`:
  - `weather://supported-cities`
  - `trips://saved`
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Annotated, Any
import uuid

from fastmcp import FastMCP
import httpx
from pydantic import BaseModel, Field

# Ensure clean UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Initialize the FastMCP Server
mcp = FastMCP(
    name="WeatherTravelAssistant",
    instructions=(
        "A Weather & Travel Planning MCP server exposing three single-purpose tools: "
        "1) get_weather_forecast (read-only Data tool for live city weather forecasts), "
        "2) save_travel_itinerary (state-mutating Action tool that saves a packing checklist to JSON storage), "
        "3) plan_weather_smart_trip (Orchestration tool that fetches the forecast, synthesizes a "
        "weather-tailored packing plan, and persists the itinerary)."
    ),
)

# Persistent storage paths for the Action tool (protected by an asyncio lock for concurrent safety)
_ENV_TRIPS_FILE = os.getenv("WEATHER_TRIPS_FILE", "").strip()
TRIPS_FILE = Path(_ENV_TRIPS_FILE) if _ENV_TRIPS_FILE else (Path(__file__).resolve().parent / "data" / "saved_trips.json")
DATA_DIR = TRIPS_FILE.parent
_STORAGE_LOCK = asyncio.Lock()

# Toggleable flag allowing deterministic unit tests without network dependency
USE_LIVE_WEATHER_API = True


# ---------------------------------------------------------------------------
# Strongly Typed Pydantic Output Models (Auto-generates MCP `outputSchema`)
# ---------------------------------------------------------------------------

class DailyForecast(BaseModel):
    """Meteorological forecast metrics for a single calendar day."""
    date: str = Field(description="ISO-8601 forecast date (YYYY-MM-DD).")
    condition: str = Field(description="Human-readable weather condition (e.g., Sunny, Light Rain, Heavy Snow).")
    temp_high: float = Field(description="Daily maximum temperature in the requested unit.")
    temp_low: float = Field(description="Daily minimum temperature in the requested unit.")
    precipitation_chance_pct: int = Field(description="Maximum probability of precipitation (0 to 100%).")
    wind_speed_kmh: float = Field(description="Maximum expected wind speed at 10m in km/h.")


class WeatherForecastResponse(BaseModel):
    """Typed output schema returned by the Data tool (`get_weather_forecast`)."""
    city: str = Field(description="Resolved name of the queried destination city.")
    country: str = Field(description="Country or territory of the resolved city.")
    latitude: float = Field(description="Geographic latitude in decimal degrees.")
    longitude: float = Field(description="Geographic longitude in decimal degrees.")
    units: str = Field(description="Normalized temperature unit ('celsius' or 'fahrenheit').")
    unit_symbol: str = Field(description="Display unit symbol ('°C' or '°F').")
    days_requested: int = Field(description="Number of forecast days returned (1 to 7).")
    summary_condition: str = Field(description="Concise meteorological summary across the forecast period.")
    average_high: float = Field(description="Mean daily high temperature across the forecast window.")
    average_low: float = Field(description="Mean daily low temperature across the forecast window.")
    max_precipitation_chance_pct: int = Field(description="Peak precipitation probability across the window.")
    max_wind_speed_kmh: float = Field(description="Peak wind speed in km/h across the forecast window.")
    source: str = Field(description="Data provider ('open_meteo_live_api' or 'offline_fallback_catalog').")
    daily_forecasts: list[DailyForecast] = Field(description="Ordered day-by-day forecast breakdown.")


class SavedItineraryResponse(BaseModel):
    """Typed output schema returned by the Action tool (`save_travel_itinerary`)."""
    status: str = Field(description="Persistence confirmation status ('saved').")
    trip_id: str = Field(description="Unique identifier assigned to the saved itinerary (e.g., 'TRIP-A1B2C3D4').")
    city: str = Field(description="Normalized destination city name.")
    trip_days: int = Field(description="Trip duration in days (1 to 30).")
    priority: str = Field(description="Normalized preparation priority ('low', 'medium', or 'high').")
    packing_items: list[str] = Field(description="Validated list of non-empty packing items persisted.")
    notes: str = Field(description="Optional advisory or traveler notes persisted with the record.")
    saved_at_utc: str = Field(description="ISO-8601 UTC timestamp recording when the itinerary was saved.")
    storage_file: str = Field(description="Filesystem path to the persistent JSON store.")
    total_saved_trips: int = Field(description="Total count of itineraries currently stored on disk.")


class WeatherOverviewSummary(BaseModel):
    """Compact weather snapshot embedded within the Orchestration tool output."""
    summary: str = Field(description="Overall weather summary text.")
    average_high: str = Field(description="Formatted average high temperature with unit.")
    average_low: str = Field(description="Formatted average low temperature with unit.")
    max_precipitation_chance_pct: int = Field(description="Peak precipitation probability (0 to 100%).")
    max_wind_speed_kmh: float = Field(description="Peak wind speed in km/h.")
    source: str = Field(description="Underlying weather data source used.")
    daily_forecasts: list[DailyForecast] = Field(description="Day-by-day forecast breakdown.")


class OrchestratedTripPlanResponse(BaseModel):
    """Typed output schema returned by the Orchestration tool (`plan_weather_smart_trip`)."""
    orchestration_status: str = Field(description="Execution status of the workflow ('completed').")
    steps_executed: list[str] = Field(description="Sequential audit trace of internal tool calls and rules.")
    destination: str = Field(description="Formatted 'City, Country' destination string.")
    weather_overview: WeatherOverviewSummary = Field(description="Forecast data retrieved in Step 1.")
    weather_advisory: str = Field(description="Actionable travel advisory synthesized in Step 2.")
    recommended_packing_items: list[str] = Field(description="Weather-adapted packing checklist generated in Step 2.")
    saved_itinerary: SavedItineraryResponse = Field(description="Persistence record returned by Step 3.")


# ---------------------------------------------------------------------------
# WMO Weather Code Mapping & Resilient Offline Fallback Catalog
# ---------------------------------------------------------------------------

WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear Sky",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing Rime Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    56: "Freezing Drizzle",
    57: "Dense Freezing Drizzle",
    61: "Light Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    66: "Freezing Rain",
    67: "Heavy Freezing Rain",
    71: "Light Snow",
    73: "Moderate Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Light Rain Showers",
    81: "Moderate Rain Showers",
    82: "Heavy Rain Showers",
    85: "Light Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorms",
    96: "Thunderstorms with Hail",
    99: "Severe Thunderstorms with Hail",
}

OFFLINE_FALLBACK_CATALOG: dict[str, dict[str, Any]] = {
    "beirut": {
        "city": "Beirut", "country": "Lebanon", "lat": 33.8938, "lon": 35.5018,
        "days": [
            {"condition": "Sunny", "high_c": 26.0, "low_c": 19.0, "precip": 10, "wind": 14.0},
            {"condition": "Partly Cloudy", "high_c": 25.0, "low_c": 18.5, "precip": 15, "wind": 16.0},
            {"condition": "Sunny", "high_c": 27.0, "low_c": 19.5, "precip": 5, "wind": 12.0},
            {"condition": "Light Rain", "high_c": 22.0, "low_c": 17.0, "precip": 60, "wind": 24.0},
            {"condition": "Partly Cloudy", "high_c": 24.0, "low_c": 18.0, "precip": 20, "wind": 15.0},
            {"condition": "Sunny", "high_c": 26.5, "low_c": 19.0, "precip": 10, "wind": 13.0},
            {"condition": "Sunny", "high_c": 27.0, "low_c": 20.0, "precip": 5, "wind": 11.0},
        ],
    },
    "london": {
        "city": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278,
        "days": [
            {"condition": "Heavy Rain", "high_c": 13.0, "low_c": 7.0, "precip": 85, "wind": 28.0},
            {"condition": "Light Rain", "high_c": 14.0, "low_c": 8.0, "precip": 70, "wind": 22.0},
            {"condition": "Overcast", "high_c": 15.0, "low_c": 8.5, "precip": 45, "wind": 18.0},
            {"condition": "Light Rain", "high_c": 12.5, "low_c": 6.5, "precip": 75, "wind": 25.0},
            {"condition": "Partly Cloudy", "high_c": 16.0, "low_c": 9.0, "precip": 30, "wind": 15.0},
            {"condition": "Moderate Rain Showers", "high_c": 13.5, "low_c": 7.5, "precip": 65, "wind": 20.0},
            {"condition": "Overcast", "high_c": 14.0, "low_c": 8.0, "precip": 40, "wind": 17.0},
        ],
    },
    "paris": {
        "city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522,
        "days": [
            {"condition": "Partly Cloudy", "high_c": 16.0, "low_c": 8.0, "precip": 25, "wind": 15.0},
            {"condition": "Light Rain", "high_c": 15.0, "low_c": 7.5, "precip": 65, "wind": 21.0},
            {"condition": "Sunny", "high_c": 18.0, "low_c": 9.0, "precip": 10, "wind": 12.0},
            {"condition": "Partly Cloudy", "high_c": 17.0, "low_c": 8.5, "precip": 20, "wind": 14.0},
            {"condition": "Sunny", "high_c": 19.0, "low_c": 10.0, "precip": 10, "wind": 11.0},
            {"condition": "Overcast", "high_c": 15.0, "low_c": 7.0, "precip": 35, "wind": 16.0},
            {"condition": "Partly Cloudy", "high_c": 16.5, "low_c": 8.0, "precip": 20, "wind": 13.0},
        ],
    },
    "tokyo": {
        "city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503,
        "days": [
            {"condition": "Clear Sky", "high_c": 21.0, "low_c": 13.0, "precip": 10, "wind": 14.0},
            {"condition": "Sunny", "high_c": 22.5, "low_c": 14.0, "precip": 5, "wind": 12.0},
            {"condition": "Light Rain Showers", "high_c": 18.0, "low_c": 12.0, "precip": 70, "wind": 26.0},
            {"condition": "Partly Cloudy", "high_c": 20.0, "low_c": 13.0, "precip": 20, "wind": 15.0},
            {"condition": "Sunny", "high_c": 23.0, "low_c": 14.5, "precip": 10, "wind": 11.0},
            {"condition": "Clear Sky", "high_c": 22.0, "low_c": 13.5, "precip": 10, "wind": 13.0},
            {"condition": "Partly Cloudy", "high_c": 19.5, "low_c": 12.5, "precip": 25, "wind": 16.0},
        ],
    },
    "reykjavik": {
        "city": "Reykjavik", "country": "Iceland", "lat": 64.1466, "lon": -21.9426,
        "days": [
            {"condition": "Light Snow Showers", "high_c": 1.0, "low_c": -4.0, "precip": 85, "wind": 38.0},
            {"condition": "Freezing Drizzle", "high_c": 0.0, "low_c": -6.0, "precip": 60, "wind": 42.0},
            {"condition": "Heavy Snow", "high_c": -1.0, "low_c": -7.0, "precip": 90, "wind": 35.0},
            {"condition": "Partly Cloudy", "high_c": 2.0, "low_c": -3.0, "precip": 25, "wind": 24.0},
            {"condition": "Light Snow Showers", "high_c": 0.5, "low_c": -5.0, "precip": 75, "wind": 32.0},
            {"condition": "Overcast", "high_c": 1.5, "low_c": -3.5, "precip": 40, "wind": 28.0},
            {"condition": "Light Snow", "high_c": 0.0, "low_c": -5.5, "precip": 65, "wind": 30.0},
        ],
    },
    "dubai": {
        "city": "Dubai", "country": "United Arab Emirates", "lat": 25.2048, "lon": 55.2708,
        "days": [
            {"condition": "Clear Sky", "high_c": 36.0, "low_c": 26.0, "precip": 0, "wind": 16.0},
            {"condition": "Clear Sky", "high_c": 37.5, "low_c": 27.0, "precip": 0, "wind": 18.0},
            {"condition": "Mainly Clear", "high_c": 35.5, "low_c": 25.5, "precip": 5, "wind": 14.0},
            {"condition": "Clear Sky", "high_c": 38.0, "low_c": 27.5, "precip": 0, "wind": 20.0},
            {"condition": "Clear Sky", "high_c": 36.0, "low_c": 26.0, "precip": 0, "wind": 15.0},
            {"condition": "Clear Sky", "high_c": 37.0, "low_c": 26.5, "precip": 0, "wind": 17.0},
            {"condition": "Mainly Clear", "high_c": 35.0, "low_c": 25.0, "precip": 5, "wind": 13.0},
        ],
    },
    "new york": {
        "city": "New York", "country": "United States", "lat": 40.7128, "lon": -74.0060,
        "days": [
            {"condition": "Partly Cloudy", "high_c": 17.0, "low_c": 9.0, "precip": 20, "wind": 19.0},
            {"condition": "Clear Sky", "high_c": 19.5, "low_c": 11.0, "precip": 10, "wind": 14.0},
            {"condition": "Thunderstorms", "high_c": 16.0, "low_c": 8.5, "precip": 80, "wind": 31.0},
            {"condition": "Clear Sky", "high_c": 18.0, "low_c": 10.0, "precip": 10, "wind": 16.0},
            {"condition": "Partly Cloudy", "high_c": 17.5, "low_c": 9.5, "precip": 25, "wind": 18.0},
            {"condition": "Mainly Clear", "high_c": 20.0, "low_c": 11.5, "precip": 5, "wind": 12.0},
            {"condition": "Light Rain", "high_c": 15.0, "low_c": 8.0, "precip": 60, "wind": 22.0},
        ],
    },
    "cairo": {
        "city": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357,
        "days": [
            {"condition": "Clear Sky", "high_c": 31.0, "low_c": 20.0, "precip": 0, "wind": 18.0},
            {"condition": "Clear Sky", "high_c": 33.0, "low_c": 21.0, "precip": 0, "wind": 20.0},
            {"condition": "Mainly Clear", "high_c": 30.5, "low_c": 19.5, "precip": 5, "wind": 16.0},
            {"condition": "Clear Sky", "high_c": 32.0, "low_c": 20.5, "precip": 0, "wind": 15.0},
            {"condition": "Clear Sky", "high_c": 31.5, "low_c": 20.0, "precip": 0, "wind": 17.0},
            {"condition": "Clear Sky", "high_c": 34.0, "low_c": 22.0, "precip": 0, "wind": 19.0},
            {"condition": "Mainly Clear", "high_c": 30.0, "low_c": 19.0, "precip": 5, "wind": 14.0},
        ],
    },
}


def _convert_temp(celsius_val: float, units: str) -> float:
    """Converts Celsius temperature to requested unit rounded to 1 decimal place."""
    if units == "fahrenheit":
        return round((celsius_val * 9.0 / 5.0) + 32.0, 1)
    return round(celsius_val, 1)


async def _fetch_open_meteo_live_async(
    city_query: str,
    days: int,
    units: str,
) -> tuple[WeatherForecastResponse | None, bool]:
    """
    Queries the Open-Meteo Geocoding & Weather Forecast REST APIs asynchronously.

    Returns:
        (response, network_ok):
          - (WeatherForecastResponse, True) when city is found and forecast retrieved.
          - (None, True) when the REST API responded cleanly that no such city exists.
          - (None, False) when offline or a network/HTTP transport timeout occurred.
    """
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            geo_resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": city_query,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

            results = geo_data.get("results")
            if not results:
                return None, True

            top = results[0]
            lat = round(float(top["latitude"]), 4)
            lon = round(float(top["longitude"]), 4)
            resolved_city = str(top.get("name", city_query.title()))
            country = str(top.get("country", top.get("country_code", "Unknown")))

            wx_resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": (
                        "weathercode,temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max,windspeed_10m_max"
                    ),
                    "temperature_unit": "fahrenheit" if units == "fahrenheit" else "celsius",
                    "windspeed_unit": "kmh",
                    "forecast_days": days,
                    "timezone": "auto",
                },
            )
            wx_resp.raise_for_status()
            wx_data = wx_resp.json()

        daily = wx_data.get("daily", {})
        dates = daily.get("time", [])[:days]
        highs = daily.get("temperature_2m_max", [])[:days]
        lows = daily.get("temperature_2m_min", [])[:days]
        precips = daily.get("precipitation_probability_max", [])[:days]
        winds = daily.get("windspeed_10m_max", [])[:days]
        codes = daily.get("weathercode", [])[:days]

        if not dates:
            return None, False

        daily_forecasts: list[DailyForecast] = []
        for idx, dt in enumerate(dates):
            wmo_code = int(codes[idx]) if idx < len(codes) and codes[idx] is not None else 2
            cond = WMO_WEATHER_CODES.get(wmo_code, "Partly Cloudy")
            daily_forecasts.append(
                DailyForecast(
                    date=str(dt),
                    condition=cond,
                    temp_high=round(float(highs[idx]), 1),
                    temp_low=round(float(lows[idx]), 1),
                    precipitation_chance_pct=int(precips[idx] or 0) if idx < len(precips) else 0,
                    wind_speed_kmh=round(float(winds[idx] or 10.0), 1) if idx < len(winds) else 10.0,
                )
            )

        avg_high = round(sum(d.temp_high for d in daily_forecasts) / len(daily_forecasts), 1)
        avg_low = round(sum(d.temp_low for d in daily_forecasts) / len(daily_forecasts), 1)
        max_precip = max(d.precipitation_chance_pct for d in daily_forecasts)
        max_wind = round(max(d.wind_speed_kmh for d in daily_forecasts), 1)
        unit_symbol = "°F" if units == "fahrenheit" else "°C"

        return (
            WeatherForecastResponse(
                city=resolved_city,
                country=country,
                latitude=lat,
                longitude=lon,
                units=units,
                unit_symbol=unit_symbol,
                days_requested=days,
                summary_condition=(
                    f"{daily_forecasts[0].condition}; avg high {avg_high}{unit_symbol}, "
                    f"avg low {avg_low}{unit_symbol}, peak precip {max_precip}%, "
                    f"max wind {max_wind} km/h."
                ),
                average_high=avg_high,
                average_low=avg_low,
                max_precipitation_chance_pct=max_precip,
                max_wind_speed_kmh=max_wind,
                daily_forecasts=daily_forecasts,
                source="open_meteo_live_api",
            ),
            True,
        )
    except (httpx.HTTPError, KeyError, ValueError, OSError):
        return None, False


# ===========================================================================
# TOOL 1: DATA TOOL (Read-Only Async Lookup, Zero Side Effects)
# ===========================================================================

@mcp.tool(
    name="get_weather_forecast",
    title="Get Weather Forecast (Data Tool)",
    description=(
        "[DATA TOOL] Retrieves a multi-day meteorological forecast for a destination city "
        "without modifying any system state. Queries the live Open-Meteo Geocoding and Forecast "
        "REST APIs (with offline catalog fallback) and returns daily temperatures, precipitation "
        "probabilities, wind speeds, and overall weather conditions."
    ),
    annotations={
        "title": "Get Weather Forecast",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_weather_forecast(
    city: Annotated[
        str,
        Field(description="Destination city name to query (e.g., 'Beirut', 'Paris', 'Reykjavik', 'Tokyo')."),
    ],
    days: Annotated[
        int,
        Field(default=3, description="Number of forecast days to retrieve (integer from 1 to 7)."),
    ] = 3,
    units: Annotated[
        str,
        Field(default="celsius", description="Temperature scale: 'celsius' or 'fahrenheit'."),
    ] = "celsius",
) -> WeatherForecastResponse:
    """
    Pure Data Tool: Retrieves typed weather forecast information without side effects.

    Predictable Failure Cases Handled:
      - Empty or whitespace-only `city` -> raises ValueError.
      - `days` outside `[1, 7]` -> raises ValueError.
      - `units` not in `('celsius', 'fahrenheit')` -> raises ValueError.
      - Unrecognized / non-existent `city` -> raises ValueError with featured city hints.
    """
    # 1. Validate city input
    if not isinstance(city, str) or not city.strip():
        raise ValueError(
            "Invalid input: 'city' cannot be empty. Provide a valid city name "
            "(for example: 'Beirut', 'Paris', 'London', 'Tokyo', 'Reykjavik', 'Dubai', 'New York')."
        )

    # 2. Validate forecast horizon
    if not isinstance(days, int) or isinstance(days, bool) or days < 1 or days > 7:
        raise ValueError(
            f"Invalid input: 'days' must be an integer between 1 and 7 (received: {days!r})."
        )

    # 3. Validate temperature scale
    normalized_units = units.strip().lower() if isinstance(units, str) else ""
    if normalized_units not in ("celsius", "fahrenheit"):
        raise ValueError(
            f"Invalid input: 'units' must be either 'celsius' or 'fahrenheit' (received: {units!r})."
        )

    cleaned_city = city.strip()
    normalized_key = cleaned_city.lower()
    unit_symbol = "°F" if normalized_units == "fahrenheit" else "°C"

    # 4. Attempt live async Open-Meteo REST lookup first when enabled
    if USE_LIVE_WEATHER_API:
        live_forecast, network_ok = await _fetch_open_meteo_live_async(
            cleaned_city,
            days,
            normalized_units,
        )
        if live_forecast is not None:
            return live_forecast
        # If the live geocoding API explicitly confirmed no matching city exists, fail predictably
        if network_ok and normalized_key not in OFFLINE_FALLBACK_CATALOG:
            supported = ", ".join(sorted(v["city"] for v in OFFLINE_FALLBACK_CATALOG.values()))
            raise ValueError(
                f"Resource unavailable: Weather forecast data could not be found for city '{cleaned_city}'. "
                f"Verify the city spelling or try a featured city ({supported})."
            )

    # 5. Resilient fallback to deterministic local catalog (when offline or in unit tests)
    if normalized_key in OFFLINE_FALLBACK_CATALOG:
        entry = OFFLINE_FALLBACK_CATALOG[normalized_key]
        raw_days = entry["days"][:days]
        today = datetime.now(timezone.utc).date()

        daily_forecasts = [
            DailyForecast(
                date=(today + timedelta(days=idx)).isoformat(),
                condition=day_info["condition"],
                temp_high=_convert_temp(day_info["high_c"], normalized_units),
                temp_low=_convert_temp(day_info["low_c"], normalized_units),
                precipitation_chance_pct=day_info["precip"],
                wind_speed_kmh=day_info["wind"],
            )
            for idx, day_info in enumerate(raw_days)
        ]

        avg_high = round(sum(d.temp_high for d in daily_forecasts) / len(daily_forecasts), 1)
        avg_low = round(sum(d.temp_low for d in daily_forecasts) / len(daily_forecasts), 1)
        max_precip = max(d.precipitation_chance_pct for d in daily_forecasts)
        max_wind = round(max(d.wind_speed_kmh for d in daily_forecasts), 1)

        return WeatherForecastResponse(
            city=entry["city"],
            country=entry["country"],
            latitude=entry["lat"],
            longitude=entry["lon"],
            units=normalized_units,
            unit_symbol=unit_symbol,
            days_requested=days,
            summary_condition=(
                f"{daily_forecasts[0].condition}; avg high {avg_high}{unit_symbol}, "
                f"avg low {avg_low}{unit_symbol}, peak precip {max_precip}%, "
                f"max wind {max_wind} km/h."
            ),
            average_high=avg_high,
            average_low=avg_low,
            max_precipitation_chance_pct=max_precip,
            max_wind_speed_kmh=max_wind,
            daily_forecasts=daily_forecasts,
            source="offline_fallback_catalog",
        )

    # 6. Predictable Failure: City unavailable in both live API and offline catalog
    supported = ", ".join(sorted(v["city"] for v in OFFLINE_FALLBACK_CATALOG.values()))
    raise ValueError(
        f"Resource unavailable: Weather forecast data could not be found for city '{cleaned_city}'. "
        f"Verify the city spelling or try a featured city ({supported})."
    )


# ===========================================================================
# TOOL 2: ACTION TOOL (Single-Purpose State-Mutating Persistence)
# ===========================================================================

def _persist_itinerary_sync(record: dict[str, Any]) -> int:
    """Synchronous file I/O helper executed in a worker thread via `asyncio.to_thread`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_trips: list[dict[str, Any]] = []
    if TRIPS_FILE.exists():
        content = TRIPS_FILE.read_text(encoding="utf-8").strip()
        if content:
            loaded = json.loads(content)
            if isinstance(loaded, list):
                existing_trips = loaded

    existing_trips.append(record)
    TRIPS_FILE.write_text(
        json.dumps(existing_trips, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(existing_trips)


@mcp.tool(
    name="save_travel_itinerary",
    title="Save Travel Itinerary (Action Tool)",
    description=(
        "[ACTION TOOL] Validates and persists a travel packing checklist and itinerary record "
        "for a destination city to local JSON storage (`data/saved_trips.json`). Performs an "
        "external state-changing effect without performing any weather lookups."
    ),
    annotations={
        "title": "Save Travel Itinerary",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def save_travel_itinerary(
    city: Annotated[
        str,
        Field(description="Destination city for the travel itinerary (e.g., 'Paris')."),
    ],
    trip_days: Annotated[
        int,
        Field(description="Duration of the trip in days (integer from 1 to 30)."),
    ],
    packing_items: Annotated[
        list[str],
        Field(description="Non-empty list of packing items to persist in the checklist."),
    ],
    priority: Annotated[
        str,
        Field(default="medium", description="Preparation priority level: 'low', 'medium', or 'high'."),
    ] = "medium",
    notes: Annotated[
        str,
        Field(default="", description="Optional reminder or travel advisory notes to store."),
    ] = "",
) -> SavedItineraryResponse:
    """
    Pure Action Tool: Persists a validated travel itinerary record to `data/saved_trips.json`.

    Predictable Failure Cases Handled:
      - Empty `city` name -> raises ValueError.
      - `trip_days` outside `[1, 30]` -> raises ValueError.
      - Empty `packing_items` list -> raises ValueError.
      - Invalid `priority` not in `('low', 'medium', 'high')` -> raises ValueError.
    """
    if not isinstance(city, str) or not city.strip():
        raise ValueError("Invalid input: 'city' cannot be empty when saving a travel itinerary.")

    if not isinstance(trip_days, int) or isinstance(trip_days, bool) or trip_days < 1 or trip_days > 30:
        raise ValueError(
            f"Invalid input: 'trip_days' must be an integer between 1 and 30 (received: {trip_days!r})."
        )

    if not isinstance(packing_items, list):
        raise ValueError("Invalid input: 'packing_items' must be a list of strings.")
    cleaned_items = [str(item).strip() for item in packing_items if str(item).strip()]
    if not cleaned_items:
        raise ValueError(
            "Invalid input: 'packing_items' must contain at least one non-empty item to save."
        )

    normalized_priority = priority.strip().lower() if isinstance(priority, str) else ""
    if normalized_priority not in ("low", "medium", "high"):
        raise ValueError(
            f"Invalid input: 'priority' must be one of 'low', 'medium', or 'high' (received: {priority!r})."
        )

    trip_id = f"TRIP-{uuid.uuid4().hex[:8].upper()}"
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    normalized_city = city.strip().title()
    cleaned_notes = notes.strip() if isinstance(notes, str) else ""

    new_record = {
        "trip_id": trip_id,
        "city": normalized_city,
        "trip_days": trip_days,
        "priority": normalized_priority,
        "packing_items": cleaned_items,
        "notes": cleaned_notes,
        "saved_at_utc": timestamp_utc,
    }

    try:
        async with _STORAGE_LOCK:
            total_saved = await asyncio.to_thread(_persist_itinerary_sync, new_record)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Storage error: Could not persist itinerary to {TRIPS_FILE}: {exc}") from exc

    return SavedItineraryResponse(
        status="saved",
        trip_id=trip_id,
        city=normalized_city,
        trip_days=trip_days,
        priority=normalized_priority,
        packing_items=cleaned_items,
        notes=cleaned_notes,
        saved_at_utc=timestamp_utc,
        storage_file=str(TRIPS_FILE),
        total_saved_trips=total_saved,
    )


# ===========================================================================
# TOOL 3: ORCHESTRATION TOOL (Coordinates Data Tool + Action Tool in Sequence)
# ===========================================================================

@mcp.tool(
    name="plan_weather_smart_trip",
    title="Plan Weather-Smart Trip (Orchestration Tool)",
    description=(
        "[ORCHESTRATION TOOL] Coordinates end-to-end trip preparation by: "
        "1) calling `get_weather_forecast` (Data tool) to inspect destination weather conditions, "
        "2) applying meteorological rules to synthesize a tailored packing checklist, priority level, "
        "and travel advisory, and 3) calling `save_travel_itinerary` (Action tool) to persist the plan."
    ),
    annotations={
        "title": "Plan Weather-Smart Trip",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def plan_weather_smart_trip(
    city: Annotated[
        str,
        Field(description="Destination city to plan the trip for (e.g., 'Reykjavik', 'London', 'Beirut')."),
    ],
    days: Annotated[
        int,
        Field(default=3, description="Number of trip days to plan for (integer from 1 to 7)."),
    ] = 3,
    units: Annotated[
        str,
        Field(default="celsius", description="Preferred temperature scale: 'celsius' or 'fahrenheit'."),
    ] = "celsius",
    traveler_notes: Annotated[
        str,
        Field(default="", description="Optional custom note from the traveler (e.g., 'Academic conference')."),
    ] = "",
) -> OrchestratedTripPlanResponse:
    """
    Orchestration Tool: Coordinates `get_weather_forecast` and `save_travel_itinerary` asynchronously.

    Predictable Failure Cases Handled:
      - Propagates any validation or lookup failure from Step 1 (`get_weather_forecast`) or
        Step 3 (`save_travel_itinerary`) with explicit orchestration stage context.
    """
    steps_executed: list[str] = []

    # Step 1: Invoke the Data tool (`get_weather_forecast`)
    try:
        forecast = await get_weather_forecast(city=city, days=days, units=units)
        steps_executed.append(
            f"1. Called Data tool `get_weather_forecast(city={city!r}, days={days}, units={units!r})` "
            f"[source={forecast.source}]"
        )
    except ValueError as exc:
        raise ValueError(
            f"Orchestration failed at Step 1 (`get_weather_forecast`): {exc}"
        ) from exc

    # Normalize temperatures to Celsius internally for uniform meteorological rule evaluation
    if forecast.units == "fahrenheit":
        avg_high_c = (forecast.average_high - 32.0) * 5.0 / 9.0
        avg_low_c = (forecast.average_low - 32.0) * 5.0 / 9.0
    else:
        avg_high_c = forecast.average_high
        avg_low_c = forecast.average_low

    max_precip = forecast.max_precipitation_chance_pct
    max_wind = forecast.max_wind_speed_kmh
    unit_sym = forecast.unit_symbol

    # Step 2: Synthesize weather-adapted packing essentials, priority, and advisory
    packing_items: list[str] = [
        f"Passport / ID and travel documents for {forecast.city}",
        f"Daily outfits for {days} day(s)",
        "Universal power adapter & phone charger",
    ]
    advisories: list[str] = []
    priority = "low"

    # Temperature-tier rules
    if avg_low_c <= 2.0 or avg_high_c <= 5.0:
        packing_items.extend([
            "Insulated thermal base layers",
            "Heavy winter parka / down coat",
            "Insulated waterproof boots",
            "Thermal gloves, wool beanie, and scarf",
        ])
        advisories.append(
            f"Cold/freezing conditions expected (avg low {forecast.average_low}{unit_sym}); pack thermal layers."
        )
        priority = "high"
    elif avg_low_c <= 12.0 or avg_high_c <= 18.0:
        packing_items.extend([
            "Medium-weight jacket or trench coat",
            "Layered sweaters / fleece",
            "Closed-toe walking shoes",
        ])
        advisories.append(
            f"Cool to mild temperatures ({forecast.average_low}{unit_sym} to {forecast.average_high}{unit_sym}); "
            "dress in versatile layers."
        )
        if priority == "low":
            priority = "medium"
    elif avg_high_c >= 30.0:
        packing_items.extend([
            "Breathable linen/cotton clothing",
            "High-SPF sunscreen and UV sunglasses",
            "Wide-brim sun hat",
            "Insulated reusable water bottle",
        ])
        advisories.append(
            f"Hot conditions expected (avg high {forecast.average_high}{unit_sym}); prioritize hydration and UV protection."
        )
        if priority == "low":
            priority = "medium"
    else:
        packing_items.extend([
            "Light breathable daywear",
            "Light evening cardigan or windbreaker",
            "Sunglasses and sunscreen",
        ])
        advisories.append(
            f"Pleasant weather expected ({forecast.average_low}{unit_sym} to {forecast.average_high}{unit_sym}); "
            "well-suited for outdoor sightseeing."
        )

    # Precipitation rules
    if max_precip >= 60:
        packing_items.extend([
            "Waterproof rain jacket with hood",
            "Compact windproof travel umbrella",
            "Water-resistant footwear",
        ])
        advisories.append(
            f"High precipitation probability (up to {max_precip}%); waterproof outerwear strongly recommended."
        )
        priority = "high" if max_precip >= 75 else ("medium" if priority == "low" else priority)
    elif max_precip >= 30:
        packing_items.append("Compact foldable umbrella")
        advisories.append(f"Moderate chance of showers ({max_precip}%); pack a compact umbrella.")

    # Wind rules
    if max_wind >= 30.0:
        packing_items.append("Windproof outer shell & lip balm")
        advisories.append(f"Strong wind gusts up to {max_wind} km/h forecasted; bring windproof outer gear.")
        priority = "high"

    advisory_text = " ".join(advisories)
    combined_notes = (
        f"{advisory_text} | Traveler Note: {traveler_notes.strip()}"
        if isinstance(traveler_notes, str) and traveler_notes.strip()
        else advisory_text
    )

    steps_executed.append(
        f"2. Synthesized {len(packing_items)} weather-tailored packing items (priority={priority!r})"
    )

    # Step 3: Invoke the Action tool (`save_travel_itinerary`)
    saved_record = await save_travel_itinerary(
        city=forecast.city,
        trip_days=days,
        packing_items=packing_items,
        priority=priority,
        notes=combined_notes,
    )
    steps_executed.append(
        f"3. Called Action tool `save_travel_itinerary` -> persisted trip_id={saved_record.trip_id!r}"
    )

    return OrchestratedTripPlanResponse(
        orchestration_status="completed",
        steps_executed=steps_executed,
        destination=f"{forecast.city}, {forecast.country}",
        weather_overview=WeatherOverviewSummary(
            summary=forecast.summary_condition,
            average_high=f"{forecast.average_high}{unit_sym}",
            average_low=f"{forecast.average_low}{unit_sym}",
            max_precipitation_chance_pct=max_precip,
            max_wind_speed_kmh=max_wind,
            source=forecast.source,
            daily_forecasts=forecast.daily_forecasts,
        ),
        weather_advisory=advisory_text,
        recommended_packing_items=packing_items,
        saved_itinerary=saved_record,
    )


# ===========================================================================
# MCP RESOURCES (Read-Only Data Context Exposed via @mcp.resource)
# ===========================================================================

@mcp.resource(
    "weather://supported-cities",
    name="supported_weather_cities",
    description="Returns featured fallback cities and live Open-Meteo geocoding configuration.",
    mime_type="application/json",
)
async def get_supported_cities_resource() -> str:
    """Exposes featured cities and live API status as a pure read-only MCP resource."""
    cities = [
        {
            "city": info["city"],
            "country": info["country"],
            "latitude": info["lat"],
            "longitude": info["lon"],
        }
        for info in OFFLINE_FALLBACK_CATALOG.values()
    ]
    return json.dumps(
        {
            "live_open_meteo_api_enabled": USE_LIVE_WEATHER_API,
            "offline_fallback_cities": cities,
        },
        indent=2,
    )


@mcp.resource(
    "trips://saved",
    name="saved_travel_itineraries",
    description="Returns all travel itineraries currently saved in persistent JSON storage.",
    mime_type="application/json",
)
async def get_saved_trips_resource() -> str:
    """Exposes all saved itineraries from `data/saved_trips.json` as a pure MCP resource."""
    if not TRIPS_FILE.exists():
        return json.dumps({"total_saved_trips": 0, "trips": []}, indent=2)
    try:
        raw_text = await asyncio.to_thread(TRIPS_FILE.read_text, "utf-8")
        trips = json.loads(raw_text) if raw_text.strip() else []
        return json.dumps({"total_saved_trips": len(trips), "trips": trips}, indent=2)
    except (json.JSONDecodeError, OSError):
        return json.dumps({"total_saved_trips": 0, "trips": []}, indent=2)


if __name__ == "__main__":
    # Suppress verbose internal FastMCP stderr tracebacks so stdio client output stays clean
    logging.getLogger("fastmcp").setLevel(logging.CRITICAL)
    mcp.run(show_banner=False)
