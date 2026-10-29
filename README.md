# Weather & Travel Planning MCP Toolkit

EECE 503P/798S — C5: Agent Tools

This project connects an OpenAI Agents SDK host to a FastMCP server over standard input/output. The server provides exactly three tools:

- **Data:** `get_weather_forecast` retrieves forecasts without writing trip data.
- **Action:** `save_travel_itinerary` validates and saves a user-supplied packing list.
- **Orchestration:** `plan_weather_smart_trip` retrieves a forecast, applies packing rules, and saves the resulting itinerary.

Two read-only resources expose featured cities (`weather://supported-cities`) and saved trips (`trips://saved`). Forecasts use Open-Meteo when available, with a fallback catalog for network outages and tests.

## Setup

Python 3.10 or later is required. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and, optionally, `OPENAI_MODEL`. Keep `.env` private. Without an API key or when `--offline` is used, the agent uses a deterministic local model.

## Run

```bash
python test_tools.py
python agent.py --interactive
python agent.py --query "Plan a smart 3-day trip to Reykjavik and save my itinerary."
python agent.py --offline --interactive
```

The action and orchestration tools create `data/saved_trips.json` as needed. The file is excluded from version control. To use a separate storage location, set the `WEATHER_TRIPS_FILE` environment variable before running the program.

## Files

- `server.py`: FastMCP server, typed tool interfaces, validation, and resources.
- `agent.py`: MCP client, host agent, and request/response tracing.
- `offline_fallback.py`: deterministic model for offline use.
- `test_tools.py`: tests for tools, validation, resources, and the MCP connection.
- `SUMMARY_AND_REFLECTION.md` / `.pdf`: reading summary, server critique, and reflection.

The narrated video walkthrough presents source code and excerpts from an actual run. It is an edited presentation rather than a continuous live screen recording.
