# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flight search agent evaluation pipeline using a two-agent architecture (LangChain + MCP) to test flight information retrieval accuracy against ground truth data.

**Two-Agent Architecture:**
1. **Agent 1 (Research)**: Uses Mistral-large with web search via MCP SearXNG to find missing flight details
2. **Agent 2 (Validation)**: Uses Mistral-small with structured JSON output to validate and score extracted data

**Tech Stack:**
- LangChain (agent orchestration) + LangGraph (agent workflows)
- MCP (Model Context Protocol) adapters for tool integration
- LiteLLM proxy (unified API for Mistral models)
- [OPTIONAL] SearXNG (privacy-respecting metasearch engine)
- Docker Compose (service orchestration)

There are currently several approaches being tested based on different search solutions:
1. Tvily MCP with liteLLM
2. Pollinate.ai with gemini + google search.
3. SearXNG with MCP


## Development Commands

### Environment Setup
```bash
# Copy environment template
cp .env.example .env
# Edit .env and add your Mistral API key from https://console.mistral.ai/

# Install dependencies
npm install
```

### Docker Services
```bash
# Start all services (LiteLLM + PostgreSQL + SearXNG + Redis + MCP)
docker-compose up -d

# View logs
docker-compose logs -f litellm      # LiteLLM proxy logs
docker-compose logs -f searxng       # SearXNG search logs
docker-compose logs -f mcp-searxng   # MCP wrapper logs

# Stop services
docker-compose down

# Restart specific service
docker-compose restart litellm
```

### Running Evaluations
```bash
# Run standard evaluation (10 flights)
node eval.js
# or
npm run eval:standard

# Quick test (3 flights)
npm test
# or
npm run eval:quick

# Custom sample size
node eval.js 15

# Full dataset evaluation (20 flights)
npm run eval:full
```

### Verify Services
```bash
curl http://localhost:4000/health  # LiteLLM health check
curl http://localhost:8080         # SearXNG web interface
curl http://localhost:3000/health  # MCP server (if available)
```

## Architecture Details

### Two-Agent Evaluation Flow
1. **Test case generation**: Random sampling from `data/sample-flights.json`
2. **Query formatting**: Converts flight data to natural language query (e.g., "Las Vegas to Albuquerque on 16 Dec 2025 with Southwest"). The query ALWAYs contains route, date and alirline and NEVER flight number, aircraft type or flight time
3. **Agent 1 execution**: LangGraph ReAct agent with web search tools finds missing flight details (flight number, duration, aircraft type)
4. **Agent 2 execution**: Structured output extraction with confidence scoring (0.0-1.0)
5. **Comparison**: Field-by-field accuracy check against ground truth
6. **Results output**: CSV export with per-field success rates and overall accuracy

### Agent 1 Prompt Strategy
Always reuse the same exact CoT-based prompt. Never deviate from it.

- Focuses on FlightAware, FlightRadar24, and aviability.com
- Requires strict format with source citations
- Distinguishes between scheduled times vs. flight duration
- Aircraft code to full name conversion (e.g., "73H" → "Boeing 737-800")
- Critical rule: Write "NOT FOUND" instead of guessing

### Agent 2 Validation Schema
- JSON Schema with strict field types and enum constraints
- Confidence scoring based on source reputation:
  - 0.95-1.0: Official sources (FlightAware/airline)
  - 0.80-0.94: FlightRadar24/reputable sources
  - 0.60-0.79: Airline schedules
  - 0.40-0.59: Inferred/uncertain
  - 0.0-0.39: NOT FOUND/guessed
- Aircraft name must map to predefined enum (30+ aircraft types)

### Comparison Logic
- **Exact match**: Flight number, airline code, airport codes, date
- **Fuzzy match**: Aircraft (family-level matching, e.g., "Boeing 737-800" matches "Boeing 737NG")
- **Tolerance-based**: Duration (±15 minutes)
- **Overall success**: All 7 fields correct (flightNumber, airlineCode, departureAirport, arrivalAirport, flightDate, flightTime, aircraftName)

### MCP Integration
- `MultiServerMCPClient` from `@langchain/mcp-adapters`
- `useStandardContentBlocks: true` for LangChain compatibility
- Tools retrieved via `mcpClient.getTools()` and passed to LangGraph ReAct agent
- SearXNG MCP wrapper exposes search functionality at `http://localhost:3000/mcp`

### LiteLLM Configuration
The `litellm-config.yaml` defines model routing:
- `mistral-large`: Agent 1 (research with web search)
- `mistral-small`: Agent 2 (validation and extraction)
- Settings: 2 retries, 600s timeout, 100 max parallel requests
- PostgreSQL backend for request logging/tracking

## Data Files

### Ground Truth Dataset
- **Path**: `data/sample-flights.json`
- **Format**: Array of flight objects with IATA codes, ICAO aircraft codes, and enriched scheduled times
- **Fields**:
  - `flight_number`, `airline_iata`, `dep_iata`, `arr_iata`, `aircraft_icao`
  - `enriched.dep_time_scheduled`, `enriched.arr_time_scheduled`, `enriched.duration`
- Contains 20 sample flights across various routes and airlines

### Airport Reference Data
- **Path**: `data/airports.json`
- **Format**: Array of airport objects with IATA codes and city names
- Used to convert airport codes to human-readable city names for natural language queries

## Evaluation Variants

The repository includes multiple evaluation script variants:

- **`eval.js`**: Main two-agent pipeline with Chain-of-Thought reasoning
- **`eval-no-cot.js`**: Two-agent without CoT (comparison baseline)
- **`eval-single-agent.js`**: Single-agent baseline (no validation stage)
- **`eval-single-agent-cot.js`**: Single-agent with CoT
- **`eval-litellm-tavily.js`**: Uses Tavily search API instead of SearXNG
- **`eval-pollen.js`**: Uses Pollinations AI API (alternative LLM provider)

All variants follow the same evaluation metrics and comparison logic.

## Output Files

### Evaluation Results
- **Format**: `eval-results-{timestamp}.csv`
- **Columns**: Test case details, extracted values, ground truth, field-by-field matches, confidence scores, duration
- Generated after each evaluation run

### Execution Traces
- **Format**: `trace-{timestamp}.md`
- **Content**: Full agent conversation history, tool calls, reasoning steps
- Useful for debugging agent behavior and prompt engineering

## Environment Variables

Required in `.env`:
```
MISTRAL_API_KEY          # From https://console.mistral.ai/
LITELLM_URL              # Default: http://localhost:4000
LITELLM_API_KEY          # Default: sk-local-dev-key-12345
LITELLM_MASTER_KEY       # Default: sk-master-local-dev-key-12345
MCP_SEARXNG_URL          # Default: http://localhost:3000/mcp
DEFAULT_SAMPLE_SIZE      # Default: 10
```

Optional:
```
DATASET_PATH             # Custom flight dataset path
AIRPORTS_PATH            # Custom airports reference path
```

## Key Implementation Details

### Aircraft Mapping
The system uses aircraft family-level matching via `AIRCRAFT_MAPPING` (100+ lines in `eval.js:75-115`). This maps ICAO codes to standardized family names:
- B738, B739, B73J → "Boeing 737NG"
- A320, A20N → "Airbus A320"
- E75L, E75S → "Embraer E175-E2"

### Duration Calculation
The `compareDuration()` function (eval.js:345-352) uses ±15 minute tolerance to account for:
- Schedule variations
- Rounding differences between sources
- Flight time vs. scheduled time discrepancies

### Date Format Conversion
- **Input format** (dataset): "2025-12-16 10:00" (ISO timestamp)
- **Query format**: "16 Dec 2025" (human-readable)
- **Comparison format**: "16-12-2025" (DD-MM-YYYY)
- Conversion handled by `formatDateFromEnriched()` and `formatDateToWords()`

## Troubleshooting

### Common Issues

**LiteLLM connection errors:**
- Verify Docker services running: `docker-compose ps`
- Check Mistral API key in `.env`
- Review logs: `docker-compose logs -f litellm`

**MCP SearXNG errors:**
- Confirm SearXNG accessible: `curl http://localhost:8080`
- Check MCP wrapper logs: `docker-compose logs -f mcp-searxng`
- Verify network connectivity: `docker network inspect flight-agent-eval_searxng-net`

**Low accuracy results:**
- Review trace files to debug agent reasoning
- Check Agent 1 prompt for search strategy issues
- Verify confidence scoring in Agent 2 validation
- Ensure test queries are date-appropriate (future flights may have limited data)

**Agent timeout errors:**
- Increase `timeout` in `litellm-config.yaml` (default: 600s)
- Reduce sample size for faster iteration
- Check if external search sources (FlightAware, FlightRadar24) are accessible
