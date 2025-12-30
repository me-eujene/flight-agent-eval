# Flight Search Agent Evaluator

Evaluation pipeline for testing flight search agent accuracy using a two-agent architecture (LangChain + MCP).

## Quick Start

```bash
# Install dependencies
npm install

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run evaluation
node eval.js 10  # Test with 10 random flights
```

## Requirements

- **Node.js** 18+
- **LiteLLM proxy** running (with Mistral models)
- **MCP SearXNG server** running
- API keys configured in `.env`

## Architecture

**Two-Agent Pipeline:**
1. **Agent 1** (Mistral-large): Research agent with web search - finds flight details from FlightAware, FlightRadar24, etc.
2. **Agent 2** (Mistral-small): Validation agent - structures data with confidence scores

## Configuration

Edit `.env`:

```env
LITELLM_URL=http://localhost:4000
LITELLM_API_KEY=your-api-key-here
MCP_SEARXNG_URL=http://localhost:3000/mcp
DEFAULT_SAMPLE_SIZE=10
```

## Usage

```bash
node eval.js        # Default (10 flights)
node eval.js 3      # Quick test (3 flights)
node eval.js 20     # Comprehensive (20 flights from sample dataset)
```

## Sample Output

```
🚀 Flight Search Agent Evaluation

📊 Dataset: 10 test cases
🔧 LiteLLM: http://localhost:4000
🔍 MCP: http://localhost:3000/mcp

[1/10] Testing: Las Vegas to Albuquerque on 16 Dec 2025 with Southwest
Expected: WN548, Boeing 737NG, 01:30
✓ 1/10 completed in 26.4s

...

════════════════════════════════════════════════════════════════════════════════
📊 BATCH EVALUATION RESULTS
════════════════════════════════════════════════════════════════════════════════

Overall Success Rate: 30.0% (3/10)
Average Duration: 25.1s per test

Field-by-Field Accuracy:
  flightNumber        : 50.0% (5/10)
  airlineCode         : 100.0% (10/10)
  departureAirport    : 100.0% (10/10)
  arrivalAirport      : 100.0% (10/10)
  flightDate          : 100.0% (10/10)
  aircraftName        : 90.0% (9/10)
  flightTime          : 40.0% (4/10)
```

## Dataset

Includes 20 sample flights in `data/sample-flights.json` with ground truth:
- Airport codes, airline codes, flight numbers
- Aircraft types (ICAO codes)
- Scheduled departure/arrival times
- Flight duration

Format:
```json
{
  "flight_number": "548",
  "airline_iata": "WN",
  "dep_iata": "LAS",
  "arr_iata": "ABQ",
  "aircraft_icao": "B738",
  "enriched": {
    "dep_time_scheduled": "2025-12-16 10:00",
    "arr_time_scheduled": "2025-12-16 12:30",
    "duration": 90
  }
}
```

## Evaluation Metrics

- **Overall Success Rate**: All 7 fields correct
- **Per-Field Accuracy**: Individual field match rates
- **Duration Tolerance**: ±15 minutes
- **Aircraft Matching**: Family-level fuzzy matching (Boeing 737-800 = Boeing 737NG)

## Extending

### Add More Flights
Add entries to `data/sample-flights.json` following the format above.

### Custom Dataset Path
Set in `.env`:
```env
DATASET_PATH=./path/to/your/dataset.json
AIRPORTS_PATH=./path/to/airports.json
```

### Modify Comparison Logic
Edit `compareResults()` function in `eval.js`:
- Duration tolerance (default ±15 min)
- Aircraft fuzzy matching rules
- Field validation logic

## License

MIT

## Contributing

PRs welcome! Please ensure:
- No API keys committed
- Tests pass with sample dataset
- Documentation updated
