/**
 * Flight Search Agent Evaluation Runner
 *
 * Evaluates flight search agent accuracy using two-agent architecture:
 * - Agent 1: Research agent with web search (finds flight details)
 * - Agent 2: Validation agent with structured output (extracts and scores)
 *
 * Usage: node eval.js [count]
 * Example: node eval.js 10  (test with 10 random flights)
 *
 * Requires:
 * - LiteLLM proxy running on configured URL
 * - MCP SearXNG server running
 * - .env file with API keys (see .env.example)
 */

import { ChatOpenAI } from '@langchain/openai';
import { createAgent } from 'langchain';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import fs from 'fs';
import { config } from 'dotenv';

// Load environment variables
config();

// Configuration from environment
const LITELLM_URL = process.env.LITELLM_URL || 'http://localhost:4000';
const LITELLM_KEY = process.env.LITELLM_API_KEY || 'sk-local-dev-key-12345';
const MCP_SEARXNG_URL = process.env.MCP_SEARXNG_URL || 'http://localhost:3000/mcp';
const DEFAULT_SAMPLE_SIZE = parseInt(process.env.DEFAULT_SAMPLE_SIZE) || 10;

// Validate required config
if (!process.env.LITELLM_API_KEY) {
    console.warn('⚠️  Warning: LITELLM_API_KEY not set in .env file. Using default dev key.');
}

// Load enriched dataset
const DATASET_PATH = process.env.DATASET_PATH || './data/sample-flights.json';
const AIRPORTS_PATH = process.env.AIRPORTS_PATH || './data/airports.json';

const enrichedDataset = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf8'));
const airports = JSON.parse(fs.readFileSync(AIRPORTS_PATH, 'utf8'));
const airportMap = new Map(airports.map(a => [a.code, a]));

// Airline name mapping
const AIRLINE_NAMES = {
  'WN': 'Southwest',
  'IB': 'Iberia',
  'EI': 'Aer Lingus',
  'KL': 'KLM',
  'B6': 'JetBlue',
  'AZ': 'ITA Airways',
  'UA': 'United',
  'AA': 'American Airlines',
  'DL': 'Delta',
  'BA': 'British Airways',
  'LH': 'Lufthansa',
  'AF': 'Air France',
  'NH': 'ANA',
  'QR': 'Qatar Airways',
  'EK': 'Emirates',
  'SQ': 'Singapore Airlines',
  'CX': 'Cathay Pacific',
  'TK': 'Turkish Airlines',
  'QF': 'Qantas',
  'VA': 'Virgin Atlantic',
  'AC': 'Air Canada',
  'FR': 'Ryanair',
  'U2': 'easyJet',
  'MH': 'Malaysia Airlines',
  'AV': 'Avianca'
};

// Aircraft ICAO code to name mapping
const AIRCRAFT_MAPPING = {
  'B738': 'Boeing 737NG',
  'B38M': 'Boeing 737MAX',
  'B737': 'Boeing 737NG',
  'B739': 'Boeing 737NG',
  'B73J': 'Boeing 737NG',
  'A320': 'Airbus A320',
  'A321': 'Airbus A321',
  'A21N': 'Airbus A321',
  'A319': 'Airbus A319',
  'A19N': 'Airbus A319',
  'A20N': 'Airbus A320',
  'A333': 'Airbus A330',
  'A332': 'Airbus A330',
  'A339': 'Airbus A330',
  'A359': 'Airbus A350',
  'A35K': 'Airbus A350',
  'A388': 'Airbus A380',
  'B78X': 'Boeing 787',
  'B788': 'Boeing 787',
  'B789': 'Boeing 787',
  'B77W': 'Boeing 777',
  'B77L': 'Boeing 777',
  'B772': 'Boeing 777',
  'B773': 'Boeing 777',
  'B744': 'Boeing 747',
  'B748': 'Boeing 747',
  'E75L': 'Embraer E175-E2',
  'E75S': 'Embraer E175-E2',
  'E170': 'Embraer E170',
  'E190': 'Embraer E190',
  'E195': 'Embraer E195-E2',
  'E290': 'Embraer E190-E2',
  'E295': 'Embraer E195-E2',
  'CRJ9': 'Bombardier CRJ',
  'CRJ7': 'Bombardier CRJ',
  'CRJ2': 'Bombardier CRJ',
  'DH8D': 'DHC Dash 8',
  'AT76': 'ATR 42/72',
  'AT72': 'ATR 42/72'
};

// Helper functions
function getAirportCity(code) {
  const airport = airportMap.get(code);
  return airport?.city || airport?.name || code;
}

function getAirlineName(code) {
  return AIRLINE_NAMES[code] || code;
}

function mapAircraftCode(icao) {
  return AIRCRAFT_MAPPING[icao] || 'Other';
}

function formatDateFromEnriched(timestamp) {
  // "2025-12-16 10:00" → "16-12-2025" (for comparison)
  const [date] = timestamp.split(' ');
  const [year, month, day] = date.split('-');
  return `${day}-${month}-${year}`;
}

function formatDateToWords(timestamp) {
  // "2025-12-16 10:00" → "16 Dec 2025" (for query)
  const [date] = timestamp.split(' ');
  const [year, month, day] = date.split('-');
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${parseInt(day)} ${months[parseInt(month) - 1]} ${year}`;
}

function formatDurationToTime(minutes) {
  // 90 → "01:30"
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
}

function timeToMinutes(timeStr) {
  // "01:30" → 90
  if (!timeStr || typeof timeStr !== 'string') return 0;
  const [hours, mins] = timeStr.split(':').map(Number);
  return hours * 60 + mins;
}

function mapToGroundTruth(flight) {
  return {
    flightNumber: flight.flight_number,
    airlineCode: flight.airline_iata,
    originCode: flight.dep_iata,
    destinationCode: flight.arr_iata,
    date: formatDateFromEnriched(flight.enriched.dep_time_scheduled),
    duration: formatDurationToTime(flight.enriched.duration),
    aircraft: mapAircraftCode(flight.aircraft_icao)
  };
}

function sampleFlights(dataset, count) {
  // Random sampling
  const shuffled = [...dataset].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(count, dataset.length));
}

// Parse CLI argument
const sampleSize = parseInt(process.argv[2]) || DEFAULT_SAMPLE_SIZE;
const sampled = sampleFlights(enrichedDataset, sampleSize);

// Convert to test cases
const DATASET = sampled.map(flight => {
  const groundTruth = mapToGroundTruth(flight);
  return {
    ...groundTruth,
    origin: getAirportCity(flight.dep_iata),
    destination: getAirportCity(flight.arr_iata),
    airlineName: getAirlineName(flight.airline_iata),
    _rawFlight: flight  // Keep reference for debugging
  };
});

// Agent 2 JSON Schema
const AGENT2_SCHEMA = {
    type: "object",
    properties: {
        flightNumber: { type: ["string", "null"] },
        airlineCode: { type: ["string", "null"] },
        departureAirportCode: { type: ["string", "null"] },
        arrivalAirportCode: { type: ["string", "null"] },
        flightDate: { type: ["string", "null"] },
        flightTime: { type: ["string", "null"] },
        aircraftName: {
            type: "string",
            enum: [
                "Airbus A220", "Airbus A319", "Airbus A320", "Airbus A321",
                "Airbus A330", "Airbus A350", "Airbus A380", "ATR 42/72",
                "Boeing 717", "Boeing 737NG", "Boeing 737MAX", "Boeing 747",
                "Boeing 757", "Boeing 767", "Boeing 777", "Boeing 787",
                "Bombardier CRJ", "DHC Dash 8", "Embraer ERJ 135",
                "Embraer ERJ 145", "Embraer E170", "Embraer E190",
                "Embraer E175-E2", "Embraer E190-E2", "Embraer E195-E2",
                "Comac C909", "Comac C919", "Superjet 100", "Tu-204/214",
                "Cessna 402", "Il-96", "Other"
            ]
        },
        flightNumberConfidence: { type: "number" },
        airlineCodeConfidence: { type: "number" },
        departureAirportConfidence: { type: "number" },
        arrivalAirportConfidence: { type: "number" },
        flightDateConfidence: { type: "number" },
        flightTimeConfidence: { type: "number" },
        aircraftNameConfidence: { type: "number" },
        validationNotes: { type: "string" }
    },
    required: [
        "flightNumberConfidence", "airlineCodeConfidence",
        "departureAirportConfidence", "arrivalAirportConfidence",
        "flightDateConfidence", "flightTimeConfidence",
        "aircraftNameConfidence", "validationNotes"
    ]
};

function generateQuery(testCase) {
    // Format: "[origin] to [destination] on [date] with [airline]"
    return `${testCase.origin} to ${testCase.destination} on ${testCase.date} with ${testCase.airlineName}`;
}

function getAgent1Prompt(userQuery) {
    const currentDate = new Date().toISOString();
    return `You are an elite flight information research assistant with access to web search tooling. Your job is to use search to accurately identify flight details according to the data you've been provided.

CURRENT DATE/TIME (UTC): ${currentDate}

USER QUERY: ${userQuery}

TASK:
Search for this flight and gather required information as possible about the flight. Focus on these three websites for your search:
1. FlightAware.com
2. FlightRadar24.com
3. aviability.com (most often shows flight time in a snippet)

SEARCH STRATEGY:
1. Extract key details from query (airline, route, date), identify missing datapoints (flight number, duration, aircraft type).
2. Formulate search queries to find missing data.
3. Review the results, focus on finding the missing pieces of data: flight number, duration and aircraft type. Pay attention to dates and trip direction.
4. Pay attention to the time: it is easy to confuse duration with scheduled departure/arrival times. Search results most likely to show scheduled times, not duration. In case search results contain only flight departure/arrival times, calculate the flight time yourself.

REQUIRED OUTPUT FORMAT:
---
FLIGHT NUMBER: [flight number found, or "NOT FOUND"]
Flight Number Source: [Website name]
Flight Number Notes: [Brief notes]

AIRLINE CODE: [IATA 2-letter code like "UA", or "NOT FOUND"]
Airline Source: [Website name]

DEPARTURE AIRPORT: [IATA 3-letter code like "JFK", or "NOT FOUND"]
Departure Source: [Website name]

ARRIVAL AIRPORT: [IATA 3-letter code like "LAX", or "NOT FOUND"]
Arrival Source: [Website name]

FLIGHT DATE: [DD-MM-YYYY format, or "NOT FOUND"]

FLIGHT TIME: [HH:MM in 24-hour format, or "NOT FOUND". This is SCHEDULED OR ACTUAL FLIGHT DURATION, not departure or arrival time.]
Time Source: [Website name]
Time Notes: [Brief notes]

AIRCRAFT TYPE: [Full aircraft name. Examples: "Airbus A321", "Boeing 737-800", "Embraer E190". If you find codes like "32B", "73H", "E90", convert them to full names. Write "NOT FOUND" only if no aircraft information exists.]
Aircraft Source: [Website name]
Aircraft Notes: [If you converted a code to full name, mention the original code]

OVERALL ASSESSMENT:
[2-3 sentences summarizing: Did you find a definitive match for this flight? What is missing, what present?]
---

CRITICAL RULES:
1. If you cannot find reliable information, write "NOT FOUND" - do NOT guess
2. For aircraft name, always convert codes to full names
3. Data provided by user is always ground truth - focus on finding missing data only

Now search and report for the user's query.`;
}

function getAgent2Prompt(userQuery, agent1Response) {
    const currentDate = new Date().toISOString();
    return `You are a flight data validator and matcher. You will receive the ORIGINAL USER QUERY and a research report from an information extraction agent. Your job is to convert it into structured JSON with confidence scores.

CURRENT DATE/TIME (UTC): ${currentDate}

ORIGINAL USER QUERY:
"${userQuery}"

AGENT 1 RESEARCH REPORT:
---
${agent1Response}
---

YOUR TASK:
1. Review the original user query to understand what data the user provided vs what Agent 1 had to find
2. Extract each data field from Agent 1's report
3. Assign confidence scores (0.0 to 1.0) based on source quality
4. Return structured JSON

FIELD EXTRACTION RULES:
1. **flightNumber**: Numeric only (e.g., "2453" not "UA2453"). Null if "NOT FOUND".
2. **airlineCode**: IATA 2-letter code (e.g., "UA", "AA", "DL"). Null if not found.
3. **departureAirportCode**: IATA 3-letter code (e.g., "JFK"). Null if not found.
4. **arrivalAirportCode**: IATA 3-letter code (e.g., "LAX"). Null if not found.
5. **flightDate**: DD-MM-YYYY format exactly. Null if not found.
6. **flightTime**: HH:MM in 24-hour format (e.g., "14:30"). Null if not found. This is flight duration, not departure/arrival time.
7. **aircraftName**: MUST use enum value from the valid aircraft list. Map to closest match.
8. **validationNotes**: 1-2 sentences summarizing key validation issues.

CONFIDENCE SCORING:
- 0.95-1.0: Data from FlightAware/official airline with confirmation
- 0.80-0.94: Data from FlightRadar24/reputable source
- 0.60-0.79: Data from airline schedule
- 0.40-0.59: Inferred/uncertain data
- 0.0-0.39: NOT FOUND or guessed

Now validate and structure the research report above.`;
}

function formatDuration(duration) {
    // Convert "1:11" or "01:11" to standardized format for comparison
    const parts = duration.split(':');
    const hours = parts[0].padStart(2, '0');
    const minutes = parts[1];
    return `${hours}:${minutes}`;
}

function compareDuration(extracted, expected, tolerance = 15) {
    // Compare duration with ±15 minute tolerance
    if (!extracted || !expected) return false;
    const extractedMins = timeToMinutes(extracted);
    const expectedMins = timeToMinutes(expected);
    const diff = Math.abs(extractedMins - expectedMins);
    return diff <= tolerance;
}

function compareAircraft(extracted, expected) {
    // Exact match or family match
    if (!extracted || !expected) return false;
    if (extracted === expected) return true;
    // Family match: "Boeing 737-800" contains "Boeing 737"
    const family = expected.split(' ')[0] + ' ' + expected.split(' ')[1]?.split(/[^0-9]/)[0];
    return extracted.toLowerCase().includes(family.toLowerCase());
}

function compareResults(extracted, groundTruth) {
    const matches = {
        flightNumber: extracted.flightNumber === groundTruth.flightNumber,
        airlineCode: extracted.airlineCode === groundTruth.airlineCode,
        departureAirport: extracted.departureAirportCode === groundTruth.originCode,
        arrivalAirport: extracted.arrivalAirportCode === groundTruth.destinationCode,
        flightDate: extracted.flightDate === groundTruth.date,  // Already in DD-MM-YYYY format
        aircraftName: compareAircraft(extracted.aircraftName, groundTruth.aircraft),
        flightTime: compareDuration(extracted.flightTime, groundTruth.duration, 15)
    };

    return matches;
}

async function runBatchEvaluation() {
    console.log('🚀 Flight Search Agent Evaluation\n');
    console.log(`📊 Dataset: ${DATASET.length} test cases`);
    console.log(`🔧 LiteLLM: ${LITELLM_URL}`);
    console.log(`🔍 MCP: ${MCP_SEARXNG_URL}\n`);
    console.log('═'.repeat(80));

    let mcpClient = null;
    let model = null;
    let tools = null;

    try {
        // Initialize MCP Client
        console.log('📡 Initializing MCP client and model...');
        mcpClient = new MultiServerMCPClient({
            mcpServers: {
                searxng: { url: MCP_SEARXNG_URL }
            }
        });

        tools = await mcpClient.getTools();
        console.log(`✓ Loaded ${tools.length} MCP tools`);

        model = new ChatOpenAI({
            model: 'mistral-large',
            configuration: {
                baseURL: `${LITELLM_URL}/v1`,
                apiKey: LITELLM_KEY
            },
            temperature: 0
        });
        console.log('✓ Model ready\n');

        const results = [];
        let totalDuration = 0;

        // Process each test case
        for (let i = 0; i < DATASET.length; i++) {
            const testCase = DATASET[i];
            const query = generateQuery(testCase);

            console.log(`\n[${i + 1}/${DATASET.length}] Testing: ${query}`);
            console.log(`Expected: ${testCase.airlineCode}${testCase.flightNumber}, ${testCase.aircraft}, ${testCase.duration}`);

            try {
                const startTime = Date.now();

                // Step 1: Agent 1 - Research
                const agent1Prompt = getAgent1Prompt(query);
                const agent = createAgent({
                    model: model,
                    tools: tools
                });

                const agent1Result = await agent.invoke({
                    messages: [{ role: 'user', content: agent1Prompt }]
                });

                const researchReport = agent1Result.messages[agent1Result.messages.length - 1].content;

                // Step 2: Agent 2 - Validation
                const agent2Prompt = getAgent2Prompt(query, researchReport);
                const agent2Response = await fetch(`${LITELLM_URL}/v1/chat/completions`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${LITELLM_KEY}`
                    },
                    body: JSON.stringify({
                        model: 'mistral-small',
                        messages: [{ role: 'user', content: agent2Prompt }],
                        response_format: {
                            type: 'json_schema',
                            json_schema: {
                                name: 'FlightData',
                                strict: true,
                                schema: AGENT2_SCHEMA
                            }
                        },
                        temperature: 0
                    })
                });

                const agent2Result = await agent2Response.json();
                const flightData = JSON.parse(agent2Result.choices[0].message.content);

                const duration = ((Date.now() - startTime) / 1000).toFixed(2);
                totalDuration += parseFloat(duration);

                // Compare with ground truth
                const matches = compareResults(flightData, testCase);
                const allMatch = Object.values(matches).every(m => m);

                results.push({
                    query,
                    groundTruth: testCase,
                    extracted: flightData,
                    matches,
                    allMatch,
                    duration
                });

                const status = allMatch ? '✓' : '✗';
                console.log(`${status} ${i + 1}/${DATASET.length} completed in ${duration}s`);

            } catch (error) {
                console.error(`✗ Error: ${error.message}`);
                results.push({
                    query,
                    groundTruth: testCase,
                    error: error.message,
                    allMatch: false
                });
            }
        }

        // Summary
        console.log('\n\n' + '═'.repeat(80));
        console.log('📊 BATCH EVALUATION RESULTS');
        console.log('═'.repeat(80));

        const successCount = results.filter(r => r.allMatch).length;
        const successRate = ((successCount / results.length) * 100).toFixed(1);

        const fieldAccuracy = {
            flightNumber: 0,
            airlineCode: 0,
            departureAirport: 0,
            arrivalAirport: 0,
            flightDate: 0,
            aircraftName: 0,
            flightTime: 0
        };

        results.forEach(r => {
            if (r.matches) {
                Object.keys(fieldAccuracy).forEach(field => {
                    if (r.matches[field]) fieldAccuracy[field]++;
                });
            }
        });

        console.log(`\nOverall Success Rate: ${successRate}% (${successCount}/${results.length})`);
        console.log(`Average Duration: ${(totalDuration / results.length).toFixed(2)}s per test`);
        console.log(`Total Duration: ${totalDuration.toFixed(2)}s`);

        console.log('\nField-by-Field Accuracy:');
        Object.entries(fieldAccuracy).forEach(([field, count]) => {
            const pct = ((count / results.length) * 100).toFixed(1);
            console.log(`  ${field.padEnd(20)}: ${pct}% (${count}/${results.length})`);
        });

        // Show failed flights
        const failedFlights = results.filter(r => !r.allMatch);
        if (failedFlights.length > 0) {
            console.log(`\nFailed Flights (${failedFlights.length}):`);
            failedFlights.forEach(f => {
                if (f.error) {
                    console.log(`  ${f.query} - ERROR: ${f.error}`);
                } else {
                    const failedFields = Object.entries(f.matches || {})
                        .filter(([_, match]) => !match)
                        .map(([field]) => field)
                        .join(', ');
                    console.log(`  ${f.query} - Failed fields: ${failedFields}`);
                }
            });
        }

        console.log('\n' + '═'.repeat(80));

    } catch (error) {
        console.error('\n❌ Fatal Error:', error.message);
        console.error(error.stack);
        process.exit(1);
    } finally {
        if (mcpClient) {
            await mcpClient.close();
        }
    }
}

// Run batch evaluation
runBatchEvaluation();
