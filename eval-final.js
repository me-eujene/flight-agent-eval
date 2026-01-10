/**
 * Flight Search Agent Evaluation - Final Implementation
 *
 * Architecture:
 * - LangChain 2-step chain (no agents, no tools)
 * - Step 1: gemini-fast → searches web, returns text report
 * - Step 2: gemini-fast + formal JSON schema → structured output
 * - Output: Human-readable markdown for manual review
 *
 * Usage: node eval-final.js
 */

import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';
import { z } from 'zod';
import fs from 'fs';
import { config } from 'dotenv';

config();

// Configuration
const POLLINATIONS_API_KEY = 'sk_szglh5aFfm2I1aiXvZEoXYvKDDGVOOl1';
const POLLINATIONS_URL = 'https://gen.pollinations.ai/v1';

// Load dataset
const DATASET_PATH = './data/sample-flights.json';
const AIRPORTS_PATH = './data/airports.json';

const enrichedDataset = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf8'));
const airports = JSON.parse(fs.readFileSync(AIRPORTS_PATH, 'utf8'));
const airportMap = new Map(airports.map(a => [a.code, a]));

// Airline name mapping
const AIRLINE_NAMES = {
    'WN': 'Southwest', 'IB': 'Iberia', 'EI': 'Aer Lingus', 'KL': 'KLM',
    'B6': 'JetBlue', 'AZ': 'ITA Airways', 'UA': 'United', 'AA': 'American Airlines',
    'DL': 'Delta', 'BA': 'British Airways', 'LH': 'Lufthansa', 'AF': 'Air France'
};

// Build dataset
const DATASET = enrichedDataset.map(flight => {
    const originAirport = airportMap.get(flight.dep_iata);
    const destAirport = airportMap.get(flight.arr_iata);
    const airlineName = AIRLINE_NAMES[flight.airline_iata] || flight.airline_iata;

    const durationMinutes = flight.enriched?.duration || 0;
    const hours = Math.floor(durationMinutes / 60);
    const mins = durationMinutes % 60;
    const durationFormatted = `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;

    return {
        airlineCode: flight.airline_iata,
        airlineName,
        flightNumber: flight.flight_number,
        originCode: flight.dep_iata,
        origin: originAirport?.city || flight.dep_iata,
        destinationCode: flight.arr_iata,
        destination: destAirport?.city || flight.arr_iata,
        date: flight.enriched?.dep_time_scheduled?.split(' ')[0]?.split('-').reverse().join('-') || '16-12-2025',
        duration: durationFormatted,
        aircraft: flight.aircraft_icao
    };
});

function generateQuery(testCase) {
    return `${testCase.origin} to ${testCase.destination} on ${testCase.date} with ${testCase.airlineName}`;
}

// Formal JSON Schema (Zod)
const FlightDataSchema = z.object({
    flightNumber: z.string().nullable().describe('Numeric part only (e.g., "548"), or null if NOT FOUND'),
    airlineCode: z.string().nullable().describe('IATA 2-letter code (e.g., "WN"), or null if NOT FOUND'),
    departureAirportCode: z.string().nullable().describe('IATA 3-letter code (e.g., "LAS"), or null if NOT FOUND'),
    arrivalAirportCode: z.string().nullable().describe('IATA 3-letter code (e.g., "ABQ"), or null if NOT FOUND'),
    flightDate: z.string().nullable().describe('DD-MM-YYYY format, or null if NOT FOUND'),
    flightTime: z.string().nullable().describe('Duration in HH:MM format, or null if NOT FOUND'),
    aircraftName: z.string().nullable().describe('Full aircraft name (e.g., "Boeing 737-800"), or null if NOT FOUND'),
    overallAssessment: z.string().describe('Brief summary of what was found')
});

// Agent 1 Prompt from eval.js (lines 240-294)
function getAgent1Prompt(userQuery) {
    const currentDate = new Date().toISOString();
    return `You are an elite flight information research assistant with access to web search. Your job is to search and accurately identify flight details. Any attempt to hallucinate or guess information is forbidden.

CURRENT DATE/TIME (UTC): ${currentDate}

USER QUERY: ${userQuery}

TASK:
Search for this flight and gather required information. Focus on these three websites:
1. FlightAware.com
2. FlightRadar24.com
3. aviability.com (most often shows flight time in a snippet)

SEARCH STRATEGY:
1. Extract key details from query (airline, route, date), identify missing datapoints (flight number, duration, aircraft type).
2. Formulate search queries to find missing data.
3. Review the results, focus on finding: flight number, duration and aircraft type. Pay attention to dates and trip direction.
4. Pay attention to the time: it is easy to confuse duration with scheduled departure/arrival times. Search results most likely to show scheduled times, not duration. In case search results contain only flight departure/arrival times, calculate the flight time yourself.

CHAIN OF THOUGHT REASONING:
Before providing your final output, you MUST think through your findings step-by-step:

1. What did I find?** - List each piece of data you discovered and from which source
2. Validate that the data matches the direction of the fligtht (origin → destination)
3. What's missing?** - Identify which fields you couldn't find
4. Are there conflicts?** - Note any contradictory information between sources
5. Duration calculation** - If you only found departure/arrival times, show your calculation for flight duration
6. Aircraft codes** - If you found codes like "73H" or "32B", explain the conversion to full names

After this reasoning, provide your structured output.

REQUIRED OUTPUT FORMAT:
---
FLIGHT NUMBER: [flight number found, or "NOT FOUND"]
Flight Number Source: [Website name]
Flight Number Notes: [Brief notes]

AIRLINE CODE: [IATA 2-letter code like "WN", or "NOT FOUND"]
Airline Source: [Website name or "Provided in query"]

DEPARTURE AIRPORT: [IATA 3-letter code like "LAS", or "NOT FOUND"]
Departure Source: [Website name or "Provided in query"]

ARRIVAL AIRPORT: [IATA 3-letter code like "ABQ", or "NOT FOUND"]
Arrival Source: [Website name or "Provided in query"]

FLIGHT DATE: [DD-MM-YYYY format, or "NOT FOUND"]
Date Source: [Website name or "Provided in query"]

FLIGHT TIME: [HH:MM in 24-hour format, or "NOT FOUND". This is SCHEDULED OR ACTUAL FLIGHT DURATION, not departure or arrival time.]
Time Source: [Website name]
Time Notes: [Brief notes - if calculated from dep/arr times, show the calculation]

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

// Structuring prompt for Step 2
const structurePromptTemplate = `Convert this search report into structured JSON according to the schema.

Search Report:
{searchResults}

Extract all fields:
- flightNumber: just the numeric part (e.g., "548" not "WN548"), or null if NOT FOUND
- airlineCode: IATA 2-letter code (e.g., "WN"), or null if NOT FOUND
- departureAirportCode: IATA 3-letter code (e.g., "LAS"), or null if NOT FOUND
- arrivalAirportCode: IATA 3-letter code (e.g., "ABQ"), or null if NOT FOUND
- flightDate: DD-MM-YYYY format, or null if NOT FOUND
- flightTime: duration in HH:MM format, or null if NOT FOUND
- aircraftName: full aircraft name (e.g., "Boeing 737-800"), or null if NOT FOUND
- overallAssessment: brief summary from the report`;

// Initialize LLMs
const searchLLM = new ChatOpenAI({
    modelName: 'gemini-fast',
    configuration: {
        baseURL: POLLINATIONS_URL,
        apiKey: POLLINATIONS_API_KEY
    },
    temperature: 0
});

const structureLLM = new ChatOpenAI({
    modelName: 'gemini-fast',
    configuration: {
        baseURL: POLLINATIONS_URL,
        apiKey: POLLINATIONS_API_KEY
    },
    temperature: 0
}).withStructuredOutput(FlightDataSchema);

// Create chain factory
function createChain(query) {
    const searchPrompt = PromptTemplate.fromTemplate(getAgent1Prompt(query));
    const structurePrompt = PromptTemplate.fromTemplate(structurePromptTemplate);

    return RunnableSequence.from([
        // Step 1: Search (returns text)
        {
            searchResults: searchPrompt.pipe(searchLLM).pipe(new StringOutputParser())
        },
        // Step 2: Structure (returns typed object)
        structurePrompt,
        structureLLM
    ]);
}

// Generate markdown report
function generateMarkdownReport(results, timestamp) {
    let markdown = `# Flight Search Evaluation - Manual Review\n\n`;
    markdown += `**Generated:** ${new Date(timestamp).toISOString()}\n`;
    markdown += `**Model:** Pollinations.ai gemini-fast\n`;
    markdown += `**Architecture:** LangChain 2-step chain (search → structure)\n`;
    markdown += `**Total Flights:** ${results.length}\n`;
    markdown += `**Avg Duration:** ${(results.reduce((s, r) => s + parseFloat(r.duration || 0), 0) / results.length).toFixed(2)}s\n\n`;
    markdown += `---\n\n`;

    results.forEach((result, idx) => {
        const { query, groundTruth, extracted, duration } = result;

        markdown += `## ${idx + 1}. ${query}\n\n`;
        markdown += `**Duration:** ${duration}s\n\n`;

        markdown += `| Field | Ground Truth | Extracted | Match |\n`;
        markdown += `|-------|--------------|-----------|-------|\n`;
        markdown += `| Flight Number | ${groundTruth.airlineCode}${groundTruth.flightNumber} | ${extracted.airlineCode || '?'}${extracted.flightNumber || '?'} | ⬜ |\n`;
        markdown += `| Airline | ${groundTruth.airlineCode} | ${extracted.airlineCode || 'null'} | ⬜ |\n`;
        markdown += `| Departure | ${groundTruth.originCode} | ${extracted.departureAirportCode || 'null'} | ⬜ |\n`;
        markdown += `| Arrival | ${groundTruth.destinationCode} | ${extracted.arrivalAirportCode || 'null'} | ⬜ |\n`;
        markdown += `| Date | ${groundTruth.date} | ${extracted.flightDate || 'null'} | ⬜ |\n`;
        markdown += `| Aircraft | ${groundTruth.aircraft} | ${extracted.aircraftName || 'null'} | ⬜ |\n`;
        markdown += `| Duration | ${groundTruth.duration} | ${extracted.flightTime || 'null'} | ⬜ |\n\n`;

        if (extracted.overallAssessment) {
            markdown += `**Assessment:** ${extracted.overallAssessment}\n\n`;
        }

        markdown += `<details>\n<summary>View Extracted JSON</summary>\n\n\`\`\`json\n${JSON.stringify(extracted, null, 2)}\n\`\`\`\n</details>\n\n`;
        markdown += `---\n\n`;
    });

    markdown += `## Manual Scoring Instructions\n\n`;
    markdown += `For each flight above, check the "Match" column:\n`;
    markdown += `- ✅ = Correct match\n`;
    markdown += `- ⚠️ = Partial (e.g., "737" vs "Boeing 737-800")\n`;
    markdown += `- ✗ = Wrong or NOT FOUND\n\n`;
    markdown += `**Total Score:** ___/140 fields (20 flights × 7 fields)\n`;

    return markdown;
}

// Main evaluation function
async function runEvaluation() {
    console.log('🚀 Flight Search Agent Evaluation - LangChain + Pollinations');
    console.log(`📊 Testing ${DATASET.length} flights\n`);

    const results = [];
    let totalDuration = 0;

    for (let i = 0; i < DATASET.length; i++) {
        const testCase = DATASET[i];
        const query = generateQuery(testCase);

        console.log(`[${i + 1}/${DATASET.length}] ${query}`);

        try {
            const startTime = Date.now();

            // Create and invoke chain
            const chain = createChain(query);
            const extracted = await chain.invoke({ query });

            const duration = ((Date.now() - startTime) / 1000).toFixed(2);
            totalDuration += parseFloat(duration);

            results.push({
                query,
                groundTruth: testCase,
                extracted,
                duration
            });

            console.log(`  ✓ Completed in ${duration}s`);

        } catch (error) {
            console.error(`  ✗ Error: ${error.message}`);
            results.push({
                query,
                groundTruth: testCase,
                extracted: {
                    flightNumber: null,
                    airlineCode: null,
                    departureAirportCode: null,
                    arrivalAirportCode: null,
                    flightDate: null,
                    flightTime: null,
                    aircraftName: null,
                    overallAssessment: `Error: ${error.message}`
                },
                duration: 0
            });
        }
    }

    // Generate markdown report
    const timestamp = Date.now();
    const markdown = generateMarkdownReport(results, timestamp);
    const outputPath = `./eval-results-${timestamp}.md`;
    fs.writeFileSync(outputPath, markdown);

    const avgDuration = (totalDuration / results.length).toFixed(1);
    console.log(`\n✅ Completed ${results.length} tests in ${totalDuration.toFixed(1)}s (avg ${avgDuration}s/test)`);
    console.log(`📄 Manual review file: ${outputPath}\n`);
    console.log(`Next: Open the file and manually score each flight!\n`);
}

runEvaluation().catch(console.error);
