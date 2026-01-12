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
import { isSameFamily, getAircraftSimilarity } from './lib/aircraft-utils.js';
import { calculateAllMetrics, getSummaryStats } from './lib/metrics.js';

config();

// Configuration
const POLLINATIONS_API_KEY = 'sk_szglh5aFfm2I1aiXvZEoXYvKDDGVOOl1';
const POLLINATIONS_URL = 'https://gen.pollinations.ai/v1';

// Load dataset
const DATASET_PATH = './flight-dataset-landed-simple.json';
const AIRPORTS_PATH = './data/airports.json';
const AIRLINES_PATH = './data/airlines.json';

const simpleDataset = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf8'));
const airports = JSON.parse(fs.readFileSync(AIRPORTS_PATH, 'utf8'));
const airlines = JSON.parse(fs.readFileSync(AIRLINES_PATH, 'utf8'));

const airportMap = new Map(airports.map(a => [a.code, a]));
const airlineMap = new Map(airlines.map(a => [a.code, a.name]));

// ICAO code to full aircraft name mapping (from Agent 1 prompt)
const AIRCRAFT_MAPPING = {
    // Boeing 737
    'B738': 'Boeing 737NG',
    'B739': 'Boeing 737NG',
    'B73J': 'Boeing 737NG',
    'B737': 'Boeing 737NG',
    'B38M': 'Boeing 737MAX',
    // Boeing other
    'B712': 'Boeing 717',
    'B753': 'Boeing 757',
    'B77W': 'Boeing 777',
    'B77L': 'Boeing 777',
    'B772': 'Boeing 777',
    'B773': 'Boeing 777',
    'B78X': 'Boeing 787',
    'B788': 'Boeing 787',
    'B789': 'Boeing 787',
    'B744': 'Boeing 747',
    'B748': 'Boeing 747',
    // Airbus A320 family
    'A320': 'Airbus A320',
    'A20N': 'Airbus A320',
    'A321': 'Airbus A321',
    'A21N': 'Airbus A321',
    'A319': 'Airbus A319',
    'A19N': 'Airbus A319',
    // Airbus wide-body
    'A333': 'Airbus A330',
    'A332': 'Airbus A330',
    'A339': 'Airbus A330',
    'A359': 'Airbus A350',
    'A35K': 'Airbus A350',
    'A388': 'Airbus A380',
    // Regional
    'E75L': 'Embraer E175-E2',
    'E75S': 'Embraer E175-E2',
    'E170': 'Embraer E170',
    'E190': 'Embraer E190',
    'E290': 'Embraer E190',
    'E195': 'Embraer E195-E2',
    'E295': 'Embraer E195-E2',
    'CRJ9': 'Bombardier CRJ',
    'CRJ7': 'Bombardier CRJ',
    'CRJ2': 'Bombardier CRJ',
    'DH8D': 'DHC Dash 8',
    'AT76': 'ATR 42/72',
    'AT72': 'ATR 42/72'
};

/**
 * Map ICAO aircraft code to full name
 * @param {string} icaoCode - ICAO aircraft code
 * @returns {string} Full aircraft name
 */
function mapAircraftCode(icaoCode) {
    return AIRCRAFT_MAPPING[icaoCode] || icaoCode;
}

// Build dataset
const DATASET = simpleDataset.map(flight => {
    const originAirport = airportMap.get(flight.dep_iata);
    const destAirport = airportMap.get(flight.arr_iata);
    const airlineName = airlineMap.get(flight.airline_iata) || flight.airline_iata;

    // Convert date from DD.MM.YYYY to DD-MM-YYYY
    const dateFormatted = flight.scheduled_flight_date.replace(/\./g, '-');

    return {
        airlineCode: flight.airline_iata,
        airlineName,
        flightNumber: flight.flight_number,
        originCode: flight.dep_iata,
        origin: originAirport?.city || flight.dep_iata,
        destinationCode: flight.arr_iata,
        destination: destAirport?.city || flight.arr_iata,
        date: dateFormatted,
        duration: flight.duration,
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
    return `You are a flight information research specialist. Your job: find specific flight details from trusted sources. NEVER guess or hallucinate.

CURRENT DATE/TIME (UTC): ${currentDate}
USER QUERY: ${userQuery}

═══════════════════════════════════════════════════════════════════
TASK: Find flight number, duration, and aircraft type
═══════════════════════════════════════════════════════════════════

SEARCH TARGETS (in priority order):
1. www.aviability.com - Best for duration (shown in snippets)
2. www.flightAware.com - Most reliable overall
3. www.flightRadar24.com - Good for aircraft types

SEARCH PROTOCOL:
1. Search for the route + airline + date combination
2. Look for EXPLICITLY STATED flight duration (e.g., "1h 30m", "Flight time: 01:30")
3. Find aircraft ICAO codes (e.g., "B738", "A320", "77W")
4. Identify flight number for this specific route/date

═══════════════════════════════════════════════════════════════════
MANDATORY REASONING (complete ALL steps before answering):
═══════════════════════════════════════════════════════════════════

STEP 1 - SEARCH FINDINGS:
List what you found from each source:
• Flight number: [value or "NOT FOUND"] from [source]
• Duration: [value or "NOT FOUND"] from [source]
• Aircraft code: [value or "NOT FOUND"] from [source]

STEP 2 - ROUTE VALIDATION:
✓ Confirmed route direction: [origin IATA] → [destination IATA]
✓ Date matches query: [YES/NO]

STEP 3 - AIRCRAFT CODE CONVERSION:
If you found an ICAO code, convert using the mapping table below.
Original code: [e.g., "B738"]
Converted name: [e.g., "Boeing 737NG"]

STEP 4 - AMBIGUITY CHECK:
• Multiple flights found? [YES/NO - if yes, list all flight numbers]
• Conflicting information? [YES/NO - if yes, describe conflict]
• Missing critical data? [List what's missing]

STEP 5 - DURATION VALIDATION:
• Is duration explicitly stated in search results? [YES/NO]
• Source that shows duration: [website name]
• If NOT found: Write "DURATION NOT FOUND" - do NOT calculate from times

STEP 6 - FINAL DECISION:
Selected flight number: [number]
Reason for selection: [if multiple found, explain why you chose this one]

═══════════════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════════

FLIGHT NUMBER: [number only, or "NOT FOUND"]
Source: [Website name]
Notes: [If multiple flights exist, note which one and why selected]

AIRLINE CODE: [IATA 2-letter code like "WN", or "NOT FOUND"]
Source: [Website name or "Provided in query"]

DEPARTURE AIRPORT: [IATA 3-letter code like "LAS", or "NOT FOUND"]
Source: [Website name or "Provided in query"]

ARRIVAL AIRPORT: [IATA 3-letter code like "ABQ", or "NOT FOUND"]
Source: [Website name or "Provided in query"]

FLIGHT DATE: [DD-MM-YYYY format, or "NOT FOUND"]
Source: [Website name or "Provided in query"]

FLIGHT TIME: [HH:MM format - ONLY from search results showing duration]
Source: [Website name that explicitly shows duration]
Notes: [CRITICAL - only fill if you found explicit duration. Write "NOT FOUND - no explicit duration in results" if you only see departure/arrival times]

AIRCRAFT TYPE: [Full name from mapping table below]
Source: [Website name]
Original Code: [ICAO code you found, e.g., "B738"]
Converted To: [Full name from mapping, e.g., "Boeing 737NG"]

═══════════════════════════════════════════════════════════════════
AIRCRAFT ICAO → FULL NAME MAPPING (use this table strictly):
═══════════════════════════════════════════════════════════════════
Boeing 737:  B738/B739/B73J/B737 → Boeing 737NG
             B38M → Boeing 737MAX
Boeing:      B712 → Boeing 717
             B753 → Boeing 757
             B77W/B77L/B772/B773 → Boeing 777
             B78X/B788/B789 → Boeing 787
             B744/B748 → Boeing 747
Airbus A320: A320/A20N → Airbus A320
             A321/A21N → Airbus A321
             A319/A19N → Airbus A319
Airbus Wide: A333/A332/A339 → Airbus A330
             A359/A35K → Airbus A350
             A388 → Airbus A380
Regional:    E75L/E75S → Embraer E175-E2
             E170 → Embraer E170
             E190/E290 → Embraer E190
             E195/E295 → Embraer E195-E2
             CRJ9/CRJ7/CRJ2 → Bombardier CRJ
             DH8D → DHC Dash 8
             AT76/AT72 → ATR 42/72

OVERALL ASSESSMENT:
[2-3 sentences: What was successfully found? What's missing? Any ambiguity?]
---

═══════════════════════════════════════════════════════════════════
CRITICAL RULES (violations will invalidate results):
═══════════════════════════════════════════════════════════════════
1. NO GUESSING: Write "NOT FOUND" if information is not in search results
2. NO CALCULATIONS: Only report duration if explicitly shown in results
3. NO ASSUMPTIONS: If multiple flights exist, note this in your reasoning
4. STRICT MAPPING: Only use aircraft names from the mapping table above
5. VERIFY DIRECTION: Ensure origin→destination matches the query

Begin your search now.`;
}

// Structuring prompt for Step 2
const structurePromptTemplate = `Extract structured data from this flight search report.

═══════════════════════════════════════════════════════════════════
SEARCH REPORT:
═══════════════════════════════════════════════════════════════════
{searchResults}

═══════════════════════════════════════════════════════════════════
EXTRACTION RULES:
═══════════════════════════════════════════════════════════════════

1. flightNumber: Extract ONLY the numeric part (e.g., "548" not "WN548")
   - If report says "NOT FOUND", set to null

2. airlineCode: IATA 2-letter code (e.g., "WN")
   - If report says "NOT FOUND", set to null

3. departureAirportCode: IATA 3-letter code (e.g., "LAS")
   - If report says "NOT FOUND", set to null

4. arrivalAirportCode: IATA 3-letter code (e.g., "ABQ")
   - If report says "NOT FOUND", set to null

5. flightDate: DD-MM-YYYY format exactly
   - If report says "NOT FOUND", set to null

6. flightTime: Duration in HH:MM format (e.g., "01:30")
   - ONLY extract if report explicitly found duration
   - If report says "NOT FOUND" or mentions only dep/arr times, set to null

7. aircraftName: Full aircraft name from the report (e.g., "Boeing 737NG")
   - Use the exact converted name from the mapping table
   - If report says "NOT FOUND", set to null

8. overallAssessment: Copy the OVERALL ASSESSMENT section from the report

Return valid JSON with these exact field names.`;

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

// Comparison and Evaluation Functions

/**
 * Convert time string to minutes
 * @param {string} timeStr - Time in HH:MM format
 * @returns {number} Total minutes
 */
function timeToMinutes(timeStr) {
    if (!timeStr || timeStr === 'null') return null;
    const [hours, mins] = timeStr.split(':').map(Number);
    return hours * 60 + mins;
}

/**
 * Get time difference in minutes
 * @param {string} time1 - First time in HH:MM format
 * @param {string} time2 - Second time in HH:MM format
 * @returns {number} Absolute difference in minutes
 */
function getTimeDiffMinutes(time1, time2) {
    const mins1 = timeToMinutes(time1);
    const mins2 = timeToMinutes(time2);

    if (mins1 === null || mins2 === null) return Infinity;

    return Math.abs(mins1 - mins2);
}

/**
 * Compare a single field and return match status and grade
 * @param {string} field - Field name
 * @param {any} extracted - Extracted value
 * @param {any} groundTruth - Ground truth value
 * @returns {Object} { match: boolean|null, grade: number }
 */
function compareField(field, extracted, groundTruth) {
    // Handle null/undefined extracted values
    if (!extracted || extracted === 'null' || extracted === null) {
        return { match: false, grade: 0.0 };
    }

    switch(field) {
        case 'flightNumber':
            // Excluded from scoring (too ambiguous with multiple flights per route)
            return { match: null, grade: null };

        case 'airlineCode':
        case 'departureAirportCode':
        case 'arrivalAirportCode':
        case 'flightDate':
            // Exact match required
            return {
                match: extracted === groundTruth,
                grade: extracted === groundTruth ? 1.0 : 0.0
            };

        case 'aircraftName':
            // Fuzzy matching: exact match or family match
            if (extracted === groundTruth) {
                return { match: true, grade: 1.0 };
            }
            if (isSameFamily(extracted, groundTruth)) {
                return { match: true, grade: 0.8 };
            }
            return { match: false, grade: 0.0 };

        case 'flightTime':
            // Tolerance-based: ±15 min = full credit, ±30 min = partial
            const diff = getTimeDiffMinutes(extracted, groundTruth);
            if (diff <= 15) {
                return { match: true, grade: 1.0 };
            }
            if (diff <= 30) {
                return { match: true, grade: 0.7 };
            }
            return { match: false, grade: 0.0 };

        default:
            return { match: false, grade: 0.0 };
    }
}

/**
 * Compare all fields for a result
 * @param {Object} extracted - Extracted data
 * @param {Object} groundTruth - Ground truth data
 * @returns {Object} Comparison results for each field
 */
function compareAllFields(extracted, groundTruth) {
    const fields = ['flightNumber', 'airlineCode', 'departureAirportCode',
                    'arrivalAirportCode', 'flightDate', 'aircraftName', 'flightTime'];

    const comparison = {};

    for (const field of fields) {
        comparison[field] = compareField(field, extracted[field], groundTruth[field]);
    }

    return comparison;
}

/**
 * Flag cases that need manual review
 * @param {Object} result - Evaluation result
 * @returns {Array} Array of flag strings
 */
function flagForReview(result) {
    const flags = [];

    const { extracted, groundTruth, comparison } = result;

    // Flag 1: Null aircraft but duration found (inconsistent)
    if ((!extracted.aircraftName || extracted.aircraftName === 'null') && extracted.flightTime) {
        flags.push('aircraft_missing');
    }

    // Flag 2: Wrong aircraft family (needs verification)
    if (comparison.aircraftName && comparison.aircraftName.grade === 0.0 && extracted.aircraftName) {
        flags.push('aircraft_mismatch');
    }

    // Flag 3: Duration off by >30min (significant error)
    if (extracted.flightTime && groundTruth.duration) {
        const durationDiff = getTimeDiffMinutes(extracted.flightTime, groundTruth.duration);
        if (durationDiff > 30) {
            flags.push('duration_error');
        }
    }

    // Flag 4: Multiple nulls (low data quality)
    const nullCount = Object.values(extracted).filter(v =>
        v === null || v === 'null' || v === undefined
    ).length;
    if (nullCount >= 3) {
        flags.push('low_quality');
    }

    return flags;
}

// Generate markdown report with automated metrics
function generateMarkdownReport(results, timestamp) {
    // Calculate metrics
    const fields = ['airlineCode', 'departureAirportCode', 'arrivalAirportCode',
                    'flightDate', 'aircraftName', 'flightTime'];
    const metrics = calculateAllMetrics(results, fields);
    const summary = getSummaryStats(results);

    // Count flagged cases
    const flaggedCases = results.filter(r => r.flags && r.flags.length > 0);

    let markdown = `# Flight Search Evaluation Report\n\n`;
    markdown += `**Generated:** ${new Date(timestamp).toISOString()}\n`;
    markdown += `**Model:** Pollinations.ai gemini-fast\n`;
    markdown += `**Architecture:** LangChain 2-step chain (search → structure)\n`;
    markdown += `**Total Flights:** ${results.length}\n`;
    markdown += `**Avg Duration:** ${(results.reduce((s, r) => s + parseFloat(r.duration || 0), 0) / results.length).toFixed(2)}s\n\n`;
    markdown += `---\n\n`;

    // Summary Metrics Section
    markdown += `## Summary Metrics\n\n`;
    markdown += `### Overall Accuracy\n\n`;
    markdown += `| Field | Extracted | Correct | Precision | Recall | F1 Score |\n`;
    markdown += `|-------|-----------|---------|-----------|--------|----------|\n`;
    markdown += `| Airline Code | ${metrics.airlineCode.extracted}/${metrics.airlineCode.total} | ${metrics.airlineCode.correct} | ${metrics.airlineCode.precision} | ${metrics.airlineCode.recall} | ${metrics.airlineCode.f1} |\n`;
    markdown += `| Departure Airport | ${metrics.departureAirportCode.extracted}/${metrics.departureAirportCode.total} | ${metrics.departureAirportCode.correct} | ${metrics.departureAirportCode.precision} | ${metrics.departureAirportCode.recall} | ${metrics.departureAirportCode.f1} |\n`;
    markdown += `| Arrival Airport | ${metrics.arrivalAirportCode.extracted}/${metrics.arrivalAirportCode.total} | ${metrics.arrivalAirportCode.correct} | ${metrics.arrivalAirportCode.precision} | ${metrics.arrivalAirportCode.recall} | ${metrics.arrivalAirportCode.f1} |\n`;
    markdown += `| Flight Date | ${metrics.flightDate.extracted}/${metrics.flightDate.total} | ${metrics.flightDate.correct} | ${metrics.flightDate.precision} | ${metrics.flightDate.recall} | ${metrics.flightDate.f1} |\n`;
    markdown += `| Aircraft Name | ${metrics.aircraftName.extracted}/${metrics.aircraftName.total} | ${metrics.aircraftName.correct} | ${metrics.aircraftName.precision} | ${metrics.aircraftName.recall} | ${metrics.aircraftName.f1} |\n`;
    markdown += `| Flight Duration | ${metrics.flightTime.extracted}/${metrics.flightTime.total} | ${metrics.flightTime.correct} | ${metrics.flightTime.precision} | ${metrics.flightTime.recall} | ${metrics.flightTime.f1} |\n\n`;
    markdown += `**Overall Weighted F1 Score:** ${metrics.overall.weightedF1}\n\n`;
    markdown += `**Note:** Flight numbers excluded from scoring (too ambiguous with multiple flights per route).\n`;
    markdown += `Query-provided fields (airline, airports, date) weighted 0.5x, searched fields (aircraft, duration) weighted 1.5x.\n\n`;

    markdown += `### Summary Statistics\n\n`;
    markdown += `- **Perfect Matches:** ${summary.perfectMatches}/${summary.totalFlights} (all fields correct)\n`;
    markdown += `- **With Data:** ${summary.withData}/${summary.totalFlights} (at least one field extracted)\n`;
    markdown += `- **Flagged for Review:** ${summary.flaggedCount}/${summary.totalFlights}\n`;
    markdown += `- **Average Grade:** ${summary.avgGrade}\n\n`;
    markdown += `---\n\n`;

    // Individual Flight Results
    markdown += `## Detailed Results\n\n`;

    results.forEach((result, idx) => {
        const { query, groundTruth, extracted, comparison, flags, duration } = result;

        // Add flag indicator in title
        const flagIndicator = flags && flags.length > 0 ? ' 🔍' : '';
        markdown += `## ${idx + 1}. ${query}${flagIndicator}\n\n`;
        markdown += `**Duration:** ${duration}s`;
        if (flags && flags.length > 0) {
            markdown += ` | **Flags:** ${flags.join(', ')}`;
        }
        markdown += `\n\n`;

        // Helper to get match emoji
        const getMatchEmoji = (comp) => {
            if (!comp) return '⬜';
            if (comp.match === null) return '➖'; // Excluded from scoring
            if (comp.match === true) {
                if (comp.grade === 1.0) return '✅';
                if (comp.grade >= 0.7) return '⚠️'; // Partial match
                return '✅';
            }
            return '❌';
        };

        // Helper to get grade display
        const getGradeDisplay = (comp) => {
            if (!comp || comp.grade === null) return '';
            return ` (${(comp.grade * 100).toFixed(0)}%)`;
        };

        markdown += `| Field | Ground Truth | Extracted | Match | Grade |\n`;
        markdown += `|-------|--------------|-----------|-------|-------|\n`;
        markdown += `| Flight Number | ${groundTruth.airlineCode}${groundTruth.flightNumber} | ${extracted.airlineCode || '?'}${extracted.flightNumber || '?'} | ${getMatchEmoji(comparison.flightNumber)} | N/A |\n`;
        markdown += `| Airline | ${groundTruth.airlineCode} | ${extracted.airlineCode || 'null'} | ${getMatchEmoji(comparison.airlineCode)} | ${comparison.airlineCode.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Departure | ${groundTruth.originCode} | ${extracted.departureAirportCode || 'null'} | ${getMatchEmoji(comparison.departureAirportCode)} | ${comparison.departureAirportCode.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Arrival | ${groundTruth.destinationCode} | ${extracted.arrivalAirportCode || 'null'} | ${getMatchEmoji(comparison.arrivalAirportCode)} | ${comparison.arrivalAirportCode.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Date | ${groundTruth.date} | ${extracted.flightDate || 'null'} | ${getMatchEmoji(comparison.flightDate)} | ${comparison.flightDate.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Aircraft | ${mapAircraftCode(groundTruth.aircraft)} | ${extracted.aircraftName || 'null'} | ${getMatchEmoji(comparison.aircraftName)} | ${comparison.aircraftName.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Duration | ${groundTruth.duration} | ${extracted.flightTime || 'null'} | ${getMatchEmoji(comparison.flightTime)} | ${comparison.flightTime.grade?.toFixed(1) || '0.0'} |\n\n`;

        if (extracted.overallAssessment) {
            markdown += `**Assessment:** ${extracted.overallAssessment}\n\n`;
        }

        markdown += `<details>\n<summary>View Extracted JSON</summary>\n\n\`\`\`json\n${JSON.stringify(extracted, null, 2)}\n\`\`\`\n</details>\n\n`;
        markdown += `---\n\n`;
    });

    // Flagged Cases Section
    if (flaggedCases.length > 0) {
        markdown += `## Cases Flagged for Manual Review\n\n`;
        markdown += `**Total Flagged:** ${flaggedCases.length}/${results.length}\n\n`;

        const flagCategories = {
            'aircraft_missing': [],
            'aircraft_mismatch': [],
            'duration_error': [],
            'low_quality': []
        };

        flaggedCases.forEach((result, idx) => {
            result.flags.forEach(flag => {
                if (flagCategories[flag]) {
                    flagCategories[flag].push({ result, idx });
                }
            });
        });

        for (const [flagType, cases] of Object.entries(flagCategories)) {
            if (cases.length > 0) {
                const flagTitles = {
                    'aircraft_missing': 'Aircraft Missing (but duration found)',
                    'aircraft_mismatch': 'Aircraft Mismatch',
                    'duration_error': 'Duration Error (>30min off)',
                    'low_quality': 'Low Quality (3+ null fields)'
                };

                markdown += `### ${flagTitles[flagType]}\n\n`;
                markdown += `**Count:** ${cases.length}\n\n`;

                cases.forEach(({ result, idx }) => {
                    const resultIdx = results.indexOf(result) + 1;
                    markdown += `- **Flight ${resultIdx}:** ${result.query}\n`;
                    markdown += `  - Ground Truth Aircraft: ${mapAircraftCode(result.groundTruth.aircraft)}, Extracted: ${result.extracted.aircraftName || 'null'}\n`;
                    markdown += `  - Ground Truth Duration: ${result.groundTruth.duration}, Extracted: ${result.extracted.flightTime || 'null'}\n`;
                });

                markdown += `\n`;
            }
        }
    }

    markdown += `## Scoring Legend\n\n`;
    markdown += `- ✅ = Correct match (grade 1.0 or family match 0.8)\n`;
    markdown += `- ⚠️ = Partial match (e.g., duration within ±30min = 0.7)\n`;
    markdown += `- ❌ = Wrong or NOT FOUND (grade 0.0)\n`;
    markdown += `- ➖ = Excluded from scoring (flight numbers)\n`;
    markdown += `- 🔍 = Flagged for manual review\n\n`;
    markdown += `**Grading System:**\n`;
    markdown += `- Aircraft: 1.0 = exact match, 0.8 = same family (e.g., Boeing 737NG ≈ Boeing 737MAX)\n`;
    markdown += `- Duration: 1.0 = within ±15min, 0.7 = within ±30min, 0.0 = >30min off\n`;
    markdown += `- Other fields: 1.0 = exact match, 0.0 = wrong\n`;

    return markdown;
}

// Generate Label Studio JSON format for manual annotation
// Format: https://labelstud.io/templates/tabular_data
function generateLabelStudioJSON(results, timestamp) {
    const labelStudioData = results.map((result, idx) => {
        const { query, groundTruth, extracted, comparison, flags, duration } = result;

        return {
            data: {
                item: {
                    "1-GT-Route": `${groundTruth.originCode}-${groundTruth.destinationCode}`,
                    "2-ACT-Route": `${extracted.departureAirportCode || '?'}-${extracted.arrivalAirportCode || '?'}`,
                    "3-GT-Airline": groundTruth.airlineCode,
                    "4-ACT-Airline": extracted.airlineCode || 'null',
                    "5-GT-FlightNumber": `${groundTruth.airlineCode}${groundTruth.flightNumber}`,
                    "6-ACT-FlightNumber": `${extracted.airlineCode || '?'}${extracted.flightNumber || '?'}`,
                    "7-GT-Date": groundTruth.date,
                    "8-ACT-Date": extracted.flightDate || 'null',
                    "9-GT-Time": groundTruth.duration,
                    "10-ACT-Time": extracted.flightTime || 'null',
                    "11-GT-ACFT": mapAircraftCode(groundTruth.aircraft),
                    "12-ACT-ACFT": extracted.aircraftName || 'null'
                }
            }
        };
    });

    return JSON.stringify(labelStudioData, null, 2);
}

// Main evaluation function
async function runEvaluation() {
    // Get sample size from command line argument (default: all flights)
    const sampleSize = parseInt(process.argv[2]) || DATASET.length;
    const testDataset = DATASET.slice(0, sampleSize);

    console.log('🚀 Flight Search Agent Evaluation - LangChain + Pollinations');
    console.log(`📊 Testing ${testDataset.length} flights\n`);

    const results = [];
    let totalDuration = 0;

    for (let i = 0; i < testDataset.length; i++) {
        const testCase = testDataset[i];
        const query = generateQuery(testCase);

        console.log(`[${i + 1}/${testDataset.length}] ${query}`);

        try {
            const startTime = Date.now();

            // Create and invoke chain
            const chain = createChain(query);
            const extracted = await chain.invoke({ query });

            const duration = ((Date.now() - startTime) / 1000).toFixed(2);
            totalDuration += parseFloat(duration);

            // Prepare ground truth data for comparison
            const groundTruthData = {
                flightNumber: testCase.flightNumber,
                airlineCode: testCase.airlineCode,
                departureAirportCode: testCase.originCode,
                arrivalAirportCode: testCase.destinationCode,
                flightDate: testCase.date,
                aircraftName: mapAircraftCode(testCase.aircraft), // Map ICAO to full name
                duration: testCase.duration
            };

            // Compare extracted vs ground truth
            const comparison = compareAllFields(extracted, groundTruthData);

            // Prepare result object for flagging
            const result = {
                query,
                groundTruth: testCase,
                extracted,
                comparison,
                duration
            };

            // Flag cases for manual review
            const flags = flagForReview(result);
            result.flags = flags;

            results.push(result);

            console.log(`  ✓ Completed in ${duration}s`);

        } catch (error) {
            console.error(`  ✗ Error: ${error.message}`);

            const extracted = {
                flightNumber: null,
                airlineCode: null,
                departureAirportCode: null,
                arrivalAirportCode: null,
                flightDate: null,
                flightTime: null,
                aircraftName: null,
                overallAssessment: `Error: ${error.message}`
            };

            const groundTruthData = {
                flightNumber: testCase.flightNumber,
                airlineCode: testCase.airlineCode,
                departureAirportCode: testCase.originCode,
                arrivalAirportCode: testCase.destinationCode,
                flightDate: testCase.date,
                aircraftName: mapAircraftCode(testCase.aircraft), // Map ICAO to full name
                duration: testCase.duration
            };

            const comparison = compareAllFields(extracted, groundTruthData);
            const result = {
                query,
                groundTruth: testCase,
                extracted,
                comparison,
                duration: 0
            };
            result.flags = flagForReview(result);

            results.push(result);
        }
    }

    // Generate markdown report
    const timestamp = Date.now();
    const markdown = generateMarkdownReport(results, timestamp);
    const outputPath = `./eval-results-${timestamp}.md`;
    fs.writeFileSync(outputPath, markdown);

    // Generate Label Studio JSON export
    const labelStudioJSON = generateLabelStudioJSON(results, timestamp);
    const jsonOutputPath = `./eval-results-${timestamp}.json`;
    fs.writeFileSync(jsonOutputPath, labelStudioJSON);

    // Calculate and display summary metrics
    const fields = ['airlineCode', 'departureAirportCode', 'arrivalAirportCode',
                    'flightDate', 'aircraftName', 'flightTime'];
    const metrics = calculateAllMetrics(results, fields);
    const summary = getSummaryStats(results);

    const avgDuration = (totalDuration / results.length).toFixed(1);
    console.log(`\n✅ Completed ${results.length} tests in ${totalDuration.toFixed(1)}s (avg ${avgDuration}s/test)`);
    console.log(`\n📊 Summary Metrics:`);
    console.log(`   Overall Weighted F1: ${metrics.overall.weightedF1}`);
    console.log(`   Perfect Matches: ${summary.perfectMatches}/${summary.totalFlights}`);
    console.log(`   Flagged for Review: ${summary.flaggedCount}/${summary.totalFlights}`);
    console.log(`   Average Grade: ${summary.avgGrade}`);
    console.log(`\n📈 Per-Field F1 Scores:`);
    console.log(`   Airline: ${metrics.airlineCode.f1}`);
    console.log(`   Departure: ${metrics.departureAirportCode.f1}`);
    console.log(`   Arrival: ${metrics.arrivalAirportCode.f1}`);
    console.log(`   Date: ${metrics.flightDate.f1}`);
    console.log(`   Aircraft: ${metrics.aircraftName.f1} (${metrics.aircraftName.correct}/${metrics.aircraftName.total} correct)`);
    console.log(`   Duration: ${metrics.flightTime.f1} (${metrics.flightTime.correct}/${metrics.flightTime.total} correct)`);
    console.log(`\n📄 Reports generated:`);
    console.log(`   Markdown: ${outputPath}`);
    console.log(`   Label Studio JSON: ${jsonOutputPath}\n`);
}

runEvaluation().catch(console.error);
