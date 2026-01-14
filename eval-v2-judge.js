/**
 * Flight Search Agent Evaluation - V2 with LLM Judge
 *
 * Architecture:
 * - LangChain 3-step chain (search → extract → validate)
 * - Step 1: gemini-fast → web search, text report
 * - Step 2: gemini-fast + Zod → structured extraction
 * - Step 3: gemini-fast + Zod → validation with judge
 * - Output: Markdown + Label Studio JSON
 *
 * Usage: node eval-v2-judge.js [sample_size]
 */

import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';
import { z } from 'zod';
import fs from 'fs';
import { config } from 'dotenv';
import { isSameFamily } from './lib/aircraft-utils.js';

config();

// Configuration
const POLLINATIONS_API_KEY = process.env.POLLINATIONS_API_KEY || 'sk_szglh5aFfm2I1aiXvZEoXYvKDDGVOOl1';
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

// ICAO to full aircraft name mapping
const AIRCRAFT_MAPPING = {
    'B738': 'Boeing 737NG', 'B739': 'Boeing 737NG', 'B73J': 'Boeing 737NG', 'B737': 'Boeing 737NG',
    'B38M': 'Boeing 737MAX',
    'B712': 'Boeing 717',
    'B753': 'Boeing 757',
    'B77W': 'Boeing 777', 'B77L': 'Boeing 777', 'B772': 'Boeing 777', 'B773': 'Boeing 777',
    'B78X': 'Boeing 787', 'B788': 'Boeing 787', 'B789': 'Boeing 787',
    'B744': 'Boeing 747', 'B748': 'Boeing 747',
    'A320': 'Airbus A320', 'A20N': 'Airbus A320',
    'A321': 'Airbus A321', 'A21N': 'Airbus A321',
    'A319': 'Airbus A319', 'A19N': 'Airbus A319',
    'A333': 'Airbus A330', 'A332': 'Airbus A330', 'A339': 'Airbus A330',
    'A359': 'Airbus A350', 'A35K': 'Airbus A350',
    'A388': 'Airbus A380',
    'E75L': 'Embraer E175-E2', 'E75S': 'Embraer E175-E2',
    'E170': 'Embraer E170',
    'E190': 'Embraer E190', 'E290': 'Embraer E190',
    'E195': 'Embraer E195-E2', 'E295': 'Embraer E195-E2',
    'CRJ9': 'Bombardier CRJ', 'CRJ7': 'Bombardier CRJ', 'CRJ2': 'Bombardier CRJ',
    'DH8D': 'DHC Dash 8',
    'AT76': 'ATR 42/72', 'AT72': 'ATR 42/72'
};

function mapAircraftCode(icaoCode) {
    return AIRCRAFT_MAPPING[icaoCode] || icaoCode;
}

// Build dataset
const DATASET = simpleDataset.map(flight => {
    const originAirport = airportMap.get(flight.dep_iata);
    const destAirport = airportMap.get(flight.arr_iata);
    const airlineName = airlineMap.get(flight.airline_iata) || flight.airline_iata;
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

// Zod Schemas
const ExtractionSchema = z.object({
    flightNumber: z.string().nullable(),
    airlineCode: z.string().nullable(),
    departureAirportCode: z.string().nullable(),
    arrivalAirportCode: z.string().nullable(),
    flightDate: z.string().nullable(),
    flightTime: z.string().nullable(),
    aircraftName: z.string().nullable()
});

const ValidationSchema = z.object({
    validationStatus: z.enum(['PASS', 'FAIL']),
    consistencyScore: z.number().min(0).max(1),
    consistencyIssues: z.array(z.string()),
    sensibilityScore: z.number().min(0).max(1),
    sensibilityIssues: z.array(z.string()),
    confidenceScore: z.number().min(0).max(1),
    hallucinationDetected: z.boolean(),
    hallucinationDetails: z.array(z.string()),
    overallQualityScore: z.number().min(0).max(1),
    reasoning: z.string()
});

// Prompts
function getSearchPrompt(userQuery) {
    const currentDate = new Date().toISOString();
    return `You are a flight information research specialist. Find specific flight details from trusted sources. NEVER guess or hallucinate.

CURRENT DATE/TIME (UTC): ${currentDate}
USER QUERY: ${userQuery}

TASK: Find flight number, duration, and aircraft type

SEARCH TARGETS (priority order):
1. www.aviability.com - Best for duration
2. www.flightAware.com - Most reliable overall
3. www.flightRadar24.com - Good for aircraft types

MANDATORY REASONING (complete ALL steps):

STEP 1 - GENERATE FLIGHT NUMBER SEARCH QUERY:
Create search query from user query: "[airline] [origin] to [destination] [date]"
Example: "Southwest LAS to ABQ December 16 2025"
Execute this search to find a flight number candidate.

STEP 2 - IDENTIFY FLIGHT NUMBER CANDIDATE:
• Flight number found: [e.g., "WN548" or "548"] from [source]
• If multiple flights: [list all, note which you'll use]
• If NO flight found: Write "NOT FOUND" and STOP

STEP 3 - SEARCH FOR MISSING DATA (using flight number):
Search specifically for: "[airline] [flight number] aircraft duration"
• Aircraft: [ICAO code or full name, or "NOT FOUND"] from [source]
• Duration: [HH:MM or "NOT FOUND"] from [source]

STEP 4 - ROUTE VALIDATION:
✓ Route: [origin IATA] → [destination IATA] matches query? [YES/NO]
✓ Date matches? [YES/NO]
✓ Airline matches? [YES/NO]

STEP 5 - DURATION VALIDATION:
• Duration explicitly stated? [YES/NO]
• Source: [website]
• If NOT found: Write "DURATION NOT FOUND" - do NOT calculate from times

STEP 6 - FINAL DECISION:
• Confirmed flight: [number]
• Confidence: [High/Medium/Low]

REQUIRED OUTPUT FORMAT:

FLIGHT NUMBER: [number only, or "NOT FOUND"]
Source: [Website]
Notes: [If multiple, explain selection]

AIRLINE CODE: [2-letter IATA like "WN", or "NOT FOUND"]
Source: [Website or "Provided in query"]

DEPARTURE AIRPORT: [3-letter IATA like "LAS", or "NOT FOUND"]
Source: [Website or "Provided in query"]

ARRIVAL AIRPORT: [3-letter IATA like "ABQ", or "NOT FOUND"]
Source: [Website or "Provided in query"]

FLIGHT DATE: [DD-MM-YYYY, or "NOT FOUND"]
Source: [Website or "Provided in query"]

FLIGHT TIME: [HH:MM duration format, or "NOT FOUND"]
Source: [Website showing duration]
Notes: [Only if explicit duration found]

AIRCRAFT TYPE: [ICAO code or full name, or "NOT FOUND"]
Source: [Website]
Notes: [Report as found, don't convert]

OVERALL ASSESSMENT: [2-3 sentences: what found, what missing, any ambiguity]

CRITICAL RULES:
1. NO GUESSING: Write "NOT FOUND" if not in search results
2. NO CALCULATIONS: Only report duration if explicitly shown
3. TWO-STAGE SEARCH: Find flight number FIRST, then aircraft/duration
4. REPORT AS-IS: Don't convert aircraft codes
5. VERIFY DIRECTION: origin→destination must match query

Begin your search now.`;
}

const extractionPromptTemplate = `Extract structured flight data from the search report below.

SEARCH REPORT:
{searchResults}

EXTRACTION RULES:
1. Use null for any field marked "NOT FOUND"
2. Flight numbers: EXTRACT DIGITS ONLY
   Algorithm: Identify flight number text → strip ALL letters/symbols → keep only digits
   Examples:
   - "WN548" → "548"
   - "9C8510" → "8510"
   - "9C9C8510" → "8510" (strip duplicate prefix)
   - "UA 1234" → "1234"
   - "548" → "548"
3. Dates: DD-MM-YYYY format
   Convert if needed: "16 Dec 2025" → "16-12-2025"
4. Duration: HH:MM format
   Convert if needed: "1h 30m" → "01:30"
5. Aircraft: Convert ICAO codes to full names using mapping below
   If code not in mapping, keep as-is

AIRCRAFT ICAO → FULL NAME MAPPING:
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

OUTPUT: JSON with 7 fields:
- flightNumber (string or null)
- airlineCode (string or null)
- departureAirportCode (string or null)
- arrivalAirportCode (string or null)
- flightDate (string or null)
- flightTime (string or null)
- aircraftName (string or null)

CRITICAL: Never guess. If report says "NOT FOUND", use null.`;

const validationPromptTemplate = `You are a RULE-FOLLOWING VALIDATOR executing mechanical quality checks.

CRITICAL INSTRUCTION:
Your ONLY job is to execute the validation logic below exactly as specified.
DO NOT add your own reasoning, judgment calls, or additional failure conditions.
DO NOT interpret or modify the rules.
DO provide reasoning that ONLY explains which condition triggered the result.

USER QUERY: {query}

SEARCH REPORT:
{searchResults}

EXTRACTED DATA:
{extractedJSON}

YOUR TASK: Parse the user query to identify what they requested, then check if extracted data matches.

PERFORM 4 VALIDATION CHECKS:

CHECK 1 - QUERY CONSISTENCY (STRICT - MUST BE 100%):
Parse the user query to extract:
1. Origin city → convert to IATA code (e.g., "Las Vegas" → "LAS")
2. Destination city → convert to IATA code (e.g., "Albuquerque" → "ABQ")
3. Airline name (e.g., "Southwest Airlines")
4. Date → convert to DD-MM-YYYY format (e.g., "16 Dec 2025" → "16-12-2025")

Then compare:
- Route: Does extracted departureAirportCode match query origin? (MUST be exact)
- Route: Does extracted arrivalAirportCode match query destination? (MUST be exact)
- Airline: Does extracted airlineCode correspond to the airline name in query?
  Use your knowledge of airline codes (Southwest Airlines = WN, United = UA, etc.)
  PASS if codes match the airline mentioned, FAIL if different airline
- Date: Does extracted flightDate match query date? (MUST be exact)

Score:
- consistencyScore = 1.0 ONLY if ALL four checks pass
- consistencyScore = 0.0 if ANY check fails

List ALL mismatches in consistencyIssues array.
Example: "Airline mismatch: query requests Southwest Airlines but extracted United (UA)"

CHECK 2 - SENSIBILITY (reasonableness of searched data):
Duration and aircraft are NOT in query, so we assess plausibility only:

Duration ranges by distance:
- Short-haul (<500mi): 0:30-2:00
- Medium (500-1500mi): 1:30-5:00
- Long-haul (1500-4000mi): 4:00-10:00
- Ultra-long (4000+mi): 8:00-18:00

Aircraft appropriate for route type and airline?

Score 0.0-1.0:
- 1.0 = completely reasonable
- 0.7 = questionable but possible
- 0.0 = impossible (duration >2x expected or wrong aircraft type)

List any issues in sensibilityIssues array.

CHECK 3 - CONFIDENCE (data completeness):
Calculate using this formula:

1. Count non-null fields in extractedJSON: X out of 7
2. Assess source quality from search report:
   - FlightAware / Official airline website → 1.0
   - FlightRadar24 / Aviability → 0.9
   - Generic search results → 0.7
   - No clear source attribution → 0.5
3. Calculate: confidenceScore = (X / 7) * 0.6 + (sourceQuality * 0.4)
4. Round to 2 decimal places

Example: 5 fields found, FlightAware source
confidenceScore = (5/7) * 0.6 + (1.0 * 0.4) = 0.43 + 0.40 = 0.83

CHECK 4 - HALLUCINATION (data fabrication):
For each non-null extracted field:
- Does search report say "NOT FOUND" for that field?
- If YES → HALLUCINATION DETECTED

List hallucinated fields: "field: search said NOT FOUND but extracted as [value]"
Set hallucinationDetected true/false.

VALIDATION DECISION (STRICT - FOLLOW EXACTLY):
Use this EXACT logic - do NOT add your own reasoning:

IF consistencyScore < 1.0 → validationStatus = "FAIL"
ELSE IF hallucinationDetected = true → validationStatus = "FAIL"
ELSE IF sensibilityScore = 0.0 → validationStatus = "FAIL"
ELSE → validationStatus = "PASS"

EXAMPLES:
✅ PASS: consistencyScore=1.0, sensibilityScore=1.0, confidenceScore=0.4
   Reason: Query fields match perfectly, data sensible, PASS despite low confidence
✅ PASS: consistencyScore=1.0, sensibilityScore=0.8, confidenceScore=0.6, flight#/duration/aircraft=null
   Reason: Query fields match, data plausible, PASS despite missing searched fields
❌ FAIL: consistencyScore=0.0, sensibilityScore=1.0, confidenceScore=1.0
   Reason: Query fields mismatch, FAIL regardless of other scores
❌ FAIL: consistencyScore=1.0, sensibilityScore=0.0, confidenceScore=1.0
   Reason: Completely impossible data, FAIL regardless of consistency

CRITICAL: Low confidence or missing searched fields do NOT cause failure.
ONLY fail if: (1) query fields mismatch, (2) hallucination, (3) completely impossible data.

overallQualityScore:
- 0.0 if validationStatus = "FAIL"
- (consistencyScore + sensibilityScore + confidenceScore) / 3 if validationStatus = "PASS"

OUTPUT JSON:
- validationStatus ("PASS" or "FAIL")
- consistencyScore (0.0 or 1.0 ONLY - no partial credit)
- consistencyIssues (array of specific mismatches)
- sensibilityScore (0.0-1.0)
- sensibilityIssues (array)
- confidenceScore (0.0-1.0)
- hallucinationDetected (boolean)
- hallucinationDetails (array)
- overallQualityScore (0.0-1.0)
- reasoning (string, 2-3 sentences explaining pass/fail)`;

// Initialize LLMs
const searchLLM = new ChatOpenAI({
    modelName: 'gemini-fast',
    configuration: { baseURL: POLLINATIONS_URL, apiKey: POLLINATIONS_API_KEY },
    temperature: 0,
    timeout: 60000  // 60 second timeout
});

const extractLLM = new ChatOpenAI({
    modelName: 'gemini-fast',
    configuration: { baseURL: POLLINATIONS_URL, apiKey: POLLINATIONS_API_KEY },
    temperature: 0,
    timeout: 60000
}).withStructuredOutput(ExtractionSchema);

const validateLLM = new ChatOpenAI({
    modelName: 'gemini-fast',
    configuration: { baseURL: POLLINATIONS_URL, apiKey: POLLINATIONS_API_KEY },
    temperature: 0,
    timeout: 60000
}).withStructuredOutput(ValidationSchema);

// Create 3-step chain
function createChain(query) {
    const searchPrompt = PromptTemplate.fromTemplate(getSearchPrompt(query));
    const extractPrompt = PromptTemplate.fromTemplate(extractionPromptTemplate);
    const validatePrompt = PromptTemplate.fromTemplate(validationPromptTemplate);

    return RunnableSequence.from([
        // Step 1: Search
        {
            searchResults: searchPrompt.pipe(searchLLM).pipe(new StringOutputParser())
        },
        // Step 2: Extract
        async (input) => {
            const extracted = await extractPrompt
                .pipe(extractLLM)
                .invoke({ searchResults: input.searchResults });
            return {
                searchResults: input.searchResults,
                extracted
            };
        },
        // Step 3: Validate
        async (input) => {
            const validation = await validatePrompt
                .pipe(validateLLM)
                .invoke({
                    query,
                    searchResults: input.searchResults,
                    extractedJSON: JSON.stringify(input.extracted, null, 2)
                });
            return {
                searchResults: input.searchResults,
                extracted: input.extracted,
                validation
            };
        }
    ]);
}

// Comparison functions
function timeToMinutes(timeStr) {
    if (!timeStr || timeStr === 'null') return null;
    const [hours, mins] = timeStr.split(':').map(Number);
    return hours * 60 + mins;
}

function compareField(field, extracted, groundTruth) {
    if (!extracted || extracted === 'null' || extracted === null) {
        return { match: false, grade: 0.0 };
    }

    switch(field) {
        case 'flightNumber':
            return { match: null, grade: null }; // Excluded from scoring

        case 'airlineCode':
        case 'departureAirportCode':
        case 'arrivalAirportCode':
        case 'flightDate':
            return {
                match: extracted === groundTruth,
                grade: extracted === groundTruth ? 1.0 : 0.0
            };

        case 'aircraftName':
            if (extracted === groundTruth) {
                return { match: true, grade: 1.0 };
            }
            if (isSameFamily(extracted, groundTruth)) {
                return { match: true, grade: 0.8 };
            }
            return { match: false, grade: 0.0 };

        case 'flightTime':
            const mins1 = timeToMinutes(extracted);
            const mins2 = timeToMinutes(groundTruth);
            if (mins1 === null || mins2 === null) return { match: false, grade: 0.0 };

            const diff = Math.abs(mins1 - mins2);
            if (diff <= 15) return { match: true, grade: 1.0 };
            if (diff <= 30) return { match: true, grade: 0.7 };
            return { match: false, grade: 0.0 };

        default:
            return { match: false, grade: 0.0 };
    }
}

function compareAllFields(extracted, groundTruth) {
    const fields = ['flightNumber', 'airlineCode', 'departureAirportCode',
                    'arrivalAirportCode', 'flightDate', 'aircraftName', 'flightTime'];

    const comparison = {};
    for (const field of fields) {
        comparison[field] = compareField(field, extracted[field], groundTruth[field]);
    }

    return comparison;
}

// Generate Label Studio JSON
function generateLabelStudioJSON(results) {
    return results.map((result) => {
        const { query, groundTruth, extracted, validation } = result;

        const combine = (gt, act) => `GT: ${gt || 'null'} || ACT: ${act || 'null'}`;

        return {
            data: {
                item: {
                    "1-Route": combine(
                        `${groundTruth.originCode}-${groundTruth.destinationCode}`,
                        `${extracted.departureAirportCode || '?'}-${extracted.arrivalAirportCode || '?'}`
                    ),
                    "2-Airline": combine(
                        groundTruth.airlineCode,
                        extracted.airlineCode
                    ),
                    "3-FlightNumber": combine(
                        `${groundTruth.airlineCode}${groundTruth.flightNumber}`,
                        `${extracted.airlineCode || '?'}${extracted.flightNumber || '?'}`
                    ),
                    "4-Date": combine(
                        groundTruth.date,
                        extracted.flightDate
                    ),
                    "5-Duration": combine(
                        groundTruth.duration,
                        extracted.flightTime
                    ),
                    "6-Aircraft": combine(
                        mapAircraftCode(groundTruth.aircraft),
                        extracted.aircraftName
                    ),
                    "7-ValidationStatus": validation.validationStatus,
                    "8-QualityScore": validation.overallQualityScore.toFixed(2),
                    "9-ValidationReasoning": validation.reasoning
                }
            }
        };
    });
}

// Generate Markdown report
function generateMarkdownReport(results, timestamp) {
    let markdown = `# Flight Search Evaluation Report - V2 with Judge\n\n`;
    markdown += `**Generated:** ${new Date(timestamp).toISOString()}\n`;
    markdown += `**Model:** Pollinations.ai gemini-fast\n`;
    markdown += `**Architecture:** LangChain 3-step chain (search → extract → validate)\n`;
    markdown += `**Total Flights:** ${results.length}\n\n`;
    markdown += `---\n\n`;

    // Validation Summary
    const passedValidation = results.filter(r => r.validation.validationStatus === 'PASS').length;
    const consistencyFailures = results.filter(r => r.validation.consistencyIssues.length > 0).length;
    const sensibilityFailures = results.filter(r => r.validation.sensibilityIssues.length > 0).length;
    const hallucinations = results.filter(r => r.validation.hallucinationDetected).length;

    const avgConsistency = results.reduce((sum, r) => sum + r.validation.consistencyScore, 0) / results.length;
    const avgSensibility = results.reduce((sum, r) => sum + r.validation.sensibilityScore, 0) / results.length;
    const avgConfidence = results.reduce((sum, r) => sum + r.validation.confidenceScore, 0) / results.length;
    const avgQuality = results.reduce((sum, r) => sum + r.validation.overallQualityScore, 0) / results.length;

    markdown += `## Validation Summary\n\n`;
    markdown += `**Judge Validation Pass Rate:** ${passedValidation}/${results.length} (${(passedValidation/results.length*100).toFixed(1)}%)\n\n`;
    markdown += `### Validation Failures\n`;
    markdown += `- Consistency Failures: ${consistencyFailures}\n`;
    markdown += `- Sensibility Failures: ${sensibilityFailures}\n`;
    markdown += `- Hallucinations Detected: ${hallucinations}\n\n`;
    markdown += `### Average Scores\n`;
    markdown += `- Consistency: ${avgConsistency.toFixed(2)}\n`;
    markdown += `- Sensibility: ${avgSensibility.toFixed(2)}\n`;
    markdown += `- Confidence: ${avgConfidence.toFixed(2)}\n`;
    markdown += `- Overall Quality: ${avgQuality.toFixed(2)}\n\n`;
    markdown += `---\n\n`;

    // Individual results
    markdown += `## Detailed Results\n\n`;

    results.forEach((result, idx) => {
        const { query, groundTruth, extracted, validation, comparison, duration } = result;

        const validationEmoji = validation.validationStatus === 'PASS' ? '✅' : '❌';
        markdown += `### ${idx + 1}. ${query} ${validationEmoji}\n\n`;
        markdown += `**Duration:** ${duration}s | **Validation:** ${validation.validationStatus} | **Quality:** ${(validation.overallQualityScore * 100).toFixed(0)}%\n\n`;

        const getMatchEmoji = (comp) => {
            if (!comp) return '⬜';
            if (comp.match === null) return '➖';
            if (comp.match === true) {
                if (comp.grade === 1.0) return '✅';
                if (comp.grade >= 0.7) return '⚠️';
                return '✅';
            }
            return '❌';
        };

        markdown += `| Field | Ground Truth | Extracted | Match | Grade |\n`;
        markdown += `|-------|--------------|-----------|-------|-------|\n`;
        markdown += `| Flight Number | ${groundTruth.airlineCode}${groundTruth.flightNumber} | ${extracted.airlineCode || '?'}${extracted.flightNumber || '?'} | ${getMatchEmoji(comparison.flightNumber)} | N/A |\n`;
        markdown += `| Airline | ${groundTruth.airlineCode} | ${extracted.airlineCode || 'null'} | ${getMatchEmoji(comparison.airlineCode)} | ${comparison.airlineCode?.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Departure | ${groundTruth.originCode} | ${extracted.departureAirportCode || 'null'} | ${getMatchEmoji(comparison.departureAirportCode)} | ${comparison.departureAirportCode?.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Arrival | ${groundTruth.destinationCode} | ${extracted.arrivalAirportCode || 'null'} | ${getMatchEmoji(comparison.arrivalAirportCode)} | ${comparison.arrivalAirportCode?.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Date | ${groundTruth.date} | ${extracted.flightDate || 'null'} | ${getMatchEmoji(comparison.flightDate)} | ${comparison.flightDate?.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Aircraft | ${mapAircraftCode(groundTruth.aircraft)} | ${extracted.aircraftName || 'null'} | ${getMatchEmoji(comparison.aircraftName)} | ${comparison.aircraftName?.grade?.toFixed(1) || '0.0'} |\n`;
        markdown += `| Duration | ${groundTruth.duration} | ${extracted.flightTime || 'null'} | ${getMatchEmoji(comparison.flightTime)} | ${comparison.flightTime?.grade?.toFixed(1) || '0.0'} |\n\n`;

        markdown += `**Judge Validation:**\n`;
        markdown += `- Consistency: ${(validation.consistencyScore * 100).toFixed(0)}%`;
        if (validation.consistencyIssues.length > 0) {
            markdown += ` ⚠️ ${validation.consistencyIssues.join(', ')}`;
        }
        markdown += `\n`;
        markdown += `- Sensibility: ${(validation.sensibilityScore * 100).toFixed(0)}%`;
        if (validation.sensibilityIssues.length > 0) {
            markdown += ` ⚠️ ${validation.sensibilityIssues.join(', ')}`;
        }
        markdown += `\n`;
        markdown += `- Confidence: ${(validation.confidenceScore * 100).toFixed(0)}%\n`;
        if (validation.hallucinationDetected) {
            markdown += `- 🚨 Hallucination: ${validation.hallucinationDetails.join(', ')}\n`;
        }
        markdown += `\n**Reasoning:** ${validation.reasoning}\n\n`;
        markdown += `---\n\n`;
    });

    return markdown;
}

// Main evaluation
async function runEvaluation() {
    const sampleSize = parseInt(process.argv[2]) || DATASET.length;
    const testDataset = DATASET.slice(0, sampleSize);

    console.log('🚀 Flight Search Agent Evaluation - V2 with LLM Judge');
    console.log(`📊 Testing ${testDataset.length} flights\n`);

    const results = [];
    let totalDuration = 0;

    for (let i = 0; i < testDataset.length; i++) {
        const testCase = testDataset[i];
        const query = generateQuery(testCase);

        console.log(`[${i + 1}/${testDataset.length}] ${query}`);

        try {
            const startTime = Date.now();

            const chain = createChain(query);
            const chainResult = await chain.invoke({ query });

            const duration = ((Date.now() - startTime) / 1000).toFixed(2);
            totalDuration += parseFloat(duration);

            const { extracted, validation, searchResults } = chainResult;

            // Ground truth for comparison
            const groundTruthData = {
                flightNumber: testCase.flightNumber,
                airlineCode: testCase.airlineCode,
                departureAirportCode: testCase.originCode,
                arrivalAirportCode: testCase.destinationCode,
                flightDate: testCase.date,
                aircraftName: mapAircraftCode(testCase.aircraft),
                duration: testCase.duration
            };

            const comparison = compareAllFields(extracted, groundTruthData);

            results.push({
                query,
                groundTruth: testCase,
                extracted,
                validation,
                comparison,
                searchResults,
                duration
            });

            const statusEmoji = validation.validationStatus === 'PASS' ? '✅' : '❌';
            console.log(`  ${statusEmoji} ${validation.validationStatus} | Quality: ${(validation.overallQualityScore * 100).toFixed(0)}% | ${duration}s\n`);

        } catch (error) {
            console.error(`  ✗ Error: ${error.message}\n`);

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
                    aircraftName: null
                },
                validation: {
                    validationStatus: 'FAIL',
                    consistencyScore: 0,
                    consistencyIssues: ['Error during processing'],
                    sensibilityScore: 0,
                    sensibilityIssues: [],
                    confidenceScore: 0,
                    hallucinationDetected: false,
                    hallucinationDetails: [],
                    overallQualityScore: 0,
                    reasoning: `Error: ${error.message}`
                },
                comparison: {},
                duration: '0'
            });
        }
    }

    // Generate outputs
    const timestamp = Date.now();

    const markdown = generateMarkdownReport(results, timestamp);
    fs.writeFileSync(`./eval-v2-results-${timestamp}.md`, markdown);

    const labelStudioJSON = generateLabelStudioJSON(results);
    fs.writeFileSync(`./eval-v2-results-${timestamp}.json`, JSON.stringify(labelStudioJSON, null, 2));

    // Summary
    const passedValidation = results.filter(r => r.validation.validationStatus === 'PASS').length;
    const avgQuality = results.reduce((sum, r) => sum + r.validation.overallQualityScore, 0) / results.length;

    console.log(`\n✅ Completed ${results.length} tests in ${totalDuration.toFixed(1)}s (avg ${(totalDuration/results.length).toFixed(1)}s/test)`);
    console.log(`\n📊 Summary:`);
    console.log(`   Validation Pass Rate: ${passedValidation}/${results.length} (${(passedValidation/results.length*100).toFixed(1)}%)`);
    console.log(`   Average Quality Score: ${avgQuality.toFixed(2)}`);
    console.log(`\n📄 Reports generated:`);
    console.log(`   Markdown: eval-v2-results-${timestamp}.md`);
    console.log(`   Label Studio JSON: eval-v2-results-${timestamp}.json\n`);
}

runEvaluation().catch(console.error);
