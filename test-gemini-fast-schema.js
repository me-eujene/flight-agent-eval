/**
 * Test gemini-fast with formal JSON schema (Step 2)
 */

import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

const POLLINATIONS_API_KEY = 'sk_szglh5aFfm2I1aiXvZEoXYvKDDGVOOl1';
const POLLINATIONS_URL = 'https://gen.pollinations.ai/v1';

// Define formal schema
const FlightDataSchema = z.object({
    flightNumber: z.string().nullable(),
    airlineCode: z.string().nullable(),
    departureAirportCode: z.string().nullable(),
    arrivalAirportCode: z.string().nullable(),
    flightDate: z.string().nullable(),
    flightTime: z.string().nullable(),
    aircraftName: z.string().nullable(),
    overallAssessment: z.string()
});

async function test() {
    console.log('Testing gemini-fast with formal JSON schema\n');

    // Simulate Step 1 output (text report)
    const searchReport = `
FLIGHT NUMBER: 3118
Source: FlightAware.com
Notes: Found for this route

AIRLINE CODE: WN
Source: Provided in query

DEPARTURE AIRPORT: LAS
Source: Provided in query

ARRIVAL AIRPORT: ABQ
Source: Provided in query

FLIGHT DATE: 16-12-2025
Source: Provided in query

FLIGHT TIME: 01:25
Source: FlightAware.com
Notes: Calculated from dep/arr times

AIRCRAFT TYPE: Boeing 737-800
Source: FlightAware.com
Notes: Converted from code 73H

OVERALL ASSESSMENT: All flight details found successfully from FlightAware.
`;

    try {
        // Step 2: gemini-fast with structured output
        const structureLLM = new ChatOpenAI({
            modelName: 'gemini-fast',
            configuration: {
                baseURL: POLLINATIONS_URL,
                apiKey: POLLINATIONS_API_KEY
            },
            temperature: 0
        }).withStructuredOutput(FlightDataSchema);

        const prompt = `Convert this search report into structured JSON.

Search Report:
${searchReport}

Extract all fields according to the schema.`;

        console.log('Calling gemini-fast with .withStructuredOutput()...\n');

        const result = await structureLLM.invoke(prompt);

        console.log('✅ Result (typed object):');
        console.log(result);
        console.log('\nType:', typeof result);
        console.log('Has flightNumber?', 'flightNumber' in result);

    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error.stack);
    }
}

test();
