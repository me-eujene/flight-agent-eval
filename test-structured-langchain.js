/**
 * Test LangChain's .withStructuredOutput() with Pollinations
 * This is the cleanest approach - LangChain handles JSON schema internally
 */

import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

const POLLINATIONS_API_KEY = 'sk_szglh5aFfm2I1aiXvZEoXYvKDDGVOOl1';
const POLLINATIONS_URL = 'https://gen.pollinations.ai/v1';

// Define schema using Zod
const FlightDataSchema = z.object({
    flightNumber: z.string().nullable(),
    aircraftName: z.string().nullable(),
    flightTime: z.string().nullable()
});

async function test() {
    console.log('Testing LangChain .withStructuredOutput()\n');

    // Create LLM with structured output
    const llm = new ChatOpenAI({
        modelName: 'gemini-search',
        configuration: {
            baseURL: POLLINATIONS_URL,
            apiKey: POLLINATIONS_API_KEY
        },
        temperature: 0
    });

    // Add structured output
    const structuredLLM = llm.withStructuredOutput(FlightDataSchema);

    const prompt = `Search for Southwest flight from Las Vegas to Albuquerque on December 16, 2025.

Find and return:
- flightNumber: just the number (e.g., "548"), or null
- aircraftName: full name (e.g., "Boeing 737-800"), or null
- flightTime: duration in HH:MM format, or null`;

    try {
        console.log('Calling LLM with structured output...\n');

        const result = await structuredLLM.invoke(prompt);

        console.log('✅ Result:');
        console.log(result);
        console.log('\nType:', typeof result);
        console.log('Is object?', typeof result === 'object');

    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error.stack);
    }
}

test();
