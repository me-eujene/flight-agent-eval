/**
 * Test LangChain LCEL (LangChain Expression Language) with Pollinations
 * Modern approach: pipe chains together
 *
 * Step 1: gemini-search (search)
 * Step 2: gemini (structure JSON)
 */

import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

const POLLINATIONS_API_KEY = 'sk_szglh5aFfm2I1aiXvZEoXYvKDDGVOOl1';
const POLLINATIONS_URL = 'https://gen.pollinations.ai/v1';

// Step 1: Search LLM (gemini-search with built-in Google Search)
const searchLLM = new ChatOpenAI({
    modelName: 'gemini-search',
    configuration: {
        baseURL: POLLINATIONS_URL,
        apiKey: POLLINATIONS_API_KEY
    },
    temperature: 0
});

// Step 2: Structuring LLM (regular gemini)
const structureLLM = new ChatOpenAI({
    modelName: 'gemini',
    configuration: {
        baseURL: POLLINATIONS_URL,
        apiKey: POLLINATIONS_API_KEY
    },
    temperature: 0
});

// Prompt 1: Search for flight info
const searchPrompt = PromptTemplate.fromTemplate(`
You are a flight search assistant. Search for this flight and provide a detailed report.

Query: {query}

Focus on these websites:
1. FlightAware.com
2. FlightRadar24.com
3. aviability.com

Find:
- Flight number
- Aircraft type (convert codes like 73H to "Boeing 737-800")
- Flight duration in HH:MM format

Provide a detailed report with sources. If you can't find something, say "NOT FOUND".
`);

// Prompt 2: Structure the results
const structurePrompt = PromptTemplate.fromTemplate(`
Convert this search report into JSON.

Search Report:
{searchResults}

Extract and return ONLY a valid JSON object with these exact fields:
- flightNumber: just the number (e.g., "548"), or null if not found
- aircraftName: full aircraft name (e.g., "Boeing 737-800"), or null if not found
- flightTime: flight duration in HH:MM format, or null if not found

Return ONLY the JSON object, no other text.
`);

// Build the chain using LCEL (pipe operator)
const chain = RunnableSequence.from([
    // Step 1: Search
    {
        searchResults: searchPrompt.pipe(searchLLM).pipe(new StringOutputParser())
    },
    // Step 2: Structure
    structurePrompt,
    structureLLM,
    new StringOutputParser()
]);

// Test it
async function test() {
    console.log('🔗 Testing LangChain LCEL Chain (no agents, no tools)\n');
    console.log('Step 1: gemini-search → searches web');
    console.log('Step 2: gemini → structures as JSON\n');

    try {
        const result = await chain.invoke({
            query: 'Southwest Airlines from Las Vegas to Albuquerque on December 16, 2025'
        });

        console.log('=== FINAL OUTPUT (Structured JSON) ===');
        console.log(result);

        console.log('\n=== PARSED ===');
        const parsed = JSON.parse(result);
        console.log(parsed);

        console.log('\n✅ Success!');
    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

test();
