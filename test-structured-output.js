/**
 * Test Pollinations.ai structured JSON output
 */

const POLLINATIONS_API_KEY = 'sk_szglh5aFfm2I1aiXvZEoXYvKDDGVOOl1';
const POLLINATIONS_URL = 'https://gen.pollinations.ai/v1/chat/completions';

const FLIGHT_SCHEMA = {
    type: "object",
    properties: {
        flightNumber: { type: "string", nullable: true },
        aircraftName: { type: "string", nullable: true },
        flightTime: { type: "string", nullable: true }
    },
    required: ["flightNumber", "aircraftName", "flightTime"],
    additionalProperties: false
};

async function testStructuredOutput() {
    console.log('Testing Pollinations structured JSON output\n');

    const prompt = `What is the flight number, aircraft type, and flight duration for Southwest Airlines from Las Vegas to Albuquerque on December 16, 2025?

Search for this information and return it as JSON with these exact fields:
- flightNumber: just the number (e.g., "548"), or null
- aircraftName: full aircraft name (e.g., "Boeing 737-800"), or null
- flightTime: flight duration in HH:MM format, or null`;

    try {
        const response = await fetch(POLLINATIONS_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${POLLINATIONS_API_KEY}`
            },
            body: JSON.stringify({
                model: 'gemini-search',
                messages: [{ role: 'user', content: prompt }],
                // response_format: {  // TEST: Comment out to see if search works
                //     type: 'json_schema',
                //     json_schema: {
                //         name: 'FlightData',
                //         strict: true,
                //         schema: FLIGHT_SCHEMA
                //     }
                // },
                temperature: 0
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ API Error:', response.status, errorText);
            return;
        }

        const result = await response.json();
        const content = result.choices[0].message.content;
        const grounding = result.choices[0].groundingMetadata;

        console.log('✅ Response received\n');
        console.log('Raw content:');
        console.log(content);

        try {
            console.log('\nParsed JSON:');
            console.log(JSON.parse(content));
        } catch (e) {
            console.log('\n(Not JSON - that\'s OK for this test)');
        }

        console.log('\nGrounding Metadata:');
        console.log('- Searches:', grounding?.webSearchQueries?.length || 0);
        console.log('- Sources:', grounding?.groundingChunks?.map(c => c.web?.domain).join(', ') || 'none');

    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

testStructuredOutput();
