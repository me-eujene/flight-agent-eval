/**
 * Test Pollinations.ai with tools parameter
 */

const POLLINATIONS_API_KEY = 'sk_szglh5aFfm2I1aiXvZEoXYvKDDGVOOl1';
const POLLINATIONS_URL = 'https://gen.pollinations.ai/v1/chat/completions';

async function testPollinationsTools() {
    console.log('Testing Pollinations.ai with google_search tool\n');

    try {
        const response = await fetch(POLLINATIONS_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${POLLINATIONS_API_KEY}`
            },
            body: JSON.stringify({
                model: 'gemini-search',  // This model has built-in search
                messages: [{
                    role: 'user',
                    content: 'What is the flight number for Southwest from Las Vegas to Albuquerque on December 16, 2025?'
                }],
                temperature: 0
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ API Error:', response.status, errorText);
            return;
        }

        const result = await response.json();

        console.log('✅ Response received\n');
        console.log('Model:', result.model);
        console.log('Provider:', result.provider);
        console.log('\nFull Response:');
        console.log(JSON.stringify(result, null, 2));

    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

testPollinationsTools();
