/**
 * Parse Label Studio CSV export and calculate field accuracy
 * Usage: node parse-label-studio.js <csv-file-path>
 */

import fs from 'fs';

// Get CSV file path from command line
const csvPath = process.argv[2] || 'C:\\Users\\eujene\\Downloads\\project-1-at-2026-01-11-20-26-89340637.csv';

// Simple CSV parser (handles quoted fields with commas)
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim());
    const headers = parseCSVLine(lines[0]);

    return lines.slice(1).map(line => {
        const values = parseCSVLine(line);
        const record = {};
        headers.forEach((header, i) => {
            record[header] = values[i] || '';
        });
        return record;
    });
}

function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];

        if (char === '"') {
            if (inQuotes && line[i + 1] === '"') {
                current += '"';
                i++; // Skip next quote
            } else {
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            result.push(current);
            current = '';
        } else {
            current += char;
        }
    }

    result.push(current);
    return result;
}

// Read and parse CSV
const csvContent = fs.readFileSync(csvPath, 'utf8');
const records = parseCSV(csvContent);

console.log(`📊 Label Studio Annotation Analysis\n`);
console.log(`Total flights annotated: ${records.length}\n`);

// Track field correctness
const fieldCounts = {
    'Direction correct': 0,
    'Airline correct': 0,
    'Flight number correct': 0,
    'Date correct': 0,
    'Time correct': 0,
    'ACFT correct': 0
};

// Count correct fields across all annotations
records.forEach((record, idx) => {
    try {
        let choices = [];

        // Try parsing as JSON first
        try {
            const choicesData = JSON.parse(record.choice);
            choices = choicesData.choices || [];
        } catch {
            // If not JSON, treat as single string choice
            if (record.choice && record.choice.trim()) {
                choices = [record.choice.trim()];
            }
        }

        // Count each correct field
        choices.forEach(choice => {
            if (fieldCounts.hasOwnProperty(choice)) {
                fieldCounts[choice]++;
            }
        });
    } catch (error) {
        console.error(`Error parsing row ${idx + 1}:`, error.message);
    }
});

// Calculate percentages and print report
console.log(`═══════════════════════════════════════════════════════`);
console.log(`Field Accuracy Report`);
console.log(`═══════════════════════════════════════════════════════\n`);

const totalFlights = records.length;
let totalCorrect = 0;
let totalFields = 0;

Object.entries(fieldCounts).forEach(([field, count]) => {
    const percentage = ((count / totalFlights) * 100).toFixed(1);
    const fieldName = field.replace(' correct', '').padEnd(20);
    console.log(`${fieldName}: ${count.toString().padStart(3)}/${totalFlights} (${percentage.padStart(5)}%)`);

    totalCorrect += count;
    totalFields += totalFlights;
});

console.log(`\n═══════════════════════════════════════════════════════`);

// Overall accuracy
const overallAccuracy = ((totalCorrect / totalFields) * 100).toFixed(1);
console.log(`Overall Accuracy   : ${totalCorrect}/${totalFields} (${overallAccuracy}%)`);

// Calculate perfect matches (all 6 fields correct)
let perfectMatches = 0;
records.forEach((record) => {
    try {
        const choicesData = JSON.parse(record.choice);
        const choices = choicesData.choices || [];
        if (choices.length === 6) {
            perfectMatches++;
        }
    } catch (error) {
        // Skip errors
    }
});

console.log(`Perfect Matches    : ${perfectMatches}/${totalFlights} (${((perfectMatches / totalFlights) * 100).toFixed(1)}%)`);
console.log(`═══════════════════════════════════════════════════════\n`);

// Breakdown by field category
console.log(`\n📈 Field Categories:\n`);
console.log(`Query-provided fields (should be ~100%):`);
console.log(`  - Direction: ${((fieldCounts['Direction correct'] / totalFlights) * 100).toFixed(1)}%`);
console.log(`  - Airline:   ${((fieldCounts['Airline correct'] / totalFlights) * 100).toFixed(1)}%`);
console.log(`  - Date:      ${((fieldCounts['Date correct'] / totalFlights) * 100).toFixed(1)}%`);

console.log(`\nSearched fields (harder to find):`);
console.log(`  - Flight #:  ${((fieldCounts['Flight number correct'] / totalFlights) * 100).toFixed(1)}%`);
console.log(`  - Time:      ${((fieldCounts['Time correct'] / totalFlights) * 100).toFixed(1)}%`);
console.log(`  - Aircraft:  ${((fieldCounts['ACFT correct'] / totalFlights) * 100).toFixed(1)}%`);
console.log();
