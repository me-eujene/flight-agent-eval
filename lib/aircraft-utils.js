/**
 * Aircraft family matching utilities for evaluation
 */

// Aircraft family definitions based on ICAO codes and full names
const AIRCRAFT_FAMILIES = {
  'Boeing 737': ['Boeing 737NG', 'Boeing 737MAX', 'Boeing 717'],
  'Boeing 777': ['Boeing 777', 'Boeing 777-200', 'Boeing 777-300ER'],
  'Boeing 787': ['Boeing 787', 'Boeing 787-8', 'Boeing 787-9', 'Boeing 787-10'],
  'Boeing 747': ['Boeing 747', 'Boeing 747-400', 'Boeing 747-8'],
  'Boeing 757': ['Boeing 757', 'Boeing 757-200', 'Boeing 757-300'],
  'Boeing 767': ['Boeing 767', 'Boeing 767-300', 'Boeing 767-400'],
  'Airbus A320': ['Airbus A319', 'Airbus A320', 'Airbus A321'],
  'Airbus A330': ['Airbus A330', 'Airbus A330-200', 'Airbus A330-300', 'Airbus A330-900'],
  'Airbus A350': ['Airbus A350', 'Airbus A350-900', 'Airbus A350-1000'],
  'Airbus A380': ['Airbus A380', 'Airbus A380-800'],
  'Embraer E170': ['Embraer E170', 'Embraer E175'],
  'Embraer E190': ['Embraer E190', 'Embraer E195'],
  'Embraer E175-E2': ['Embraer E175-E2', 'Embraer E170-E2'],
  'Embraer E190-E2': ['Embraer E190-E2', 'Embraer E195-E2'],
  'Bombardier CRJ': ['Bombardier CRJ', 'Bombardier CRJ-200', 'Bombardier CRJ-700', 'Bombardier CRJ-900'],
  'ATR 42/72': ['ATR 42/72', 'ATR 42', 'ATR 72'],
  'DHC Dash 8': ['DHC Dash 8', 'DHC Dash 8-400']
};

/**
 * Check if two aircraft types belong to the same family
 * @param {string} aircraft1 - First aircraft name
 * @param {string} aircraft2 - Second aircraft name
 * @returns {boolean} True if same family
 */
function isSameFamily(aircraft1, aircraft2) {
  if (!aircraft1 || !aircraft2) return false;

  // Exact match
  if (aircraft1 === aircraft2) return true;

  // Check if both belong to the same family
  for (const [family, members] of Object.entries(AIRCRAFT_FAMILIES)) {
    const has1 = members.includes(aircraft1);
    const has2 = members.includes(aircraft2);

    if (has1 && has2) return true;
  }

  return false;
}

/**
 * Get the family name for an aircraft type
 * @param {string} aircraft - Aircraft name
 * @returns {string} Family name or the aircraft name if no family found
 */
function getAircraftFamily(aircraft) {
  if (!aircraft) return null;

  for (const [family, members] of Object.entries(AIRCRAFT_FAMILIES)) {
    if (members.includes(aircraft)) return family;
  }

  return aircraft; // No family found, return original
}

/**
 * Get similarity score between two aircraft types
 * @param {string} aircraft1 - First aircraft name
 * @param {string} aircraft2 - Second aircraft name
 * @returns {number} Similarity score (0.0 to 1.0)
 */
function getAircraftSimilarity(aircraft1, aircraft2) {
  if (!aircraft1 || !aircraft2) return 0.0;

  // Exact match
  if (aircraft1 === aircraft2) return 1.0;

  // Same family
  if (isSameFamily(aircraft1, aircraft2)) return 0.8;

  // Same manufacturer (Boeing vs Boeing, Airbus vs Airbus)
  const getManufacturer = (ac) => ac.split(' ')[0];
  if (getManufacturer(aircraft1) === getManufacturer(aircraft2)) return 0.3;

  return 0.0;
}

export {
  AIRCRAFT_FAMILIES,
  isSameFamily,
  getAircraftFamily,
  getAircraftSimilarity
};
