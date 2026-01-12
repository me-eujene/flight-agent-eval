/**
 * Evaluation metrics calculations (Precision, Recall, F1)
 */

/**
 * Calculate precision, recall, and F1 score for a single field
 * @param {Array} results - Array of evaluation results
 * @param {string} field - Field name to evaluate
 * @returns {Object} Metrics object with precision, recall, F1
 */
function calculateFieldMetrics(results, field) {
  const total = results.length;

  // Count how many results have this field extracted (not null)
  const extracted = results.filter(r => {
    const value = r.extracted[field];
    return value !== null && value !== undefined && value !== 'null';
  }).length;

  // Count how many are correct (comparison.match === true)
  const correct = results.filter(r => {
    const comparison = r.comparison && r.comparison[field];
    return comparison && comparison.match === true;
  }).length;

  // Calculate metrics
  const precision = extracted > 0 ? correct / extracted : 0;
  const recall = total > 0 ? correct / total : 0;
  const f1 = (precision + recall) > 0
    ? 2 * (precision * recall) / (precision + recall)
    : 0;

  return {
    total,
    extracted,
    correct,
    precision: (precision * 100).toFixed(1) + '%',
    recall: (recall * 100).toFixed(1) + '%',
    f1: (f1 * 100).toFixed(1) + '%',
    // Raw values for calculations
    precisionRaw: precision,
    recallRaw: recall,
    f1Raw: f1
  };
}

/**
 * Calculate metrics for all fields
 * @param {Array} results - Array of evaluation results
 * @param {Array} fields - List of field names to evaluate
 * @returns {Object} Metrics for each field
 */
function calculateAllMetrics(results, fields) {
  const metrics = {};

  for (const field of fields) {
    metrics[field] = calculateFieldMetrics(results, field);
  }

  // Calculate overall weighted F1 score
  // Weight fields by importance (query-provided fields have lower weight)
  const weights = {
    airlineCode: 0.5,           // Provided in query
    departureAirportCode: 0.5,  // Provided in query
    arrivalAirportCode: 0.5,    // Provided in query
    flightDate: 0.5,            // Provided in query
    aircraftName: 1.5,          // Searched by agent
    flightTime: 1.5             // Searched by agent
  };

  let weightedF1Sum = 0;
  let totalWeight = 0;

  for (const field of fields) {
    const weight = weights[field] || 1.0;
    const f1 = metrics[field].f1Raw;
    weightedF1Sum += f1 * weight;
    totalWeight += weight;
  }

  const overallF1 = totalWeight > 0 ? weightedF1Sum / totalWeight : 0;

  metrics.overall = {
    weightedF1: (overallF1 * 100).toFixed(1) + '%',
    weightedF1Raw: overallF1
  };

  return metrics;
}

/**
 * Get summary statistics from results
 * @param {Array} results - Array of evaluation results
 * @returns {Object} Summary statistics
 */
function getSummaryStats(results) {
  const totalFlights = results.length;

  // Count perfect matches (all fields correct)
  const perfectMatches = results.filter(r => {
    if (!r.comparison) return false;
    return Object.values(r.comparison).every(c => c && c.match === true);
  }).length;

  // Count flights with at least one field extracted
  const withData = results.filter(r => {
    if (!r.extracted) return false;
    return Object.values(r.extracted).some(v => v !== null && v !== undefined && v !== 'null');
  }).length;

  // Count flagged for review
  const flaggedCount = results.filter(r => r.flags && r.flags.length > 0).length;

  // Average grades
  let totalGrade = 0;
  let gradeCount = 0;

  results.forEach(r => {
    if (r.comparison) {
      Object.values(r.comparison).forEach(c => {
        if (c && typeof c.grade === 'number') {
          totalGrade += c.grade;
          gradeCount++;
        }
      });
    }
  });

  const avgGrade = gradeCount > 0 ? totalGrade / gradeCount : 0;

  return {
    totalFlights,
    perfectMatches,
    withData,
    flaggedCount,
    avgGrade: (avgGrade * 100).toFixed(1) + '%',
    avgGradeRaw: avgGrade
  };
}

export {
  calculateFieldMetrics,
  calculateAllMetrics,
  getSummaryStats
};
