// src/utils/ml-utils.js

// Dummy expense prediction based on income, city, and gender
export function predictExpenses({ income, city, gender, age }) {
  // Simple mock logic for demo:
  const cityCostIndex = {
    Mumbai: 1.0,
    Delhi: 0.95,
    Bangalore: 0.88,
    Chennai: 0.82,
    Pune: 0.78,
    Hyderabad: 0.73,
    Kolkata: 0.75,
    Ahmedabad: 0.70,
    Jaipur: 0.65,
    Lucknow: 0.60,
  };
  const base = income * (cityCostIndex[city] || 1.0);

  // Gender/utilization adjustment (mock)
  const genderAdj = gender === "female" ? 0.97 : 1.0;

  // Age adjustment (mock)
  const ageAdj = age < 25 ? 0.9 : age > 40 ? 1.1 : 1.0;

  // Split into categories
  return {
    Housing: Math.round(base * 0.35 * genderAdj * ageAdj),
    "Food & Groceries": Math.round(base * 0.20 * genderAdj),
    Transport: Math.round(base * 0.12),
    Utilities: Math.round(base * 0.08),
    Entertainment: Math.round(base * 0.07),
    "Self Care": Math.round(base * 0.05),
    Other: Math.round(base * 0.13),
  };
}

// Simple categorization based on keywords
export function categorizeTransaction(description) {
  const desc = description.toLowerCase();
  if (desc.includes("zomato") || desc.includes("swiggy") || desc.includes("restaurant")) return "Food & Groceries";
  if (desc.includes("rent")) return "Housing";
  if (desc.includes("ola") || desc.includes("uber") || desc.includes("cab")) return "Transport";
  if (desc.includes("electricity") || desc.includes("water") || desc.includes("gas")) return "Utilities";
  if (desc.includes("movie") || desc.includes("netflix") || desc.includes("entertainment")) return "Entertainment";
  if (desc.includes("salon") || desc.includes("spa") || desc.includes("self")) return "Self Care";
  if (desc.includes("salary") || desc.includes("credit")) return "Income";
  return "Other";
}
