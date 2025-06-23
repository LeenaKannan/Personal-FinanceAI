// src/utils/formatters.js

// Format number as INR currency
export function formatINR(amount) {
  return "₹" + Number(amount).toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

// Format ISO date (YYYY-MM-DD) to readable format
export function formatDate(dateStr) {
  if (!dateStr) return "";
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-IN", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

// Capitalize first letter of each word
export function capitalizeWords(str) {
  return str.replace(/\b\w/g, (l) => l.toUpperCase());
}
