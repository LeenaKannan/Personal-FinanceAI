// src/utils/api.js

// Uncomment and set your backend URL when ready
// const API_BASE = "http://localhost:8000/api";

export async function fetchUserProfile() {
  // // Real API call:
  // const res = await fetch(`${API_BASE}/user/profile`);
  // return await res.json();

  // Mock for local testing:
  return {
    name: "Amit Sharma",
    email: "amit.sharma@email.com",
    city: "Mumbai",
    gender: "male",
    income: 80000,
    age: 28,
  };
}

export async function fetchTransactions() {
  // // Real API call:
  // const res = await fetch(`${API_BASE}/transactions`);
  // return await res.json();

  // Mock for local testing:
  return [
    { id: 1, date: "2025-06-01", description: "ZOMATO ONLINE PAYMENT", category: "Food & Groceries", amount: -850 },
    { id: 2, date: "2025-06-02", description: "RENT TRANSFER", category: "Housing", amount: -28000 },
    { id: 3, date: "2025-06-03", description: "SALARY CREDIT", category: "Income", amount: 80000 },
    { id: 4, date: "2025-06-04", description: "OLA CABS", category: "Transport", amount: -1200 },
    { id: 5, date: "2025-06-05", description: "ELECTRICITY BILL", category: "Utilities", amount: -2000 },
  ];
}

export async function fetchInsights() {
  // // Real API call:
  // const res = await fetch(`${API_BASE}/insights`);
  // return await res.json();

  // Mock for local testing:
  return [
    {
      type: "warning",
      title: "High Food Spending",
      message: "You spent 30% more on food this month compared to last month.",
      action: "Consider cooking at home more often.",
    },
    {
      type: "tip",
      title: "Investment Opportunity",
      message: "Your savings rate allows you to invest more this month.",
      action: "Check mutual fund recommendations.",
    },
  ];
}

export async function updateUserProfile(profile) {
  // // Real API call:
  // const res = await fetch(`${API_BASE}/user/profile`, {
  //   method: "PUT",
  //   headers: { "Content-Type": "application/json" },
  //   body: JSON.stringify(profile),
  // });
  // return await res.json();

  // Mock for local testing:
  return { success: true, ...profile };
}
