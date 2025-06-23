// src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route, Link, NavLink } from "react-router-dom";
import Home from "./pages/Home";
import Profile from "./pages/Profile";
import Analytics from "./pages/Analytics";
import Settings from "./pages/Settings";
import Dashboard from "./components/Dashboard";
import { PiggyBank } from "lucide-react";

const navLinks = [
  { to: "/", label: "Home" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/profile", label: "Profile" },
  { to: "/analytics", label: "Analytics" },
  { to: "/settings", label: "Settings" },
];

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
        {/* Navbar */}
        <nav className="bg-white shadow sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
            <Link to="/" className="flex items-center gap-2 text-green-600 font-bold text-xl">
              <PiggyBank className="text-green-500" size={28} />
              SmartFinance AI
            </Link>
            <div className="flex gap-4">
              {navLinks.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={({ isActive }) =>
                    `px-3 py-1 rounded transition ${isActive
                      ? "bg-green-100 text-green-700 font-semibold"
                      : "text-gray-700 hover:bg-green-50"
                    }`
                  }
                  end={link.to === "/"}
                >
                  {link.label}
                </NavLink>
              ))}
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="py-6 px-2 md:px-0">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard
              user={{ firstName: "Amit" }}
              balances={{ total: 120000, savings: 50000, investments: 30000 }}
              expenses={{
                monthly: 42000,
                budget: 60000,
                byCategory: {
                  "Housing": 18000,
                  "Food & Groceries": 9000,
                  "Transport": 4000,
                  "Utilities": 3000,
                  "Entertainment": 2000,
                  "Self Care": 1000,
                  "Other": 7000,
                },
              }}
              insights={[
                {
                  type: "tip",
                  title: "Good Savings",
                  message: "You saved 30% of your income this month.",
                  action: "Consider investing more.",
                },
              ]}
              transactions={[
                { id: 1, date: "2025-06-01", description: "ZOMATO ONLINE PAYMENT", category: "Food & Groceries", amount: -850 },
                { id: 2, date: "2025-06-02", description: "RENT TRANSFER", category: "Housing", amount: -18000 },
                { id: 3, date: "2025-06-03", description: "SALARY CREDIT", category: "Income", amount: 60000 },
                { id: 4, date: "2025-06-04", description: "OLA CABS", category: "Transport", amount: -1200 },
                { id: 5, date: "2025-06-05", description: "ELECTRICITY BILL", category: "Utilities", amount: -2000 },
              ]}
            />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
