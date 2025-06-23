import React from "react";
import ExpenseTracker from "./ExpenseTracker";
import AIInsights from "./AIInsights";
import { SummaryCharts } from "./Charts/SummaryCharts";

const Dashboard = ({ user, balances, expenses, insights, transactions }) => (
  <div className="p-6 space-y-8">
    <h1 className="text-2xl font-bold mb-4">Welcome, {user.firstName}!</h1>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="font-semibold text-gray-700">Total Balance</h2>
        <p className="text-2xl font-bold text-green-600">₹{balances.total.toLocaleString()}</p>
        <div className="flex justify-between mt-2 text-sm text-gray-500">
          <span>Savings: ₹{balances.savings}</span>
          <span>Investments: ₹{balances.investments}</span>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="font-semibold text-gray-700">This Month's Spend</h2>
        <p className="text-2xl font-bold text-red-500">₹{expenses.monthly.toLocaleString()}</p>
        <div className="flex justify-between mt-2 text-sm text-gray-500">
          <span>Budget: ₹{expenses.budget}</span>
          <span>Left: ₹{expenses.budget - expenses.monthly}</span>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="font-semibold text-gray-700">Key Insight</h2>
        <p className="text-md text-blue-600">{insights[0]?.message || "No insights yet!"}</p>
      </div>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <SummaryCharts expenses={expenses.byCategory} />
      </div>
      <div>
        <ExpenseTracker transactions={transactions} />
      </div>
    </div>
    <AIInsights insights={insights} />
  </div>
);

export default Dashboard;
