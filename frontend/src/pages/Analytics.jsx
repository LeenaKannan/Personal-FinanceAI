import React from "react";
import { SummaryCharts } from "../components/Charts/SummaryCharts";
import { TrendsChart } from "../components/Charts/TrendsChart";
import { BarChart3 } from "lucide-react";

const dummyExpenses = {
  "Housing": 28000,
  "Food & Groceries": 16000,
  "Transport": 12000,
  "Utilities": 5000,
  "Entertainment": 4000,
  "Self Care": 3000,
};

const dummyTrends = [
  { month: "Jan", expenses: 35000, income: 80000 },
  { month: "Feb", expenses: 42000, income: 80000 },
  { month: "Mar", expenses: 39000, income: 80000 },
  { month: "Apr", expenses: 41000, income: 80000 },
];

const Analytics = () => (
  <div className="max-w-5xl mx-auto mt-10 p-6 bg-white rounded-xl shadow-lg">
    <div className="flex items-center mb-6">
      <BarChart3 className="text-green-500 mr-3" size={36} />
      <h2 className="text-2xl font-bold text-gray-800">Analytics & Insights</h2>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <div className="bg-blue-50 rounded-lg p-4">
        <SummaryCharts expenses={dummyExpenses} />
      </div>
      <div className="bg-green-50 rounded-lg p-4">
        <TrendsChart data={dummyTrends} />
      </div>
    </div>
    <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-white border rounded-lg p-4 text-center shadow">
        <div className="text-gray-500">Total Expenses (Apr)</div>
        <div className="text-2xl font-bold text-red-500">₹41,000</div>
      </div>
      <div className="bg-white border rounded-lg p-4 text-center shadow">
        <div className="text-gray-500">Total Income (Apr)</div>
        <div className="text-2xl font-bold text-green-600">₹80,000</div>
      </div>
      <div className="bg-white border rounded-lg p-4 text-center shadow">
        <div className="text-gray-500">Savings Rate</div>
        <div className="text-2xl font-bold text-blue-600">48.8%</div>
      </div>
    </div>
  </div>
);

export default Analytics;
