import React, { useState } from "react";

const ExpenseTracker = ({ transactions }) => {
  const [filter, setFilter] = useState("");
  const filtered = transactions.filter(
    (t) =>
      t.description.toLowerCase().includes(filter.toLowerCase()) ||
      t.category.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="font-semibold text-gray-700 mb-3">Recent Transactions</h2>
      <input
        type="text"
        placeholder="Search by description or category"
        className="border rounded px-2 py-1 w-full mb-2"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />
      <div className="overflow-y-auto max-h-64">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left py-1">Date</th>
              <th className="text-left py-1">Description</th>
              <th className="text-left py-1">Category</th>
              <th className="text-right py-1">Amount</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length > 0 ? (
              filtered.map((t) => (
                <tr key={t.id}>
                  <td className="py-1">{t.date}</td>
                  <td className="py-1">{t.description}</td>
                  <td className="py-1">{t.category}</td>
                  <td className={`py-1 text-right ${t.amount < 0 ? "text-red-500" : "text-green-600"}`}>
                    ₹{Math.abs(t.amount).toLocaleString()}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={4} className="text-center text-gray-400 py-3">
                  No transactions found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ExpenseTracker;
