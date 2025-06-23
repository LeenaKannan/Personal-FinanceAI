import React from "react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";

const COLORS = ["#34d399", "#fbbf24", "#60a5fa", "#f87171", "#a78bfa", "#f472b6"];

export const SummaryCharts = ({ expenses }) => {
  const data = Object.entries(expenses || {}).map(([category, value]) => ({
    name: category,
    value,
  }));

  return (
    <div>
      <h3 className="font-semibold mb-2">Spending by Category</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};
