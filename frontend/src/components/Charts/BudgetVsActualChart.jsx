import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const BudgetVsActualChart = ({ data, title = "Budget vs Actual Spending" }) => {
  const chartData = data || [
    { category: 'Housing', budget: 30000, actual: 28000 },
    { category: 'Food', budget: 15000, actual: 18000 },
    { category: 'Transport', budget: 8000, actual: 12000 },
    { category: 'Utilities', budget: 5000, actual: 4500 },
    { category: 'Entertainment', budget: 6000, actual: 4000 },
    { category: 'Healthcare', budget: 3000, actual: 2500 },
  ]

  const formatCurrency = (value) => `₹${value.toLocaleString()}`

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const budget = payload.find(p => p.dataKey === 'budget')?.value || 0
      const actual = payload.find(p => p.dataKey === 'actual')?.value || 0
      const difference = budget - actual
      const percentage = budget > 0 ? ((actual / budget) * 100).toFixed(1) : 0

      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">{label}</p>
          <div className="space-y-1">
            <p className="text-blue-600">
              <span className="inline-block w-3 h-3 bg-blue-500 rounded-full mr-2"></span>
              Budget: {formatCurrency(budget)}
            </p>
            <p className="text-green-600">
              <span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-2"></span>
              Actual: {formatCurrency(actual)}
            </p>
            <p className={`font-medium ${difference >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {difference >= 0 ? 'Under' : 'Over'} by: {formatCurrency(Math.abs(difference))}
            </p>
            <p className="text-gray-600 text-sm">
              {percentage}% of budget used
            </p>
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          barCategoryGap="20%"
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="category" 
            tick={{ fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            tickFormatter={formatCurrency}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Bar 
            dataKey="budget" 
            fill="#3b82f6" 
            name="Budget"
            radius={[2, 2, 0, 0]}
          />
          <Bar 
            dataKey="actual" 
            fill="#10b981" 
            name="Actual"
            radius={[2, 2, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
      
      {/* Summary stats */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
        <div className="bg-blue-50 p-3 rounded-lg">
          <p className="text-sm text-blue-600 font-medium">Total Budget</p>
          <p className="text-lg font-bold text-blue-800">
            {formatCurrency(chartData.reduce((sum, item) => sum + item.budget, 0))}
          </p>
        </div>
        <div className="bg-green-50 p-3 rounded-lg">
          <p className="text-sm text-green-600 font-medium">Total Actual</p>
          <p className="text-lg font-bold text-green-800">
            {formatCurrency(chartData.reduce((sum, item) => sum + item.actual, 0))}
          </p>
        </div>
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="text-sm text-gray-600 font-medium">Difference</p>
          <p className={`text-lg font-bold ${
            chartData.reduce((sum, item) => sum + (item.budget - item.actual), 0) >= 0 
              ? 'text-green-800' 
              : 'text-red-800'
          }`}>
            {formatCurrency(Math.abs(chartData.reduce((sum, item) => sum + (item.budget - item.actual), 0)))}
          </p>
        </div>
        <div className="bg-purple-50 p-3 rounded-lg">
          <p className="text-sm text-purple-600 font-medium">Budget Used</p>
          <p className="text-lg font-bold text-purple-800">
            {(
              (chartData.reduce((sum, item) => sum + item.actual, 0) / 
               chartData.reduce((sum, item) => sum + item.budget, 0)) * 100
            ).toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  )
}


export default BudgetVsActualChart

