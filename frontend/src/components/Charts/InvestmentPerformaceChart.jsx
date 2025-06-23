import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts'

const InvestmentPerformanceChart = ({ data, title = "Investment Performance" }) => {
  const chartData = data || [
    { month: 'Jan', portfolio: 100000, benchmark: 100000, invested: 95000 },
    { month: 'Feb', portfolio: 105000, benchmark: 102000, invested: 100000 },
    { month: 'Mar', portfolio: 98000, benchmark: 99000, invested: 105000 },
    { month: 'Apr', portfolio: 112000, benchmark: 105000, invested: 110000 },
    { month: 'May', portfolio: 118000, benchmark: 108000, invested: 115000 },
    { month: 'Jun', portfolio: 125000, benchmark: 112000, invested: 120000 },
    { month: 'Jul', portfolio: 122000, benchmark: 110000, invested: 125000 },
    { month: 'Aug', portfolio: 135000, benchmark: 118000, invested: 130000 },
    { month: 'Sep', portfolio: 142000, benchmark: 125000, invested: 135000 },
    { month: 'Oct', portfolio: 138000, benchmark: 122000, invested: 140000 },
    { month: 'Nov', portfolio: 148000, benchmark: 128000, invested: 145000 },
    { month: 'Dec', portfolio: 155000, benchmark: 135000, invested: 150000 },
  ]

  const formatCurrency = (value) => `₹${(value / 1000).toFixed(0)}K`

  const calculateReturns = (current, initial) => {
    return ((current - initial) / initial * 100).toFixed(2)
  }

  const currentValue = chartData[chartData.length - 1]?.portfolio || 0
  const initialValue = chartData[0]?.portfolio || 0
  const totalInvested = chartData[chartData.length - 1]?.invested || 0
  
  const absoluteReturns = currentValue - totalInvested
  const percentageReturns = totalInvested > 0 ? ((absoluteReturns / totalInvested) * 100).toFixed(2) : 0

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">{label}</p>
          <div className="space-y-1">
            {payload.map((entry, index) => (
              <p key={index} style={{ color: entry.color }}>
                <span className="inline-block w-3 h-3 rounded-full mr-2" style={{ backgroundColor: entry.color }}></span>
                {entry.name}: {formatCurrency(entry.value)}
              </p>
            ))}
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
        <div className="flex space-x-4 text-sm">
          <div className="text-center">
            <p className="text-gray-600">Total Returns</p>
            <p className={`font-bold ${absoluteReturns >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {absoluteReturns >= 0 ? '+' : ''}{formatCurrency(absoluteReturns)} ({percentageReturns}%)
            </p>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <defs>
            <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="benchmarkGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="month" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} tickFormatter={formatCurrency} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          <Area
            type="monotone"
            dataKey="portfolio"
            stroke="#10b981"
            strokeWidth={3}
            fill="url(#portfolioGradient)"
            name="Portfolio Value"
          />
          <Area
            type="monotone"
            dataKey="benchmark"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#benchmarkGradient)"
            name="Benchmark (Nifty 50)"
          />
          <Line
            type="monotone"
            dataKey="invested"
            stroke="#f59e0b"
            strokeWidth={2}
            strokeDasharray="5 5"
            name="Amount Invested"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Performance metrics */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-green-50 p-4 rounded-lg text-center">
          <p className="text-sm text-green-600 font-medium">Current Value</p>
          <p className="text-xl font-bold text-green-800">{formatCurrency(currentValue)}</p>
        </div>
        <div className="bg-blue-50 p-4 rounded-lg text-center">
          <p className="text-sm text-blue-600 font-medium">Total Invested</p>
          <p className="text-xl font-bold text-blue-800">{formatCurrency(totalInvested)}</p>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg text-center">
          <p className="text-sm text-purple-600 font-medium">Absolute Returns</p>
          <p className={`text-xl font-bold ${absoluteReturns >= 0 ? 'text-green-800' : 'text-red-800'}`}>
            {absoluteReturns >= 0 ? '+' : ''}{formatCurrency(absoluteReturns)}
          </p>
        </div>
        <div className="bg-orange-50 p-4 rounded-lg text-center">
          <p className="text-sm text-orange-600 font-medium">Returns %</p>
          <p className={`text-xl font-bold ${percentageReturns >= 0 ? 'text-green-800' : 'text-red-800'}`}>
            {percentageReturns >= 0 ? '+' : ''}{percentageReturns}%
          </p>
        </div>
      </div>

      {/* Investment allocation */}
      <div className="mt-6">
        <h4 className="text-md font-semibold text-gray-700 mb-3">Asset Allocation</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <span className="text-sm text-gray-600">Equity</span>
            <span className="font-semibold text-gray-800">65%</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <span className="text-sm text-gray-600">Debt</span>
            <span className="font-semibold text-gray-800">25%</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <span className="text-sm text-gray-600">Gold</span>
            <span className="font-semibold text-gray-800">5%</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <span className="text-sm text-gray-600">Cash</span>
            <span className="font-semibold text-gray-800">5%</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default InvestmentPerformanceChart
