import React from "react";
import { Lightbulb, AlertTriangle, TrendingUp } from "lucide-react";

const iconMap = {
  tip: <Lightbulb className="inline w-5 h-5 text-yellow-400 mr-2" />,
  warning: <AlertTriangle className="inline w-5 h-5 text-red-500 mr-2" />,
  opportunity: <TrendingUp className="inline w-5 h-5 text-green-500 mr-2" />,
};

const AIInsights = ({ insights }) => (
  <div className="bg-white rounded-lg shadow p-4 mt-8">
    <h2 className="font-semibold text-gray-700 mb-3">AI Insights</h2>
    {insights && insights.length > 0 ? (
      <ul className="space-y-4">
        {insights.map((insight, i) => (
          <li key={i} className="flex items-start">
            <span>{iconMap[insight.type] || <Lightbulb className="inline w-5 h-5 mr-2" />}</span>
            <div>
              <div className="font-semibold">{insight.title}</div>
              <div className="text-gray-600">{insight.message}</div>
              {insight.action && (
                <div className="text-blue-500 text-sm mt-1">Action: {insight.action}</div>
              )}
            </div>
          </li>
        ))}
      </ul>
    ) : (
      <div className="text-gray-400">No insights available yet.</div>
    )}
  </div>
);

export default AIInsights;
